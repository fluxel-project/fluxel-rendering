//! The shader-visible descriptor heap, and the free-range allocator that hands
//! slots out of it.
//!
//! One heap per device, created with the device and living as long as it. It is
//! shader-visible because a descriptor table is set by *GPU* address, and only a
//! shader-visible heap has one; a non-visible heap would be reachable from the
//! CPU only and could never be named in `Set*RootDescriptorTable`.
//!
//! # Why one heap and not one per bind group
//!
//! Direct3D 12 allows at most one CBV/SRV/UAV heap to be bound at a time on the
//! command list (`SetDescriptorHeaps` takes an array, but the API documents that
//! only one of each type may be set, and the driver refuses more). A bind group
//! created on its own heap would therefore be un-bindable beside any other group,
//! which is the opposite of what a bind group is for. So the heap is a device
//! resource and a group is a *range* of it.
//!
//! # Why the allocator coalesces
//!
//! A frame that creates and drops bind groups every frame would otherwise shred
//! the free list into runs of one, and a request for four contiguous descriptors
//! would fail against a heap with four hundred free descriptors in it. Merging
//! adjacent free runs on release is what keeps the answer to "is there room" a
//! fact about the heap's occupancy rather than about the order things were
//! dropped in.
//!
//! # What this module does not own
//!
//! Which descriptors an allocation needs, and what gets written into them. That
//! is [`super::layout`]'s plan and [`super::group`]'s writes. This module knows
//! only about runs of integers.
//!
//! TODO(perf): This persistent shader-visible descriptor allocator is correct
//! because a bind-group-owned range is not returned until the group drops, so a
//! submitted list cannot see its slots rewritten. A frame/ring allocator or
//! descriptor cache must instead defer reuse to the last `CompletionPoint` of
//! every list that can reference a range. Rust `Drop` alone is not sufficient
//! once a range can be recycled independently. Heap slots and retirement stay
//! backend-private; the portable binding and completion APIs already suffice.

use std::ops::Range;
use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DESCRIPTOR_HEAP_DESC,
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, D3D12_DESCRIPTOR_HEAP_TYPE,
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
    D3D12_GPU_DESCRIPTOR_HANDLE, ID3D12DescriptorHeap, ID3D12Device,
};

use crate::backend::dx12::ffi;

/// How many descriptors the one heap holds.
///
/// A fixed capacity rather than a growing set of heaps, because the heap is
/// bound to the command list as a whole: growing it would mean binding a second
/// heap mid-list, and a group allocated from the new heap could not be bound
/// beside one from the old. 65536 CBV/SRV/UAV descriptors is 2 MiB at the
/// increment size Direct3D 12 reports for this heap type, which is a fixed cost
/// this backend pays once per device rather than per group.
///
/// Exhaustion is a refusal ([`super::group`]'s), not a wrong answer: a device
/// handed more live descriptors than this says so, rather than silently binding
/// a group whose range it never got.
const CAPACITY: u32 = 65_536;
/// D3D12 limits a shader-visible sampler heap to 2,048 descriptors.
const SAMPLER_CAPACITY: u32 = 2_048;

/// A shader-visible CBV/SRV/UAV descriptor heap and its free runs.
pub(crate) struct DescriptorHeap {
    heap: ID3D12DescriptorHeap,
    /// How far apart two adjacent descriptors are.
    ///
    /// Read from the device rather than assumed: Direct3D 12 does not fix the
    /// size of a descriptor, and the value differs between heap types and can
    /// differ between drivers. Computing an address by multiplying a literal
    /// would be wrong on any adapter that chose another size.
    increment: u32,
    /// The address of descriptor zero, as the CPU writes to it.
    base_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    /// The address of descriptor zero, as the GPU reads it.
    base_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    capacity: u32,
    /// The runs not handed out, sorted by start and never adjacent to each other.
    ///
    /// The two invariants are what make [`Self::allocate`] a first-fit scan and
    /// [`Self::release`] a single merge, and both are restored by every mutation
    /// below rather than checked by a reader.
    free: Mutex<Vec<Range<u32>>>,
}

impl DescriptorHeap {
    /// Creates the heap and its starting free run.
    pub(crate) fn new(device: &ID3D12Device) -> Result<Self, ffi::NativeError> {
        Self::with_type(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, CAPACITY)
    }

    /// Creates the device-wide shader-visible sampler heap.
    ///
    /// D3D12 permits one sampler heap beside one CBV/SRV/UAV heap on a command
    /// list.  It must therefore be device-wide just like the view heap: a
    /// per-group heap would make two independently live groups unbindable.
    pub(crate) fn new_sampler(device: &ID3D12Device) -> Result<Self, ffi::NativeError> {
        Self::with_type(device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, SAMPLER_CAPACITY)
    }

    fn with_type(
        device: &ID3D12Device,
        heap_type: D3D12_DESCRIPTOR_HEAP_TYPE,
        capacity: u32,
    ) -> Result<Self, ffi::NativeError> {
        let description = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: heap_type,
            NumDescriptors: capacity,
            // Shader-visible, for the reason the module doc gives: a table is
            // named by GPU address and only this heap type has one.
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            // One node, matching the queue and the fence. Linked-node adapters
            // are the multi-GPU feature this backend does not expose.
            NodeMask: 0,
        };
        // SAFETY: `CreateDescriptorHeap` reads the descriptor it is given — a
        // local that outlives the call — and writes one interface pointer the
        // binding converts only on success. The two handle getters take no
        // argument and return a plain address each.
        unsafe {
            let heap = device
                .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&description)
                .map_err(|error| ffi::NativeError::new(&error, "Dx12Device::create_bind_group"))?;
            let increment = device.GetDescriptorHandleIncrementSize(heap_type);
            Ok(Self {
                base_cpu: heap.GetCPUDescriptorHandleForHeapStart(),
                base_gpu: heap.GetGPUDescriptorHandleForHeapStart(),
                heap,
                increment,
                capacity,
                free: Mutex::new(vec![0..capacity]),
            })
        }
    }

    /// The heap interface, for `SetDescriptorHeaps`.
    pub(crate) fn handle(&self) -> &ID3D12DescriptorHeap {
        &self.heap
    }

    /// Claims `count` consecutive descriptors, or `None` if none such run is free.
    ///
    /// First fit, which is what the sorted-and-merged free list makes cheap: the
    /// scan stops at the first run long enough, and a run is split rather than
    /// consumed so the remainder stays available to a smaller request.
    pub(crate) fn allocate(&self, count: u32) -> Option<u32> {
        if count == 0 || count > self.capacity {
            return None;
        }
        let mut free = self.free();
        let index = free.iter().position(|run| run.len() as u32 >= count)?;
        let run = free[index].clone();
        let start = run.start;
        if run.len() as u32 == count {
            free.remove(index);
        } else {
            free[index] = run.start + count..run.end;
        }
        Some(start)
    }

    /// Returns a run claimed by [`Self::allocate`].
    ///
    /// Restores both free-list invariants: the run is inserted in start order,
    /// and any neighbour it touches is merged into one run.
    pub(crate) fn release(&self, start: u32, count: u32) {
        if count == 0 {
            return;
        }
        let mut free = self.free();
        let end = start + count;
        // `partition_point` on a list sorted by start gives the one index the run
        // belongs at, so the insert costs no comparison of its own.
        let at = free.partition_point(|run| run.start < start);
        // The merge is written as "absorb the next, then absorb the previous"
        // rather than as four cases, because the two absorptions are independent
        // and doing them in this order means each touches only the index it is
        // given. A run adjacent to neither is simply inserted.
        let mut merged = start..end;
        if let Some(next) = free.get(at) {
            if next.start == merged.end {
                merged.end = next.end;
                free.remove(at);
            }
        }
        if at > 0 {
            let previous = &mut free[at - 1];
            if previous.end == merged.start {
                previous.end = merged.end;
                return;
            }
        }
        free.insert(at, merged);
    }

    /// The CPU address of the descriptor at `index`.
    pub(crate) fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            // `ptr` is a `usize` here and a `u64` on the GPU-side handle below.
            // The asymmetry is Direct3D 12's own and is the reason the two
            // computations are written out rather than shared through a generic.
            ptr: self.base_cpu.ptr + (index as usize) * (self.increment as usize),
        }
    }

    /// The GPU address of the descriptor at `index`, for a root descriptor table.
    pub(crate) fn gpu(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: self.base_gpu.ptr + (index as u64) * (self.increment as u64),
        }
    }

    /// Borrows the free list, surviving a poisoned lock.
    ///
    /// Recovering rather than propagating, for the reason the spine's own `lock`
    /// gives: the guarded value is a list of integer ranges with no invariant a
    /// panicking holder could have left half-written — every mutation below is a
    /// single `Vec` operation that either happened or did not.
    fn free(&self) -> std::sync::MutexGuard<'_, Vec<Range<u32>>> {
        self.free
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
