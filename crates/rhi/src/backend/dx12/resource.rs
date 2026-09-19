//! Buffer allocation on Direct3D 12.
//!
//! This is the first lowering in this backend, and its job is the smallest one a
//! lowering has: turn an already-validated [`BufferDescriptor`] into one
//! `ID3D12Resource`. Nothing here decides whether the descriptor is acceptable —
//! size, usage, and the device's own ceiling have all been checked in
//! `Device::create_buffer` before this is reached — so a failure returned from
//! here is one only Direct3D 12 can know, and
//! [`crate::api::RhiErrorKind::InvalidUsage`] is not among the errors this file
//! can produce.
//!
//! # The heap, and why the preference does not select it
//!
//! [`ResourceMemoryPreference`] is a performance hint and never a correctness
//! guarantee (section 11.2), and both of its variants lower onto
//! `D3D12_HEAP_TYPE_DEFAULT`. The reason is not that the distinction was skipped:
//! section 11.2 deleted host-visible buffers from the portable surface, so the
//! only two mutation paths are upload and readback, and both of those are the
//! *transfer* chapter's staging resources — which this backend allocates for
//! itself and which a caller never names. There is therefore no caller-stated
//! preference that could select `UPLOAD` or `READBACK`, and a backend that picked
//! one anyway would be placing a GPU-only resource in host-visible memory where
//! every later `CopyBufferRegion` touching it is slower.
//!
//! # Initial state, and the one flag that is not a hint
//!
//! The initial state is `D3D12_RESOURCE_STATE_COMMON`. Direct3D 12 promotes a
//! resource out of `COMMON` implicitly on first use, so recording a narrower
//! state here would claim knowledge of what the resource will be used for, which
//! is the command chapter's to know and not this one's.
//!
//! `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS` is different in kind: it is not a
//! preference but a *creation-time* grant, and a buffer created without it cannot
//! be a UAV at all — no later barrier or descriptor makes it one. Section 11.1
//! makes `BufferUsage` a creation-time correctness contract for exactly this
//! reason, so `STORAGE` is what turns it on. The flag is granted only when the
//! bit is set, because granting it unconditionally would make every buffer a UAV
//! candidate and quietly relax the contract the caller stated.

use std::any::Any;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE,
    D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_TYPE_UPLOAD,
    D3D12_MEMORY_POOL_UNKNOWN, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_FLAGS,
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ,
    D3D12_RESOURCE_STATES, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ID3D12Device, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

use crate::api::resource::buffer::{BufferDescriptor, BufferUsage, ResourceMemoryPreference};
use crate::base::resource::BufferBackend;

use super::ffi;

/// The native one-byte-granular allocation behind a portable buffer.
pub(crate) struct Dx12Buffer {
    /// The committed resource.
    ///
    /// Held by value: this handle is the allocation, and section 18.6's
    /// last-owner rule is served by `Arc<dyn BufferBackend>` on the portable
    /// side, so dropping this is what actually frees the memory.
    resource: ID3D12Resource,
}

impl Dx12Buffer {
    /// The committed resource.
    ///
    /// Reached by the copy and readback lowering in [`super::command`], which
    /// downcasts through [`BufferBackend::as_any`] from the device's own backend,
    /// and by the provider's test set, which asserts the native description
    /// against the portable descriptor that produced it.
    ///
    /// This accessor carried an `expect(dead_code)` while the command lowering was
    /// unwritten and only the test set reached it. The expectation's stated
    /// reason came true rather than expiring, which is why the attribute is gone
    /// and the field's name never needed an underscore: `resource` is the
    /// allocation, and dropping it is what frees the memory.
    pub(crate) fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }
}

impl BufferBackend for Dx12Buffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Allocates one buffer on `device`.
///
/// `descriptor` has passed
/// [`crate::api::resource::buffer::validate_buffer_descriptor`], so this asks the
/// driver and reports what it says.
///
/// The failure type is [`ffi::NativeError`] rather than [`RhiError`] because the
/// device layer above has one further question about a failure — whether it ended
/// the device — and only the raw code can answer it. Converting here would throw
/// that away and leave the caller to recover it from a message.
pub(super) fn create_buffer(
    device: &ID3D12Device,
    descriptor: &BufferDescriptor,
) -> Result<Dx12Buffer, ffi::NativeError> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: heap_type(descriptor.memory),
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        // One node, visible to one node. Direct3D 12's linked-node adapters are a
        // multi-GPU feature this backend does not expose, and the masks are how a
        // resource says which nodes may touch it; a single-node mask is the
        // honest statement for an adapter this backend selected as one device.
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    };

    let native = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        // Zero asks the driver to choose. A buffer has no mip chain, no array
        // layers, and one sample, and `Format` is `UNKNOWN` because a buffer is
        // byte-addressed in API v1 — the element stride section 12.2 deletes is
        // exactly the thing that would have needed a typed format here.
        Alignment: 0,
        Width: descriptor.size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: resource_flags(descriptor.usage),
    };

    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: `CreateCommittedResource` reads the two descriptors it is given —
    // both are locals that outlive the call — and writes one interface pointer
    // into `resource`, converting only on success. `poptimizedclearvalue` is
    // `None`, which the binding lowers to a null pointer; that is the documented
    // way to say "no optimized clear value", and a buffer has no clear value to
    // optimize for anyway.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &native,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
            .map_err(|error| ffi::NativeError::new(&error, "Device::create_buffer"))?;
    }

    // `S_OK` with a null out-parameter is a contract violation by the driver, not
    // a fact about the request, and there is no `HRESULT` to classify because the
    // call reported success. It is therefore not a `NativeError` but a plain
    // error, and it cannot be terminal for the device: a driver that lies about
    // one allocation has not said it is gone.
    let Some(resource) = resource else {
        return Err(ffi::NativeError::driver_contract_violation(
            "CreateCommittedResource reported success without producing a resource",
            "Device::create_buffer",
        ));
    };

    Ok(Dx12Buffer { resource })
}

/// Which host-visible heap a staging allocation lives in.
///
/// Two variants rather than a bool, because the two are not opposites of one
/// choice: they differ in the heap, in the state Direct3D 12 requires the
/// resource be created in, and in which way the barrier rules permit data to
/// flow. Naming them is what lets each of those three be stated once below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StagingHeap {
    /// Written by the CPU and read by the GPU: an upload's source.
    Upload,
    /// Written by the GPU and read by the CPU: a readback's destination.
    Readback,
}

impl StagingHeap {
    /// The heap type this variant allocates in.
    fn heap_type(self) -> D3D12_HEAP_TYPE {
        match self {
            Self::Upload => D3D12_HEAP_TYPE_UPLOAD,
            Self::Readback => D3D12_HEAP_TYPE_READBACK,
        }
    }

    /// The state Direct3D 12 requires a resource in this heap to be created in.
    ///
    /// This is not a choice and not a hint: `CreateCommittedResource` fails with
    /// `E_INVALIDARG` for any other state in either of these heaps. It is also
    /// the state the resource must *stay* in — neither heap permits a transition
    /// — which is why the command lowering never names a staging resource in a
    /// barrier. See [`super::command`], whose closing invariant would otherwise
    /// have to account for them.
    fn created_state(self) -> D3D12_RESOURCE_STATES {
        match self {
            // The GPU reads it, so the CPU-side union of read states is what the
            // API asks for; a copy source is the only use this backend puts it to.
            Self::Upload => D3D12_RESOURCE_STATE_GENERIC_READ,
            // The GPU writes it, and `COPY_DEST` is the state it must both be
            // created in and remain in.
            Self::Readback => D3D12_RESOURCE_STATE_COPY_DEST,
        }
    }
}

/// Allocates one host-visible staging buffer of `size` bytes.
///
/// # Why this is not `create_buffer` with a different argument
///
/// A portable [`BufferDescriptor`] cannot name one of these, and that is section
/// 11.2's decision rather than an omission here: host-visible buffers were
/// deleted from the portable surface, so the only two mutation paths are upload
/// and readback, and both are *transfer chapter* staging that the backend
/// allocates for itself and a caller never names. The portable descriptor's
/// [`ResourceMemoryPreference`] is a hint about where on the device a resource
/// should live, not a request for a heap the caller may name — which is why the
/// two are separate functions with separate signatures rather than one function
/// reading a field.
///
/// # Test reach
///
/// Nothing outside this backend can call this: the portable upload verb that
/// would reach it through [`super::command`] is not built. It is nevertheless
/// non-test code, because the readback half of the command lowering *is* built
/// and calls it in every build — which is the difference between this and the
/// provider, whose whole module is unreachable outside its tests.
pub(super) fn create_staging(
    device: &ID3D12Device,
    size: u64,
    heap: StagingHeap,
) -> Result<Dx12Buffer, ffi::NativeError> {
    let properties = D3D12_HEAP_PROPERTIES {
        Type: heap.heap_type(),
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    };

    let native = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        // No creation-time grant, and the absence is deliberate: neither heap can
        // hold a resource that is an unordered-access target, so the flag would
        // be refused rather than ignored.
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };

    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: `CreateCommittedResource` reads the two descriptors it is given —
    // both are locals that outlive the call — and writes one interface pointer
    // into `resource`, converting only on success. `poptimizedclearvalue` is
    // `None`, the documented way to say "no optimized clear value", and a buffer
    // has no clear value to optimize for.
    unsafe {
        device
            .CreateCommittedResource(
                &properties,
                D3D12_HEAP_FLAG_NONE,
                &native,
                heap.created_state(),
                None,
                &mut resource,
            )
            .map_err(|error| ffi::NativeError::new(&error, "Dx12Device::submit"))?;
    }

    let Some(resource) = resource else {
        return Err(ffi::NativeError::driver_contract_violation(
            "CreateCommittedResource reported success without producing a staging resource",
            "Dx12Device::submit",
        ));
    };

    Ok(Dx12Buffer { resource })
}

/// The heap type `preference` lowers onto.
///
/// One answer for both variants, and the module documentation states why: the
/// variants differ in *where on the device* the caller would like the memory, and
/// Direct3D 12's heap types differ in *who may touch it at all*. There is no
/// second device-local heap type to choose between, so the mapping is total
/// rather than lossy. The match is written out rather than collapsed so that
/// adding a variant to the portable enum is a compile error here rather than a
/// silently inherited default.
fn heap_type(preference: ResourceMemoryPreference) -> D3D12_HEAP_TYPE {
    match preference {
        ResourceMemoryPreference::Automatic | ResourceMemoryPreference::DeviceLocalPreferred => {
            D3D12_HEAP_TYPE_DEFAULT
        }
    }
}

/// The creation-time flags `usage` requires.
fn resource_flags(usage: BufferUsage) -> D3D12_RESOURCE_FLAGS {
    if usage.contains(BufferUsage::STORAGE) {
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    } else {
        D3D12_RESOURCE_FLAG_NONE
    }
}
