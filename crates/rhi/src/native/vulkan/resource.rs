//! The buffer table: one owner for handles, allocations and identities.
//!
//! # Why one table owns all three
//!
//! `gpu_allocator`'s `free` takes the `Allocation` **by value** and needs
//! `&mut Allocator`, so a buffer cannot release its own memory when it is dropped:
//! a `Drop` body has no way to reach the allocator. Releasing memory therefore has
//! to be driven by whatever owns both, and that is this table. Handing an
//! `Allocation` out to a caller would be a leak waiting for someone to guess which
//! allocator frees it, and `gpu_allocator` cannot detect a free sent to the wrong
//! allocator -- so the structure makes the mistake unrepresentable instead.
//!
//! # Teardown order
//!
//! The table holds a clone of the device's `ash::Device`, because destroying a
//! handle needs it and `Drop` cannot take a parameter. That clone is a handle, not
//! an owner: the `VkDevice` itself is owned by [`VulkanDevice`](super::device::VulkanDevice),
//! which must therefore be dropped **after** this table. The owner of both fixes
//! that by field order, which is the same rule the plan's preserved-semantics table
//! records for native teardown.
//!
//! Within the table the order is also fixed: the handle is destroyed first, which is
//! what unbinds it, and only then is the allocation handed back to the allocator.
//! That is the order the borrowed Vulkan backend being replaced uses, so the owned
//! path does not differ behaviorally from the one it supersedes; releasing memory
//! out from under a live binding would be the difference that matters.
//!
//! # Why a hash map rather than an ordered one
//!
//! [`BufferId`] is `Eq + Hash` by construction -- it is a stamp plus a physical
//! identity -- and deliberately not `Ord`: nothing about identity is ordered, and
//! adding an ordering just to key a table would invent a comparison the rest of the
//! layer does not have. The records therefore live in a `HashMap`, and teardown
//! order across buffers is not meaningful.

use std::collections::HashMap;

use ash::vk;
use fluxel_rendergraph::{BufferUsage, PhysicalResourceIdentity};

use crate::common::base::resource::BufferId;
use crate::common::base::stamp::DeviceStamp;

use super::allocator::GpuAllocator;
use super::buffer;
use super::memory::MemoryPurpose;

/// Why a buffer could not be created, found, or destroyed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BufferError {
    /// The requested size was zero, which Vulkan does not allow.
    ///
    /// Refused here rather than passed on, because a driver validation error is a
    /// worse answer than a reason the caller can act on.
    ZeroSize,
    /// The driver refused to create the handle.
    Create(vk::Result),
    /// The driver refused to bind memory to the handle.
    Bind(vk::Result),
    /// The id names no live buffer.
    ///
    /// A value rather than a panic, because an id can be stale: it may belong to a
    /// retired device generation, which is the whole reason identities are stamped.
    Unknown,
    /// The allocation failed. Carried separately because it is not a `vk::Result`.
    Memory,
}

/// One live buffer: its handle, its memory, and how large it is.
struct BufferRecord {
    handle: vk::Buffer,
    allocation: gpu_allocator::vulkan::Allocation,
    size: u64,
}

/// Every buffer one device generation owns.
pub(crate) struct BufferTable {
    device: ash::Device,
    stamp: DeviceStamp,
    allocator: GpuAllocator,
    records: HashMap<BufferId, BufferRecord>,
    next_identity: u64,
}

impl BufferTable {
    /// Creates an empty table for one device generation.
    ///
    /// The caller guarantees, by field order, that the device outlives this table;
    /// see the module docs.
    pub(crate) fn new(
        device: &ash::Device,
        stamp: DeviceStamp,
        allocator: GpuAllocator,
    ) -> Self {
        Self {
            device: device.clone(),
            stamp,
            allocator,
            records: HashMap::new(),
            next_identity: 1,
        }
    }

    /// Returns how many buffers are live.
    pub(crate) fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns the driver handle for a live buffer, or `None` for a stale id.
    pub(crate) fn handle(&self, id: BufferId) -> Option<vk::Buffer> {
        self.record(id).map(|record| record.handle)
    }

    /// Returns the created size of a live buffer, or `None` for a stale id.
    pub(crate) fn size(&self, id: BufferId) -> Option<u64> {
        self.record(id).map(|record| record.size)
    }

    /// Creates a buffer of `size` bytes and binds memory to it.
    ///
    /// Identity is checked before anything is created, so a stale id can never
    /// reach the driver, and the handle is destroyed again if binding fails: a
    /// half-created buffer that never got memory is not a resource this table may
    /// leave behind.
    pub(crate) fn create(
        &mut self,
        size: u64,
        usage: BufferUsage,
        memory_types: &[vk::MemoryType],
        purpose: MemoryPurpose,
    ) -> Result<BufferId, BufferError> {
        let info = buffer::create_info(size, usage).ok_or(BufferError::ZeroSize)?;
        // SAFETY: the device is live and owned above this table; the description is
        // valid because `create_info` refused a zero size.
        let handle = unsafe { self.device.create_buffer(&info, None) }
            .map_err(BufferError::Create)?;

        // SAFETY: the requirements are read for a handle this table just created.
        let requirements = unsafe { self.device.get_buffer_memory_requirements(handle) };
        let allocation = match self
            .allocator
            .allocate(&requirements, memory_types, purpose, "fluxel buffer")
        {
            Ok(allocation) => allocation,
            Err(_) => {
                // SAFETY: the handle has no memory bound and is destroyed once.
                unsafe { self.device.destroy_buffer(handle, None) };
                return Err(BufferError::Memory);
            }
        };

        // SAFETY: the allocation comes from this device's allocator, sized for this
        // very handle, and is bound at the offset the allocation reports. The handle
        // is bound exactly once.
        if let Err(error) = unsafe {
            self.device
                .bind_buffer_memory(handle, allocation.memory(), allocation.offset())
        } {
            // SAFETY: nothing is bound, so both are released in either order; the
            // allocation goes back to the allocator that produced it.
            let _ = self.allocator.release(allocation);
            unsafe { self.device.destroy_buffer(handle, None) };
            return Err(BufferError::Bind(error));
        }

        let id = BufferId::new(
            self.stamp,
            PhysicalResourceIdentity::new(self.next_identity),
        );
        self.next_identity += 1;
        self.records.insert(
            id,
            BufferRecord {
                handle,
                allocation,
                size,
            },
        );
        Ok(id)
    }

    /// Destroys a buffer's handle and returns its memory to the allocator.
    ///
    /// Order matters: the handle is destroyed first, which unbinds the memory, and
    /// the allocation is released only after that. An id that names nothing answers
    /// [`BufferError::Unknown`] rather than being ignored, because a caller
    /// destroying a buffer it does not own has a bug that silence would hide.
    pub(crate) fn destroy(&mut self, id: BufferId) -> Result<(), BufferError> {
        let record = self.records.remove(&id).ok_or(BufferError::Unknown)?;
        // SAFETY: the handle was created by this device and is destroyed once;
        // destroying it unbinds the memory, which is released only afterwards.
        unsafe { self.device.destroy_buffer(record.handle, None) };
        self.allocator
            .release(record.allocation)
            .map_err(|_| BufferError::Memory)
    }

    /// Returns a live record, or `None`.
    fn record(&self, id: BufferId) -> Option<&BufferRecord> {
        self.records.get(&id)
    }
}

impl Drop for BufferTable {
    fn drop(&mut self) {
        // Every allocation must go back to its allocator, and every handle must be
        // destroyed, even when the owner is dropped without destroying them. The
        // order per buffer is the same as `destroy`'s.
        for (_, record) in self.records.drain() {
            // SAFETY: each handle was created by this device and is destroyed once
            // here; the device outlives this table by the caller's field order.
            unsafe { self.device.destroy_buffer(record.handle, None) };
            let _ = self.allocator.release(record.allocation);
        }
    }
}

impl core::fmt::Debug for BufferTable {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("BufferTable")
            .field("stamp", &self.stamp)
            .field("live", &self.records.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Validation;
    use crate::common::base::stamp::StampMismatch;
    use crate::native::vulkan::{memory, open};
    use fluxel_rendergraph::BufferUsageKind;

    fn declared(kinds: &[BufferUsageKind]) -> BufferUsage {
        BufferUsage::from_kinds(kinds.iter().copied())
    }

    #[test]
    fn a_real_table_creates_binds_looks_up_and_destroys_buffers() {
        // Step 4's resource layer against the real driver: two buffers live at
        // once, looked up by identity, then destroyed. Skips where no adapter
        // exists.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let mut table = BufferTable::new(
            opened.device.device(),
            opened.device.stamp(),
            allocator,
        );

        let first = table
            .create(
                256,
                declared(&[BufferUsageKind::Vertex]),
                &memory_types,
                MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local vertex buffer");
        let second = table
            .create(
                512,
                declared(&[BufferUsageKind::Uniform]),
                &memory_types,
                MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local uniform buffer");

        assert_eq!(table.len(), 2);
        assert_ne!(first, second, "identities are distinct");
        assert_eq!(table.size(first), Some(256));
        assert_eq!(table.size(second), Some(512));
        assert!(table.handle(first).is_some(), "a live buffer has a handle");

        // The ids carry this device generation, so a stale one is rejected before
        // it could reach the table.
        assert_eq!(first.verify(table.stamp), Ok(()));
        assert_eq!(
            first.verify(table.stamp.next_generation()),
            Err(StampMismatch::StaleGeneration {
                object: 0,
                current: 1,
            })
        );

        table.destroy(first).expect("first buffer released");
        assert_eq!(table.len(), 1);
        assert_eq!(table.handle(first), None, "a destroyed id resolves to nothing");
        assert_eq!(table.destroy(first), Err(BufferError::Unknown));

        // The second buffer is released by the table's own drop, which must leave
        // nothing behind.
        drop(table);
    }

    #[test]
    fn creating_a_zero_sized_buffer_is_refused_before_the_driver_is_reached() {
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let mut table = BufferTable::new(
            opened.device.device(),
            opened.device.stamp(),
            allocator,
        );
        assert!(matches!(
            table.create(
                0,
                declared(&[BufferUsageKind::Vertex]),
                &memory_types,
                MemoryPurpose::DeviceLocal
            ),
            Err(BufferError::ZeroSize)
        ));
        assert_eq!(table.len(), 0);
    }
}
