//! Step 3's second half: suballocation, and binding memory to a handle.
//!
//! # Why the allocator and the allocations must be owned together
//!
//! `gpu_allocator`'s Vulkan API decides this, and it is worth stating because it is
//! not the shape one would guess: `allocate` takes `&mut self`, and `free` takes the
//! `Allocation` **by value**. An allocation therefore cannot free itself when its
//! resource is dropped, because freeing needs the allocator and a `Drop` body has no
//! way to reach it. Releasing memory has to be driven by whatever owns both.
//!
//! That is why this type exists as a single owner rather than as a helper: the
//! resource table that holds the allocator is also the only thing that may release
//! an allocation, and an `Allocation` handed out without that context would be a
//! leak waiting for someone to guess which allocator frees it. `gpu_allocator`
//! cannot detect a free sent to the wrong allocator, so the structure has to make
//! the mistake unrepresentable rather than detectable.
//!
//! # Memory location is chosen by purpose
//!
//! [`MemoryPurpose`](super::memory::MemoryPurpose) already names the three kinds of
//! memory this backend needs, and `gpu_allocator::MemoryLocation` has a variant for
//! each, so the mapping is one to one and is written once here rather than at each
//! allocation site.

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::{AllocationError, MemoryLocation};

use super::device::VulkanDevice;
use super::memory::{MemoryPurpose, required_flags};

/// Why memory could not be obtained or released.
#[derive(Debug)]
pub(crate) enum AllocatorError {
    /// The allocator could not be created for this device.
    Creation(AllocationError),
    /// The allocation request failed.
    Allocation(AllocationError),
    /// Releasing an allocation failed.
    Release(AllocationError),
    /// No memory type satisfies the purpose, before the allocator is involved.
    ///
    /// Reported with the type count so the diagnostic can say how many types were
    /// considered, which is what distinguishes "this device has no host-visible
    /// memory" from "the request was malformed".
    NoSuitableType {
        /// The purpose that could not be satisfied.
        purpose: MemoryPurpose,
        /// How many memory types the adapter reported.
        considered: usize,
    },
}

/// The suballocator for one device.
pub(crate) struct GpuAllocator {
    inner: Allocator,
}

impl GpuAllocator {
    /// Creates the suballocator for one opened device.
    ///
    /// The instance, device and physical-device handles are cloned into the
    /// allocator, which is why the allocator must be owned beside the device rather
    /// than inside a single resource: it holds its own references to them and must
    /// outlive everything it hands out.
    pub(crate) fn new(
        instance: &ash::Instance,
        device: &VulkanDevice,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, AllocatorError> {
        Self::from_handles(instance, device.device(), physical_device)
    }

    /// The same construction from the bare device handle.
    ///
    /// Device creation needs this shape and nothing else: the table the device owns
    /// is built before the [`VulkanDevice`] value exists, so `new`'s borrow of one is
    /// not available there. The two share this body, so the descriptor -- and the
    /// `buffer_device_address: false` that keeps the allocator off a feature this
    /// device never enables -- is written once.
    pub(crate) fn from_handles(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, AllocatorError> {
        let desc = gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        };
        Allocator::new(&desc)
            .map(|inner| Self { inner })
            .map_err(AllocatorError::Creation)
    }

    /// Allocates memory for `requirements`, in the location `purpose` names.
    ///
    /// `memory_types` is the adapter's reported list, consulted first so that a
    /// purpose no type can serve is refused with a reason that names the purpose
    /// rather than being passed to the allocator, whose message would be about the
    /// request instead of about the device.
    pub(crate) fn allocate(
        &mut self,
        requirements: &vk::MemoryRequirements,
        memory_types: &[vk::MemoryType],
        purpose: MemoryPurpose,
        name: &str,
    ) -> Result<Allocation, AllocatorError> {
        if memory_types.is_empty()
            || !memory_types.iter().any(|memory_type| {
                memory_type.property_flags.contains(required_flags(purpose))
            })
        {
            return Err(AllocatorError::NoSuitableType {
                purpose,
                considered: memory_types.len(),
            });
        }
        let desc = AllocationCreateDesc {
            name,
            requirements: *requirements,
            location: location(purpose),
            // Linear layout is a Direct3D 12 concept that Vulkan's allocator ignores
            // for a non-linear resource; stating it false is the honest value rather
            // than a copied one.
            linear: false,
            // The suballocating scheme: this is what makes the allocator an
            // allocator rather than one `vkAllocateMemory` per resource.
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        self.inner
            .allocate(&desc)
            .map_err(AllocatorError::Allocation)
    }

    /// Releases one allocation.
    ///
    /// Takes the allocation by value because that is what `gpu_allocator` requires,
    /// and that is the mechanism which stops a released allocation from being used
    /// again: there is no value left to bind with.
    pub(crate) fn release(&mut self, allocation: Allocation) -> Result<(), AllocatorError> {
        self.inner.free(allocation).map_err(AllocatorError::Release)
    }
}

impl core::fmt::Debug for GpuAllocator {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.debug_struct("GpuAllocator").finish_non_exhaustive()
    }
}

/// The allocator location for one purpose.
fn location(purpose: MemoryPurpose) -> MemoryLocation {
    match purpose {
        MemoryPurpose::DeviceLocal => MemoryLocation::GpuOnly,
        MemoryPurpose::UploadStaging => MemoryLocation::CpuToGpu,
        MemoryPurpose::ReadbackStaging => MemoryLocation::GpuToCpu,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Validation;
    use crate::native::vulkan::{buffer, memory, open};
    use fluxel_rendergraph::{BufferUsage, BufferUsageKind};

    #[test]
    fn the_purpose_mapping_is_one_to_one() {
        let locations = [
            location(MemoryPurpose::DeviceLocal),
            location(MemoryPurpose::UploadStaging),
            location(MemoryPurpose::ReadbackStaging),
        ];
        assert_eq!(locations[0], MemoryLocation::GpuOnly);
        assert_eq!(locations[1], MemoryLocation::CpuToGpu);
        assert_eq!(locations[2], MemoryLocation::GpuToCpu);
        assert_ne!(locations[0], locations[1]);
        assert_ne!(locations[1], locations[2]);
    }

    #[test]
    fn a_real_buffer_is_allocated_bound_and_released_on_this_machine() {
        // The whole of step 3 against the real driver: create a handle, read its
        // memory requirements, suballocate, bind, then release and destroy in the
        // order Vulkan requires. Skips where no adapter exists.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let mut allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);

        let declared = BufferUsage::from_kinds([BufferUsageKind::Vertex]);
        let info = buffer::create_info(256, declared).expect("a non-zero size");
        // SAFETY: live device, valid description, handle destroyed below.
        let handle = unsafe { opened.device.device().create_buffer(&info, None) }
            .expect("a valid buffer description");
        // SAFETY: the requirements are read for a handle this device created.
        let requirements = unsafe { opened.device.device().get_buffer_memory_requirements(handle) };

        let allocation = allocator
            .allocate(
                &requirements,
                &memory_types,
                MemoryPurpose::DeviceLocal,
                "step-3 contract buffer",
            )
            .expect("device-local memory for a vertex buffer");
        // SAFETY: the memory comes from this device's allocator and is bound once,
        // at the offset the allocation reports; the handle is not bound elsewhere.
        unsafe {
            opened
                .device
                .device()
                .bind_buffer_memory(handle, allocation.memory(), allocation.offset())
        }
        .expect("binding an allocation to the handle it was sized for");

        allocator.release(allocation).expect("release succeeds");
        // SAFETY: the handle is unbound and destroyed once, after its memory was
        // released -- the order Vulkan requires.
        unsafe { opened.device.device().destroy_buffer(handle, None) };
    }
}
