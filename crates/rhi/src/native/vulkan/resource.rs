//! The resource table: one owner for handles, allocations and identities.
//!
//! # Why one table owns all of them
//!
//! `gpu_allocator`'s `free` takes the `Allocation` **by value** and needs
//! `&mut Allocator`, so a resource cannot release its own memory when it is dropped:
//! a `Drop` body has no way to reach the allocator. Releasing memory therefore has
//! to be driven by whatever owns both, and that is this table. Handing an
//! `Allocation` out to a caller would be a leak waiting for someone to guess which
//! allocator frees it, and `gpu_allocator` cannot detect a free sent to the wrong
//! allocator -- so the structure makes the mistake unrepresentable instead.
//!
//! Buffers and textures share the one table rather than each owning an allocator,
//! because `gpu_allocator` suballocates whole `vkDeviceMemory` blocks: a second
//! allocator on the same device would be a second set of blocks for memory the
//! driver cannot move between them. One table also means one identity counter, so a
//! buffer and a texture can never be handed the same physical identity.
//!
//! Samplers are in this table for the second of those reasons rather than the first:
//! a sampler owns no memory and needs no allocation, but it is a device-owned handle
//! that must be destroyed exactly once and must carry an identity a stale generation
//! cannot reuse. Keeping them beside the rest means one owner answers for every
//! handle this backend creates, and a caller has one identity space to check.
//!
//! # Teardown order
//!
//! The table holds a clone of the device's `ash::Device`, because destroying a
//! handle needs it and `Drop` cannot take a parameter. That clone is a handle, not
//! an owner: the `VkDevice` itself is owned by
//! [`VulkanDevice`](super::device::VulkanDevice), which must therefore be dropped
//! **after** this table. The owner of both fixes that by field order, which is the
//! same rule the plan's preserved-semantics table records for native teardown.
//!
//! Within the table the order is also fixed. A buffer's handle is destroyed first,
//! which is what unbinds it, and only then is the allocation handed back to the
//! allocator. A texture adds one step in front: its view is destroyed before its
//! image, because a view refers to an image and not the other way round, and the
//! image is what binds the memory. Samplers are destroyed first, before both, and
//! that placement claims nothing: a sampler refers to no image and binds no memory,
//! so it has no ordering dependency on any other record. That is the order the
//! borrowed Vulkan backend being replaced uses, so the owned path does not differ
//! behaviorally from the one it supersedes; releasing memory out from under a live
//! binding would be the difference that matters.
//!
//! # Why a hash map rather than an ordered one
//!
//! [`BufferId`] and [`TextureId`] are `Eq + Hash` by construction -- a stamp plus a
//! physical identity -- and deliberately not `Ord`: nothing about identity is
//! ordered, and adding an ordering just to key a table would invent a comparison the
//! rest of the layer does not have. The records therefore live in `HashMap`s, and
//! teardown order across resources is not meaningful.

use std::collections::HashMap;

use ash::vk;
use fluxel_rendergraph::{BufferUsage, PhysicalResourceIdentity, TextureDesc, TextureUsage};

use crate::common::base::resource::{BufferId, ResourceId, ResourceKind, SamplerId, TextureId};
use crate::common::base::stamp::DeviceStamp;
use crate::common::sampler::SamplerDescriptor;

use super::allocator::GpuAllocator;
use super::memory::MemoryPurpose;
use super::{buffer, sampler, texture};

/// Why a resource could not be created, found, or destroyed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResourceError {
    /// The requested buffer size was zero, which Vulkan does not allow.
    ///
    /// Refused here rather than passed on, because a driver validation error is a
    /// worse answer than a reason the caller can act on.
    ZeroSize,
    /// The texture description names a dimension, format, sample count or extent
    /// this backend cannot honour.
    ///
    /// A different sentence from [`Self::ZeroSize`] because the caller's mistake is
    /// different: a zero-sized buffer is a bad length, while an unsupported texture
    /// is a description this backend has not been taught.
    UnsupportedTexture,
    /// The sampler description is internally inconsistent, so it is not a description
    /// the driver could accept.
    ///
    /// Today that is an inverted LOD range. Kept separate from
    /// [`Self::UnsupportedTexture`] because "this backend has not been taught the
    /// description" and "this description contradicts itself" are different things
    /// for a caller to fix.
    InvalidSampler,
    /// The driver refused to create a handle (buffer, image, view or sampler).
    Create(vk::Result),
    /// The driver refused to bind memory to the handle.
    Bind(vk::Result),
    /// The driver refused to create the image view of a texture.
    View(vk::Result),
    /// The id names no live resource of its kind.
    ///
    /// A value rather than a panic, because an id can be stale: it may belong to a
    /// retired device generation, which is the whole reason identities are stamped.
    Unknown,
    /// The allocation failed, or could not be released. Carried separately because
    /// it is not a `vk::Result`.
    Memory,
}

/// One live buffer: its handle, its memory, how large it is and what it was declared
/// for.
///
/// The declared usage is kept beside the size for the same reason the size is: a
/// binding built after creation must be checked against what the caller declared, not
/// against what the driver happens to allow. The buffer step's rule is that the
/// mapping never widens (section 18 of the lead 3F plan), so a storage binding over a
/// buffer created for vertices is a claim the graph never made.
struct BufferRecord {
    handle: vk::Buffer,
    allocation: gpu_allocator::vulkan::Allocation,
    size: u64,
    usage: BufferUsage,
}

/// One live texture: its image, the view it is sampled through, and its memory.
struct TextureRecord {
    image: vk::Image,
    view: vk::ImageView,
    allocation: gpu_allocator::vulkan::Allocation,
    desc: TextureDesc,
}

/// Every resource one device generation owns, over one allocator.
pub(crate) struct ResourceTable {
    device: ash::Device,
    stamp: DeviceStamp,
    allocator: GpuAllocator,
    buffers: HashMap<BufferId, BufferRecord>,
    textures: HashMap<TextureId, TextureRecord>,
    samplers: HashMap<SamplerId, vk::Sampler>,
    next_identity: u64,
}

impl ResourceTable {
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
            buffers: HashMap::new(),
            textures: HashMap::new(),
            samplers: HashMap::new(),
            next_identity: 1,
        }
    }

    /// Returns how many buffers are live.
    pub(crate) fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Returns how many textures are live.
    pub(crate) fn texture_count(&self) -> usize {
        self.textures.len()
    }

    /// Returns how many samplers are live.
    pub(crate) fn sampler_count(&self) -> usize {
        self.samplers.len()
    }

    /// Returns the driver handle for a live buffer, or `None` for a stale id.
    pub(crate) fn buffer_handle(&self, id: BufferId) -> Option<vk::Buffer> {
        self.buffer(id).map(|record| record.handle)
    }

    /// Returns the created size of a live buffer, or `None` for a stale id.
    pub(crate) fn buffer_size(&self, id: BufferId) -> Option<u64> {
        self.buffer(id).map(|record| record.size)
    }

    /// Returns the usage a live buffer was created with, or `None` for a stale id.
    ///
    /// The *declared* usage, not a reading of the driver's create-info: it is the
    /// same value the graph's own capability check saw, so a binding built later
    /// cannot be checked against a weaker fact than the one that admitted the buffer.
    pub(crate) fn buffer_usage(&self, id: BufferId) -> Option<BufferUsage> {
        self.buffer(id).map(|record| record.usage)
    }

    /// Returns the image handle for a live texture, or `None` for a stale id.
    ///
    /// The image rather than the view, because the copy path addresses subresources
    /// of the image itself.
    pub(crate) fn texture_image(&self, id: TextureId) -> Option<vk::Image> {
        self.texture(id).map(|record| record.image)
    }

    /// Returns the view a live texture is sampled through, or `None` for a stale id.
    pub(crate) fn texture_view(&self, id: TextureId) -> Option<vk::ImageView> {
        self.texture(id).map(|record| record.view)
    }

    /// Returns the description a live texture was created from, or `None` for a
    /// stale id.
    ///
    /// The description is what a copy's range checks address against, so it is kept
    /// rather than recovered from the driver.
    pub(crate) fn texture_desc(&self, id: TextureId) -> Option<TextureDesc> {
        self.texture(id).map(|record| record.desc)
    }

    /// Returns the handle for a live sampler, or `None` for a stale id.
    ///
    /// A sampler has no memory and no view, so the handle is the whole resource and
    /// this is the only accessor it needs.
    pub(crate) fn sampler_handle(&self, id: SamplerId) -> Option<vk::Sampler> {
        self.samplers.get(&id).copied()
    }

    /// Creates a buffer of `size` bytes and binds memory to it.
    ///
    /// Identity is checked before anything is created, so a stale id can never
    /// reach the driver, and the handle is destroyed again if binding fails: a
    /// half-created buffer that never got memory is not a resource this table may
    /// leave behind.
    pub(crate) fn create_buffer(
        &mut self,
        size: u64,
        usage: BufferUsage,
        memory_types: &[vk::MemoryType],
        purpose: MemoryPurpose,
    ) -> Result<BufferId, ResourceError> {
        let info = buffer::create_info(size, usage).ok_or(ResourceError::ZeroSize)?;
        // SAFETY: the device is live and owned above this table; the description is
        // valid because `create_info` refused a zero size.
        let handle =
            unsafe { self.device.create_buffer(&info, None) }.map_err(ResourceError::Create)?;

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
                return Err(ResourceError::Memory);
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
            return Err(ResourceError::Bind(error));
        }

        let id = self.next_id();
        self.buffers.insert(
            id,
            BufferRecord {
                handle,
                allocation,
                size,
                usage,
            },
        );
        Ok(id)
    }

    /// Creates a texture and the view it is sampled through, and binds memory to it.
    ///
    /// The view is created as part of the texture rather than on demand, because the
    /// two share one lifetime: a texture with no view is not a resource a graph may
    /// sample, and a view outliving its image is invalid. Every failure path
    /// destroys what it created, in the order Vulkan requires, so a refused texture
    /// leaves neither a handle nor an allocation behind.
    pub(crate) fn create_texture(
        &mut self,
        desc: TextureDesc,
        usage: TextureUsage,
        memory_types: &[vk::MemoryType],
        purpose: MemoryPurpose,
    ) -> Result<TextureId, ResourceError> {
        let info =
            texture::image_create_info(&desc, usage).ok_or(ResourceError::UnsupportedTexture)?;
        // SAFETY: the device is live and owned above this table, and the description
        // was validated above.
        let image =
            unsafe { self.device.create_image(&info, None) }.map_err(ResourceError::Create)?;

        // SAFETY: the requirements are read for a handle this table just created.
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let allocation = match self
            .allocator
            .allocate(&requirements, memory_types, purpose, "fluxel texture")
        {
            Ok(allocation) => allocation,
            Err(_) => {
                // SAFETY: the image has no memory bound and is destroyed once.
                unsafe { self.device.destroy_image(image, None) };
                return Err(ResourceError::Memory);
            }
        };

        // SAFETY: the allocation comes from this device's allocator, sized for this
        // very image, and is bound at the offset the allocation reports. The image is
        // bound exactly once.
        if let Err(error) = unsafe {
            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        } {
            // SAFETY: nothing is bound, so both are released in either order.
            let _ = self.allocator.release(allocation);
            unsafe { self.device.destroy_image(image, None) };
            return Err(ResourceError::Bind(error));
        }

        // From here on the image is bound, so teardown is always image first -- that
        // is what unbinds it -- and allocation release only afterwards.
        let Some(view_info) = texture::view_create_info(&desc, image) else {
            // A description that got this far but cannot build a view is one
            // `image_create_info` should have refused; the image is torn down anyway
            // rather than left as a resource with no way to be sampled.
            // SAFETY: the image is bound and destroyed once.
            unsafe { self.device.destroy_image(image, None) };
            let _ = self.allocator.release(allocation);
            return Err(ResourceError::UnsupportedTexture);
        };
        // SAFETY: the device is live and the view description refers to the image
        // this table just created and bound.
        let view = match unsafe { self.device.create_image_view(&view_info, None) } {
            Ok(view) => view,
            Err(error) => {
                // SAFETY: the image is bound and destroyed once.
                unsafe { self.device.destroy_image(image, None) };
                let _ = self.allocator.release(allocation);
                return Err(ResourceError::View(error));
            }
        };

        let id = self.next_id();
        self.textures.insert(
            id,
            TextureRecord {
                image,
                view,
                allocation,
                desc,
            },
        );
        Ok(id)
    }

    /// Destroys a buffer's handle and returns its memory to the allocator.
    ///
    /// Order matters: the handle is destroyed first, which unbinds the memory, and
    /// the allocation is released only after that. An id that names nothing answers
    /// [`ResourceError::Unknown`] rather than being ignored, because a caller
    /// destroying a buffer it does not own has a bug that silence would hide.
    pub(crate) fn destroy_buffer(&mut self, id: BufferId) -> Result<(), ResourceError> {
        let record = self.buffers.remove(&id).ok_or(ResourceError::Unknown)?;
        // SAFETY: the handle was created by this device and is destroyed once;
        // destroying it unbinds the memory, which is released only afterwards.
        unsafe { self.device.destroy_buffer(record.handle, None) };
        self.allocator
            .release(record.allocation)
            .map_err(|_| ResourceError::Memory)
    }

    /// Destroys a texture's view and image and returns its memory to the allocator.
    ///
    /// The view goes first because it refers to the image, then the image, which
    /// unbinds the memory, and the allocation only after that.
    pub(crate) fn destroy_texture(&mut self, id: TextureId) -> Result<(), ResourceError> {
        let record = self.textures.remove(&id).ok_or(ResourceError::Unknown)?;
        // SAFETY: the view and image were created by this device and are each
        // destroyed once, view before image; destroying the image unbinds the memory,
        // which is released only afterwards.
        unsafe { self.device.destroy_image_view(record.view, None) };
        unsafe { self.device.destroy_image(record.image, None) };
        self.allocator
            .release(record.allocation)
            .map_err(|_| ResourceError::Memory)
    }

    /// Creates a sampler from `desc`.
    ///
    /// A sampler is the one resource here with nothing to bind and nothing to
    /// allocate, so creation is a single call and there is no failure path to undo.
    /// The description is validated first anyway, so an inconsistent one is refused
    /// with a reason rather than by the driver with a validation error.
    pub(crate) fn create_sampler(
        &mut self,
        desc: &SamplerDescriptor,
    ) -> Result<SamplerId, ResourceError> {
        let info = sampler::create_info(desc).ok_or(ResourceError::InvalidSampler)?;
        // SAFETY: the device is live and owned above this table, and the description
        // was validated above.
        let handle = unsafe { self.device.create_sampler(&info, None) }
            .map_err(ResourceError::Create)?;
        let id = self.next_id();
        self.samplers.insert(id, handle);
        Ok(id)
    }

    /// Destroys a sampler's handle.
    ///
    /// The only teardown step: a sampler owns no memory, refers to no image and has
    /// no dependency to honour, which is why it neither releases an allocation nor
    /// waits on another record.
    pub(crate) fn destroy_sampler(&mut self, id: SamplerId) -> Result<(), ResourceError> {
        let handle = self.samplers.remove(&id).ok_or(ResourceError::Unknown)?;
        // SAFETY: the handle was created by this device and is destroyed once.
        unsafe { self.device.destroy_sampler(handle, None) };
        Ok(())
    }

    /// Mints the next identity for this device generation.
    ///
    /// One counter serves both kinds, so a buffer and a texture can never be handed
    /// the same identity even though they live in different maps.
    fn next_id<K: ResourceKind>(&mut self) -> ResourceId<K> {
        let identity = PhysicalResourceIdentity::new(self.next_identity);
        self.next_identity += 1;
        ResourceId::new(self.stamp, identity)
    }

    /// Returns a live buffer record, or `None`.
    fn buffer(&self, id: BufferId) -> Option<&BufferRecord> {
        self.buffers.get(&id)
    }

    /// Returns a live texture record, or `None`.
    fn texture(&self, id: TextureId) -> Option<&TextureRecord> {
        self.textures.get(&id)
    }
}

impl Drop for ResourceTable {
    fn drop(&mut self) {
        // Every allocation must go back to its allocator, and every handle must be
        // destroyed, even when the owner is dropped without destroying them. The
        // order per resource is the same as `destroy_buffer` / `destroy_texture` /
        // `destroy_sampler`; the sampler loop is first only because a sampler has no
        // dependency on anything, not because anything requires that order.
        for (_, handle) in self.samplers.drain() {
            // SAFETY: each handle was created by this device and is destroyed once
            // here; the device outlives this table by the caller's field order.
            unsafe { self.device.destroy_sampler(handle, None) };
        }
        for (_, record) in self.textures.drain() {
            // SAFETY: each view and image was created by this device and is destroyed
            // once here, view before image; the device outlives this table by the
            // caller's field order.
            unsafe { self.device.destroy_image_view(record.view, None) };
            unsafe { self.device.destroy_image(record.image, None) };
            let _ = self.allocator.release(record.allocation);
        }
        for (_, record) in self.buffers.drain() {
            // SAFETY: each handle was created by this device and is destroyed once
            // here; the device outlives this table by the caller's field order.
            unsafe { self.device.destroy_buffer(record.handle, None) };
            let _ = self.allocator.release(record.allocation);
        }
    }
}

impl core::fmt::Debug for ResourceTable {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ResourceTable")
            .field("stamp", &self.stamp)
            .field("buffers", &self.buffers.len())
            .field("textures", &self.textures.len())
            .field("samplers", &self.samplers.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Validation;
    use crate::common::base::stamp::StampMismatch;
    use crate::common::sampler::{CompareFunction, SamplerDescriptor};
    use crate::native::vulkan::{memory, open};
    use fluxel_rendergraph::{
        BufferUsageKind, Extent3d, TextureDimension, TextureFormat, TextureUsageKind,
    };

    fn declared(kinds: &[BufferUsageKind]) -> BufferUsage {
        BufferUsage::from_kinds(kinds.iter().copied())
    }

    fn texture_desc(format: TextureFormat, sample_count: u32) -> TextureDesc {
        TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: 16,
                height: 8,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count,
            format,
        }
    }

    fn texture_usage(kinds: &[TextureUsageKind]) -> TextureUsage {
        TextureUsage::from_kinds(kinds.iter().copied())
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
        let mut table = ResourceTable::new(
            opened.device.device(),
            opened.device.stamp(),
            allocator,
        );

        let first = table
            .create_buffer(
                256,
                declared(&[BufferUsageKind::Vertex]),
                &memory_types,
                MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local vertex buffer");
        let second = table
            .create_buffer(
                512,
                declared(&[BufferUsageKind::Uniform]),
                &memory_types,
                MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local uniform buffer");

        assert_eq!(table.buffer_count(), 2);
        assert_ne!(first, second, "identities are distinct");
        assert_eq!(table.buffer_size(first), Some(256));
        assert_eq!(table.buffer_size(second), Some(512));
        assert!(
            table.buffer_handle(first).is_some(),
            "a live buffer has a handle"
        );

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

        table.destroy_buffer(first).expect("first buffer released");
        assert_eq!(table.buffer_count(), 1);
        assert_eq!(
            table.buffer_handle(first),
            None,
            "a destroyed id resolves to nothing"
        );
        assert_eq!(table.destroy_buffer(first), Err(ResourceError::Unknown));

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
        let mut table = ResourceTable::new(
            opened.device.device(),
            opened.device.stamp(),
            allocator,
        );
        assert!(matches!(
            table.create_buffer(
                0,
                declared(&[BufferUsageKind::Vertex]),
                &memory_types,
                MemoryPurpose::DeviceLocal
            ),
            Err(ResourceError::ZeroSize)
        ));
        assert_eq!(table.buffer_count(), 0);
    }

    #[test]
    fn a_real_table_creates_an_image_view_and_destroys_them_together() {
        // The texture half of step 4 against the real driver: the table owns the
        // image, the memory bound to it and the view it is sampled through, and
        // releases all three together. Skips where no adapter exists.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let mut table = ResourceTable::new(
            opened.device.device(),
            opened.device.stamp(),
            allocator,
        );

        let described = texture_desc(TextureFormat::Rgba8Unorm, 1);
        let texture = table
            .create_texture(
                described,
                texture_usage(&[TextureUsageKind::Sampled, TextureUsageKind::CopyDestination]),
                &memory_types,
                MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local sampled texture");

        assert_eq!(table.texture_count(), 1);
        assert_eq!(table.texture_desc(texture), Some(described));
        assert!(
            table.texture_image(texture).is_some(),
            "a live texture has an image"
        );
        assert!(
            table.texture_view(texture).is_some(),
            "a live texture has the view it is sampled through"
        );
        assert_eq!(texture.verify(table.stamp), Ok(()));

        // Both live at once, and one identity counter means the two kinds never
        // share one: the buffer and the texture are distinguishable even though
        // their ids are different types.
        let buffer = table
            .create_buffer(
                256,
                declared(&[BufferUsageKind::Vertex]),
                &memory_types,
                MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local vertex buffer");
        assert_ne!(buffer.identity(), texture.identity());

        table.destroy_texture(texture).expect("texture released");
        assert_eq!(table.texture_count(), 0);
        assert_eq!(table.texture_view(texture), None);
        assert_eq!(table.texture_image(texture), None);
        assert_eq!(
            table.destroy_texture(texture),
            Err(ResourceError::Unknown)
        );

        // The buffer is released by the table's own drop, which must leave nothing
        // behind.
        drop(table);
    }

    #[test]
    fn an_unsupported_texture_is_refused_before_the_driver_is_reached() {
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let mut table = ResourceTable::new(
            opened.device.device(),
            opened.device.stamp(),
            allocator,
        );

        // Nothing has proved a multisample row and no recipe asks for one.
        assert!(matches!(
            table.create_texture(
                texture_desc(TextureFormat::Rgba8Unorm, 4),
                texture_usage(&[TextureUsageKind::Sampled]),
                &memory_types,
                MemoryPurpose::DeviceLocal
            ),
            Err(ResourceError::UnsupportedTexture)
        ));
        assert_eq!(table.texture_count(), 0);

        let mut zero = texture_desc(TextureFormat::Rgba8Unorm, 1);
        zero.extent.width = 0;
        assert!(matches!(
            table.create_texture(
                zero,
                texture_usage(&[TextureUsageKind::Sampled]),
                &memory_types,
                MemoryPurpose::DeviceLocal
            ),
            Err(ResourceError::UnsupportedTexture)
        ));
        assert_eq!(table.texture_count(), 0);
    }

    #[test]
    fn a_real_table_creates_looks_up_and_destroys_samplers() {
        // Step 4's sampler half against the real driver: a sampler owns no memory
        // and no view, so identity, the handle and destruction are the whole
        // contract. Skips where no adapter exists.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let mut table = ResourceTable::new(
            opened.device.device(),
            opened.device.stamp(),
            allocator,
        );

        let linear = table
            .create_sampler(&SamplerDescriptor::linear_clamp())
            .expect("a linear-clamp sampler");
        let comparing = table
            .create_sampler(&SamplerDescriptor {
                compare: Some(CompareFunction::LessEqual),
                ..SamplerDescriptor::nearest_clamp()
            })
            .expect("a comparison sampler");

        assert_eq!(table.sampler_count(), 2);
        assert_ne!(linear, comparing, "identities are distinct");
        assert!(
            table.sampler_handle(linear).is_some(),
            "a live sampler has a handle"
        );
        // The one identity counter serves every kind, so a sampler cannot be handed
        // an identity a buffer or texture already used.
        let buffer = table
            .create_buffer(
                256,
                declared(&[BufferUsageKind::Vertex]),
                &memory_types,
                MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local vertex buffer");
        assert_ne!(buffer.identity(), linear.identity());
        assert_ne!(buffer.identity(), comparing.identity());

        // A sampler id carries this device generation like any other resource.
        assert_eq!(linear.verify(table.stamp), Ok(()));
        assert_eq!(linear.kind_name(), "sampler");

        table.destroy_sampler(linear).expect("sampler released");
        assert_eq!(table.sampler_count(), 1);
        assert_eq!(
            table.sampler_handle(linear),
            None,
            "a destroyed id resolves to nothing"
        );
        assert_eq!(table.destroy_sampler(linear), Err(ResourceError::Unknown));

        // The second sampler and the buffer are released by the table's own drop,
        // which must leave nothing behind.
        drop(table);
    }

    #[test]
    fn an_inconsistent_sampler_is_refused_before_the_driver_is_reached() {
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let mut table = ResourceTable::new(
            opened.device.device(),
            opened.device.stamp(),
            allocator,
        );

        let inverted = SamplerDescriptor {
            lod_min_clamp: 4.0,
            lod_max_clamp: 2.0,
            ..SamplerDescriptor::nearest_clamp()
        };
        assert_eq!(
            table.create_sampler(&inverted),
            Err(ResourceError::InvalidSampler)
        );
        assert_eq!(table.sampler_count(), 0);
    }
}
