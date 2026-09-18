//! Step 12's binding selection: the descriptor set a raster draw reads through.
//!
//! # A bind group is a set layout, an allocation and the writes that fill it
//!
//! `Vulkan` has no "bind group" object: a binding is a `VkDescriptorSet` allocated
//! from a pool against a `VkDescriptorSetLayout`, filled by
//! `vkUpdateDescriptorSets`, and selected by `vkCmdBindDescriptorSets` together with
//! the `VkPipelineLayout` it is compatible with. [`BindGroup`] owns the pool, the set
//! and the pipeline-layout handle, so this backend's one descriptor object is the
//! shape the raster recorder's `set_bindings` verb needs.
//!
//! The set layout a group is created against is the pipeline layout's own
//! [`SetLayout`](super::descriptor::SetLayout), never a second one created from an
//! equal description: [`PipelineLayout::set_layout`] hands back the exact handle the
//! pipeline was built with, so "the set was allocated from a compatible layout" is a
//! fact of construction rather than a compatibility check this layer cannot make.
//!
//! # The pool is owned by the group, and that is a bounded decision
//!
//! `Vulkan` frees a descriptor set when its pool is destroyed, so the smallest owner
//! that makes the set's lifetime the group's is one pool per group. A device-owned
//! pool with a free list is the alternative, and it is a *pooling* policy this
//! backend deliberately does not have yet: the transient-reuse rows are the
//! rejecting value (step 11's lowering), and nothing here reuses a descriptor set
//! across recordings. When a consumer needs to recycle sets, that policy arrives with
//! the device that owns the pool, and this type becomes a handle into it.
//!
//! # Every refusal is a value, and most are pure
//!
//! An entry that names a binding the layout does not declare, a layout binding with
//! no entry, a duplicate entry, a resource of the wrong kind, a buffer range that
//! contradicts the layout's minimum size or the buffer's own size, and a layout that
//! asks for a bind-time dynamic offset are all refused before the pool exists. The
//! last one is deliberate: this layer's bind-group vocabulary has no place to state
//! the offset a *bind* supplies, so a dynamic binding cannot be honoured, and
//! recording one anyway would hand the driver a descriptor it reads at the wrong
//! offset. It arrives with the first recipe that needs bind-time offsets.
//!
//! The range checks are the copy step's rule applied here: the layout's own minimum
//! is checked first because it is a fact of the description, and the buffer's size is
//! checked with the end computed in checked arithmetic so a `u64::MAX` offset is a
//! refusal rather than a wrapped range that passes.
//!
//! # What is lowered, and what is deliberately not
//!
//! The descriptor *type* comes from [`descriptor::descriptor_type`], so layout
//! creation and bind-group creation cannot disagree about a dynamic spelling. The
//! image layout comes from [`image_layout`], whose one format-dependent answer asks
//! the mapped `Vulkan` format through [`format::is_depth`], the single source of
//! truth step 4 established. The sample type, view dimension and minimum binding size
//! are deliberately not written into the descriptor: `VkDescriptorImageInfo` and
//! `VkDescriptorBufferInfo` have nowhere to put them, and the layout is where
//! `Vulkan` validates them.

use ash::vk;
use fluxel_rendergraph::TextureFormat;

use crate::common::base::resource::{BufferId, SamplerId, TextureId};
use crate::common::binding::{
    BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingKind, BindingResource,
};

use super::descriptor::{self, SetLayout};
use super::format;
use super::pipeline::PipelineLayout;
use super::resource::ResourceTable;

/// Why a bind group could not be created.
///
/// The variants stay separate because they name different fixes: a binding number no
/// layout declares, a layout binding no entry fills, a resource of the wrong kind, a
/// buffer range that contradicts a fact, a dynamic binding this vocabulary cannot
/// honour, and the driver's own refusal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BindGroupError {
    /// The pipeline layout has no set layout at this index.
    NoSuchSet {
        /// The requested set index.
        index: u32,
    },
    /// An entry names a binding number the layout does not declare.
    UnknownBinding {
        /// The undeclared binding number.
        binding: u32,
    },
    /// A binding the layout declares has no entry.
    MissingBinding {
        /// The binding number with no entry.
        binding: u32,
    },
    /// Two entries name the same binding number.
    DuplicateBinding {
        /// The repeated binding number.
        binding: u32,
    },
    /// The entry's resource is not the kind the layout declares at that binding.
    KindMismatch {
        /// The binding whose kinds disagree.
        binding: u32,
    },
    /// The layout's minimum binding size is zero, which is not a meaningful minimum.
    ///
    /// [`crate::common::binding::BindingKind`] documents this as the bind-group
    /// check's refusal rather than the layout's, because no driver sees the value at
    /// layout creation.
    ZeroMinimum {
        /// The binding with the meaningless minimum.
        binding: u32,
    },
    /// The layout declares a bind-time dynamic offset, which this vocabulary cannot
    /// supply.
    DynamicOffset {
        /// The dynamic binding number.
        binding: u32,
    },
    /// A buffer entry's range is zero bytes.
    ZeroRange {
        /// The binding with the empty range.
        binding: u32,
    },
    /// A buffer entry's range is smaller than the layout's minimum binding size.
    RangeTooSmall {
        /// The binding whose range is short.
        binding: u32,
        /// The minimum size the layout declares.
        required: u64,
    },
    /// A buffer entry's range does not fit inside the buffer it names.
    RangeOutOfBounds {
        /// The binding whose range does not fit.
        binding: u32,
    },
    /// The buffer id names no live buffer of this device generation.
    UnknownBuffer(BufferId),
    /// The texture id names no live texture of this device generation.
    UnknownTexture(TextureId),
    /// The sampler id names no live sampler of this device generation.
    UnknownSampler(SamplerId),
    /// The texture's portable format has no `Vulkan` equivalent, so its view cannot be
    /// described to the driver through this layer.
    UnsupportedFormat(TextureFormat),
    /// The driver refused to create the descriptor pool.
    Pool(vk::Result),
    /// The driver refused to allocate the descriptor set.
    Allocate(vk::Result),
    /// The driver reported a successful allocation but returned no descriptor set.
    ///
    /// `ash` grows the output vector only on success, so this cannot happen with a
    /// conformant loader; it is a value rather than a panic so the impossible shape
    /// still destroys the pool it just created.
    NoDescriptorSet,
}

/// One descriptor write to issue, with the index of its payload in the info vector.
///
/// The payloads are collected into two vectors first so every `WriteDescriptorSet`
/// can borrow a slice of exactly the length it names; building them in one pass would
/// require borrowing a vector while it is still growing.
enum Planned {
    /// A buffer binding's write.
    Buffer {
        /// The binding number.
        binding: u32,
        /// The descriptor type the layout lowered to.
        ty: vk::DescriptorType,
        /// The index of the buffer info this write names.
        index: usize,
    },
    /// An image or sampler binding's write.
    Image {
        /// The binding number.
        binding: u32,
        /// The descriptor type the layout lowered to.
        ty: vk::DescriptorType,
        /// The index of the image info this write names.
        index: usize,
    },
}

/// A `VkDescriptorSet` and the pool that owns its memory.
///
/// The pipeline layout the set was allocated against is stored as a handle, and the
/// caller keeps its owner alive for as long as the recording can name this group --
/// the same obligation the encoder's vertex buffers state. The set is freed by
/// destroying the pool in `Drop`, which is the only release `Vulkan` needs here
/// because the pool was created without `FREE_DESCRIPTOR_SET`.
pub(crate) struct BindGroup {
    device: ash::Device,
    pipeline_layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    set_index: u32,
}

impl BindGroup {
    /// Returns the descriptor set the recorder binds.
    pub(crate) const fn set(&self) -> vk::DescriptorSet {
        self.set
    }

    /// Returns the pipeline-layout handle the set is compatible with.
    pub(crate) const fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    /// Returns the set index this group fills in its pipeline layout.
    pub(crate) const fn set_index(&self) -> u32 {
        self.set_index
    }
}

impl Drop for BindGroup {
    fn drop(&mut self) {
        // Destroying the pool implicitly frees the descriptor set allocated from it,
        // which is the whole reason the group owns the pool. The pipeline layout is a
        // handle, not an owner: its owner outlives this group by the caller's field
        // order, the same invariant the encoder's command pool states.
        // SAFETY: this is the only owner of the pool, no allocation callbacks were
        // supplied at creation, and the device outlives it.
        unsafe { self.device.destroy_descriptor_pool(self.pool, None) };
    }
}

impl core::fmt::Debug for BindGroup {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("BindGroup")
            .field("set_index", &self.set_index)
            .field("set", &self.set)
            .finish_non_exhaustive()
    }
}

/// The pool sizes one layout's descriptors need.
///
/// One entry per *distinct* descriptor type, counting the bindings that lower to it:
/// `VkDescriptorPoolCreateInfo` rejects two entries for one type, and a second entry
/// would silently make the total a lie. The fold is over
/// [`descriptor::descriptor_type`], so a dynamic buffer binding and a fixed one are
/// the two different types their layout creation already spelled.
pub(crate) fn pool_sizes(layout: &BindGroupLayout) -> Vec<vk::DescriptorPoolSize> {
    let mut sizes: Vec<vk::DescriptorPoolSize> = Vec::new();
    for entry in &layout.entries {
        let ty = descriptor::descriptor_type(&entry.kind);
        match sizes.iter_mut().find(|size| size.ty == ty) {
            Some(size) => size.descriptor_count += 1,
            None => sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(ty)
                    .descriptor_count(1),
            ),
        }
    }
    sizes
}

/// The image layout one texture binding's descriptor must name.
///
/// A sampled read's layout is the one case whose answer depends on the format: a
/// depth texture cannot be read through the colour read-only layout, so `is_depth`
/// -- asked of the *mapped* `Vulkan` format -- selects the depth-stencil read-only
/// layout instead. A storage texture has no optimal layout, so both directions share
/// `GENERAL`, which is the layout [`super::barrier::image_state`] gives the storage
/// states.
///
/// `None` means the binding is not a texture binding, which the caller refused
/// before reaching here.
pub(crate) fn image_layout(kind: &BindingKind, is_depth: bool) -> Option<vk::ImageLayout> {
    match kind {
        BindingKind::Texture { .. } => Some(if is_depth {
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        }),
        BindingKind::StorageTexture { .. } => Some(vk::ImageLayout::GENERAL),
        BindingKind::Buffer { .. } | BindingKind::Sampler(_) => None,
    }
}

/// The buffer descriptor one binding writes, with its range checked at the boundary.
///
/// The layout's minimum is checked first because it is a fact of the description; the
/// buffer's own size is checked with the end computed in checked arithmetic, so an
/// offset that would wrap is an overrun rather than a range that passes.
pub(crate) fn buffer_info(
    binding: u32,
    min_binding_size: Option<u64>,
    buffer: vk::Buffer,
    buffer_size: u64,
    offset: u64,
    size: u64,
) -> Result<vk::DescriptorBufferInfo, BindGroupError> {
    if size == 0 {
        return Err(BindGroupError::ZeroRange { binding });
    }
    if let Some(required) = min_binding_size {
        if size < required {
            return Err(BindGroupError::RangeTooSmall { binding, required });
        }
    }
    let fits = offset
        .checked_add(size)
        .is_some_and(|end| end <= buffer_size);
    if !fits {
        return Err(BindGroupError::RangeOutOfBounds { binding });
    }
    Ok(vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(offset)
        .range(size))
}

/// The kind agreement one declared binding and its entry must satisfy.
///
/// A texture binding accepts a texture entry whatever the entry's own view dimension
/// is: the layout's `BindingKind::Texture` versus `StorageTexture` decides the
/// descriptor type and the image layout, so a second distinction here could disagree
/// with it. The dynamic-offset refusal comes after the kind check, because "the wrong
/// resource kind" is the more specific mistake when both are wrong.
fn entry_kind(declared: &BindGroupLayoutEntry, entry: &BindGroupEntry) -> Result<(), BindGroupError> {
    let agrees = matches!(
        (&declared.kind, &entry.resource),
        (BindingKind::Buffer { .. }, BindingResource::Buffer { .. })
            | (
                BindingKind::Texture { .. } | BindingKind::StorageTexture { .. },
                BindingResource::Texture(_)
            )
            | (BindingKind::Sampler(_), BindingResource::Sampler(_))
    );
    if !agrees {
        return Err(BindGroupError::KindMismatch {
            binding: entry.binding,
        });
    }
    if let BindingKind::Buffer {
        has_dynamic_offset: true,
        ..
    } = declared.kind
    {
        return Err(BindGroupError::DynamicOffset {
            binding: entry.binding,
        });
    }
    Ok(())
}

/// Refuses every description that cannot fill one descriptor set.
///
/// The check is total over the entries and the layout: every declared binding is
/// filled exactly once, no entry names an undeclared binding, each entry's resource
/// kind matches, and no layout minimum is zero. It is pure, so it runs before the
/// pool exists and every refusal is a sentence a caller can act on rather than a
/// driver validation error naming a handle.
fn validate(layout: &BindGroupLayout, entries: &[BindGroupEntry]) -> Result<(), BindGroupError> {
    for declared in &layout.entries {
        if let BindingKind::Buffer {
            min_binding_size: Some(0),
            ..
        } = declared.kind
        {
            return Err(BindGroupError::ZeroMinimum {
                binding: declared.binding,
            });
        }
        let mut seen = false;
        for entry in entries {
            if entry.binding != declared.binding {
                continue;
            }
            if seen {
                return Err(BindGroupError::DuplicateBinding {
                    binding: declared.binding,
                });
            }
            seen = true;
            entry_kind(declared, entry)?;
        }
        if !seen {
            return Err(BindGroupError::MissingBinding {
                binding: declared.binding,
            });
        }
    }
    for entry in entries {
        if layout.entry(entry.binding).is_none() {
            return Err(BindGroupError::UnknownBinding {
                binding: entry.binding,
            });
        }
    }
    Ok(())
}

/// Creates the descriptor set `entries` describe against `set_index` of
/// `pipeline_layout`.
///
/// The description is validated and every resource resolved before the pool exists,
/// so a refused bind group reaches no driver entry point at all. The pool holds room
/// for exactly the descriptors this set writes, and the pipeline layout must be kept
/// alive by the caller for as long as the group is used.
pub(crate) fn create(
    device: &ash::Device,
    table: &ResourceTable,
    pipeline_layout: &PipelineLayout,
    set_index: u32,
    entries: &[BindGroupEntry],
) -> Result<BindGroup, BindGroupError> {
    let set_layout: &SetLayout = pipeline_layout
        .set_layout(set_index)
        .ok_or(BindGroupError::NoSuchSet { index: set_index })?;
    let layout = set_layout.layout();
    validate(layout, entries)?;

    // Resolution and lowering first: no pool exists while a refusal is still
    // possible. The infos are collected before the writes because a write borrows a
    // one-element slice of the vector it names.
    let mut buffers: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(entries.len());
    let mut images: Vec<vk::DescriptorImageInfo> = Vec::with_capacity(entries.len());
    let mut planned: Vec<Planned> = Vec::with_capacity(entries.len());
    for entry in entries {
        let declared = layout
            .entry(entry.binding)
            .expect("validation refused an undeclared binding");
        let ty = descriptor::descriptor_type(&declared.kind);
        match entry.resource {
            BindingResource::Buffer {
                buffer,
                offset,
                size,
            } => {
                let handle = table
                    .buffer_handle(buffer)
                    .ok_or(BindGroupError::UnknownBuffer(buffer))?;
                let buffer_size = table
                    .buffer_size(buffer)
                    .ok_or(BindGroupError::UnknownBuffer(buffer))?;
                let min_binding_size = match declared.kind {
                    BindingKind::Buffer {
                        min_binding_size, ..
                    } => min_binding_size,
                    _ => None,
                };
                let info = buffer_info(
                    entry.binding,
                    min_binding_size,
                    handle,
                    buffer_size,
                    offset,
                    size,
                )?;
                planned.push(Planned::Buffer {
                    binding: entry.binding,
                    ty,
                    index: buffers.len(),
                });
                buffers.push(info);
            }
            BindingResource::Texture(texture) => {
                let view = table
                    .texture_view(texture)
                    .ok_or(BindGroupError::UnknownTexture(texture))?;
                let desc = table
                    .texture_desc(texture)
                    .ok_or(BindGroupError::UnknownTexture(texture))?;
                let mapped = format::image_format(desc.format)
                    .ok_or(BindGroupError::UnsupportedFormat(desc.format))?;
                let required = image_layout(&declared.kind, format::is_depth(mapped))
                    .expect("validation matched a texture binding");
                let info = vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(required);
                planned.push(Planned::Image {
                    binding: entry.binding,
                    ty,
                    index: images.len(),
                });
                images.push(info);
            }
            BindingResource::Sampler(sampler) => {
                let handle = table
                    .sampler_handle(sampler)
                    .ok_or(BindGroupError::UnknownSampler(sampler))?;
                // A sampler descriptor names the sampler and nothing else; the null
                // view and the undefined layout are that fact, not an unset value.
                let info = vk::DescriptorImageInfo::default().sampler(handle);
                planned.push(Planned::Image {
                    binding: entry.binding,
                    ty,
                    index: images.len(),
                });
                images.push(info);
            }
        }
    }

    let sizes = pool_sizes(layout);
    // One set, no flags: the group owns the pool and destroys it whole, so the
    // individual-free flag would be a permission nothing uses.
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&sizes);
    // SAFETY: the device is live; `pool_info` borrows `sizes`, a local that outlives
    // the call, and no allocation callbacks are supplied.
    let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
        .map_err(BindGroupError::Pool)?;

    let set_layouts = [set_layout.handle()];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&set_layouts);
    // SAFETY: the pool was just created by this device and is live; the set layout is
    // a live handle this device created and the caller keeps alive; `allocate_info`
    // borrows locals that outlive the call.
    let set = match unsafe { device.allocate_descriptor_sets(&allocate_info) } {
        // Exactly one set was asked for, so anything else is a loader that broke its
        // own contract: the pool is destroyed before the refusal, leaving nothing.
        Ok(sets) => match sets.as_slice() {
            [only] => *only,
            _ => {
                // SAFETY: the pool was created just above and is destroyed once.
                unsafe { device.destroy_descriptor_pool(pool, None) };
                return Err(BindGroupError::NoDescriptorSet);
            }
        },
        Err(error) => {
            // SAFETY: the pool was created just above and is destroyed once.
            unsafe { device.destroy_descriptor_pool(pool, None) };
            return Err(BindGroupError::Allocate(error));
        }
    };

    // Each write borrows the one-element slice of the info vector it names, and both
    // vectors outlive the update call. `buffer_info`/`image_info` derive the
    // descriptor count from the slice length, so a write cannot name a count its
    // pointer does not cover.
    let writes: Vec<vk::WriteDescriptorSet<'_>> = planned
        .iter()
        .map(|planned| match planned {
            Planned::Buffer {
                binding,
                ty,
                index,
            } => vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(*binding)
                .descriptor_type(*ty)
                .buffer_info(&buffers[*index..=*index]),
            Planned::Image {
                binding,
                ty,
                index,
            } => vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(*binding)
                .descriptor_type(*ty)
                .image_info(&images[*index..=*index]),
        })
        .collect();
    // SAFETY: the set was allocated from the live pool above, every write names that
    // set and a binding the layout declares, and each write's info slice is a live
    // local for the duration of the call. No copies are performed.
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    Ok(BindGroup {
        device: device.clone(),
        pipeline_layout: pipeline_layout.handle(),
        pool,
        set,
        set_index,
    })
}

#[cfg(test)]
mod tests {
    use core::time::Duration;

    use fluxel_rendergraph::{
        BufferUsage, BufferUsageKind, CompletionStatus, Extent3d, ResourceAccessState, ScissorRect,
        TextureDesc, TextureDimension, TextureFormat, TextureRange, TextureUsage, TextureUsageKind,
        Viewport,
    };

    use super::*;
    use crate::Validation;
    use crate::common::base::stamp::DeviceStamp;
    use crate::common::binding::{
        BufferBindingType, SamplerBindingType, ShaderVisibility, StorageTextureAccess,
        TextureSampleType, ViewDimension,
    };
    use crate::common::sampler::SamplerDescriptor;
    use crate::native::vulkan::allocator::GpuAllocator;
    use crate::native::vulkan::command::{CommandPool, RecordError};
    use crate::native::vulkan::open::OpenedVulkan;
    use crate::native::vulkan::pipeline::{create_layout, create_raster};
    use crate::native::vulkan::test_support::{
        colour_only_state, colour_target_pass, position_stream, raster_shaders,
    };
    use crate::native::vulkan::{memory, open, submission};
    use fluxel_rendergraph::{DeviceIdentity, PhysicalResourceIdentity};

    /// The exact bind-group layout the linear-clamp raster artifact declares.
    fn textured_frame() -> BindGroupLayout {
        let entry = |binding, visibility, kind| BindGroupLayoutEntry {
            binding,
            visibility,
            kind,
        };
        BindGroupLayout {
            entries: vec![
                entry(
                    0,
                    ShaderVisibility::VERTEX_FRAGMENT,
                    BindingKind::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(80),
                    },
                ),
                entry(
                    1,
                    ShaderVisibility::FRAGMENT,
                    BindingKind::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: ViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(
                    2,
                    ShaderVisibility::FRAGMENT,
                    BindingKind::Sampler(SamplerBindingType::Filtering),
                ),
            ],
        }
    }

    fn stamp() -> DeviceStamp {
        DeviceStamp::initial(DeviceIdentity::new(1))
    }

    fn identity(value: u64) -> PhysicalResourceIdentity {
        PhysicalResourceIdentity::new(value)
    }

    /// A buffer entry naming an id, for the pure validation tests.
    fn buffer_entry(binding: u32, offset: u64, size: u64) -> BindGroupEntry {
        BindGroupEntry {
            binding,
            resource: BindingResource::Buffer {
                buffer: BufferId::new(stamp(), identity(u64::from(binding) + 1)),
                offset,
                size,
            },
        }
    }

    fn texture_entry(binding: u32) -> BindGroupEntry {
        BindGroupEntry {
            binding,
            resource: BindingResource::Texture(TextureId::new(stamp(), identity(9))),
        }
    }

    fn sampler_entry(binding: u32) -> BindGroupEntry {
        BindGroupEntry {
            binding,
            resource: BindingResource::Sampler(SamplerId::new(stamp(), identity(10))),
        }
    }

    fn declared_buffer(kinds: &[BufferUsageKind]) -> BufferUsage {
        BufferUsage::from_kinds(kinds.iter().copied())
    }

    fn declared_texture(kinds: &[TextureUsageKind]) -> TextureUsage {
        TextureUsage::from_kinds(kinds.iter().copied())
    }

    #[test]
    fn the_retained_layout_needs_one_descriptor_of_each_type() {
        // Three bindings, three distinct descriptor types: a pool that collapsed two
        // of them into one entry would under-count and fail only at allocation.
        let sizes = pool_sizes(&textured_frame());
        assert_eq!(sizes.len(), 3);
        assert_eq!(sizes[0].ty, vk::DescriptorType::UNIFORM_BUFFER);
        assert_eq!(sizes[1].ty, vk::DescriptorType::SAMPLED_IMAGE);
        assert_eq!(sizes[2].ty, vk::DescriptorType::SAMPLER);
        for size in &sizes {
            assert_eq!(size.descriptor_count, 1);
        }
        assert!(
            pool_sizes(&BindGroupLayout {
                entries: Vec::new()
            })
            .is_empty()
        );
    }

    #[test]
    fn two_bindings_of_one_descriptor_type_share_one_pool_size_entry() {
        // `Vulkan` rejects two sizes for one type, and a second entry would make the
        // total a lie rather than a duplicate.
        let entry = |binding| BindGroupLayoutEntry {
            binding,
            visibility: ShaderVisibility::FRAGMENT,
            kind: BindingKind::Sampler(SamplerBindingType::Filtering),
        };
        let sizes = pool_sizes(&BindGroupLayout {
            entries: vec![entry(0), entry(1)],
        });
        assert_eq!(sizes.len(), 1);
        assert_eq!(sizes[0].ty, vk::DescriptorType::SAMPLER);
        assert_eq!(sizes[0].descriptor_count, 2);
    }

    #[test]
    fn a_sampled_image_layout_follows_the_format_and_a_storage_image_is_general() {
        let sampled = BindingKind::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: ViewDimension::D2,
            multisampled: false,
        };
        assert_eq!(
            image_layout(&sampled, false),
            Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        );
        assert_eq!(
            image_layout(&sampled, true),
            Some(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        );
        let storage = BindingKind::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format: TextureFormat::Rgba8Unorm,
            view_dimension: ViewDimension::D2,
        };
        assert_eq!(
            image_layout(&storage, false),
            Some(vk::ImageLayout::GENERAL)
        );
        // A storage read and a storage write share `GENERAL`, so the format cannot
        // change the answer.
        assert_eq!(image_layout(&storage, true), image_layout(&storage, false));
        // A buffer or a sampler is not a texture binding, so it has no image layout.
        assert_eq!(
            image_layout(
                &BindingKind::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                false,
            ),
            None
        );
        assert_eq!(
            image_layout(&BindingKind::Sampler(SamplerBindingType::Filtering), false),
            None
        );
    }

    #[test]
    fn the_buffer_range_is_checked_at_the_boundary() {
        // The valid shape carries all three fields through unchanged.
        let info = buffer_info(4, Some(16), vk::Buffer::null(), 256, 32, 64).expect("a valid range");
        assert_eq!(info.offset, 32);
        assert_eq!(info.range, 64);

        // A zero range, a range under the layout's minimum, a range past the buffer,
        // and an offset that would wrap each answer with their own sentence.
        //
        // `ash` derives no `PartialEq` for `DescriptorBufferInfo`, so the refusals are
        // compared as errors rather than as whole results -- the same shape the barrier,
        // copy and surface increments recorded.
        assert_eq!(
            buffer_info(4, None, vk::Buffer::null(), 256, 0, 0).err(),
            Some(BindGroupError::ZeroRange { binding: 4 })
        );
        assert_eq!(
            buffer_info(4, Some(80), vk::Buffer::null(), 256, 0, 64).err(),
            Some(BindGroupError::RangeTooSmall {
                binding: 4,
                required: 80
            })
        );
        assert_eq!(
            buffer_info(4, None, vk::Buffer::null(), 256, 128, 256).err(),
            Some(BindGroupError::RangeOutOfBounds { binding: 4 })
        );
        // `u64::MAX - 3` plus eight overflows, and the checked arithmetic is what
        // makes it an overrun rather than a wrapped range that fits.
        assert_eq!(
            buffer_info(4, None, vk::Buffer::null(), u64::MAX, u64::MAX - 3, 8).err(),
            Some(BindGroupError::RangeOutOfBounds { binding: 4 })
        );
    }

    #[test]
    fn validation_refuses_every_description_a_set_cannot_be_filled_from() {
        let layout = textured_frame();
        let valid = [buffer_entry(0, 0, 80), texture_entry(1), sampler_entry(2)];
        assert_eq!(validate(&layout, &valid), Ok(()));

        // A binding the layout does not declare.
        assert_eq!(
            validate(
                &layout,
                &[valid[0], valid[1], valid[2], buffer_entry(7, 0, 80)]
            ),
            Err(BindGroupError::UnknownBinding { binding: 7 })
        );
        // A declared binding with no entry.
        assert_eq!(
            validate(&layout, &[valid[0], valid[1]]),
            Err(BindGroupError::MissingBinding { binding: 2 })
        );
        // Two entries for one binding.
        assert_eq!(
            validate(
                &layout,
                &[valid[0], buffer_entry(0, 0, 80), valid[1], valid[2]]
            ),
            Err(BindGroupError::DuplicateBinding { binding: 0 })
        );
        // A texture placed at a buffer binding.
        let mismatched = BindGroupEntry {
            binding: 0,
            resource: valid[1].resource,
        };
        assert_eq!(
            validate(&layout, &[mismatched, valid[1], valid[2]]),
            Err(BindGroupError::KindMismatch { binding: 0 })
        );
        // A minimum binding size of zero is the documented bind-group refusal.
        let zero_minimum = BindGroupLayout {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderVisibility::VERTEX,
                kind: BindingKind::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(0),
                },
            }],
        };
        assert_eq!(
            validate(&zero_minimum, &[buffer_entry(0, 0, 80)]),
            Err(BindGroupError::ZeroMinimum { binding: 0 })
        );
        // A dynamic binding cannot be honoured by this vocabulary, so it is refused
        // rather than recorded with an offset the driver would apply itself.
        let dynamic = BindGroupLayout {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderVisibility::VERTEX,
                kind: BindingKind::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
            }],
        };
        assert_eq!(
            validate(&dynamic, &[buffer_entry(0, 0, 80)]),
            Err(BindGroupError::DynamicOffset { binding: 0 })
        );
    }

    /// Opens a headless device, its memory types and a real table, or `None` where no
    /// adapter exists. Having no GPU is not what these tests are about.
    fn fixture() -> Option<(OpenedVulkan, Vec<vk::MemoryType>, ResourceTable)> {
        let opened = open::open(Validation::Disabled, 0).ok()?;
        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let table = ResourceTable::new(opened.device.device(), opened.device.stamp(), allocator);
        Some((opened, memory_types, table))
    }

    /// The three resources the retained textured layout binds, plus the entries that
    /// name them.
    fn retained_entries(
        table: &mut ResourceTable,
        memory_types: &[vk::MemoryType],
    ) -> [BindGroupEntry; 3] {
        let buffer = table
            .create_buffer(
                80,
                declared_buffer(&[BufferUsageKind::Uniform]),
                memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local uniform buffer");
        let texture = table
            .create_texture(
                TextureDesc {
                    dimension: TextureDimension::D2,
                    extent: Extent3d {
                        width: 16,
                        height: 8,
                        depth: 1,
                    },
                    mip_levels: 1,
                    array_layers: 1,
                    sample_count: 1,
                    format: TextureFormat::Rgba8Unorm,
                },
                declared_texture(&[TextureUsageKind::Sampled]),
                memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local sampled texture");
        let sampler = table
            .create_sampler(&SamplerDescriptor::linear_clamp())
            .expect("a linear-clamp sampler");
        [
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer,
                    offset: 0,
                    size: 80,
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Texture(texture),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(sampler),
            },
        ]
    }

    #[test]
    fn a_real_bind_group_is_created_against_the_pipeline_layouts_own_set_layout() {
        // Step 12's bind-group half against the real driver: a real
        // `VkDescriptorPool`, a real `VkDescriptorSet` allocated from the pipeline
        // layout's own set layout, and the three real writes that fill it. Skips where
        // no adapter exists.
        let Some((opened, memory_types, mut table)) = fixture() else {
            return;
        };
        let entries = retained_entries(&mut table, &memory_types);
        let set_layout = descriptor::create_set_layout(opened.device.device(), &textured_frame())
            .expect("a valid set layout");
        let layout = create_layout(opened.device.device(), vec![set_layout])
            .expect("a pipeline layout over one set layout");

        let group = create(opened.device.device(), &table, &layout, 0, &entries)
            .expect("a real descriptor set over the retained layout");
        assert_ne!(group.set(), vk::DescriptorSet::null());
        assert_eq!(group.set_index(), 0);
        assert_eq!(group.pipeline_layout(), layout.handle());
        // The group owns the pool, so its drop is the whole release.
        drop(group);
    }

    #[test]
    fn a_bind_group_naming_a_set_the_layout_does_not_have_is_refused_before_the_driver() {
        let Some((opened, memory_types, mut table)) = fixture() else {
            return;
        };
        let entries = retained_entries(&mut table, &memory_types);
        let set_layout = descriptor::create_set_layout(opened.device.device(), &textured_frame())
            .expect("a valid set layout");
        let layout = create_layout(opened.device.device(), vec![set_layout])
            .expect("a pipeline layout over one set layout");
        assert_eq!(
            create(opened.device.device(), &table, &layout, 1, &entries).err(),
            Some(BindGroupError::NoSuchSet { index: 1 })
        );
    }

    #[test]
    fn a_stale_or_wrong_resource_id_is_refused_rather_than_reaching_the_driver() {
        // The entries name ids the table resolves, so a destroyed resource and a
        // resource of the wrong kind both answer before a pool exists.
        let Some((opened, memory_types, mut table)) = fixture() else {
            return;
        };
        let entries = retained_entries(&mut table, &memory_types);
        let set_layout = descriptor::create_set_layout(opened.device.device(), &textured_frame())
            .expect("a valid set layout");
        let layout = create_layout(opened.device.device(), vec![set_layout])
            .expect("a pipeline layout over one set layout");

        // A destroyed id resolves to nothing, so the buffer entry is refused by name.
        let BindingResource::Buffer { buffer, .. } = entries[0].resource else {
            panic!("the first entry is a buffer");
        };
        table.destroy_buffer(buffer).expect("the buffer is released");
        assert_eq!(
            create(opened.device.device(), &table, &layout, 0, &entries).err(),
            Some(BindGroupError::UnknownBuffer(buffer))
        );

        // A sampler placed at a buffer binding is a kind mismatch, refused before any
        // id is looked up.
        let BindingResource::Sampler(sampler) = entries[2].resource else {
            panic!("the third entry is a sampler");
        };
        let mismatched = [
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Sampler(sampler),
            },
            entries[1],
            entries[2],
        ];
        assert_eq!(
            create(opened.device.device(), &table, &layout, 0, &mismatched).err(),
            Some(BindGroupError::KindMismatch { binding: 0 })
        );
    }

    #[test]
    fn a_bound_bind_group_reaches_a_draw_the_driver_completes() {
        // The end of step 12's binding path: a real group created over the pipeline
        // layout of a real raster pipeline is bound inside a real pass, and the draw
        // that reads through it is submitted and reported complete. Skips where no
        // adapter exists.
        let Some((opened, memory_types, mut table)) = fixture() else {
            return;
        };
        let entries = retained_entries(&mut table, &memory_types);
        let (framebuffer, image, mapped) = colour_target_pass(&opened, &mut table, &memory_types);
        let vertices = table
            .create_buffer(
                256,
                declared_buffer(&[BufferUsageKind::Vertex]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local vertex buffer");
        let vertex_handle = table
            .buffer_handle(vertices)
            .expect("a live vertex buffer");

        // The group is created against the layout *before* the layout is moved into
        // the pipeline; the pipeline then owns it for the recording, which is what
        // keeps the stored `VkPipelineLayout` handle alive.
        let set_layout = descriptor::create_set_layout(opened.device.device(), &textured_frame())
            .expect("a valid set layout");
        let layout = create_layout(opened.device.device(), vec![set_layout])
            .expect("a pipeline layout over one set layout");
        let group = create(opened.device.device(), &table, &layout, 0, &entries)
            .expect("a real descriptor set over the retained layout");
        let pipeline = create_raster(
            opened.device.device(),
            layout,
            &raster_shaders(),
            &position_stream(),
            &colour_only_state(),
        )
        .expect("a raster pipeline over the textured layout");

        let pool = CommandPool::new(
            opened.device.device(),
            opened.device.selected_queue().family,
        )
        .expect("a command pool on an opened device");
        let mut encoder = pool.begin().expect("a recording encoder");
        encoder
            .transition_image(
                image,
                mapped,
                TextureRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::ColorAttachmentWrite,
            )
            .expect("undefined to a colour attachment");
        encoder.begin_raster(&framebuffer).expect("the pass opens");
        encoder
            .set_raster_pipeline(&pipeline)
            .expect("the pipeline binds");
        encoder.set_bindings(&group).expect("the bind group binds");
        encoder
            .set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: 16.0,
                height: 8.0,
                min_depth: 0.0,
                max_depth: 1.0,
            })
            .expect("the viewport sets");
        encoder
            .set_scissor(ScissorRect {
                x: 0,
                y: 0,
                width: 16,
                height: 8,
            })
            .expect("the scissor sets");
        encoder
            .set_vertex_buffer(0, vertex_handle, 0)
            .expect("the vertex buffer binds");
        encoder.draw(0..3, 1).expect("a draw records");
        encoder.end_raster().expect("the pass closes");

        let finished = encoder.finish().expect("the recording ends");
        let mut submitted =
            submission::submit(opened.device.device(), opened.device.queue(), finished)
                .expect("the driver accepts one submission");
        assert_eq!(
            submitted.wait(Duration::from_secs(10)),
            Ok(CompletionStatus::Complete),
            "a draw through a bound descriptor set runs to completion"
        );
        assert!(submitted.is_terminal());
    }

    #[test]
    fn set_bindings_refuses_outside_a_pass_without_poisoning_the_recording() {
        // The verb is a raster verb like the draws: it belongs to an open pass, and
        // the same sentence covers "not recording" and "no pass open".
        let Some((opened, memory_types, mut table)) = fixture() else {
            return;
        };
        let entries = retained_entries(&mut table, &memory_types);
        let set_layout = descriptor::create_set_layout(opened.device.device(), &textured_frame())
            .expect("a valid set layout");
        let layout = create_layout(opened.device.device(), vec![set_layout])
            .expect("a pipeline layout over one set layout");
        let group = create(opened.device.device(), &table, &layout, 0, &entries)
            .expect("a real descriptor set");

        let pool = CommandPool::new(
            opened.device.device(),
            opened.device.selected_queue().family,
        )
        .expect("a command pool on an opened device");
        let mut encoder = pool.begin().expect("a recording encoder");
        assert_eq!(encoder.set_bindings(&group), Err(RecordError::NoPass));
        // The refusal did not poison the recording, which still ends.
        encoder.end().expect("the recording ends");
    }

    #[test]
    fn an_empty_layout_needs_no_descriptor_and_binds_nothing() {
        // A shader that declares no bindings asks for an empty set; the pool has no
        // sizes, the allocation succeeds and no write is issued. That is the shape the
        // retained compute artifact's layout already describes.
        let Some((opened, _memory_types, table)) = fixture() else {
            return;
        };
        let set_layout = descriptor::create_set_layout(
            opened.device.device(),
            &BindGroupLayout {
                entries: Vec::new(),
            },
        )
        .expect("an empty set layout");
        let layout = create_layout(opened.device.device(), vec![set_layout])
            .expect("a pipeline layout over an empty set layout");
        let group =
            create(opened.device.device(), &table, &layout, 0, &[]).expect("an empty descriptor set");
        assert_ne!(group.set(), vk::DescriptorSet::null());
    }
}
