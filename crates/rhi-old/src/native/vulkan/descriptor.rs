//! Step 5's descriptor half: the descriptor set layout.
//!
//! # A descriptor set layout is a value plus a handle
//!
//! The layout a recipe declares is [`BindGroupLayout`]: binding numbers, shader
//! visibility and a binding kind. `Vulkan` copies that description at creation, so
//! [`SetLayout`] owns the `VkDescriptorSetLayout` and the device that must destroy
//! it.
//!
//! It keeps the description as well, and that is a deliberate change from the first
//! draft: a bind group is created against a set layout and validated against the
//! entries the same description declares, so the description is the layout's own
//! fact rather than a second copy a caller has to keep in step. A set layout is a
//! long-lived object while the description is a handful of bytes, and the
//! alternative -- asking a caller that moved its [`SetLayout`] into a
//! [`PipelineLayout`](super::pipeline::PipelineLayout) for a description it no
//! longer holds -- cannot be answered at all.
//!
//! # What is lowered, and what is deliberately not
//!
//! Only two things reach the driver from an entry: the descriptor type and the
//! stage flags. Everything else the entry carries -- a view dimension, a sample
//! type, a storage format, a minimum binding size -- is a fact a *bind group* or a
//! *pipeline* is validated against later. Lowering them here would be a capability
//! claim this call cannot back: `VkDescriptorSetLayoutCreateInfo` has nowhere to
//! put them, and inventing a field would be a second truth about the same layout.
//!
//! The descriptor type is the one place the lowering is not a plain name change:
//! a buffer binding with a dynamic offset gets `*_DYNAMIC`, because the offset is
//! supplied at bind time and `Vulkan` spells the two as different descriptor types.
//! Choosing the non-dynamic type there would produce a layout the driver accepts and
//! a bind that then fails.
//!
//! # Binding arrays are not in the vocabulary, so the count is one
//!
//! Every retained binding is a single descriptor, and [`crate::common::binding`]
//! deliberately has no count. The count is therefore written as one here rather
//! than taken from the description. `descriptor_count` is *not* set through
//! `immutable_samplers`: `ash`'s builder derives the count from the slice length,
//! so passing an empty slice would silently lower every binding to zero descriptors.
//!
//! # The refusals happen before the driver
//!
//! A duplicate binding number and an entry no stage can see are refused by
//! [`BindGroupLayout::validate`](crate::common::binding::BindGroupLayout::validate),
//! which is a pure decision, so no invalid description is ever handed to
//! `vkCreateDescriptorSetLayout`. The remaining failure is the driver refusing a
//! well-formed description, and that is reported as its own value.

use ash::vk;

use crate::common::binding::{
    BindGroupLayout, BindGroupLayoutError, BindingKind, BufferBindingType, ShaderVisibility,
};

/// Why a descriptor set layout could not be described or created.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DescriptorError {
    /// The description contradicts itself and was refused before the driver saw it.
    Layout(BindGroupLayoutError),
    /// The driver refused a well-formed description.
    Create(vk::Result),
}

/// A `VkDescriptorSetLayout` this backend owns and destroys exactly once.
pub(crate) struct SetLayout {
    device: ash::Device,
    handle: vk::DescriptorSetLayout,
    /// The description the handle was created from.
    ///
    /// Kept because a bind group is created against this layout and validated
    /// against these entries -- which binding numbers exist, which kind each holds,
    /// and the descriptor types the pool must size for. The driver does not need it
    /// again; the bind-group creation does.
    layout: BindGroupLayout,
}

impl SetLayout {
    /// Returns the driver handle a pipeline layout is created against.
    pub(crate) const fn handle(&self) -> vk::DescriptorSetLayout {
        self.handle
    }

    /// Returns the description this layout was created from.
    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }
}

impl Drop for SetLayout {
    fn drop(&mut self) {
        // SAFETY: the handle was created by this device and is destroyed once here.
        // The device outlives this layout because the caller that created it holds
        // both, and every pipeline layout created against this set is destroyed
        // first -- `PipelineLayout` owns these, so field order enforces that.
        unsafe { self.device.destroy_descriptor_set_layout(self.handle, None) };
    }
}

impl core::fmt::Debug for SetLayout {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SetLayout")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

/// The descriptor type one binding kind lowers to.
///
/// Exhaustive over this crate's own closed enums, with no wildcard: a kind added
/// later must be taught to this match rather than silently lowered to the wrong
/// descriptor.
pub(crate) fn descriptor_type(kind: &BindingKind) -> vk::DescriptorType {
    match kind {
        BindingKind::Buffer {
            ty,
            has_dynamic_offset,
            ..
        } => match (ty, has_dynamic_offset) {
            (BufferBindingType::Uniform, false) => vk::DescriptorType::UNIFORM_BUFFER,
            (BufferBindingType::Uniform, true) => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            (BufferBindingType::Storage { .. }, false) => vk::DescriptorType::STORAGE_BUFFER,
            (BufferBindingType::Storage { .. }, true) => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
        },
        // A sampled image and a storage image are different descriptor types even
        // when the same texture view could serve either: the shader's access is
        // part of the layout, and the driver validates against it.
        BindingKind::Texture { .. } => vk::DescriptorType::SAMPLED_IMAGE,
        BindingKind::Sampler(_) => vk::DescriptorType::SAMPLER,
        BindingKind::StorageTexture { .. } => vk::DescriptorType::STORAGE_IMAGE,
    }
}

/// The stage bits one visibility names.
///
/// Folds the bits the value holds rather than mapping a closed set of combinations,
/// because visibility composes: a new pair is a caller's union, not a new variant.
/// The representation is private, so every bit that can be set has a stage here.
pub(crate) fn stage_flags(visibility: ShaderVisibility) -> vk::ShaderStageFlags {
    let mut flags = vk::ShaderStageFlags::empty();
    if visibility.contains(ShaderVisibility::VERTEX) {
        flags |= vk::ShaderStageFlags::VERTEX;
    }
    if visibility.contains(ShaderVisibility::FRAGMENT) {
        flags |= vk::ShaderStageFlags::FRAGMENT;
    }
    if visibility.contains(ShaderVisibility::COMPUTE) {
        flags |= vk::ShaderStageFlags::COMPUTE;
    }
    flags
}

/// The driver bindings one description lowers to.
///
/// `descriptor_count` is one because binding arrays are not in the closed
/// vocabulary; see the module docs for why `immutable_samplers` must not be used to
/// say so.
pub(crate) fn bindings(layout: &BindGroupLayout) -> Vec<vk::DescriptorSetLayoutBinding<'static>> {
    layout
        .entries
        .iter()
        .map(|entry| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(entry.binding)
                .descriptor_type(descriptor_type(&entry.kind))
                .descriptor_count(1)
                .stage_flags(stage_flags(entry.visibility))
        })
        .collect()
}

/// Creates the descriptor set layout `layout` describes, or reports why not.
///
/// The description is validated first, so a contradictory one never reaches the
/// driver. The returned value owns the handle and destroys it on drop.
pub(crate) fn create_set_layout(
    device: &ash::Device,
    layout: &BindGroupLayout,
) -> Result<SetLayout, DescriptorError> {
    layout.validate().map_err(DescriptorError::Layout)?;
    let bindings = bindings(layout);
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    // SAFETY: the device is live; the create-info borrows `bindings`, which outlives
    // this call, and every binding it names is a value this layer described.
    let handle = unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(DescriptorError::Create)?;
    Ok(SetLayout {
        device: device.clone(),
        handle,
        layout: layout.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Validation;
    use crate::common::binding::{
        BindGroupLayoutEntry, SamplerBindingType, StorageTextureAccess, TextureSampleType,
        ViewDimension,
    };
    use crate::native::vulkan::open;
    use fluxel_rendergraph::TextureFormat;

    fn entry(binding: u32, visibility: ShaderVisibility, kind: BindingKind) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility,
            kind,
        }
    }

    /// The exact layout the linear-clamp raster artifact declares.
    fn textured_frame() -> BindGroupLayout {
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

    #[test]
    fn each_binding_kind_lowers_to_the_descriptor_type_that_serves_it() {
        let uniform = BindingKind::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(80),
        };
        let storage = BindingKind::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let sampled = BindingKind::Texture {
            sample_type: TextureSampleType::Float { filterable: false },
            view_dimension: ViewDimension::D2,
            multisampled: false,
        };
        let sampler = BindingKind::Sampler(SamplerBindingType::Comparison);
        let storage_image = BindingKind::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format: TextureFormat::Rgba8Unorm,
            view_dimension: ViewDimension::D2,
        };
        assert_eq!(
            descriptor_type(&uniform),
            vk::DescriptorType::UNIFORM_BUFFER
        );
        assert_eq!(
            descriptor_type(&storage),
            vk::DescriptorType::STORAGE_BUFFER
        );
        assert_eq!(descriptor_type(&sampled), vk::DescriptorType::SAMPLED_IMAGE);
        assert_eq!(descriptor_type(&sampler), vk::DescriptorType::SAMPLER);
        assert_eq!(
            descriptor_type(&storage_image),
            vk::DescriptorType::STORAGE_IMAGE
        );
        // Five kinds, five distinct types: a copy-paste in the match shows up as
        // two kinds sharing one descriptor type.
        let types = [
            descriptor_type(&uniform),
            descriptor_type(&storage),
            descriptor_type(&sampled),
            descriptor_type(&sampler),
            descriptor_type(&storage_image),
        ];
        for (index, ty) in types.iter().enumerate() {
            for (other_index, other) in types.iter().enumerate() {
                if index != other_index {
                    assert_ne!(ty, other);
                }
            }
        }
    }

    #[test]
    fn a_dynamic_offset_selects_the_dynamic_descriptor_type() {
        // `Vulkan` spells the dynamic case as a different descriptor type, so a
        // lowering that ignored the flag would produce a layout a bind could not
        // satisfy.
        for (ty, fixed, dynamic) in [
            (
                BufferBindingType::Uniform,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            ),
            (
                BufferBindingType::Storage { read_only: true },
                vk::DescriptorType::STORAGE_BUFFER,
                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
            ),
        ] {
            assert_eq!(
                descriptor_type(&BindingKind::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
                fixed
            );
            assert_eq!(
                descriptor_type(&BindingKind::Buffer {
                    ty,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                }),
                dynamic
            );
        }
    }

    #[test]
    fn a_storage_binding_reads_the_same_descriptor_type_either_direction() {
        // `read_only` gates the shader's declaration, not the descriptor type: both
        // directions are a storage buffer.
        for read_only in [false, true] {
            assert_eq!(
                descriptor_type(&BindingKind::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
                vk::DescriptorType::STORAGE_BUFFER
            );
        }
    }

    #[test]
    fn visibility_lowers_to_exactly_the_stages_it_names() {
        let both = stage_flags(ShaderVisibility::VERTEX_FRAGMENT);
        assert!(both.contains(vk::ShaderStageFlags::VERTEX));
        assert!(both.contains(vk::ShaderStageFlags::FRAGMENT));
        assert!(!both.intersects(vk::ShaderStageFlags::COMPUTE));

        let fragment = stage_flags(ShaderVisibility::FRAGMENT);
        assert!(!fragment.intersects(vk::ShaderStageFlags::VERTEX));
        assert_eq!(fragment, vk::ShaderStageFlags::FRAGMENT);

        let compute = stage_flags(ShaderVisibility::COMPUTE);
        assert!(!compute.intersects(vk::ShaderStageFlags::VERTEX));
        assert!(!compute.intersects(vk::ShaderStageFlags::FRAGMENT));
        assert_eq!(compute, vk::ShaderStageFlags::COMPUTE);
    }

    #[test]
    fn a_binding_lowers_to_one_descriptor_and_no_immutable_sampler() {
        let lowered = bindings(&textured_frame());
        assert_eq!(lowered.len(), 3);
        for (index, binding) in lowered.iter().enumerate() {
            assert_eq!(binding.binding, index as u32);
            // Count one, not the zero `immutable_samplers(&[])` would write.
            assert_eq!(binding.descriptor_count, 1);
            assert!(
                binding.p_immutable_samplers.is_null(),
                "the retained recipes bind samplers separately"
            );
        }
        assert_eq!(lowered[1].descriptor_type, vk::DescriptorType::SAMPLED_IMAGE);
        assert_eq!(lowered[2].descriptor_type, vk::DescriptorType::SAMPLER);
        assert_eq!(
            lowered[0].stage_flags,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
        );
    }

    #[test]
    fn a_real_descriptor_set_layout_is_created_and_destroyed_on_this_machine() {
        // Step 5's descriptor half against the real driver: the description is
        // validated here, the handle is the driver's, and dropping the owner
        // destroys it. Skips where no adapter exists.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let layout =
            create_set_layout(opened.device.device(), &textured_frame()).expect("a valid layout");
        assert_ne!(layout.handle(), vk::DescriptorSetLayout::null());
        // The drop is the destruction; nothing else releases the handle.
        drop(layout);
    }

    #[test]
    fn an_empty_layout_is_created_on_this_machine() {
        // A shader that declares no bindings asks for a layout with no entries,
        // which `Vulkan` permits; a lowering that refused it would break step 5's
        // compute path.
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let empty = BindGroupLayout {
            entries: Vec::new(),
        };
        let layout = create_set_layout(opened.device.device(), &empty).expect("an empty layout");
        assert_ne!(layout.handle(), vk::DescriptorSetLayout::null());
    }

    #[test]
    fn a_duplicate_binding_is_refused_before_the_driver_is_reached() {
        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let contradictory = BindGroupLayout {
            entries: vec![
                entry(
                    0,
                    ShaderVisibility::FRAGMENT,
                    BindingKind::Sampler(SamplerBindingType::Filtering),
                ),
                entry(
                    0,
                    ShaderVisibility::FRAGMENT,
                    BindingKind::Sampler(SamplerBindingType::Filtering),
                ),
            ],
        };
        assert_eq!(
            create_set_layout(opened.device.device(), &contradictory).err(),
            Some(DescriptorError::Layout(
                BindGroupLayoutError::DuplicateBinding { binding: 0 }
            ))
        );
    }
}
