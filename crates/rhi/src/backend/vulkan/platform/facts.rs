//! Conservative, device-probed facts for Vulkan.
//!
//! A Vulkan format name is not by itself an RHI texture capability. The image
//! creator uses optimal tiling, an exact image type/usage/flag tuple, and a
//! concrete sample count; all of those reach the native image-format query
//! below. Consequently this module records only finite v13 keys which the
//! physical device accepted. Missing entries mean unsupported, never "the
//! driver will probably accept it".
//!
//! Alternate view-format pairs deliberately remain absent. The current image
//! path can request `MUTABLE_FORMAT`, but has not yet enabled/validated Vulkan's
//! image-format-list contract nor supplied conformance cases for each view
//! creation pair. `CapabilityFacts` therefore rejects non-empty `view_formats`,
//! rather than promising a texture whose later view can fail.

use ash::vk;

use crate::api::binding::vocabulary::BindableKind;
use crate::api::binding::{BindingLimitClass, BindingSupport, BufferBindingAccess};
use crate::api::capability::{BindingSupportKey, CapabilityFacts};
use crate::api::error::RhiResult;
use crate::api::format::{
    TextureSupport, TextureSupportLimits, TextureSupportQuery, format_aspects,
};
use crate::api::platform::{LimitKey, OptionalFeature};
use crate::api::resource::TextureAspects;
use crate::api::resource::buffer::{BufferSupport, BufferSupportLimits, BufferUsage};
use crate::api::resource::route::{
    BufferCopyLayoutLimits, RouteCapabilities, RouteQuery, RouteSupport, TexelCopyLayoutLimits,
};
use crate::api::resource::texture::{
    Extent3d, TextureDimension, TextureUsage, TextureViewCompatibility,
};
use crate::api::shader::vocabulary::AcceptedCodeForm;
use crate::api::shader::{ShaderStage, ShaderStages};
use crate::backend::vulkan::ffi;

use crate::backend::vulkan::format::{FORMATS, vk_format};

#[derive(Clone, Copy)]
pub(super) struct VulkanCapabilityLimits {
    pub(super) max_bindings_per_group: u32,
    pub(super) max_bound_descriptor_sets: u32,
    pub(super) max_per_stage_uniform_buffers: u32,
    pub(super) max_per_stage_storage_buffers: u32,
    pub(super) min_uniform_buffer_offset_alignment: u64,
    pub(super) min_storage_buffer_offset_alignment: u64,
    pub(super) max_compute_work_group_invocations: u32,
    pub(super) max_compute_work_group_size: [u32; 3],
    pub(super) max_compute_work_group_count: [u32; 3],
    pub(super) max_compute_shared_memory_size: u32,
}

/// Probes the resource-creation subset of Vulkan 1.0 exposed by this backend.
pub(super) fn probe(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    general_ceiling: u64,
    uniform_ceiling: u64,
    storage_ceiling: u64,
    limits: VulkanCapabilityLimits,
) -> RhiResult<CapabilityFacts> {
    let mut facts = CapabilityFacts::empty();
    facts.record_code_form(AcceptedCodeForm::SpirV);
    record_compute_and_binding(&mut facts, limits);
    facts.record_limit(LimitKey::MaxUniformBufferBindingSize, uniform_ceiling);
    record_buffer_support(
        &mut facts,
        general_ceiling,
        uniform_ceiling,
        storage_ceiling,
    );
    facts.record_limit(LimitKey::MaxBufferSize, general_ceiling);
    facts.record_limit(LimitKey::MaxStorageBufferBindingSize, storage_ceiling);
    facts.record_route(
        RouteQuery::BufferToBuffer,
        RouteSupport::Supported(RouteCapabilities::new(
            Some(BufferCopyLayoutLimits::new(4, 4)),
            None,
        )),
    );

    for &format in FORMATS {
        let native = vk_format(format).expect("FORMATS contains only mapped Vulkan formats");
        let properties =
            unsafe { instance.get_physical_device_format_properties(physical, native) };
        // These routes are only published after the exact native format has
        // reported the matching optimal-tiling transfer feature.  The command
        // spine has real `vkCmdCopy*` lowering for this subset; resolve and
        // blit deliberately remain absent until their own conformance slices.
        let features = properties.optimal_tiling_features;
        let texel_limits = Some(TexelCopyLayoutLimits::new(4, 4));
        for dimension in [
            TextureDimension::D1,
            TextureDimension::D2,
            TextureDimension::D3,
        ]
        .into_iter()
        .filter(|_| format_aspects(format).contains(TextureAspects::COLOR))
        {
            if features.contains(vk::FormatFeatureFlags::TRANSFER_DST) {
                facts.record_route(
                    RouteQuery::BufferToTexture {
                        dimension,
                        format,
                        aspect: crate::api::resource::subresource::TextureAspect::Color,
                    },
                    RouteSupport::Supported(RouteCapabilities::new(None, texel_limits)),
                );
            }
            if features.contains(vk::FormatFeatureFlags::TRANSFER_SRC) {
                facts.record_route(
                    RouteQuery::TextureToBuffer {
                        dimension,
                        format,
                        aspect: crate::api::resource::subresource::TextureAspect::Color,
                    },
                    RouteSupport::Supported(RouteCapabilities::new(None, texel_limits)),
                );
            }
            if features.contains(
                vk::FormatFeatureFlags::TRANSFER_SRC | vk::FormatFeatureFlags::TRANSFER_DST,
            ) {
                facts.record_route(
                    RouteQuery::TextureToTexture {
                        src_dimension: dimension,
                        src_format: format,
                        src_aspect: crate::api::resource::subresource::TextureAspect::Color,
                        src_sample_count: 1,
                        dst_dimension: dimension,
                        dst_format: format,
                        dst_aspect: crate::api::resource::subresource::TextureAspect::Color,
                        dst_sample_count: 1,
                    },
                    RouteSupport::Supported(RouteCapabilities::new(None, None)),
                );
            }
        }
        for dimension in [
            TextureDimension::D1,
            TextureDimension::D2,
            TextureDimension::D3,
        ] {
            for usage in TextureUsage::all().filter(|usage| !usage.is_empty()) {
                for view_compatibility in [
                    TextureViewCompatibility::NONE,
                    TextureViewCompatibility::CUBE,
                ] {
                    let Some(limits) = texture_properties(
                        instance,
                        physical,
                        properties.optimal_tiling_features,
                        native,
                        dimension,
                        usage,
                        view_compatibility,
                    )?
                    else {
                        continue;
                    };
                    // `vkGetPhysicalDeviceImageFormatProperties` answers the
                    // native image tuple once and returns its whole sample-mask.
                    // The RHI key then selects a member from that mask; querying
                    // the driver again for each member would be identical work.
                    for sample_count in [1, 2, 4, 8, 16, 32, 64] {
                        // Keep requirements queries inside the public P0
                        // descriptor domain too: only 2D images may be
                        // multisampled, and cube-compatible images must be 1x.
                        if (dimension != TextureDimension::D2
                            || view_compatibility == TextureViewCompatibility::CUBE)
                            && sample_count != 1
                        {
                            continue;
                        }
                        if !limits
                            .sample_counts
                            .contains(vk::SampleCountFlags::from_raw(sample_count))
                        {
                            continue;
                        }
                        let query =
                            TextureSupportQuery::new(dimension, format, usage, sample_count)
                                .with_view_compatibility(view_compatibility);
                        facts.record_texture_support(
                            &query,
                            TextureSupport::Supported(TextureSupportLimits::new(
                                Extent3d {
                                    width: limits.max_extent.width,
                                    height: limits.max_extent.height,
                                    depth: limits.max_extent.depth,
                                },
                                limits.max_mip_levels,
                                limits.max_array_layers,
                            )),
                        );
                    }
                }
            }
        }
    }
    Ok(facts)
}

fn record_compute_and_binding(facts: &mut CapabilityFacts, limits: VulkanCapabilityLimits) {
    facts.record_feature(OptionalFeature::Compute);
    facts.record_limit(
        LimitKey::MaxBindGroups,
        u64::from(limits.max_bound_descriptor_sets),
    );
    // The portable key bounds both entry count and slot number, while Vulkan
    // separates sparse binding numbers from descriptor-count limits. Publish a
    // conservative buffer-only subset derived from both per-stage and per-set
    // limits; accepting fewer sparse slot numbers is preferable to validating a
    // layout the device cannot populate.
    facts.record_limit(
        LimitKey::MaxBindingsPerGroup,
        u64::from(limits.max_bindings_per_group),
    );
    facts.record_limit(LimitKey::MaxDynamicUniformBuffersPerPipelineLayout, 0);
    facts.record_limit(LimitKey::MaxDynamicStorageBuffersPerPipelineLayout, 0);
    facts.record_limit(
        LimitKey::MinUniformBufferOffsetAlignment,
        limits.min_uniform_buffer_offset_alignment,
    );
    facts.record_limit(
        LimitKey::MinStorageBufferOffsetAlignment,
        limits.min_storage_buffer_offset_alignment,
    );
    facts.record_limit(
        LimitKey::MaxComputeInvocationsPerWorkgroup,
        u64::from(limits.max_compute_work_group_invocations),
    );
    facts.record_limit(
        LimitKey::MaxComputeWorkgroupSizeX,
        u64::from(limits.max_compute_work_group_size[0]),
    );
    facts.record_limit(
        LimitKey::MaxComputeWorkgroupSizeY,
        u64::from(limits.max_compute_work_group_size[1]),
    );
    facts.record_limit(
        LimitKey::MaxComputeWorkgroupSizeZ,
        u64::from(limits.max_compute_work_group_size[2]),
    );
    facts.record_limit(
        LimitKey::MaxComputeWorkgroupsPerDimension,
        u64::from(
            *limits
                .max_compute_work_group_count
                .iter()
                .min()
                .unwrap_or(&0),
        ),
    );
    facts.record_limit(
        LimitKey::MaxComputeWorkgroupStorageSize,
        u64::from(limits.max_compute_shared_memory_size),
    );
    facts.record_binding_limit(
        ShaderStage::Compute,
        BindingLimitClass::UniformBuffers,
        limits.max_per_stage_uniform_buffers,
    );
    facts.record_binding_limit(
        ShaderStage::Compute,
        BindingLimitClass::StorageBuffers,
        limits.max_per_stage_storage_buffers,
    );
    // Fixed descriptor arrays are implemented by the descriptor writer, but
    // remain unadvertised until a native conformance case closes that route.
    // Capability publication follows proven lowering, not Vulkan's theoretical
    // descriptor-set vocabulary.
    for array in [false] {
        facts.record_binding_support(
            BindingSupportKey {
                visibility: ShaderStages::COMPUTE,
                kind: BindableKind::UniformBuffer,
                array,
                dynamic_offset: false,
            },
            BindingSupport::Supported,
        );
        for access in [
            BufferBindingAccess::ReadOnly,
            BufferBindingAccess::ReadWrite,
        ] {
            facts.record_binding_support(
                BindingSupportKey {
                    visibility: ShaderStages::COMPUTE,
                    kind: BindableKind::StorageBuffer { access },
                    array,
                    dynamic_offset: false,
                },
                BindingSupport::Supported,
            );
        }
    }
}

fn record_buffer_support(
    facts: &mut CapabilityFacts,
    general_ceiling: u64,
    uniform_ceiling: u64,
    storage_ceiling: u64,
) {
    for usage in BufferUsage::all() {
        let support = if usage.is_empty() {
            BufferSupport::Unsupported
        } else {
            let mut ceiling = general_ceiling;
            if usage.contains(BufferUsage::UNIFORM) {
                ceiling = ceiling.min(uniform_ceiling);
            }
            if usage.contains(BufferUsage::STORAGE) {
                ceiling = ceiling.min(storage_ceiling);
            }
            BufferSupport::Supported(BufferSupportLimits::new(ceiling))
        };
        facts.record_buffer_support(usage, support);
    }
}

fn texture_properties(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    features: vk::FormatFeatureFlags,
    format: vk::Format,
    dimension: TextureDimension,
    usage: TextureUsage,
    view_compatibility: TextureViewCompatibility,
) -> RhiResult<Option<vk::ImageFormatProperties>> {
    if usage.is_empty() || !format_features_cover(features, usage) {
        return Ok(None);
    }
    if view_compatibility == TextureViewCompatibility::CUBE && dimension != TextureDimension::D2 {
        return Ok(None);
    }
    let flags = if view_compatibility == TextureViewCompatibility::CUBE {
        vk::ImageCreateFlags::CUBE_COMPATIBLE
    } else {
        vk::ImageCreateFlags::empty()
    };
    let result = unsafe {
        instance.get_physical_device_image_format_properties(
            physical,
            format,
            image_type(dimension),
            vk::ImageTiling::OPTIMAL,
            image_usage(usage),
            flags,
        )
    };
    let properties = match result {
        Ok(properties) => properties,
        Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => return Ok(None),
        Err(result) => {
            return Err(ffi::to_rhi(
                result,
                "VulkanProvider::probe_texture_image_format",
            ));
        }
    };
    // Cube intent has a descriptor-level minimum of six layers.  A native
    // format query that cannot allocate that many layers is not a useful cube
    // capability even if it accepted the creation flag.
    if view_compatibility == TextureViewCompatibility::CUBE && properties.max_array_layers < 6 {
        return Ok(None);
    }
    Ok(Some(properties))
}

fn format_features_cover(features: vk::FormatFeatureFlags, usage: TextureUsage) -> bool {
    (!usage.contains(TextureUsage::SAMPLED)
        || features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE))
        && (!usage.contains(TextureUsage::STORAGE)
            || features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE))
        && (!usage.contains(TextureUsage::COLOR_ATTACHMENT)
            || features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT))
        && (!usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT)
            || features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT))
}

fn image_type(dimension: TextureDimension) -> vk::ImageType {
    match dimension {
        TextureDimension::D1 => vk::ImageType::TYPE_1D,
        TextureDimension::D2 => vk::ImageType::TYPE_2D,
        TextureDimension::D3 => vk::ImageType::TYPE_3D,
    }
}

fn image_usage(usage: TextureUsage) -> vk::ImageUsageFlags {
    let mut native = vk::ImageUsageFlags::empty();
    if usage.contains(TextureUsage::COPY_SRC) {
        native |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(TextureUsage::COPY_DST) {
        native |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    if usage.contains(TextureUsage::SAMPLED) {
        native |= vk::ImageUsageFlags::SAMPLED;
    }
    if usage.contains(TextureUsage::STORAGE) {
        native |= vk::ImageUsageFlags::STORAGE;
    }
    if usage.contains(TextureUsage::COLOR_ATTACHMENT) {
        native |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    if usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
        native |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    }
    native
}
