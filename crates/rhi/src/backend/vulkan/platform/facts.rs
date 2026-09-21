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

use crate::api::binding::vocabulary::{BindableKind, TextureSampleType};
use crate::api::binding::{
    BindingLimitClass, BindingSupport, BufferBindingAccess, SamplerKind, StorageAccess,
};
use crate::api::capability::{BindingSupportKey, CapabilityFacts};
use crate::api::error::RhiResult;
use crate::api::format::{
    FormatFacts, StorageAccessSupport, TextureSupport, TextureSupportLimits, TextureSupportQuery,
    format_aspects, sample_type,
};
use crate::api::platform::{LimitKey, OptionalFeature};
use crate::api::query::{PipelineStatistics, TimestampQueryCapabilities};
use crate::api::resource::TextureAspects;
use crate::api::resource::buffer::{BufferSupport, BufferSupportLimits, BufferUsage};
use crate::api::resource::route::{
    BufferCopyLayoutLimits, RouteCapabilities, RouteQuery, RouteSupport, TexelCopyLayoutLimits,
};
use crate::api::resource::texture::{
    Extent3d, TextureDimension, TextureUsage, TextureViewCompatibility,
};
use crate::api::resource::view::TextureViewDimension;
use crate::api::shader::ShaderStage;
use crate::api::shader::vocabulary::AcceptedCodeForm;
use crate::backend::vulkan::ffi;

use crate::backend::vulkan::format::{FORMATS, is_astc_hdr, vk_format};

#[derive(Clone, Copy)]
pub(super) struct VulkanCapabilityLimits {
    /// The extension was present and its feature bit was enabled in the
    /// logical-device feature chain. `*_SFLOAT_BLOCK` format properties alone
    /// are insufficient to publish ASTC HDR support.
    pub(super) astc_hdr: bool,
    /// `None` means `VkPhysicalDeviceFeatures::samplerAnisotropy` was not
    /// enabled on the logical device.  The numeric property alone is not a
    /// capability: Vulkan requires both the feature and a non-zero limit.
    pub(super) max_sampler_anisotropy: Option<u32>,
    /// Core `VkPhysicalDeviceFeatures` bits which must be enabled on the
    /// logical device before their portable raster semantics are published.
    pub(super) depth_bias_clamp: bool,
    pub(super) dual_src_blend: bool,
    pub(super) independent_blend: bool,
    pub(super) pipeline_statistics_query: bool,
    pub(super) timestamp_compute_and_graphics: bool,
    pub(super) timestamp_period: f32,
    pub(super) timestamp_valid_bits: u32,
    pub(super) max_push_constants_size: u32,
    pub(super) draw_indirect_first_instance: bool,
    pub(super) multi_draw_indirect: bool,
    /// `VK_KHR_draw_indirect_count` was present during physical-device probe
    /// and will be enabled on the logical device. The baseline creates a 1.0
    /// instance, so even Vulkan 1.2 implementations use this extension route.
    pub(super) draw_indirect_count: bool,
    /// Native `maxDrawIndirectCount`; checked again by lowering because the
    /// public vocabulary has no separate capability-limit key for it yet.
    pub(super) max_draw_indirect_count: u32,
    pub(super) max_bindings_per_group: u32,
    pub(super) max_bound_descriptor_sets: u32,
    pub(super) max_per_stage_uniform_buffers: u32,
    pub(super) max_per_stage_storage_buffers: u32,
    pub(super) max_per_stage_sampled_images: u32,
    pub(super) max_per_stage_storage_images: u32,
    pub(super) max_per_stage_samplers: u32,
    pub(super) min_uniform_buffer_offset_alignment: u64,
    pub(super) min_storage_buffer_offset_alignment: u64,
    pub(super) max_compute_work_group_invocations: u32,
    pub(super) max_compute_work_group_size: [u32; 3],
    pub(super) max_compute_work_group_count: [u32; 3],
    pub(super) max_compute_shared_memory_size: u32,
    pub(super) max_color_attachments: u32,
    pub(super) max_vertex_input_bindings: u32,
    pub(super) max_vertex_input_attributes: u32,
    pub(super) max_vertex_input_binding_stride: u32,
    pub(super) max_inter_stage_variables: u32,
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
    record_pipeline_and_binding(&mut facts, limits);
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
        // `VK_EXT_texture_compression_astc_hdr` owns both the format family
        // and feature bit. Do not even record FormatFacts before device
        // creation has enabled that exact contract.
        if is_astc_hdr(format) && !limits.astc_hdr {
            continue;
        }
        let native = vk_format(format).expect("FORMATS contains only mapped Vulkan formats");
        let properties =
            unsafe { instance.get_physical_device_format_properties(physical, native) };
        let features = properties.optimal_tiling_features;
        let aspects = format_aspects(format);
        let storage = features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE);
        let depth_stencil = features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT);
        facts.record_format(
            format,
            FormatFacts::new(
                format,
                StorageAccessSupport::new(storage, storage, storage),
                features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT),
                depth_stencil && aspects.contains(TextureAspects::DEPTH),
                depth_stencil && aspects.contains(TextureAspects::STENCIL),
                features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND),
            ),
        );
        if storage {
            for visibility in crate::api::capability::visibilities() {
                for dimension in [
                    TextureViewDimension::D1,
                    TextureViewDimension::D2,
                    TextureViewDimension::D2Array,
                    TextureViewDimension::D3,
                ] {
                    for access in [
                        StorageAccess::ReadOnly,
                        StorageAccess::WriteOnly,
                        StorageAccess::ReadWrite,
                    ] {
                        facts.record_binding_support(
                            BindingSupportKey {
                                visibility,
                                kind: BindableKind::StorageTexture {
                                    dimension,
                                    format,
                                    access,
                                },
                                array: false,
                                runtime_sized: false,
                                dynamic_offset: false,
                            },
                            BindingSupport::Supported,
                        );
                    }
                }
            }
        }
        // These routes are only published after the exact native format has
        // reported the matching optimal-tiling transfer feature.  The command
        // spine has real `vkCmdCopy*` lowering for this subset; resolve and
        // blit deliberately remain absent until their own conformance slices.
        let texel_limits = Some(TexelCopyLayoutLimits::new(4, 4).with_image_layout(1, false));
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
                // Vulkan 1.0 framebuffer attachments are 1D/2D image views;
                // the current raster lowering deliberately has no 3D-slice
                // attachment route. Do not let a successful generic image
                // format query advertise a D3 texture that later fails at
                // `vkCreateFramebuffer`.
                if dimension == TextureDimension::D3
                    && (usage.contains(TextureUsage::COLOR_ATTACHMENT)
                        || usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT))
                {
                    continue;
                }
                for view_compatibility in [
                    TextureViewCompatibility::NONE,
                    TextureViewCompatibility::CUBE,
                ] {
                    let Some(limits) = texture_properties(
                        instance,
                        physical,
                        properties.optimal_tiling_features,
                        format,
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
                    // Keep the published raster slice at 1x until its Vulkan
                    // render-pass lowering includes resolve attachments. The
                    // portable scope can request resolve whenever an MSAA color
                    // attachment exists, so advertising native MSAA creation
                    // ahead of that lowering would leave a capability hole.
                    for sample_count in [1] {
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

fn record_pipeline_and_binding(facts: &mut CapabilityFacts, limits: VulkanCapabilityLimits) {
    facts.record_feature(OptionalFeature::Compute);
    facts.record_feature(OptionalFeature::BaseVertex);
    facts.record_feature(OptionalFeature::BaseInstance);
    // VkPipelineMultisampleStateCreateInfo always carries pSampleMask. Sample
    // shading remains separate because it needs the sampleRateShading device
    // feature and this backend currently keeps sampleShadingEnable false.
    facts.record_feature(OptionalFeature::MultisampleMask);
    // `firstInstance` is GPU-provided indirect data, so it cannot be checked
    // by portable recording. Do not publish even one raster indirect draw
    // unless the native feature guarantees that field is honored.
    if limits.draw_indirect_first_instance {
        facts.record_feature(OptionalFeature::IndirectFirstInstance);
        facts.record_feature(OptionalFeature::IndirectDraw);
    }
    // Multi-draw has its own physical feature bit. It still requires the
    // first-instance guarantee because the portable indirect command contains
    // that GPU-owned field and `IndirectDraw` is not a lossy subset.
    if limits.draw_indirect_first_instance && limits.multi_draw_indirect {
        facts.record_feature(OptionalFeature::MultiDrawIndirect);
        if limits.draw_indirect_count {
            facts.record_feature(OptionalFeature::MultiDrawIndirectCount);
        }
    }
    // vkCmdDispatchIndirect is core and has no analogous optional feature bit.
    facts.record_feature(OptionalFeature::IndirectDispatch);
    if limits.max_push_constants_size != 0 {
        facts.record_feature(OptionalFeature::Immediates);
        facts.record_limit(
            LimitKey::MaxImmediateSize,
            u64::from(limits.max_push_constants_size),
        );
        // Vulkan push-constant offset and size are always multiples of four.
        facts.record_limit(LimitKey::ImmediateDataAlignment, 4);
    }
    facts.record_feature(OptionalFeature::ClearBuffer);
    facts.record_feature(OptionalFeature::ClearTexture);
    // VkPipelineCache is core Vulkan. The backend owns the native cache,
    // validates serialized bytes against pipelineCacheUUID/device metadata,
    // and passes it to both compute and graphics creation paths.
    facts.record_feature(OptionalFeature::PipelineCache);
    facts.record_feature(OptionalFeature::PipelineCacheSerialization);
    // MAP_* buffers, including combinations with broader primary GPU usages,
    // are allocated from a compatible HOST_VISIBLE memory type by
    // resource::buffer. This is precisely the additional contract named by
    // MappablePrimaryBuffers; ordinary staging maps are decided by their exact
    // buffer-support rows even on backends that do not publish this feature.
    // Non-coherent allocations use explicit whole-allocation flush/invalidate,
    // so coherence is intentionally not a blanket fact. Mapping maps from
    // allocation offset zero then slices the portable range, making
    // byte-granular map ranges safe.
    facts.record_feature(OptionalFeature::MappablePrimaryBuffers);
    facts.record_limit(LimitKey::MapOffsetAlignment, 1);
    facts.record_limit(LimitKey::MapSizeAlignment, 1);
    // Query pools and result copies are Vulkan core. Query-pool creation can
    // still report OOM, but has no device feature bit beyond the exact
    // pipeline-statistics feature handled below.
    facts.record_feature(OptionalFeature::OcclusionQuery);
    facts.record_feature(OptionalFeature::QueryResolve);
    facts.record_limit(LimitKey::MaxQueriesPerQuerySet, u64::from(u32::MAX));
    facts.record_limit(LimitKey::QueryResolveBufferAlignment, 8);
    if limits.timestamp_compute_and_graphics && limits.timestamp_valid_bits != 0 {
        if let Some(timestamp) = TimestampQueryCapabilities::new(
            f64::from(limits.timestamp_period),
            (limits.timestamp_valid_bits < 64).then_some(limits.timestamp_valid_bits as u8),
            false,
        ) {
            facts.record_feature(OptionalFeature::TimestampQuery);
            facts.record_feature(OptionalFeature::TimestampInsideEncoder);
            // A compute scope is not a Vulkan render pass, so legacy
            // vkCmdWriteTimestamp is valid there. Raster scopes use legacy
            // render passes and deliberately do not publish the raster form.
            facts.record_feature(OptionalFeature::TimestampInsideComputeScope);
            facts.record_timestamp_queries(timestamp);
        }
    }
    if limits.pipeline_statistics_query {
        facts.record_feature(OptionalFeature::PipelineStatisticsQuery);
        facts.record_pipeline_statistics(PipelineStatistics::ALL);
    }
    // Fixed-size descriptor arrays are core Vulkan descriptor-set semantics
    // and the descriptor writer emits all elements atomically. Runtime-sized
    // arrays, partial binding and descriptor indexing are intentionally not
    // implied by this feature and stay unavailable without VK_EXT_descriptor_indexing.
    facts.record_feature(OptionalFeature::BindingArrays);
    facts.record_limit(
        LimitKey::MaxBindingArrayElementsPerShaderStage,
        u64::from(
            limits
                .max_per_stage_uniform_buffers
                .min(limits.max_per_stage_storage_buffers)
                .min(limits.max_per_stage_sampled_images)
                .min(limits.max_per_stage_storage_images)
                .min(limits.max_per_stage_samplers),
        ),
    );
    // These are Vulkan core sampler and blend semantics. Feature-gated raster
    // extensions (depth clamp, non-fill modes, conservative raster) are not
    // advertised here until logical-device feature negotiation enables them.
    facts.record_feature(OptionalFeature::ComparisonSamplers);
    facts.record_feature(OptionalFeature::SamplerClampToBorder);
    if limits.depth_bias_clamp {
        facts.record_feature(OptionalFeature::DepthBiasClamp);
    }
    if limits.dual_src_blend {
        facts.record_feature(OptionalFeature::DualSourceBlending);
    }
    if limits.independent_blend {
        facts.record_feature(OptionalFeature::IndependentBlend);
    }
    if let Some(max_anisotropy) = limits.max_sampler_anisotropy.filter(|max| *max > 1) {
        facts.record_feature(OptionalFeature::SamplerAnisotropy);
        facts.record_limit(LimitKey::MaxSamplerAnisotropy, u64::from(max_anisotropy));
    }
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
    facts.record_limit(
        LimitKey::MaxColorAttachments,
        u64::from(limits.max_color_attachments),
    );
    facts.record_limit(
        LimitKey::MaxVertexBuffers,
        u64::from(limits.max_vertex_input_bindings),
    );
    facts.record_limit(
        LimitKey::MaxVertexAttributes,
        u64::from(limits.max_vertex_input_attributes),
    );
    facts.record_limit(
        LimitKey::MaxVertexBufferArrayStride,
        u64::from(limits.max_vertex_input_binding_stride),
    );
    facts.record_limit(
        LimitKey::MaxInterStageShaderVariables,
        u64::from(limits.max_inter_stage_variables),
    );
    for stage in [
        ShaderStage::Vertex,
        ShaderStage::Fragment,
        ShaderStage::Compute,
    ] {
        facts.record_binding_limit(
            stage,
            BindingLimitClass::UniformBuffers,
            limits.max_per_stage_uniform_buffers,
        );
        facts.record_binding_limit(
            stage,
            BindingLimitClass::StorageBuffers,
            limits.max_per_stage_storage_buffers,
        );
        facts.record_binding_limit(
            stage,
            BindingLimitClass::SampledTextures,
            limits.max_per_stage_sampled_images,
        );
        facts.record_binding_limit(
            stage,
            BindingLimitClass::StorageTextures,
            limits.max_per_stage_storage_images,
        );
        facts.record_binding_limit(
            stage,
            BindingLimitClass::Samplers,
            limits.max_per_stage_samplers,
        );
    }
    // Both scalar and fixed-size packets lower to core Vulkan descriptor sets.
    // Runtime-sized arrays remain absent: their declared count is not known at
    // VkDescriptorSetLayout creation without descriptor-indexing semantics.
    for visibility in crate::api::capability::visibilities() {
        for array in [false, true] {
            facts.record_binding_support(
                BindingSupportKey {
                    visibility,
                    kind: BindableKind::UniformBuffer,
                    array,
                    runtime_sized: false,
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
                        visibility,
                        kind: BindableKind::StorageBuffer { access },
                        array,
                        runtime_sized: false,
                        dynamic_offset: false,
                    },
                    BindingSupport::Supported,
                );
            }

            for dimension in [
                TextureViewDimension::D1,
                TextureViewDimension::D2,
                TextureViewDimension::D2Array,
                TextureViewDimension::Cube,
                TextureViewDimension::CubeArray,
                TextureViewDimension::D3,
            ] {
                for sample_type in [
                    TextureSampleType::Float,
                    TextureSampleType::UnfilterableFloat,
                    TextureSampleType::Sint,
                    TextureSampleType::Uint,
                    TextureSampleType::Depth,
                ] {
                    facts.record_binding_support(
                        BindingSupportKey {
                            visibility,
                            kind: BindableKind::SampledTexture {
                                dimension,
                                sample_type,
                                multisampled: false,
                            },
                            array,
                            runtime_sized: false,
                            dynamic_offset: false,
                        },
                        BindingSupport::Supported,
                    );
                }
            }

            for kind in [
                SamplerKind::Filtering,
                SamplerKind::NonFiltering,
                SamplerKind::Comparison,
            ] {
                facts.record_binding_support(
                    BindingSupportKey {
                        visibility,
                        kind: BindableKind::Sampler { kind },
                        array,
                        runtime_sized: false,
                        dynamic_offset: false,
                    },
                    BindingSupport::Supported,
                );
            }
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
        let support = if usage.is_empty()
            || usage.contains(BufferUsage::BLAS_INPUT)
            || usage.contains(BufferUsage::TLAS_INPUT)
            || usage.contains(BufferUsage::ACCELERATION_STRUCTURE_SCRATCH)
        {
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
    portable_format: crate::api::format::TextureFormat,
    format: vk::Format,
    dimension: TextureDimension,
    usage: TextureUsage,
    view_compatibility: TextureViewCompatibility,
) -> RhiResult<Option<vk::ImageFormatProperties>> {
    if usage.is_empty() || !format_features_cover(portable_format, features, usage) {
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

fn format_features_cover(
    format: crate::api::format::TextureFormat,
    features: vk::FormatFeatureFlags,
    usage: TextureUsage,
) -> bool {
    (!usage.contains(TextureUsage::SAMPLED)
        || features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE))
        // `TextureSampleType::Float` permits filtering in the frozen public
        // vocabulary. Vulkan exposes linear filtering per format, so publishing
        // this tuple without the native bit would make a Filtering sampler
        // pairing pass validation and fail only later at execution.
        && (!usage.contains(TextureUsage::SAMPLED)
            || sample_type(format) != Some(TextureSampleType::Float)
            || features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR))
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
