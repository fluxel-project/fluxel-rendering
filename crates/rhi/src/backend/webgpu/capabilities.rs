//! WebGPU capability discovery translated into the frozen RHI vocabulary.
//!
//! This module deliberately does not turn a WebGPU feature string into an RHI
//! feature by itself.  A string says that a browser/device *could* expose an
//! operation; a public fact is emitted only when it is part of the baseline
//! lowering below.  Keeping this bridge small is particularly important for
//! browsers, where implementations commonly expose experimental feature names
//! ahead of the command path we can actually exercise.

use std::collections::BTreeSet;

use crate::api::binding::vocabulary::{BindableKind, StorageAccess};
use crate::api::binding::{BindingSupport, BufferBindingAccess, SamplerKind, TextureSampleType};
use crate::api::capability::{BindingSupportKey, CapabilityFacts, visibilities};
use crate::api::format::{
    FormatFacts, StorageAccessSupport, TextureFormat, TextureSupport, TextureSupportLimits,
    TextureSupportQuery,
};
use crate::api::platform::{LimitKey, OptionalFeature};
use crate::api::resource::buffer::{BufferSupport, BufferSupportLimits, BufferUsage};
use crate::api::resource::route::{
    BufferCopyLayoutLimits, RouteCapabilities, RouteQuery, RouteSupport, TexelCopyLayoutLimits,
};
use crate::api::resource::subresource::TextureAspect;
use crate::api::resource::texture::{Extent3d, TextureDimension, TextureUsage};
use crate::api::resource::view::TextureViewDimension;
use crate::api::shader::vocabulary::AcceptedCodeForm;
use crate::api::submission::{
    LaneWorkDomains, SubmissionCapabilities, SubmissionLaneClass, SubmissionLaneId,
    SubmissionLaneInfo,
};

/// Raw numeric answers obtained from `GPUSupportedLimits` after device
/// creation.  Zero means that the browser did not provide a usable value; it
/// is never silently replaced by a WebGPU specification minimum.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WebGpuLimits {
    pub(crate) max_buffer_size: u64,
    pub(crate) max_texture_dimension_1d: u32,
    pub(crate) max_texture_dimension_2d: u32,
    pub(crate) max_texture_dimension_3d: u32,
    pub(crate) max_texture_array_layers: u32,
    pub(crate) max_bind_groups: u32,
    pub(crate) max_bindings_per_bind_group: u32,
    pub(crate) max_uniform_buffer_binding_size: u64,
    pub(crate) max_storage_buffer_binding_size: u64,
    pub(crate) min_uniform_buffer_offset_alignment: u64,
    pub(crate) min_storage_buffer_offset_alignment: u64,
    pub(crate) max_color_attachments: u32,
    pub(crate) max_vertex_buffers: u32,
    pub(crate) max_vertex_attributes: u32,
    pub(crate) max_vertex_buffer_array_stride: u64,
    pub(crate) max_inter_stage_shader_variables: u32,
    pub(crate) max_compute_invocations_per_workgroup: u32,
    pub(crate) max_compute_workgroup_size_x: u32,
    pub(crate) max_compute_workgroup_size_y: u32,
    pub(crate) max_compute_workgroup_size_z: u32,
    pub(crate) max_compute_workgroups_per_dimension: u32,
    pub(crate) max_compute_workgroup_storage_size: u64,
}

/// Backend-private discovery input.  Feature names are retained verbatim from
/// WebGPU (`GPUSupportedFeatures` / the enabled `GPUDevice.features`) rather
/// than projected into a public enum at the JS boundary.  That prevents a
/// browser-private spelling from leaking through the RHI API.
#[derive(Clone, Debug, Default)]
pub(crate) struct WebGpuCapabilityInput {
    pub(crate) adapter_features: BTreeSet<String>,
    pub(crate) device_features: BTreeSet<String>,
    pub(crate) limits: WebGpuLimits,
}

impl WebGpuCapabilityInput {
    /// True only if the adapter offered and the created device actually enabled
    /// the exact WebGPU feature.  Testing just `adapter_features` here is a
    /// common browser bug: optional features must be requested when the device
    /// is created, and are unavailable otherwise.
    fn enabled(&self, name: &str) -> bool {
        self.adapter_features.contains(name) && self.device_features.contains(name)
    }

    /// Builds the public fact table.  The core WebGPU command encoder is one
    /// ordered queue, so its one lane deliberately accepts raster, compute, and
    /// copy work.  WebGPU has no portable multi-queue promise.
    pub(crate) fn into_capabilities(self) -> (CapabilityFacts, SubmissionCapabilities) {
        let mut facts = CapabilityFacts::empty();
        facts.record_code_form(AcceptedCodeForm::Wgsl);
        self.record_limits(&mut facts);
        self.record_buffers(&mut facts);
        self.record_bindings(&mut facts);
        self.record_formats_and_texture_routes(&mut facts);

        // These are core WebGPU encoder operations and have a corresponding
        // WebGPU lowering path. Compression is represented per exact texture
        // format below rather than as a coarse public boolean. Deliberately
        // absent: the complete query family, external texture, mesh and ray
        // features. This is stricter than merely checking `timestamp-query`:
        // WebGPU fixes an occlusion query set in the render-pass descriptor,
        // while a portable RasterScope can legally use more than one QuerySet
        // sequentially. WebGPU pass timestamps describe boundaries rather than
        // the portable API's exact arbitrary command positions. Publishing
        // either fact would turn valid recorded work into a submit-time refusal.
        // SamplerAnisotropy is also deliberately absent:
        // WebGPU accepts maxAnisotropy but does not expose a GPUSupportedLimits
        // ceiling that is equivalent to RHI MaxSamplerAnisotropy. A browser's
        // private clamp is not a portable device promise.
        facts.record_feature(OptionalFeature::Compute);
        // `mapAsync`/getMappedRange/unmap is a real browser WebGPU staging-map
        // route. WebGPU restricts MAP_READ to COPY_DST and MAP_WRITE to
        // COPY_SRC, so those exact buffer rows publish ordinary mapping while
        // MappablePrimaryBuffers correctly remains absent.
        facts.record_feature(OptionalFeature::ComparisonSamplers);
        facts.record_feature(OptionalFeature::BaseVertex);
        facts.record_feature(OptionalFeature::ClearBuffer);
        facts.record_feature(OptionalFeature::IndirectDispatch);
        facts.record_feature(OptionalFeature::IndependentBlend);
        // WebGPU's indirect draw commands exist in core, but their portable
        // `first_instance` semantics are gated by this exact optional feature.
        // Fluxel's IndirectDraw contract includes that semantic, so do not
        // publish a weaker half-route on browsers that omit it.
        if self.enabled("indirect-first-instance") {
            facts.record_feature(OptionalFeature::IndirectFirstInstance);
            facts.record_feature(OptionalFeature::IndirectDraw);
        }

        let lane = SubmissionLaneInfo::new(
            SubmissionLaneId::new(0),
            SubmissionLaneClass::General,
            LaneWorkDomains::RASTER
                .union(LaneWorkDomains::COMPUTE)
                .union(LaneWorkDomains::COPY),
        );
        (facts, SubmissionCapabilities::new(vec![lane]))
    }

    fn record_limits(&self, facts: &mut CapabilityFacts) {
        let l = self.limits;
        for (key, value) in [
            (LimitKey::MaxBufferSize, l.max_buffer_size),
            (
                LimitKey::MaxTexture1dDimension,
                u64::from(l.max_texture_dimension_1d),
            ),
            (
                LimitKey::MaxTexture2dDimension,
                u64::from(l.max_texture_dimension_2d),
            ),
            (
                LimitKey::MaxTexture3dDimension,
                u64::from(l.max_texture_dimension_3d),
            ),
            (
                LimitKey::MaxTextureArrayLayers,
                u64::from(l.max_texture_array_layers),
            ),
            (LimitKey::MaxBindGroups, u64::from(l.max_bind_groups)),
            (
                LimitKey::MaxBindingsPerGroup,
                u64::from(l.max_bindings_per_bind_group),
            ),
            (
                LimitKey::MaxUniformBufferBindingSize,
                l.max_uniform_buffer_binding_size,
            ),
            (
                LimitKey::MaxStorageBufferBindingSize,
                l.max_storage_buffer_binding_size,
            ),
            (
                LimitKey::MinUniformBufferOffsetAlignment,
                l.min_uniform_buffer_offset_alignment,
            ),
            (
                LimitKey::MinStorageBufferOffsetAlignment,
                l.min_storage_buffer_offset_alignment,
            ),
            (
                LimitKey::MaxColorAttachments,
                u64::from(l.max_color_attachments),
            ),
            (LimitKey::MaxVertexBuffers, u64::from(l.max_vertex_buffers)),
            (
                LimitKey::MaxVertexAttributes,
                u64::from(l.max_vertex_attributes),
            ),
            (
                LimitKey::MaxVertexBufferArrayStride,
                l.max_vertex_buffer_array_stride,
            ),
            (
                LimitKey::MaxInterStageShaderVariables,
                u64::from(l.max_inter_stage_shader_variables),
            ),
            (
                LimitKey::MaxComputeInvocationsPerWorkgroup,
                u64::from(l.max_compute_invocations_per_workgroup),
            ),
            (
                LimitKey::MaxComputeWorkgroupSizeX,
                u64::from(l.max_compute_workgroup_size_x),
            ),
            (
                LimitKey::MaxComputeWorkgroupSizeY,
                u64::from(l.max_compute_workgroup_size_y),
            ),
            (
                LimitKey::MaxComputeWorkgroupSizeZ,
                u64::from(l.max_compute_workgroup_size_z),
            ),
            (
                LimitKey::MaxComputeWorkgroupsPerDimension,
                u64::from(l.max_compute_workgroups_per_dimension),
            ),
            (
                LimitKey::MaxComputeWorkgroupStorageSize,
                l.max_compute_workgroup_storage_size,
            ),
        ] {
            if value != 0 {
                facts.record_limit(key, value);
            }
        }
    }

    fn record_buffers(&self, facts: &mut CapabilityFacts) {
        let ordinary = BufferUsage::COPY_SRC
            .union(BufferUsage::COPY_DST)
            .union(BufferUsage::VERTEX)
            .union(BufferUsage::INDEX)
            .union(BufferUsage::UNIFORM)
            .union(BufferUsage::STORAGE)
            .union(BufferUsage::INDIRECT)
            .union(BufferUsage::QUERY_RESOLVE)
            .union(BufferUsage::MAP_READ)
            .union(BufferUsage::MAP_WRITE);
        for usage in BufferUsage::all() {
            // WebGPU's map usages are intentionally asymmetric.  They may only
            // accompany their respective copy direction and may not be mixed
            // with each other or with binding/indirect/query usages.
            let map_read_ok = !usage.contains(BufferUsage::MAP_READ)
                || usage.is_subset_of(BufferUsage::MAP_READ.union(BufferUsage::COPY_DST));
            let map_write_ok = !usage.contains(BufferUsage::MAP_WRITE)
                || usage.is_subset_of(BufferUsage::MAP_WRITE.union(BufferUsage::COPY_SRC));
            let supported = !usage.is_empty()
                && usage.is_subset_of(ordinary)
                && map_read_ok
                && map_write_ok
                && self.limits.max_buffer_size != 0;
            facts.record_buffer_support(
                usage,
                if supported {
                    BufferSupport::Supported(BufferSupportLimits::new(self.limits.max_buffer_size))
                } else {
                    BufferSupport::Unsupported
                },
            );
        }
        facts.record_route(
            RouteQuery::BufferToBuffer,
            RouteSupport::Supported(RouteCapabilities::new(
                Some(BufferCopyLayoutLimits::new(4, 4)),
                None,
            )),
        );
    }

    fn record_bindings(&self, facts: &mut CapabilityFacts) {
        // WebGPU's limits are layout-wide; it has no native per-stage-class
        // counters.  Do not manufacture per-stage limits from unrelated
        // `maxBindingsPerBindGroup` data.
        for visibility in visibilities() {
            for kind in [
                BindableKind::UniformBuffer,
                BindableKind::StorageBuffer {
                    access: BufferBindingAccess::ReadOnly,
                },
                BindableKind::StorageBuffer {
                    access: BufferBindingAccess::ReadWrite,
                },
            ] {
                facts.record_binding_support(
                    BindingSupportKey {
                        visibility,
                        kind,
                        array: false,
                        runtime_sized: false,
                        dynamic_offset: false,
                    },
                    BindingSupport::Supported,
                );
            }
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
                            dimension: TextureViewDimension::D2,
                            sample_type,
                            multisampled: false,
                        },
                        array: false,
                        runtime_sized: false,
                        dynamic_offset: false,
                    },
                    BindingSupport::Supported,
                );
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
                        array: false,
                        runtime_sized: false,
                        dynamic_offset: false,
                    },
                    BindingSupport::Supported,
                );
            }
            // WebGPU core exposes only write-only storage textures.  The
            // read-only and read-write WGSL access forms are guarded by this
            // exact optional feature *on the created device*, not merely by a
            // browser advertising it on the adapter.
            let storage_accesses: &[StorageAccess] =
                if self.enabled("readonly_and_readwrite_storage_textures") {
                    &[
                        StorageAccess::ReadOnly,
                        StorageAccess::WriteOnly,
                        StorageAccess::ReadWrite,
                    ]
                } else {
                    &[StorageAccess::WriteOnly]
                };
            // Storage textures are deliberately only emitted for the formats
            // below that have a closed core WebGPU storage route.
            for format in storage_formats() {
                for &access in storage_accesses {
                    facts.record_binding_support(
                        BindingSupportKey {
                            visibility,
                            kind: BindableKind::StorageTexture {
                                dimension: TextureViewDimension::D2,
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

    fn record_formats_and_texture_routes(&self, facts: &mut CapabilityFacts) {
        let l = self.limits;
        let texture_limits = TextureSupportLimits::new(
            Extent3d::d2(l.max_texture_dimension_2d, l.max_texture_dimension_2d),
            mip_levels(l.max_texture_dimension_2d),
            l.max_texture_array_layers,
        );
        let formats = core_formats()
            .chain(compressed_formats(self))
            .collect::<Vec<_>>();
        for format in formats.iter().copied() {
            let info = format_info(format, self.enabled("float32-filterable"));
            let extended_storage_access = self.enabled("readonly_and_readwrite_storage_textures");
            facts.record_format(
                format,
                FormatFacts::new(
                    format,
                    StorageAccessSupport::new(
                        info.storage && extended_storage_access,
                        info.storage,
                        info.storage && extended_storage_access,
                    ),
                    info.color_attachment,
                    info.depth,
                    info.stencil,
                    info.blendable,
                )
                .with_sampling_and_atomic(info.filterable, false),
            );

            let allowed = texture_allowed_usage(info);
            for usage in TextureUsage::all() {
                let query = TextureSupportQuery::new(TextureDimension::D2, format, usage, 1);
                let support = if !usage.is_empty()
                    && allowed.contains(usage)
                    && l.max_texture_dimension_2d != 0
                {
                    TextureSupport::Supported(texture_limits)
                } else {
                    TextureSupport::Unsupported
                };
                facts.record_texture_support(&query, support.clone());

                // WebGPU's `viewFormats` is not an implicit reinterpretation:
                // the alternate format must have been declared at texture
                // creation.  The portable query therefore includes it, and this
                // table must include that exact key rather than treating a
                // compatible view fact as a substitute for creation support.
                if let Some(alternate) = srgb_view_alternate(format) {
                    facts.record_view_compatibility(format, alternate);
                    facts.record_texture_support(
                        &query.clone().with_view_format(alternate),
                        support.clone(),
                    );
                }
                // WebGPU does not need a creation flag for cube views, but the
                // RHI key retains the intent for Vulkan.  A 2D color texture
                // with this intent remains the same native WebGPU texture.
                if info.color {
                    facts.record_texture_support(
                        &query.clone().with_view_compatibility(
                            crate::api::resource::TextureViewCompatibility::CUBE,
                        ),
                        support.clone(),
                    );
                    if let Some(alternate) = srgb_view_alternate(format) {
                        facts.record_texture_support(
                            &query
                                .clone()
                                .with_view_format(alternate)
                                .with_view_compatibility(
                                    crate::api::resource::TextureViewCompatibility::CUBE,
                                ),
                            support,
                        );
                    }
                }
            }
            if info.color {
                facts.record_route(
                    RouteQuery::BufferToTexture {
                        dimension: TextureDimension::D2,
                        format,
                        aspect: TextureAspect::Color,
                    },
                    RouteSupport::Supported(RouteCapabilities::new(
                        None,
                        Some(TexelCopyLayoutLimits::new(4, 256)),
                    )),
                );
                facts.record_route(
                    RouteQuery::TextureToBuffer {
                        dimension: TextureDimension::D2,
                        format,
                        aspect: TextureAspect::Color,
                    },
                    RouteSupport::Supported(RouteCapabilities::new(
                        None,
                        Some(TexelCopyLayoutLimits::new(4, 256)),
                    )),
                );
                facts.record_route(
                    RouteQuery::TextureToTexture {
                        src_dimension: TextureDimension::D2,
                        src_format: format,
                        src_aspect: TextureAspect::Color,
                        src_sample_count: 1,
                        dst_dimension: TextureDimension::D2,
                        dst_format: format,
                        dst_aspect: TextureAspect::Color,
                        dst_sample_count: 1,
                    },
                    RouteSupport::Supported(RouteCapabilities::new(None, None)),
                );
            }
        }

        // WebGPU's core linear/sRGB reinterpretation pairs are not inferred
        // from matching byte widths. They are the exact `viewFormats` pairs
        // defined by WebGPU, and both ends must be in this device's format
        // table before the portable capability may publish either direction.
        // (Compressed sRGB formats remain creatable per-format above; their
        // view-reinterpretation pairs are deliberately withheld until an
        // explicit browser conformance case covers them.)
        for (linear, srgb) in [
            (TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb),
            (TextureFormat::Bgra8Unorm, TextureFormat::Bgra8UnormSrgb),
        ] {
            if formats.contains(&linear) && formats.contains(&srgb) {
                facts.record_view_compatibility(linear, srgb);
                facts.record_view_compatibility(srgb, linear);
            }
        }

        // WebGPU can create 1D and 3D textures, but render/depth attachments
        // and multisampling are 2D-only.  Keep these answers separate from the
        // D2 table above so a valid 3D copy/storage texture does not accidentally
        // inherit a render-target claim.
        for (dimension, limits) in [
            (
                TextureDimension::D1,
                TextureSupportLimits::new(
                    Extent3d::d1(l.max_texture_dimension_1d),
                    mip_levels(l.max_texture_dimension_1d),
                    // WebGPU has no `1d-array` view dimension.  Do not turn
                    // the unrelated 2D array-layer limit into a claim that a
                    // portable 1D array texture/view route exists.
                    1,
                ),
            ),
            (
                TextureDimension::D3,
                TextureSupportLimits::new(
                    Extent3d::d3(
                        l.max_texture_dimension_3d,
                        l.max_texture_dimension_3d,
                        l.max_texture_dimension_3d,
                    ),
                    mip_levels(l.max_texture_dimension_3d),
                    1,
                ),
            ),
        ] {
            let maximum = match dimension {
                TextureDimension::D1 => l.max_texture_dimension_1d,
                TextureDimension::D3 => l.max_texture_dimension_3d,
                TextureDimension::D2 => unreachable!("D2 is handled above"),
            };
            for format in core_formats() {
                let info = format_info(format, self.enabled("float32-filterable"));
                // WebGPU depth/stencil textures are 2D-only.  Do not publish a
                // copy/sample-only subset for 1D/3D merely because those usages
                // themselves do not name an attachment; the native texture
                // descriptor would still be invalid.
                if !info.color {
                    continue;
                }
                let allowed = non_attachment_texture_usage(info);
                for usage in TextureUsage::all() {
                    let query = TextureSupportQuery::new(dimension, format, usage, 1);
                    facts.record_texture_support(
                        &query,
                        if !usage.is_empty() && allowed.contains(usage) && maximum != 0 {
                            TextureSupport::Supported(limits)
                        } else {
                            TextureSupport::Unsupported
                        },
                    );
                }
                if info.color {
                    record_color_copy_routes(facts, dimension, format);
                }
            }
        }
    }
}

fn record_color_copy_routes(
    facts: &mut CapabilityFacts,
    dimension: TextureDimension,
    format: TextureFormat,
) {
    facts.record_route(
        RouteQuery::BufferToTexture {
            dimension,
            format,
            aspect: TextureAspect::Color,
        },
        RouteSupport::Supported(RouteCapabilities::new(
            None,
            Some(TexelCopyLayoutLimits::new(4, 256)),
        )),
    );
    facts.record_route(
        RouteQuery::TextureToBuffer {
            dimension,
            format,
            aspect: TextureAspect::Color,
        },
        RouteSupport::Supported(RouteCapabilities::new(
            None,
            Some(TexelCopyLayoutLimits::new(4, 256)),
        )),
    );
    facts.record_route(
        RouteQuery::TextureToTexture {
            src_dimension: dimension,
            src_format: format,
            src_aspect: TextureAspect::Color,
            src_sample_count: 1,
            dst_dimension: dimension,
            dst_format: format,
            dst_aspect: TextureAspect::Color,
            dst_sample_count: 1,
        },
        RouteSupport::Supported(RouteCapabilities::new(None, None)),
    );
}

#[derive(Clone, Copy)]
struct FormatInfo {
    color: bool,
    color_attachment: bool,
    depth: bool,
    stencil: bool,
    blendable: bool,
    filterable: bool,
    sampled: bool,
    storage: bool,
}

fn texture_allowed_usage(info: FormatInfo) -> TextureUsage {
    let mut usage = TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST);
    if info.sampled {
        usage = usage.union(TextureUsage::SAMPLED);
    }
    if info.storage {
        usage = usage.union(TextureUsage::STORAGE);
    }
    if info.color_attachment {
        usage = usage.union(TextureUsage::COLOR_ATTACHMENT);
    }
    if info.depth || info.stencil {
        usage = usage.union(TextureUsage::DEPTH_STENCIL_ATTACHMENT);
    }
    usage
}

fn non_attachment_texture_usage(info: FormatInfo) -> TextureUsage {
    let mut usage = TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST);
    if info.sampled {
        usage = usage.union(TextureUsage::SAMPLED);
    }
    if info.storage {
        usage = usage.union(TextureUsage::STORAGE);
    }
    usage
}

/// The exact WebGPU `viewFormats` reinterpretation pairs.  No same-byte-size
/// heuristic belongs here: e.g. RGBA and BGRA have the same size but are not a
/// legal WebGPU alternate view pair.
fn srgb_view_alternate(format: TextureFormat) -> Option<TextureFormat> {
    use TextureFormat::*;
    Some(match format {
        Rgba8Unorm => Rgba8UnormSrgb,
        Rgba8UnormSrgb => Rgba8Unorm,
        Bgra8Unorm => Bgra8UnormSrgb,
        Bgra8UnormSrgb => Bgra8Unorm,
        Bc1RgbaUnorm => Bc1RgbaUnormSrgb,
        Bc1RgbaUnormSrgb => Bc1RgbaUnorm,
        Bc2RgbaUnorm => Bc2RgbaUnormSrgb,
        Bc2RgbaUnormSrgb => Bc2RgbaUnorm,
        Bc3RgbaUnorm => Bc3RgbaUnormSrgb,
        Bc3RgbaUnormSrgb => Bc3RgbaUnorm,
        Bc7RgbaUnorm => Bc7RgbaUnormSrgb,
        Bc7RgbaUnormSrgb => Bc7RgbaUnorm,
        Etc2Rgb8Unorm => Etc2Rgb8UnormSrgb,
        Etc2Rgb8UnormSrgb => Etc2Rgb8Unorm,
        Etc2Rgb8A1Unorm => Etc2Rgb8A1UnormSrgb,
        Etc2Rgb8A1UnormSrgb => Etc2Rgb8A1Unorm,
        Etc2Rgba8Unorm => Etc2Rgba8UnormSrgb,
        Etc2Rgba8UnormSrgb => Etc2Rgba8Unorm,
        Astc4x4Unorm => Astc4x4UnormSrgb,
        Astc4x4UnormSrgb => Astc4x4Unorm,
        Astc5x4Unorm => Astc5x4UnormSrgb,
        Astc5x4UnormSrgb => Astc5x4Unorm,
        Astc5x5Unorm => Astc5x5UnormSrgb,
        Astc5x5UnormSrgb => Astc5x5Unorm,
        Astc6x5Unorm => Astc6x5UnormSrgb,
        Astc6x5UnormSrgb => Astc6x5Unorm,
        Astc6x6Unorm => Astc6x6UnormSrgb,
        Astc6x6UnormSrgb => Astc6x6Unorm,
        Astc8x5Unorm => Astc8x5UnormSrgb,
        Astc8x5UnormSrgb => Astc8x5Unorm,
        Astc8x6Unorm => Astc8x6UnormSrgb,
        Astc8x6UnormSrgb => Astc8x6Unorm,
        Astc8x8Unorm => Astc8x8UnormSrgb,
        Astc8x8UnormSrgb => Astc8x8Unorm,
        Astc10x5Unorm => Astc10x5UnormSrgb,
        Astc10x5UnormSrgb => Astc10x5Unorm,
        Astc10x6Unorm => Astc10x6UnormSrgb,
        Astc10x6UnormSrgb => Astc10x6Unorm,
        Astc10x8Unorm => Astc10x8UnormSrgb,
        Astc10x8UnormSrgb => Astc10x8Unorm,
        Astc10x10Unorm => Astc10x10UnormSrgb,
        Astc10x10UnormSrgb => Astc10x10Unorm,
        Astc12x10Unorm => Astc12x10UnormSrgb,
        Astc12x10UnormSrgb => Astc12x10Unorm,
        Astc12x12Unorm => Astc12x12UnormSrgb,
        Astc12x12UnormSrgb => Astc12x12Unorm,
        _ => return None,
    })
}

fn format_info(format: TextureFormat, float32_filterable: bool) -> FormatInfo {
    use TextureFormat::*;
    if compression_feature(format).is_some() {
        // WebGPU's BC/ETC2/ASTC LDR feature sets are sampled, filterable color
        // textures. They never acquire storage or attachment semantics merely
        // by being enabled. ASTC HDR is deliberately absent from
        // `compression_feature`: it has no browser WebGPU texture-format token.
        return FormatInfo {
            color: true,
            color_attachment: false,
            depth: false,
            stencil: false,
            blendable: false,
            filterable: true,
            sampled: true,
            storage: false,
        };
    }
    let depth = matches!(
        format,
        Depth16Unorm | Depth24Plus | Depth24PlusStencil8 | Depth32Float | Depth32FloatStencil8
    );
    let stencil = matches!(
        format,
        Stencil8 | Depth24PlusStencil8 | Depth32FloatStencil8
    );
    let color = !depth && !stencil;
    let integer = matches!(
        format,
        R8Uint
            | R8Sint
            | Rg8Uint
            | Rg8Sint
            | Rgba8Uint
            | Rgba8Sint
            | R16Uint
            | R16Sint
            | Rg16Uint
            | Rg16Sint
            | Rgba16Uint
            | Rgba16Sint
            | R32Uint
            | R32Sint
            | Rg32Uint
            | Rg32Sint
            | Rgba32Uint
            | Rgba32Sint
    );
    let float32 = matches!(format, R32Float | Rg32Float | Rgba32Float);
    let storage = matches!(
        format,
        Rgba8Unorm
            | Rgba8Uint
            | Rgba8Sint
            | Rgba16Float
            | Rgba16Uint
            | Rgba16Sint
            | R32Float
            | R32Uint
            | R32Sint
            | Rg32Float
            | Rg32Uint
            | Rg32Sint
            | Rgba32Float
            | Rgba32Uint
            | Rgba32Sint
    );
    FormatInfo {
        color,
        color_attachment: color && !matches!(format, Rgba32Float | Rgba32Uint | Rgba32Sint),
        depth,
        stencil,
        blendable: color && !integer,
        filterable: !integer && !float32 && !depth && !stencil || (float32 && float32_filterable),
        // Stencil aspects have no WebGPU sampled-texture binding class.  A
        // depth-stencil format remains sampleable through its depth aspect;
        // pure Stencil8 must not inherit that fact merely from the texture
        // usage vocabulary containing SAMPLED.
        sampled: color || depth,
        storage,
    }
}

fn core_formats() -> impl Iterator<Item = TextureFormat> {
    use TextureFormat::*;
    [
        R8Unorm,
        R8Snorm,
        R8Uint,
        R8Sint,
        Rg8Unorm,
        Rg8Snorm,
        Rg8Uint,
        Rg8Sint,
        Rgba8Unorm,
        Rgba8UnormSrgb,
        Rgba8Snorm,
        Rgba8Uint,
        Rgba8Sint,
        Bgra8Unorm,
        Bgra8UnormSrgb,
        R16Float,
        R16Uint,
        R16Sint,
        Rg16Float,
        Rg16Uint,
        Rg16Sint,
        Rgba16Float,
        Rgba16Uint,
        Rgba16Sint,
        R32Float,
        R32Uint,
        R32Sint,
        Rg32Float,
        Rg32Uint,
        Rg32Sint,
        Rgba32Float,
        Rgba32Uint,
        Rgba32Sint,
        Depth24Plus,
        Depth24PlusStencil8,
        Depth32Float,
        Stencil8,
    ]
    .into_iter()
}

/// Emits exactly the LDR compressed formats whose *specific WebGPU feature* is
/// both present on the adapter and enabled on this device.  This is intentionally
/// not a `CompressedTextureApi` bit: an ETC2-only Android adapter must not
/// answer BC or ASTC support.
fn compressed_formats(input: &WebGpuCapabilityInput) -> impl Iterator<Item = TextureFormat> + '_ {
    TextureFormat::all().filter(move |format| {
        compression_feature(*format).is_some_and(|feature| input.enabled(feature))
    })
}

fn compression_feature(format: TextureFormat) -> Option<&'static str> {
    use TextureFormat::*;
    Some(match format {
        Bc1RgbaUnorm | Bc1RgbaUnormSrgb | Bc2RgbaUnorm | Bc2RgbaUnormSrgb | Bc3RgbaUnorm
        | Bc3RgbaUnormSrgb | Bc4RUnorm | Bc4RSnorm | Bc5RgUnorm | Bc5RgSnorm | Bc6hRgbUfloat
        | Bc6hRgbFloat | Bc7RgbaUnorm | Bc7RgbaUnormSrgb => "texture-compression-bc",
        Etc2Rgb8Unorm | Etc2Rgb8UnormSrgb | Etc2Rgb8A1Unorm | Etc2Rgb8A1UnormSrgb
        | Etc2Rgba8Unorm | Etc2Rgba8UnormSrgb | EacR11Unorm | EacR11Snorm | EacRg11Unorm
        | EacRg11Snorm => "texture-compression-etc2",
        Astc4x4Unorm | Astc4x4UnormSrgb | Astc5x4Unorm | Astc5x4UnormSrgb | Astc5x5Unorm
        | Astc5x5UnormSrgb | Astc6x5Unorm | Astc6x5UnormSrgb | Astc6x6Unorm | Astc6x6UnormSrgb
        | Astc8x5Unorm | Astc8x5UnormSrgb | Astc8x6Unorm | Astc8x6UnormSrgb | Astc8x8Unorm
        | Astc8x8UnormSrgb | Astc10x5Unorm | Astc10x5UnormSrgb | Astc10x6Unorm
        | Astc10x6UnormSrgb | Astc10x8Unorm | Astc10x8UnormSrgb | Astc10x10Unorm
        | Astc10x10UnormSrgb | Astc12x10Unorm | Astc12x10UnormSrgb | Astc12x12Unorm
        | Astc12x12UnormSrgb => "texture-compression-astc",
        _ => return None,
    })
}

fn storage_formats() -> impl Iterator<Item = TextureFormat> {
    use TextureFormat::*;
    [
        Rgba8Unorm,
        Rgba8Uint,
        Rgba8Sint,
        Rgba16Float,
        Rgba16Uint,
        Rgba16Sint,
        R32Float,
        R32Uint,
        R32Sint,
        Rg32Float,
        Rg32Uint,
        Rg32Sint,
        Rgba32Float,
        Rgba32Uint,
        Rgba32Sint,
    ]
    .into_iter()
}

fn mip_levels(max_dimension: u32) -> u32 {
    if max_dimension == 0 {
        0
    } else {
        u32::BITS - max_dimension.leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::binding::{BindingCount, BindingKind, BindingSupportQuery};
    use crate::api::capability::AvailableCapabilities;
    use crate::api::resource::buffer::BufferSupportQuery;
    use crate::api::shader::ShaderStages;

    fn input() -> WebGpuCapabilityInput {
        WebGpuCapabilityInput {
            adapter_features: ["float32-filterable".to_owned()].into_iter().collect(),
            device_features: ["float32-filterable".to_owned()].into_iter().collect(),
            limits: WebGpuLimits {
                max_buffer_size: 1024,
                max_texture_dimension_2d: 256,
                max_texture_array_layers: 8,
                max_bind_groups: 4,
                max_bindings_per_bind_group: 16,
                max_uniform_buffer_binding_size: 256,
                max_storage_buffer_binding_size: 256,
                min_uniform_buffer_offset_alignment: 256,
                min_storage_buffer_offset_alignment: 256,
                max_color_attachments: 4,
                max_vertex_buffers: 8,
                max_vertex_attributes: 16,
                max_vertex_buffer_array_stride: 2048,
                max_inter_stage_shader_variables: 16,
                max_compute_invocations_per_workgroup: 256,
                max_compute_workgroup_size_x: 256,
                max_compute_workgroup_size_y: 256,
                max_compute_workgroup_size_z: 64,
                max_compute_workgroups_per_dimension: 65535,
                max_compute_workgroup_storage_size: 16384,
                ..Default::default()
            },
        }
    }

    #[test]
    fn core_path_has_one_general_lane_and_complete_buffer_answers() {
        let (facts, submission) = input().into_capabilities();
        let available = AvailableCapabilities::from_facts(facts);
        assert!(available.supports_feature(OptionalFeature::Compute));
        assert!(!available.supports_feature(OptionalFeature::MappablePrimaryBuffers));
        assert_eq!(submission.lanes().len(), 1);
        assert!(
            submission.lanes()[0].domains().contains(
                LaneWorkDomains::RASTER
                    .union(LaneWorkDomains::COMPUTE)
                    .union(LaneWorkDomains::COPY)
            )
        );
        for usage in BufferUsage::all() {
            let _ = available.buffer_support(&BufferSupportQuery::new(usage));
        }
    }

    #[test]
    fn map_usage_combinations_follow_webgpu_exclusivity() {
        let available = AvailableCapabilities::from_facts(input().into_capabilities().0);
        assert!(
            available
                .buffer_support(&BufferSupportQuery::new(
                    BufferUsage::MAP_READ.union(BufferUsage::COPY_DST)
                ))
                .is_supported()
        );
        assert!(
            !available
                .buffer_support(&BufferSupportQuery::new(
                    BufferUsage::MAP_READ.union(BufferUsage::VERTEX)
                ))
                .is_supported()
        );
        assert!(
            !available
                .buffer_support(&BufferSupportQuery::new(BufferUsage::BLAS_INPUT))
                .is_supported()
        );
    }

    #[test]
    fn adapter_feature_without_device_enablement_is_not_a_public_fact() {
        let mut input = input();
        input.device_features.clear();
        let available = AvailableCapabilities::from_facts(input.into_capabilities().0);
        assert!(
            !available
                .format(TextureFormat::R32Float)
                .expect("core format")
                .filterable()
        );
    }

    #[test]
    fn query_feature_strings_remain_fail_closed_until_recording_contract_matches() {
        let mut input = input();
        for name in ["timestamp-query", "timestamp-query-inside-passes"] {
            input.adapter_features.insert(name.to_owned());
            input.device_features.insert(name.to_owned());
        }
        let (facts, submission) = input.into_capabilities();
        let available = AvailableCapabilities::from_facts(facts.clone());
        let enabled = crate::api::capability::EnabledCapabilities::from_facts(facts, submission);

        // The native feature names are not enough: see the WebGPU
        // query-profile boundary in the design document.
        for feature in [
            OptionalFeature::OcclusionQuery,
            OptionalFeature::TimestampQuery,
            OptionalFeature::TimestampInsideEncoder,
            OptionalFeature::TimestampInsideRasterScope,
            OptionalFeature::TimestampInsideComputeScope,
            OptionalFeature::QueryResolve,
        ] {
            assert!(!available.supports_feature(feature));
        }
        assert_eq!(available.limit(LimitKey::MaxQueriesPerQuerySet), None);
        assert_eq!(available.limit(LimitKey::QueryResolveBufferAlignment), None);
        assert_eq!(enabled.timestamp_queries().period_nanos, None);
    }

    #[test]
    fn extended_storage_texture_access_requires_device_enablement() {
        let available = AvailableCapabilities::from_facts(input().into_capabilities().0);
        let storage = available
            .format(TextureFormat::Rgba8Unorm)
            .expect("core format")
            .storage_access();
        assert!(!storage.supports(StorageAccess::ReadOnly));
        assert!(storage.supports(StorageAccess::WriteOnly));
        assert!(!storage.supports(StorageAccess::ReadWrite));
        let read_only = BindingSupportQuery {
            visibility: ShaderStages::FRAGMENT,
            kind: BindingKind::StorageTexture {
                dimension: TextureViewDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                access: StorageAccess::ReadOnly,
            },
            count: BindingCount::One,
            dynamic_offset: false,
        };
        assert!(!available.binding_support(&read_only).is_supported());

        let mut input = input();
        input
            .adapter_features
            .insert("readonly_and_readwrite_storage_textures".to_owned());
        input
            .device_features
            .insert("readonly_and_readwrite_storage_textures".to_owned());
        let extended = AvailableCapabilities::from_facts(input.into_capabilities().0);
        let storage = extended
            .format(TextureFormat::Rgba8Unorm)
            .expect("core format")
            .storage_access();
        assert!(storage.supports(StorageAccess::ReadOnly));
        assert!(storage.supports(StorageAccess::WriteOnly));
        assert!(storage.supports(StorageAccess::ReadWrite));
        assert!(extended.binding_support(&read_only).is_supported());
    }

    #[test]
    fn compressed_formats_are_gated_per_family_and_never_publish_astc_hdr() {
        let mut input = input();
        input
            .adapter_features
            .insert("texture-compression-bc".to_owned());
        input
            .device_features
            .insert("texture-compression-bc".to_owned());
        let available = AvailableCapabilities::from_facts(input.into_capabilities().0);

        let bc = TextureSupportQuery::new(
            TextureDimension::D2,
            TextureFormat::Bc7RgbaUnormSrgb,
            TextureUsage::COPY_DST.union(TextureUsage::SAMPLED),
            1,
        );
        assert!(available.texture_support(&bc).is_supported());
        assert!(available.format(TextureFormat::Bc7RgbaUnormSrgb).is_some());
        assert!(available.format(TextureFormat::Etc2Rgba8Unorm).is_none());
        assert!(available.format(TextureFormat::Astc4x4Unorm).is_none());
        assert!(available.format(TextureFormat::Astc4x4Hdr).is_none());
    }

    #[test]
    fn adapter_compression_feature_without_device_enablement_is_not_published() {
        let mut input = input();
        input
            .adapter_features
            .insert("texture-compression-etc2".to_owned());
        let available = AvailableCapabilities::from_facts(input.into_capabilities().0);
        assert!(available.format(TextureFormat::Etc2Rgba8Unorm).is_none());
    }

    #[test]
    fn one_and_three_dimensional_textures_do_not_inherit_d2_attachment_claims() {
        let available = AvailableCapabilities::from_facts(input().into_capabilities().0);
        let d1_copy = TextureSupportQuery::new(
            TextureDimension::D1,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COPY_DST.union(TextureUsage::SAMPLED),
            1,
        );
        let d3_storage = TextureSupportQuery::new(
            TextureDimension::D3,
            TextureFormat::Rgba8Unorm,
            TextureUsage::STORAGE,
            1,
        );
        let d3_attachment = TextureSupportQuery::new(
            TextureDimension::D3,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COLOR_ATTACHMENT,
            1,
        );
        assert!(available.texture_support(&d1_copy).is_supported());
        assert!(available.texture_support(&d3_storage).is_supported());
        assert!(!available.texture_support(&d3_attachment).is_supported());
    }
}
