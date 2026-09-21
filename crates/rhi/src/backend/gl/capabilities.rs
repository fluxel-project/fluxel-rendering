//! Translation from observed GL-family evidence into the frozen RHI facts.
//!
//! This is intentionally a *narrowing* bridge.  A GL version, extension string,
//! or a format enum alone is never a public capability.  Native and browser
//! discovery first provide this module with the complete entry-point and exact
//! format/limit observations; this module then publishes only operations for
//! which the v13 command lowering has opted in.

use crate::api::binding::vocabulary::BindableKind;
use crate::api::binding::{BindingLimitClass, BindingSupport, SamplerKind, TextureSampleType};
use crate::api::capability::BindingSupportKey;
use crate::api::capability::CapabilityFacts;
use crate::api::format::{
    FormatFacts, StorageAccessSupport, TextureFormat, TextureSupport, TextureSupportLimits,
    TextureSupportQuery,
};
use crate::api::platform::{LimitKey, OptionalFeature};
use crate::api::resource::buffer::{BufferSupport, BufferSupportLimits, BufferUsage};
use crate::api::resource::route::{
    BufferCopyLayoutLimits, RouteCapabilities, RouteQuery, RouteSupport,
};
use crate::api::resource::texture::TextureUsage;
use crate::api::resource::view::TextureViewDimension;
use crate::api::shader::vocabulary::AcceptedCodeForm;
use crate::api::shader::{ShaderStage, ShaderStages};

use super::api::{GlFamilyProfile, GlLimits};
use super::facts::{GlFeature, GlFeatureProbe};

/// Exact native format facts.  A provider creates one only after the matching
/// `glGetInternalformativ` / WebGL extension route and all creation/upload/view
/// lowering paths have been established for this context generation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GlFormatEvidence {
    pub(crate) format: TextureFormat,
    pub(crate) storage_read: bool,
    pub(crate) storage_write: bool,
    pub(crate) storage_read_write: bool,
    pub(crate) color_attachment: bool,
    pub(crate) depth_attachment: bool,
    pub(crate) stencil_attachment: bool,
    pub(crate) blendable: bool,
    pub(crate) filterable: bool,
    pub(crate) storage_atomic: bool,
}

/// An exact descriptor support result obtained from the same context that owns
/// this device.  Supplying this separately prevents the bridge from inventing
/// texture combinations from `FormatFacts`.
#[derive(Clone, Debug)]
pub(crate) struct GlTextureEvidence {
    pub(crate) query: TextureSupportQuery,
    pub(crate) limits: TextureSupportLimits,
}

/// Backend-lowering closure, not a driver capability.  Every `true` means that
/// the corresponding v13 command path exists and has a conformance case.  This
/// makes it impossible for a freshly-discovered extension to become public
/// support merely because its native entry points were loaded.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GlLoweringClosure {
    pub(crate) buffers: bool,
    pub(crate) textures: bool,
    pub(crate) buffer_copy: bool,
    pub(crate) texture_copy: bool,
    /// Scalar UBO, sampled-texture, and sampler packet lowering, including a
    /// trustworthy shader-artifact mapping from logical `(group, slot)` pairs
    /// to native GL reflection names. Fixed/runtime arrays and dynamic offsets
    /// remain separately fail-closed.
    pub(crate) bindings: bool,
    pub(crate) raster_indirect: bool,
    pub(crate) compute: bool,
    pub(crate) indirect_dispatch: bool,
    pub(crate) multi_draw_indirect: bool,
    pub(crate) occlusion_query: bool,
    pub(crate) timestamp_query: bool,
    pub(crate) sampler_anisotropy: bool,
    /// Compressed upload/copy lowering. Format family admission remains
    /// independently controlled by `GlFeatureProbe`.
    pub(crate) compressed_upload: bool,
}

/// One immutable input gathered by native WGL/EGL or browser WebGL discovery.
/// It owns no context handle and contains no browser session/token concept.
#[derive(Clone, Debug)]
pub(crate) struct GlCapabilitySnapshot {
    pub(crate) profile: GlFamilyProfile,
    pub(crate) feature_probe: GlFeatureProbe,
    pub(crate) limits: GlLimits,
    pub(crate) maximum_buffer_size: u64,
    pub(crate) formats: Vec<GlFormatEvidence>,
    pub(crate) textures: Vec<GlTextureEvidence>,
    pub(crate) lowering: GlLoweringClosure,
}

impl GlCapabilitySnapshot {
    /// Produces facts for this exact context generation.  Missing texture or
    /// format entries deliberately remain negative; no generic GL default is
    /// inferred here.
    pub(crate) fn into_facts(self) -> CapabilityFacts {
        let mut facts = CapabilityFacts::empty();
        facts.record_code_form(match self.profile {
            GlFamilyProfile::Desktop { .. } => AcceptedCodeForm::Glsl,
            GlFamilyProfile::Embedded { .. } | GlFamilyProfile::WebGl2 => AcceptedCodeForm::GlslEs,
        });
        self.record_limits(&mut facts);
        self.record_features(&mut facts);
        self.record_bindings(&mut facts);
        if self.lowering.buffer_copy {
            facts.record_route(
                RouteQuery::BufferToBuffer,
                RouteSupport::Supported(RouteCapabilities::new(
                    Some(BufferCopyLayoutLimits::new(1, 1)),
                    None,
                )),
            );
        }
        if self.lowering.texture_copy {
            for texture in &self.textures {
                let query = &texture.query;
                if query.sample_count() != 1
                    || !query.usage().contains(TextureUsage::COPY_SRC)
                    || !query.usage().contains(TextureUsage::COPY_DST)
                    || !crate::api::format::format_aspects(query.format())
                        .contains(crate::api::resource::TextureAspects::COLOR)
                {
                    continue;
                }
                facts.record_route(
                    RouteQuery::TextureToTexture {
                        src_dimension: query.dimension(),
                        src_format: query.format(),
                        src_aspect: crate::api::resource::TextureAspect::Color,
                        src_sample_count: 1,
                        dst_dimension: query.dimension(),
                        dst_format: query.format(),
                        dst_aspect: crate::api::resource::TextureAspect::Color,
                        dst_sample_count: 1,
                    },
                    RouteSupport::Supported(RouteCapabilities::new(None, None)),
                );
            }
        }
        // Texture uploads use this route key even though GL supplies client
        // memory directly. A texture support row with COPY_DST is emitted only
        // when the matching Phase-A packet can lower it.
        if self.lowering.textures {
            for texture in &self.textures {
                let query = &texture.query;
                if query.sample_count() == 1
                    && query.usage().contains(TextureUsage::COPY_DST)
                    && (is_compressed(query.format())
                        || matches!(
                            query.format(),
                            TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
                        ))
                    && crate::api::format::format_aspects(query.format())
                        .contains(crate::api::resource::TextureAspects::COLOR)
                {
                    facts.record_route(
                        RouteQuery::BufferToTexture {
                            dimension: query.dimension(),
                            format: query.format(),
                            aspect: crate::api::resource::TextureAspect::Color,
                        },
                        RouteSupport::Supported(RouteCapabilities::new(None, None)),
                    );
                }
            }
        }
        // Buffer support is an enumerable capability family: every mask must
        // be recorded, including empty and masks containing a route this GL
        // lowering has not closed.  Recording only the positive singleton
        // masks would make a perfectly ordinary VERTEX|COPY_DST query panic.
        let mut ordinary = BufferUsage::COPY_SRC
            .union(BufferUsage::COPY_DST)
            .union(BufferUsage::VERTEX)
            .union(BufferUsage::INDEX)
            .union(BufferUsage::UNIFORM);
        if self.lowering.raster_indirect || self.lowering.indirect_dispatch {
            ordinary = ordinary.union(BufferUsage::INDIRECT);
        }
        for usage in BufferUsage::all() {
            let support = if self.lowering.buffers
                && self.maximum_buffer_size != 0
                && !usage.is_empty()
                && usage.is_subset_of(ordinary)
            {
                BufferSupport::Supported(BufferSupportLimits::new(self.maximum_buffer_size))
            } else {
                BufferSupport::Unsupported
            };
            facts.record_buffer_support(usage, support);
        }
        if self.lowering.textures {
            for evidence in &self.formats {
                if !self.format_family_admitted(evidence.format) {
                    continue;
                }
                let format = evidence.format;
                facts.record_format(
                    format,
                    FormatFacts::new(
                        format,
                        StorageAccessSupport::new(
                            evidence.storage_read,
                            evidence.storage_write,
                            evidence.storage_read_write,
                        ),
                        evidence.color_attachment,
                        evidence.depth_attachment,
                        evidence.stencil_attachment,
                        evidence.blendable,
                    )
                    .with_sampling_and_atomic(evidence.filterable, evidence.storage_atomic),
                );
            }
            for texture in &self.textures {
                if self.format_family_admitted(texture.query.format()) {
                    facts.record_texture_support(
                        &texture.query,
                        TextureSupport::Supported(texture.limits),
                    );
                }
            }
        }
        facts
    }

    fn record_limits(&self, facts: &mut CapabilityFacts) {
        let l = self.limits;
        // Limits are portable lowering promises, not a dump of GL integer
        // state. Derive texture limits from the exact descriptor evidence so
        // the current 2D-only backends do not accidentally advertise native
        // 1D/3D/array ceilings for shapes they cannot create yet.
        let mut max_1d = 0;
        let mut max_2d = 0;
        let mut max_3d = 0;
        let mut max_layers = 0;
        for texture in &self.textures {
            let limits = texture.limits;
            let extent = limits.max_extent();
            match texture.query.dimension() {
                crate::api::resource::texture::TextureDimension::D1 => {
                    max_1d = max_1d.max(extent.width)
                }
                crate::api::resource::texture::TextureDimension::D2 => {
                    max_2d = max_2d.max(extent.width.max(extent.height));
                    max_layers = max_layers.max(limits.max_array_layers());
                }
                crate::api::resource::texture::TextureDimension::D3 => {
                    max_3d = max_3d.max(extent.width.max(extent.height).max(extent.depth))
                }
            }
        }
        record_nonzero(facts, LimitKey::MaxTexture1dDimension, u64::from(max_1d));
        record_nonzero(facts, LimitKey::MaxTexture2dDimension, u64::from(max_2d));
        record_nonzero(facts, LimitKey::MaxTexture3dDimension, u64::from(max_3d));
        record_nonzero(
            facts,
            LimitKey::MaxTextureArrayLayers,
            u64::from(max_layers),
        );
        record_nonzero(
            facts,
            LimitKey::MaxColorAttachments,
            u64::from(l.max_color_attachments.min(l.max_draw_buffers)),
        );
        record_nonzero(
            facts,
            LimitKey::MaxVertexAttributes,
            u64::from(l.max_vertex_attributes),
        );
        record_nonzero(
            facts,
            LimitKey::MaxUniformBufferBindingSize,
            l.max_uniform_block_size,
        );
        record_nonzero(
            facts,
            LimitKey::MinUniformBufferOffsetAlignment,
            l.uniform_buffer_offset_alignment,
        );
        if self.lowering.compute && self.feature_probe.supports(GlFeature::Compute) {
            record_nonzero(
                facts,
                LimitKey::MaxComputeInvocationsPerWorkgroup,
                u64::from(l.max_compute_work_group_invocations),
            );
            record_nonzero(
                facts,
                LimitKey::MaxComputeWorkgroupSizeX,
                u64::from(l.max_compute_work_group_size[0]),
            );
            record_nonzero(
                facts,
                LimitKey::MaxComputeWorkgroupSizeY,
                u64::from(l.max_compute_work_group_size[1]),
            );
            record_nonzero(
                facts,
                LimitKey::MaxComputeWorkgroupSizeZ,
                u64::from(l.max_compute_work_group_size[2]),
            );
            let count = l
                .max_compute_work_group_count
                .into_iter()
                .min()
                .unwrap_or(0);
            record_nonzero(
                facts,
                LimitKey::MaxComputeWorkgroupsPerDimension,
                u64::from(count),
            );
        }
        if self.lowering.sampler_anisotropy
            && self.feature_probe.supports(GlFeature::SamplerAnisotropy)
        {
            let max = l
                .max_texture_anisotropy
                .map(|value| value.get() as u64)
                .unwrap_or(0);
            if max > 1 {
                facts.record_feature(OptionalFeature::SamplerAnisotropy);
                facts.record_limit(LimitKey::MaxSamplerAnisotropy, max);
            }
        }
        record_nonzero(facts, LimitKey::MaxBufferSize, self.maximum_buffer_size);
    }

    fn record_features(&self, facts: &mut CapabilityFacts) {
        let p = &self.feature_probe;
        let c = self.lowering;
        if c.compute && p.supports(GlFeature::Compute) && self.limits.supports_compute() {
            facts.record_feature(OptionalFeature::Compute);
        }
        if c.indirect_dispatch && p.supports(GlFeature::IndirectDispatch) {
            facts.record_feature(OptionalFeature::IndirectDispatch);
        }
        if c.raster_indirect && p.supports(GlFeature::IndirectDraw) {
            facts.record_feature(OptionalFeature::IndirectDraw);
        }
        if c.multi_draw_indirect
            && p.supports(GlFeature::MultiDrawIndirect)
            && self.limits.supports_multi_draw_indirect()
        {
            facts.record_feature(OptionalFeature::MultiDrawIndirect);
        }
        if c.occlusion_query && p.supports(GlFeature::OcclusionQuery) {
            facts.record_feature(OptionalFeature::OcclusionQuery);
        }
        if c.timestamp_query
            && p.supports(GlFeature::TimerQuery)
            && self.limits.query_counter_bits != 0
        {
            facts.record_feature(OptionalFeature::TimestampQuery);
        }
        // There is no separate GL extension to discover for comparison
        // samplers in the profiles this backend accepts. It is nevertheless
        // not a free-standing "GL has it" fact: the public feature licenses a
        // descriptor *and* a `SamplerKind::Comparison` binding, so its one
        // authority is the complete sampler binding lowering route below.
        // Keep this predicate shared with `record_bindings`; publishing either
        // half by itself would let descriptor validation and layout validation
        // disagree about the same operation.
        if self.comparison_sampler_contract_closed() {
            facts.record_feature(OptionalFeature::ComparisonSamplers);
        }
    }

    fn record_bindings(&self, facts: &mut CapabilityFacts) {
        if !self.lowering.bindings {
            return;
        }
        let l = self.limits;
        for (stage, uniforms, textures) in [
            (
                ShaderStage::Vertex,
                l.max_vertex_uniform_blocks,
                l.max_vertex_texture_image_units,
            ),
            (
                ShaderStage::Fragment,
                l.max_fragment_uniform_blocks,
                l.max_fragment_texture_image_units,
            ),
            (
                ShaderStage::Compute,
                l.max_compute_uniform_blocks,
                l.max_combined_texture_image_units,
            ),
        ] {
            if stage == ShaderStage::Compute && !self.lowering.compute {
                continue;
            }
            if uniforms != 0 {
                facts.record_binding_limit(stage, BindingLimitClass::UniformBuffers, uniforms);
            }
            if textures != 0 {
                facts.record_binding_limit(stage, BindingLimitClass::SampledTextures, textures);
                facts.record_binding_limit(stage, BindingLimitClass::Samplers, textures);
            }
        }

        for visibility in crate::api::capability::visibilities() {
            if visibility.contains(ShaderStages::COMPUTE) && !self.lowering.compute {
                continue;
            }
            let stages_have_uniforms = visibility_stages(visibility).all(|stage| {
                (match stage {
                    ShaderStage::Vertex => l.max_vertex_uniform_blocks,
                    ShaderStage::Fragment => l.max_fragment_uniform_blocks,
                    ShaderStage::Compute => l.max_compute_uniform_blocks,
                    _ => 0,
                }) != 0
            });
            if stages_have_uniforms {
                facts.record_binding_support(
                    BindingSupportKey {
                        visibility,
                        kind: BindableKind::UniformBuffer,
                        array: false,
                        runtime_sized: false,
                        dynamic_offset: false,
                    },
                    BindingSupport::Supported,
                );
            }
            let stages_have_textures = visibility_stages(visibility).all(|stage| {
                (match stage {
                    ShaderStage::Vertex => l.max_vertex_texture_image_units,
                    ShaderStage::Fragment => l.max_fragment_texture_image_units,
                    ShaderStage::Compute => l.max_combined_texture_image_units,
                    _ => 0,
                }) != 0
            });
            if !stages_have_textures {
                continue;
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
            // Filtering and non-filtering samplers use the ordinary packet
            // route. Comparison is deliberately separate: it has the same
            // native carrier on accepted GL profiles, but is a public optional
            // feature and must use the exact predicate that publishes it.
            for kind in [SamplerKind::Filtering, SamplerKind::NonFiltering] {
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
            if self.comparison_sampler_contract_closed() {
                facts.record_binding_support(
                    BindingSupportKey {
                        visibility,
                        kind: BindableKind::Sampler {
                            kind: SamplerKind::Comparison,
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

    /// The one GL-family truth for the portable comparison-sampler contract.
    ///
    /// Desktop GL 4.x, GLES 3.x, and WebGL2 all provide the native comparison
    /// state, but a native state bit is insufficient. The RHI must also be
    /// able to carry the sampler through a reflected binding packet. Until
    /// that packet lowering is closed, both creation's optional feature and
    /// the comparison binding answer remain negative.
    fn comparison_sampler_contract_closed(&self) -> bool {
        self.lowering.bindings
    }

    fn format_family_admitted(&self, format: TextureFormat) -> bool {
        use TextureFormat::*;
        let p = &self.feature_probe;
        let compressed = match format {
            Bc1RgbaUnorm | Bc1RgbaUnormSrgb | Bc2RgbaUnorm | Bc2RgbaUnormSrgb | Bc3RgbaUnorm
            | Bc3RgbaUnormSrgb => p.supports(GlFeature::CompressionBcS3tc),
            Bc4RUnorm | Bc4RSnorm | Bc5RgUnorm | Bc5RgSnorm => {
                p.supports(GlFeature::CompressionBcRgtc)
            }
            Bc6hRgbUfloat | Bc6hRgbFloat | Bc7RgbaUnorm | Bc7RgbaUnormSrgb => {
                p.supports(GlFeature::CompressionBcBptc)
            }
            Etc2Rgb8Unorm | Etc2Rgb8UnormSrgb | Etc2Rgb8A1Unorm | Etc2Rgb8A1UnormSrgb
            | Etc2Rgba8Unorm | Etc2Rgba8UnormSrgb | EacR11Unorm | EacR11Snorm | EacRg11Unorm
            | EacRg11Snorm => p.supports(GlFeature::CompressionEtc2Eac),
            Astc4x4Hdr | Astc5x4Hdr | Astc5x5Hdr | Astc6x5Hdr | Astc6x6Hdr | Astc8x5Hdr
            | Astc8x6Hdr | Astc8x8Hdr | Astc10x5Hdr | Astc10x6Hdr | Astc10x8Hdr | Astc10x10Hdr
            | Astc12x10Hdr | Astc12x12Hdr => p.supports(GlFeature::CompressionAstcHdr),
            Astc4x4Unorm | Astc4x4UnormSrgb | Astc5x4Unorm | Astc5x4UnormSrgb | Astc5x5Unorm
            | Astc5x5UnormSrgb | Astc6x5Unorm | Astc6x5UnormSrgb | Astc6x6Unorm
            | Astc6x6UnormSrgb | Astc8x5Unorm | Astc8x5UnormSrgb | Astc8x6Unorm
            | Astc8x6UnormSrgb | Astc8x8Unorm | Astc8x8UnormSrgb | Astc10x5Unorm
            | Astc10x5UnormSrgb | Astc10x6Unorm | Astc10x6UnormSrgb | Astc10x8Unorm
            | Astc10x8UnormSrgb | Astc10x10Unorm | Astc10x10UnormSrgb | Astc12x10Unorm
            | Astc12x10UnormSrgb | Astc12x12Unorm | Astc12x12UnormSrgb => {
                p.supports(GlFeature::CompressionAstcLdr)
            }
            _ => true,
        };
        !is_compressed(format) || (self.lowering.compressed_upload && compressed)
    }
}

fn visibility_stages(visibility: ShaderStages) -> impl Iterator<Item = ShaderStage> {
    [
        ShaderStage::Vertex,
        ShaderStage::Fragment,
        ShaderStage::Compute,
    ]
    .into_iter()
    .filter(move |stage| {
        let flag = match stage {
            ShaderStage::Vertex => ShaderStages::VERTEX,
            ShaderStage::Fragment => ShaderStages::FRAGMENT,
            ShaderStage::Compute => ShaderStages::COMPUTE,
            _ => return false,
        };
        visibility.contains(flag)
    })
}

fn record_nonzero(facts: &mut CapabilityFacts, key: LimitKey, value: u64) {
    if value != 0 {
        facts.record_limit(key, value);
    }
}

fn is_compressed(format: TextureFormat) -> bool {
    // All compressed variants currently have a block footprint rather than a
    // texel byte size. Keeping this predicate next to the family gate makes a
    // newly added compressed format a compile-time review point.
    crate::api::format::logical_bytes_per_block(format).is_some_and(|_| {
        matches!(
            format,
            TextureFormat::Bc1RgbaUnorm
                | TextureFormat::Bc1RgbaUnormSrgb
                | TextureFormat::Bc2RgbaUnorm
                | TextureFormat::Bc2RgbaUnormSrgb
                | TextureFormat::Bc3RgbaUnorm
                | TextureFormat::Bc3RgbaUnormSrgb
                | TextureFormat::Bc4RUnorm
                | TextureFormat::Bc4RSnorm
                | TextureFormat::Bc5RgUnorm
                | TextureFormat::Bc5RgSnorm
                | TextureFormat::Bc6hRgbUfloat
                | TextureFormat::Bc6hRgbFloat
                | TextureFormat::Bc7RgbaUnorm
                | TextureFormat::Bc7RgbaUnormSrgb
                | TextureFormat::Etc2Rgb8Unorm
                | TextureFormat::Etc2Rgb8UnormSrgb
                | TextureFormat::Etc2Rgb8A1Unorm
                | TextureFormat::Etc2Rgb8A1UnormSrgb
                | TextureFormat::Etc2Rgba8Unorm
                | TextureFormat::Etc2Rgba8UnormSrgb
                | TextureFormat::EacR11Unorm
                | TextureFormat::EacR11Snorm
                | TextureFormat::EacRg11Unorm
                | TextureFormat::EacRg11Snorm
                | TextureFormat::Astc4x4Unorm
                | TextureFormat::Astc4x4UnormSrgb
                | TextureFormat::Astc4x4Hdr
                | TextureFormat::Astc5x4Unorm
                | TextureFormat::Astc5x4UnormSrgb
                | TextureFormat::Astc5x4Hdr
                | TextureFormat::Astc5x5Unorm
                | TextureFormat::Astc5x5UnormSrgb
                | TextureFormat::Astc5x5Hdr
                | TextureFormat::Astc6x5Unorm
                | TextureFormat::Astc6x5UnormSrgb
                | TextureFormat::Astc6x5Hdr
                | TextureFormat::Astc6x6Unorm
                | TextureFormat::Astc6x6UnormSrgb
                | TextureFormat::Astc6x6Hdr
                | TextureFormat::Astc8x5Unorm
                | TextureFormat::Astc8x5UnormSrgb
                | TextureFormat::Astc8x5Hdr
                | TextureFormat::Astc8x6Unorm
                | TextureFormat::Astc8x6UnormSrgb
                | TextureFormat::Astc8x6Hdr
                | TextureFormat::Astc8x8Unorm
                | TextureFormat::Astc8x8UnormSrgb
                | TextureFormat::Astc8x8Hdr
                | TextureFormat::Astc10x5Unorm
                | TextureFormat::Astc10x5UnormSrgb
                | TextureFormat::Astc10x5Hdr
                | TextureFormat::Astc10x6Unorm
                | TextureFormat::Astc10x6UnormSrgb
                | TextureFormat::Astc10x6Hdr
                | TextureFormat::Astc10x8Unorm
                | TextureFormat::Astc10x8UnormSrgb
                | TextureFormat::Astc10x8Hdr
                | TextureFormat::Astc10x10Unorm
                | TextureFormat::Astc10x10UnormSrgb
                | TextureFormat::Astc10x10Hdr
                | TextureFormat::Astc12x10Unorm
                | TextureFormat::Astc12x10UnormSrgb
                | TextureFormat::Astc12x10Hdr
                | TextureFormat::Astc12x12Unorm
                | TextureFormat::Astc12x12UnormSrgb
                | TextureFormat::Astc12x12Hdr
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::binding::{BindingCount, BindingKind, BindingSupportQuery};
    use crate::api::capability::AvailableCapabilities;
    use crate::api::resource::buffer::BufferSupportQuery;
    use crate::api::resource::texture::{TextureDimension, TextureUsage};
    use crate::backend::gl::facts::{GlExtension, GlFunction};

    fn baseline(profile: GlFamilyProfile) -> GlCapabilitySnapshot {
        let mut limits = GlLimits::unavailable();
        limits.max_texture_size = 4096;
        limits.max_3d_texture_size = 256;
        limits.max_array_texture_layers = 256;
        limits.max_color_attachments = 4;
        limits.max_draw_buffers = 4;
        limits.max_vertex_attributes = 16;
        limits.max_uniform_block_size = 16_384;
        limits.uniform_buffer_offset_alignment = 256;
        GlCapabilitySnapshot {
            profile,
            feature_probe: GlFeatureProbe::new(profile),
            limits,
            maximum_buffer_size: 1 << 20,
            formats: vec![],
            textures: vec![],
            lowering: GlLoweringClosure {
                buffers: true,
                textures: true,
                ..Default::default()
            },
        }
    }

    #[test]
    fn extension_without_lowering_never_becomes_a_public_feature() {
        let mut snapshot = baseline(GlFamilyProfile::Embedded { major: 3, minor: 1 });
        snapshot
            .feature_probe
            .report_function(GlFunction::DispatchCompute);
        snapshot.limits.max_compute_work_group_count = [1, 1, 1];
        snapshot.limits.max_compute_work_group_size = [1, 1, 1];
        snapshot.limits.max_compute_work_group_invocations = 1;
        let facts = AvailableCapabilities::from_facts(snapshot.into_facts());
        assert!(!facts.supports_feature(OptionalFeature::Compute));
    }

    #[test]
    fn buffer_table_is_total_and_accepts_supported_combinations() {
        let facts = AvailableCapabilities::from_facts(
            baseline(GlFamilyProfile::Desktop { major: 4, minor: 0 }).into_facts(),
        );
        let ordinary = BufferUsage::VERTEX.union(BufferUsage::COPY_DST);
        assert!(
            facts
                .buffer_support(&BufferSupportQuery::new(ordinary))
                .is_supported()
        );
        assert!(
            !facts
                .buffer_support(&BufferSupportQuery::new(BufferUsage::STORAGE))
                .is_supported()
        );
        let empty = BufferUsage::all()
            .next()
            .expect("empty usage is enumerated");
        assert!(
            !facts
                .buffer_support(&BufferSupportQuery::new(empty))
                .is_supported()
        );
    }

    #[test]
    fn feature_needs_native_evidence_lowering_and_limit() {
        let mut snapshot = baseline(GlFamilyProfile::Desktop { major: 4, minor: 6 });
        snapshot
            .feature_probe
            .report_function(GlFunction::DispatchCompute);
        snapshot.limits.max_compute_work_group_count = [8, 8, 8];
        snapshot.limits.max_compute_work_group_size = [8, 8, 8];
        snapshot.limits.max_compute_work_group_invocations = 64;
        snapshot.lowering.compute = true;
        let facts = AvailableCapabilities::from_facts(snapshot.into_facts());
        assert!(facts.supports_feature(OptionalFeature::Compute));
    }

    #[test]
    fn bc_and_etc_are_not_inferred_from_each_other() {
        let mut snapshot = baseline(GlFamilyProfile::WebGl2);
        snapshot.lowering.compressed_upload = true;
        snapshot
            .feature_probe
            .report_extension(GlExtension::OesCompressedEtc2Rgb8Texture);
        snapshot
            .feature_probe
            .report_function(GlFunction::CompressedTexImage2d);
        snapshot
            .feature_probe
            .report_function(GlFunction::CompressedTexSubImage2d);
        snapshot.formats = vec![
            GlFormatEvidence {
                format: TextureFormat::Etc2Rgb8Unorm,
                storage_read: false,
                storage_write: false,
                storage_read_write: false,
                color_attachment: false,
                depth_attachment: false,
                stencil_attachment: false,
                blendable: false,
                filterable: true,
                storage_atomic: false,
            },
            GlFormatEvidence {
                format: TextureFormat::Bc1RgbaUnorm,
                storage_read: false,
                storage_write: false,
                storage_read_write: false,
                color_attachment: false,
                depth_attachment: false,
                stencil_attachment: false,
                blendable: false,
                filterable: true,
                storage_atomic: false,
            },
        ];
        snapshot.textures.push(GlTextureEvidence {
            query: TextureSupportQuery::new(
                TextureDimension::D2,
                TextureFormat::Etc2Rgb8Unorm,
                TextureUsage::SAMPLED,
                1,
            ),
            limits: TextureSupportLimits::new(
                crate::api::resource::texture::Extent3d::d2(4096, 4096),
                8,
                1,
            ),
        });
        let facts = AvailableCapabilities::from_facts(snapshot.into_facts());
        assert!(facts.format(TextureFormat::Etc2Rgb8Unorm).is_some());
        assert!(facts.format(TextureFormat::Bc1RgbaUnorm).is_none());
    }

    #[test]
    fn anisotropy_requires_an_extension_entry_point_lowering_and_real_limit() {
        let mut snapshot = baseline(GlFamilyProfile::WebGl2);
        snapshot.lowering.sampler_anisotropy = true;
        snapshot
            .feature_probe
            .report_extension(GlExtension::ExtTextureFilterAnisotropic);
        snapshot
            .feature_probe
            .report_function(GlFunction::TexParameterAnisotropy);
        snapshot.limits.max_texture_anisotropy = crate::backend::gl::api::GlFiniteF32::new(8.0);
        let facts = AvailableCapabilities::from_facts(snapshot.into_facts());
        assert!(facts.supports_feature(OptionalFeature::SamplerAnisotropy));
        assert_eq!(facts.limit(LimitKey::MaxSamplerAnisotropy), Some(8));
    }

    fn sampler_binding_query(kind: SamplerKind) -> BindingSupportQuery {
        BindingSupportQuery {
            visibility: ShaderStages::VERTEX,
            kind: BindingKind::Sampler { kind },
            count: BindingCount::One,
            dynamic_offset: false,
        }
    }

    #[test]
    fn comparison_sampler_feature_and_binding_share_the_same_closed_route() {
        let mut snapshot = baseline(GlFamilyProfile::WebGl2);
        snapshot.lowering.bindings = true;
        // Binding support additionally needs a real texture-unit ceiling; a
        // closed lowering route alone is not permission to invent one.
        snapshot.limits.max_vertex_texture_image_units = 8;
        snapshot.limits.max_fragment_texture_image_units = 8;
        let facts = AvailableCapabilities::from_facts(snapshot.into_facts());

        // Positive: one closed packet route publishes both halves.
        assert!(facts.supports_feature(OptionalFeature::ComparisonSamplers));
        assert_eq!(
            facts.binding_support(&sampler_binding_query(SamplerKind::Comparison)),
            BindingSupport::Supported
        );

        // Boundary: comparison admission does not change the ordinary sampler
        // families that share the packet route.
        assert_eq!(
            facts.binding_support(&sampler_binding_query(SamplerKind::Filtering)),
            BindingSupport::Supported
        );
        assert_eq!(
            facts.binding_support(&sampler_binding_query(SamplerKind::NonFiltering)),
            BindingSupport::Supported
        );
    }

    #[test]
    fn comparison_sampler_is_fail_closed_until_binding_lowering_exists() {
        let snapshot = baseline(GlFamilyProfile::Desktop { major: 4, minor: 6 });
        // The profile has native compare state, but `baseline` intentionally
        // leaves the sampler binding packet lowering closed.
        let facts = AvailableCapabilities::from_facts(snapshot.into_facts());
        assert!(!facts.supports_feature(OptionalFeature::ComparisonSamplers));
        assert_eq!(
            facts.binding_support(&sampler_binding_query(SamplerKind::Comparison)),
            BindingSupport::Unsupported
        );
    }
}
