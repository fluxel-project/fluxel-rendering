//! Private resource owners for the first Deferred G-buffer consumer.
//!
//! This module owns only concrete texture allocation facts.  It does not
//! declare Graph accesses, select shaders, or record commands.

use std::{ops::Range, sync::Arc};

use fluxel_rhi::{
    api::{
        BindGroup, Device, DeviceId, Error, ImageLayout, RasterPipeline, Result, SamplerDesc,
        Support, TextureDesc, TextureDimension, TextureFormat, TextureUsage, TextureViewDesc,
        TextureViewDimension, VertexBufferLayout,
    },
    backend::ResourceBackend,
};

use crate::graph::ResourceInstanceId;

use super::{
    ForwardFrameBuffers, ForwardResidentBuffers, ForwardResidentUpload, ForwardTexture,
    ForwardUniformUpload, SurfaceForwardFrameBuffers,
};

/// One fully resolved Deferred geometry draw.
///
/// The Graph and recorder consume the actual ranges and binding owners retained
/// by the enclosing frame. This type grants no recording authority.
pub(crate) struct DeferredDraw<B: ResourceBackend> {
    /// Babylon's optional preliminary opposite-winding pass. It uses the same
    /// shader layout and bindings as `pipeline`, so the Graph footprint stays
    /// identical to the primary draw.
    pub(crate) preliminary_pipeline: Option<RasterPipeline<B>>,
    pub(crate) pipeline: RasterPipeline<B>,
    pub(crate) material_group: BindGroup<B>,
    pub(crate) transform_group: BindGroup<B>,
    pub(crate) vertex_layouts: Vec<VertexBufferLayout>,
    pub(crate) vertex_ranges: Vec<Range<u64>>,
    /// Indices into the frame-wide resident stream owner.
    pub(crate) vertex_streams: Vec<usize>,
    /// Index into the frame-wide resident texture owner.
    /// Resident texture slots in `binding_schema.textures` order.
    pub(crate) textures: Vec<usize>,
    /// Parallel logical binding facts: true selects the same-Graph capture product.
    pub(crate) capture_textures: Vec<bool>,
    pub(crate) vertices: Range<u32>,
    /// Physical instance range supplied to the raster draw and geometry
    /// footprint calculation.
    pub(crate) instances: Range<u32>,
    /// Dynamic material uniform ranges in binding order.
    pub(crate) material_uniform_ranges: Vec<Range<u64>>,
    /// Exact byte footprint for each dynamic material binding.
    pub(crate) material_uniform_sizes: Vec<u64>,
    pub(crate) transform_range: Range<u64>,
    /// Exact transform binding footprint selected by the material route.
    pub(crate) transform_bytes: u64,
}

impl<B: ResourceBackend> DeferredDraw<B> {
    pub(crate) fn pipelines_on(&self, device: DeviceId) -> bool {
        self.pipeline.device() == device
            && self
                .preliminary_pipeline
                .as_ref()
                .is_none_or(|pipeline| pipeline.device() == device)
    }
}

/// Exact immutable resident products eligible for completion-gated reuse.
#[derive(Default)]
pub(crate) struct DeferredPublication {
    pub(crate) geometry: Vec<(super::ForwardGeometryCacheKey, Vec<usize>)>,
    pub(crate) textures: Vec<(super::ForwardTextureCacheKey, usize)>,
}

/// Concrete resources retained through the Deferred submission's completion.
///
/// Resident geometry/material resources and frame upload ownership are exactly
/// the existing Forward owners. The three G-buffer textures are frame-local;
/// their descriptors, incoming state, instances, and views remain together in
/// [`ForwardTexture`].
pub(crate) struct DeferredFrame<B: ResourceBackend> {
    /// Selects the PBR G-buffer encoding and its wider base-color attachment.
    pub(crate) pbr: bool,
    pub(crate) publication: DeferredPublication,
    pub(crate) deformation: Option<super::DeferredDeformation<B>>,
    pub(crate) resident: Arc<ForwardResidentBuffers<B>>,
    pub(crate) resident_upload: ForwardResidentUpload,
    pub(crate) frame: ForwardFrameBuffers<B>,
    pub(crate) uniform_upload: ForwardUniformUpload,
    pub(crate) uniform_bytes: Vec<u8>,
    pub(crate) environment: Option<super::EnvironmentFrameOwner<B>>,
    /// Concrete per-frame specular resources and camera upload.
    pub(crate) specular: Option<super::SpecularEnvironmentFrameOwner<B>>,
    pub(crate) gbuffer: [ForwardTexture<B>; 3],
    pub(crate) effects: Option<ForwardTexture<B>>,
    /// Optional multisample G-buffer inputs and their resolve program.
    pub(crate) msaa: Option<DeferredMsaa<B>>,
    /// Frame-local lighting output. It is sampled by the composite pass before
    /// the final readback target is written.
    pub(crate) scene_color: ForwardTexture<B>,
    pub(crate) scene_color_sampler: fluxel_rhi::api::Sampler<B>,
    pub(crate) scene_color_sampler_desc: SamplerDesc,
    pub(crate) draws: Vec<DeferredDraw<B>>,
    pub(crate) lighting_pipeline: RasterPipeline<B>,
    pub(crate) lighting_group: BindGroup<B>,
    pub(crate) lighting_view: Option<std::ops::Range<u64>>,
    pub(crate) lighting_environment_group: Option<BindGroup<B>>,
    pub(crate) composite_pipeline: RasterPipeline<B>,
    pub(crate) composite_group: BindGroup<B>,
    pub(crate) extent: [u32; 2],
    pub(crate) readback_row_pitch: u32,
}

/// Concrete owners for one Deferred submission whose final lighting pass writes
/// to a transient surface lease. The lease itself stays in execution ownership;
/// it is intentionally absent here and cannot enter any reusable cache.
pub(crate) struct DeferredSurfaceFrame<B: ResourceBackend> {
    /// Selects the PBR G-buffer encoding and its wider base-color attachment.
    pub(crate) pbr: bool,
    pub(crate) publication: DeferredPublication,
    pub(crate) deformation: Option<super::DeferredDeformation<B>>,
    pub(crate) resident: Arc<ForwardResidentBuffers<B>>,
    pub(crate) resident_upload: ForwardResidentUpload,
    pub(crate) frame: SurfaceForwardFrameBuffers<B>,
    pub(crate) uniform_upload: ForwardUniformUpload,
    pub(crate) uniform_bytes: Vec<u8>,
    pub(crate) environment: Option<super::EnvironmentFrameOwner<B>>,
    /// Concrete per-frame specular resources and camera upload.
    pub(crate) specular: Option<super::SpecularEnvironmentFrameOwner<B>>,
    pub(crate) gbuffer: [ForwardTexture<B>; 3],
    pub(crate) effects: Option<ForwardTexture<B>>,
    /// Optional multisample G-buffer inputs and their resolve program.
    pub(crate) msaa: Option<DeferredMsaa<B>>,
    pub(crate) scene_color: ForwardTexture<B>,
    pub(crate) scene_color_sampler: fluxel_rhi::api::Sampler<B>,
    pub(crate) scene_color_sampler_desc: SamplerDesc,
    pub(crate) draws: Vec<DeferredDraw<B>>,
    pub(crate) lighting_pipeline: RasterPipeline<B>,
    pub(crate) lighting_group: BindGroup<B>,
    pub(crate) lighting_view: Option<std::ops::Range<u64>>,
    pub(crate) lighting_environment_group: Option<BindGroup<B>>,
    pub(crate) composite_pipeline: RasterPipeline<B>,
    pub(crate) composite_group: BindGroup<B>,
    pub(crate) extent: [u32; 2],
    pub(crate) color_format: TextureFormat,
}

/// Multisample G-buffer owners retained through the Deferred frame completion.
///
/// The ordinary frame G-buffer remains single-sample and is populated by this
/// resolve program before lighting. This owner has no alternate Graph path.
pub(crate) struct DeferredMsaa<B: ResourceBackend> {
    pub(crate) gbuffer: [ForwardTexture<B>; 3],
    pub(crate) effects: Option<ForwardTexture<B>>,
    pub(crate) pipeline: RasterPipeline<B>,
    pub(crate) group: BindGroup<B>,
}

/// Validated, handle-free G-buffer allocation facts for one device and extent.
///
/// The plan is deliberately private and carries no Graph resource identity.
/// Allocation rechecks the recorded device and exact descriptors before its
/// first backend allocation.
pub(crate) struct DeferredGbufferPlan {
    device: DeviceId,
    descs: [TextureDesc; 3],
    effects: Option<TextureDesc>,
}

impl DeferredGbufferPlan {
    pub(crate) fn allocate_with_effects<B: ResourceBackend>(
        self,
        device: &Device<B>,
    ) -> Result<([ForwardTexture<B>; 3], Option<ForwardTexture<B>>)> {
        if self.device != device.id() {
            return Err(Error::WrongDevice);
        }
        for desc in &self.descs {
            require_texture(device, desc)?;
        }
        if let Some(desc) = &self.effects {
            require_texture(device, desc)?;
        }
        let [albedo_desc, normal_desc, depth_desc] = self.descs;
        let gbuffer = [
            allocate_texture(device, albedo_desc, fluxel_rhi::api::TextureAspects::COLOR)?,
            allocate_texture(device, normal_desc, fluxel_rhi::api::TextureAspects::COLOR)?,
            allocate_texture(device, depth_desc, fluxel_rhi::api::TextureAspects::DEPTH)?,
        ];
        let effects = self
            .effects
            .map(|desc| allocate_texture(device, desc, fluxel_rhi::api::TextureAspects::COLOR))
            .transpose()?;
        Ok((gbuffer, effects))
    }
}

/// Validates the Deferred G-buffer resource request without allocating.
///
/// PBR uses a linear RGBA16F albedo attachment; unlit keeps RGBA8 compatibility.
/// PBR stores signed normals and roughness in RGBA32Float. Both use a single-sample
/// Depth32Float attachment. All attachments must also be
/// sampled by the later lighting pass, so support is checked for the exact
/// combined `RENDER_ATTACHMENT | TEXTURE_BINDING` descriptors now.
pub(crate) fn plan_deferred_gbuffer_with_effects<B: ResourceBackend>(
    device: &Device<B>,
    extent: [u32; 2],
    pbr: bool,
    effects: bool,
) -> Result<DeferredGbufferPlan> {
    plan_gbuffer(device, extent, pbr, effects, 1)
}

fn plan_gbuffer<B: ResourceBackend>(
    device: &Device<B>,
    extent: [u32; 2],
    pbr: bool,
    effects: bool,
    samples: u32,
) -> Result<DeferredGbufferPlan> {
    validate_request(
        extent,
        device.capabilities().limits.max_color_attachments,
        effects,
    )?;
    let descs = gbuffer_descs(extent, pbr, samples);
    for desc in &descs {
        require_texture(device, desc)?;
    }
    let effects = effects.then(|| {
        gbuffer_desc(
            "fluxel deferred surface effects",
            extent,
            TextureFormat::Rgba16Float,
            samples,
        )
    });
    if let Some(desc) = &effects {
        require_texture(device, desc)?;
    }
    Ok(DeferredGbufferPlan {
        device: device.id(),
        descs,
        effects,
    })
}

/// Plans separate multisample G-buffer attachments for a later explicit
/// resolve into the frame's single-sample G-buffer.
///
/// `None` preserves the existing single-sample path. Every admitted attachment
/// carries the exact render-attachment and multisample-texture-binding usage
/// needed by the resolve pass.
pub(crate) fn plan_deferred_msaa_with_effects<B: ResourceBackend>(
    device: &Device<B>,
    extent: [u32; 2],
    pbr: bool,
    effects: bool,
    samples: u32,
) -> Result<Option<DeferredGbufferPlan>> {
    match samples {
        0 => Err(Error::InvalidRange),
        1 => Ok(None),
        _ => plan_gbuffer(device, extent, pbr, effects, samples).map(Some),
    }
}

pub(crate) fn allocate_deferred_scene_color_desc<B: ResourceBackend>(
    device: &Device<B>,
    desc: TextureDesc,
) -> Result<ForwardTexture<B>> {
    let support = match device.texture_support(&desc) {
        Support::Supported(support) => support,
        Support::Unsupported(reason) => return Err(Error::Unsupported(reason)),
    };
    if !support.filterable {
        return Err(Error::Unsupported("filterable Deferred scene-color"));
    }
    allocate_texture(device, desc, fluxel_rhi::api::TextureAspects::COLOR)
}

/// Allocates the default linear Deferred lighting result for owner-validation
/// fixtures. Product paths select their format explicitly before allocation.
#[cfg(test)]
pub(crate) fn allocate_deferred_scene_color<B: ResourceBackend>(
    device: &Device<B>,
    extent: [u32; 2],
) -> Result<ForwardTexture<B>> {
    validate_request(
        extent,
        device.capabilities().limits.max_color_attachments,
        false,
    )?;
    allocate_deferred_scene_color_desc(
        device,
        deferred_scene_color_desc_with_format(extent, TextureFormat::Rgba8Unorm),
    )
}

/// The exact linear lighting descriptor selected for one Deferred submission.
/// The terminal remains a distinct LDR or Surface target, so this format never
/// changes the byte layout of an offscreen readback.
pub(crate) fn deferred_scene_color_desc_with_format(
    extent: [u32; 2],
    format: TextureFormat,
) -> TextureDesc {
    gbuffer_desc("fluxel deferred scene color", extent, format, 1)
}

fn validate_request(extent: [u32; 2], max_color_attachments: u32, effects: bool) -> Result<()> {
    if extent.contains(&0) {
        return Err(Error::InvalidRange);
    }
    if max_color_attachments < if effects { 3 } else { 2 } {
        return Err(Error::Unsupported(if effects {
            "deferred surface effects require three color attachments"
        } else {
            "deferred G-buffer requires two color attachments"
        }));
    }
    Ok(())
}

pub(crate) fn deferred_albedo_format(pbr: bool) -> TextureFormat {
    if pbr {
        TextureFormat::Rgba16Float
    } else {
        TextureFormat::Rgba8Unorm
    }
}

pub(crate) fn deferred_normal_format(pbr: bool) -> TextureFormat {
    if pbr {
        TextureFormat::Rgba32Float
    } else {
        TextureFormat::Rgba8Unorm
    }
}

fn gbuffer_descs(extent: [u32; 2], pbr: bool, samples: u32) -> [TextureDesc; 3] {
    [
        gbuffer_desc(
            "fluxel deferred albedo",
            extent,
            deferred_albedo_format(pbr),
            samples,
        ),
        gbuffer_desc(
            "fluxel deferred normal",
            extent,
            deferred_normal_format(pbr),
            samples,
        ),
        gbuffer_desc(
            "fluxel deferred depth",
            extent,
            TextureFormat::Depth32Float,
            samples,
        ),
    ]
}

fn gbuffer_desc(label: &str, extent: [u32; 2], format: TextureFormat, samples: u32) -> TextureDesc {
    TextureDesc {
        label: Some(label.into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format,
        mip_levels: 1,
        samples,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        view_formats: Vec::new(),
    }
}

fn require_texture<B: ResourceBackend>(device: &Device<B>, desc: &TextureDesc) -> Result<()> {
    match device.texture_support(desc) {
        Support::Supported(support)
            if desc.samples == 1 || support.sample_counts.contains(&desc.samples) =>
        {
            Ok(())
        }
        Support::Supported(_) => Err(Error::Unsupported(
            "Deferred multisample G-buffer attachment",
        )),
        Support::Unsupported(reason) => Err(Error::Unsupported(reason)),
    }
}

fn allocate_texture<B: ResourceBackend>(
    device: &Device<B>,
    desc: TextureDesc,
    aspects: fluxel_rhi::api::TextureAspects,
) -> Result<ForwardTexture<B>> {
    let filterable = match device.texture_support(&desc) {
        Support::Supported(support) => support.filterable,
        Support::Unsupported(reason) => return Err(Error::Unsupported(reason)),
    };
    let texture = device.create_texture(&desc)?;
    let view = device.create_view(
        &texture,
        &TextureViewDesc {
            format: desc.format,
            dimension: TextureViewDimension::D2,
            aspects,
            mips: 0..1,
            layers: 0..1,
        },
    )?;
    Ok(ForwardTexture {
        texture,
        view,
        view_dimension: TextureViewDimension::D2,
        desc,
        instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
        incoming: fluxel_rhi::api::TextureUses {
            stages: fluxel_rhi::api::StageMask::EMPTY,
            accesses: fluxel_rhi::api::AccessMask::EMPTY,
        },
        layout: ImageLayout::Undefined,
        device: device.id(),
        filterable,
    })
}

#[cfg(test)]
#[path = "../tests/residency/deferred.rs"]
mod tests;
