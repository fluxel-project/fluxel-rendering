//! Private resource owners for the first Deferred G-buffer consumer.
//!
//! This module owns only concrete texture allocation facts.  It does not
//! declare Graph accesses, select shaders, or record commands.

use std::{ops::Range, sync::Arc};

use fluxel_rhi::{
    api::{
        BindGroup, Device, DeviceId, Error, ImageLayout, RasterPipeline, Result, Support,
        TextureDesc, TextureDimension, TextureFormat, TextureUsage, TextureViewDesc,
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
    pub(crate) pipeline: RasterPipeline<B>,
    pub(crate) material_group: BindGroup<B>,
    pub(crate) transform_group: BindGroup<B>,
    pub(crate) vertex_layouts: Vec<VertexBufferLayout>,
    pub(crate) vertex_ranges: Vec<Range<u64>>,
    pub(crate) vertices: Range<u32>,
    pub(crate) tint_range: Range<u64>,
    pub(crate) transform_range: Range<u64>,
    /// Exact transform binding footprint selected by the material route.
    pub(crate) transform_bytes: u64,
}

/// Concrete resources retained through the Deferred submission's completion.
///
/// Resident geometry/material resources and frame upload ownership are exactly
/// the existing Forward owners. The three G-buffer textures are frame-local;
/// their descriptors, incoming state, instances, and views remain together in
/// [`ForwardTexture`].
pub(crate) struct DeferredFrame<B: ResourceBackend> {
    pub(crate) geometry_key: Option<super::ForwardGeometryCacheKey>,
    pub(crate) texture_key: Option<super::ForwardTextureCacheKey>,
    pub(crate) deformation: Option<super::DeferredDeformation<B>>,
    pub(crate) resident: Arc<ForwardResidentBuffers<B>>,
    pub(crate) resident_upload: ForwardResidentUpload,
    pub(crate) frame: ForwardFrameBuffers<B>,
    pub(crate) uniform_upload: ForwardUniformUpload,
    pub(crate) uniform_bytes: Vec<u8>,
    pub(crate) gbuffer: [ForwardTexture<B>; 3],
    pub(crate) draw: DeferredDraw<B>,
    pub(crate) lighting_pipeline: RasterPipeline<B>,
    pub(crate) lighting_group: BindGroup<B>,
    pub(crate) extent: [u32; 2],
    pub(crate) readback_row_pitch: u32,
}

/// Concrete owners for one Deferred submission whose final lighting pass writes
/// to a transient surface lease. The lease itself stays in execution ownership;
/// it is intentionally absent here and cannot enter any reusable cache.
pub(crate) struct DeferredSurfaceFrame<B: ResourceBackend> {
    pub(crate) geometry_key: Option<super::ForwardGeometryCacheKey>,
    pub(crate) texture_key: Option<super::ForwardTextureCacheKey>,
    pub(crate) deformation: Option<super::DeferredDeformation<B>>,
    pub(crate) resident: Arc<ForwardResidentBuffers<B>>,
    pub(crate) resident_upload: ForwardResidentUpload,
    pub(crate) frame: SurfaceForwardFrameBuffers<B>,
    pub(crate) uniform_upload: ForwardUniformUpload,
    pub(crate) uniform_bytes: Vec<u8>,
    pub(crate) gbuffer: [ForwardTexture<B>; 3],
    pub(crate) draw: DeferredDraw<B>,
    pub(crate) lighting_pipeline: RasterPipeline<B>,
    pub(crate) lighting_group: BindGroup<B>,
    pub(crate) extent: [u32; 2],
    pub(crate) color_format: TextureFormat,
}

/// Validated, handle-free G-buffer allocation facts for one device and extent.
///
/// The plan is deliberately private and carries no Graph resource identity.
/// Allocation rechecks the recorded device and exact descriptors before its
/// first backend allocation.
pub(crate) struct DeferredGbufferPlan {
    device: DeviceId,
    descs: [TextureDesc; 3],
}

impl DeferredGbufferPlan {
    /// Allocates the validated albedo, normal, and depth owners.
    pub(crate) fn allocate<B: ResourceBackend>(
        self,
        device: &Device<B>,
    ) -> Result<[ForwardTexture<B>; 3]> {
        if self.device != device.id() {
            return Err(Error::WrongDevice);
        }
        for desc in &self.descs {
            require_texture(device, desc)?;
        }
        let [albedo_desc, normal_desc, depth_desc] = self.descs;
        Ok([
            allocate_texture(device, albedo_desc, fluxel_rhi::api::TextureAspects::COLOR)?,
            allocate_texture(device, normal_desc, fluxel_rhi::api::TextureAspects::COLOR)?,
            allocate_texture(device, depth_desc, fluxel_rhi::api::TextureAspects::DEPTH)?,
        ])
    }
}

/// Validates the Deferred G-buffer resource request without allocating.
///
/// The resulting plan always has two single-sample RGBA8 color attachments and
/// one single-sample Depth32Float attachment. All attachments must also be
/// sampled by the later lighting pass, so support is checked for the exact
/// combined `RENDER_ATTACHMENT | TEXTURE_BINDING` descriptors now.
pub(crate) fn plan_deferred_gbuffer<B: ResourceBackend>(
    device: &Device<B>,
    extent: [u32; 2],
) -> Result<DeferredGbufferPlan> {
    validate_request(extent, device.capabilities().limits.max_color_attachments)?;
    let descs = gbuffer_descs(extent);
    for desc in &descs {
        require_texture(device, desc)?;
    }
    Ok(DeferredGbufferPlan {
        device: device.id(),
        descs,
    })
}

fn validate_request(extent: [u32; 2], max_color_attachments: u32) -> Result<()> {
    if extent.contains(&0) {
        return Err(Error::InvalidRange);
    }
    if max_color_attachments < 2 {
        return Err(Error::Unsupported(
            "deferred G-buffer requires two color attachments",
        ));
    }
    Ok(())
}

fn gbuffer_descs(extent: [u32; 2]) -> [TextureDesc; 3] {
    [
        gbuffer_desc("fluxel deferred albedo", extent, TextureFormat::Rgba8Unorm),
        gbuffer_desc("fluxel deferred normal", extent, TextureFormat::Rgba8Unorm),
        gbuffer_desc("fluxel deferred depth", extent, TextureFormat::Depth32Float),
    ]
}

fn gbuffer_desc(label: &str, extent: [u32; 2], format: TextureFormat) -> TextureDesc {
    TextureDesc {
        label: Some(label.into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        view_formats: Vec::new(),
    }
}

fn require_texture<B: ResourceBackend>(device: &Device<B>, desc: &TextureDesc) -> Result<()> {
    match device.texture_support(desc) {
        Support::Supported(_) => Ok(()),
        Support::Unsupported(reason) => Err(Error::Unsupported(reason)),
    }
}

fn allocate_texture<B: ResourceBackend>(
    device: &Device<B>,
    desc: TextureDesc,
    aspects: fluxel_rhi::api::TextureAspects,
) -> Result<ForwardTexture<B>> {
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
        desc,
        instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
        incoming: fluxel_rhi::api::TextureUses {
            stages: fluxel_rhi::api::StageMask::EMPTY,
            accesses: fluxel_rhi::api::AccessMask::EMPTY,
        },
        layout: ImageLayout::Undefined,
    })
}

#[cfg(test)]
#[path = "../tests/residency/deferred.rs"]
mod tests;
