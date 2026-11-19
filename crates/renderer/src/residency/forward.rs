//! Device resource owners for the bounded Forward consumer.
//!
//! Allocation records concrete descriptors and incoming states alongside every
//! RHI handle. Graph validation consumes these owner facts rather than copying
//! expected declarations back into the resource owner.

use std::{ops::Range, sync::Arc};

use fluxel_rhi::{
    api::{
        Buffer, BufferDesc, BufferUsage, BufferUses, Device, DeviceId, Error, ImageLayout,
        MemoryClass, Result, Sampler, SamplerDesc, Support, Texture, TextureDesc, TextureDimension,
        TextureFormat, TextureUsage, TextureUses, TextureView, TextureViewDesc,
        TextureViewDimension, VertexBufferLayout,
    },
    backend::ResourceBackend,
};

use crate::graph::ResourceInstanceId;

#[path = "forward_depth.rs"]
mod forward_depth;
pub(crate) use forward_depth::ForwardDepthAttachment;
#[path = "forward_frame_allocation.rs"]
mod forward_frame_allocation;
pub(crate) use forward_frame_allocation::plan_forward_hdr_frame;

#[path = "forward_support.rs"]
mod forward_support;
use forward_support::{
    empty_buffer_uses, empty_texture_uses, pad_to, require_buffer, require_texture,
};
#[path = "forward_texture_plan.rs"]
mod forward_texture_plan;
pub(crate) use forward_support::{ForwardFramePlan, SurfaceForwardFramePlan};
use forward_texture_plan::plan_texture_uploads;
#[path = "forward_resources.rs"]
mod forward_resources;
pub(crate) use forward_resources::*;

/// Per-frame uniform/attachment/readback owners.
pub(crate) struct ForwardFrameBuffers<B: ResourceBackend> {
    pub(crate) upload: Buffer<B>,
    pub(crate) upload_desc: BufferDesc,
    pub(crate) upload_instance: ResourceInstanceId,
    pub(crate) upload_incoming: BufferUses,
    pub(crate) uniform: Buffer<B>,
    pub(crate) uniform_desc: BufferDesc,
    pub(crate) uniform_instance: ResourceInstanceId,
    pub(crate) uniform_incoming: BufferUses,
    pub(crate) target: Texture<B>,
    pub(crate) target_desc: TextureDesc,
    pub(crate) target_instance: ResourceInstanceId,
    pub(crate) target_view: TextureView<B>,
    pub(crate) target_incoming: TextureUses,
    pub(crate) target_layout: ImageLayout,
    /// Display-referred terminal target for an HDR scene-color frame.
    /// Ordinary Forward frames retain no second color owner.
    pub(crate) output: Option<ForwardTexture<B>>,
    /// Raster color owner when the single-sample target receives a resolve.
    pub(crate) multisample_color: Option<ForwardTexture<B>>,
    pub(crate) depth: Option<ForwardDepthAttachment<B>>,
    pub(crate) readback: Buffer<B>,
    pub(crate) readback_desc: BufferDesc,
    pub(crate) readback_instance: ResourceInstanceId,
    pub(crate) readback_incoming: BufferUses,
}

/// Per-frame owners for direct surface Forward rendering. Surface images are
/// transient RHI leases and therefore deliberately do not appear here.
pub(crate) struct SurfaceForwardFrameBuffers<B: ResourceBackend> {
    pub(crate) upload: Buffer<B>,
    pub(crate) upload_desc: BufferDesc,
    pub(crate) upload_instance: ResourceInstanceId,
    pub(crate) upload_incoming: BufferUses,
    pub(crate) uniform: Buffer<B>,
    pub(crate) uniform_desc: BufferDesc,
    pub(crate) uniform_instance: ResourceInstanceId,
    pub(crate) uniform_incoming: BufferUses,
    pub(crate) depth: Option<ForwardDepthAttachment<B>>,
}

/// One source-to-device dynamic-uniform copy.
#[derive(Clone, Debug)]
pub(crate) struct ForwardUniformUpload {
    pub(crate) source: Range<u64>,
    pub(crate) destination: Range<u64>,
}

/// One group selected for a draw and its physical dynamic offsets.
pub(crate) struct ForwardBindGroup<B: ResourceBackend> {
    pub(crate) slot: u32,
    pub(crate) group: fluxel_rhi::api::BindGroup<B>,
    pub(crate) dynamic_offsets: Vec<u32>,
    /// Concrete uniform ranges whose dynamic offsets are consumed by this group.
    pub(crate) uniform_ranges: Vec<Range<u64>>,
}

/// A prepared non-indexed opaque draw.
pub(crate) struct ForwardDraw<B: ResourceBackend> {
    /// The contiguous compatibility batch that owns this draw.
    pub(crate) batch: usize,
    /// Complete resolved program facts selected during preparation.  The
    /// Graph verifies every draw in a batch shares this key before eliding a
    /// pipeline bind.
    pub(crate) compatibility: crate::preparation::ForwardCompatibilityKey,
    pub(crate) pipeline: fluxel_rhi::api::RasterPipeline<B>,
    /// Optional first pass with opposite winding and back-face culling.
    /// It is recorded immediately before the ordinary material draw.
    pub(crate) preliminary_pipeline: Option<fluxel_rhi::api::RasterPipeline<B>>,
    /// Maps pipeline vertex-buffer slots to resident stream indices.
    pub(crate) vertex_streams: Vec<usize>,
    pub(crate) vertex_layouts: Vec<VertexBufferLayout>,
    pub(crate) vertex_ranges: Vec<Range<u64>>,
    pub(crate) vertices: Range<u32>,
    pub(crate) instances: Range<u32>,
    pub(crate) bind_groups: Vec<ForwardBindGroup<B>>,
    /// Index of the concrete material texture sampled by this draw.
    pub(crate) textures: Vec<usize>,
    /// True exactly where the logical material texture is the same-submission
    /// capture product rather than an immutable asset owner.
    pub(crate) capture_textures: Vec<bool>,
}

impl<B: ResourceBackend> ForwardDraw<B> {
    pub(crate) fn pipelines_on(&self, device: DeviceId) -> bool {
        self.pipeline.device() == device
            && self
                .preliminary_pipeline
                .as_ref()
                .is_none_or(|pipeline| pipeline.device() == device)
    }
}

/// Validates the one-to-one indirect-product ownership carried by a submitted
/// Forward frame.
pub(crate) fn validate_indirect_draws<B: ResourceBackend>(
    draws: &[ForwardDraw<B>],
    indirect: &[(usize, crate::renderer::indirect_producer::IndirectProduct<B>)],
) -> Result<()> {
    let mut selected = std::collections::BTreeSet::new();
    for (draw_index, product) in indirect {
        let draw = draws.get(*draw_index).ok_or(Error::InvalidRange)?;
        if !selected.insert(*draw_index) {
            return Err(Error::InvalidRange);
        }
        product.validate_draw(&draw.vertices, &draw.instances)?;
    }
    Ok(())
}

/// Returns the unique GPU command owner for one prepared draw.
pub(crate) fn indirect_for_draw<'a, B: ResourceBackend>(
    indirect: &'a [(usize, crate::renderer::indirect_producer::IndirectProduct<B>)],
    draw_index: usize,
) -> Option<&'a crate::renderer::indirect_producer::IndirectProduct<B>> {
    indirect
        .iter()
        .find_map(|(index, product)| (*index == draw_index).then_some(product))
}

/// Complete concrete owners and uploads admitted for one Forward submission.
///
/// The coordinator keeps the resident `Arc` in its cache and moves this frame
/// into pending completion ownership after recording succeeds or has unknown
/// native effects.
pub(crate) struct ForwardFrame<B: ResourceBackend> {
    pub(crate) resident: Arc<ForwardResidentBuffers<B>>,
    pub(crate) resident_upload: ForwardResidentUpload,
    pub(crate) frame: ForwardFrameBuffers<B>,
    pub(crate) uniform_upload: ForwardUniformUpload,
    pub(crate) uniform_bytes: Vec<u8>,
    /// Optional scene-selected diffuse irradiance owner. This is deliberately
    /// frame-local: selection may change while earlier submissions are pending.
    pub(crate) environment: Option<super::EnvironmentFrameOwner<B>>,
    /// Concrete per-frame specular resources and camera upload.
    pub(crate) specular: Option<super::SpecularEnvironmentFrameOwner<B>>,
    pub(crate) draws: Vec<ForwardDraw<B>>,
    /// GPU-produced indirect argument records keyed by their prepared draw.
    ///
    /// The product stays with this submitted frame through completion. Graph
    /// declaration and recording use the same concrete owner.
    pub(crate) indirect: Vec<(usize, crate::renderer::indirect_producer::IndirectProduct<B>)>,
    /// Contiguous draw ranges that share one raster program without changing
    /// the prepared candidate order.
    pub(crate) draw_batches: Vec<Range<usize>>,
    pub(crate) extent: [u32; 2],
    pub(crate) readback_row_pitch: u32,
}

/// Concrete owners admitted for one surface Forward submission.
pub(crate) struct SurfaceForwardFrame<B: ResourceBackend> {
    pub(crate) resident: Arc<ForwardResidentBuffers<B>>,
    pub(crate) resident_upload: ForwardResidentUpload,
    pub(crate) frame: SurfaceForwardFrameBuffers<B>,
    pub(crate) uniform_upload: ForwardUniformUpload,
    pub(crate) uniform_bytes: Vec<u8>,
    /// Optional scene-selected diffuse irradiance owner retained until the
    /// submission's presentation completion covers its final use.
    pub(crate) environment: Option<super::EnvironmentFrameOwner<B>>,
    /// Concrete per-frame specular resources and camera upload.
    pub(crate) specular: Option<super::SpecularEnvironmentFrameOwner<B>>,
    pub(crate) draws: Vec<ForwardDraw<B>>,
    /// GPU-produced indirect argument records keyed by their prepared draw.
    pub(crate) indirect: Vec<(usize, crate::renderer::indirect_producer::IndirectProduct<B>)>,
    pub(crate) draw_batches: Vec<Range<usize>>,
    pub(crate) extent: [u32; 2],
    /// The color target format used to create every pipeline in this frame.
    pub(crate) color_format: TextureFormat,
}

/// Validated, handle-free allocation facts for one resident Forward product.
///
/// This stays private to residency: it is not a Graph declaration or a public
/// resource model. Keeping it separate makes all descriptor and copy-layout
/// failures observable before the first backend allocation.
pub(crate) struct ForwardResidentPlan {
    pub(crate) bytes: Vec<u8>,
    pub(crate) vertices: Vec<ForwardVertexUpload>,
    pub(crate) textures: Vec<ForwardTextureUpload>,
    pub(crate) stream_descs: Vec<BufferDesc>,
    pub(crate) texture_allocations: Vec<ForwardTextureAllocation>,
    pub(crate) upload_desc: BufferDesc,
}

#[cfg(test)]
pub(crate) fn plan_forward_resident(
    streams: Vec<ForwardStreamInput>,
    textures: Vec<ForwardTextureInput>,
) -> Result<ForwardResidentPlan> {
    plan_forward_resident_with_texture_inputs(
        streams,
        textures
            .into_iter()
            .map(ForwardResidentTextureInput::from)
            .collect(),
    )
}

/// Plans one resident product with ordinary material textures and complete
/// authored `RGBA16Float` texture chains.
pub(crate) fn plan_forward_resident_with_texture_inputs(
    streams: Vec<ForwardStreamInput>,
    textures: Vec<ForwardResidentTextureInput>,
) -> Result<ForwardResidentPlan> {
    // Complete input validation precedes the first backend allocation. The
    // vertex source may end before its padded stride; only its physical byte
    // allocation and prepared draw footprint establish later bounds.
    if streams
        .iter()
        .any(|stream| stream.layout.stride == 0 || stream.bytes.is_empty())
    {
        return Err(Error::InvalidRange);
    }
    let mut bytes = Vec::new();
    let mut vertices = Vec::with_capacity(streams.len());
    let mut stream_descs = Vec::with_capacity(streams.len());
    for (index, stream) in streams.into_iter().enumerate() {
        pad_to(&mut bytes, 4)?;
        let source_start = u64::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?;
        bytes.extend_from_slice(&stream.bytes);
        let source_end = u64::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?;
        let desc = BufferDesc {
            label: Some("fluxel forward vertex stream".into()),
            size: source_end - source_start,
            usage: BufferUsage::COPY_DST | BufferUsage::VERTEX,
            memory: MemoryClass::Device,
        };
        vertices.push(ForwardVertexUpload {
            stream: index,
            source: source_start..source_end,
            destination: 0..desc.size,
        });
        stream_descs.push(desc);
    }
    let mut texture_uploads = Vec::with_capacity(textures.len());
    let mut texture_allocations = Vec::with_capacity(textures.len());
    for (index, texture) in textures.into_iter().enumerate() {
        texture_allocations.push(plan_texture_uploads(
            &mut bytes,
            &mut texture_uploads,
            index,
            texture,
        )?);
    }
    let upload_desc = BufferDesc {
        label: Some("fluxel forward resident upload".into()),
        size: u64::try_from(bytes.len().max(16)).map_err(|_| Error::InvalidRange)?,
        usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
        memory: MemoryClass::Upload,
    };
    bytes.resize(
        usize::try_from(upload_desc.size).map_err(|_| Error::InvalidRange)?,
        0,
    );
    Ok(ForwardResidentPlan {
        bytes,
        vertices,
        textures: texture_uploads,
        stream_descs,
        texture_allocations,
        upload_desc,
    })
}

impl<B: ResourceBackend> ForwardResidentBuffers<B> {
    /// Allocates exact owners from a previously validated handle-free plan.
    pub(crate) fn allocate_planned(
        device: &Device<B>,
        plan: ForwardResidentPlan,
    ) -> Result<(Arc<Self>, ForwardResidentUpload)> {
        let mut resident_streams = Vec::with_capacity(plan.stream_descs.len());
        for desc in plan.stream_descs {
            require_buffer(device, &desc)?;
            let buffer = device.create_buffer(&desc)?;
            let instance = ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?;
            resident_streams.push(ForwardVertexStream {
                buffer,
                desc,
                instance,
                incoming: empty_buffer_uses(),
            });
        }
        let mut resident_textures = Vec::with_capacity(plan.texture_allocations.len());
        for allocation in plan.texture_allocations {
            let texture_desc = allocation.desc;
            let filterable = match device.texture_support(&texture_desc) {
                Support::Supported(support) if texture_desc.mip_levels <= support.max_mips => {
                    support.filterable
                }
                Support::Supported(_) => {
                    return Err(Error::Unsupported("material texture mip levels"));
                }
                Support::Unsupported(reason) => return Err(Error::Unsupported(reason)),
            };
            let texture_handle = device.create_texture(&texture_desc)?;
            let view = device.create_view(
                &texture_handle,
                &TextureViewDesc {
                    format: texture_desc.format,
                    dimension: allocation.view_dimension,
                    aspects: if texture_desc.format == TextureFormat::Depth32Float {
                        fluxel_rhi::api::TextureAspects::DEPTH
                    } else {
                        fluxel_rhi::api::TextureAspects::COLOR
                    },
                    mips: 0..texture_desc.mip_levels,
                    layers: 0..texture_desc.array_layers,
                },
            )?;
            resident_textures.push(ForwardTexture {
                texture: texture_handle,
                view,
                view_dimension: allocation.view_dimension,
                desc: texture_desc,
                instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
                incoming: empty_texture_uses(),
                layout: ImageLayout::Undefined,
                device: device.id(),
                filterable,
            });
        }
        let upload_desc = plan.upload_desc;
        require_buffer(device, &upload_desc)?;
        let upload = device.create_buffer(&upload_desc)?;
        let resident = Arc::new(Self {
            device: device.id(),
            upload,
            upload_desc,
            upload_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
            upload_incoming: empty_buffer_uses(),
            streams: resident_streams,
            textures: resident_textures,
            samplers: Vec::new(),
        });
        Ok((
            resident,
            ForwardResidentUpload {
                bytes: plan.bytes,
                vertices: plan.vertices,
                textures: plan.textures,
            },
        ))
    }
}

pub(crate) fn plan_forward_frame(
    extent: [u32; 2],
    uniform_capacity: u64,
    uniform_len: u64,
    uniform_range: Range<u64>,
    row_pitch: u32,
    alignment: u64,
    sampled_target: bool,
) -> Result<ForwardFramePlan> {
    if extent.contains(&0)
        || uniform_len == 0
        || uniform_range.start >= uniform_range.end
        || uniform_range.end > uniform_capacity
        || row_pitch == 0
        || !row_pitch.is_multiple_of(256)
        || u64::from(row_pitch) < u64::from(extent[0]) * 4
        || uniform_range.end - uniform_range.start != uniform_len
        || alignment == 0
        || !uniform_range.start.is_multiple_of(alignment)
    {
        return Err(Error::InvalidRange);
    }
    let upload_desc = BufferDesc {
        label: Some("fluxel forward frame upload".into()),
        size: uniform_len,
        usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
        memory: MemoryClass::Upload,
    };
    let uniform_desc = BufferDesc {
        label: Some("fluxel forward frame uniform".into()),
        size: uniform_capacity,
        usage: BufferUsage::COPY_DST | BufferUsage::UNIFORM,
        memory: MemoryClass::Device,
    };
    let readback_size = u64::from(row_pitch)
        .checked_mul(u64::from(extent[1]))
        .ok_or(Error::InvalidRange)?;
    let readback_desc = BufferDesc {
        label: Some("fluxel forward readback".into()),
        size: readback_size,
        usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
        memory: MemoryClass::Readback,
    };
    let mut target_usage = TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC;
    if sampled_target {
        target_usage |= TextureUsage::TEXTURE_BINDING;
    }
    let target_desc = TextureDesc {
        label: Some("fluxel forward target".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: target_usage,
        view_formats: Vec::new(),
    };
    Ok(ForwardFramePlan {
        upload_desc,
        uniform_desc,
        target_desc,
        output_desc: None,
        readback_desc,
        uniform_upload: ForwardUniformUpload {
            source: 0..uniform_len,
            destination: uniform_range,
        },
    })
}

impl<B: ResourceBackend> SurfaceForwardFrameBuffers<B> {
    pub(crate) fn allocate_planned(
        device: &Device<B>,
        plan: SurfaceForwardFramePlan,
    ) -> Result<(Self, ForwardUniformUpload)> {
        require_buffer(device, &plan.upload_desc)?;
        require_buffer(device, &plan.uniform_desc)?;
        Ok((
            Self {
                upload: device.create_buffer(&plan.upload_desc)?,
                upload_desc: plan.upload_desc,
                upload_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
                upload_incoming: empty_buffer_uses(),
                uniform: device.create_buffer(&plan.uniform_desc)?,
                uniform_desc: plan.uniform_desc,
                uniform_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
                uniform_incoming: empty_buffer_uses(),
                depth: None,
            },
            plan.uniform_upload,
        ))
    }
}

pub(crate) fn plan_surface_forward_frame(
    uniform_capacity: u64,
    uniform_len: u64,
    uniform_range: Range<u64>,
    alignment: u64,
) -> Result<SurfaceForwardFramePlan> {
    if uniform_len == 0
        || uniform_range.start >= uniform_range.end
        || uniform_range.end > uniform_capacity
        || uniform_range.end - uniform_range.start != uniform_len
        || alignment == 0
        || !uniform_range.start.is_multiple_of(alignment)
    {
        return Err(Error::InvalidRange);
    }
    Ok(SurfaceForwardFramePlan {
        upload_desc: BufferDesc {
            label: Some("fluxel forward surface frame upload".into()),
            size: uniform_len,
            usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
            memory: MemoryClass::Upload,
        },
        uniform_desc: BufferDesc {
            label: Some("fluxel forward surface frame uniform".into()),
            size: uniform_capacity,
            usage: BufferUsage::COPY_DST | BufferUsage::UNIFORM,
            memory: MemoryClass::Device,
        },
        uniform_upload: ForwardUniformUpload {
            source: 0..uniform_len,
            destination: uniform_range,
        },
    })
}

#[cfg(test)]
#[path = "../tests/residency/forward.rs"]
mod test_helpers;
