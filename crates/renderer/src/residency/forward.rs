//! Device resource owners for the bounded Forward consumer.
//!
//! Allocation records concrete descriptors and incoming states alongside every
//! RHI handle. Graph validation consumes these owner facts rather than copying
//! expected declarations back into the resource owner.

use std::{ops::Range, sync::Arc};

use fluxel_rhi::{
    api::{
        Buffer, BufferDesc, BufferUsage, BufferUses, Device, DeviceId, Error, ImageLayout,
        MemoryClass, Result, Texture, TextureDesc, TextureDimension, TextureFormat, TextureUsage,
        TextureUses, TextureView, TextureViewDesc, TextureViewDimension, VertexBufferLayout,
    },
    backend::ResourceBackend,
};

use crate::graph::ResourceInstanceId;

#[path = "forward_support.rs"]
mod forward_support;
pub(crate) use forward_support::{ForwardFramePlan, SurfaceForwardFramePlan};
use forward_support::{
    empty_buffer_uses, empty_texture_uses, pad_to, require_buffer, require_texture,
};

/// Source bytes and declared layout for one complete vertex stream.
pub(crate) struct ForwardStreamInput {
    pub(crate) bytes: Vec<u8>,
    pub(crate) layout: VertexBufferLayout,
}

/// Source bytes and copied extent for the initial RGBA8 material texture.
pub(crate) struct ForwardTextureInput {
    pub(crate) bytes: Vec<u8>,
    pub(crate) extent: [u32; 2],
    pub(crate) bytes_per_row: u32,
}

/// One device-local vertex stream and its allocation facts.
pub(crate) struct ForwardVertexStream<B: ResourceBackend> {
    pub(crate) buffer: Buffer<B>,
    pub(crate) desc: BufferDesc,
    pub(crate) instance: ResourceInstanceId,
    pub(crate) incoming: BufferUses,
}

/// Device-local texture and its allocation facts.
pub(crate) struct ForwardTexture<B: ResourceBackend> {
    pub(crate) texture: Texture<B>,
    pub(crate) view: TextureView<B>,
    pub(crate) desc: TextureDesc,
    pub(crate) instance: ResourceInstanceId,
    pub(crate) incoming: TextureUses,
    pub(crate) layout: ImageLayout,
}

/// Persistent logical geometry/material realization for one device.
///
/// A renderer cache keys this owner by logical asset identity, content
/// generation and [`DeviceId`]. Submitted frames retain an [`Arc`] until the
/// completion that covers their final use.
pub(crate) struct ForwardResidentBuffers<B: ResourceBackend> {
    pub(crate) device: DeviceId,
    pub(crate) upload: Buffer<B>,
    pub(crate) upload_desc: BufferDesc,
    pub(crate) upload_instance: ResourceInstanceId,
    pub(crate) upload_incoming: BufferUses,
    pub(crate) streams: Vec<ForwardVertexStream<B>>,
    pub(crate) textures: Vec<ForwardTexture<B>>,
}

/// One source-to-device vertex copy, with physical byte offsets.
#[derive(Clone, Debug)]
pub(crate) struct ForwardVertexUpload {
    pub(crate) stream: usize,
    pub(crate) source: Range<u64>,
    pub(crate) destination: Range<u64>,
}

/// One source-to-device texture copy, with physical copy layout.
#[derive(Clone, Debug)]
pub(crate) struct ForwardTextureUpload {
    pub(crate) texture: usize,
    pub(crate) source_offset: u64,
    pub(crate) bytes_per_row: u32,
    pub(crate) rows_per_image: u32,
    pub(crate) extent: [u32; 3],
}

/// Exact upload map contents and declared copy ranges for a resident product.
pub(crate) struct ForwardResidentUpload {
    pub(crate) bytes: Vec<u8>,
    pub(crate) vertices: Vec<ForwardVertexUpload>,
    pub(crate) textures: Vec<ForwardTextureUpload>,
}

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
    /// Maps pipeline vertex-buffer slots to resident stream indices.
    pub(crate) vertex_streams: Vec<usize>,
    pub(crate) vertex_layouts: Vec<VertexBufferLayout>,
    pub(crate) vertex_ranges: Vec<Range<u64>>,
    pub(crate) vertices: Range<u32>,
    pub(crate) instances: Range<u32>,
    pub(crate) bind_groups: Vec<ForwardBindGroup<B>>,
    /// Index of the concrete material texture sampled by this draw.
    pub(crate) texture: usize,
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
    pub(crate) draws: Vec<ForwardDraw<B>>,
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
    pub(crate) draws: Vec<ForwardDraw<B>>,
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
    pub(crate) texture_descs: Vec<TextureDesc>,
    pub(crate) upload_desc: BufferDesc,
}

pub(crate) fn plan_forward_resident(
    streams: Vec<ForwardStreamInput>,
    textures: Vec<ForwardTextureInput>,
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
    for texture in &textures {
        if texture.extent.contains(&0)
            || texture.bytes_per_row == 0
            || texture.bytes_per_row % 256 != 0
            || texture.bytes_per_row
                < texture.extent[0]
                    .checked_mul(4)
                    .ok_or(Error::InvalidRange)?
        {
            return Err(Error::InvalidRange);
        }
        let texture_bytes = u64::from(texture.bytes_per_row)
            .checked_mul(u64::from(texture.extent[1]))
            .ok_or(Error::InvalidRange)?;
        if u64::try_from(texture.bytes.len()).map_err(|_| Error::InvalidRange)? != texture_bytes {
            return Err(Error::InvalidRange);
        }
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
    let mut texture_descs = Vec::with_capacity(textures.len());
    for (index, texture) in textures.into_iter().enumerate() {
        pad_to(&mut bytes, 512)?;
        let source_offset = u64::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?;
        bytes.extend_from_slice(&texture.bytes);
        texture_uploads.push(ForwardTextureUpload {
            texture: index,
            source_offset,
            bytes_per_row: texture.bytes_per_row,
            rows_per_image: texture.extent[1],
            extent: [texture.extent[0], texture.extent[1], 1],
        });
        texture_descs.push(TextureDesc {
            label: Some("fluxel forward material texture".into()),
            dimension: TextureDimension::D2,
            extent: [texture.extent[0], texture.extent[1], 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
            view_formats: Vec::new(),
        });
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
        texture_descs,
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
        let mut resident_textures = Vec::with_capacity(plan.texture_descs.len());
        for texture_desc in plan.texture_descs {
            require_texture(device, &texture_desc)?;
            let texture_handle = device.create_texture(&texture_desc)?;
            let view = device.create_view(
                &texture_handle,
                &TextureViewDesc {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2,
                    aspects: fluxel_rhi::api::TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
            )?;
            resident_textures.push(ForwardTexture {
                texture: texture_handle,
                view,
                desc: texture_desc,
                instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
                incoming: empty_texture_uses(),
                layout: ImageLayout::Undefined,
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

impl<B: ResourceBackend> ForwardFrameBuffers<B> {
    pub(crate) fn allocate_planned(
        device: &Device<B>,
        plan: ForwardFramePlan,
    ) -> Result<(Self, ForwardUniformUpload)> {
        for desc in [&plan.upload_desc, &plan.uniform_desc, &plan.readback_desc] {
            require_buffer(device, desc)?;
        }
        require_texture(device, &plan.target_desc)?;
        let upload = device.create_buffer(&plan.upload_desc)?;
        let uniform = device.create_buffer(&plan.uniform_desc)?;
        let target = device.create_texture(&plan.target_desc)?;
        let target_view = device.create_view(
            &target,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: fluxel_rhi::api::TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        let readback = device.create_buffer(&plan.readback_desc)?;
        Ok((
            Self {
                upload,
                upload_desc: plan.upload_desc,
                upload_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
                upload_incoming: empty_buffer_uses(),
                uniform,
                uniform_desc: plan.uniform_desc,
                uniform_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
                uniform_incoming: empty_buffer_uses(),
                target,
                target_desc: plan.target_desc,
                target_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
                target_view,
                target_incoming: empty_texture_uses(),
                target_layout: ImageLayout::Undefined,
                readback,
                readback_desc: plan.readback_desc,
                readback_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
                readback_incoming: empty_buffer_uses(),
            },
            plan.uniform_upload,
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
    let target_desc = TextureDesc {
        label: Some("fluxel forward target".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
        view_formats: Vec::new(),
    };
    Ok(ForwardFramePlan {
        upload_desc,
        uniform_desc,
        target_desc,
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
