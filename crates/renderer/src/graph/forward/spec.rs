//! Owner-derived facts and recording shapes for Forward declarations.
use super::*;

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct ForwardShape {
    pub(crate) extent: [u32; 2],
    pub(crate) vertex_uploads: Vec<(usize, std::ops::Range<u64>, std::ops::Range<u64>)>,
    pub(crate) texture_uploads: Vec<(usize, u64, u32, u32, [u32; 3])>,
    pub(crate) uniform_upload: (std::ops::Range<u64>, std::ops::Range<u64>),
    /// Batch boundaries affect pipeline binding and therefore are part of the
    /// compiled recording shape even though the batches contain no resources.
    pub(crate) batches: Vec<std::ops::Range<usize>>,
    pub(crate) draws: Vec<ForwardDrawShape>,
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct ForwardDrawShape {
    pub(crate) compatibility: crate::preparation::ForwardCompatibilityKey,
    pub(crate) streams: Vec<usize>,
    pub(crate) layouts: Vec<crate::preparation::VertexLayoutKey>,
    pub(crate) ranges: Vec<std::ops::Range<u64>>,
    pub(crate) vertices: std::ops::Range<u32>,
    pub(crate) instances: std::ops::Range<u32>,
    pub(crate) texture: usize,
    pub(crate) groups: Vec<(u32, Vec<u32>, Vec<std::ops::Range<u64>>)>,
}

/// Handle-free facts consumed while declaring the offscreen Forward graph.
/// This remains a private compiled-product input, not a second public graph
/// language; concrete owners are checked separately before binding.
#[derive(Clone)]
pub(crate) struct ForwardGraphSpec {
    pub(crate) device: DeviceId,
    pub(crate) extent: [u32; 2],
    pub(crate) readback_row_pitch: u32,
    pub(crate) resident_upload: BufferGraphFact,
    pub(crate) frame_upload: BufferGraphFact,
    pub(crate) uniform: BufferGraphFact,
    pub(crate) streams: Vec<BufferGraphFact>,
    pub(crate) textures: Vec<TextureGraphFact>,
    pub(crate) target: TextureGraphFact,
    pub(crate) readback: BufferGraphFact,
    pub(crate) vertex_uploads: Vec<ForwardGraphVertexUpload>,
    pub(crate) texture_uploads: Vec<ForwardGraphTextureUpload>,
    pub(crate) uniform_upload: ForwardGraphUniformUpload,
    pub(crate) draws: Vec<ForwardGraphDraw>,
    pub(crate) shape: ForwardShape,
}

#[derive(Clone)]
pub(crate) struct BufferGraphFact {
    pub(crate) desc: BufferDesc,
    pub(crate) incoming: BufferUses,
}

#[derive(Clone)]
pub(crate) struct TextureGraphFact {
    pub(crate) desc: TextureDesc,
    pub(crate) incoming: TextureUses,
    pub(crate) layout: ImageLayout,
}

#[derive(Clone)]
pub(crate) struct ForwardGraphDraw {
    pub(crate) vertex_streams: Vec<usize>,
    pub(crate) vertex_layouts: Vec<fluxel_rhi::api::VertexBufferLayout>,
    pub(crate) vertex_ranges: Vec<Range<u64>>,
    pub(crate) vertices: Range<u32>,
    pub(crate) instances: Range<u32>,
    pub(crate) uniform_ranges: Vec<Range<u64>>,
    pub(crate) texture: usize,
    pub(crate) accesses: Vec<Option<Range<u64>>>,
}

#[derive(Clone)]
pub(crate) struct ForwardGraphVertexUpload {
    pub(crate) stream: usize,
    pub(crate) source: Range<u64>,
    pub(crate) destination: Range<u64>,
}

#[derive(Clone)]
pub(crate) struct ForwardGraphTextureUpload {
    pub(crate) texture: usize,
    pub(crate) source_offset: u64,
    pub(crate) bytes_per_row: u32,
    pub(crate) rows_per_image: u32,
}

#[derive(Clone)]
pub(crate) struct ForwardGraphUniformUpload {
    pub(crate) source: Range<u64>,
    pub(crate) destination: Range<u64>,
}

impl ForwardGraphSpec {
    /// Captures descriptor and access facts without retaining a backend handle.
    pub(crate) fn from_frame<B: ResourceBackend>(frame: &ForwardFrame<B>) -> Result<Self> {
        let streams = frame
            .resident
            .streams
            .iter()
            .map(|stream| BufferGraphFact {
                desc: stream.desc.clone(),
                incoming: stream.incoming,
            })
            .collect();
        let textures = frame
            .resident
            .textures
            .iter()
            .map(|texture| TextureGraphFact {
                desc: texture.desc.clone(),
                incoming: texture.incoming,
                layout: texture.layout,
            })
            .collect();
        let draws = frame
            .draws
            .iter()
            .map(|draw| {
                Ok(ForwardGraphDraw {
                    vertex_streams: draw.vertex_streams.clone(),
                    vertex_layouts: draw.vertex_layouts.clone(),
                    vertex_ranges: draw.vertex_ranges.clone(),
                    vertices: draw.vertices.clone(),
                    instances: draw.instances.clone(),
                    uniform_ranges: draw
                        .bind_groups
                        .iter()
                        .flat_map(|group| group.uniform_ranges.iter().cloned())
                        .collect(),
                    texture: draw.texture,
                    accesses: draw_accesses(frame, draw)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            device: frame.resident.device,
            extent: frame.extent,
            readback_row_pitch: frame.readback_row_pitch,
            resident_upload: BufferGraphFact {
                desc: frame.resident.upload_desc.clone(),
                incoming: frame.resident.upload_incoming,
            },
            frame_upload: BufferGraphFact {
                desc: frame.frame.upload_desc.clone(),
                incoming: frame.frame.upload_incoming,
            },
            uniform: BufferGraphFact {
                desc: frame.frame.uniform_desc.clone(),
                incoming: frame.frame.uniform_incoming,
            },
            streams,
            textures,
            target: TextureGraphFact {
                desc: frame.frame.target_desc.clone(),
                incoming: frame.frame.target_incoming,
                layout: frame.frame.target_layout,
            },
            readback: BufferGraphFact {
                desc: frame.frame.readback_desc.clone(),
                incoming: frame.frame.readback_incoming,
            },
            vertex_uploads: frame
                .resident_upload
                .vertices
                .iter()
                .map(|upload| ForwardGraphVertexUpload {
                    stream: upload.stream,
                    source: upload.source.clone(),
                    destination: upload.destination.clone(),
                })
                .collect(),
            texture_uploads: frame
                .resident_upload
                .textures
                .iter()
                .map(|upload| ForwardGraphTextureUpload {
                    texture: upload.texture,
                    source_offset: upload.source_offset,
                    bytes_per_row: upload.bytes_per_row,
                    rows_per_image: upload.rows_per_image,
                })
                .collect(),
            uniform_upload: ForwardGraphUniformUpload {
                source: frame.uniform_upload.source.clone(),
                destination: frame.uniform_upload.destination.clone(),
            },
            draws,
            shape: forward_shape(frame),
        })
    }
}

/// Handle-free recording facts for the direct-surface product. Surface leases
/// are transient, but every persistent input and batch boundary still affects
/// the compiled Graph and must therefore participate in reuse validation.
#[derive(PartialEq, Eq)]
pub(super) struct SurfaceForwardShape {
    extent: [u32; 2],
    vertex_uploads: Vec<(usize, std::ops::Range<u64>, std::ops::Range<u64>)>,
    texture_uploads: Vec<(usize, u64, u32, u32, [u32; 3])>,
    uniform_upload: (std::ops::Range<u64>, std::ops::Range<u64>),
    batches: Vec<std::ops::Range<usize>>,
    draws: Vec<ForwardDrawShape>,
}

/// A handle-free contract snapshot used for plan reuse. It deliberately keeps
/// allocation identities out of the comparison; [`FrameBindings`] validates
/// each new owner separately against this plan's declared resource facts.
pub(super) fn forward_shape<B: ResourceBackend>(frame: &ForwardFrame<B>) -> ForwardShape {
    ForwardShape {
        extent: frame.extent,
        vertex_uploads: frame
            .resident_upload
            .vertices
            .iter()
            .map(|upload| {
                (
                    upload.stream,
                    upload.source.clone(),
                    upload.destination.clone(),
                )
            })
            .collect(),
        texture_uploads: frame
            .resident_upload
            .textures
            .iter()
            .map(|upload| {
                (
                    upload.texture,
                    upload.source_offset,
                    upload.bytes_per_row,
                    upload.rows_per_image,
                    upload.extent,
                )
            })
            .collect(),
        uniform_upload: (
            frame.uniform_upload.source.clone(),
            frame.uniform_upload.destination.clone(),
        ),
        batches: frame.draw_batches.clone(),
        draws: frame
            .draws
            .iter()
            .map(|draw| ForwardDrawShape {
                compatibility: draw.compatibility.clone(),
                streams: draw.vertex_streams.clone(),
                layouts: draw
                    .vertex_layouts
                    .iter()
                    .map(crate::preparation::vertex_layout_key)
                    .collect(),
                ranges: draw.vertex_ranges.clone(),
                vertices: draw.vertices.clone(),
                instances: draw.instances.clone(),
                texture: draw.texture,
                groups: draw
                    .bind_groups
                    .iter()
                    .map(|group| {
                        (
                            group.slot,
                            group.dynamic_offsets.clone(),
                            group.uniform_ranges.clone(),
                        )
                    })
                    .collect(),
            })
            .collect(),
    }
}

pub(super) fn surface_forward_shape<B: ResourceBackend>(
    frame: &SurfaceForwardFrame<B>,
) -> SurfaceForwardShape {
    SurfaceForwardShape {
        extent: frame.extent,
        vertex_uploads: frame
            .resident_upload
            .vertices
            .iter()
            .map(|upload| {
                (
                    upload.stream,
                    upload.source.clone(),
                    upload.destination.clone(),
                )
            })
            .collect(),
        texture_uploads: frame
            .resident_upload
            .textures
            .iter()
            .map(|upload| {
                (
                    upload.texture,
                    upload.source_offset,
                    upload.bytes_per_row,
                    upload.rows_per_image,
                    upload.extent,
                )
            })
            .collect(),
        uniform_upload: (
            frame.uniform_upload.source.clone(),
            frame.uniform_upload.destination.clone(),
        ),
        batches: frame.draw_batches.clone(),
        draws: frame
            .draws
            .iter()
            .map(|draw| ForwardDrawShape {
                compatibility: draw.compatibility.clone(),
                streams: draw.vertex_streams.clone(),
                layouts: draw
                    .vertex_layouts
                    .iter()
                    .map(crate::preparation::vertex_layout_key)
                    .collect(),
                ranges: draw.vertex_ranges.clone(),
                vertices: draw.vertices.clone(),
                instances: draw.instances.clone(),
                texture: draw.texture,
                groups: draw
                    .bind_groups
                    .iter()
                    .map(|group| {
                        (
                            group.slot,
                            group.dynamic_offsets.clone(),
                            group.uniform_ranges.clone(),
                        )
                    })
                    .collect(),
            })
            .collect(),
    }
}
