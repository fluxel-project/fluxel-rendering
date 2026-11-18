//! Owner-derived facts and recording shapes for Forward declarations.
use super::*;
use crate::graph::ResourceInstanceId;

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct ForwardShape {
    pub(crate) extent: [u32; 2],
    pub(crate) vertex_uploads: Vec<(usize, std::ops::Range<u64>, std::ops::Range<u64>)>,
    pub(crate) texture_uploads: Vec<(usize, u64, u32, u32, [u32; 3])>,
    pub(crate) uniform_upload: (std::ops::Range<u64>, std::ops::Range<u64>),
    pub(crate) environment: Option<(u64, u32, u32, [u32; 3])>,
    /// All physical specular IBL copies and the separate view-uniform copy.
    /// Keeping this in the reusable shape prevents a plan made for one mip
    /// chain/layout from being rebound to a different one.
    pub(crate) specular: Option<SpecularIblShape>,
    pub(crate) multisample_color: Option<TextureShape>,
    pub(crate) depth: Option<DepthShape>,
    /// Batch boundaries affect pipeline binding and therefore are part of the
    /// compiled recording shape even though the batches contain no resources.
    pub(crate) batches: Vec<std::ops::Range<usize>>,
    pub(crate) draws: Vec<ForwardDrawShape>,
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct SpecularIblShape {
    pub(crate) upload_size: u64,
    pub(crate) radiance: TextureShape,
    pub(crate) brdf_lut: TextureShape,
    pub(crate) radiance_copies: Vec<SpecularIblCopyShape>,
    pub(crate) brdf_lut_copy: SpecularIblCopyShape,
    pub(crate) view_uniform: (u64, std::ops::Range<u64>),
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct TextureShape {
    pub(crate) dimension: fluxel_rhi::api::TextureDimension,
    pub(crate) extent: [u32; 3],
    pub(crate) array_layers: u32,
    pub(crate) format: TextureFormat,
    pub(crate) mip_levels: u32,
    pub(crate) samples: u32,
    pub(crate) usage: TextureUsage,
    pub(crate) view_formats: Vec<TextureFormat>,
    pub(crate) incoming: TextureUses,
    pub(crate) layout: ImageLayout,
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct SpecularIblCopyShape {
    pub(crate) mip: u32,
    pub(crate) source_offset: u64,
    pub(crate) bytes_per_row: u32,
    pub(crate) rows_per_image: u32,
    pub(crate) extent: [u32; 3],
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct DepthShape {
    dimension: fluxel_rhi::api::TextureDimension,
    extent: [u32; 3],
    array_layers: u32,
    format: TextureFormat,
    mip_levels: u32,
    samples: u32,
    usage: TextureUsage,
    view_formats: Vec<TextureFormat>,
    incoming: TextureUses,
    layout: ImageLayout,
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct ForwardDrawShape {
    pub(crate) compatibility: crate::preparation::ForwardCompatibilityKey,
    pub(crate) streams: Vec<usize>,
    pub(crate) layouts: Vec<crate::preparation::VertexLayoutKey>,
    pub(crate) ranges: Vec<std::ops::Range<u64>>,
    pub(crate) vertices: std::ops::Range<u32>,
    pub(crate) instances: std::ops::Range<u32>,
    pub(crate) textures: Vec<usize>,
    pub(crate) capture_textures: Vec<bool>,
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
    pub(crate) environment_upload: Option<BufferGraphFact>,
    pub(crate) streams: Vec<BufferGraphFact>,
    pub(crate) stream_instances: Vec<ResourceInstanceId>,
    pub(crate) textures: Vec<TextureGraphFact>,
    pub(crate) texture_instances: Vec<ResourceInstanceId>,
    pub(crate) environment: Option<TextureGraphFact>,
    pub(crate) specular_upload: Option<BufferGraphFact>,
    pub(crate) specular_radiance: Option<TextureGraphFact>,
    pub(crate) specular_brdf_lut: Option<TextureGraphFact>,
    pub(crate) specular_view_uniform: Option<BufferGraphFact>,
    pub(crate) target: TextureGraphFact,
    /// Optional multisampled raster attachment whose resolve target is the
    /// always-single-sampled `target` above.
    pub(crate) multisample_color: Option<TextureGraphFact>,
    pub(crate) depth: Option<TextureGraphFact>,
    pub(crate) readback: BufferGraphFact,
    pub(crate) vertex_uploads: Vec<ForwardGraphVertexUpload>,
    pub(crate) texture_uploads: Vec<ForwardGraphTextureUpload>,
    pub(crate) uniform_upload: ForwardGraphUniformUpload,
    pub(crate) environment_copy: Option<ForwardGraphTextureUpload>,
    pub(crate) specular: Option<ForwardGraphSpecularIbl>,
    pub(crate) draws: Vec<ForwardGraphDraw>,
    pub(crate) shape: ForwardShape,
}

#[derive(Clone)]
pub(crate) struct ForwardGraphSpecularIbl {
    pub(crate) radiance_copies: Vec<SpecularIblCopyShape>,
    pub(crate) brdf_lut_copy: SpecularIblCopyShape,
    pub(crate) view_uniform_copy: Range<u64>,
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
    pub(crate) textures: Vec<usize>,
    pub(crate) capture_textures: Vec<bool>,
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
        let stream_instances = frame
            .resident
            .streams
            .iter()
            .map(|stream| stream.instance)
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
        let texture_instances = frame
            .resident
            .textures
            .iter()
            .map(|texture| texture.instance)
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
                    textures: draw.textures.clone(),
                    capture_textures: draw.capture_textures.clone(),
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
            environment_upload: frame.environment.as_ref().map(|owner| BufferGraphFact {
                desc: owner.upload_desc.clone(),
                incoming: owner.upload_incoming,
            }),
            streams,
            stream_instances,
            textures,
            texture_instances,
            environment: frame.environment.as_ref().map(|owner| TextureGraphFact {
                desc: owner.desc.clone(),
                incoming: owner.incoming,
                layout: owner.layout,
            }),
            specular_upload: frame.specular.as_ref().map(|owner| BufferGraphFact {
                desc: owner.upload_desc.clone(),
                incoming: owner.upload_incoming,
            }),
            specular_radiance: frame.specular.as_ref().map(|owner| TextureGraphFact {
                desc: owner.radiance.desc.clone(),
                incoming: owner.radiance.incoming,
                layout: owner.radiance.layout,
            }),
            specular_brdf_lut: frame.specular.as_ref().map(|owner| TextureGraphFact {
                desc: owner.brdf_lut.desc.clone(),
                incoming: owner.brdf_lut.incoming,
                layout: owner.brdf_lut.layout,
            }),
            specular_view_uniform: frame.specular.as_ref().map(|owner| BufferGraphFact {
                desc: owner.view_uniform_desc.clone(),
                incoming: owner.view_uniform_incoming,
            }),
            target: TextureGraphFact {
                desc: frame.frame.target_desc.clone(),
                incoming: frame.frame.target_incoming,
                layout: frame.frame.target_layout,
            },
            multisample_color: frame.frame.multisample_color.as_ref().map(|owner| {
                TextureGraphFact {
                    desc: owner.desc.clone(),
                    incoming: owner.incoming,
                    layout: owner.layout,
                }
            }),
            depth: frame.frame.depth.as_ref().map(|owner| TextureGraphFact {
                desc: owner.desc.clone(),
                incoming: owner.incoming,
                layout: owner.layout,
            }),
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
            environment_copy: frame
                .environment
                .as_ref()
                .map(|owner| ForwardGraphTextureUpload {
                    texture: 0,
                    source_offset: 0,
                    bytes_per_row: owner.bytes_per_row,
                    rows_per_image: owner.desc.extent[1],
                }),
            specular: frame
                .specular
                .as_ref()
                .map(|owner| ForwardGraphSpecularIbl {
                    radiance_copies: owner
                        .radiance_copies
                        .iter()
                        .map(specular_copy_shape)
                        .collect(),
                    brdf_lut_copy: specular_copy_shape(&owner.brdf_lut_copy),
                    view_uniform_copy: owner.view_uniform_copy.clone(),
                }),
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
    environment: Option<(u64, u32, u32, [u32; 3])>,
    specular: Option<SpecularIblShape>,
    depth: Option<DepthShape>,
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
        environment: frame.environment.as_ref().map(|owner| {
            (
                owner.upload_desc.size,
                owner.bytes_per_row,
                owner.desc.extent[1],
                owner.desc.extent,
            )
        }),
        specular: frame.specular.as_ref().map(specular_shape),
        multisample_color: frame
            .frame
            .multisample_color
            .as_ref()
            .map(|owner| texture_shape(&owner.desc, owner.incoming, owner.layout)),
        depth: frame.frame.depth.as_ref().map(depth_shape),
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
                textures: draw.textures.clone(),
                capture_textures: draw.capture_textures.clone(),
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
        environment: frame.environment.as_ref().map(|owner| {
            (
                owner.upload_desc.size,
                owner.bytes_per_row,
                owner.desc.extent[1],
                owner.desc.extent,
            )
        }),
        specular: frame.specular.as_ref().map(specular_shape),
        depth: frame.frame.depth.as_ref().map(depth_shape),
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
                textures: draw.textures.clone(),
                capture_textures: draw.capture_textures.clone(),
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

fn specular_copy_shape(copy: &crate::residency::SpecularIblCopyFootprint) -> SpecularIblCopyShape {
    SpecularIblCopyShape {
        mip: copy.mip,
        source_offset: copy.source_offset,
        bytes_per_row: copy.bytes_per_row,
        rows_per_image: copy.rows_per_image,
        extent: copy.extent,
    }
}

fn specular_shape<B: ResourceBackend>(
    owner: &crate::residency::SpecularEnvironmentFrameOwner<B>,
) -> SpecularIblShape {
    SpecularIblShape {
        upload_size: owner.upload_desc.size,
        radiance: texture_shape(
            &owner.radiance.desc,
            owner.radiance.incoming,
            owner.radiance.layout,
        ),
        brdf_lut: texture_shape(
            &owner.brdf_lut.desc,
            owner.brdf_lut.incoming,
            owner.brdf_lut.layout,
        ),
        radiance_copies: owner
            .radiance_copies
            .iter()
            .map(specular_copy_shape)
            .collect(),
        brdf_lut_copy: specular_copy_shape(&owner.brdf_lut_copy),
        view_uniform: (
            owner.view_uniform_desc.size,
            owner.view_uniform_copy.clone(),
        ),
    }
}

fn texture_shape(desc: &TextureDesc, incoming: TextureUses, layout: ImageLayout) -> TextureShape {
    TextureShape {
        dimension: desc.dimension,
        extent: desc.extent,
        array_layers: desc.array_layers,
        format: desc.format,
        mip_levels: desc.mip_levels,
        samples: desc.samples,
        usage: desc.usage,
        view_formats: desc.view_formats.clone(),
        incoming,
        layout,
    }
}

fn depth_shape<B: ResourceBackend>(
    owner: &crate::residency::ForwardDepthAttachment<B>,
) -> DepthShape {
    DepthShape {
        dimension: owner.desc.dimension,
        extent: owner.desc.extent,
        array_layers: owner.desc.array_layers,
        format: owner.desc.format,
        mip_levels: owner.desc.mip_levels,
        samples: owner.desc.samples,
        usage: owner.desc.usage,
        view_formats: owner.desc.view_formats.clone(),
        incoming: owner.incoming,
        layout: owner.layout,
    }
}
