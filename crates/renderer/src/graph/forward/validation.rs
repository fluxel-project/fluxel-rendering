//! Concrete Forward ranges, uploads, and batch validation.
use super::*;

/// Depth is an owner-selected optional attachment, but when selected its
/// allocation contract is fixed: one Depth32Float 2D render target covering
/// the frame and still in its saved incoming state.
pub(super) fn validate_depth_spec(spec: &ForwardGraphSpec) -> Result<()> {
    let Some(depth) = spec.depth.as_ref() else {
        return Ok(());
    };
    if depth.desc.dimension != fluxel_rhi::api::TextureDimension::D2
        || depth.desc.extent != [spec.extent[0], spec.extent[1], 1]
        || depth.desc.array_layers != 1
        || depth.desc.format != TextureFormat::Depth32Float
        || depth.desc.mip_levels != 1
        || depth.desc.samples
            != spec
                .multisample_color
                .as_ref()
                .map_or(1, |color| color.desc.samples)
        || !depth.desc.usage.contains(TextureUsage::RENDER_ATTACHMENT)
        || depth.incoming
            != (TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            })
        || depth.layout != ImageLayout::Undefined
    {
        return Err(Error::InvalidState);
    }
    Ok(())
}

/// The resolve target is the ordinary single-sample frame target.  The
/// multisample owner exists solely for the raster attachment and is never
/// sampled or copied by this graph path.
pub(super) fn validate_multisample_color_spec(spec: &ForwardGraphSpec) -> Result<()> {
    let Some(color) = spec.multisample_color.as_ref() else {
        return Ok(());
    };
    if color.desc.dimension != fluxel_rhi::api::TextureDimension::D2
        || color.desc.extent != [spec.extent[0], spec.extent[1], 1]
        || color.desc.array_layers != 1
        || color.desc.format != spec.target.desc.format
        || color.desc.mip_levels != 1
        || color.desc.samples < 2
        || !color.desc.usage.contains(TextureUsage::RENDER_ATTACHMENT)
        || spec.target.desc.samples != 1
        || !spec
            .target
            .desc
            .usage
            .contains(TextureUsage::RENDER_ATTACHMENT)
        || color.incoming
            != (TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            })
        || color.layout != ImageLayout::Undefined
    {
        return Err(Error::InvalidState);
    }
    Ok(())
}

/// Validates the topology's depth requirement independently of allocation.
///
/// Sorted Forward always needs the shared depth attachment: opaque draws write
/// it and transparent draws test it in the same raster pass. Keeping this
/// check separate from owner validation ensures cached plans and direct record
/// paths reject a subsequently mutated frame instead of trusting allocation.
pub(super) fn validate_depth_topology(topology: FrameTopology, has_depth: bool) -> Result<()> {
    match (topology, has_depth) {
        (FrameTopology::SortedForward, false) => Err(Error::InvalidState),
        (FrameTopology::OpaqueForward | FrameTopology::SortedForward, _) | (_, false) => Ok(()),
        _ => Err(Error::InvalidState),
    }
}

pub(super) fn validate_surface_depth<B: ResourceBackend>(
    frame: &SurfaceForwardFrame<B>,
) -> Result<()> {
    let Some(depth) = frame.frame.depth.as_ref() else {
        return Ok(());
    };
    if depth.device != frame.resident.device
        || depth.texture.device() != frame.resident.device
        || depth.view.device() != frame.resident.device
        || depth.desc.dimension != fluxel_rhi::api::TextureDimension::D2
        || depth.desc.extent != [frame.extent[0], frame.extent[1], 1]
        || depth.desc.array_layers != 1
        || depth.desc.format != TextureFormat::Depth32Float
        || depth.desc.mip_levels != 1
        || depth.desc.samples != 1
        || !depth.desc.usage.contains(TextureUsage::RENDER_ATTACHMENT)
        || depth.incoming
            != (TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            })
        || depth.layout != ImageLayout::Undefined
    {
        return Err(Error::InvalidState);
    }
    Ok(())
}

pub(super) fn validate_offsets<B: ResourceBackend>(
    draw: &crate::residency::ForwardDraw<B>,
    alignment: u64,
) -> Result<()> {
    if alignment == 0
        || draw
            .bind_groups
            .iter()
            .flat_map(|group| &group.dynamic_offsets)
            .any(|offset| !u64::from(*offset).is_multiple_of(alignment))
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

pub(super) fn validate_batches<B: ResourceBackend>(frame: &ForwardFrame<B>) -> Result<()> {
    let mut next = 0;
    for (batch, range) in frame.draw_batches.iter().enumerate() {
        if range.start != next || range.start >= range.end || range.end > frame.draws.len() {
            return Err(Error::InvalidRange);
        }
        if frame.draws[range.clone()]
            .iter()
            .any(|draw| draw.batch != batch)
        {
            return Err(Error::InvalidState);
        }
        let first = &frame.draws[range.start].compatibility;
        if frame.draws[range.clone()]
            .iter()
            .any(|draw| &draw.compatibility != first)
        {
            return Err(Error::InvalidState);
        }
        next = range.end;
    }
    if next != frame.draws.len() {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

pub(super) fn validate_draw<B: ResourceBackend>(
    draw: &crate::residency::ForwardDraw<B>,
    stream_count: usize,
) -> Result<()> {
    if draw.vertex_streams.len() != draw.vertex_layouts.len()
        || draw.vertex_streams.len() != draw.vertex_ranges.len()
        || draw.textures.len() != draw.capture_textures.len()
        || draw
            .vertex_streams
            .iter()
            .any(|stream| *stream >= stream_count)
        || draw.vertices.start >= draw.vertices.end
        || draw.instances.start >= draw.instances.end
        || draw.bind_groups.is_empty()
        || draw.bind_groups.iter().any(|group| {
            group.uniform_ranges.len() != group.dynamic_offsets.len()
                || group
                    .uniform_ranges
                    .iter()
                    .zip(&group.dynamic_offsets)
                    .any(|(range, offset)| {
                        range.start >= range.end || u32::try_from(range.start).ok() != Some(*offset)
                    })
        })
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

pub(super) fn validate_graph_draw(draw: &ForwardGraphDraw, stream_count: usize) -> Result<()> {
    if draw.vertex_streams.len() != draw.vertex_layouts.len()
        || draw.vertex_streams.len() != draw.vertex_ranges.len()
        || draw.vertex_streams.len() != draw.accesses.len()
        || draw.textures.len() != draw.capture_textures.len()
        || draw
            .vertex_streams
            .iter()
            .any(|stream| *stream >= stream_count)
        || draw.vertices.start >= draw.vertices.end
        || draw.instances.start >= draw.instances.end
        || draw
            .uniform_ranges
            .iter()
            .any(|range| range.start >= range.end)
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

/// Derives the declared vertex fetches from owner facts rather than trusting a
/// coordinator-provided access range. The RHI binding range remains unchanged;
/// only the Graph access narrows to the bytes the draw can actually consume.
pub(crate) fn draw_accesses<B: ResourceBackend>(
    frame: &ForwardFrame<B>,
    draw: &crate::residency::ForwardDraw<B>,
) -> Result<Vec<Option<std::ops::Range<u64>>>> {
    validate_draw(draw, frame.resident.streams.len())?;
    let mut bindings = Vec::with_capacity(draw.vertex_streams.len());
    for (stream, range) in draw.vertex_streams.iter().zip(&draw.vertex_ranges) {
        let owner = frame
            .resident
            .streams
            .get(*stream)
            .ok_or(Error::InvalidRange)?;
        bindings.push(GeometryStreamBinding {
            owner: GeometryBufferOwner::new(
                frame.resident.device,
                owner.desc.clone(),
                owner.incoming,
            ),
            range: range.clone(),
        });
    }
    PreparedGeometry {
        streams: draw.vertex_layouts.clone(),
    }
    .bind(
        frame.resident.device,
        &bindings,
        GeometryDraw {
            vertices: draw.vertices.clone(),
            instances: draw.instances.clone(),
        },
    )
    .map(|prepared| prepared.accessed_ranges().to_vec())
    .map_err(|_| Error::InvalidRange)
}

pub(super) fn validate_uploads<B: ResourceBackend>(frame: &ForwardFrame<B>) -> Result<()> {
    if frame.readback_row_pitch == 0
        || !frame.readback_row_pitch.is_multiple_of(256)
        || u64::from(frame.readback_row_pitch) < u64::from(frame.extent[0]) * 4
    {
        return Err(Error::InvalidRange);
    }
    let mut uploaded_streams = vec![false; frame.resident.streams.len()];
    for upload in &frame.resident_upload.vertices {
        let stream = frame
            .resident
            .streams
            .get(upload.stream)
            .ok_or(Error::InvalidRange)?;
        let already_uploaded = std::mem::replace(
            uploaded_streams
                .get_mut(upload.stream)
                .ok_or(Error::InvalidRange)?,
            true,
        );
        if already_uploaded
            || upload.source.start >= upload.source.end
            || upload.destination.start >= upload.destination.end
            || upload.destination.end > stream.desc.size
            || upload.source.end > frame.resident.upload_desc.size
            || upload.source.end - upload.source.start
                != upload.destination.end - upload.destination.start
        {
            return Err(Error::InvalidRange);
        }
    }
    validate_texture_uploads(
        &frame.resident,
        &frame.resident_upload.textures,
        frame.resident.upload_desc.size,
    )?;
    Ok(())
}

/// Validates only owners that exist in the direct-surface frame. In
/// particular, a surface frame must not manufacture an offscreen target or a
/// readback buffer merely to satisfy Graph validation.
pub(super) fn validate_surface_uploads<B: ResourceBackend>(
    frame: &SurfaceForwardFrame<B>,
) -> Result<()> {
    let mut uploaded_streams = vec![false; frame.resident.streams.len()];
    for upload in &frame.resident_upload.vertices {
        let stream = frame
            .resident
            .streams
            .get(upload.stream)
            .ok_or(Error::InvalidRange)?;
        let already = std::mem::replace(
            uploaded_streams
                .get_mut(upload.stream)
                .ok_or(Error::InvalidRange)?,
            true,
        );
        if already
            || upload.source.start >= upload.source.end
            || upload.destination.start >= upload.destination.end
            || upload.destination.end > stream.desc.size
            || upload.source.end > frame.resident.upload_desc.size
            || upload.source.end - upload.source.start
                != upload.destination.end - upload.destination.start
        {
            return Err(Error::InvalidRange);
        }
    }
    validate_texture_uploads(
        &frame.resident,
        &frame.resident_upload.textures,
        frame.resident.upload_desc.size,
    )?;
    Ok(())
}

fn validate_texture_uploads<B: ResourceBackend>(
    resident: &crate::residency::ForwardResidentBuffers<B>,
    uploads: &[crate::residency::ForwardTextureUpload],
    upload_size: u64,
) -> Result<()> {
    let mut uploaded = resident
        .textures
        .iter()
        .map(|texture| {
            vec![vec![false; texture.desc.array_layers as usize]; texture.desc.mip_levels as usize]
        })
        .collect::<Vec<_>>();
    for upload in uploads {
        let texture = resident
            .textures
            .get(upload.texture)
            .ok_or(Error::InvalidRange)?;
        if !matches!(
            texture.desc.dimension,
            fluxel_rhi::api::TextureDimension::D2 | fluxel_rhi::api::TextureDimension::D3
        ) || (texture.desc.dimension == fluxel_rhi::api::TextureDimension::D2
            && texture.desc.extent[2] != 1)
            || (texture.desc.dimension == fluxel_rhi::api::TextureDimension::D3
                && texture.desc.array_layers != 1)
            || texture.desc.array_layers == 0
        {
            return Err(Error::InvalidRange);
        }
        let layer_uploaded = uploaded
            .get_mut(upload.texture)
            .and_then(|mips| mips.get_mut(upload.mip_level as usize))
            .and_then(|layers| layers.get_mut(upload.array_layer as usize))
            .ok_or(Error::InvalidRange)?;
        let expected = [
            (texture.desc.extent[0] >> upload.mip_level).max(1),
            (texture.desc.extent[1] >> upload.mip_level).max(1),
            if texture.desc.dimension == fluxel_rhi::api::TextureDimension::D3 {
                (texture.desc.extent[2] >> upload.mip_level).max(1)
            } else {
                1
            },
        ];
        let bytes_per_texel = texture_bytes_per_texel(texture.desc.format)?;
        if std::mem::replace(layer_uploaded, true)
            || !upload.source_offset.is_multiple_of(512)
            || upload.extent != expected
            || upload.rows_per_image != upload.extent[1]
            || upload.bytes_per_row == 0
            || !upload.bytes_per_row.is_multiple_of(256)
            || u64::from(upload.bytes_per_row)
                < u64::from(upload.extent[0]) * u64::from(bytes_per_texel)
            || texture_copy_end(upload)? > upload_size
        {
            return Err(Error::InvalidRange);
        }
    }
    for (texture, mips) in resident.textures.iter().zip(uploaded) {
        if !matches!(
            texture.desc.dimension,
            fluxel_rhi::api::TextureDimension::D2 | fluxel_rhi::api::TextureDimension::D3
        ) || (texture.desc.dimension == fluxel_rhi::api::TextureDimension::D2
            && texture.desc.extent[2] != 1)
            || (texture.desc.dimension == fluxel_rhi::api::TextureDimension::D3
                && texture.desc.array_layers != 1)
            || texture.desc.array_layers == 0
        {
            return Err(Error::InvalidRange);
        }
        let any_uploaded = mips.iter().flatten().any(|uploaded| *uploaded);
        if texture.desc.usage.contains(TextureUsage::COPY_SRC) {
            // Graph generation starts from the complete base mip, including
            // every array layer or the full depth of a volume.
            if any_uploaded
                && (!mips
                    .first()
                    .is_some_and(|layers| layers.iter().all(|layer| *layer))
                    || mips.iter().skip(1).flatten().any(|uploaded| *uploaded))
            {
                return Err(Error::InvalidRange);
            }
        } else if any_uploaded && mips.iter().flatten().any(|uploaded| !uploaded) {
            // Without COPY_SRC every physical subresource is authored.
            return Err(Error::InvalidRange);
        }
    }
    Ok(())
}

fn texture_bytes_per_texel(format: TextureFormat) -> Result<u32> {
    match format {
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb | TextureFormat::Depth32Float => Ok(4),
        TextureFormat::Rgba16Float => Ok(8),
        _ => Err(Error::InvalidRange),
    }
}

pub(super) fn validate_surface_batches<B: ResourceBackend>(
    frame: &SurfaceForwardFrame<B>,
) -> Result<()> {
    let mut next = 0;
    for (batch, range) in frame.draw_batches.iter().enumerate() {
        if range.start != next || range.start >= range.end || range.end > frame.draws.len() {
            return Err(Error::InvalidRange);
        }
        if frame.draws[range.clone()]
            .iter()
            .any(|draw| draw.batch != batch)
        {
            return Err(Error::InvalidState);
        }
        let first = &frame.draws[range.start].compatibility;
        if frame.draws[range.clone()]
            .iter()
            .any(|draw| &draw.compatibility != first)
        {
            return Err(Error::InvalidState);
        }
        next = range.end;
    }
    if next != frame.draws.len() {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

/// A clear-only product owns no material, vertex, or texture accesses. Keeping
/// this invariant separate from batch validation prevents a caller from
/// recording draws that the clear-only Graph did not declare.
pub(super) fn validate_forward_topology(
    topology: FrameTopology,
    draw_count: usize,
    batch_count: usize,
) -> Result<()> {
    if topology == FrameTopology::ClearOnly && (draw_count != 0 || batch_count != 0) {
        return Err(Error::InvalidState);
    }
    Ok(())
}

pub(super) fn validate_surface_topology(
    topology: FrameTopology,
    draw_count: usize,
    batch_count: usize,
) -> Result<()> {
    validate_forward_topology(topology, draw_count, batch_count)
}

pub(super) fn surface_draw_accesses<B: ResourceBackend>(
    frame: &SurfaceForwardFrame<B>,
    draw: &crate::residency::ForwardDraw<B>,
) -> Result<Vec<Option<std::ops::Range<u64>>>> {
    validate_draw(draw, frame.resident.streams.len())?;
    let mut bindings = Vec::with_capacity(draw.vertex_streams.len());
    for (stream, range) in draw.vertex_streams.iter().zip(&draw.vertex_ranges) {
        let owner = frame
            .resident
            .streams
            .get(*stream)
            .ok_or(Error::InvalidRange)?;
        bindings.push(GeometryStreamBinding {
            owner: GeometryBufferOwner::new(
                frame.resident.device,
                owner.desc.clone(),
                owner.incoming,
            ),
            range: range.clone(),
        });
    }
    PreparedGeometry {
        streams: draw.vertex_layouts.clone(),
    }
    .bind(
        frame.resident.device,
        &bindings,
        GeometryDraw {
            vertices: draw.vertices.clone(),
            instances: draw.instances.clone(),
        },
    )
    .map(|prepared| prepared.accessed_ranges().to_vec())
    .map_err(|_| Error::InvalidRange)
}
