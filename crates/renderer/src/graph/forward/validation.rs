//! Concrete Forward ranges, uploads, and batch validation.
use super::*;

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
pub(super) fn draw_accesses<B: ResourceBackend>(
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
    let mut uploaded_textures = vec![false; frame.resident.textures.len()];
    for upload in &frame.resident_upload.textures {
        let texture = frame
            .resident
            .textures
            .get(upload.texture)
            .ok_or(Error::InvalidRange)?;
        let already_uploaded = std::mem::replace(
            uploaded_textures
                .get_mut(upload.texture)
                .ok_or(Error::InvalidRange)?,
            true,
        );
        if already_uploaded
            || !upload.source_offset.is_multiple_of(512)
            || upload.extent != texture.desc.extent
            || upload.rows_per_image != upload.extent[1]
            || upload.bytes_per_row == 0
            || !upload.bytes_per_row.is_multiple_of(256)
            || u64::from(upload.bytes_per_row) < u64::from(upload.extent[0]) * 4
            || texture_copy_end(upload)? > frame.resident.upload_desc.size
        {
            return Err(Error::InvalidRange);
        }
    }
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
    let mut uploaded_textures = vec![false; frame.resident.textures.len()];
    for upload in &frame.resident_upload.textures {
        let texture = frame
            .resident
            .textures
            .get(upload.texture)
            .ok_or(Error::InvalidRange)?;
        let already = std::mem::replace(
            uploaded_textures
                .get_mut(upload.texture)
                .ok_or(Error::InvalidRange)?,
            true,
        );
        if already
            || !upload.source_offset.is_multiple_of(512)
            || upload.extent != texture.desc.extent
            || upload.rows_per_image != upload.extent[1]
            || upload.bytes_per_row == 0
            || !upload.bytes_per_row.is_multiple_of(256)
            || u64::from(upload.bytes_per_row) < u64::from(upload.extent[0]) * 4
            || texture_copy_end(upload)? > frame.resident.upload_desc.size
        {
            return Err(Error::InvalidRange);
        }
    }
    Ok(())
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
