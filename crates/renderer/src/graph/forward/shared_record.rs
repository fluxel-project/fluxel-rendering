//! Checked recording shared by ordinary and captured Forward frames.
use super::*;

pub(super) struct RecordSlots<'a> {
    pub(super) device: DeviceId,
    pub(super) topology: FrameTopology,
    pub(super) resident_upload_slot: BufferSlot,
    pub(super) frame_upload_slot: BufferSlot,
    pub(super) uniform_slot: BufferSlot,
    pub(super) environment_upload_slot: Option<BufferSlot>,
    pub(super) stream_slots: &'a [BufferSlot],
    pub(super) texture_slots: &'a [TextureSlot],
    pub(super) environment_slot: Option<TextureSlot>,
    pub(super) target_slot: TextureSlot,
    pub(super) readback_slot: BufferSlot,
}

pub(super) fn record_upload<B: ResourceBackend>(
    slots: &RecordSlots<'_>,
    pass_id: GraphPass,
    bindings: &FrameBindings<'_, B>,
    encoder: &mut CommandEncoder<B>,
    frame: &ForwardFrame<B>,
) -> Result<()>
where
    B: CommandBackend,
{
    let pass = bindings.pass(pass_id)?;
    for upload in &frame.resident_upload.vertices {
        let slot = *slots
            .stream_slots
            .get(upload.stream)
            .ok_or(Error::InvalidRange)?;
        pass.buffer(
            slots.resident_upload_slot,
            upload.source.clone(),
            AccessKind::Read,
            copy_read_uses(),
        )?;
        pass.buffer(
            slot,
            upload.destination.clone(),
            AccessKind::Write,
            copy_write_uses(),
        )?;
        if upload.source.end - upload.source.start
            != upload.destination.end - upload.destination.start
        {
            return Err(Error::InvalidRange);
        }
    }
    pass.buffer(
        slots.frame_upload_slot,
        frame.uniform_upload.source.clone(),
        AccessKind::Read,
        copy_read_uses(),
    )?;
    pass.buffer(
        slots.uniform_slot,
        frame.uniform_upload.destination.clone(),
        AccessKind::Write,
        copy_write_uses(),
    )?;
    if frame.uniform_upload.source.end - frame.uniform_upload.source.start
        != frame.uniform_upload.destination.end - frame.uniform_upload.destination.start
    {
        return Err(Error::InvalidRange);
    }
    for copy in &frame.resident_upload.textures {
        let slot = *slots
            .texture_slots
            .get(copy.texture)
            .ok_or(Error::InvalidRange)?;
        pass.buffer(
            slots.resident_upload_slot,
            copy.source_offset..texture_copy_end(copy)?,
            AccessKind::Read,
            copy_read_uses(),
        )?;
        pass.texture(slot, whole(), AccessKind::Write, texture_copy_write_uses())?;
    }
    if let (Some(slot), Some(owner)) = (slots.environment_upload_slot, frame.environment.as_ref()) {
        pass.buffer(
            slot,
            0..owner.upload_desc.size,
            AccessKind::Read,
            copy_read_uses(),
        )?;
        pass.texture(
            slots.environment_slot.ok_or(Error::InvalidState)?,
            whole(),
            AccessKind::Write,
            texture_copy_write_uses(),
        )?;
    }
    pass.transition(encoder)?;
    for upload in &frame.resident_upload.vertices {
        let slot = *slots
            .stream_slots
            .get(upload.stream)
            .ok_or(Error::InvalidRange)?;
        let src = pass.buffer(
            slots.resident_upload_slot,
            upload.source.clone(),
            AccessKind::Read,
            copy_read_uses(),
        )?;
        let dst = pass.buffer(
            slot,
            upload.destination.clone(),
            AccessKind::Write,
            copy_write_uses(),
        )?;
        encoder.copy_buffer(
            src,
            upload.source.start,
            dst,
            upload.destination.start,
            upload.destination.end - upload.destination.start,
        )?;
    }
    let src = pass.buffer(
        slots.frame_upload_slot,
        frame.uniform_upload.source.clone(),
        AccessKind::Read,
        copy_read_uses(),
    )?;
    let dst = pass.buffer(
        slots.uniform_slot,
        frame.uniform_upload.destination.clone(),
        AccessKind::Write,
        copy_write_uses(),
    )?;
    encoder.copy_buffer(
        src,
        frame.uniform_upload.source.start,
        dst,
        frame.uniform_upload.destination.start,
        frame.uniform_upload.destination.end - frame.uniform_upload.destination.start,
    )?;
    for copy in &frame.resident_upload.textures {
        let src = pass.buffer(
            slots.resident_upload_slot,
            copy.source_offset..texture_copy_end(copy)?,
            AccessKind::Read,
            copy_read_uses(),
        )?;
        let texture = pass.texture(
            *slots
                .texture_slots
                .get(copy.texture)
                .ok_or(Error::InvalidRange)?,
            whole(),
            AccessKind::Write,
            texture_copy_write_uses(),
        )?;
        encoder.buffer_to_texture(
            src,
            TexelCopyBufferLayout {
                offset: copy.source_offset,
                bytes_per_row: copy.bytes_per_row,
                rows_per_image: copy.rows_per_image,
            },
            TextureRegion {
                texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: copy.extent,
                aspect: TextureAspects::COLOR,
            },
        )?;
    }
    if let (Some(slot), Some(owner)) = (slots.environment_upload_slot, frame.environment.as_ref()) {
        let src = pass.buffer(
            slot,
            0..owner.upload_desc.size,
            AccessKind::Read,
            copy_read_uses(),
        )?;
        let texture = pass.texture(
            slots.environment_slot.ok_or(Error::InvalidState)?,
            whole(),
            AccessKind::Write,
            texture_copy_write_uses(),
        )?;
        encoder.buffer_to_texture(
            src,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: owner.bytes_per_row,
                rows_per_image: owner.desc.extent[1],
            },
            TextureRegion {
                texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: owner.desc.extent,
                aspect: TextureAspects::COLOR,
            },
        )?;
    }
    Ok(())
}

pub(super) fn record_raster<B: ResourceBackend>(
    slots: &RecordSlots<'_>,
    pass_id: GraphPass,
    bindings: &FrameBindings<'_, B>,
    encoder: &mut CommandEncoder<B>,
    frame: &ForwardFrame<B>,
) -> Result<()>
where
    B: RasterBackend,
{
    validate_forward_topology(slots.topology, frame.draws.len(), frame.draw_batches.len())?;
    validate_batches(frame)?;
    let pass = bindings.pass(pass_id)?;
    for draw in &frame.draws {
        validate_draw(draw, slots.stream_slots.len())?;
        if draw.pipeline.device() != slots.device {
            return Err(Error::WrongDevice);
        }
        if draw
            .bind_groups
            .iter()
            .any(|group| group.group.device() != slots.device)
        {
            return Err(Error::WrongDevice);
        }
        for (stream_index, range) in draw.vertex_streams.iter().zip(draw_accesses(frame, draw)?) {
            let slot = *slots
                .stream_slots
                .get(*stream_index)
                .ok_or(Error::InvalidRange)?;
            let Some(range) = range else {
                continue;
            };
            pass.buffer(slot, range.clone(), AccessKind::Read, vertex_read_uses())?;
        }
        for group in &draw.bind_groups {
            for range in &group.uniform_ranges {
                pass.buffer(
                    slots.uniform_slot,
                    range.clone(),
                    AccessKind::Read,
                    shader_read_uses(),
                )?;
            }
        }
    }
    if frame.environment.is_some() {
        pass.texture(
            slots.environment_slot.ok_or(Error::InvalidState)?,
            whole(),
            AccessKind::Read,
            texture_read_uses(),
        )?;
    }
    for draw in &frame.draws {
        pass.texture(
            *slots
                .texture_slots
                .get(draw.texture)
                .ok_or(Error::InvalidRange)?,
            whole(),
            AccessKind::Read,
            texture_read_uses(),
        )?;
    }
    pass.texture(
        slots.target_slot,
        whole(),
        AccessKind::Write,
        color_write_uses(),
    )?;
    pass.transition(encoder)?;
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&frame.frame.target_view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
        store: StoreOp::Store,
    })];
    let mut raster = encoder.begin_raster(&RasterPassDesc {
        label: Some("fluxel forward opaque"),
        colors: &colors,
        depth_stencil: None,
        render_area: Rect {
            x: 0,
            y: 0,
            width: frame.extent[0],
            height: frame.extent[1],
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    })?;
    raster.viewport(Viewport {
        x: 0.0,
        y: 0.0,
        width: frame.extent[0] as f32,
        height: frame.extent[1] as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    })?;
    raster.scissor(Rect {
        x: 0,
        y: 0,
        width: frame.extent[0],
        height: frame.extent[1],
    })?;
    for (batch, range) in frame.draw_batches.iter().enumerate() {
        let draws = frame.draws.get(range.clone()).ok_or(Error::InvalidRange)?;
        let first = draws.first().ok_or(Error::InvalidRange)?;
        raster.set_pipeline(&first.pipeline)?;
        for draw in draws {
            if draw.batch != batch {
                return Err(Error::InvalidState);
            }
            for group in &draw.bind_groups {
                raster.set_bind_group(group.slot, &group.group, &group.dynamic_offsets)?;
            }
            for (slot, (stream, range)) in draw
                .vertex_streams
                .iter()
                .zip(&draw.vertex_ranges)
                .enumerate()
            {
                raster.set_vertex_buffer(
                    u32::try_from(slot).map_err(|_| Error::InvalidRange)?,
                    &frame.resident.streams[*stream].buffer,
                    range.clone(),
                )?;
            }
            raster.draw(draw.vertices.clone(), draw.instances.clone())?;
        }
    }
    raster.finish()
}

pub(super) fn record_readback<B: ResourceBackend>(
    slots: &RecordSlots<'_>,
    pass_id: GraphPass,
    bindings: &FrameBindings<'_, B>,
    encoder: &mut CommandEncoder<B>,
    frame: &ForwardFrame<B>,
) -> Result<fluxel_rhi::api::ReadbackTicket<B>>
where
    B: ReadbackBackend,
{
    let length = u64::from(frame.readback_row_pitch)
        .checked_mul(u64::from(frame.extent[1]))
        .ok_or(Error::InvalidRange)?;
    let pass = bindings.pass(pass_id)?;
    let target = pass.texture(
        slots.target_slot,
        whole(),
        AccessKind::Read,
        texture_copy_read_uses(),
    )?;
    let readback = pass.buffer(
        slots.readback_slot,
        0..length,
        AccessKind::Write,
        copy_write_uses(),
    )?;
    pass.transition(encoder)?;
    encoder.texture_to_buffer(
        TextureRegion {
            texture: target,
            mip: 0,
            origin: [0, 0, 0],
            extent: [frame.extent[0], frame.extent[1], 1],
            aspect: TextureAspects::COLOR,
        },
        readback,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: frame.readback_row_pitch,
            rows_per_image: frame.extent[1],
        },
    )?;
    encoder.readback_buffer(readback, 0..length)
}
