//! Offscreen Record for the bounded Forward product.
use super::*;

impl<B: ResourceBackend> ForwardPlan<B> {
    pub(crate) fn record_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
    ) -> Result<()>
    where
        B: CommandBackend + ResourceBackend,
    {
        let bindings = self.bindings(frame)?;
        let pass = bindings.pass(self.upload_pass)?;
        for upload in &frame.resident_upload.vertices {
            let slot = *self
                .stream_slots
                .get(upload.stream)
                .ok_or(Error::InvalidRange)?;
            pass.buffer(
                self.resident_upload_slot,
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
            self.frame_upload_slot,
            frame.uniform_upload.source.clone(),
            AccessKind::Read,
            copy_read_uses(),
        )?;
        pass.buffer(
            self.uniform_slot,
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
            let slot = *self
                .texture_slots
                .get(copy.texture)
                .ok_or(Error::InvalidRange)?;
            pass.buffer(
                self.resident_upload_slot,
                copy.source_offset..texture_copy_end(copy)?,
                AccessKind::Read,
                copy_read_uses(),
            )?;
            pass.texture(slot, whole(), AccessKind::Write, texture_copy_write_uses())?;
        }
        pass.transition(encoder)?;
        for upload in &frame.resident_upload.vertices {
            let slot = *self
                .stream_slots
                .get(upload.stream)
                .ok_or(Error::InvalidRange)?;
            let src = pass.buffer(
                self.resident_upload_slot,
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
            self.frame_upload_slot,
            frame.uniform_upload.source.clone(),
            AccessKind::Read,
            copy_read_uses(),
        )?;
        let dst = pass.buffer(
            self.uniform_slot,
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
                self.resident_upload_slot,
                copy.source_offset..texture_copy_end(copy)?,
                AccessKind::Read,
                copy_read_uses(),
            )?;
            let texture = pass.texture(
                *self
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
        Ok(())
    }

    pub(crate) fn record_raster(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
    ) -> Result<()>
    where
        B: RasterBackend + ResourceBackend,
    {
        validate_forward_topology(self.topology, frame.draws.len(), frame.draw_batches.len())?;
        validate_batches(frame)?;
        let bindings = self.bindings(frame)?;
        let pass = bindings.pass(self.raster_pass)?;
        for draw in &frame.draws {
            validate_draw(draw, self.stream_slots.len())?;
            if draw.pipeline.device() != self.id.device {
                return Err(Error::WrongDevice);
            }
            if draw
                .bind_groups
                .iter()
                .any(|group| group.group.device() != self.id.device)
            {
                return Err(Error::WrongDevice);
            }
            for (stream_index, range) in draw.vertex_streams.iter().zip(draw_accesses(frame, draw)?)
            {
                let slot = *self
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
                        self.uniform_slot,
                        range.clone(),
                        AccessKind::Read,
                        shader_read_uses(),
                    )?;
                }
            }
        }
        for draw in &frame.draws {
            pass.texture(
                *self
                    .texture_slots
                    .get(draw.texture)
                    .ok_or(Error::InvalidRange)?,
                whole(),
                AccessKind::Read,
                texture_read_uses(),
            )?;
        }
        pass.texture(
            self.target_slot,
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

    pub(crate) fn record_readback(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
    ) -> Result<fluxel_rhi::api::ReadbackTicket<B>>
    where
        B: ReadbackBackend + ResourceBackend,
    {
        let length = u64::from(frame.readback_row_pitch)
            .checked_mul(u64::from(frame.extent[1]))
            .ok_or(Error::InvalidRange)?;
        let bindings = self.bindings(frame)?;
        let pass = bindings.pass(self.readback_pass)?;
        let target = pass.texture(
            self.target_slot,
            whole(),
            AccessKind::Read,
            texture_copy_read_uses(),
        )?;
        let readback = pass.buffer(
            self.readback_slot,
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
}
