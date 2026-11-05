//! RHI recording for the already declared Deferred graph.

use super::*;

impl<B: ResourceBackend> DeferredPlan<B> {
    pub(crate) fn record_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredFrame<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.record_body_upload(
            encoder,
            super::record_view::DeferredRecordView::offscreen(frame),
        )
    }

    pub(super) fn record_body_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: super::record_view::DeferredRecordView<'_, B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        let bindings = self.record_bindings(&frame)?;
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
                copy_read(),
            )?;
            pass.buffer(
                slot,
                upload.destination.clone(),
                AccessKind::Write,
                copy_write(),
            )?;
        }
        pass.buffer(
            self.frame_upload_slot,
            frame.uniform_upload.source.clone(),
            AccessKind::Read,
            copy_read(),
        )?;
        pass.buffer(
            self.uniform_slot,
            frame.uniform_upload.destination.clone(),
            AccessKind::Write,
            copy_write(),
        )?;
        for copy in &frame.resident_upload.textures {
            pass.buffer(
                self.resident_upload_slot,
                copy.source_offset..texture_copy_end(copy)?,
                AccessKind::Read,
                copy_read(),
            )?;
            pass.texture(
                self.texture_slot,
                whole_color(),
                AccessKind::Write,
                texture_copy_write(),
            )?;
        }
        if let (Some(slots), Some(deformation)) = (self.deformation_slots, &frame.deformation) {
            pass.buffer(
                slots[0],
                0..deformation.source_bytes,
                AccessKind::Read,
                copy_read(),
            )?;
            pass.buffer(
                slots[1],
                0..deformation.source_bytes,
                AccessKind::Write,
                copy_write(),
            )?;
            pass.buffer(
                slots[0],
                deformation.parameter_source.clone(),
                AccessKind::Read,
                copy_read(),
            )?;
            pass.buffer(slots[3], 0..16, AccessKind::Write, copy_write())?;
        }
        pass.transition(encoder)?;
        for upload in &frame.resident_upload.vertices {
            let slot = *self
                .stream_slots
                .get(upload.stream)
                .ok_or(Error::InvalidRange)?;
            let source = pass.buffer(
                self.resident_upload_slot,
                upload.source.clone(),
                AccessKind::Read,
                copy_read(),
            )?;
            let destination = pass.buffer(
                slot,
                upload.destination.clone(),
                AccessKind::Write,
                copy_write(),
            )?;
            encoder.copy_buffer(
                source,
                upload.source.start,
                destination,
                upload.destination.start,
                upload.destination.end - upload.destination.start,
            )?;
        }
        let source = pass.buffer(
            self.frame_upload_slot,
            frame.uniform_upload.source.clone(),
            AccessKind::Read,
            copy_read(),
        )?;
        let destination = pass.buffer(
            self.uniform_slot,
            frame.uniform_upload.destination.clone(),
            AccessKind::Write,
            copy_write(),
        )?;
        encoder.copy_buffer(
            source,
            frame.uniform_upload.source.start,
            destination,
            frame.uniform_upload.destination.start,
            frame.uniform_upload.destination.end - frame.uniform_upload.destination.start,
        )?;
        for copy in &frame.resident_upload.textures {
            let source = pass.buffer(
                self.resident_upload_slot,
                copy.source_offset..texture_copy_end(copy)?,
                AccessKind::Read,
                copy_read(),
            )?;
            let material = pass.texture(
                self.texture_slot,
                whole_color(),
                AccessKind::Write,
                texture_copy_write(),
            )?;
            encoder.buffer_to_texture(
                source,
                TexelCopyBufferLayout {
                    offset: copy.source_offset,
                    bytes_per_row: copy.bytes_per_row,
                    rows_per_image: copy.rows_per_image,
                },
                TextureRegion {
                    texture: material,
                    mip: 0,
                    origin: [0, 0, 0],
                    extent: copy.extent,
                    aspect: TextureAspects::COLOR,
                },
            )?;
        }
        if let (Some(slots), Some(deformation)) = (self.deformation_slots, &frame.deformation) {
            let upload = pass.buffer(
                slots[0],
                0..deformation.source_bytes,
                AccessKind::Read,
                copy_read(),
            )?;
            let source = pass.buffer(
                slots[1],
                0..deformation.source_bytes,
                AccessKind::Write,
                copy_write(),
            )?;
            encoder.copy_buffer(upload, 0, source, 0, deformation.source_bytes)?;
            let upload = pass.buffer(
                slots[0],
                deformation.parameter_source.clone(),
                AccessKind::Read,
                copy_read(),
            )?;
            let parameters = pass.buffer(slots[3], 0..16, AccessKind::Write, copy_write())?;
            encoder.copy_buffer(
                upload,
                deformation.parameter_source.start,
                parameters,
                0,
                16,
            )?;
        }
        Ok(())
    }

    pub(crate) fn record_deformation(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredFrame<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.record_body_deformation(
            encoder,
            super::record_view::DeferredRecordView::offscreen(frame),
        )
    }

    pub(super) fn record_body_deformation(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: super::record_view::DeferredRecordView<'_, B>,
    ) -> Result<()>
    where
        B: CommandBackend + ResourceBackend,
    {
        let (pass_id, slots, deformation) = match (
            self.deformation_pass,
            self.deformation_slots,
            frame.deformation.as_ref(),
        ) {
            (Some(pass), Some(slots), Some(deformation)) => (pass, slots, deformation),
            _ => return Err(Error::InvalidState),
        };
        let bindings = self.record_bindings(&frame)?;
        let pass = bindings.pass(pass_id)?;
        pass.buffer(
            slots[1],
            0..deformation.source_bytes,
            AccessKind::Read,
            compute_read(),
        )?;
        pass.buffer(slots[3], 0..16, AccessKind::Read, compute_parameter_read())?;
        pass.buffer(
            slots[2],
            0..deformation.source_bytes,
            AccessKind::Write,
            compute_write(),
        )?;
        pass.transition(encoder)?;
        let domain = self.device.compute().ok_or(Error::Unsupported("compute"))?;
        let mut compute = domain.begin_labeled(encoder, "fluxel deferred deformation")?;
        compute.set_pipeline(&deformation.pipeline)?;
        compute.set_bind_group(0, &deformation.group, &[])?;
        compute.dispatch(deformation.groups)?;
        compute.finish()
    }

    pub(crate) fn record_geometry(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredFrame<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        self.record_body_geometry(
            encoder,
            super::record_view::DeferredRecordView::offscreen(frame),
        )
    }

    pub(super) fn record_body_geometry(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: super::record_view::DeferredRecordView<'_, B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let bindings = self.record_bindings(&frame)?;
        let pass = bindings.pass(self.geometry_pass)?;
        let accesses = frame.draw_accesses()?;
        if let (Some(slots), Some(_)) = (self.deformation_slots, &frame.deformation) {
            let range = accesses
                .first()
                .and_then(|range| range.clone())
                .ok_or(Error::InvalidRange)?;
            pass.buffer(slots[2], range, AccessKind::Read, vertex_read())?;
        } else {
            for (slot, access) in self.stream_slots.iter().copied().zip(accesses) {
                if let Some(range) = access {
                    pass.buffer(slot, range, AccessKind::Read, vertex_read())?;
                }
            }
        }
        for range in [&frame.draw.tint_range, &frame.draw.transform_range] {
            pass.buffer(
                self.uniform_slot,
                range.clone(),
                AccessKind::Read,
                uniform_read(),
            )?;
        }
        pass.texture(
            self.texture_slot,
            whole_color(),
            AccessKind::Read,
            sampled_read(),
        )?;
        pass.texture(
            self.gbuffer_slots[0],
            whole_color(),
            AccessKind::Write,
            color_write(),
        )?;
        pass.texture(
            self.gbuffer_slots[1],
            whole_color(),
            AccessKind::Write,
            color_write(),
        )?;
        pass.texture(
            self.gbuffer_slots[2],
            whole_depth(),
            AccessKind::Write,
            depth_write(),
        )?;
        pass.transition(encoder)?;
        let colors = [
            Some(ColorAttachment {
                view: AttachmentView::Texture(&frame.gbuffer[0].view),
                resolve_target: None,
                load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
                store: StoreOp::Store,
            }),
            Some(ColorAttachment {
                view: AttachmentView::Texture(&frame.gbuffer[1].view),
                resolve_target: None,
                load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
                store: StoreOp::Store,
            }),
        ];
        let depth_stencil = DepthStencilAttachment {
            view: AttachmentView::Texture(&frame.gbuffer[2].view),
            depth: Some((LoadOp::Clear(1.0), StoreOp::Store)),
            stencil: None,
            depth_read_only: false,
            stencil_read_only: true,
        };
        let mut raster = encoder.begin_raster(&RasterPassDesc {
            label: Some("fluxel deferred geometry"),
            colors: &colors,
            depth_stencil: Some(depth_stencil),
            render_area: render_area(frame.extent),
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        configure_raster(&mut raster, frame.extent)?;
        raster.set_pipeline(&frame.draw.pipeline)?;
        raster.set_bind_group(
            0,
            &frame.draw.material_group,
            &[u32::try_from(frame.draw.tint_range.start).map_err(|_| Error::InvalidRange)?],
        )?;
        raster.set_bind_group(
            1,
            &frame.draw.transform_group,
            &[u32::try_from(frame.draw.transform_range.start).map_err(|_| Error::InvalidRange)?],
        )?;
        if let Some(deformation) = &frame.deformation {
            raster.set_vertex_buffer(
                0,
                &deformation.product.buffer,
                frame.draw.vertex_ranges[0].clone(),
            )?;
        } else {
            for (index, (stream, range)) in frame
                .resident
                .streams
                .iter()
                .zip(&frame.draw.vertex_ranges)
                .enumerate()
            {
                raster.set_vertex_buffer(
                    u32::try_from(index).map_err(|_| Error::InvalidRange)?,
                    &stream.buffer,
                    range.clone(),
                )?;
            }
        }
        raster.draw(frame.draw.vertices.clone(), 0..1)?;
        raster.finish()
    }

    pub(crate) fn record_lighting(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredFrame<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let target_slot = self.target_slot.ok_or(Error::InvalidState)?;
        let bindings = self.bindings(frame)?;
        let pass = bindings.pass(self.lighting_pass)?;
        pass.texture(
            self.gbuffer_slots[0],
            whole_color(),
            AccessKind::Read,
            sampled_read(),
        )?;
        pass.texture(
            self.gbuffer_slots[1],
            whole_color(),
            AccessKind::Read,
            sampled_read(),
        )?;
        pass.texture(
            self.gbuffer_slots[2],
            whole_depth(),
            AccessKind::Read,
            sampled_read(),
        )?;
        pass.texture(target_slot, whole_color(), AccessKind::Write, color_write())?;
        pass.transition(encoder)?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&frame.frame.target_view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
            store: StoreOp::Store,
        })];
        let mut raster = encoder.begin_raster(&RasterPassDesc {
            label: Some("fluxel deferred lighting"),
            colors: &colors,
            depth_stencil: None,
            render_area: render_area(frame.extent),
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        configure_raster(&mut raster, frame.extent)?;
        raster.set_pipeline(&frame.lighting_pipeline)?;
        raster.set_bind_group(0, &frame.lighting_group, &[])?;
        raster.draw(0..3, 0..1)?;
        raster.finish()
    }

    pub(crate) fn record_readback(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredFrame<B>,
    ) -> Result<fluxel_rhi::api::ReadbackTicket<B>>
    where
        B: ReadbackBackend,
    {
        let target_slot = self.target_slot.ok_or(Error::InvalidState)?;
        let readback_slot = self.readback_slot.ok_or(Error::InvalidState)?;
        let length = u64::from(frame.readback_row_pitch)
            .checked_mul(u64::from(frame.extent[1]))
            .ok_or(Error::InvalidRange)?;
        let bindings = self.bindings(frame)?;
        let readback_pass = match self.output {
            DeferredOutputPlan::Offscreen { readback_pass } => readback_pass,
            DeferredOutputPlan::Surface { .. } => return Err(Error::InvalidState),
        };
        let pass = bindings.pass(readback_pass)?;
        let target = pass.texture(
            target_slot,
            whole_color(),
            AccessKind::Read,
            texture_copy_read(),
        )?;
        let readback = pass.buffer(readback_slot, 0..length, AccessKind::Write, copy_write())?;
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

fn render_area(extent: [u32; 2]) -> Rect {
    Rect {
        x: 0,
        y: 0,
        width: extent[0],
        height: extent[1],
    }
}

fn configure_raster<B: RasterBackend>(
    raster: &mut fluxel_rhi::api::RasterPass<'_, B>,
    extent: [u32; 2],
) -> Result<()> {
    raster.viewport(Viewport {
        x: 0.0,
        y: 0.0,
        width: extent[0] as f32,
        height: extent[1] as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    })?;
    raster.scissor(render_area(extent))
}
