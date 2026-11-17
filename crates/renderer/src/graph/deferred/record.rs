//! Geometry, deformation, composite and readback recording for the Deferred Graph.

use super::*;

impl<B: ResourceBackend> DeferredPlan<B> {
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
        if frame.draws.len() != 1 {
            return Err(Error::InvalidRange);
        }
        let (pass_id, slots, deformation) = match (
            self.deformation_pass,
            self.deformation_slots,
            frame.deformation.as_ref(),
        ) {
            (Some(pass), Some(slots), Some(deformation)) => (pass, slots, deformation),
            _ => return Err(Error::InvalidState),
        };
        super::declaration_validation::validate_deformation(
            &self.device,
            &frame.draws[0],
            deformation,
        )?;
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
        if frame.draws.iter().any(|draw| {
            draw.pipeline.device() != self.id.device
                || draw.material_group.device() != self.id.device
                || draw.transform_group.device() != self.id.device
        }) {
            return Err(Error::WrongDevice);
        }
        for draw in frame.draws {
            super::declaration_validation::validate_draw_uniform_ranges(
                draw,
                frame.frame.uniform_desc.size,
                self.uniform_alignment,
            )?;
        }
        let bindings = self.record_bindings(&frame)?;
        let pass = bindings.pass(self.geometry_pass)?;
        let draw_data = frame.draw_graph_data()?;
        if let (Some(slots), Some(_)) = (self.deformation_slots, &frame.deformation) {
            let range = draw_data
                .first()
                .and_then(|draw| draw.accesses.first())
                .and_then(|range| range.clone())
                .ok_or(Error::InvalidRange)?;
            pass.buffer(slots[2], range, AccessKind::Read, vertex_read())?;
        } else {
            for draw in &draw_data {
                for (&stream, access) in draw.streams.iter().zip(&draw.accesses) {
                    if let Some(range) = access {
                        let slot = *self.stream_slots.get(stream).ok_or(Error::InvalidRange)?;
                        pass.buffer(slot, range.clone(), AccessKind::Read, vertex_read())?;
                    }
                }
            }
        }
        for draw in &draw_data {
            for range in [&draw.tint, &draw.transform] {
                pass.buffer(
                    self.uniform_slot,
                    range.clone(),
                    AccessKind::Read,
                    uniform_read(),
                )?;
            }
            if draw.textures.is_empty() {
                return Err(Error::InvalidRange);
            }
            for texture in &draw.textures {
                pass.texture(
                    *self
                        .texture_slots
                        .get(*texture)
                        .ok_or(Error::InvalidRange)?,
                    whole_color(),
                    AccessKind::Read,
                    sampled_read(),
                )?;
            }
        }
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
        if let (Some(slot), Some(_)) = (self.effects_slot, frame.effects) {
            pass.texture(slot, whole_color(), AccessKind::Write, color_write())?;
        }
        pass.texture(
            self.gbuffer_slots[2],
            whole_depth(),
            AccessKind::Write,
            depth_write(),
        )?;
        pass.transition(encoder)?;
        let mut colors = vec![
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
        if let Some(effects) = frame.effects {
            colors.push(Some(ColorAttachment {
                view: AttachmentView::Texture(&effects.view),
                resolve_target: None,
                load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
                store: StoreOp::Store,
            }));
        }
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
        for draw in frame.draws {
            raster.set_pipeline(&draw.pipeline)?;
            raster.set_bind_group(
                0,
                &draw.material_group,
                &[u32::try_from(draw.tint_range.start).map_err(|_| Error::InvalidRange)?],
            )?;
            raster.set_bind_group(
                1,
                &draw.transform_group,
                &[u32::try_from(draw.transform_range.start).map_err(|_| Error::InvalidRange)?],
            )?;
            if let Some(deformation) = &frame.deformation {
                raster.set_vertex_buffer(
                    0,
                    &deformation.product.buffer,
                    draw.vertex_ranges[0].clone(),
                )?;
            } else {
                for (slot, (&stream_index, range)) in draw
                    .vertex_streams
                    .iter()
                    .zip(&draw.vertex_ranges)
                    .enumerate()
                {
                    let stream = frame
                        .resident
                        .streams
                        .get(stream_index)
                        .ok_or(Error::InvalidRange)?;
                    raster.set_vertex_buffer(
                        u32::try_from(slot).map_err(|_| Error::InvalidRange)?,
                        &stream.buffer,
                        range.clone(),
                    )?;
                }
            }
            raster.draw(draw.vertices.clone(), 0..1)?;
        }
        raster.finish()
    }

    pub(crate) fn record_composite(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredFrame<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let composite_pass = self.composite_pass.ok_or(Error::InvalidState)?;
        let scene_color_slot = self.scene_color_slot.ok_or(Error::InvalidState)?;
        let target_slot = self.target_slot.ok_or(Error::InvalidState)?;
        let bindings = self.bindings(frame)?;
        let pass = bindings.pass(composite_pass)?;
        pass.texture(
            scene_color_slot,
            whole_color(),
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
            label: Some("fluxel deferred scene-color composite"),
            colors: &colors,
            depth_stencil: None,
            render_area: render_area(frame.extent),
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        configure_raster(&mut raster, frame.extent)?;
        raster.set_pipeline(&frame.composite_pipeline)?;
        raster.set_bind_group(0, &frame.composite_group, &[])?;
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
