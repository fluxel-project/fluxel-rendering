//! Surface Record for the bounded Forward product.
use super::*;

impl<B: ResourceBackend> SurfaceForwardPlan<B> {
    pub(crate) fn record_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &SurfaceForwardFrame<B>,
        image: &SurfaceImage<B>,
    ) -> Result<()>
    where
        B: CommandBackend + ResourceBackend,
    {
        self.record_surface_upload(encoder, frame, image)
    }

    pub(crate) fn record_raster(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &SurfaceForwardFrame<B>,
        image: &SurfaceImage<B>,
    ) -> Result<()>
    where
        B: RasterBackend + ResourceBackend,
    {
        if image.device() != self.id.device {
            return Err(Error::WrongDevice);
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated);
        }
        if image.extent() != frame.extent {
            return Err(Error::InvalidRange);
        }
        if image.format() != frame.color_format {
            return Err(Error::InvalidRange);
        }
        validate_surface_topology(self.topology, frame.draws.len(), frame.draw_batches.len())?;
        validate_surface_batches(frame)?;
        let bindings = self.bindings(frame, image)?;
        let pass = bindings.pass(self.raster_pass)?;
        for draw in &frame.draws {
            validate_draw(draw, self.stream_slots.len())?;
            if draw.pipeline.device() != self.id.device
                || draw
                    .bind_groups
                    .iter()
                    .any(|group| group.group.device() != self.id.device)
            {
                return Err(Error::WrongDevice);
            }
            for (stream_index, range) in draw
                .vertex_streams
                .iter()
                .zip(surface_draw_accesses(frame, draw)?)
            {
                let slot = *self
                    .stream_slots
                    .get(*stream_index)
                    .ok_or(Error::InvalidRange)?;
                if let Some(range) = range {
                    pass.buffer(slot, range, AccessKind::Read, vertex_read_uses())?;
                }
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
            if draw.textures.is_empty() {
                return Err(Error::InvalidRange);
            }
            for texture in &draw.textures {
                pass.texture(
                    *self
                        .texture_slots
                        .get(*texture)
                        .ok_or(Error::InvalidRange)?,
                    whole(),
                    AccessKind::Read,
                    texture_read_uses(),
                )?;
            }
        }
        if frame.environment.is_some() {
            pass.texture(
                self.environment_slot.ok_or(Error::InvalidState)?,
                whole(),
                AccessKind::Read,
                texture_read_uses(),
            )?;
        }
        if let Some(owner) = frame.specular.as_ref() {
            owner.validate_for_sampling(self.id.device)?;
            pass.texture(
                self.specular_radiance_slot.ok_or(Error::InvalidState)?,
                all_color_mips(owner.radiance.desc.mip_levels),
                AccessKind::Read,
                texture_read_uses(),
            )?;
            pass.texture(
                self.specular_brdf_lut_slot.ok_or(Error::InvalidState)?,
                whole(),
                AccessKind::Read,
                texture_read_uses(),
            )?;
            pass.buffer(
                self.specular_view_uniform_slot.ok_or(Error::InvalidState)?,
                0..112,
                AccessKind::Read,
                shader_read_uses(),
            )?;
        }
        pass.surface_texture(
            self.surface_target_slot,
            whole(),
            AccessKind::Write,
            color_write_uses(),
        )?;
        let depth_stencil = match (self.depth_slot, frame.frame.depth.as_ref()) {
            (None, None) => None,
            (Some(slot), Some(owner)) => {
                if owner.device != self.id.device
                    || owner.texture.device() != self.id.device
                    || owner.view.device() != self.id.device
                {
                    return Err(Error::WrongDevice);
                }
                pass.texture(slot, whole_depth(), AccessKind::Write, depth_write_uses())?;
                Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(&owner.view),
                    depth: Some((LoadOp::Clear(1.0), StoreOp::Discard)),
                    stencil: None,
                    depth_read_only: false,
                    stencil_read_only: true,
                })
            }
            _ => return Err(Error::InvalidState),
        };
        pass.transition(encoder)?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
        })];
        let mut raster = encoder.begin_raster(&RasterPassDesc {
            label: Some("fluxel forward surface"),
            colors: &colors,
            depth_stencil,
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

    fn record_surface_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &SurfaceForwardFrame<B>,
        image: &SurfaceImage<B>,
    ) -> Result<()>
    where
        B: CommandBackend + ResourceBackend,
    {
        let bindings = self.bindings(frame, image)?;
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
        if let (Some(slot), Some(owner)) =
            (self.environment_upload_slot, frame.environment.as_ref())
        {
            pass.buffer(
                slot,
                0..owner.upload_desc.size,
                AccessKind::Read,
                copy_read_uses(),
            )?;
            pass.texture(
                self.environment_slot.ok_or(Error::InvalidState)?,
                whole(),
                AccessKind::Write,
                texture_copy_write_uses(),
            )?;
        }
        if let (Some(upload), Some(uniform), Some(radiance), Some(lut), Some(owner)) = (
            self.specular_upload_slot,
            self.specular_view_uniform_slot,
            self.specular_radiance_slot,
            self.specular_brdf_lut_slot,
            frame.specular.as_ref(),
        ) {
            owner.validate_for_sampling(self.id.device)?;
            for copy in &owner.radiance_copies {
                let end = copy
                    .source_offset
                    .checked_add(u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image))
                    .ok_or(Error::InvalidRange)?;
                pass.buffer(
                    upload,
                    copy.source_offset..end,
                    AccessKind::Read,
                    copy_read_uses(),
                )?;
                pass.texture(
                    radiance,
                    one_color_mip(copy.mip),
                    AccessKind::Write,
                    texture_copy_write_uses(),
                )?;
            }
            let copy = &owner.brdf_lut_copy;
            let end = copy
                .source_offset
                .checked_add(u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image))
                .ok_or(Error::InvalidRange)?;
            pass.buffer(
                upload,
                copy.source_offset..end,
                AccessKind::Read,
                copy_read_uses(),
            )?;
            pass.texture(lut, whole(), AccessKind::Write, texture_copy_write_uses())?;
            pass.buffer(
                upload,
                owner.view_uniform_copy.clone(),
                AccessKind::Read,
                copy_read_uses(),
            )?;
            pass.buffer(uniform, 0..112, AccessKind::Write, copy_write_uses())?;
        }
        pass.transition(encoder)?;
        for upload in &frame.resident_upload.vertices {
            let src = pass.buffer(
                self.resident_upload_slot,
                upload.source.clone(),
                AccessKind::Read,
                copy_read_uses(),
            )?;
            let dst = pass.buffer(
                *self
                    .stream_slots
                    .get(upload.stream)
                    .ok_or(Error::InvalidRange)?,
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
        if let (Some(slot), Some(owner)) =
            (self.environment_upload_slot, frame.environment.as_ref())
        {
            let src = pass.buffer(
                slot,
                0..owner.upload_desc.size,
                AccessKind::Read,
                copy_read_uses(),
            )?;
            let texture = pass.texture(
                self.environment_slot.ok_or(Error::InvalidState)?,
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
        if let (Some(upload), Some(uniform), Some(radiance), Some(lut), Some(owner)) = (
            self.specular_upload_slot,
            self.specular_view_uniform_slot,
            self.specular_radiance_slot,
            self.specular_brdf_lut_slot,
            frame.specular.as_ref(),
        ) {
            for copy in &owner.radiance_copies {
                let end = copy
                    .source_offset
                    .checked_add(u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image))
                    .ok_or(Error::InvalidRange)?;
                let src = pass.buffer(
                    upload,
                    copy.source_offset..end,
                    AccessKind::Read,
                    copy_read_uses(),
                )?;
                let texture = pass.texture(
                    radiance,
                    one_color_mip(copy.mip),
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
                        mip: copy.mip,
                        origin: [0, 0, 0],
                        extent: copy.extent,
                        aspect: TextureAspects::COLOR,
                    },
                )?;
            }
            let copy = &owner.brdf_lut_copy;
            let end = copy
                .source_offset
                .checked_add(u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image))
                .ok_or(Error::InvalidRange)?;
            let src = pass.buffer(
                upload,
                copy.source_offset..end,
                AccessKind::Read,
                copy_read_uses(),
            )?;
            let texture =
                pass.texture(lut, whole(), AccessKind::Write, texture_copy_write_uses())?;
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
            let src = pass.buffer(
                upload,
                owner.view_uniform_copy.clone(),
                AccessKind::Read,
                copy_read_uses(),
            )?;
            let dst = pass.buffer(uniform, 0..112, AccessKind::Write, copy_write_uses())?;
            encoder.copy_buffer(src, owner.view_uniform_copy.start, dst, 0, 112)?;
        }
        Ok(())
    }
}
