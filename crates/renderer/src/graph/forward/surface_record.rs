//! Surface Record for the bounded Forward product.
use super::*;
use crate::frame_data::specular_view::SPECULAR_VIEW_BYTES;

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
        surface_raster_record::record(self, None, encoder, frame, image)
    }

    pub(crate) fn record_raster_indirect(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        frame: &SurfaceForwardFrame<B>,
        image: &SurfaceImage<B>,
    ) -> Result<()>
    where
        B: RasterBackend + ResourceBackend,
    {
        surface_raster_record::record(self, Some(device), encoder, frame, image)
    }

    pub(crate) fn record_indirect_upload(
        &self,
        index: usize,
        encoder: &mut CommandEncoder<B>,
        frame: &SurfaceForwardFrame<B>,
        image: &SurfaceImage<B>,
    ) -> Result<()>
    where
        B: CommandBackend + ResourceBackend,
    {
        let declared = self.indirect.get(index).ok_or(Error::InvalidRange)?;
        if frame.indirect.get(index).is_none() {
            return Err(Error::InvalidRange);
        }
        declared
            .graph
            .record_upload(encoder, &self.bindings(frame, image)?)
    }

    pub(crate) fn record_indirect_compute(
        &self,
        device: &Device<B>,
        index: usize,
        encoder: &mut CommandEncoder<B>,
        frame: &SurfaceForwardFrame<B>,
        image: &SurfaceImage<B>,
    ) -> Result<()>
    where
        B: CommandBackend + ResourceBackend,
    {
        let declared = self.indirect.get(index).ok_or(Error::InvalidRange)?;
        let (_, product) = frame.indirect.get(index).ok_or(Error::InvalidRange)?;
        declared
            .graph
            .record_compute(device, encoder, &self.bindings(frame, image)?, product)
    }

    pub(crate) fn record_mip(
        &self,
        index: usize,
        encoder: &mut CommandEncoder<B>,
        frame: &SurfaceForwardFrame<B>,
        image: &SurfaceImage<B>,
    ) -> Result<()>
    where
        B: CommandBackend + ResourceBackend,
    {
        let generation = self.mip_passes.get(index).ok_or(Error::InvalidRange)?;
        let slot = *self
            .texture_slots
            .get(generation.texture)
            .ok_or(Error::InvalidRange)?;
        let texture = frame
            .resident
            .textures
            .get(generation.texture)
            .ok_or(Error::InvalidRange)?;
        super::super::mip_generation::record(
            generation,
            &self.bindings(frame, image)?,
            slot,
            texture,
            encoder,
        )
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
            let owner = frame
                .resident
                .textures
                .get(copy.texture)
                .ok_or(Error::InvalidRange)?;
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
            pass.texture(
                slot,
                material_mip_layer(&owner.desc, copy.mip_level, copy.array_layer),
                AccessKind::Write,
                texture_copy_write_uses(),
            )?;
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
            pass.buffer(
                uniform,
                0..SPECULAR_VIEW_BYTES,
                AccessKind::Write,
                copy_write_uses(),
            )?;
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
            let owner = frame
                .resident
                .textures
                .get(copy.texture)
                .ok_or(Error::InvalidRange)?;
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
                material_mip_layer(&owner.desc, copy.mip_level, copy.array_layer),
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
                    mip: copy.mip_level,
                    origin: [0, 0, copy.array_layer],
                    extent: copy.extent,
                    aspect: material_aspect(&owner.desc),
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
            let dst = pass.buffer(
                uniform,
                0..SPECULAR_VIEW_BYTES,
                AccessKind::Write,
                copy_write_uses(),
            )?;
            encoder.copy_buffer(
                src,
                owner.view_uniform_copy.start,
                dst,
                0,
                SPECULAR_VIEW_BYTES,
            )?;
        }
        Ok(())
    }
}
