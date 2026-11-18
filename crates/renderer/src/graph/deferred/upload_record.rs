//! Upload recording for the Deferred Graph.

use super::*;
use crate::frame_data::specular_view::SPECULAR_VIEW_BYTES;

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
        self.record_body_upload_with_bindings(encoder, frame, &bindings)
    }

    pub(super) fn record_body_upload_with_bindings(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: super::record_view::DeferredRecordView<'_, B>,
        bindings: &FrameBindings<'_, B>,
    ) -> Result<()>
    where
        B: CommandBackend + ResourceBackend,
    {
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
                *self
                    .texture_slots
                    .get(copy.texture)
                    .ok_or(Error::InvalidRange)?,
                one_color_mip(copy.mip_level),
                AccessKind::Write,
                texture_copy_write(),
            )?;
        }
        if let (Some(upload_slot), Some(texture_slot), Some(environment)) = (
            self.environment_upload_slot,
            self.environment_slot,
            frame.environment.as_ref(),
        ) {
            pass.buffer(
                upload_slot,
                0..environment.upload_desc.size,
                AccessKind::Read,
                copy_read(),
            )?;
            pass.texture(
                texture_slot,
                whole_color(),
                AccessKind::Write,
                texture_copy_write(),
            )?;
        }
        if let (
            Some(upload_slot),
            Some(radiance_slot),
            Some(lut_slot),
            Some(uniform_slot),
            Some(owner),
        ) = (
            self.specular_upload_slot,
            self.specular_radiance_slot,
            self.specular_brdf_lut_slot,
            self.specular_view_uniform_slot,
            frame.specular,
        ) {
            for copy in &owner.radiance_copies {
                let end = copy
                    .source_offset
                    .checked_add(u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image))
                    .ok_or(Error::InvalidRange)?;
                pass.buffer(
                    upload_slot,
                    copy.source_offset..end,
                    AccessKind::Read,
                    copy_read(),
                )?;
                pass.texture(
                    radiance_slot,
                    one_color_mip(copy.mip),
                    AccessKind::Write,
                    texture_copy_write(),
                )?;
            }
            let copy = &owner.brdf_lut_copy;
            let end = copy
                .source_offset
                .checked_add(u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image))
                .ok_or(Error::InvalidRange)?;
            pass.buffer(
                upload_slot,
                copy.source_offset..end,
                AccessKind::Read,
                copy_read(),
            )?;
            pass.texture(
                lut_slot,
                one_color_mip(0),
                AccessKind::Write,
                texture_copy_write(),
            )?;
            pass.buffer(
                upload_slot,
                owner.view_uniform_copy.clone(),
                AccessKind::Read,
                copy_read(),
            )?;
            pass.buffer(
                uniform_slot,
                0..SPECULAR_VIEW_BYTES,
                AccessKind::Write,
                copy_write(),
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
                *self
                    .texture_slots
                    .get(copy.texture)
                    .ok_or(Error::InvalidRange)?,
                one_color_mip(copy.mip_level),
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
                    mip: copy.mip_level,
                    origin: [0, 0, 0],
                    extent: copy.extent,
                    aspect: TextureAspects::COLOR,
                },
            )?;
        }
        if let (
            Some(upload_slot),
            Some(radiance_slot),
            Some(lut_slot),
            Some(uniform_slot),
            Some(owner),
        ) = (
            self.specular_upload_slot,
            self.specular_radiance_slot,
            self.specular_brdf_lut_slot,
            self.specular_view_uniform_slot,
            frame.specular,
        ) {
            for copy in &owner.radiance_copies {
                let source = pass.buffer(
                    upload_slot,
                    copy.source_offset
                        ..copy.source_offset
                            + u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image),
                    AccessKind::Read,
                    copy_read(),
                )?;
                let texture = pass.texture(
                    radiance_slot,
                    one_color_mip(copy.mip),
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
                        texture,
                        mip: copy.mip,
                        origin: [0, 0, 0],
                        extent: copy.extent,
                        aspect: TextureAspects::COLOR,
                    },
                )?;
            }
            let copy = &owner.brdf_lut_copy;
            let source = pass.buffer(
                upload_slot,
                copy.source_offset
                    ..copy.source_offset
                        + u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image),
                AccessKind::Read,
                copy_read(),
            )?;
            let texture = pass.texture(
                lut_slot,
                one_color_mip(0),
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
                    texture,
                    mip: 0,
                    origin: [0, 0, 0],
                    extent: copy.extent,
                    aspect: TextureAspects::COLOR,
                },
            )?;
            let source = pass.buffer(
                upload_slot,
                owner.view_uniform_copy.clone(),
                AccessKind::Read,
                copy_read(),
            )?;
            let destination = pass.buffer(
                uniform_slot,
                0..SPECULAR_VIEW_BYTES,
                AccessKind::Write,
                copy_write(),
            )?;
            encoder.copy_buffer(
                source,
                owner.view_uniform_copy.start,
                destination,
                0,
                SPECULAR_VIEW_BYTES,
            )?;
        }
        if let (Some(upload_slot), Some(texture_slot), Some(environment)) = (
            self.environment_upload_slot,
            self.environment_slot,
            frame.environment.as_ref(),
        ) {
            let upload = pass.buffer(
                upload_slot,
                0..environment.upload_desc.size,
                AccessKind::Read,
                copy_read(),
            )?;
            let texture = pass.texture(
                texture_slot,
                whole_color(),
                AccessKind::Write,
                texture_copy_write(),
            )?;
            encoder.buffer_to_texture(
                upload,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: environment.bytes_per_row,
                    rows_per_image: environment.desc.extent[1],
                },
                TextureRegion {
                    texture,
                    mip: 0,
                    origin: [0, 0, 0],
                    extent: environment.desc.extent,
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
}
