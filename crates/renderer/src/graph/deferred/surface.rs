//! Surface output route for the Deferred Graph.
//!
//! This only changes the terminal owner: the upload/geometry/lighting Graph
//! transactions are the same as the offscreen route and the lease is recorded
//! after that Graph has been compiled.

use crate::frame_data::specular_view::SPECULAR_VIEW_BYTES;
use super::surface_validate::validate_surface_inputs;
use super::*;

impl<B: ResourceBackend> DeferredPlan<B> {
    pub(crate) fn declare_surface_with<F>(
        device: &Device<B>,
        frame: &DeferredSurfaceFrame<B>,
        declare: F,
    ) -> Result<DeferredDeclaration>
    where
        F: FnOnce(
            &mut crate::DeferredPhaseContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        validate_surface_inputs(device, frame)?;
        Self::declare_body(
            device,
            super::record_view::DeferredRecordView::surface(frame),
            declare,
        )
    }

    pub(crate) fn validate_surface_frame(&self, frame: &DeferredSurfaceFrame<B>) -> Result<()> {
        validate_surface_inputs(&self.device, frame)?;
        super::validate_resident_texture_devices(&frame.resident, self.id.device)?;
        if !matches!(self.output, DeferredOutputPlan::Surface { .. })
            || frame.extent != self.extent
            || frame
                .draws
                .iter()
                .map(|draw| draw.transform_bytes)
                .collect::<Vec<_>>()
                != self.transform_bytes
            || self.stream_slots.len() != frame.resident.streams.len()
            || self.texture_slots.len() != frame.resident.textures.len()
            || self.deformation_pass.is_some() != frame.deformation.is_some()
            || self.composite_pass.is_none()
            || self.scene_color_slot.is_none()
        {
            return Err(Error::InvalidState);
        }
        if self.specular_upload_slot.is_some() != frame.specular.is_some()
            || self.specular_view_uniform_slot.is_some() != frame.specular.is_some()
            || self.specular_radiance_slot.is_some() != frame.specular.is_some()
            || self.specular_brdf_lut_slot.is_some() != frame.specular.is_some()
        {
            return Err(Error::InvalidState);
        }
        if let Some(specular) = &frame.specular {
            specular.validate_for_sampling(self.id.device)?;
        }
        Ok(())
    }

    pub(crate) fn validate_surface_frame_with_image(
        &self,
        frame: &DeferredSurfaceFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
    ) -> Result<()> {
        self.validate_surface_frame(frame)?;
        self.surface_bindings(frame, image).map(|_| ())
    }

    pub(super) fn surface_bindings<'a>(
        &'a self,
        frame: &'a DeferredSurfaceFrame<B>,
        image: &'a fluxel_rhi::api::SurfaceImage<B>,
    ) -> Result<FrameBindings<'a, B>> {
        super::validate_resident_texture_devices(&frame.resident, self.id.device)?;
        let uses = |slot| {
            self.execution
                .logical()
                .buffer_bindings()
                .requirements()
                .iter()
                .any(|item| item.slot == slot)
        };
        let mut buffers = vec![
            BoundBuffer::new(
                self.frame_upload_slot,
                frame.frame.upload_instance,
                &frame.frame.upload,
                &frame.frame.upload_desc,
                frame.frame.upload_incoming,
            ),
            BoundBuffer::new(
                self.uniform_slot,
                frame.frame.uniform_instance,
                &frame.frame.uniform,
                &frame.frame.uniform_desc,
                frame.frame.uniform_incoming,
            ),
        ];
        if uses(self.resident_upload_slot) {
            buffers.push(BoundBuffer::new(
                self.resident_upload_slot,
                frame.resident.upload_instance,
                &frame.resident.upload,
                &frame.resident.upload_desc,
                frame.resident.upload_incoming,
            ));
        }
        if let (Some(slot), Some(environment)) = (self.environment_upload_slot, &frame.environment)
        {
            if uses(slot) {
                buffers.push(BoundBuffer::new(
                    slot,
                    environment.upload_instance,
                    &environment.upload,
                    &environment.upload_desc,
                    environment.upload_incoming,
                ));
            }
        }
        if let (Some(upload_slot), Some(uniform_slot), Some(specular)) = (
            self.specular_upload_slot,
            self.specular_view_uniform_slot,
            &frame.specular,
        ) {
            if uses(upload_slot) {
                buffers.push(BoundBuffer::new(
                    upload_slot,
                    specular.upload_instance,
                    &specular.upload,
                    &specular.upload_desc,
                    specular.upload_incoming,
                ));
            }
            if uses(uniform_slot) {
                buffers.push(BoundBuffer::new(
                    uniform_slot,
                    specular.view_uniform_instance,
                    &specular.view_uniform,
                    &specular.view_uniform_desc,
                    specular.view_uniform_incoming,
                ));
            }
        }
        for (slot, stream) in self
            .stream_slots
            .iter()
            .copied()
            .zip(&frame.resident.streams)
        {
            if uses(slot) {
                buffers.push(BoundBuffer::new(
                    slot,
                    stream.instance,
                    &stream.buffer,
                    &stream.desc,
                    stream.incoming,
                ));
            }
        }
        if let (Some(slots), Some(deformation)) = (self.deformation_slots, &frame.deformation) {
            for (slot, owner) in [
                (slots[0], &deformation.upload),
                (slots[1], &deformation.source),
                (slots[2], &deformation.product),
                (slots[3], &deformation.parameters),
            ] {
                if uses(slot) {
                    buffers.push(BoundBuffer::new(
                        slot,
                        owner.instance,
                        &owner.buffer,
                        &owner.desc,
                        owner.incoming,
                    ));
                }
            }
        }
        let textures =
            self.texture_slots
                .iter()
                .copied()
                .zip(&frame.resident.textures)
                .map(|(slot, texture)| {
                    BoundTexture::new(
                        slot,
                        texture.instance,
                        &texture.texture,
                        &texture.desc,
                        texture.incoming,
                        texture.layout,
                    )
                })
                .chain(self.gbuffer_slots.iter().copied().zip(&frame.gbuffer).map(
                    |(slot, texture)| {
                        BoundTexture::new(
                            slot,
                            texture.instance,
                            &texture.texture,
                            &texture.desc,
                            texture.incoming,
                            texture.layout,
                        )
                    },
                ))
                .chain(
                    self.effects_slot
                        .zip(frame.effects.as_ref())
                        .map(|(slot, texture)| {
                            BoundTexture::new(
                                slot,
                                texture.instance,
                                &texture.texture,
                                &texture.desc,
                                texture.incoming,
                                texture.layout,
                            )
                        }),
                )
                .chain(self.environment_slot.zip(frame.environment.as_ref()).map(
                    |(slot, environment)| {
                        BoundTexture::new(
                            slot,
                            environment.texture_instance,
                            &environment.texture,
                            &environment.desc,
                            environment.incoming,
                            environment.layout,
                        )
                    },
                ))
                .chain(
                    self.specular_radiance_slot
                        .zip(frame.specular.as_ref())
                        .map(|(slot, owner)| {
                            BoundTexture::new(
                                slot,
                                owner.radiance.instance,
                                &owner.radiance.texture,
                                &owner.radiance.desc,
                                owner.radiance.incoming,
                                owner.radiance.layout,
                            )
                        }),
                )
                .chain(
                    self.specular_brdf_lut_slot
                        .zip(frame.specular.as_ref())
                        .map(|(slot, owner)| {
                            BoundTexture::new(
                                slot,
                                owner.brdf_lut.instance,
                                &owner.brdf_lut.texture,
                                &owner.brdf_lut.desc,
                                owner.brdf_lut.incoming,
                                owner.brdf_lut.layout,
                            )
                        }),
                )
                .chain(std::iter::once(BoundTexture::new(
                    self.scene_color_slot.ok_or(Error::InvalidState)?,
                    frame.scene_color.instance,
                    &frame.scene_color.texture,
                    &frame.scene_color.desc,
                    frame.scene_color.incoming,
                    frame.scene_color.layout,
                )))
                .collect();
        let surface_target = self.surface_target_slot.ok_or(Error::InvalidState)?;
        let lease = BoundSurfaceTexture::new(surface_target, image)?;
        FrameBindings::new_with_surface_textures(
            &self.execution,
            self.id.device,
            buffers,
            textures,
            vec![lease],
        )
        .map_err(binding_error)
    }

    pub(crate) fn surface_record_lighting(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        if image.device() != self.id.device {
            return Err(Error::WrongDevice);
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated);
        }
        if image.extent() != frame.extent || image.format() != frame.color_format {
            return Err(Error::InvalidRange);
        }
        let bindings = self.surface_bindings(frame, image)?;
        let pass = bindings.pass(self.lighting_pass)?;
        for (slot, range) in [
            (self.gbuffer_slots[0], whole_color()),
            (self.gbuffer_slots[1], whole_color()),
            (self.gbuffer_slots[2], whole_depth()),
        ] {
            pass.texture(slot, range, AccessKind::Read, sampled_read())?;
        }
        if let (Some(slot), Some(_)) = (self.effects_slot, frame.effects.as_ref()) {
            pass.texture(slot, whole_color(), AccessKind::Read, sampled_read())?;
        }
        pass.texture(
            self.scene_color_slot.ok_or(Error::InvalidState)?,
            whole_color(),
            AccessKind::Write,
            color_write(),
        )?;
        if let (Some(slot), Some(_)) = (self.environment_slot, frame.environment.as_ref()) {
            pass.texture(slot, whole_color(), AccessKind::Read, sampled_read())?;
        }
        if let (Some(radiance), Some(lut), Some(view_uniform), Some(owner)) = (
            self.specular_radiance_slot,
            self.specular_brdf_lut_slot,
            self.specular_view_uniform_slot,
            frame.specular.as_ref(),
        ) {
            pass.texture(
                radiance,
                all_color_mips(owner.radiance.desc.mip_levels),
                AccessKind::Read,
                sampled_read(),
            )?;
            pass.texture(lut, one_color_mip(0), AccessKind::Read, sampled_read())?;
            pass.buffer(view_uniform, 0..SPECULAR_VIEW_BYTES, AccessKind::Read, uniform_read())?;
        }
        if let Some(range) = &frame.lighting_view {
            pass.buffer(
                self.uniform_slot,
                range.clone(),
                AccessKind::Read,
                lighting_uniform_read(),
            )?;
        }
        pass.transition(encoder)?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&frame.scene_color.view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
            store: StoreOp::Store,
        })];
        let mut raster = encoder.begin_raster(&RasterPassDesc {
            label: Some("fluxel deferred surface lighting"),
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
        raster.set_pipeline(&frame.lighting_pipeline)?;
        raster.set_bind_group(0, &frame.lighting_group, &[])?;
        if let Some(group) = &frame.lighting_environment_group {
            raster.set_bind_group(1, group, &[])?;
        }
        raster.draw(0..3, 0..1)?;
        raster.finish()
    }

    pub(crate) fn surface_record_composite(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        if image.device() != self.id.device {
            return Err(Error::WrongDevice);
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated);
        }
        if image.extent() != frame.extent || image.format() != frame.color_format {
            return Err(Error::InvalidRange);
        }
        let bindings = self.surface_bindings(frame, image)?;
        let pass = bindings.pass(self.composite_pass.ok_or(Error::InvalidState)?)?;
        pass.texture(
            self.scene_color_slot.ok_or(Error::InvalidState)?,
            whole_color(),
            AccessKind::Read,
            sampled_read(),
        )?;
        pass.surface_texture(
            self.surface_target_slot.ok_or(Error::InvalidState)?,
            whole_color(),
            AccessKind::Write,
            color_write(),
        )?;
        pass.transition(encoder)?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
            store: StoreOp::Store,
        })];
        let mut raster = encoder.begin_raster(&RasterPassDesc {
            label: Some("fluxel deferred surface scene-color composite"),
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
        raster.set_pipeline(&frame.composite_pipeline)?;
        raster.set_bind_group(0, &frame.composite_group, &[])?;
        raster.draw(0..3, 0..1)?;
        raster.finish()
    }

    // These entry points deliberately remain separate from the output route:
    // their Graph passes are shared with offscreen Deferred.  The Surface
    // recorder supplies the same concrete owners, so an absent deformation is
    // an ordinary no-op rather than a second Graph shape.
    pub(crate) fn surface_record_deformation(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.record_body_deformation(
            encoder,
            super::record_view::DeferredRecordView::surface_with_image(frame, image),
        )
    }

    pub(crate) fn surface_record_mip_generation(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
        generation: &super::super::mip_generation::MipGenerationPass,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.record_body_mip_generation(
            encoder,
            super::record_view::DeferredRecordView::surface_with_image(frame, image),
            generation,
        )
    }

    pub(crate) fn surface_record_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.record_body_upload(
            encoder,
            super::record_view::DeferredRecordView::surface_with_image(frame, image),
        )
    }
    pub(crate) fn surface_record_geometry(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        self.record_body_geometry(
            encoder,
            super::record_view::DeferredRecordView::surface_with_image(frame, image),
        )
    }
}
