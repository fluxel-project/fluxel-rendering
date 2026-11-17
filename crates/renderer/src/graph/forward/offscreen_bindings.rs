//! Actual owner collection for ordinary and OIT Forward Graph bindings.
use super::*;

impl<B: ResourceBackend> ForwardPlan<B> {
    pub(super) fn bindings<'a>(
        &'a self,
        frame: &'a ForwardFrame<B>,
    ) -> Result<FrameBindings<'a, B>> {
        if self.oit.is_some() {
            return Err(Error::InvalidState);
        }
        self.bindings_inner(frame, None, None)
    }
    pub(super) fn bindings_with_oit<'a>(
        &'a self,
        frame: &'a ForwardFrame<B>,
        input: &'a ForwardOitInput<'_, B>,
        image: Option<&'a SurfaceImage<B>>,
    ) -> Result<FrameBindings<'a, B>> {
        if self.oit.is_none() {
            return Err(Error::InvalidState);
        }
        self.bindings_inner(frame, Some(input), image)
    }
    fn bindings_inner<'a>(
        &'a self,
        frame: &'a ForwardFrame<B>,
        oit: Option<&'a ForwardOitInput<'_, B>>,
        image: Option<&'a SurfaceImage<B>>,
    ) -> Result<FrameBindings<'a, B>> {
        validate_resident_texture_devices(&frame.resident, self.id.device)?;
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
        let uses_buffer = |slot| {
            self.execution
                .logical()
                .buffer_bindings()
                .requirements()
                .iter()
                .any(|requirement| requirement.slot == slot)
        };
        if uses_buffer(self.resident_upload_slot) {
            buffers.push(BoundBuffer::new(
                self.resident_upload_slot,
                frame.resident.upload_instance,
                &frame.resident.upload,
                &frame.resident.upload_desc,
                frame.resident.upload_incoming,
            ));
        }
        if let (Some(slot), Some(owner)) =
            (self.environment_upload_slot, frame.environment.as_ref())
        {
            buffers.push(BoundBuffer::new(
                slot,
                owner.upload_instance,
                &owner.upload,
                &owner.upload_desc,
                owner.upload_incoming,
            ));
        }
        if let (Some(upload_slot), Some(uniform_slot), Some(owner)) = (
            self.specular_upload_slot,
            self.specular_view_uniform_slot,
            frame.specular.as_ref(),
        ) {
            buffers.push(BoundBuffer::new(
                upload_slot,
                owner.upload_instance,
                &owner.upload,
                &owner.upload_desc,
                owner.upload_incoming,
            ));
            buffers.push(BoundBuffer::new(
                uniform_slot,
                owner.view_uniform_instance,
                &owner.view_uniform,
                &owner.view_uniform_desc,
                owner.view_uniform_incoming,
            ));
        }
        for (slot, stream) in self
            .stream_slots
            .iter()
            .copied()
            .zip(&frame.resident.streams)
        {
            if !uses_buffer(slot) {
                continue;
            }
            buffers.push(BoundBuffer::new(
                slot,
                stream.instance,
                &stream.buffer,
                &stream.desc,
                stream.incoming,
            ));
        }
        if uses_buffer(self.readback_slot) {
            buffers.push(BoundBuffer::new(
                self.readback_slot,
                frame.frame.readback_instance,
                &frame.frame.readback,
                &frame.frame.readback_desc,
                frame.frame.readback_incoming,
            ));
        }
        let mut textures: Vec<_> =
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
                .chain(std::iter::once(BoundTexture::new(
                    self.target_slot,
                    frame.frame.target_instance,
                    &frame.frame.target,
                    &frame.frame.target_desc,
                    frame.frame.target_incoming,
                    frame.frame.target_layout,
                )))
                .chain(self.environment_slot.zip(frame.environment.as_ref()).map(
                    |(slot, owner)| {
                        BoundTexture::new(
                            slot,
                            owner.texture_instance,
                            &owner.texture,
                            &owner.desc,
                            owner.incoming,
                            owner.layout,
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
                .chain(
                    self.depth_slot
                        .zip(frame.frame.depth.as_ref())
                        .map(|(slot, owner)| {
                            BoundTexture::new(
                                slot,
                                owner.instance,
                                &owner.texture,
                                &owner.desc,
                                owner.incoming,
                                owner.layout,
                            )
                        }),
                )
                .collect();
        if let Some(input) = oit {
            let product = self.oit.as_ref().ok_or(Error::InvalidState)?;
            for (slots, side) in product.slots.ping_pong.iter().zip(&input.owners.ping_pong) {
                for (slot, owner) in [
                    (slots.min_max, &side.min_max),
                    (slots.front, &side.front),
                    (slots.back, &side.back),
                ] {
                    textures.push(BoundTexture::new(
                        slot,
                        owner.instance,
                        &owner.texture,
                        &owner.desc,
                        owner.incoming,
                        owner.layout,
                    ));
                }
            }
            for (slot, owner) in [
                (
                    product.slots.back_accumulator,
                    &input.owners.back_accumulator,
                ),
                (product.slots.output, input.output),
            ] {
                textures.push(BoundTexture::new(
                    slot,
                    owner.instance,
                    &owner.texture,
                    &owner.desc,
                    owner.incoming,
                    owner.layout,
                ));
            }
        }
        let surface_textures = match (self.oit.as_ref().and_then(|oit| oit.surface), image) {
            (None, None) => Vec::new(),
            (Some(surface), Some(image)) => vec![BoundSurfaceTexture::new(surface.slot, image)?],
            _ => return Err(Error::InvalidState),
        };
        FrameBindings::new_with_surface_textures(
            &self.execution,
            self.id.device,
            buffers,
            textures,
            surface_textures,
        )
        .map_err(binding_error)
    }
}
