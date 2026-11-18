//! Concrete Deferred Graph owner bindings.

use super::*;

impl<B: ResourceBackend> DeferredPlan<B> {
    /// Collects exact ordinary Deferred owners so a combined Graph may append
    /// its own declared overlay owners before constructing FrameBindings.
    pub(super) fn collect_bindings<'a>(
        &'a self,
        frame: &'a DeferredFrame<B>,
    ) -> Result<(Vec<BoundBuffer<'a, B>>, Vec<BoundTexture<'a, B>>)> {
        super::validate_resident_texture_devices(&frame.resident, self.id.device)?;
        let uses_buffer = |slot| {
            self.execution
                .logical()
                .buffer_bindings()
                .requirements()
                .iter()
                .any(|requirement| requirement.slot == slot)
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
        if let Some(slot) = self.readback_slot.filter(|slot| uses_buffer(*slot)) {
            buffers.push(BoundBuffer::new(
                slot,
                frame.frame.readback_instance,
                &frame.frame.readback,
                &frame.frame.readback_desc,
                frame.frame.readback_incoming,
            ));
        }
        if uses_buffer(self.resident_upload_slot) {
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
            if uses_buffer(slot) {
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
            if uses_buffer(upload_slot) {
                buffers.push(BoundBuffer::new(
                    upload_slot,
                    specular.upload_instance,
                    &specular.upload,
                    &specular.upload_desc,
                    specular.upload_incoming,
                ));
            }
            if uses_buffer(uniform_slot) {
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
            if uses_buffer(slot) {
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
                if uses_buffer(slot) {
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
                .chain(
                    self.msaa_gbuffer_slots
                        .into_iter()
                        .flatten()
                        .zip(frame.msaa.iter().flat_map(|msaa| msaa.gbuffer.iter()))
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
                .chain(
                    self.msaa_effects_slot
                        .zip(frame.msaa.as_ref().and_then(|msaa| msaa.effects.as_ref()))
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
                .chain(self.target_slot.into_iter().map(|slot| {
                    BoundTexture::new(
                        slot,
                        frame.frame.target_instance,
                        &frame.frame.target,
                        &frame.frame.target_desc,
                        frame.frame.target_incoming,
                        frame.frame.target_layout,
                    )
                }))
                .collect();
        Ok((buffers, textures))
    }

    pub(super) fn bindings<'a>(
        &'a self,
        frame: &'a DeferredFrame<B>,
    ) -> Result<FrameBindings<'a, B>> {
        let (buffers, textures) = self.collect_bindings(frame)?;
        FrameBindings::new(&self.execution, self.id.device, buffers, textures)
            .map_err(binding_error)
    }
}
