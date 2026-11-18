//! Recording for the appended Forward capture receiver namespace.
use super::*;

impl ForwardReceiverComposition {
    fn record_slots(&self, device: DeviceId) -> super::shared_record::RecordSlots<'_> {
        let s = &self.slots;
        super::shared_record::RecordSlots {
            device,
            topology: FrameTopology::SortedForward,
            resident_upload_slot: s.resident_upload,
            frame_upload_slot: s.frame_upload,
            uniform_slot: s.uniform,
            environment_upload_slot: s.environment_upload,
            specular_upload_slot: s.specular_upload,
            specular_view_uniform_slot: s.specular_view_uniform,
            stream_slots: &s.streams,
            texture_slots: &s.textures,
            environment_slot: s.environment,
            specular_radiance_slot: s.specular_radiance,
            specular_brdf_lut_slot: s.specular_brdf_lut,
            depth_slot: s.depth,
            target_slot: s.target,
            readback_slot: s.readback,
        }
    }

    pub(in crate::graph) fn record_prefix_pass<B: ResourceBackend + CommandBackend>(
        &self,
        encoder: &mut CommandEncoder<B>,
        bindings: &FrameBindings<'_, B>,
        frame: &ForwardFrame<B>,
        pass: GraphPass,
    ) -> Result<bool> {
        if pass == self.upload {
            super::shared_record::record_upload(
                &self.record_slots(frame.resident.device),
                pass,
                bindings,
                encoder,
                frame,
            )?;
            return Ok(true);
        }
        if let Some(generation) = self.mip_generations.iter().find(|item| item.pass == pass) {
            let slot = *self
                .slots
                .textures
                .get(generation.texture)
                .ok_or(Error::InvalidRange)?;
            let owner = frame
                .resident
                .textures
                .get(generation.texture)
                .ok_or(Error::InvalidRange)?;
            crate::graph::mip_generation::record(generation, bindings, slot, owner, encoder)?;
            return Ok(true);
        }
        Ok(false)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::graph) fn record_pass<B: ResourceBackend + RasterBackend + CommandBackend>(
        &self,
        encoder: &mut CommandEncoder<B>,
        bindings: &FrameBindings<'_, B>,
        frame: &ForwardFrame<B>,
        producer_depth: Option<&crate::residency::ForwardTexture<B>>,
        input: Option<&ForwardOitInput<'_, B>>,
        commands: Option<&OitRecordCommands<'_, B>>,
        pass: GraphPass,
    ) -> Result<bool> {
        if self.oit.is_some() {
            self.validate_oit_input_commands(frame.resident.device, frame, input, commands)?;
        }
        if self.record_prefix_pass(encoder, bindings, frame, pass)? {
            return Ok(true);
        }
        if let Some((background_copy, source)) = self.background_copy {
            if pass == background_copy {
                super::capture_terminal::record_background_copy(
                    bindings,
                    pass,
                    encoder,
                    source,
                    self.slots.target,
                    frame.extent,
                )?;
                return Ok(true);
            }
        }
        let slots = self.record_slots(frame.resident.device);
        let depth = frame.frame.depth.as_ref().ok_or(Error::InvalidState)?;
        if pass == self.raster {
            if let Some(oit) = &self.oit {
                super::shared_record::record_raster_selection(
                    &slots,
                    pass,
                    bindings,
                    encoder,
                    frame,
                    super::shared_record::RasterSelection {
                        draws: 0..oit.opaque_count,
                        color: super::shared_record::RasterColorSelection {
                            slot: self.slots.target,
                            texture: &frame.frame.target,
                            view: &frame.frame.target_view,
                            instance: frame.frame.target_instance,
                            load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
                            read_write: false,
                        },
                        depth: Some(super::shared_record::RasterDepthSelection {
                            slot: self.slots.depth.ok_or(Error::InvalidState)?,
                            texture: &depth.texture,
                            view: &depth.view,
                            instance: depth.instance,
                            load: LoadOp::Clear(1.0),
                            store: StoreOp::Store,
                            read_only: false,
                            read_write: false,
                        }),
                    },
                )?;
            } else if self.scene_refraction {
                let producer_depth = producer_depth.ok_or(Error::InvalidState)?;
                super::shared_record::record_raster_selection(
                    &slots,
                    pass,
                    bindings,
                    encoder,
                    frame,
                    super::shared_record::RasterSelection {
                        draws: 0..frame.draws.len(),
                        color: super::shared_record::RasterColorSelection {
                            slot: self.slots.target,
                            texture: &frame.frame.target,
                            view: &frame.frame.target_view,
                            instance: frame.frame.target_instance,
                            load: LoadOp::Load,
                            read_write: true,
                        },
                        depth: Some(super::shared_record::RasterDepthSelection {
                            slot: self.slots.depth.ok_or(Error::InvalidState)?,
                            texture: &producer_depth.texture,
                            view: &producer_depth.view,
                            instance: producer_depth.instance,
                            load: LoadOp::Load,
                            store: StoreOp::Store,
                            read_only: true,
                            read_write: false,
                        }),
                    },
                )?;
            } else {
                super::shared_record::record_raster(&slots, pass, bindings, encoder, frame)?;
            }
            return Ok(true);
        }
        let (oit, input, commands) = match (&self.oit, input, commands) {
            (Some(oit), Some(input), Some(commands)) => (oit, input, commands),
            (None, _, _) => return Ok(false),
            _ => return Err(Error::InvalidState),
        };
        if oit.passes().contains(&pass) {
            crate::graph::oit_record::record_pass(
                frame.resident.device,
                encoder,
                bindings,
                &oit.product,
                pass,
                &oit.slots,
                input.owners,
                &OitDepthTargetRef::from(depth),
                frame,
                &OitDrawSlots {
                    streams: &self.slots.streams,
                    uniform: self.slots.uniform,
                    textures: &self.slots.textures,
                    environment: self.slots.environment,
                    specular_radiance: self.slots.specular_radiance,
                    specular_brdf_lut: self.slots.specular_brdf_lut,
                    specular_view_uniform: self.slots.specular_view_uniform,
                },
                commands,
            )?;
            return Ok(true);
        }
        if oit.fallback == Some(pass) {
            super::shared_record::record_raster_selection(
                &slots,
                pass,
                bindings,
                encoder,
                frame,
                super::shared_record::RasterSelection {
                    draws: (oit.opaque_count + oit.peel_count)..frame.draws.len(),
                    color: super::shared_record::RasterColorSelection {
                        slot: oit.slots.output,
                        texture: &input.output.texture,
                        view: &input.output.view,
                        instance: input.output.instance,
                        load: LoadOp::Load,
                        read_write: true,
                    },
                    depth: Some(super::shared_record::RasterDepthSelection {
                        slot: self.slots.depth.ok_or(Error::InvalidState)?,
                        texture: &depth.texture,
                        view: &depth.view,
                        instance: depth.instance,
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                        read_only: true,
                        read_write: false,
                    }),
                },
            )?;
            return Ok(true);
        }
        Ok(false)
    }

    pub(in crate::graph) fn record_final_readback<B: ResourceBackend + ReadbackBackend>(
        &self,
        encoder: &mut CommandEncoder<B>,
        bindings: &FrameBindings<'_, B>,
        frame: &ForwardFrame<B>,
        pass: GraphPass,
        source: TextureSlot,
    ) -> Result<fluxel_rhi::api::ReadbackTicket<B>> {
        let mut slots = self.record_slots(frame.resident.device);
        slots.target_slot = source;
        super::shared_record::record_readback(&slots, pass, bindings, encoder, frame)
    }
}
