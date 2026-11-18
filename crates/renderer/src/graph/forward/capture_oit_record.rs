//! Recording one already-compiled capture OIT Graph pass.

use super::shared_record::{self, RasterColorSelection, RasterDepthSelection, RasterSelection};
use super::*;
use crate::graph::oit_record::{OitDepthTargetRef, OitDrawSlots, OitRecordCommands};

impl<B: ResourceBackend> CaptureOitPlan<B> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_pass(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        capture_input: Option<&ForwardOitInput<'_, B>>,
        receiver_input: Option<&ForwardOitInput<'_, B>>,
        capture_commands: Option<&OitRecordCommands<'_, B>>,
        receiver_commands: Option<&OitRecordCommands<'_, B>>,
        composite: Option<&crate::renderer::SurfaceCompositeOwner<B>>,
        pass: GraphPass,
    ) -> Result<Option<fluxel_rhi::api::ReadbackTicket<B>>>
    where
        B: RasterBackend + CommandBackend + ReadbackBackend,
    {
        self.record_inner(
            encoder,
            capture,
            receiver,
            capture_input,
            receiver_input,
            capture_commands,
            receiver_commands,
            None,
            composite.map(|composite| (device, composite)),
            pass,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_surface_pass(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        capture_input: Option<&ForwardOitInput<'_, B>>,
        receiver_input: Option<&ForwardOitInput<'_, B>>,
        capture_commands: Option<&OitRecordCommands<'_, B>>,
        receiver_commands: Option<&OitRecordCommands<'_, B>>,
        image: &SurfaceImage<B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
        pass: GraphPass,
    ) -> Result<()>
    where
        B: RasterBackend + CommandBackend + ReadbackBackend,
    {
        if device.id() != self.id.device {
            return Err(Error::WrongDevice);
        }
        self.record_inner(
            encoder,
            capture,
            receiver,
            capture_input,
            receiver_input,
            capture_commands,
            receiver_commands,
            Some((image, composite)),
            Some((device, composite)),
            pass,
        )
        .and_then(|ticket| {
            if ticket.is_none() {
                Ok(())
            } else {
                Err(Error::InvalidState)
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn record_inner(
        &self,
        encoder: &mut CommandEncoder<B>,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        capture_input: Option<&ForwardOitInput<'_, B>>,
        receiver_input: Option<&ForwardOitInput<'_, B>>,
        capture_commands: Option<&OitRecordCommands<'_, B>>,
        receiver_commands: Option<&OitRecordCommands<'_, B>>,
        surface: Option<(&SurfaceImage<B>, &crate::renderer::SurfaceCompositeOwner<B>)>,
        hdr: Option<(&Device<B>, &crate::renderer::SurfaceCompositeOwner<B>)>,
        pass: GraphPass,
    ) -> Result<Option<fluxel_rhi::api::ReadbackTicket<B>>>
    where
        B: RasterBackend + CommandBackend + ReadbackBackend,
    {
        let bindings = match surface {
            Some((image, _)) => {
                self.bindings_with_image(capture, receiver, capture_input, receiver_input, image)?
            }
            None => self.bindings(capture, receiver, capture_input, receiver_input)?,
        };
        self.validate_command_sets(
            capture,
            receiver,
            capture_input,
            receiver_input,
            capture_commands,
            receiver_commands,
        )?;
        if pass == self.capture_upload {
            shared_record::record_upload(
                &self.slots(&self.capture),
                pass,
                &bindings,
                encoder,
                capture,
            )?;
            return Ok(None);
        }
        if pass == self.receiver_upload {
            shared_record::record_upload(
                &self.slots(&self.receiver),
                pass,
                &bindings,
                encoder,
                receiver,
            )?;
            return Ok(None);
        }
        if let Some(generation) = self
            .capture_mips
            .iter()
            .find(|generation| generation.pass == pass)
        {
            crate::graph::mip_generation::record(
                generation,
                &bindings,
                *self
                    .capture
                    .textures
                    .get(generation.texture)
                    .ok_or(Error::InvalidRange)?,
                capture
                    .resident
                    .textures
                    .get(generation.texture)
                    .ok_or(Error::InvalidRange)?,
                encoder,
            )?;
            return Ok(None);
        }
        if let Some(generation) = self
            .receiver_mips
            .iter()
            .find(|generation| generation.pass == pass)
        {
            crate::graph::mip_generation::record(
                generation,
                &bindings,
                *self
                    .receiver
                    .textures
                    .get(generation.texture)
                    .ok_or(Error::InvalidRange)?,
                receiver
                    .resident
                    .textures
                    .get(generation.texture)
                    .ok_or(Error::InvalidRange)?,
                encoder,
            )?;
            return Ok(None);
        }
        if let Some((background_copy, source)) = self.receiver_background_copy {
            if pass == background_copy {
                super::capture_terminal::record_background_copy(
                    &bindings,
                    pass,
                    encoder,
                    source,
                    self.receiver.target,
                    receiver.extent,
                )?;
                return Ok(None);
            }
        }
        if pass == self.capture_raster && self.capture_oit.is_some() {
            self.record_oit_opaque(
                &bindings,
                encoder,
                capture,
                &self.capture,
                self.capture_oit.as_ref().ok_or(Error::InvalidState)?,
                pass,
            )?;
            return Ok(None);
        }
        if pass == self.receiver_raster && self.receiver_oit.is_some() {
            self.record_oit_opaque(
                &bindings,
                encoder,
                receiver,
                &self.receiver,
                self.receiver_oit.as_ref().ok_or(Error::InvalidState)?,
                pass,
            )?;
            return Ok(None);
        }
        if pass == self.capture_raster {
            if self.scene_refraction {
                shared_record::record_raster_with_loads(
                    &self.slots(&self.capture),
                    pass,
                    &bindings,
                    encoder,
                    capture,
                    capture.frame.depth.as_ref(),
                    LoadOp::Clear(ClearColor::Float([0.0; 4])),
                    LoadOp::Clear(1.0),
                    StoreOp::Store,
                    false,
                )?;
                return Ok(None);
            }
            shared_record::record_raster(
                &self.slots(&self.capture),
                pass,
                &bindings,
                encoder,
                capture,
            )?;
            return Ok(None);
        }
        if pass == self.receiver_raster {
            if self.scene_refraction {
                shared_record::record_raster_with_loads(
                    &self.slots(&self.receiver),
                    pass,
                    &bindings,
                    encoder,
                    receiver,
                    capture.frame.depth.as_ref(),
                    LoadOp::Load,
                    LoadOp::Load,
                    StoreOp::Store,
                    true,
                )?;
                return Ok(None);
            }
            shared_record::record_raster(
                &self.slots(&self.receiver),
                pass,
                &bindings,
                encoder,
                receiver,
            )?;
            return Ok(None);
        }
        if let Some(oit) = self.capture_oit.as_ref() {
            if oit.passes().contains(&pass) {
                self.record_oit_stage(
                    &bindings,
                    encoder,
                    capture,
                    capture_input.ok_or(Error::InvalidState)?,
                    capture_commands.ok_or(Error::InvalidState)?,
                    &self.capture,
                    oit,
                    pass,
                )?;
                return Ok(None);
            }
            if oit.fallback == Some(pass) {
                self.record_oit_fallback(
                    &bindings,
                    encoder,
                    capture,
                    capture_input.ok_or(Error::InvalidState)?,
                    &self.capture,
                    oit,
                    pass,
                )?;
                return Ok(None);
            }
        }
        if let Some(oit) = self.receiver_oit.as_ref() {
            if oit.passes().contains(&pass) {
                self.record_oit_stage(
                    &bindings,
                    encoder,
                    receiver,
                    receiver_input.ok_or(Error::InvalidState)?,
                    receiver_commands.ok_or(Error::InvalidState)?,
                    &self.receiver,
                    oit,
                    pass,
                )?;
                return Ok(None);
            }
            if oit.fallback == Some(pass) {
                self.record_oit_fallback(
                    &bindings,
                    encoder,
                    receiver,
                    receiver_input.ok_or(Error::InvalidState)?,
                    &self.receiver,
                    oit,
                    pass,
                )?;
                return Ok(None);
            }
        }
        if matches!(self.terminal, super::capture_oit_declaration::CaptureOitTerminal::Readback { pass: terminal } if pass == terminal)
        {
            return self
                .record_final_readback(&bindings, encoder, receiver, pass)
                .map(Some);
        }
        if let super::capture_oit_declaration::CaptureOitTerminal::HdrReadback(terminal) =
            self.terminal
        {
            if pass == terminal.composite_pass {
                return self
                    .record_hdr_terminal(
                        hdr.ok_or(Error::InvalidState)?.0,
                        &bindings,
                        encoder,
                        receiver,
                        receiver_input,
                        terminal,
                        hdr.ok_or(Error::InvalidState)?.1,
                    )
                    .map(|()| None);
            }
            if pass == terminal.readback_pass {
                return self
                    .record_hdr_readback(&bindings, encoder, receiver, terminal)
                    .map(Some);
            }
        }
        if let (
            super::capture_oit_declaration::CaptureOitTerminal::Surface {
                pass: terminal,
                slot,
                extent,
                format,
            },
            Some((image, composite)),
            Some(device),
        ) = (self.terminal, surface, hdr.map(|(device, _)| device))
        {
            if pass == terminal {
                return self
                    .record_surface_composite(
                        &bindings,
                        encoder,
                        receiver,
                        receiver_input,
                        image,
                        composite,
                        device,
                        slot,
                        extent,
                        format,
                        pass,
                    )
                    .map(|()| None);
            }
        }
        Err(Error::InvalidState)
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_command_sets(
        &self,
        capture: &ForwardFrame<B>,
        receiver: &ForwardFrame<B>,
        capture_input: Option<&ForwardOitInput<'_, B>>,
        receiver_input: Option<&ForwardOitInput<'_, B>>,
        capture_commands: Option<&OitRecordCommands<'_, B>>,
        receiver_commands: Option<&OitRecordCommands<'_, B>>,
    ) -> Result<()> {
        validate_command_side(
            self.id.device,
            capture,
            self.capture_oit.as_ref(),
            capture_input,
            capture_commands,
        )?;
        validate_command_side(
            self.id.device,
            receiver,
            self.receiver_oit.as_ref(),
            receiver_input,
            receiver_commands,
        )
    }

    fn record_oit_opaque(
        &self,
        bindings: &FrameBindings<'_, B>,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        slots: &CaptureFrameSlots,
        oit: &super::oit_plan::ForwardOitPlan,
        pass: GraphPass,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let depth = frame.frame.depth.as_ref().ok_or(Error::InvalidState)?;
        shared_record::record_raster_selection(
            &self.slots(slots),
            pass,
            bindings,
            encoder,
            frame,
            RasterSelection {
                draws: 0..oit.opaque_count,
                color: RasterColorSelection {
                    slot: slots.target,
                    texture: &frame.frame.target,
                    view: &frame.frame.target_view,
                    instance: frame.frame.target_instance,
                    load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
                    read_write: false,
                },
                depth: Some(RasterDepthSelection {
                    slot: slots.depth.ok_or(Error::InvalidState)?,
                    texture: &depth.texture,
                    view: &depth.view,
                    instance: depth.instance,
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                    read_only: false,
                    read_write: false,
                }),
            },
        )
    }

    fn record_oit_stage(
        &self,
        bindings: &FrameBindings<'_, B>,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
        commands: &OitRecordCommands<'_, B>,
        slots: &CaptureFrameSlots,
        oit: &super::oit_plan::ForwardOitPlan,
        pass: GraphPass,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let depth = OitDepthTargetRef::from(frame.frame.depth.as_ref().ok_or(Error::InvalidState)?);
        crate::graph::oit_record::record_pass(
            self.id.device,
            encoder,
            bindings,
            &oit.product,
            pass,
            &oit.slots,
            input.owners,
            &depth,
            frame,
            &OitDrawSlots {
                streams: &slots.streams,
                uniform: slots.uniform,
                textures: &slots.textures,
                environment: slots.environment,
                specular_radiance: slots.specular_radiance,
                specular_brdf_lut: slots.specular_brdf_lut,
                specular_view_uniform: slots.specular_view_uniform,
            },
            commands,
        )
    }

    fn record_oit_fallback(
        &self,
        bindings: &FrameBindings<'_, B>,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
        slots: &CaptureFrameSlots,
        oit: &super::oit_plan::ForwardOitPlan,
        pass: GraphPass,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let depth = frame.frame.depth.as_ref().ok_or(Error::InvalidState)?;
        shared_record::record_raster_selection(
            &self.slots(slots),
            pass,
            bindings,
            encoder,
            frame,
            RasterSelection {
                draws: (oit.opaque_count + oit.peel_count)..frame.draws.len(),
                color: RasterColorSelection {
                    slot: oit.slots.output,
                    texture: &input.output.texture,
                    view: &input.output.view,
                    instance: input.output.instance,
                    load: LoadOp::Load,
                    read_write: true,
                },
                depth: Some(RasterDepthSelection {
                    slot: slots.depth.ok_or(Error::InvalidState)?,
                    texture: &depth.texture,
                    view: &depth.view,
                    instance: depth.instance,
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                    read_only: true,
                    read_write: false,
                }),
            },
        )
    }
}

fn validate_command_side<B: ResourceBackend>(
    device: DeviceId,
    frame: &ForwardFrame<B>,
    plan: Option<&super::oit_plan::ForwardOitPlan>,
    input: Option<&ForwardOitInput<'_, B>>,
    commands: Option<&OitRecordCommands<'_, B>>,
) -> Result<()> {
    let (plan, input, commands) = match (plan, input, commands) {
        (None, None, None) => return Ok(()),
        (Some(plan), Some(input), Some(commands)) => (plan, input, commands),
        _ => return Err(Error::InvalidState),
    };
    let candidate_end = plan
        .opaque_count
        .checked_add(plan.peel_count)
        .ok_or(Error::InvalidRange)?;
    if plan.peel_count == 0
        || candidate_end > frame.draws.len()
        || commands.draws.len() != plan.peel_count
        || commands
            .draws
            .iter()
            .enumerate()
            .any(|(offset, draw)| draw.draw != plan.opaque_count + offset)
        || commands.opaque_color.instance != frame.frame.target_instance
        || !std::ptr::eq(commands.opaque_color.texture, &frame.frame.target)
        || !std::ptr::eq(commands.opaque_color.view, &frame.frame.target_view)
        || commands.output.instance != input.output.instance
        || !std::ptr::eq(commands.output.texture, &input.output.texture)
        || !std::ptr::eq(commands.output.view, &input.output.view)
    {
        return Err(Error::InvalidState);
    }
    if commands.opaque_color.device != device
        || commands.opaque_color.texture.device() != device
        || commands.opaque_color.view.device() != device
        || commands.output.device != device
        || commands.output.texture.device() != device
        || commands.output.view.device() != device
    {
        return Err(Error::WrongDevice);
    }
    if commands.draws.iter().any(|draw| {
        draw.initial.device() != device
            || draw.peel.device() != device
            || draw.initial_group3.group.device() != device
            || draw
                .peel_group3
                .iter()
                .any(|group| group.group.device() != device)
            || draw
                .empty_group2
                .is_some_and(|group| group.device() != device)
    }) || std::iter::once(&commands.init_back)
        .chain(commands.back_accumulate.iter())
        .chain(commands.final_composite.iter())
        .any(|command| {
            command.pipeline.device() != device || command.bind_group.group.device() != device
        })
    {
        return Err(Error::WrongDevice);
    }
    Ok(())
}
