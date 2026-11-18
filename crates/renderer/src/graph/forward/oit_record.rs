//! Recording every phase of the one OIT Forward submission.

use super::shared_record::{self, RasterColorSelection, RasterDepthSelection, RasterSelection};
use super::*;
use crate::graph::oit_record::{OitDepthTargetRef, OitDrawSlots, OitRecordCommands};
use crate::renderer::SurfaceCompositeOwner;

impl<B: ResourceBackend> ForwardPlan<B> {
    pub(crate) fn record_upload_oit(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
        image: Option<&SurfaceImage<B>>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.validate_frame_with_oit(frame, input, image)?;
        shared_record::record_upload(
            &self.record_slots(),
            self.upload_pass,
            &self.bindings_with_oit(frame, input, image)?,
            encoder,
            frame,
        )
    }

    pub(crate) fn record_mip_oit(
        &self,
        index: usize,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
        image: Option<&SurfaceImage<B>>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.validate_frame_with_oit(frame, input, image)?;
        let generation = self.mip_passes.get(index).ok_or(Error::InvalidRange)?;
        let slot = *self
            .texture_slots
            .get(generation.texture)
            .ok_or(Error::InvalidRange)?;
        let owner = frame
            .resident
            .textures
            .get(generation.texture)
            .ok_or(Error::InvalidRange)?;
        crate::graph::mip_generation::record(
            generation,
            &self.bindings_with_oit(frame, input, image)?,
            slot,
            owner,
            encoder,
        )
    }

    pub(crate) fn record_opaque_oit(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
        image: Option<&SurfaceImage<B>>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        self.validate_frame_with_oit(frame, input, image)?;
        let product = self.oit.as_ref().ok_or(Error::InvalidState)?;
        let depth = frame.frame.depth.as_ref().ok_or(Error::InvalidState)?;
        shared_record::record_raster_selection(
            &self.record_slots(),
            self.raster_pass,
            &self.bindings_with_oit(frame, input, image)?,
            encoder,
            frame,
            RasterSelection {
                draws: 0..product.opaque_count,
                color: RasterColorSelection {
                    slot: self.target_slot,
                    texture: &frame.frame.target,
                    view: &frame.frame.target_view,
                    instance: frame.frame.target_instance,
                    load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
                    read_write: false,
                },
                depth: Some(RasterDepthSelection {
                    slot: self.depth_slot.ok_or(Error::InvalidState)?,
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

    pub(crate) fn record_oit_pass(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
        commands: &OitRecordCommands<'_, B>,
        pass: GraphPass,
        image: Option<&SurfaceImage<B>>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        self.validate_oit_commands(frame, input, commands, image)?;
        let product = self.oit.as_ref().ok_or(Error::InvalidState)?;
        let depth = OitDepthTargetRef::from(frame.frame.depth.as_ref().ok_or(Error::InvalidState)?);
        crate::graph::oit_record::record_pass(
            self.id.device,
            encoder,
            &self.bindings_with_oit(frame, input, image)?,
            &product.product,
            pass,
            &product.slots,
            input.owners,
            &depth,
            frame,
            &OitDrawSlots {
                streams: &self.stream_slots,
                uniform: self.uniform_slot,
                textures: &self.texture_slots,
                environment: self.environment_slot,
                specular_radiance: self.specular_radiance_slot,
                specular_brdf_lut: self.specular_brdf_lut_slot,
                specular_view_uniform: self.specular_view_uniform_slot,
            },
            commands,
        )
    }

    pub(crate) fn record_fallback_oit(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
        commands: &OitRecordCommands<'_, B>,
        image: Option<&SurfaceImage<B>>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        self.validate_oit_commands(frame, input, commands, image)?;
        let product = self.oit.as_ref().ok_or(Error::InvalidState)?;
        let depth = frame.frame.depth.as_ref().ok_or(Error::InvalidState)?;
        shared_record::record_raster_selection(
            &self.record_slots(),
            product.fallback.ok_or(Error::InvalidState)?,
            &self.bindings_with_oit(frame, input, image)?,
            encoder,
            frame,
            RasterSelection {
                draws: (product.opaque_count + product.peel_count)..frame.draws.len(),
                color: RasterColorSelection {
                    slot: product.slots.output,
                    texture: &input.output.texture,
                    view: &input.output.view,
                    instance: input.output.instance,
                    load: LoadOp::Load,
                    read_write: true,
                },
                depth: Some(RasterDepthSelection {
                    slot: self.depth_slot.ok_or(Error::InvalidState)?,
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

    pub(crate) fn record_readback_oit(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
        commands: &OitRecordCommands<'_, B>,
        image: Option<&SurfaceImage<B>>,
    ) -> Result<fluxel_rhi::api::ReadbackTicket<B>>
    where
        B: ReadbackBackend,
    {
        if image.is_some() {
            return Err(Error::InvalidState);
        }
        self.validate_oit_commands(frame, input, commands, image)?;
        let mut slots = self.record_slots();
        slots.target_slot = self.oit.as_ref().ok_or(Error::InvalidState)?.slots.output;
        shared_record::record_readback(
            &slots,
            self.readback_pass,
            &self.bindings_with_oit(frame, input, image)?,
            encoder,
            frame,
        )
    }

    pub(crate) fn record_hdr_oit_composite(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
        composite: &SurfaceCompositeOwner<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        self.validate_frame_with_oit(frame, input, None)?;
        let product = self.oit.as_ref().ok_or(Error::InvalidState)?;
        let terminal = product.hdr_readback.ok_or(Error::InvalidState)?;
        terminal.record(
            device,
            &self.bindings_with_oit(frame, input, None)?,
            encoder,
            product.slots.output,
            input.output.instance,
            frame,
            composite,
        )
    }

    pub(crate) fn record_hdr_oit_readback(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
    ) -> Result<fluxel_rhi::api::ReadbackTicket<B>>
    where
        B: ReadbackBackend,
    {
        self.validate_frame_with_oit(frame, input, None)?;
        let terminal = self
            .oit
            .as_ref()
            .and_then(|product| product.hdr_readback)
            .ok_or(Error::InvalidState)?;
        let mut slots = self.record_slots();
        slots.target_slot = terminal.output_slot;
        shared_record::record_readback(
            &slots,
            terminal.readback_pass,
            &self.bindings_with_oit(frame, input, None)?,
            encoder,
            frame,
        )
    }

    pub(crate) fn record_oit_surface_composite(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
        composite: &SurfaceCompositeOwner<B>,
        image: &SurfaceImage<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        if device.id() != self.id.device {
            return Err(Error::WrongDevice);
        }
        self.validate_frame_with_oit(frame, input, Some(image))?;
        let product = self.oit.as_ref().ok_or(Error::InvalidState)?;
        let surface = product.surface.ok_or(Error::InvalidState)?;
        if composite.extent() != surface.extent
            || composite.target_instance() != input.output.instance
        {
            return Err(Error::InvalidState);
        }
        composite.validate_for(device, image)?;
        let bindings = self.bindings_with_oit(frame, input, Some(image))?;
        let pass = bindings.pass(self.readback_pass)?;
        pass.texture(
            product.slots.output,
            whole(),
            AccessKind::Read,
            texture_read_uses(),
        )?;
        pass.surface_texture(surface.slot, whole(), AccessKind::Write, color_write_uses())?;
        pass.transition(encoder)?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
        })];
        let mut raster = encoder.begin_raster(&RasterPassDesc {
            label: Some("fluxel forward OIT surface composite"),
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: surface.extent[0],
                height: surface.extent[1],
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        raster.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: surface.extent[0] as f32,
            height: surface.extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        raster.scissor(Rect {
            x: 0,
            y: 0,
            width: surface.extent[0],
            height: surface.extent[1],
        })?;
        raster.set_pipeline(composite.pipeline())?;
        raster.set_bind_group(0, composite.bind_group(), &[])?;
        raster.draw(0..3, 0..1)?;
        raster.finish()
    }

    fn validate_oit_commands(
        &self,
        frame: &ForwardFrame<B>,
        input: &ForwardOitInput<'_, B>,
        commands: &OitRecordCommands<'_, B>,
        image: Option<&SurfaceImage<B>>,
    ) -> Result<()> {
        self.validate_frame_with_oit(frame, input, image)?;
        let product = self.oit.as_ref().ok_or(Error::InvalidState)?;
        let device = self.id.device;
        if commands.draws.iter().any(|draw| {
            !draw.pipelines_on(device)
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
        if commands.draws.len() != product.peel_count
            || commands
                .draws
                .iter()
                .enumerate()
                .any(|(index, draw)| draw.draw != product.opaque_count + index)
            || commands.output.instance != input.output.instance
            || !std::ptr::eq(commands.output.texture, &input.output.texture)
            || !std::ptr::eq(commands.output.view, &input.output.view)
            || commands.opaque_color.instance != frame.frame.target_instance
            || !std::ptr::eq(commands.opaque_color.texture, &frame.frame.target)
            || !std::ptr::eq(commands.opaque_color.view, &frame.frame.target_view)
        {
            return Err(Error::InvalidState);
        }
        Ok(())
    }
}
