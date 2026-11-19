//! Native recording for the terminal of a linear Forward HDR frame.

use super::*;

impl<B: ResourceBackend> ForwardPlan<B> {
    fn hdr_bindings<'a>(
        &'a self,
        frame: &'a ForwardFrame<B>,
        image: Option<&'a SurfaceImage<B>>,
    ) -> Result<FrameBindings<'a, B>> {
        match (self.hdr_terminal.as_ref(), image) {
            (Some(HdrForwardTerminal::Offscreen { .. }), None) => self.bindings(frame),
            (Some(HdrForwardTerminal::Surface { .. }), Some(image)) => {
                self.bindings_with_hdr_surface(frame, image)
            }
            _ => Err(Error::InvalidState),
        }
    }

    pub(crate) fn record_hdr_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        image: Option<&SurfaceImage<B>>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        shared_record::record_upload(
            &self.record_slots(),
            self.upload_pass,
            &self.hdr_bindings(frame, image)?,
            encoder,
            frame,
        )
    }

    pub(crate) fn record_hdr_mip(
        &self,
        index: usize,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        image: Option<&SurfaceImage<B>>,
    ) -> Result<()>
    where
        B: CommandBackend,
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
            &self.hdr_bindings(frame, image)?,
            slot,
            texture,
            encoder,
        )
    }

    pub(crate) fn record_hdr_raster(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        image: Option<&SurfaceImage<B>>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        shared_record::record_raster(
            &self.record_slots(),
            self.raster_pass,
            &self.hdr_bindings(frame, image)?,
            encoder,
            frame,
        )
    }

    pub(crate) fn record_hdr_raster_indirect(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        image: Option<&SurfaceImage<B>>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let bindings = self.hdr_bindings(frame, image)?;
        for (declared, (_, product)) in self.indirect.iter().zip(&frame.indirect) {
            declared
                .graph
                .validate_raster_read(self.raster_pass, &bindings, product)?;
        }
        shared_record::record_raster_with_loads_indirect(
            &self.record_slots(),
            self.raster_pass,
            &bindings,
            encoder,
            frame,
            frame.frame.depth.as_ref(),
            LoadOp::Clear(ClearColor::Float([0.0; 4])),
            LoadOp::Clear(1.0),
            StoreOp::Discard,
            false,
            device,
        )
    }

    pub(crate) fn record_hdr_indirect_upload(
        &self,
        index: usize,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        image: Option<&SurfaceImage<B>>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        let declared = self.indirect.get(index).ok_or(Error::InvalidRange)?;
        if frame.indirect.get(index).is_none() {
            return Err(Error::InvalidRange);
        }
        declared
            .graph
            .record_upload(encoder, &self.hdr_bindings(frame, image)?)
    }

    pub(crate) fn record_hdr_indirect_compute(
        &self,
        device: &Device<B>,
        index: usize,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        image: Option<&SurfaceImage<B>>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        let declared = self.indirect.get(index).ok_or(Error::InvalidRange)?;
        let (_, product) = frame.indirect.get(index).ok_or(Error::InvalidRange)?;
        declared
            .graph
            .record_compute(device, encoder, &self.hdr_bindings(frame, image)?, product)
    }

    pub(crate) fn record_hdr_offscreen_composite(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let (pass_id, output_slot) = match self.hdr_terminal.as_ref() {
            Some(HdrForwardTerminal::Offscreen {
                composite_pass,
                output_slot,
                ..
            }) => (*composite_pass, *output_slot),
            _ => return Err(Error::InvalidState),
        };
        let output = frame.frame.output.as_ref().ok_or(Error::InvalidState)?;
        if composite.target_instance() != frame.frame.target_instance
            || composite.extent() != frame.extent
        {
            return Err(Error::InvalidState);
        }
        composite.validate_offscreen(device, frame.extent, output.desc.format)?;
        let bindings = self.bindings(frame)?;
        let pass = bindings.pass(pass_id)?;
        pass.texture(
            self.target_slot,
            whole(),
            AccessKind::Read,
            texture_read_uses(),
        )?;
        pass.texture(output_slot, whole(), AccessKind::Write, color_write_uses())?;
        pass.transition(encoder)?;
        record_composite_raster(
            encoder,
            AttachmentView::Texture(&output.view),
            frame.extent,
            composite,
            "fluxel Forward HDR output composite",
        )
    }

    pub(crate) fn record_hdr_readback(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
    ) -> Result<fluxel_rhi::api::ReadbackTicket<B>>
    where
        B: ReadbackBackend,
    {
        let (pass, output_slot) = match self.hdr_terminal.as_ref() {
            Some(HdrForwardTerminal::Offscreen {
                readback_pass,
                output_slot,
                ..
            }) => (*readback_pass, *output_slot),
            _ => return Err(Error::InvalidState),
        };
        let mut slots = self.record_slots();
        slots.target_slot = output_slot;
        shared_record::record_readback(&slots, pass, &self.bindings(frame)?, encoder, frame)
    }

    pub(crate) fn record_hdr_surface_composite(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        frame: &ForwardFrame<B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
        image: &SurfaceImage<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let (pass_id, surface_slot, extent, format) = match self.hdr_terminal.as_ref() {
            Some(HdrForwardTerminal::Surface {
                composite_pass,
                surface_slot,
                extent,
                format,
            }) => (*composite_pass, *surface_slot, *extent, *format),
            _ => return Err(Error::InvalidState),
        };
        if composite.target_instance() != frame.frame.target_instance
            || composite.extent() != extent
            || image.format() != format
        {
            return Err(Error::InvalidState);
        }
        composite.validate_for(device, image)?;
        let bindings = self.bindings_with_hdr_surface(frame, image)?;
        let pass = bindings.pass(pass_id)?;
        pass.texture(
            self.target_slot,
            whole(),
            AccessKind::Read,
            texture_read_uses(),
        )?;
        pass.surface_texture(surface_slot, whole(), AccessKind::Write, color_write_uses())?;
        pass.transition(encoder)?;
        record_composite_raster(
            encoder,
            AttachmentView::Surface(image),
            extent,
            composite,
            "fluxel Forward HDR surface composite",
        )
    }
}

pub(super) fn record_composite_raster<B: ResourceBackend + RasterBackend>(
    encoder: &mut CommandEncoder<B>,
    view: AttachmentView<'_, B>,
    extent: [u32; 2],
    composite: &crate::renderer::SurfaceCompositeOwner<B>,
    label: &'static str,
) -> Result<()> {
    let colors = [Some(ColorAttachment {
        view,
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
        store: StoreOp::Store,
    })];
    let mut raster = encoder.begin_raster(&RasterPassDesc {
        label: Some(label),
        colors: &colors,
        depth_stencil: None,
        render_area: Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    })?;
    raster.viewport(Viewport {
        x: 0.0,
        y: 0.0,
        width: extent[0] as f32,
        height: extent[1] as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    })?;
    raster.scissor(Rect {
        x: 0,
        y: 0,
        width: extent[0],
        height: extent[1],
    })?;
    raster.set_pipeline(composite.pipeline())?;
    raster.set_bind_group(0, composite.bind_group(), &[])?;
    raster.draw(0..3, 0..1)?;
    raster.finish()
}
