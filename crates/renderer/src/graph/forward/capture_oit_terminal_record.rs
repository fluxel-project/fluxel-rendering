//! Native terminal recording for one capture OIT Graph.

use super::shared_record::RecordSlots;
use super::*;

impl<B: ResourceBackend> CaptureOitPlan<B> {
    pub(super) fn slots<'a>(&self, slots: &'a CaptureFrameSlots) -> RecordSlots<'a> {
        RecordSlots {
            device: self.id.device,
            topology: FrameTopology::SortedForward,
            resident_upload_slot: slots.resident_upload,
            frame_upload_slot: slots.frame_upload,
            uniform_slot: slots.uniform,
            environment_upload_slot: slots.environment_upload,
            specular_upload_slot: slots.specular_upload,
            specular_view_uniform_slot: slots.specular_view_uniform,
            stream_slots: &slots.streams,
            texture_slots: &slots.textures,
            environment_slot: slots.environment,
            specular_radiance_slot: slots.specular_radiance,
            specular_brdf_lut_slot: slots.specular_brdf_lut,
            depth_slot: slots.depth,
            target_slot: slots.target,
            multisample_color_slot: None,
            readback_slot: slots.readback,
        }
    }

    pub(super) fn record_final_readback(
        &self,
        bindings: &FrameBindings<'_, B>,
        encoder: &mut CommandEncoder<B>,
        receiver: &ForwardFrame<B>,
        pass: GraphPass,
    ) -> Result<fluxel_rhi::api::ReadbackTicket<B>>
    where
        B: ReadbackBackend,
    {
        let mut slots = self.slots(&self.receiver);
        slots.target_slot = self.receiver_final;
        shared_record::record_readback(&slots, pass, bindings, encoder, receiver)
    }

    pub(super) fn record_hdr_terminal(
        &self,
        device: &Device<B>,
        bindings: &FrameBindings<'_, B>,
        encoder: &mut CommandEncoder<B>,
        receiver: &ForwardFrame<B>,
        receiver_input: Option<&ForwardOitInput<'_, B>>,
        terminal: super::hdr_output::HdrReadbackTerminal,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let instance = receiver_input.map_or(receiver.frame.target_instance, |input| {
            input.output.instance
        });
        terminal.record(
            device,
            bindings,
            encoder,
            self.receiver_final,
            instance,
            receiver,
            composite,
        )
    }

    pub(super) fn record_hdr_readback(
        &self,
        bindings: &FrameBindings<'_, B>,
        encoder: &mut CommandEncoder<B>,
        receiver: &ForwardFrame<B>,
        terminal: super::hdr_output::HdrReadbackTerminal,
    ) -> Result<fluxel_rhi::api::ReadbackTicket<B>>
    where
        B: ReadbackBackend,
    {
        let mut slots = self.slots(&self.receiver);
        slots.target_slot = terminal.output_slot;
        shared_record::record_readback(&slots, terminal.readback_pass, bindings, encoder, receiver)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_surface_composite(
        &self,
        bindings: &FrameBindings<'_, B>,
        encoder: &mut CommandEncoder<B>,
        receiver: &ForwardFrame<B>,
        receiver_input: Option<&ForwardOitInput<'_, B>>,
        image: &SurfaceImage<B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
        device: &Device<B>,
        surface: TextureSlot,
        extent: [u32; 2],
        format: TextureFormat,
        pass: GraphPass,
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
        if image.extent() != extent || image.format() != format || composite.extent() != extent {
            return Err(Error::InvalidRange);
        }
        composite.validate_for(device, image)?;
        let expected = receiver_input.map_or(receiver.frame.target_instance, |input| {
            input.output.instance
        });
        if composite.target_instance() != expected {
            return Err(Error::InvalidState);
        }
        let checked = bindings.pass(pass)?;
        checked.texture(
            self.receiver_final,
            whole(),
            AccessKind::Read,
            texture_read_uses(),
        )?;
        checked.surface_texture(surface, whole(), AccessKind::Write, color_write_uses())?;
        checked.transition(encoder)?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
        })];
        let mut raster = encoder.begin_raster(&RasterPassDesc {
            label: Some("fluxel capture OIT surface composite"),
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
}
