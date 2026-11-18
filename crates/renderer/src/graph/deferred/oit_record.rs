//! Recording dispatch for the single Deferred OIT Graph.

use super::*;
use crate::graph::oit_record::{OitColorTargetRef, OitDepthTargetRef, OitRecordCommands};
use crate::residency::ForwardFrame;

impl<B: ResourceBackend> DeferredOitPlan<B> {
    pub(crate) fn record_pass(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        opaque: &DeferredFrame<B>,
        overlay: &ForwardFrame<B>,
        input: &crate::graph::ForwardOitInput<'_, B>,
        commands: &OitRecordCommands<'_, B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
        pass: GraphPass,
    ) -> Result<Option<fluxel_rhi::api::ReadbackTicket<B>>>
    where
        B: CommandBackend + RasterBackend + ReadbackBackend,
    {
        self.validate_frames(device, opaque, overlay, input, composite)?;
        self.validate_commands(opaque, input, commands)?;
        let bindings = self.bindings(opaque, overlay, input)?;
        self.record_bound(
            encoder, opaque, overlay, input, commands, composite, pass, &bindings, None,
        )
    }

    pub(crate) fn record_surface_pass(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        opaque: &DeferredFrame<B>,
        overlay: &ForwardFrame<B>,
        input: &crate::graph::ForwardOitInput<'_, B>,
        commands: &OitRecordCommands<'_, B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
        pass: GraphPass,
    ) -> Result<()>
    where
        B: CommandBackend + RasterBackend + ReadbackBackend,
    {
        self.validate_frames_with_image(device, opaque, overlay, input, image, composite)?;
        self.validate_commands(opaque, input, commands)?;
        let bindings = self.surface_bindings(opaque, overlay, input, image)?;
        self.record_bound(
            encoder,
            opaque,
            overlay,
            input,
            commands,
            composite,
            pass,
            &bindings,
            Some(image),
        )
        .and_then(|ticket| match ticket {
            None => Ok(()),
            Some(_) => Err(Error::InvalidState),
        })
    }

    fn validate_commands(
        &self,
        opaque: &DeferredFrame<B>,
        input: &crate::graph::ForwardOitInput<'_, B>,
        commands: &OitRecordCommands<'_, B>,
    ) -> Result<()> {
        let device = self.base.id.device;
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
        if commands.draws.len() != self.peel_count
            || commands
                .draws
                .iter()
                .enumerate()
                .any(|(i, draw)| draw.draw != i)
            || commands.output.instance != input.output.instance
            || !std::ptr::eq(commands.output.texture, &input.output.texture)
            || !std::ptr::eq(commands.output.view, &input.output.view)
            || commands.opaque_color.instance != opaque.scene_color.instance
            || !std::ptr::eq(commands.opaque_color.texture, &opaque.scene_color.texture)
            || !std::ptr::eq(commands.opaque_color.view, &opaque.scene_color.view)
        {
            Err(Error::InvalidState)
        } else {
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_bound<'a>(
        &self,
        encoder: &mut CommandEncoder<B>,
        opaque: &'a DeferredFrame<B>,
        overlay: &'a ForwardFrame<B>,
        input: &'a crate::graph::ForwardOitInput<'a, B>,
        commands: &'a OitRecordCommands<'a, B>,
        composite: &'a crate::renderer::SurfaceCompositeOwner<B>,
        pass: GraphPass,
        bindings: &FrameBindings<'a, B>,
        image: Option<&'a fluxel_rhi::api::SurfaceImage<B>>,
    ) -> Result<Option<fluxel_rhi::api::ReadbackTicket<B>>>
    where
        B: CommandBackend + RasterBackend + ReadbackBackend,
    {
        if pass == self.overlay_upload {
            self.overlay
                .record_upload(encoder, bindings, overlay, pass)?;
            return Ok(None);
        }
        if let Some(g) = self.overlay_mips.iter().find(|g| g.pass == pass) {
            self.overlay.record_mip(encoder, bindings, overlay, g)?;
            return Ok(None);
        }
        if pass == self.base.upload_pass {
            self.base
                .record_upload_with_bindings(encoder, opaque, bindings)?;
            return Ok(None);
        }
        if let Some(g) = self.base.mip_generation.iter().find(|g| g.pass == pass) {
            self.base
                .record_mip_with_bindings(encoder, opaque, g, bindings)?;
            return Ok(None);
        }
        if pass == self.base.geometry_pass {
            self.base
                .record_geometry_with_bindings(encoder, opaque, bindings)?;
            return Ok(None);
        }
        if pass == self.base.lighting_pass {
            self.base
                .record_lighting_with_bindings(encoder, opaque, bindings)?;
            return Ok(None);
        }
        if self.fallback == Some(pass) {
            self.overlay.record_raster_range(
                encoder,
                bindings,
                overlay,
                OitColorTargetRef {
                    texture: &input.output.texture,
                    view: &input.output.view,
                    desc: &input.output.desc,
                    instance: input.output.instance,
                    device: input.output.device,
                },
                self.final_slot,
                &opaque.gbuffer[2],
                self.peel_count..overlay.draws.len(),
                pass,
            )?;
            return Ok(None);
        }
        if self.is_product_pass(pass) {
            crate::graph::oit_record::record_pass(
                device_id(self),
                encoder,
                bindings,
                &self.product,
                pass,
                &self.slots,
                &input.owners,
                &OitDepthTargetRef::from(&opaque.gbuffer[2]),
                overlay,
                &self.overlay.draw_slots(),
                commands,
            )?;
            return Ok(None);
        }
        if self.base.composite_pass == Some(pass) {
            if let Some(image) = image {
                self.record_surface_composite(encoder, opaque, composite, bindings, image)?;
            } else {
                self.record_composite(encoder, opaque, composite, bindings)?;
            }
            return Ok(None);
        }
        if matches!(self.base.output, DeferredOutputPlan::Offscreen { readback_pass } if readback_pass == pass)
        {
            return self
                .base
                .record_readback_with_bindings(encoder, opaque, bindings)
                .map(Some);
        }
        Err(Error::InvalidState)
    }

    fn is_product_pass(&self, pass: GraphPass) -> bool {
        let p = &self.product.passes;
        pass == p.initial_clear[0].pass
            || pass == p.initial_clear[1].pass
            || pass == p.initialize_back_accumulator
            || pass == p.initial_depth
            || pass == p.final_composite
            || p.iterations
                .iter()
                .any(|x| pass == x.clear.pass || pass == x.peel || pass == x.back_accumulate)
    }

    fn record_composite(
        &self,
        encoder: &mut CommandEncoder<B>,
        opaque: &DeferredFrame<B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
        bindings: &FrameBindings<'_, B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let pass = bindings.pass(self.base.composite_pass.ok_or(Error::InvalidState)?)?;
        pass.texture(
            self.final_slot,
            whole_color(),
            AccessKind::Read,
            sampled_read(),
        )?;
        pass.texture(
            self.base.target_slot.ok_or(Error::InvalidState)?,
            whole_color(),
            AccessKind::Write,
            color_write(),
        )?;
        pass.transition(encoder)?;
        self.composite_raster(
            encoder,
            AttachmentView::Texture(&opaque.frame.target_view),
            opaque.extent,
            composite,
        )
    }

    fn record_surface_composite(
        &self,
        encoder: &mut CommandEncoder<B>,
        opaque: &DeferredFrame<B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
        bindings: &FrameBindings<'_, B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let pass = bindings.pass(self.base.composite_pass.ok_or(Error::InvalidState)?)?;
        pass.texture(
            self.final_slot,
            whole_color(),
            AccessKind::Read,
            sampled_read(),
        )?;
        pass.surface_texture(
            self.base.surface_target_slot.ok_or(Error::InvalidState)?,
            whole_color(),
            AccessKind::Write,
            color_write(),
        )?;
        pass.transition(encoder)?;
        self.composite_raster(
            encoder,
            AttachmentView::Surface(image),
            opaque.extent,
            composite,
        )
    }

    fn composite_raster<'a>(
        &self,
        encoder: &mut CommandEncoder<B>,
        view: AttachmentView<'a, B>,
        extent: [u32; 2],
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let colors = [Some(ColorAttachment {
            view,
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
            store: StoreOp::Store,
        })];
        let mut raster = encoder.begin_raster(&RasterPassDesc {
            label: Some("fluxel deferred oit composite"),
            colors: &colors,
            depth_stencil: None,
            render_area: render_area(extent),
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        configure_raster(&mut raster, extent)?;
        raster.set_pipeline(composite.pipeline())?;
        raster.set_bind_group(0, composite.bind_group(), &[])?;
        raster.draw(0..3, 0..1)?;
        raster.finish()
    }
}

fn device_id<B: ResourceBackend>(plan: &DeferredOitPlan<B>) -> fluxel_rhi::api::DeviceId {
    plan.base.id.device
}
