//! Per-pass recording for the combined Deferred capture Graph.

use super::*;

impl<B: ResourceBackend> DeferredCapturePlan<B> {
    pub(crate) fn record_pass(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        producer: &DeferredFrame<B>,
        producer_overlay: Option<&super::capture_producer::DeferredCaptureOverlay<'_, B>>,
        oit_commands: Option<&crate::graph::oit_record::OitRecordCommands<'_, B>>,
        receiver: &DeferredFrame<B>,
        image: Option<&fluxel_rhi::api::SurfaceImage<B>>,
        composite: Option<&crate::renderer::SurfaceCompositeOwner<B>>,
        pass: GraphPass,
    ) -> Result<Option<fluxel_rhi::api::ReadbackTicket<B>>>
    where
        B: CommandBackend + RasterBackend + ReadbackBackend,
    {
        self.validate_frames(
            device,
            producer,
            producer_overlay,
            receiver,
            image,
            composite,
        )?;
        super::capture_producer::validate_capture_producer_commands(
            self.producer_composition.as_ref(),
            producer,
            producer_overlay,
            oit_commands,
        )?;
        let bindings = match image {
            Some(image) => self.surface_bindings(producer, producer_overlay, receiver, image)?,
            None => self.bindings(producer, producer_overlay, receiver)?,
        };
        if record_namespace(&self.producer, encoder, producer, pass, &bindings)? {
            return Ok(None);
        }
        if let (Some(composition), Some(overlay)) =
            (self.producer_composition.as_ref(), producer_overlay)
        {
            if super::capture_producer::record_capture_producer_composition(
                composition,
                encoder,
                producer,
                overlay,
                oit_commands,
                &bindings,
                pass,
            )? {
                return Ok(None);
            }
        }
        if record_namespace(&self.receiver, encoder, receiver, pass, &bindings)? {
            return Ok(None);
        }
        if self.receiver.composite_pass == Some(pass) {
            if let (Some(image), Some(composite)) = (image, composite) {
                self.record_surface_composite(encoder, receiver, image, composite, &bindings)?;
            } else {
                self.receiver
                    .record_composite_with_bindings(encoder, receiver, &bindings)?;
            }
            return Ok(None);
        }
        if matches!(self.receiver.output, DeferredOutputPlan::Offscreen { readback_pass } if readback_pass == pass)
        {
            return self
                .receiver
                .record_readback_with_bindings(encoder, receiver, &bindings)
                .map(Some);
        }
        Err(Error::InvalidState)
    }

    fn record_surface_composite(
        &self,
        encoder: &mut CommandEncoder<B>,
        receiver: &DeferredFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
        composite: &crate::renderer::SurfaceCompositeOwner<B>,
        bindings: &FrameBindings<'_, B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        let pass = bindings.pass(self.receiver.composite_pass.ok_or(Error::InvalidState)?)?;
        pass.texture(
            self.receiver.scene_color_slot.ok_or(Error::InvalidState)?,
            whole_color(),
            AccessKind::Read,
            sampled_read(),
        )?;
        pass.surface_texture(
            self.receiver
                .surface_target_slot
                .ok_or(Error::InvalidState)?,
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
            label: Some("fluxel deferred capture surface composite"),
            colors: &colors,
            depth_stencil: None,
            render_area: render_area(receiver.extent),
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        configure_raster(&mut raster, receiver.extent)?;
        raster.set_pipeline(composite.pipeline())?;
        raster.set_bind_group(0, composite.bind_group(), &[])?;
        raster.draw(0..3, 0..1)?;
        raster.finish()
    }
}

pub(super) fn record_namespace<B: ResourceBackend + CommandBackend + RasterBackend>(
    plan: &DeferredPlan<B>,
    encoder: &mut CommandEncoder<B>,
    frame: &DeferredFrame<B>,
    pass: GraphPass,
    bindings: &FrameBindings<'_, B>,
) -> Result<bool> {
    if pass == plan.upload_pass {
        plan.record_upload_with_bindings(encoder, frame, bindings)?;
        return Ok(true);
    }
    if let Some(g) = plan.mip_generation.iter().find(|x| x.pass == pass) {
        plan.record_mip_with_bindings(encoder, frame, g, bindings)?;
        return Ok(true);
    }
    if plan.deformation_pass == Some(pass) {
        return Err(Error::Unsupported("deferred capture deformation"));
    }
    if pass == plan.geometry_pass {
        plan.record_geometry_with_bindings(encoder, frame, bindings)?;
        return Ok(true);
    }
    if pass == plan.lighting_pass {
        plan.record_lighting_with_bindings(encoder, frame, bindings)?;
        return Ok(true);
    }
    Ok(false)
}
