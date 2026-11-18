//! Per-pass native recording for Forward-to-Deferred capture.

use super::*;
use crate::{
    graph::{oit_record::OitRecordCommands, ForwardOitInput},
    residency::ForwardFrame,
};

impl<B: ResourceBackend> ForwardDeferredCapturePlan<B> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_pass(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        producer: &ForwardFrame<B>,
        producer_oit: Option<&ForwardOitInput<'_, B>>,
        producer_commands: Option<&OitRecordCommands<'_, B>>,
        receiver: &DeferredFrame<B>,
        image: Option<&fluxel_rhi::api::SurfaceImage<B>>,
        composite: Option<&crate::renderer::SurfaceCompositeOwner<B>>,
        pass: GraphPass,
    ) -> Result<Option<fluxel_rhi::api::ReadbackTicket<B>>>
    where
        B: CommandBackend + RasterBackend + ReadbackBackend,
    {
        self.validate_frames(device, producer, producer_oit, receiver, image, composite)?;
        self.producer.validate_oit_input_commands(
            device.id(),
            producer,
            producer_oit,
            producer_commands,
        )?;
        let bindings = self.bindings(producer, producer_oit, receiver, image)?;
        if self.producer.record_pass(
            encoder,
            &bindings,
            producer,
            None,
            producer_oit,
            producer_commands,
            pass,
        )? {
            return Ok(None);
        }
        if super::capture_record::record_namespace(
            &self.receiver,
            encoder,
            receiver,
            pass,
            &bindings,
        )? {
            return Ok(None);
        }
        if self.receiver.composite_pass == Some(pass) {
            match (self.surface, image, composite) {
                (Some(_), Some(image), Some(composite)) => {
                    self.record_surface_composite(encoder, receiver, image, composite, &bindings)?
                }
                (None, None, None) => self
                    .receiver
                    .record_composite_with_bindings(encoder, receiver, &bindings)?,
                _ => return Err(Error::InvalidState),
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
            label: Some("fluxel forward deferred capture surface composite"),
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
