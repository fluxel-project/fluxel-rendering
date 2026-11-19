//! Per-pass native recording for Deferred-to-Forward capture.

use super::*;
use crate::graph::forward::ForwardReceiverTerminal;
use crate::{
    graph::{ForwardOitInput, oit_record::OitRecordCommands},
    residency::ForwardFrame,
};

impl<B: ResourceBackend> DeferredForwardCapturePlan<B> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_pass(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        producer: &DeferredFrame<B>,
        overlay: Option<&super::capture_producer::DeferredCaptureOverlay<'_, B>>,
        producer_commands: Option<&OitRecordCommands<'_, B>>,
        receiver: &ForwardFrame<B>,
        receiver_oit: Option<&ForwardOitInput<'_, B>>,
        receiver_commands: Option<&OitRecordCommands<'_, B>>,
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
            overlay,
            receiver,
            receiver_oit,
            image,
            composite,
        )?;
        super::capture_producer::validate_capture_producer_commands(
            self.producer_composition.as_ref(),
            producer,
            overlay,
            producer_commands,
        )?;
        self.receiver.validate_oit_input_commands(
            device.id(),
            receiver,
            receiver_oit,
            receiver_commands,
        )?;
        let bindings = self.bindings(producer, overlay, receiver, receiver_oit, image)?;
        if super::capture_record::record_namespace(
            &self.producer,
            encoder,
            producer,
            pass,
            &bindings,
        )? {
            return Ok(None);
        }
        if let (Some(composition), Some(overlay)) = (self.producer_composition.as_ref(), overlay) {
            if super::capture_producer::record_capture_producer_composition(
                composition,
                encoder,
                producer,
                overlay,
                producer_commands,
                &bindings,
                pass,
            )? {
                return Ok(None);
            }
        }
        if self.receiver.record_pass(
            encoder,
            &bindings,
            receiver,
            self.receiver
                .scene_refraction()
                .then_some(&producer.gbuffer[2]),
            receiver_oit,
            receiver_commands,
            pass,
        )? {
            return Ok(None);
        }
        match self.terminal {
            ForwardReceiverTerminal::Offscreen { pass: terminal } if pass == terminal => self
                .receiver
                .record_final_readback(
                    encoder,
                    &bindings,
                    receiver,
                    pass,
                    self.receiver.final_slot(),
                )
                .map(Some),
            ForwardReceiverTerminal::HdrReadback(terminal) if pass == terminal.composite_pass => {
                let source = receiver_oit.map_or(receiver.frame.target_instance, |input| {
                    input.output.instance
                });
                terminal.record(
                    device,
                    &bindings,
                    encoder,
                    self.receiver.final_slot(),
                    source,
                    receiver,
                    composite.ok_or(Error::InvalidState)?,
                )?;
                Ok(None)
            }
            ForwardReceiverTerminal::HdrReadback(terminal) if pass == terminal.readback_pass => {
                self.receiver
                    .record_final_readback(encoder, &bindings, receiver, pass, terminal.output_slot)
                    .map(Some)
            }
            ForwardReceiverTerminal::Surface {
                pass: terminal,
                slot,
                extent,
                ..
            } if pass == terminal => {
                let image = image.ok_or(Error::InvalidState)?;
                let composite = composite.ok_or(Error::InvalidState)?;
                let checked = bindings.pass(pass)?;
                checked.texture(
                    self.receiver.final_slot(),
                    whole_color(),
                    AccessKind::Read,
                    sampled_read(),
                )?;
                checked.surface_texture(slot, whole_color(), AccessKind::Write, color_write())?;
                checked.transition(encoder)?;
                let colors = [Some(ColorAttachment {
                    view: AttachmentView::Surface(image),
                    resolve_target: None,
                    load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
                    store: StoreOp::Store,
                })];
                let mut raster = encoder.begin_raster(&RasterPassDesc {
                    label: Some("fluxel deferred forward capture surface"),
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
                raster.finish()?;
                Ok(None)
            }
            _ => Err(Error::InvalidState),
        }
    }
}
