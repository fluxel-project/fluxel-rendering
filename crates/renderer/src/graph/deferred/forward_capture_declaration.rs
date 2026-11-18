//! Declaration of one Deferred producer followed by an actual Forward receiver.

use super::*;
use crate::{
    graph::{
        forward::{
            declare_receiver_after_deferred, declare_receiver_terminal, ForwardCaptureReceiver,
            ForwardReceiverTerminal, SharedResidentImports,
        },
        ForwardOitInput,
    },
    residency::ForwardFrame,
};

pub(crate) struct DeferredForwardCaptureDeclaration {
    pub(super) producer: DeferredDeclaration,
    pub(super) producer_composition:
        Option<super::capture_producer::DeferredCaptureProducerComposition>,
    pub(super) receiver: ForwardCaptureReceiver,
    pub(super) terminal: ForwardReceiverTerminal,
}

impl<B: ResourceBackend> super::DeferredForwardCapturePlan<B> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn declare<F, C>(
        device: &Device<B>,
        producer: &DeferredFrame<B>,
        producer_overlay: Option<&super::capture_producer::DeferredCaptureOverlay<'_, B>>,
        receiver: &ForwardFrame<B>,
        receiver_oit: Option<&ForwardOitInput<'_, B>>,
        scene_refraction: bool,
        image: Option<&fluxel_rhi::api::SurfaceImage<B>>,
        producer_contribute: F,
        mut overlay_contribute: C,
        overlay_error: &mut Option<crate::PipelineError>,
    ) -> Result<DeferredForwardCaptureDeclaration>
    where
        F: FnOnce(
            &mut crate::DeferredPhaseContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
        C: FnMut(
            &mut crate::graph::ForwardOitReadContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        super::validation::validate_frame_inputs(device, producer)?;
        if producer.extent != receiver.extent || producer.deformation.is_some() {
            return Err(Error::InvalidState);
        }
        if let Some(image) = image {
            if image.device() != device.id() {
                return Err(Error::WrongDevice);
            }
            if !image.is_current() {
                return Err(Error::SurfaceOutdated);
            }
            if image.extent() != receiver.extent {
                return Err(Error::InvalidRange);
            }
        }
        let mut shared = SharedResidentImports::default();
        let mut prefix = DeferredPlan::declare_prefix_in_graph(
            device,
            super::record_view::DeferredRecordView::offscreen(producer),
            GraphBuilder::default(),
            None,
            Some(&mut shared),
            producer_contribute,
        )?;
        let composition = producer_overlay
            .map(|overlay| {
                super::capture_producer::declare_capture_producer_composition_with_reads(
                    device,
                    &mut prefix,
                    producer,
                    overlay,
                    Some(&mut shared),
                    &mut overlay_contribute,
                    overlay_error,
                )
            })
            .transpose()?;
        let producer_final = composition
            .as_ref()
            .map(super::capture_producer::DeferredCaptureProducerComposition::final_source)
            .unwrap_or(prefix.scene_color);
        let mut producer_declaration = prefix.clone().capture_namespace_declaration();
        let output = receiver.frame.output.as_ref();
        if image.is_none()
            && (receiver.frame.target_desc.format == TextureFormat::Rgba16Float) != output.is_some()
        {
            return Err(Error::InvalidState);
        }
        let receiver = declare_receiver_after_deferred(
            device,
            &mut prefix.graph,
            receiver,
            producer_final,
            prefix.gbuffer[2],
            scene_refraction,
            receiver_oit,
            &mut shared,
        )
        .map_err(|_| Error::InvalidState)?;
        let terminal = declare_receiver_terminal(&mut prefix.graph, &receiver, output, image)
            .map_err(|_| Error::InvalidState)?;
        // The Deferred namespace metadata deliberately remains a capture
        // projection, while its declaration owns the complete one-Graph
        // producer -> Forward receiver -> terminal schedule.
        producer_declaration.graph = prefix.graph;
        Ok(DeferredForwardCaptureDeclaration {
            producer: producer_declaration,
            producer_composition: composition,
            receiver,
            terminal,
        })
    }
}
