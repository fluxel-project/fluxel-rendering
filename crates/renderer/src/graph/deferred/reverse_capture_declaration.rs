//! One actual Forward producer followed by a Deferred capture receiver.

use super::*;
use crate::{
    graph::{
        forward::{declare_forward_capture_namespace_with_contribution, SharedResidentImports},
        ForwardOitInput,
    },
    residency::ForwardFrame,
};

pub(crate) struct ForwardDeferredCaptureDeclaration {
    pub(super) producer: crate::graph::forward::ForwardCaptureReceiver,
    pub(super) receiver: DeferredDeclaration,
    pub(super) surface: Option<([u32; 2], TextureFormat)>,
}

impl<B: ResourceBackend> ForwardDeferredCapturePlan<B> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn declare_with_contribution<N, C>(
        device: &Device<B>,
        producer: &ForwardFrame<B>,
        producer_oit: Option<&ForwardOitInput<'_, B>>,
        receiver: &DeferredFrame<B>,
        image: Option<&fluxel_rhi::api::SurfaceImage<B>>,
        ordinary: N,
        contribute: &mut C,
    ) -> std::result::Result<ForwardDeferredCaptureDeclaration, crate::graph::ForwardPlanBuildError>
    where
        N: FnOnce(
            &mut crate::ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), crate::PipelineError>,
        C: FnMut(
            &mut crate::graph::ForwardOitReadContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        super::validation::validate_frame_inputs(device, receiver)?;
        if producer.extent != receiver.extent || receiver.deformation.is_some() {
            return Err(Error::InvalidState.into());
        }
        if let Some(image) = image {
            if image.device() != device.id() {
                return Err(Error::WrongDevice.into());
            }
            if !image.is_current() {
                return Err(Error::SurfaceOutdated.into());
            }
            if image.extent() != receiver.extent {
                return Err(Error::InvalidRange.into());
            }
        }
        let mut shared = SharedResidentImports::default();
        let mut graph = GraphBuilder::default();
        let producer = declare_forward_capture_namespace_with_contribution(
            device,
            &mut graph,
            producer,
            None,
            None,
            producer_oit,
            &mut shared,
            ordinary,
            contribute,
        )?;
        let aliases =
            super::capture_declaration::receiver_aliases(receiver, producer.final_handle)?;
        let mut prefix = DeferredPlan::declare_prefix_in_graph(
            device,
            super::record_view::DeferredRecordView::offscreen(receiver),
            graph,
            Some(&aliases),
            Some(&mut shared),
            |context| super::capture_declaration::builtin_receiver(context),
        )?;
        let surface = image.map(|image| (image.extent(), image.format()));
        if let Some((extent, format)) = surface {
            prefix.target = None;
            prefix.readback = None;
            prefix.row_pitch = None;
            prefix.surface_target = Some(
                prefix
                    .graph
                    .import_surface_texture(fluxel_rhi::api::TextureDesc {
                        label: Some("fluxel forward deferred capture surface lease".into()),
                        dimension: fluxel_rhi::api::TextureDimension::D2,
                        extent: [extent[0], extent[1], 1],
                        array_layers: 1,
                        format,
                        mip_levels: 1,
                        samples: 1,
                        usage: TextureUsage::RENDER_ATTACHMENT,
                        view_formats: Vec::new(),
                    })
                    .map_err(|_| Error::InvalidState)?,
            );
        }
        let receiver = super::terminal::declare_terminal(prefix, None)?;
        Ok(ForwardDeferredCaptureDeclaration {
            producer,
            receiver,
            surface,
        })
    }
}
