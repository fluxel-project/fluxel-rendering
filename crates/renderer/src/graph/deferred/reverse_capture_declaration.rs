//! One actual Forward producer followed by a Deferred capture receiver.

use super::*;
use crate::{
    graph::{
        forward::{declare_forward_capture_namespace, SharedResidentImports},
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
    pub(crate) fn declare(
        device: &Device<B>,
        producer: &ForwardFrame<B>,
        producer_oit: Option<&ForwardOitInput<'_, B>>,
        receiver: &DeferredFrame<B>,
        image: Option<&fluxel_rhi::api::SurfaceImage<B>>,
    ) -> Result<ForwardDeferredCaptureDeclaration> {
        super::validation::validate_frame_inputs(device, receiver)?;
        if producer.extent != receiver.extent || receiver.deformation.is_some() {
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
        let mut graph = GraphBuilder::default();
        let producer = declare_forward_capture_namespace(
            device,
            &mut graph,
            producer,
            None,
            producer_oit,
            &mut shared,
        )
        .map_err(|_| Error::InvalidState)?;
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
