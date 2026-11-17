//! One actual Deferred producer followed by a Deferred capture receiver.

use super::*;
use crate::graph::forward::SharedResidentImports;

pub(crate) struct DeferredCaptureDeclaration {
    pub(super) producer: DeferredDeclaration,
    pub(super) producer_composition:
        Option<super::capture_producer::DeferredCaptureProducerComposition>,
    pub(super) receiver: DeferredDeclaration,
    pub(super) surface: Option<([u32; 2], TextureFormat)>,
}

impl<B: ResourceBackend> DeferredCapturePlan<B> {
    pub(crate) fn declare<F>(
        device: &Device<B>,
        producer: &DeferredFrame<B>,
        producer_overlay: Option<&super::capture_producer::DeferredCaptureOverlay<'_, B>>,
        receiver: &DeferredFrame<B>,
        image: Option<&fluxel_rhi::api::SurfaceImage<B>>,
        producer_contribute: F,
    ) -> Result<DeferredCaptureDeclaration>
    where
        F: FnOnce(
            &mut crate::DeferredPhaseContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        super::validation::validate_frame_inputs(device, producer)?;
        super::validation::validate_frame_inputs(device, receiver)?;
        if producer.extent != receiver.extent {
            return Err(Error::InvalidRange);
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
        let mut producer_prefix = DeferredPlan::declare_prefix_in_graph(
            device,
            super::record_view::DeferredRecordView::offscreen(producer),
            GraphBuilder::default(),
            None,
            Some(&mut shared),
            producer_contribute,
        )?;
        let producer_composition = producer_overlay
            .map(|overlay| {
                super::capture_producer::declare_capture_producer_composition(
                    device,
                    &mut producer_prefix,
                    producer,
                    overlay,
                    None,
                )
            })
            .transpose()?;
        let producer_final = producer_composition
            .as_ref()
            .map(super::capture_producer::DeferredCaptureProducerComposition::final_source)
            .unwrap_or(producer_prefix.scene_color);
        let producer_declaration = producer_prefix.clone().capture_namespace_declaration();
        let aliases = receiver_aliases(receiver, producer_final)?;
        let mut receiver_prefix = DeferredPlan::declare_prefix_in_graph(
            device,
            super::record_view::DeferredRecordView::offscreen(receiver),
            producer_prefix.graph,
            Some(&aliases),
            Some(&mut shared),
            |context| builtin_receiver(context),
        )?;
        let surface = image.map(|image| (image.extent(), image.format()));
        if let Some((extent, format)) = surface {
            receiver_prefix.target = None;
            receiver_prefix.readback = None;
            receiver_prefix.row_pitch = None;
            receiver_prefix.surface_target = Some(
                receiver_prefix
                    .graph
                    .import_surface_texture(fluxel_rhi::api::TextureDesc {
                        label: Some("fluxel deferred capture surface lease".into()),
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
        let receiver = super::terminal::declare_terminal(receiver_prefix, None)?;
        Ok(DeferredCaptureDeclaration {
            producer: producer_declaration,
            producer_composition,
            receiver,
            surface,
        })
    }
}

pub(super) fn receiver_aliases<B: ResourceBackend>(
    receiver: &DeferredFrame<B>,
    producer: crate::graph::definition::TextureHandle,
) -> Result<Vec<Option<crate::graph::definition::TextureHandle>>> {
    let mut aliases = vec![None; receiver.resident.textures.len()];
    let mut roles = vec![None; aliases.len()];
    for draw in &receiver.draws {
        if draw.textures.len() != draw.capture_textures.len() {
            return Err(Error::InvalidRange);
        }
        for (&index, &capture) in draw.textures.iter().zip(&draw.capture_textures) {
            let role = roles.get_mut(index).ok_or(Error::InvalidRange)?;
            if role.is_some_and(|old| old != capture) {
                return Err(Error::InvalidState);
            }
            *role = Some(capture);
            if capture {
                *aliases.get_mut(index).ok_or(Error::InvalidRange)? = Some(producer);
            }
        }
    }
    Ok(aliases)
}

pub(super) fn builtin_receiver(
    context: &mut crate::DeferredPhaseContext<'_, '_>,
) -> std::result::Result<(), crate::PipelineError> {
    context.geometry(|pass| {
        pass.depends_on_uploads()?;
        pass.read_vertex_streams()?;
        pass.read_frame_uniforms()?;
        pass.read_sampled_textures()?;
        pass.write_gbuffer()
    })?;
    context.lighting(|pass| {
        pass.read_gbuffer()?;
        pass.write_main_color()
    })
}
