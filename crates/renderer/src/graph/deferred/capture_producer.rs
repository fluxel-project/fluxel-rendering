//! Optional producer-side transparency appended before a Deferred capture alias.

use super::*;
use crate::{graph::ForwardOitInput, residency::ForwardFrame};

/// Producer transparency supplied to the combined Deferred capture Graph.
///
/// `oit` selects dual-depth peeling; without it the frame is an ordinary
/// depth-tested Forward overlay.  The overlay frame remains independently
/// owned and is retained by the renderer through the combined completion.
pub(crate) struct DeferredCaptureOverlay<'a, B: ResourceBackend> {
    pub(crate) frame: &'a ForwardFrame<B>,
    pub(crate) oit: Option<ForwardOitInput<'a, B>>,
}

pub(super) enum DeferredCaptureProducerComposition {
    Ordinary(super::transparent_declaration::DeferredTransparentComposition),
    Oit(super::oit_declaration::DeferredOitComposition),
}

impl DeferredCaptureProducerComposition {
    pub(super) fn final_source(&self) -> crate::graph::definition::TextureHandle {
        match self {
            Self::Ordinary(composition) => composition.final_source,
            Self::Oit(composition) => composition.final_source,
        }
    }

    pub(super) fn append_passes(&self, passes: &mut Vec<GraphPass>) {
        match self {
            Self::Ordinary(composition) => {
                passes.push(composition.overlay_upload);
                passes.extend(composition.overlay_mips.iter().map(|item| item.pass));
                passes.push(composition.overlay_raster);
            }
            Self::Oit(composition) => {
                passes.push(composition.overlay_upload);
                passes.extend(composition.overlay_mips.iter().map(|item| item.pass));
                let product = &composition.product.passes;
                passes.extend([
                    product.initial_clear[0].pass,
                    product.initial_clear[1].pass,
                    product.initialize_back_accumulator,
                    product.initial_depth,
                ]);
                for iteration in &product.iterations {
                    passes.extend([
                        iteration.clear.pass,
                        iteration.peel,
                        iteration.back_accumulate,
                    ]);
                }
                passes.push(product.final_composite);
                if let Some(pass) = composition.fallback {
                    passes.push(pass);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn record_capture_producer_composition<B>(
    composition: &DeferredCaptureProducerComposition,
    encoder: &mut CommandEncoder<B>,
    producer: &DeferredFrame<B>,
    overlay: &DeferredCaptureOverlay<'_, B>,
    commands: Option<&crate::graph::oit_record::OitRecordCommands<'_, B>>,
    bindings: &FrameBindings<'_, B>,
    pass: GraphPass,
) -> Result<bool>
where
    B: ResourceBackend + CommandBackend + RasterBackend,
{
    match composition {
        DeferredCaptureProducerComposition::Ordinary(composition) => {
            if overlay.oit.is_some() || commands.is_some() {
                return Err(Error::InvalidState);
            }
            if pass == composition.overlay_upload {
                composition
                    .overlay
                    .record_upload(encoder, bindings, overlay.frame, pass)?;
                return Ok(true);
            }
            if let Some(generation) = composition
                .overlay_mips
                .iter()
                .find(|item| item.pass == pass)
            {
                composition
                    .overlay
                    .record_mip(encoder, bindings, overlay.frame, generation)?;
                return Ok(true);
            }
            if pass == composition.overlay_raster {
                composition.overlay.record_raster(
                    encoder,
                    bindings,
                    overlay.frame,
                    &producer.scene_color,
                    &producer.gbuffer[2],
                    pass,
                )?;
                return Ok(true);
            }
            Ok(false)
        }
        DeferredCaptureProducerComposition::Oit(composition) => {
            let input = overlay.oit.as_ref().ok_or(Error::InvalidState)?;
            let commands = commands.ok_or(Error::InvalidState)?;
            validate_oit_commands(producer, input, composition, commands)?;
            if pass == composition.overlay_upload {
                composition
                    .overlay
                    .record_upload(encoder, bindings, overlay.frame, pass)?;
                return Ok(true);
            }
            if let Some(generation) = composition
                .overlay_mips
                .iter()
                .find(|item| item.pass == pass)
            {
                composition
                    .overlay
                    .record_mip(encoder, bindings, overlay.frame, generation)?;
                return Ok(true);
            }
            if composition.fallback == Some(pass) {
                composition.overlay.record_raster_range(
                    encoder,
                    bindings,
                    overlay.frame,
                    crate::graph::oit_record::OitColorTargetRef {
                        texture: &input.output.texture,
                        view: &input.output.view,
                        desc: &input.output.desc,
                        instance: input.output.instance,
                        device: input.output.device,
                    },
                    composition.final_slot,
                    &producer.gbuffer[2],
                    composition.peel_count..overlay.frame.draws.len(),
                    pass,
                )?;
                return Ok(true);
            }
            if is_oit_product_pass(&composition.product, pass) {
                crate::graph::oit_record::record_pass(
                    producer.resident.device,
                    encoder,
                    bindings,
                    &composition.product,
                    pass,
                    &composition.slots,
                    input.owners,
                    &crate::graph::oit_record::OitDepthTargetRef::from(&producer.gbuffer[2]),
                    overlay.frame,
                    &composition.overlay.draw_slots(),
                    commands,
                )?;
                return Ok(true);
            }
            Ok(false)
        }
    }
}

pub(super) fn validate_capture_producer_commands<B: ResourceBackend>(
    composition: Option<&DeferredCaptureProducerComposition>,
    producer: &DeferredFrame<B>,
    overlay: Option<&DeferredCaptureOverlay<'_, B>>,
    commands: Option<&crate::graph::oit_record::OitRecordCommands<'_, B>>,
) -> Result<()> {
    match (composition, overlay, commands) {
        (None, None, None) => Ok(()),
        (Some(DeferredCaptureProducerComposition::Ordinary(_)), Some(overlay), None)
            if overlay.oit.is_none() =>
        {
            Ok(())
        }
        (
            Some(DeferredCaptureProducerComposition::Oit(composition)),
            Some(overlay),
            Some(commands),
        ) => validate_oit_commands(
            producer,
            overlay.oit.as_ref().ok_or(Error::InvalidState)?,
            composition,
            commands,
        ),
        _ => Err(Error::InvalidState),
    }
}

fn is_oit_product_pass(product: &crate::graph::oit::DualDepthPeelGraph, pass: GraphPass) -> bool {
    let stages = &product.passes;
    pass == stages.initial_clear[0].pass
        || pass == stages.initial_clear[1].pass
        || pass == stages.initialize_back_accumulator
        || pass == stages.initial_depth
        || pass == stages.final_composite
        || stages.iterations.iter().any(|item| {
            pass == item.clear.pass || pass == item.peel || pass == item.back_accumulate
        })
}

fn validate_oit_commands<B: ResourceBackend>(
    producer: &DeferredFrame<B>,
    input: &ForwardOitInput<'_, B>,
    composition: &super::oit_declaration::DeferredOitComposition,
    commands: &crate::graph::oit_record::OitRecordCommands<'_, B>,
) -> Result<()> {
    let device = producer.resident.device;
    if commands.draws.len() != composition.peel_count
        || commands
            .draws
            .iter()
            .enumerate()
            .any(|(index, draw)| draw.draw != index)
        || commands.output.instance != input.output.instance
        || !std::ptr::eq(commands.output.texture, &input.output.texture)
        || !std::ptr::eq(commands.output.view, &input.output.view)
        || commands.opaque_color.instance != producer.scene_color.instance
        || !std::ptr::eq(commands.opaque_color.texture, &producer.scene_color.texture)
        || !std::ptr::eq(commands.opaque_color.view, &producer.scene_color.view)
        || commands.draws.iter().any(|draw| {
            draw.initial.device() != device
                || draw.peel.device() != device
                || draw.initial_group3.group.device() != device
                || draw
                    .peel_group3
                    .iter()
                    .any(|group| group.group.device() != device)
                || draw
                    .empty_group2
                    .is_some_and(|group| group.device() != device)
        })
        || std::iter::once(&commands.init_back)
            .chain(commands.back_accumulate.iter())
            .chain(commands.final_composite.iter())
            .any(|command| {
                command.pipeline.device() != device || command.bind_group.group.device() != device
            })
    {
        Err(Error::InvalidState)
    } else {
        Ok(())
    }
}

pub(super) fn declare_capture_producer_composition<B: ResourceBackend>(
    device: &Device<B>,
    prefix: &mut super::prefix::DeferredGraphPrefix,
    producer: &DeferredFrame<B>,
    overlay: &DeferredCaptureOverlay<'_, B>,
    shared: Option<&mut crate::graph::forward::SharedResidentImports>,
) -> Result<DeferredCaptureProducerComposition> {
    match overlay.oit.as_ref() {
        Some(input) => super::oit_declaration::declare_oit_composition_with_shared(
            device,
            prefix,
            producer,
            overlay.frame,
            input,
            shared,
        )
        .map(DeferredCaptureProducerComposition::Oit),
        None => {
            if producer.deformation.is_some() {
                return Err(Error::Unsupported("deferred transparent deformation"));
            }
            super::transparent_declaration::declare_transparent_composition_with_shared(
                prefix,
                overlay.frame,
                shared,
            )
            .map(DeferredCaptureProducerComposition::Ordinary)
        }
    }
}
