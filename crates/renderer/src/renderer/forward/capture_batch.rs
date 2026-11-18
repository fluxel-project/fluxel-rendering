//! Preparation and phase ordering for projected capture receiver batches.

use super::{CaptureTextureSlot, RendererError, capture_entry, composed, transparent_order};
use crate::{
    DualDepthPeelingConfig, ForwardAssets, ForwardDrawInput, PreparedCaptureReceiver, RenderView,
    preparation::{OitDrawPartition, partition_oit_draws},
};
use fluxel_rhi::api::Error;

fn prepare_capture_receiver_batch(
    receivers: Vec<PreparedCaptureReceiver>,
    producer_slot: &CaptureTextureSlot,
    producer_view: &RenderView,
    assets: &ForwardAssets,
    environment: bool,
    specular: bool,
) -> Result<(Vec<ForwardDrawInput>, ForwardAssets), RendererError> {
    if receivers.is_empty() {
        return Err(Error::InvalidRange.into());
    }
    let mut composer = fluxel_shader::ShaderComposer::new();
    let mut draws = Vec::with_capacity(receivers.len());
    let mut retained_assets = assets.clone();
    for receiver in receivers {
        let draw = receiver
            .with_capture_lighting(&mut composer, environment, specular)
            .map_err(RendererError::ForwardPreparation)?
            .into_draw_input();
        if !draw.prepared().uses_scene_refraction() || !transparent_order::is_transparent(&draw) {
            return Err(Error::Unsupported(
                "refraction batch requires Blend or MaskBlend projected receivers",
            )
            .into());
        }
        capture_entry::validate_receiver_sources(&draw, producer_slot)?;
        capture_entry::validate_receiver_view(&draw, producer_view)?;
        capture_entry::retain_capture_assets(&mut retained_assets, &draw);
        draws.push(draw);
    }
    composed::validate_draw_sources(&draws).map_err(RendererError::ForwardPipeline)?;
    Ok((draws, retained_assets))
}

fn partition_capture_receiver_draws(
    inputs: Vec<ForwardDrawInput>,
    config: Option<&DualDepthPeelingConfig>,
) -> Result<OitDrawPartition, RendererError> {
    let mut partition = match config {
        Some(config) => partition_oit_draws(inputs, config),
        None => OitDrawPartition {
            draws: inputs,
            opaque_count: 0,
            peel_count: 0,
        },
    };
    // Ordinary receivers consume their authored alpha indices. Peeling
    // candidates retain their received order and resolve visibility on the GPU.
    let ordinary = partition.draws.split_off(partition.peel_count);
    partition.draws.extend(
        transparent_order::sort(ordinary, ForwardDrawInput::order)
            .map_err(RendererError::ForwardPipeline)?,
    );
    Ok(partition)
}

pub(in crate::renderer) enum CaptureReceivers {
    Single(PreparedCaptureReceiver),
    RefractionBatch(Vec<PreparedCaptureReceiver>),
}

impl CaptureReceivers {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) fn prepare(
        self,
        producer_slot: &CaptureTextureSlot,
        producer_view: &RenderView,
        assets: &ForwardAssets,
        environment: bool,
        specular: bool,
        composer: &mut fluxel_shader::ShaderComposer,
        config: Option<&DualDepthPeelingConfig>,
    ) -> Result<(OitDrawPartition, ForwardAssets, bool), RendererError> {
        match self {
            Self::Single(receiver) => {
                let receiver = receiver
                    .with_capture_lighting(composer, environment, specular)
                    .map_err(RendererError::ForwardPreparation)?
                    .into_draw_input();
                capture_entry::validate_receiver_sources(&receiver, producer_slot)?;
                capture_entry::validate_receiver_view(&receiver, producer_view)?;
                let scene_refraction = receiver.prepared().uses_scene_refraction();
                let assets = capture_entry::capture_assets(assets, &receiver);
                let partition =
                    super::capture_oit_entry::capture_oit_receiver_partition(receiver, config);
                Ok((partition, assets, scene_refraction))
            }
            Self::RefractionBatch(receivers) => {
                let (receivers, assets) = prepare_capture_receiver_batch(
                    receivers,
                    producer_slot,
                    producer_view,
                    assets,
                    environment,
                    specular,
                )?;
                Ok((
                    partition_capture_receiver_draws(receivers, config)?,
                    assets,
                    true,
                ))
            }
        }
    }
}

pub(super) fn order_capture_producer_draws<P: crate::FramePipeline<crate::ForwardResources>>(
    pipeline: &P,
    frame: &crate::PreparedFrame<'_, crate::ForwardResources>,
    inputs: Vec<ForwardDrawInput>,
    config: Option<&DualDepthPeelingConfig>,
) -> Result<OitDrawPartition, crate::PipelineError> {
    match config {
        Some(config) => super::oit_order::order_oit_draws(pipeline, frame, inputs, config),
        None => Ok(OitDrawPartition {
            draws: composed::order_draws(
                pipeline,
                frame,
                inputs,
                crate::FrameTopology::SortedForward,
            )?,
            opaque_count: 0,
            peel_count: 0,
        }),
    }
}
