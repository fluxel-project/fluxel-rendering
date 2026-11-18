//! Preparation and phase ordering for projected capture receiver batches.

use super::{CaptureTextureSlot, RendererError, capture_entry, composed, transparent_order};
use crate::{
    DualDepthPeelingConfig, ForwardAssets, ForwardDrawInput, PreparedCaptureReceiver, RenderView,
    preparation::{OitDrawPartition, partition_oit_draws},
};
use fluxel_material::SurfaceAlphaMode;
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
        if !draw.prepared().uses_scene_refraction() {
            return Err(Error::Unsupported(
                "refraction batch requires projected scene-refraction receivers",
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
    let (prefix, peel, excluded) = match config {
        Some(config) => {
            let partition = partition_oit_draws(inputs, config);
            let mut draws = partition.draws;
            let mut transparent = draws.split_off(partition.opaque_count);
            let excluded = transparent.split_off(partition.peel_count);
            (draws, transparent, excluded)
        }
        None => {
            let mut prefix = Vec::new();
            let mut ordinary = Vec::new();
            for input in inputs {
                if transparent_order::is_transparent(&input) {
                    ordinary.push(input);
                } else {
                    prefix.push(input);
                }
            }
            (prefix, Vec::new(), ordinary)
        }
    };
    // Babylon renders opaque and alpha-test queues before transparent work.
    // Preserve submitted order inside either queue. OIT owns visibility for
    // candidates, while excluded and non-OIT transparent draws retain the
    // authored alpha-index and camera order.
    let mut opaque = Vec::with_capacity(prefix.len());
    let mut mask = Vec::new();
    for input in prefix {
        match input.prepared().material().surface_policy().alpha_mode {
            SurfaceAlphaMode::Opaque => opaque.push(input),
            SurfaceAlphaMode::Mask => mask.push(input),
            SurfaceAlphaMode::Blend | SurfaceAlphaMode::MaskBlend => {
                return Err(RendererError::ForwardPipeline(
                    crate::PipelineError::InvalidDrawOrder,
                ));
            }
        }
    }
    let opaque_count = opaque.len() + mask.len();
    opaque.extend(mask);
    let peel_count = peel.len();
    opaque.extend(peel);
    opaque.extend(
        transparent_order::sort(excluded, ForwardDrawInput::order)
            .map_err(RendererError::ForwardPipeline)?,
    );
    let partition = OitDrawPartition {
        draws: opaque,
        opaque_count,
        peel_count,
    };
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
