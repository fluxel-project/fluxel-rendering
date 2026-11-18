//! Capture preparation when the producer has no Deferred G-buffer prefix.

use super::*;
use crate::renderer::forward::{
    CaptureTextureSlot, OitExecutionTarget, capture_oit_receiver_partition,
};
use std::{num::NonZeroU32, time::Instant};

impl<B> Renderer<B>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    /// Keeps an all-transparent producer in its actual Forward namespace.
    ///
    /// The caller has already selected, prepared, and ordered both the scene
    /// and producer partition. This branch must consequently neither create a
    /// placeholder G-buffer nor prepare that scene a second time.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn submit_forward_only_capture_producer(
        &mut self,
        snapshot: FrameSnapshot<ForwardResources>,
        producer_inputs: Vec<crate::ForwardDrawInput>,
        peel_count: usize,
        rounds: NonZeroU32,
        receiver_input: crate::ForwardDrawInput,
        producer_slot: CaptureTextureSlot,
        retained_assets: &ForwardAssets,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
        surface: Option<SurfaceImage<B>>,
    ) -> Result<Submission, RendererError> {
        let extent = receiver_input.extent();
        let started = Instant::now();
        let scene_refraction = receiver_input.prepared().uses_scene_refraction();
        if peel_count != 0 {
            rounds
                .get()
                .checked_mul(3)
                .and_then(|passes| passes.checked_add(5))
                .ok_or(Error::InvalidRange)?;
            crate::residency::OitPlan::admit(&self.device, extent)?;
        }
        let producer = self
            .assemble_capture_oit_side(
                snapshot.clone(),
                crate::preparation::OitDrawPartition {
                    draws: producer_inputs,
                    opaque_count: 0,
                    peel_count,
                },
                extent,
                retained_assets,
                None,
                true,
                rounds,
                started,
            )
            .await?;
        let capture = producer
            .capture_owner(&self.device, producer_slot, scene_refraction)
            .map_err(|error| self.observe_rhi_error(error))?;

        if scene_refraction
            || matches!(
                receiver_input
                    .prepared()
                    .material()
                    .surface_policy()
                    .alpha_mode,
                fluxel_material::SurfaceAlphaMode::Blend
                    | fluxel_material::SurfaceAlphaMode::MaskBlend,
            )
        {
            let receiver_partition =
                capture_oit_receiver_partition(receiver_input, receiver_config);
            let receiver_rounds =
                receiver_config.map_or(NonZeroU32::MIN, crate::DualDepthPeelingConfig::rounds);
            let receiver = self
                .assemble_capture_oit_side(
                    snapshot,
                    receiver_partition,
                    extent,
                    retained_assets,
                    Some(&capture),
                    true,
                    receiver_rounds,
                    started,
                )
                .await?;
            let target = match surface {
                Some(image) => {
                    let composite = match receiver.oit.as_ref() {
                        Some(oit) => {
                            self.oit_surface_composite_owner(&oit.output, extent, &image)
                                .await?
                        }
                        None => {
                            self.surface_composite_owner(&receiver.frame, &image)
                                .await?
                        }
                    };
                    OitExecutionTarget::Surface { image, composite }
                }
                None => OitExecutionTarget::Readback,
            };
            return self
                .execute_capture_oit(producer, receiver, target, started)
                .await;
        }

        self.deferred_residency.retain_assets(retained_assets);
        let receiver = self
            .prepare_deferred_frames_with_capture(
                std::slice::from_ref(&receiver_input),
                Some(&capture),
            )
            .await?;
        self.execute_forward_deferred_capture(
            PendingInputs::Frame {
                _snapshot: snapshot,
                _inputs: vec![receiver_input],
            },
            producer,
            receiver,
            surface,
        )
        .await
    }
}
