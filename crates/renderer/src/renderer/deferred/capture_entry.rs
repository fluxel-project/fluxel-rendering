//! Deferred capture with real opaque or transparent-only producer namespaces.

use super::capture_feature::{CaptureDrawGroups, CaptureProducerReads, SceneCaptureProducer};
use super::*;
use crate::renderer::forward::{CaptureReceivers, CaptureTextureOwner, CaptureTextureSlot};
use std::{num::NonZeroU32, time::Instant};

pub(super) enum CaptureReceiverInput {
    Forward {
        partition: crate::preparation::OitDrawPartition,
        rounds: NonZeroU32,
    },
    Deferred(Vec<crate::ForwardDrawInput>),
}

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
    /// Lights the producer and samples its color in a Deferred receiver within
    /// one Graph submission. The producer may append sorted transparency to
    /// its opaque/masked prefix, or use the actual Forward path when that
    /// partition has no opaque prefix. Blended and projected scene-refraction
    /// receivers compose in the Forward branch against the selected capture.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_deferred_capture<V>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
    {
        self.submit_deferred_capture_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            producer_slot,
            receiver,
            None,
            None,
            None,
        )
        .await
    }

    /// Presents the capture receiver without a CPU readback. A nonempty
    /// producer partition with no opaque prefix remains in the actual Forward
    /// namespace rather than creating an empty G-buffer.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_deferred_capture_surface<V>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
        image: SurfaceImage<B>,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
    {
        self.submit_deferred_capture_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            producer_slot,
            receiver,
            None,
            None,
            Some(image),
        )
        .await
    }

    /// Captures one Deferred producer and composites a same-view batch of
    /// projected transparent receivers in one Graph submission.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_deferred_capture_batch<V>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        producer_slot: CaptureTextureSlot,
        receivers: Vec<crate::PreparedCaptureReceiver>,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
    {
        self.submit_deferred_capture_target_inputs(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            producer_slot,
            CaptureReceivers::RefractionBatch(receivers),
            producer_config,
            receiver_config,
            None,
        )
        .await
    }

    /// Surface form of [`Self::submit_deferred_capture_batch`].
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_deferred_capture_batch_surface<V>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        producer_slot: CaptureTextureSlot,
        receivers: Vec<crate::PreparedCaptureReceiver>,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
        image: SurfaceImage<B>,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
    {
        self.submit_deferred_capture_target_inputs(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            producer_slot,
            CaptureReceivers::RefractionBatch(receivers),
            producer_config,
            receiver_config,
            Some(image),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn submit_deferred_capture_target<V>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
        surface: Option<SurfaceImage<B>>,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
    {
        self.submit_deferred_capture_target_inputs(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            producer_slot,
            CaptureReceivers::Single(receiver),
            producer_config,
            receiver_config,
            surface,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn submit_deferred_capture_target_inputs<V>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        producer_slot: CaptureTextureSlot,
        receivers: CaptureReceivers,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
        surface: Option<SurfaceImage<B>>,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
    {
        self.ensure_open()?;
        self.free_slot().ok_or(RendererError::Backpressure)?;
        if let Some(image) = surface.as_ref() {
            if image.device() != self.device.id() {
                return Err(Error::WrongDevice.into());
            }
            if !image.is_current() {
                return Err(Error::SurfaceOutdated.into());
            }
        }
        let selected = pipeline
            .prepare(&snapshot, view, visibility)
            .map_err(RendererError::DeferredPipeline)?;
        selected
            .validate_for(&snapshot, view)
            .map_err(RendererError::DeferredPipeline)?;
        if surface
            .as_ref()
            .is_some_and(|image| image.extent() != selected.view_record().extent)
        {
            return Err(Error::InvalidRange.into());
        }
        let (receiver, retained_assets, scene_refraction) = receivers.prepare(
            &producer_slot,
            selected.view_record(),
            assets,
            self.environment_irradiance.is_some(),
            self.environment_specular.is_some(),
            &mut fluxel_shader::ShaderComposer::new(),
            receiver_config,
        )?;
        let producer_reads = SceneCaptureProducer {
            pipeline,
            selected: &selected,
        };
        self.submit_deferred_capture_prepared(
            pipeline,
            snapshot.clone(),
            &selected,
            assets,
            producer_slot,
            receiver,
            scene_refraction,
            retained_assets,
            producer_config,
            receiver_config,
            surface,
            Vec::new(),
            &producer_reads,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn submit_deferred_capture_prepared<V>(
        &mut self,
        pipeline: &crate::DeferredCpuPipeline<V>,
        snapshot: FrameSnapshot<ForwardResources>,
        selected: &crate::PreparedFrame<'_, ForwardResources>,
        assets: &ForwardAssets,
        producer_slot: CaptureTextureSlot,
        partition: crate::preparation::OitDrawPartition,
        scene_refraction: bool,
        retained_assets: ForwardAssets,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
        surface: Option<SurfaceImage<B>>,
        additional_producer_draws: Vec<crate::ForwardDrawInput>,
        producer_reads: &impl CaptureProducerReads<B>,
    ) -> Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<ForwardResources>,
    {
        let extent = selected.view_record().extent;
        let mut producer_inputs = crate::preparation::prepare_forward_with_lighting(
            selected,
            assets,
            &mut fluxel_shader::ShaderComposer::new(),
            self.environment_irradiance.is_some(),
            self.environment_specular.is_some(),
        )
        .map_err(RendererError::ForwardPreparation)?;
        producer_inputs.extend(additional_producer_draws);
        if producer_inputs.is_empty() {
            return Err(Error::Unsupported("Deferred capture requires visible geometry").into());
        }
        let receiver_blended = partition.draws.iter().any(|input| {
            matches!(
                input.effective_alpha_mode(),
                fluxel_material::SurfaceAlphaMode::Blend
                    | fluxel_material::SurfaceAlphaMode::MaskBlend,
            )
        });
        let receiver = if scene_refraction || receiver_blended {
            let rounds =
                receiver_config.map_or(NonZeroU32::MIN, crate::DualDepthPeelingConfig::rounds);
            if partition.peel_count != 0 {
                rounds
                    .get()
                    .checked_mul(3)
                    .and_then(|passes| passes.checked_add(5))
                    .ok_or(Error::InvalidRange)?;
                crate::residency::OitPlan::admit(&self.device, extent)?;
            }
            CaptureReceiverInput::Forward { partition, rounds }
        } else {
            CaptureReceiverInput::Deferred(partition.draws)
        };
        if scene_refraction {
            super::super::forward::target_admission::admit(
                &self.device,
                crate::FrameTopology::SortedForward,
                extent,
                self.deferred_scene_color_format(),
                false,
                true,
                true,
            )
            .map_err(|error| self.observe_rhi_error(error))?;
        }
        let (producer_inputs, opaque_count, peel_count, rounds) = match producer_config {
            Some(config) => {
                let partition = super::super::forward::order_forward_oit_draws(
                    pipeline,
                    selected,
                    producer_inputs,
                    config,
                )
                .map_err(RendererError::DeferredPipeline)?;
                (
                    partition.draws,
                    partition.opaque_count,
                    partition.peel_count,
                    config.rounds(),
                )
            }
            None => {
                let ordered = super::super::forward::order_forward_draws(
                    pipeline,
                    selected,
                    producer_inputs,
                    crate::FrameTopology::SortedForward,
                )
                .map_err(RendererError::DeferredPipeline)?;
                let (transparent, mut opaque): (Vec<_>, Vec<_>) =
                    ordered.into_iter().partition(|input| {
                        matches!(
                            input.effective_alpha_mode(),
                            fluxel_material::SurfaceAlphaMode::Blend
                                | fluxel_material::SurfaceAlphaMode::MaskBlend,
                        )
                    });
                let opaque_count = opaque.len();
                opaque.extend(transparent);
                (opaque, opaque_count, 0, NonZeroU32::MIN)
            }
        };
        let handles = producer_reads.feature_handles();
        let opaque_groups =
            CaptureDrawGroups::from_draws(&producer_inputs[..opaque_count], handles)
                .map_err(RendererError::DeferredPipeline)?;
        let overlay_groups =
            CaptureDrawGroups::from_draws(&producer_inputs[opaque_count..], handles)
                .map_err(RendererError::DeferredPipeline)?;
        if opaque_count == 0 {
            return self
                .submit_forward_only_capture_producer(
                    snapshot.clone(),
                    producer_inputs,
                    peel_count,
                    rounds,
                    receiver,
                    scene_refraction,
                    producer_slot,
                    &retained_assets,
                    surface,
                    producer_reads,
                    &overlay_groups,
                )
                .await;
        }
        let (producer, composition, producer_inputs) = self
            .prepare_deferred_capture_producer(
                snapshot.clone(),
                producer_inputs,
                opaque_count,
                peel_count,
                rounds,
                &retained_assets,
                Instant::now(),
                scene_refraction,
            )
            .await?;
        let mut capture = match composition.as_ref() {
            Some(composition) => match composition.oit.as_ref() {
                Some(oit) => {
                    CaptureTextureOwner::from_oit_output(&self.device, producer_slot, &oit.output)
                }
                None => CaptureTextureOwner::from_capture_frame(
                    &self.device,
                    producer_slot,
                    &composition.frame,
                ),
            },
            None => CaptureTextureOwner::from_deferred_scene_color(
                &self.device,
                producer_slot,
                &producer.scene_color,
                extent,
            ),
        }
        .map_err(|error| self.observe_rhi_error(error))?;
        if scene_refraction {
            capture.retain_deferred_refraction_depth(&producer.gbuffer[2]);
        }
        if let CaptureReceiverInput::Forward { partition, rounds } = receiver {
            let receiver = self
                .assemble_capture_oit_side(
                    snapshot.clone(),
                    partition,
                    extent,
                    &retained_assets,
                    Some(&capture),
                    true,
                    rounds,
                    Instant::now(),
                )
                .await?;
            return self
                .execute_deferred_forward_capture(
                    PendingInputs::Frame {
                        _snapshot: snapshot.clone(),
                        _inputs: producer_inputs,
                    },
                    producer,
                    composition,
                    receiver,
                    surface,
                    scene_refraction,
                    |context| producer_reads.deferred(context, &opaque_groups),
                    |context| producer_reads.overlay(context, &overlay_groups),
                )
                .await;
        }
        let CaptureReceiverInput::Deferred(receiver_inputs) = receiver else {
            return Err(Error::InvalidState.into());
        };
        let receiver = self
            .prepare_deferred_frames_with_capture(&receiver_inputs, Some(&capture))
            .await?;
        let mut inputs = producer_inputs;
        inputs.extend(receiver_inputs);
        self.execute_deferred_capture(
            PendingInputs::Frame {
                _snapshot: snapshot.clone(),
                _inputs: inputs,
            },
            producer,
            composition,
            receiver,
            surface,
            |context| producer_reads.deferred(context, &opaque_groups),
            |context| producer_reads.overlay(context, &overlay_groups),
        )
        .await
    }
}
