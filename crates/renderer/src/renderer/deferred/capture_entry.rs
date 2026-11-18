//! Deferred capture with real opaque or transparent-only producer namespaces.

use super::*;
use crate::renderer::forward::{
    CaptureTextureOwner, CaptureTextureSlot, capture_assets, capture_oit_receiver_partition,
    validate_receiver_sources, validate_receiver_view,
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
        let extent = selected.view_record().extent;
        if surface
            .as_ref()
            .is_some_and(|image| image.extent() != extent)
        {
            return Err(Error::InvalidRange.into());
        }
        let producer_inputs = crate::preparation::prepare_forward_with_lighting(
            &selected,
            assets,
            &mut fluxel_shader::ShaderComposer::new(),
            self.environment_irradiance.is_some(),
            self.environment_specular.is_some(),
        )
        .map_err(RendererError::ForwardPreparation)?;
        if producer_inputs.is_empty() {
            return Err(Error::Unsupported("Deferred capture requires visible geometry").into());
        }
        let receiver_input = receiver
            .with_capture_lighting(
                &mut fluxel_shader::ShaderComposer::new(),
                self.environment_irradiance.is_some(),
                self.environment_specular.is_some(),
            )
            .map_err(RendererError::ForwardPreparation)?
            .into_draw_input();
        validate_receiver_sources(&receiver_input, &producer_slot)?;
        validate_receiver_view(&receiver_input, selected.view_record())?;
        let scene_refraction = receiver_input.prepared().uses_scene_refraction();
        let receiver_blended = matches!(
            receiver_input
                .prepared()
                .material()
                .surface_policy()
                .alpha_mode,
            fluxel_material::SurfaceAlphaMode::Blend | fluxel_material::SurfaceAlphaMode::MaskBlend,
        );
        if let Some(config) = receiver_config
            .filter(|config| receiver_blended && !config.excludes_source(receiver_input.source()))
        {
            config
                .rounds()
                .get()
                .checked_mul(3)
                .and_then(|passes| passes.checked_add(5))
                .ok_or(Error::InvalidRange)?;
            crate::residency::OitPlan::admit(&self.device, extent)?;
        }
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
        let retained_assets = capture_assets(assets, &receiver_input);
        let (producer_inputs, opaque_count, peel_count, rounds) = match producer_config {
            Some(config) => {
                let partition = super::super::forward::order_forward_oit_draws(
                    pipeline,
                    &selected,
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
                    &selected,
                    producer_inputs,
                    crate::FrameTopology::SortedForward,
                )
                .map_err(RendererError::DeferredPipeline)?;
                let (transparent, mut opaque): (Vec<_>, Vec<_>) =
                    ordered.into_iter().partition(|input| {
                        matches!(
                            input.prepared().material().surface_policy().alpha_mode,
                            fluxel_material::SurfaceAlphaMode::Blend
                                | fluxel_material::SurfaceAlphaMode::MaskBlend,
                        )
                    });
                let opaque_count = opaque.len();
                opaque.extend(transparent);
                (opaque, opaque_count, 0, NonZeroU32::MIN)
            }
        };
        if opaque_count == 0 {
            return self
                .submit_forward_only_capture_producer(
                    snapshot,
                    producer_inputs,
                    peel_count,
                    rounds,
                    receiver_input,
                    producer_slot,
                    &retained_assets,
                    receiver_config,
                    surface,
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
        if scene_refraction || receiver_blended {
            let receiver_partition =
                capture_oit_receiver_partition(receiver_input, receiver_config);
            let rounds =
                receiver_config.map_or(NonZeroU32::MIN, crate::DualDepthPeelingConfig::rounds);
            let receiver = self
                .assemble_capture_oit_side(
                    snapshot.clone(),
                    receiver_partition,
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
                    |context| pipeline.contribute_deferred(&selected, context),
                )
                .await;
        }
        let receiver = self
            .prepare_deferred_frames_with_capture(
                std::slice::from_ref(&receiver_input),
                Some(&capture),
            )
            .await?;
        let mut inputs = producer_inputs;
        inputs.push(receiver_input);
        self.execute_deferred_capture(
            PendingInputs::Frame {
                _snapshot: snapshot.clone(),
                _inputs: inputs,
            },
            producer,
            composition,
            receiver,
            surface,
            |context| pipeline.contribute_deferred(&selected, context),
        )
        .await
    }
}
