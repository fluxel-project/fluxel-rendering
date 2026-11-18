//! Two typed features in the actual capture producer OIT declarations.

use super::*;
use crate::{FeatureBinding, FramePipeline, RenderFeature, pipeline::read_all};
use std::{num::NonZeroU32, time::Instant};

use super::capture_batch::{CaptureReceivers, order_capture_producer_draws};

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
    /// Captures meshes and two typed Features through selected OIT stages,
    /// then shades one material receiver in the same Graph submission.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_oit_capture_with_features<V, F, G>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        features: (FeatureBinding<'_, F>, FeatureBinding<'_, G>),
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
    ) -> Result<Submission>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
        F: RenderFeature,
        G: RenderFeature,
    {
        self.submit_composed_oit_capture_pair_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            features,
            producer_slot,
            CaptureReceivers::Single(receiver),
            producer_config,
            receiver_config,
            None,
        )
        .await
    }

    /// Surface form of [`Self::submit_forward_oit_capture_with_features`].
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_oit_capture_surface_with_features<V, F, G>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        features: (FeatureBinding<'_, F>, FeatureBinding<'_, G>),
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
        image: SurfaceImage<B>,
    ) -> Result<Submission>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
        F: RenderFeature,
        G: RenderFeature,
    {
        self.submit_composed_oit_capture_pair_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            features,
            producer_slot,
            CaptureReceivers::Single(receiver),
            producer_config,
            receiver_config,
            Some(image),
        )
        .await
    }

    /// Captures meshes and two typed Features, then shades a validated
    /// projected refraction receiver batch in one Graph submission.
    ///
    /// Omit either peeling configuration to retain the ordinary sorted stage.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_oit_capture_batch_with_features<V, F, G>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        features: (FeatureBinding<'_, F>, FeatureBinding<'_, G>),
        producer_slot: CaptureTextureSlot,
        receivers: Vec<crate::PreparedCaptureReceiver>,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
    ) -> Result<Submission>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
        F: RenderFeature,
        G: RenderFeature,
    {
        self.submit_composed_oit_capture_pair_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            features,
            producer_slot,
            CaptureReceivers::RefractionBatch(receivers),
            producer_config,
            receiver_config,
            None,
        )
        .await
    }

    /// Surface form of [`Self::submit_forward_oit_capture_batch_with_features`].
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_oit_capture_batch_surface_with_features<V, F, G>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        features: (FeatureBinding<'_, F>, FeatureBinding<'_, G>),
        producer_slot: CaptureTextureSlot,
        receivers: Vec<crate::PreparedCaptureReceiver>,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
        image: SurfaceImage<B>,
    ) -> Result<Submission>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
        F: RenderFeature,
        G: RenderFeature,
    {
        self.submit_composed_oit_capture_pair_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            features,
            producer_slot,
            CaptureReceivers::RefractionBatch(receivers),
            producer_config,
            receiver_config,
            Some(image),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn submit_composed_oit_capture_pair_target<V, F, G>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        features: (FeatureBinding<'_, F>, FeatureBinding<'_, G>),
        producer_slot: CaptureTextureSlot,
        receivers: CaptureReceivers,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
        surface: Option<SurfaceImage<B>>,
    ) -> Result<Submission>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
        F: RenderFeature,
        G: RenderFeature,
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
        let started = Instant::now();
        let selected = pipeline
            .prepare(&snapshot, view, visibility)
            .map_err(RendererError::ForwardPipeline)?;
        selected
            .validate_for(&snapshot, view)
            .map_err(RendererError::ForwardPipeline)?;
        let extent = selected.view_record().extent;
        if extent.contains(&0)
            || surface
                .as_ref()
                .is_some_and(|image| image.extent() != extent)
        {
            return Err(Error::InvalidRange.into());
        }
        let (first, second) = features;
        let first_handle = first.feature.handle();
        let second_handle = second.feature.handle();
        if first_handle == second_handle {
            return Err(RendererError::ForwardFeature(
                crate::FeatureError::DuplicateFeatureBinding {
                    feature: first_handle,
                },
            ));
        }
        let first_context = first
            .issuer
            .context(&snapshot, crate::CpuPhase::Reference, first_handle)
            .map_err(RendererError::ForwardFeature)?;
        crate::validate_context(&first_context, first.input, first_handle)
            .map_err(RendererError::ForwardFeature)?;
        let first_prepared = first
            .feature
            .prepare(&first_context, first.input, &selected, visibility)
            .map_err(RendererError::ForwardFeature)?;
        let second_context = second
            .issuer
            .context(&snapshot, crate::CpuPhase::Reference, second_handle)
            .map_err(RendererError::ForwardFeature)?;
        crate::validate_context(&second_context, second.input, second_handle)
            .map_err(RendererError::ForwardFeature)?;
        let second_prepared = second
            .feature
            .prepare(&second_context, second.input, &selected, visibility)
            .map_err(RendererError::ForwardFeature)?;

        let environment = self.environment_irradiance.is_some();
        let specular = self.environment_specular.is_some();
        let mut composer = fluxel_shader::ShaderComposer::new();
        let (receiver, retained_assets, scene_refraction) = receivers.prepare(
            &producer_slot,
            selected.view_record(),
            assets,
            environment,
            specular,
            &mut composer,
            receiver_config,
        )?;
        let mut inputs = crate::preparation::prepare_forward_with_lighting(
            &selected,
            assets,
            &mut composer,
            environment,
            specular,
        )
        .map_err(RendererError::ForwardPreparation)?;
        inputs.extend(
            crate::preparation::prepare_feature_draws_with_lighting(
                &selected,
                first_handle,
                first.feature.draws(&first_prepared),
                assets,
                &mut composer,
                environment,
                specular,
            )
            .map_err(RendererError::ForwardPreparation)?,
        );
        inputs.extend(
            crate::preparation::prepare_feature_draws_with_lighting(
                &selected,
                second_handle,
                second.feature.draws(&second_prepared),
                assets,
                &mut composer,
                environment,
                specular,
            )
            .map_err(RendererError::ForwardPreparation)?,
        );
        let producer = order_capture_producer_draws(pipeline, &selected, inputs, producer_config)
            .map_err(RendererError::ForwardPipeline)?;
        let (meshes, first_draws, second_draws) =
            super::oit_composed_pair::pair_draws(&producer.draws, first_handle, second_handle)?;
        let producer_rounds =
            producer_config.map_or(NonZeroU32::MIN, crate::DualDepthPeelingConfig::rounds);
        let receiver_rounds =
            receiver_config.map_or(NonZeroU32::MIN, crate::DualDepthPeelingConfig::rounds);
        if scene_refraction {
            self.admit_oit_capture_refraction(
                extent,
                surface.is_some() || receiver.peel_count != 0,
            )?;
        }
        for (partition, rounds) in [(&producer, producer_rounds), (&receiver, receiver_rounds)] {
            if partition.peel_count != 0 {
                rounds
                    .get()
                    .checked_mul(3)
                    .and_then(|passes| passes.checked_add(5))
                    .ok_or(Error::InvalidRange)?;
                crate::residency::OitPlan::admit(&self.device, extent)?;
            }
        }
        self.execute_prepared_oit_capture(
            snapshot.clone(),
            producer,
            receiver,
            extent,
            retained_assets,
            producer_slot,
            scene_refraction,
            surface,
            producer_rounds,
            receiver_rounds,
            started,
            |phase| {
                let mut context = crate::ForwardPairCompositionContext::new(
                    phase,
                    first.feature,
                    &first_prepared,
                    second.feature,
                    &second_prepared,
                    &meshes,
                    &first_draws,
                    &second_draws,
                );
                pipeline.contribute_composed_pair(&selected, &mut context)
            },
            |context| {
                context.scope(&meshes, read_all)?;
                context.scope(&first_draws, |pass| {
                    first.feature.contribute(&first_prepared, pass)
                })?;
                context.scope(&second_draws, |pass| {
                    second.feature.contribute(&second_prepared, pass)
                })
            },
        )
        .await
    }
}

type Result<T> = std::result::Result<T, RendererError>;
