//! One typed feature in the actual capture producer OIT declarations.

use super::*;
use crate::{FeatureBinding, ForwardDrawSource, FramePipeline, RenderFeature};
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
    /// Captures scene meshes and one typed Feature through the producer's
    /// selected OIT stages, then shades the material receiver in one Graph.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_oit_capture_with_feature<V, F>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        feature: FeatureBinding<'_, F>,
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
        producer_config: &crate::DualDepthPeelingConfig,
        receiver_config: &crate::DualDepthPeelingConfig,
    ) -> Result<Submission>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
        F: RenderFeature,
    {
        self.submit_composed_oit_capture_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            feature,
            producer_slot,
            CaptureReceivers::Single(receiver),
            Some(producer_config),
            Some(receiver_config),
            None,
        )
        .await
    }

    /// Surface form of [`Self::submit_forward_oit_capture_with_feature`].
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_oit_capture_surface_with_feature<V, F>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        feature: FeatureBinding<'_, F>,
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
        producer_config: &crate::DualDepthPeelingConfig,
        receiver_config: &crate::DualDepthPeelingConfig,
        image: SurfaceImage<B>,
    ) -> Result<Submission>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
        F: RenderFeature,
    {
        self.submit_composed_oit_capture_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            feature,
            producer_slot,
            CaptureReceivers::Single(receiver),
            Some(producer_config),
            Some(receiver_config),
            Some(image),
        )
        .await
    }

    /// Captures scene meshes and one typed Feature through the producer's
    /// selected OIT stages, then shades a validated scene-refraction receiver
    /// batch in the same Graph submission.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_oit_capture_batch_with_feature<V, F>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        feature: FeatureBinding<'_, F>,
        producer_slot: CaptureTextureSlot,
        receivers: Vec<crate::PreparedCaptureReceiver>,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
    ) -> Result<Submission>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
        F: RenderFeature,
    {
        self.submit_composed_oit_capture_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            feature,
            producer_slot,
            CaptureReceivers::RefractionBatch(receivers),
            producer_config,
            receiver_config,
            None,
        )
        .await
    }

    /// Surface form of [`Self::submit_forward_oit_capture_batch_with_feature`].
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_oit_capture_batch_surface_with_feature<V, F>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        feature: FeatureBinding<'_, F>,
        producer_slot: CaptureTextureSlot,
        receivers: Vec<crate::PreparedCaptureReceiver>,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
        image: SurfaceImage<B>,
    ) -> Result<Submission>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
        F: RenderFeature,
    {
        self.submit_composed_oit_capture_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            feature,
            producer_slot,
            CaptureReceivers::RefractionBatch(receivers),
            producer_config,
            receiver_config,
            Some(image),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn submit_composed_oit_capture_target<V, F>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        feature: FeatureBinding<'_, F>,
        producer_slot: CaptureTextureSlot,
        receiver: CaptureReceivers,
        producer_config: Option<&crate::DualDepthPeelingConfig>,
        receiver_config: Option<&crate::DualDepthPeelingConfig>,
        surface: Option<SurfaceImage<B>>,
    ) -> Result<Submission>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
        F: RenderFeature,
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
        let handle = feature.feature.handle();
        let context = feature
            .issuer
            .context(&snapshot, crate::CpuPhase::Reference, handle)
            .map_err(RendererError::ForwardFeature)?;
        crate::validate_context(&context, feature.input, handle)
            .map_err(RendererError::ForwardFeature)?;
        let prepared = feature
            .feature
            .prepare(&context, feature.input, &selected, visibility)
            .map_err(RendererError::ForwardFeature)?;
        let environment = self.environment_irradiance.is_some();
        let specular = self.environment_specular.is_some();
        let mut composer = fluxel_shader::ShaderComposer::new();
        let (receiver, retained_assets, scene_refraction) = receiver.prepare(
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
                handle,
                feature.feature.draws(&prepared),
                assets,
                &mut composer,
                environment,
                specular,
            )
            .map_err(RendererError::ForwardPreparation)?,
        );
        let producer = order_capture_producer_draws(pipeline, &selected, inputs, producer_config)
            .map_err(RendererError::ForwardPipeline)?;
        let producer_rounds =
            producer_config.map_or(NonZeroU32::MIN, crate::DualDepthPeelingConfig::rounds);
        let receiver_rounds =
            receiver_config.map_or(NonZeroU32::MIN, crate::DualDepthPeelingConfig::rounds);
        let mut meshes = Vec::new();
        let mut feature_draws = Vec::new();
        for (index, input) in producer.draws.iter().enumerate() {
            match input.source() {
                ForwardDrawSource::Scene(_) => meshes.push(index),
                ForwardDrawSource::Feature {
                    feature: source, ..
                } if source == handle => feature_draws.push(index),
                _ => {
                    return Err(RendererError::ForwardPipeline(
                        crate::PipelineError::InvalidDrawOrder,
                    ));
                }
            }
        }
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
        let producer = self
            .assemble_capture_oit_side(
                snapshot.clone(),
                producer,
                extent,
                &retained_assets,
                None,
                true,
                producer_rounds,
                started,
            )
            .await?;
        let capture_owner = producer
            .capture_owner(&self.device, producer_slot, scene_refraction)
            .map_err(|error| {
                self.observe_forward_error("typed OIT capture product admission", error)
            })?;
        let receiver = self
            .assemble_capture_oit_side(
                snapshot.clone(),
                receiver,
                extent,
                &retained_assets,
                Some(&capture_owner),
                surface.is_some(),
                receiver_config
                    .map(crate::DualDepthPeelingConfig::rounds)
                    .unwrap_or(std::num::NonZeroU32::MIN),
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
                super::oit_execute::OitExecutionTarget::Surface { image, composite }
            }
            None => super::oit_execute::OitExecutionTarget::Readback,
        };
        self.execute_capture_oit_with_capture_contribution(
            producer,
            receiver,
            target,
            started,
            |phase| {
                let mut context = crate::ForwardCompositionContext::new(
                    phase,
                    feature.feature,
                    &prepared,
                    &meshes,
                    &feature_draws,
                );
                pipeline.contribute_composed(&selected, &mut context)
            },
            |context| {
                context.scope(&meshes, read_all)?;
                context.scope(&feature_draws, |pass| {
                    feature.feature.contribute(&prepared, pass)
                })
            },
        )
        .await
    }
}

fn read_all(
    pass: &mut crate::ForwardFeaturePass<'_, '_>,
) -> std::result::Result<(), crate::PipelineError> {
    pass.read_vertex_streams()?;
    pass.read_frame_uniforms()?;
    pass.read_sampled_textures()
}

type Result<T> = std::result::Result<T, RendererError>;
