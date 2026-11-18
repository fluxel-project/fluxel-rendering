//! Typed feature preparation for one same-Graph Forward capture producer.

use super::*;
use crate::{FeatureBinding, ForwardDrawSource, FramePipeline, RenderFeature};
use std::time::Instant;

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
    /// Captures built-in meshes and one concrete typed feature, then shades a
    /// material receiver from that capture in one Graph submission.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_capture_with_feature<P, F>(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        feature: FeatureBinding<'_, F>,
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
    ) -> Result<Submission>
    where
        P: FramePipeline<crate::ForwardResources>,
        F: RenderFeature,
    {
        self.submit_composed_capture_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            feature,
            producer_slot,
            receiver,
            None,
        )
        .await
    }

    /// Presents a typed capture composition to a host-acquired Surface image.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_capture_surface_with_feature<P, F>(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        feature: FeatureBinding<'_, F>,
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
        image: SurfaceImage<B>,
    ) -> Result<Submission>
    where
        P: FramePipeline<crate::ForwardResources>,
        F: RenderFeature,
    {
        if image.device() != self.device.id() {
            return Err(Error::WrongDevice.into());
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated.into());
        }
        self.submit_composed_capture_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            feature,
            producer_slot,
            receiver,
            Some(image),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn submit_composed_capture_target<P, F>(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        feature: FeatureBinding<'_, F>,
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
        surface: Option<SurfaceImage<B>>,
    ) -> Result<Submission>
    where
        P: FramePipeline<crate::ForwardResources>,
        F: RenderFeature,
    {
        self.ensure_open()?;
        self.free_slot().ok_or(RendererError::Backpressure)?;
        let preparation_started = Instant::now();
        let selected = pipeline
            .prepare(&snapshot, view, visibility)
            .map_err(RendererError::ForwardPipeline)?;
        selected
            .validate_for(&snapshot, view)
            .map_err(RendererError::ForwardPipeline)?;
        let depth_mode = pipeline.forward_depth_mode(&selected);
        let topology = pipeline
            .topology(&selected)
            .map_err(RendererError::ForwardPipeline)?;
        if !matches!(
            topology,
            crate::FrameTopology::OpaqueForward | crate::FrameTopology::SortedForward
        ) {
            return Err(RendererError::ForwardPipeline(
                crate::PipelineError::FeatureRequiresOpaque,
            ));
        }
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
        let receiver_input = receiver
            .with_lighting(&mut composer, environment, specular)
            .map_err(RendererError::ForwardPreparation)?
            .into_draw_input();
        super::capture_entry::validate_receiver_sources(&receiver_input, &producer_slot)?;
        let receiver_topology = super::capture_entry::receiver_topology(&receiver_input)?;
        let retained_assets = super::capture_entry::capture_assets(assets, &receiver_input);
        if receiver_input.extent() != extent {
            return Err(Error::InvalidRange.into());
        }

        let mut capture_inputs = crate::preparation::prepare_forward_with_lighting(
            &selected,
            assets,
            &mut composer,
            environment,
            specular,
        )
        .map_err(RendererError::ForwardPreparation)?;
        capture_inputs.extend(
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
        let capture_inputs =
            super::composed::order_draws(pipeline, &selected, capture_inputs, topology)
                .map_err(RendererError::ForwardPipeline)?;
        let mut meshes = Vec::new();
        let mut feature_draws = Vec::new();
        for (index, input) in capture_inputs.iter().enumerate() {
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
        let capture_environment = self.environment_irradiance.clone().filter(|_| {
            capture_inputs
                .iter()
                .any(|input| input.prepared().uses_environment_irradiance())
        });
        self.submit_prepared_forward_capture(
            snapshot.clone(),
            capture_inputs,
            receiver_input,
            producer_slot,
            extent,
            topology,
            receiver_topology,
            depth_mode,
            &retained_assets,
            surface,
            preparation_started,
            capture_environment,
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
        )
        .await
    }
}

type Result<T> = std::result::Result<T, RendererError>;
