//! Preparation and one-pass composition for two statically typed features.

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
    /// Submits meshes and two independently typed features in one Forward frame.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_with_features<P, F, G>(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        features: (FeatureBinding<'_, F>, FeatureBinding<'_, G>),
    ) -> Result<super::Submission>
    where
        P: FramePipeline<crate::ForwardResources>,
        F: RenderFeature,
        G: RenderFeature,
    {
        self.submit_composed_pair_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            features,
            None,
            TextureFormat::Rgba8Unorm,
        )
        .await
    }

    /// Submits the pair composition to a host-acquired Surface image.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_surface_with_features<P, F, G>(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        features: (FeatureBinding<'_, F>, FeatureBinding<'_, G>),
        image: SurfaceImage<B>,
    ) -> Result<super::Submission>
    where
        P: FramePipeline<crate::ForwardResources>,
        F: RenderFeature,
        G: RenderFeature,
    {
        if image.device() != self.device.id() {
            return Err(Error::WrongDevice.into());
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated.into());
        }
        let format = image.format();
        let extent = image.extent();
        self.submit_composed_pair_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            features,
            Some((image, extent)),
            format,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn submit_composed_pair_target<P, F, G>(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        features: (FeatureBinding<'_, F>, FeatureBinding<'_, G>),
        surface: Option<(SurfaceImage<B>, [u32; 2])>,
        color_format: TextureFormat,
    ) -> Result<super::Submission>
    where
        P: FramePipeline<crate::ForwardResources>,
        F: RenderFeature,
        G: RenderFeature,
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
                .is_some_and(|(_, expected)| *expected != extent)
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

        let environment_snapshot = self.environment_irradiance.clone();
        let mut composer = fluxel_shader::ShaderComposer::new();
        let mut inputs = crate::preparation::prepare_forward_with_lighting(
            &selected,
            assets,
            &mut composer,
            environment_snapshot.is_some(),
            self.environment_specular.is_some(),
        )
        .map_err(RendererError::ForwardPreparation)?;
        inputs.extend(
            crate::preparation::prepare_feature_draws_with_lighting(
                &selected,
                first_handle,
                first.feature.draws(&first_prepared),
                assets,
                &mut composer,
                environment_snapshot.is_some(),
                self.environment_specular.is_some(),
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
                environment_snapshot.is_some(),
                self.environment_specular.is_some(),
            )
            .map_err(RendererError::ForwardPreparation)?,
        );
        let inputs = super::composed::order_draws(pipeline, &selected, inputs, topology)
            .map_err(RendererError::ForwardPipeline)?;
        let environment_snapshot = environment_snapshot.filter(|_| {
            inputs
                .iter()
                .any(|input| input.prepared().uses_environment_irradiance())
        });
        let mut meshes = Vec::new();
        let mut first_draws = Vec::new();
        let mut second_draws = Vec::new();
        for (index, input) in inputs.iter().enumerate() {
            match input.source() {
                ForwardDrawSource::Scene(_) => meshes.push(index),
                ForwardDrawSource::Feature { feature, .. } if feature == first_handle => {
                    first_draws.push(index)
                }
                ForwardDrawSource::Feature { feature, .. } if feature == second_handle => {
                    second_draws.push(index)
                }
                ForwardDrawSource::Feature { .. } => {
                    return Err(RendererError::ForwardPipeline(
                        crate::PipelineError::InvalidDrawOrder,
                    ));
                }
            }
        }

        self.submit_prepared_forward(
            snapshot.clone(),
            inputs,
            extent,
            topology,
            depth_mode,
            assets,
            surface,
            color_format,
            preparation_started,
            environment_snapshot,
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
        )
        .await
    }
}

type Result<T> = std::result::Result<T, RendererError>;
