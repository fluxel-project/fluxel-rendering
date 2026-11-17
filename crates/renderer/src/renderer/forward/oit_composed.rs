//! One typed feature in the actual opaque, peel and excluded OIT read scopes.

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
    /// Submits scene and one typed feature through finite dual depth peeling.
    /// Each actual material stage invokes the feature in its restricted scope.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_oit_with_feature<V, F>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        feature: FeatureBinding<'_, F>,
        config: &crate::DualDepthPeelingConfig,
    ) -> Result<Submission>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
        F: RenderFeature,
    {
        self.submit_composed_oit_target(
            pipeline, snapshot, view, visibility, assets, feature, config, None,
        )
        .await
    }

    /// Presents the typed feature OIT composition in the same submission.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_oit_surface_with_feature<V, F>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        feature: FeatureBinding<'_, F>,
        config: &crate::DualDepthPeelingConfig,
        image: SurfaceImage<B>,
    ) -> Result<Submission>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
        F: RenderFeature,
    {
        self.submit_composed_oit_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            feature,
            config,
            Some(image),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn submit_composed_oit_target<V, F>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        feature: FeatureBinding<'_, F>,
        config: &crate::DualDepthPeelingConfig,
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
        let mut composer = fluxel_shader::ShaderComposer::new();
        let environment = self.environment_irradiance.is_some();
        let specular = self.environment_specular.is_some();
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
        let partition = super::oit_order::order_oit_draws(pipeline, &selected, inputs, config)
            .map_err(RendererError::ForwardPipeline)?;
        let mut meshes = Vec::new();
        let mut feature_draws = Vec::new();
        for (index, input) in partition.draws.iter().enumerate() {
            match input.source() {
                ForwardDrawSource::Scene(_) => meshes.push(index),
                ForwardDrawSource::Feature { feature, .. } if feature == handle => {
                    feature_draws.push(index)
                }
                _ => {
                    return Err(RendererError::ForwardPipeline(
                        crate::PipelineError::InvalidDrawOrder,
                    ));
                }
            }
        }
        self.submit_prepared_oit(
            snapshot.clone(),
            partition,
            extent,
            assets,
            config,
            surface,
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
