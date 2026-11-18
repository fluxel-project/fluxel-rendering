//! Real dual-depth-peeling submission for two statically typed features.

use super::*;
use crate::{FeatureBinding, ForwardDrawSource, RenderFeature};
use crate::{ForwardDrawInput, FramePipeline};
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
    /// Submits scene and two independently typed features through finite OIT.
    /// Every actual material stage invokes both restricted feature scopes.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_oit_with_features<V, F, G>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        features: (FeatureBinding<'_, F>, FeatureBinding<'_, G>),
        config: &crate::DualDepthPeelingConfig,
    ) -> std::result::Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
        F: RenderFeature,
        G: RenderFeature,
    {
        self.submit_oit_pair_target(
            pipeline, snapshot, view, visibility, assets, features, config, None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    /// Presents the dual-feature OIT composition in the same submission.
    pub async fn submit_forward_oit_surface_with_features<V, F, G>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        features: (FeatureBinding<'_, F>, FeatureBinding<'_, G>),
        config: &crate::DualDepthPeelingConfig,
        image: SurfaceImage<B>,
    ) -> std::result::Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
        F: RenderFeature,
        G: RenderFeature,
    {
        if image.device() != self.device.id() {
            return Err(Error::WrongDevice.into());
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated.into());
        }
        self.submit_oit_pair_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            features,
            config,
            Some(image),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn submit_oit_pair_target<V, F, G>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        features: (FeatureBinding<'_, F>, FeatureBinding<'_, G>),
        config: &crate::DualDepthPeelingConfig,
        surface: Option<SurfaceImage<B>>,
    ) -> std::result::Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
        F: RenderFeature,
        G: RenderFeature,
    {
        self.ensure_open()?;
        self.free_slot().ok_or(RendererError::Backpressure)?;
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
        let partition = super::oit_order::order_oit_draws(pipeline, &selected, inputs, config)
            .map_err(RendererError::ForwardPipeline)?;
        let (meshes, first_draws, second_draws) =
            pair_draws(&partition.draws, first_handle, second_handle)?;
        self.submit_prepared_oit(
            snapshot.clone(),
            partition,
            extent,
            assets,
            config,
            surface,
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
            |ctx| {
                ctx.scope(&meshes, read_all)?;
                ctx.scope(&first_draws, |pass| {
                    first.feature.contribute(&first_prepared, pass)
                })?;
                ctx.scope(&second_draws, |pass| {
                    second.feature.contribute(&second_prepared, pass)
                })?;
                Ok(())
            },
        )
        .await
    }
}

fn read_all(
    pass: &mut crate::pipeline::ForwardFeaturePass<'_, '_>,
) -> std::result::Result<(), crate::PipelineError> {
    pass.read_vertex_streams()?;
    pass.read_frame_uniforms()?;
    pass.read_sampled_textures()
}

pub(super) fn pair_draws(
    draws: &[ForwardDrawInput],
    first: crate::FeatureHandle,
    second: crate::FeatureHandle,
) -> std::result::Result<(Vec<usize>, Vec<usize>, Vec<usize>), RendererError> {
    let mut meshes = Vec::new();
    let mut first_draws = Vec::new();
    let mut second_draws = Vec::new();
    for (index, input) in draws.iter().enumerate() {
        match input.source() {
            ForwardDrawSource::Scene(_) => meshes.push(index),
            ForwardDrawSource::Feature { feature, .. } if feature == first => {
                first_draws.push(index)
            }
            ForwardDrawSource::Feature { feature, .. } if feature == second => {
                second_draws.push(index)
            }
            ForwardDrawSource::Feature { .. } => {
                return Err(RendererError::ForwardPipeline(
                    crate::PipelineError::InvalidDrawOrder,
                ));
            }
        }
    }
    Ok((meshes, first_draws, second_draws))
}
