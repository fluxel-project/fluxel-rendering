//! Explicit dual-depth-peeling submission for the built-in CPU scene pipeline.

use super::*;
use crate::FramePipeline;
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
    /// Submits the built-in scene consumer with finite dual depth peeling.
    ///
    /// The pipeline supplies visibility and scene ordering. This entry owns
    /// the required depth attachment and OIT pass selection. Excluded blend
    /// draws follow the ordinary sorted path after OIT composition; only that
    /// fallback range needs a camera position for distance ordering. Selected
    /// OIT capability or program failures remain errors.
    ///
    /// This entry handles scene records. Typed-feature and capture
    /// composition require their corresponding OIT integration.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_oit<V>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        config: &crate::DualDepthPeelingConfig,
    ) -> std::result::Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
    {
        self.submit_forward_oit_target(pipeline, snapshot, view, visibility, assets, config, None)
            .await
    }

    /// Presents scene-only dual depth peeling to this acquired image in the
    /// same submission. No intermediate CPU readback is performed.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_oit_surface<V>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        config: &crate::DualDepthPeelingConfig,
        image: SurfaceImage<B>,
    ) -> std::result::Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
    {
        self.submit_forward_oit_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            config,
            Some(image),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn submit_forward_oit_target<V>(
        &mut self,
        pipeline: &crate::ForwardCpuPipeline<V>,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        config: &crate::DualDepthPeelingConfig,
        surface: Option<SurfaceImage<B>>,
    ) -> std::result::Result<Submission, RendererError>
    where
        V: crate::VisibilityProvider<crate::ForwardResources>,
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
        let preparation_started = Instant::now();
        let selected = pipeline
            .prepare(&snapshot, view, visibility)
            .map_err(RendererError::ForwardPipeline)?;
        selected
            .validate_for(&snapshot, view)
            .map_err(RendererError::ForwardPipeline)?;
        let extent = selected.view_record().extent;
        if surface
            .as_ref()
            .is_some_and(|image| image.extent() != extent)
        {
            return Err(Error::InvalidRange.into());
        }
        let inputs = crate::preparation::prepare_forward_with_lighting(
            &selected,
            assets,
            &mut fluxel_shader::ShaderComposer::new(),
            self.environment_irradiance.is_some(),
            self.environment_specular.is_some(),
        )
        .map_err(RendererError::ForwardPreparation)?;
        let partition = super::oit_order::order_oit_draws(pipeline, &selected, inputs, config)
            .map_err(RendererError::ForwardPipeline)?;
        self.submit_prepared_oit(
            snapshot.clone(),
            partition,
            extent,
            assets,
            config,
            surface,
            preparation_started,
            |context| pipeline.contribute(&selected, context),
            |context| context.all_reads(),
        )
        .await
    }
}
