//! Built-in Scene selection for the shared Forward execution path.
use super::*;
use crate::{GpuIndirectDraw, renderer::indirect_producer::plan_draws};
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
    /// Prepares and submits the bounded material-backed opaque Forward path.
    ///
    /// Scene records name exact CPU asset generations. The concrete pipeline
    /// selects candidates; only this renderer records and submits their Graph.
    /// Each draw resolves its own exact texture generation.
    pub async fn submit_forward<P: crate::FramePipeline<crate::ForwardResources>>(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
    ) -> std::result::Result<super::Submission, RendererError> {
        self.submit_forward_target(pipeline, snapshot, view, visibility, assets, &[], None)
            .await
    }

    /// Submits Forward work to an acquired surface image.
    ///
    /// The acquired lease is the authority for the attachment format and
    /// extent; callers cannot supply substitute surface facts.
    #[allow(clippy::too_many_arguments)] // The scene, view, visibility, assets, and lease are separate ownership inputs.
    pub async fn submit_forward_surface<P: crate::FramePipeline<crate::ForwardResources>>(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        image: SurfaceImage<B>,
    ) -> std::result::Result<super::Submission, RendererError> {
        self.submit_forward_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            &[],
            Some(image),
        )
        .await
    }

    /// Produces instance counts on the GPU for selected Forward scene draws.
    #[allow(clippy::too_many_arguments)] // Indirect requests are independent of the scene assets.
    pub async fn submit_forward_with_indirect<P: crate::FramePipeline<crate::ForwardResources>>(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        requests: &[GpuIndirectDraw],
    ) -> std::result::Result<super::Submission, RendererError> {
        self.submit_forward_target(pipeline, snapshot, view, visibility, assets, requests, None)
            .await
    }

    /// Produces instance counts for Forward scene draws and presents the frame.
    #[allow(clippy::too_many_arguments)] // The caller owns the surface lease and indirect requests.
    pub async fn submit_forward_surface_with_indirect<
        P: crate::FramePipeline<crate::ForwardResources>,
    >(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        requests: &[GpuIndirectDraw],
        image: SurfaceImage<B>,
    ) -> std::result::Result<super::Submission, RendererError> {
        self.submit_forward_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            requests,
            Some(image),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)] // Shared offscreen/surface preparation keeps their ownership paths identical.
    async fn submit_forward_target<P: crate::FramePipeline<crate::ForwardResources>>(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        indirect_draws: &[GpuIndirectDraw],
        surface: Option<SurfaceImage<B>>,
    ) -> std::result::Result<super::Submission, RendererError> {
        if let Some(image) = surface.as_ref() {
            if image.device() != self.device.id() {
                return Err(Error::WrongDevice.into());
            }
            if !image.is_current() {
                return Err(Error::SurfaceOutdated.into());
            }
        }
        self.ensure_open()?;
        self.free_slot().ok_or(RendererError::Backpressure)?;
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
            .is_some_and(|image| extent != image.extent() || extent.contains(&0))
        {
            return Err(Error::InvalidRange.into());
        }
        let depth_mode = pipeline.forward_depth_mode(&selected);
        let topology = pipeline
            .topology(&selected)
            .map_err(RendererError::ForwardPipeline)?;
        let use_environment = self.environment_irradiance.is_some();
        let prepared_inputs = match topology {
            crate::FrameTopology::OpaqueForward | crate::FrameTopology::SortedForward => {
                crate::preparation::prepare_forward_with_lighting(
                    &selected,
                    assets,
                    &mut fluxel_shader::ShaderComposer::new(),
                    use_environment,
                    self.environment_specular.is_some(),
                )
                .map_err(RendererError::ForwardPreparation)?
            }
            crate::FrameTopology::ClearOnly => Vec::new(),
            crate::FrameTopology::Deferred => {
                return Err(RendererError::ForwardPipeline(
                    crate::PipelineError::InvalidPhaseContribution,
                ));
            }
        };
        let prepared_inputs =
            super::composed::order_draws(pipeline, &selected, prepared_inputs, topology)
                .map_err(RendererError::ForwardPipeline)?;
        let environment_snapshot = self.environment_irradiance.clone().filter(|_| {
            prepared_inputs
                .iter()
                .any(|input| input.prepared().uses_environment_irradiance())
        });
        let indirect = plan_draws(&self.device, &snapshot, &prepared_inputs, indirect_draws)?;
        let color_format = surface
            .as_ref()
            .map_or(TextureFormat::Rgba8Unorm, SurfaceImage::format);
        self.submit_prepared_forward_with_indirect(
            snapshot.clone(),
            prepared_inputs,
            extent,
            topology,
            depth_mode,
            self.forward_sample_count(),
            assets,
            surface.map(|image| (image, extent)),
            color_format,
            preparation_started,
            environment_snapshot,
            indirect,
            |context| pipeline.contribute(&selected, context),
        )
        .await
    }
}
