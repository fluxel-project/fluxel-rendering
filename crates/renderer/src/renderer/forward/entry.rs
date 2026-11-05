//! Built-in Scene selection for the shared Forward execution path.
use super::*;
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
        self.submit_forward_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            None,
            TextureFormat::Rgba8Unorm,
        )
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
        if image.device() != self.device.id() {
            return Err(Error::WrongDevice.into());
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated.into());
        }
        let format = image.format();
        let extent = image.extent();
        self.submit_forward_target(
            pipeline,
            snapshot,
            view,
            visibility,
            assets,
            Some((image, extent)),
            format,
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
        surface: Option<(SurfaceImage<B>, [u32; 2])>,
        color_format: TextureFormat,
    ) -> std::result::Result<super::Submission, RendererError> {
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
        if let Some((_, surface_extent)) = surface.as_ref() {
            if extent != *surface_extent || extent.contains(&0) {
                return Err(Error::InvalidRange.into());
            }
        }
        let topology = pipeline
            .topology(&selected)
            .map_err(RendererError::ForwardPipeline)?;
        let prepared_inputs = match topology {
            crate::FrameTopology::OpaqueForward => {
                crate::prepare_forward(&selected, assets, &mut fluxel_shader::ShaderComposer::new())
                    .map_err(RendererError::ForwardPreparation)?
            }
            crate::FrameTopology::ClearOnly => Vec::new(),
            crate::FrameTopology::Deferred => {
                return Err(RendererError::ForwardPipeline(
                    crate::PipelineError::InvalidPhaseContribution,
                ));
            }
        };
        self.submit_prepared_forward(
            snapshot.clone(),
            prepared_inputs,
            extent,
            topology,
            assets,
            surface,
            color_format,
            preparation_started,
            |context| pipeline.contribute(&selected, context),
        )
        .await
    }
}
