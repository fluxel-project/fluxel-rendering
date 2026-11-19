//! Scene selection and terminal choice for ordinary Deferred submission.

use super::*;
use crate::{
    GpuIndirectDraw,
    renderer::{
        forward::{order_forward_draws, order_opaque_mask},
        indirect_producer::{plan_draws, realize_draws},
    },
};

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
    /// Submits compatible scene draws using a concrete Deferred frame policy.
    ///
    /// The policy declares geometry and lighting accesses on the actual Graph.
    /// The renderer retains resources through the covering completion.
    pub async fn submit_deferred<P: FramePipeline<ForwardResources>>(
        &mut self,
        pipeline: &P,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
    ) -> Result<Submission, RendererError> {
        self.submit_deferred_target(pipeline, snapshot, view, visibility, assets, &[], None)
            .await
    }

    /// Submits the Deferred Graph directly to an acquired surface image.
    ///
    /// The surface lease is validated before scene preparation and device work.
    pub async fn submit_deferred_surface<P: FramePipeline<ForwardResources>>(
        &mut self,
        pipeline: &P,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        image: SurfaceImage<B>,
    ) -> Result<Submission, RendererError> {
        self.submit_deferred_target(
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

    /// Produces instance counts on the GPU for selected Deferred geometry draws.
    #[allow(clippy::too_many_arguments)] // The requests are independent of scene assets and view.
    pub async fn submit_deferred_with_indirect<P: FramePipeline<ForwardResources>>(
        &mut self,
        pipeline: &P,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        requests: &[GpuIndirectDraw],
    ) -> Result<Submission, RendererError> {
        self.submit_deferred_target(pipeline, snapshot, view, visibility, assets, requests, None)
            .await
    }

    /// Produces instance counts for Deferred geometry and presents the frame.
    #[allow(clippy::too_many_arguments)] // The caller owns the surface lease and indirect requests.
    pub async fn submit_deferred_surface_with_indirect<P: FramePipeline<ForwardResources>>(
        &mut self,
        pipeline: &P,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        requests: &[GpuIndirectDraw],
        image: SurfaceImage<B>,
    ) -> Result<Submission, RendererError> {
        self.submit_deferred_target(
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

    #[allow(clippy::too_many_arguments)] // Shared admission covers the same scene, request and lease inputs.
    async fn submit_deferred_target<P: FramePipeline<ForwardResources>>(
        &mut self,
        pipeline: &P,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        requests: &[GpuIndirectDraw],
        surface: Option<SurfaceImage<B>>,
    ) -> Result<Submission, RendererError> {
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
        let selected = pipeline
            .prepare(&snapshot, view, visibility)
            .map_err(RendererError::DeferredPipeline)?;
        selected
            .validate_for(&snapshot, view)
            .map_err(RendererError::DeferredPipeline)?;
        let extent = selected.view_record().extent;
        if surface
            .as_ref()
            .is_some_and(|image| image.extent() != extent || extent.contains(&0))
        {
            return Err(Error::InvalidRange.into());
        }
        if pipeline
            .topology(&selected)
            .map_err(RendererError::DeferredPipeline)?
            != crate::FrameTopology::Deferred
        {
            return Err(RendererError::DeferredPipeline(
                crate::PipelineError::InvalidPhaseContribution,
            ));
        }
        let mut inputs = crate::preparation::prepare_forward_with_lighting(
            &selected,
            assets,
            &mut fluxel_shader::ShaderComposer::new(),
            self.environment_irradiance.is_some(),
            self.environment_specular.is_some(),
        )
        .map_err(RendererError::ForwardPreparation)?;
        if inputs.is_empty() {
            let message = if surface.is_some() {
                "Deferred surface requires visible geometry"
            } else {
                "Deferred requires visible geometry"
            };
            return Err(Error::Unsupported(message).into());
        }
        let has_overlay = inputs.iter().any(requires_forward_overlay);
        if !has_overlay {
            order_opaque_mask(&mut inputs);
        }
        let producers = plan_draws(&self.device, &snapshot, &inputs, requests)?;
        let contribute = |context: &mut crate::DeferredPhaseContext<'_, '_>| {
            pipeline.contribute_deferred(&selected, context)
        };
        if has_overlay {
            if !producers.is_empty() {
                return Err(Error::Unsupported(
                    "Deferred indirect draws require the geometry-only scene route",
                )
                .into());
            }
            let inputs = order_forward_draws(
                pipeline,
                &selected,
                inputs,
                crate::FrameTopology::SortedForward,
            )
            .map_err(RendererError::DeferredPipeline)?;
            return self
                .submit_prepared_deferred_transparent(
                    snapshot.clone(),
                    inputs,
                    extent,
                    assets,
                    surface,
                    contribute,
                )
                .await;
        }
        self.deferred_residency.retain_assets(assets);
        match surface {
            Some(image) => {
                let mut frame = self
                    .prepare_deferred_surface_frames(&inputs, image.format())
                    .await
                    .map_err(|error| self.observe_renderer_error(error))?;
                frame.indirect = realize_draws(&self.device, producers)
                    .await
                    .map_err(|error| self.observe_renderer_error(error))?;
                let retained = PendingInputs::Frame {
                    _snapshot: snapshot.clone(),
                    _inputs: inputs,
                };
                self.execute_deferred_surface(retained, frame, image, contribute)
                    .await
            }
            None => {
                let mut frame = self
                    .prepare_deferred_frames(&inputs)
                    .await
                    .map_err(|error| self.observe_renderer_error(error))?;
                frame.indirect = realize_draws(&self.device, producers)
                    .await
                    .map_err(|error| self.observe_renderer_error(error))?;
                let retained = PendingInputs::Frame {
                    _snapshot: snapshot.clone(),
                    _inputs: inputs,
                };
                self.execute_deferred_probe(retained, frame, contribute)
                    .await
            }
        }
    }
}
