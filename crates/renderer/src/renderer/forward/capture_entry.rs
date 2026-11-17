//! Public submission entry for one same-Graph Forward capture receiver.

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
    /// Submits capture and receiver work to one acquired Surface lease.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_capture_surface<
        P: crate::FramePipeline<crate::ForwardResources>,
    >(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        receiver: crate::PreparedCaptureReceiver,
        image: SurfaceImage<B>,
    ) -> std::result::Result<super::Submission, RendererError> {
        self.ensure_open()?;
        if image.device() != self.device.id() {
            return Err(Error::WrongDevice.into());
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated.into());
        }
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
        if topology != crate::FrameTopology::OpaqueForward {
            return Err(RendererError::ForwardPipeline(
                crate::PipelineError::InvalidPhaseContribution,
            ));
        }
        let extent = selected.view_record().extent;
        if extent != image.extent() {
            return Err(Error::InvalidRange.into());
        }
        let receiver_input = receiver.into_draw_input();
        if receiver_input.extent() != extent {
            return Err(Error::InvalidRange.into());
        }
        let capture_inputs = crate::preparation::prepare_forward_with_lighting(
            &selected,
            assets,
            &mut fluxel_shader::ShaderComposer::new(),
            self.environment_irradiance.is_some(),
            self.environment_specular.is_some(),
        )
        .map_err(RendererError::ForwardPreparation)?;
        let capture_inputs = super::composed::order_draws(pipeline, &selected, capture_inputs)
            .map_err(RendererError::ForwardPipeline)?;
        let environment = self.environment_irradiance.clone().filter(|_| {
            capture_inputs
                .iter()
                .any(|input| input.prepared().uses_environment_irradiance())
        });
        self.submit_prepared_forward_capture(
            snapshot.clone(),
            capture_inputs,
            receiver_input,
            extent,
            topology,
            depth_mode,
            assets,
            Some(image),
            preparation_started,
            environment,
            |context| pipeline.contribute(&selected, context),
        )
        .await
    }

    /// Submits a capture producer and its material receiver in one Graph.
    ///
    /// The receiver has already resolved its exact geometry and material
    /// generations. Its texture binding is replaced only with the producer
    /// frame's target owner; no CPU texture is synthesized or uploaded.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_capture<P: crate::FramePipeline<crate::ForwardResources>>(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        receiver: crate::PreparedCaptureReceiver,
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
        let depth_mode = pipeline.forward_depth_mode(&selected);
        let topology = pipeline
            .topology(&selected)
            .map_err(RendererError::ForwardPipeline)?;
        if topology != crate::FrameTopology::OpaqueForward {
            return Err(RendererError::ForwardPipeline(
                crate::PipelineError::InvalidPhaseContribution,
            ));
        }
        let capture_inputs = crate::preparation::prepare_forward_with_lighting(
            &selected,
            assets,
            &mut fluxel_shader::ShaderComposer::new(),
            self.environment_irradiance.is_some(),
            self.environment_specular.is_some(),
        )
        .map_err(RendererError::ForwardPreparation)?;
        let extent = selected.view_record().extent;
        let receiver_input = receiver.into_draw_input();
        if receiver_input.extent() != extent {
            return Err(Error::InvalidRange.into());
        }
        let capture_inputs = super::composed::order_draws(pipeline, &selected, capture_inputs)
            .map_err(RendererError::ForwardPipeline)?;
        let capture_environment = self.environment_irradiance.clone().filter(|_| {
            capture_inputs
                .iter()
                .any(|input| input.prepared().uses_environment_irradiance())
        });
        self.submit_prepared_forward_capture(
            snapshot.clone(),
            capture_inputs,
            receiver_input,
            extent,
            topology,
            depth_mode,
            assets,
            None,
            preparation_started,
            capture_environment,
            |context| pipeline.contribute(&selected, context),
        )
        .await
    }
}
