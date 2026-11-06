//! Typed external deformation admission and shared Deferred submission.

use super::*;
use crate::{DeferredPositionDeformer, DeformationBinding};
use fluxel_rhi::api::SurfaceImage;

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
    /// Runs a typed external GPU position producer and Deferred in one Graph.
    ///
    /// Each submission owns its immutable source, parameters and generated
    /// vertex stream until the completion covering the consumer. Preparation
    /// validates scene/issuer/version, shader interfaces and exact device support
    /// before allocating resources. This bounded route accepts one packed
    /// POSITION Float32x3 mesh and has no CPU intermediate readback.
    #[allow(clippy::too_many_arguments)] // Typed Feature input has its own issuer and lifetime.
    pub async fn submit_deferred_with_deformation<P, F>(
        &mut self,
        pipeline: &P,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        feature: DeformationBinding<'_, F>,
    ) -> Result<Submission, RendererError>
    where
        P: FramePipeline<ForwardResources>,
        F: DeferredPositionDeformer,
    {
        self.ensure_open()?;
        self.free_slot().ok_or(RendererError::Backpressure)?;
        let selected = pipeline
            .prepare(&snapshot, view, visibility)
            .map_err(RendererError::DeferredPipeline)?;
        selected
            .validate_for(&snapshot, view)
            .map_err(RendererError::DeferredPipeline)?;
        if pipeline
            .topology(&selected)
            .map_err(RendererError::DeferredPipeline)?
            != crate::FrameTopology::Deferred
        {
            return Err(RendererError::DeferredPipeline(
                crate::PipelineError::InvalidPhaseContribution,
            ));
        }
        let handle = feature.feature.handle();
        let context = feature
            .issuer
            .context(&snapshot, crate::CpuPhase::Reference, handle)
            .map_err(RendererError::DeferredFeature)?;
        crate::validate_context(&context, feature.input, handle)
            .map_err(RendererError::DeferredFeature)?;
        let request = feature
            .feature
            .prepare(&context, feature.input, &selected)
            .map_err(RendererError::DeferredFeature)?;
        let inputs =
            crate::prepare_forward(&selected, assets, &mut fluxel_shader::ShaderComposer::new())
                .map_err(RendererError::ForwardPreparation)?;
        let [input] = inputs.as_slice() else {
            return Err(Error::Unsupported(
                "Deferred deformation requires exactly one visible mesh",
            )
            .into());
        };
        // No backend resources exist for this request until both the shared
        // input and the external producer's complete CPU contract are admitted.
        let producer = self.plan_deformation(input, &request)?;
        self.deferred_residency.retain_assets(assets);
        let mut frame = self
            .prepare_deferred_frame(input)
            .await
            .map_err(|error| self.observe_renderer_error(error))?;
        frame.deformation = Some(self.realize_deformation(producer).await?);
        self.execute_deferred_probe(
            PendingInputs::Frame {
                _snapshot: snapshot.clone(),
                _inputs: inputs,
            },
            frame,
            |context| pipeline.contribute_deformed(&selected, feature.feature, context),
        )
        .await
    }

    /// Runs a typed position producer and Deferred directly to a surface.
    ///
    /// The acquired image remains a transient present lease. Its device,
    /// generation and extent are accepted before producer realization or any
    /// GPU owner allocation.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_deferred_surface_with_deformation<P, F>(
        &mut self,
        pipeline: &P,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        feature: DeformationBinding<'_, F>,
        image: SurfaceImage<B>,
    ) -> Result<Submission, RendererError>
    where
        P: FramePipeline<ForwardResources>,
        F: DeferredPositionDeformer,
    {
        if image.device() != self.device.id() {
            return Err(Error::WrongDevice.into());
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated.into());
        }
        let surface_extent = image.extent();
        let surface_format = image.format();
        self.ensure_open()?;
        self.free_slot().ok_or(RendererError::Backpressure)?;
        let selected = pipeline
            .prepare(&snapshot, view, visibility)
            .map_err(RendererError::DeferredPipeline)?;
        selected
            .validate_for(&snapshot, view)
            .map_err(RendererError::DeferredPipeline)?;
        if selected.view_record().extent != surface_extent || surface_extent.contains(&0) {
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
        let handle = feature.feature.handle();
        let context = feature
            .issuer
            .context(&snapshot, crate::CpuPhase::Reference, handle)
            .map_err(RendererError::DeferredFeature)?;
        crate::validate_context(&context, feature.input, handle)
            .map_err(RendererError::DeferredFeature)?;
        let request = feature
            .feature
            .prepare(&context, feature.input, &selected)
            .map_err(RendererError::DeferredFeature)?;
        let inputs =
            crate::prepare_forward(&selected, assets, &mut fluxel_shader::ShaderComposer::new())
                .map_err(RendererError::ForwardPreparation)?;
        let [input] = inputs.as_slice() else {
            return Err(Error::Unsupported(
                "Deferred surface deformation requires exactly one visible mesh",
            )
            .into());
        };
        if crate::pbr::is_pbr(input.material().value().output()) {
            return Err(Error::Unsupported(
                "Deferred surface deformation supports Color POSITION-only geometry, not PBR",
            )
            .into());
        }
        let producer = self.plan_deformation(input, &request)?;
        self.deferred_residency.retain_assets(assets);
        let mut frame = self
            .prepare_deferred_surface_frame(input, surface_format)
            .await
            .map_err(|error| self.observe_renderer_error(error))?;
        frame.deformation = Some(self.realize_deformation(producer).await?);
        self.execute_deferred_surface(
            PendingInputs::Frame {
                _snapshot: snapshot.clone(),
                _inputs: inputs,
            },
            frame,
            image,
            |context| pipeline.contribute_deformed(&selected, feature.feature, context),
        )
        .await
    }
}
