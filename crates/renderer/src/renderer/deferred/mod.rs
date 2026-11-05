//! Provisional one-mesh Deferred consumer of the existing Scene/Material assets.
//!
//! The fixed geometry and lighting passes prove real MRT, depth sampling and
//! completion ownership before a wider public pipeline authority is chosen.
//! No Forward Graph, independent scheduler or CPU G-buffer round-trip is used.

mod cache;
mod deformation;
mod deformed;
mod execute;
mod prepare;
mod program;
mod surface_prepare;
mod templates;

pub use cache::DeferredCacheStats;
pub(crate) use program::DeferredProgramCache;

#[cfg(test)]
#[path = "../../tests/renderer/deferred_preparation.rs"]
mod test_preparation;

use super::{Renderer, RendererError, Submission};
use crate::{
    ForwardAssets, ForwardResources, FramePipeline, FrameSnapshot, RenderBounds, RenderViewId,
};
use fluxel_rhi::{
    api::{Error, PlanId, SurfaceImage, TextureFormat},
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};

/// Observed owner and compiled-Graph facts for the last accepted Deferred probe.
///
/// This is diagnostic evidence for the provisional one-mesh path, not permission
/// to name or mutate internal Graph resources.
#[derive(Clone, Copy, Debug)]
pub struct DeferredProbeFacts {
    /// Actual RHI execution plan identity.
    pub plan: PlanId,
    /// Number of passes in the compiled Graph.
    pub passes: usize,
    /// Number of buffer bindings required by the compiled Graph.
    pub buffer_bindings: usize,
    /// Number of texture bindings required by the compiled Graph.
    pub texture_bindings: usize,
    /// Formats saved by the albedo, normal and depth allocation owners.
    pub gbuffer_formats: [TextureFormat; 3],
    /// Extents saved by those three allocation owners.
    pub gbuffer_extents: [[u32; 3]; 3],
}

impl<B> Renderer<B>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    /// Returns facts from the last accepted provisional Deferred submission.
    pub fn deferred_probe_facts(&self) -> Option<DeferredProbeFacts> {
        self.deferred_probe_facts
    }

    /// Describes the actual Graph and owners of a pending Deferred probe.
    ///
    /// The text is diagnostic, not a stable serialization or resource authority.
    /// Read it before retiring the submission. Other workload types return None.
    pub fn deferred_probe_diagnostic(
        &self,
        submission: &Submission,
    ) -> Result<Option<String>, RendererError> {
        if !std::sync::Arc::ptr_eq(&self.owner, &submission.owner)
            || submission.device != self.device.id()
        {
            return Err(RendererError::ForeignSubmission);
        }
        let pending = self.pending_for(submission)?;
        Ok(match &pending.workload {
            super::PendingWork::Deferred { _frame, plan, .. } => Some(plan.diagnostic(_frame)),
            super::PendingWork::DeferredSurface { _frame, plan } => {
                Some(plan.surface_diagnostic(_frame))
            }
            _ => None,
        })
    }
}

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
    /// Submits the provisional one-mesh, offscreen Deferred proof path.
    ///
    /// Uses the same immutable assets and CPU scene selection as Forward. It
    /// writes albedo/normal/depth, samples them in a separate lighting pass, then
    /// reads the final color through the normal completion-owned submission.
    /// Multi-draw policy and Surface remain unproved. The separate deformation
    /// entry accepts the selected position-only Feature.
    pub async fn submit_deferred_probe(
        &mut self,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
    ) -> Result<Submission, RendererError> {
        self.submit_deferred(
            &crate::DeferredCpuPipeline::new(crate::BoundsVisibility),
            snapshot,
            view,
            visibility,
            assets,
        )
        .await
    }

    /// Submits one mesh using a concrete Deferred frame policy.
    ///
    /// The policy selects from the same immutable scene/material inputs as
    /// Forward and declares geometry/lighting accesses on the actual private
    /// Graph. Resources remain owned by this renderer through completion.
    /// Color consumes position-only geometry; MetallicRoughness requires
    /// POSITION/NORMAL geometry and uses the selected fixed world-space light.
    /// This entry does not accept external GPU products or Surface; position
    /// deformation has a separate entry and remains restricted to Color.
    pub async fn submit_deferred<P: FramePipeline<ForwardResources>>(
        &mut self,
        pipeline: &P,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
    ) -> Result<Submission, RendererError> {
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
        let inputs =
            crate::prepare_forward(&selected, assets, &mut fluxel_shader::ShaderComposer::new())
                .map_err(RendererError::ForwardPreparation)?;
        let [input] = inputs.as_slice() else {
            return Err(
                Error::Unsupported("Deferred probe requires exactly one visible mesh").into(),
            );
        };
        self.deferred_residency.retain_assets(assets);
        let frame = self
            .prepare_deferred_frame(input)
            .await
            .map_err(|error| self.observe_renderer_error(error))?;
        self.execute_deferred_probe((snapshot.clone(), inputs), frame, |context| {
            pipeline.contribute_deferred(&selected, context)
        })
        .await
    }

    /// Submits the Deferred graph directly to an acquired surface image.
    ///
    /// The lease is checked before scene preparation, shader admission, or
    /// allocation.  The private preparation route owns only dynamic frame
    /// buffers and the G-buffer; the image remains a transient present lease.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_deferred_surface<P: FramePipeline<ForwardResources>>(
        &mut self,
        pipeline: &P,
        snapshot: FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
        assets: &ForwardAssets,
        image: SurfaceImage<B>,
    ) -> Result<Submission, RendererError> {
        if image.device() != self.device.id() {
            return Err(Error::WrongDevice.into());
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated.into());
        }
        let format = image.format();
        let surface_extent = image.extent();
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
        let inputs =
            crate::prepare_forward(&selected, assets, &mut fluxel_shader::ShaderComposer::new())
                .map_err(RendererError::ForwardPreparation)?;
        let [input] = inputs.as_slice() else {
            return Err(
                Error::Unsupported("Deferred surface requires exactly one visible mesh").into(),
            );
        };
        self.deferred_residency.retain_assets(assets);
        let frame = self
            .prepare_deferred_surface_frame(input, format)
            .await
            .map_err(|error| self.observe_renderer_error(error))?;
        self.execute_deferred_surface((snapshot.clone(), inputs), frame, image, |context| {
            pipeline.contribute_deferred(&selected, context)
        })
        .await
    }
}
