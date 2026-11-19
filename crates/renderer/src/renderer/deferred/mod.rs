//! Deferred lighting, transparency and capture for immutable Scene and Material assets.
//!
//! Opaque geometry and lighting own the G-buffer prefix. Forward transparency
//! and capture namespaces extend the same Graph and retain actual owners
//! through the receiver terminal's covering completion.

mod cache;
mod capture_empty_prepare;
mod capture_entry;
mod capture_execute;
mod capture_feature;
mod capture_feature_entry;
mod capture_forward_execute;
mod capture_oit_entry;
mod capture_prepare;
mod capture_reverse_execute;
pub(in crate::renderer) use capture_prepare::DeferredCaptureComposition;
mod deformation;
mod deformed;
mod draw_assembly;
mod execute;
mod msaa;
mod oit_entry;
mod oit_execute;
mod oit_prepare;
mod prepare;
mod program;
mod resident_assembly;
mod scene_entry;
mod scene_prepare;
mod surface_prepare;
mod templates;
mod transparent_entry;
mod transparent_execute;

pub use cache::DeferredCacheStats;
pub(crate) use program::DeferredProgramCache;

#[cfg(test)]
#[path = "../../tests/renderer/deferred_preparation.rs"]
mod test_preparation;

use super::{PendingInputs, Renderer, RendererError, Submission};
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

/// Selects the ordinary Deferred Forward overlay. Environment transmission
/// uses its selected IBL bindings there; scene-color refraction has a separate
/// capture composition entry because it needs a distinct sampled scene image.
fn requires_forward_overlay(input: &crate::ForwardDrawInput) -> bool {
    (input.prepared().uses_transmission() && !input.prepared().uses_scene_refraction())
        || matches!(
            input.effective_alpha_mode(),
            fluxel_material::SurfaceAlphaMode::Blend | fluxel_material::SurfaceAlphaMode::MaskBlend,
        )
}

/// Observed owner and compiled-Graph facts for the last accepted Deferred probe.
///
/// This is diagnostic evidence for the selected Deferred path, not permission
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
            super::PendingWork::DeferredSurface { _frame, plan, .. } => {
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
}
