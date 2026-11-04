//! Consumer-owned policies for the public Forward pipeline boundary.
//!
//! The types in this crate deliberately provide scene policy only. They cannot
//! construct a Graph, record commands, or submit work.

use fluxel_renderer::{
    ForwardCpuPipeline, ForwardPhaseContext, FramePipeline, FrameSnapshot, FrameTopology,
    PipelineError, RenderBounds, RenderId, VisibilityProvider,
};

pub mod quad;
pub use quad::{
    ExternalComposedPipeline, PreparedQuads, QuadAssets, QuadFeature, QuadInput, QuadMarker,
    quad_geometry_data, quad_marker_material_definition,
};

/// A concrete external opaque policy using interval-separation visibility.
#[derive(Clone, Debug, Default)]
pub struct ExternalOpaquePipeline(ForwardCpuPipeline<IntervalVisibility>);

impl ExternalOpaquePipeline {
    /// Creates the external opaque policy.
    pub fn new() -> Self {
        Self::default()
    }
}

/// A concrete external clear policy using the same candidate preparation.
#[derive(Clone, Debug, Default)]
pub struct ExternalClearPipeline(ForwardCpuPipeline<IntervalVisibility>);

impl ExternalClearPipeline {
    /// Creates the external clear policy.
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct IntervalVisibility;

impl<R> FramePipeline<R> for ExternalOpaquePipeline {
    fn prepare<'a>(
        &self,
        snapshot: &'a FrameSnapshot<R>,
        view: fluxel_renderer::RenderViewId,
        visibility: RenderBounds,
    ) -> Result<fluxel_renderer::PreparedFrame<'a, R>, PipelineError> {
        self.0.prepare(snapshot, view, visibility)
    }

    fn topology(
        &self,
        _: &fluxel_renderer::PreparedFrame<'_, R>,
    ) -> Result<FrameTopology, PipelineError> {
        Ok(FrameTopology::OpaqueForward)
    }

    fn contribute<B: fluxel_rhi::backend::ResourceBackend>(
        &self,
        _: &fluxel_renderer::PreparedFrame<'_, R>,
        context: &mut ForwardPhaseContext<'_, '_, '_, B>,
    ) -> Result<(), PipelineError> {
        context.opaque(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            pass.write_main_color()
        })
    }
}

impl<R> FramePipeline<R> for ExternalClearPipeline {
    fn prepare<'a>(
        &self,
        snapshot: &'a FrameSnapshot<R>,
        view: fluxel_renderer::RenderViewId,
        visibility: RenderBounds,
    ) -> Result<fluxel_renderer::PreparedFrame<'a, R>, PipelineError> {
        self.0.prepare(snapshot, view, visibility)
    }

    fn topology(
        &self,
        _: &fluxel_renderer::PreparedFrame<'_, R>,
    ) -> Result<FrameTopology, PipelineError> {
        Ok(FrameTopology::ClearOnly)
    }

    fn contribute<B: fluxel_rhi::backend::ResourceBackend>(
        &self,
        _: &fluxel_renderer::PreparedFrame<'_, R>,
        context: &mut ForwardPhaseContext<'_, '_, '_, B>,
    ) -> Result<(), PipelineError> {
        context.clear(|pass| {
            pass.depends_on_uploads()?;
            pass.write_main_color()
        })
    }
}

impl<R> VisibilityProvider<R> for IntervalVisibility {
    fn visible(
        &self,
        snapshot: &FrameSnapshot<R>,
        volume: RenderBounds,
        candidates: &[RenderId],
    ) -> Result<Vec<RenderId>, PipelineError> {
        let mut result = Vec::new();
        for &id in candidates.iter().rev() {
            let bound = snapshot.record(id).map_err(PipelineError::Scene)?.bounds;
            // This consumer treats touching the supplied volume boundary as
            // outside. Its conservative convention is part of the external
            // policy, and makes the S5 moved-to-clip-boundary record cull.
            let separated = (0..3).any(|axis| {
                bound.maximum()[axis] <= volume.minimum()[axis]
                    || volume.maximum()[axis] <= bound.minimum()[axis]
            });
            if !separated {
                result.push(id);
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
#[path = "tests/quad.rs"]
mod quad_tests;
