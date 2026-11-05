//! External policy for the bounded position-only Deferred Graph product.
//!
//! This type selects and declares the existing Deferred product; it cannot
//! obtain Graph resources, record commands, or submit GPU work itself.

use fluxel_renderer::{
    DeferredPhaseContext, ForwardCpuPipeline, FramePipeline, FrameSnapshot, FrameTopology,
    PipelineError, PreparedFrame, RenderBounds, RenderViewId,
};

/// An external Deferred policy using the same interval visibility convention
/// as the existing external Forward consumers.
#[derive(Clone, Debug, Default)]
pub struct ExternalDeferredPipeline(ForwardCpuPipeline<super::IntervalVisibility>);

impl ExternalDeferredPipeline {
    /// Creates the external policy for the real four-pass Deferred product.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<R> FramePipeline<R> for ExternalDeferredPipeline {
    fn prepare<'a>(
        &self,
        snapshot: &'a FrameSnapshot<R>,
        view: RenderViewId,
        visibility: RenderBounds,
    ) -> Result<PreparedFrame<'a, R>, PipelineError> {
        self.0.prepare(snapshot, view, visibility)
    }

    fn topology(&self, _: &PreparedFrame<'_, R>) -> Result<FrameTopology, PipelineError> {
        Ok(FrameTopology::Deferred)
    }

    fn contribute_deferred(
        &self,
        _: &PreparedFrame<'_, R>,
        context: &mut DeferredPhaseContext<'_, '_>,
    ) -> Result<(), PipelineError> {
        context.geometry(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            pass.write_gbuffer()
        })?;
        context.lighting(|pass| {
            pass.read_gbuffer()?;
            pass.write_main_color()
        })
    }
}
