use super::*;

impl<'context, 'graph, 'frame, B: ResourceBackend>
    ForwardPhaseContext<'context, 'graph, 'frame, B>
{
    /// Declares the canonical opaque phase for crate-private legacy callers.
    ///
    /// New pipeline implementations use [`Self::opaque`] so the required
    /// accesses remain visible at the policy boundary.
    pub(crate) fn opaque_forward(&mut self) -> Result<(), PipelineError> {
        self.opaque(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            pass.write_main_color()
        })
    }

    /// Declares the canonical clear-only phase for crate-private legacy callers.
    pub(crate) fn clear_only(&mut self) -> Result<(), PipelineError> {
        self.clear(|pass| {
            pass.depends_on_uploads()?;
            pass.write_main_color()
        })
    }
}
