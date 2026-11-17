//! Program-cache selection for one assembled Forward frame.

use super::*;

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
    /// Retains only pipelines selected by this frame's concrete draw facts.
    pub(super) fn retain_programs_for(
        &mut self,
        inputs: &[crate::ForwardDrawInput],
        color_format: TextureFormat,
        depth_mode: crate::ForwardDepthMode,
    ) {
        let active: std::collections::HashSet<_> = inputs
            .iter()
            .map(|input| {
                ForwardProgramKey::for_material(
                    input.prepared(),
                    color_format,
                    input.transform_uniform_bytes(),
                    depth_mode,
                    crate::renderer::raster_surface_state(input),
                )
            })
            .collect();
        self.forward_programs.retain(|key, _| active.contains(key));
    }
}
