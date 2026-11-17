//! Public access wrapper for the sorted Forward Graph phase.

use super::PipelineError;
use fluxel_rhi::backend::ResourceBackend;

/// The bounded sorted Forward authority. It declares the same material reads as
/// opaque Forward, but its color and selected depth attachment contracts cover
/// the blend reads performed later in the one real raster pass.
pub struct SortedForwardPass<'callback, 'pass, 'builder, 'frame, B: ResourceBackend> {
    pub(super) pass: &'callback mut crate::graph::OpaqueForwardPass<'pass, 'builder, 'frame, B>,
}

impl<B: ResourceBackend> SortedForwardPass<'_, '_, '_, '_, B> {
    /// Declares the dependency on the frame upload pass.
    pub fn depends_on_uploads(&mut self) -> Result<(), PipelineError> {
        self.pass
            .depends_on_uploads()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
    /// Declares the physical vertex-stream reads for the prepared draws.
    pub fn read_vertex_streams(&mut self) -> Result<(), PipelineError> {
        self.pass
            .read_vertex_streams()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
    /// Declares the prepared frame-uniform reads.
    pub fn read_frame_uniforms(&mut self) -> Result<(), PipelineError> {
        self.pass
            .read_frame_uniforms()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
    /// Declares the sampled texture reads for the prepared draws.
    pub fn read_sampled_textures(&mut self) -> Result<(), PipelineError> {
        self.pass
            .read_sampled_textures()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
    /// Declares the selected depth attachment's read/write footprint.
    pub fn write_selected_depth(&mut self) -> Result<(), PipelineError> {
        self.pass
            .write_selected_depth()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
    /// Declares the read/write color attachment used by the blend subrange.
    pub fn read_write_main_color(&mut self) -> Result<(), PipelineError> {
        self.pass
            .write_main_color()
            .map_err(|_| PipelineError::InvalidPhaseContribution)
    }
}
