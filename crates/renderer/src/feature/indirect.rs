//! Scene selection for GPU-generated indirect material draw arguments.

use fluxel_shader::SourceUnit;

use crate::RenderId;

/// Selects one scene draw whose active instance count is produced on the GPU.
///
/// The named WGSL unit defines
/// `visible_instance_count(parameters: vec4<u32>, capacity: u32) -> u32`.
/// The renderer clamps its result to the prepared instance capacity and
/// generates the indirect command from the validated geometry range.
/// A live record outside the view's selected draw set contributes no work.
#[derive(Clone, Debug)]
pub struct GpuIndirectDraw {
    draw: RenderId,
    source: SourceUnit,
    parameters: [u32; 4],
}

impl GpuIndirectDraw {
    /// Selects a scene record and supplies its GPU count function and values.
    #[must_use]
    pub fn new(draw: RenderId, source: SourceUnit, parameters: [u32; 4]) -> Self {
        Self {
            draw,
            source,
            parameters,
        }
    }

    pub(crate) fn draw(&self) -> RenderId {
        self.draw
    }

    pub(crate) fn source(&self) -> &SourceUnit {
        &self.source
    }

    pub(crate) fn parameters(&self) -> [u32; 4] {
        self.parameters
    }
}
