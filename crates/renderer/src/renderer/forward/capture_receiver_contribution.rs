//! Graph declaration policy for the capture receiver draw.

use super::*;

pub(in crate::renderer) fn receiver_contribution<B: ResourceBackend>(
    context: &mut crate::ForwardPhaseContext<'_, '_, '_, B>,
    topology: crate::FrameTopology,
    reads_producer_depth: bool,
) -> std::result::Result<(), crate::PipelineError> {
    match topology {
        crate::FrameTopology::OpaqueForward => context.opaque(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            pass.write_selected_depth()?;
            pass.write_main_color()
        }),
        crate::FrameTopology::SortedForward => context.sorted(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            if reads_producer_depth {
                pass.read_selected_depth()?;
            } else {
                pass.write_selected_depth()?;
            }
            pass.read_write_main_color()
        }),
        _ => Err(crate::PipelineError::InvalidPhaseContribution),
    }
}
