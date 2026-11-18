//! Per-frame Forward program-cache selection.

use std::collections::HashSet;

use fluxel_rhi::api::TextureFormat;

use super::ForwardProgramKey;

/// Returns every concrete fixed-function program the pending frame may use.
pub(super) fn active_programs(
    inputs: &[crate::ForwardDrawInput],
    color_format: TextureFormat,
    depth_mode: crate::ForwardDepthMode,
    samples: u32,
) -> HashSet<ForwardProgramKey> {
    inputs
        .iter()
        .map(|input| {
            ForwardProgramKey::for_material(
                input.prepared(),
                color_format,
                input.transform_uniform_bytes(),
                depth_mode,
                samples,
                crate::renderer::raster_surface_state(input),
            )
        })
        .collect()
}
