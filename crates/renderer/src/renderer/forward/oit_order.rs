//! OIT partition ordering atop the ordinary Forward ordering authority.

use crate::{ForwardDrawInput, FramePipeline};

/// Validates all inputs, then sorts only the ordinary opaque/excluded ranges.
/// Peel candidates deliberately never enter distance sorting: OIT resolves
/// their depth order on the GPU.
pub(in crate::renderer) fn order_oit_draws<P: FramePipeline<crate::ForwardResources>>(
    pipeline: &P,
    frame: &crate::PreparedFrame<'_, crate::ForwardResources>,
    inputs: Vec<ForwardDrawInput>,
    config: &crate::DualDepthPeelingConfig,
) -> std::result::Result<crate::preparation::OitDrawPartition, crate::PipelineError> {
    super::composed::validate_draw_sources(&inputs)?;
    let partition = crate::preparation::partition_oit_draws(inputs, config);
    let opaque_count = partition.opaque_count;
    let peel_count = partition.peel_count;
    let mut ordinary = partition.draws;
    let mut peel = ordinary.split_off(opaque_count);
    let excluded = peel.split_off(peel_count);
    ordinary.extend(excluded);
    let mut ordered = super::composed::order_draws(
        pipeline,
        frame,
        ordinary,
        crate::FrameTopology::SortedForward,
    )?;
    let excluded = ordered.split_off(opaque_count);
    ordered.extend(peel);
    ordered.extend(excluded);
    Ok(crate::preparation::OitDrawPartition {
        draws: ordered,
        opaque_count,
        peel_count,
    })
}
