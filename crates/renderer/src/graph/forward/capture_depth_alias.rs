//! Depth-alias validation and attachment import for scene-refraction capture.

use super::capture_declaration::{
    CaptureFrameSlots, FrameHandles, SharedResidentImports, import_frame_inner,
};
use super::*;

pub(in crate::graph) fn same_texture_owner_desc(
    left: &fluxel_rhi::api::TextureDesc,
    right: &fluxel_rhi::api::TextureDesc,
) -> bool {
    left.dimension == right.dimension
        && left.extent == right.extent
        && left.array_layers == right.array_layers
        && left.format == right.format
        && left.mip_levels == right.mip_levels
        && left.samples == right.samples
        && left.usage == right.usage
        && left.view_formats == right.view_formats
}

pub(in crate::graph) fn same_buffer_owner_desc(
    left: &fluxel_rhi::api::BufferDesc,
    right: &fluxel_rhi::api::BufferDesc,
) -> bool {
    left.size == right.size && left.usage == right.usage && left.memory == right.memory
}

pub(super) fn validate_depth_alias<B: ResourceBackend>(
    device: &Device<B>,
    capture: &ForwardFrame<B>,
    receiver: &ForwardFrame<B>,
    receiver_topology: FrameTopology,
    enabled: bool,
) -> Result<()> {
    if enabled
        && (receiver_topology != FrameTopology::SortedForward
            || !matches!(
                (&capture.frame.depth, &receiver.frame.depth),
                (Some(capture_depth), Some(receiver_depth))
                    if capture_depth.device == device.id()
                        && receiver_depth.device == device.id()
                        && capture_depth.instance == receiver_depth.instance
                        && capture_depth.texture.device() == device.id()
                        && receiver_depth.texture.device() == device.id()
                        && capture_depth.view.device() == device.id()
                        && receiver_depth.view.device() == device.id()
                        && capture_depth.incoming == receiver_depth.incoming
                        && capture_depth.layout == receiver_depth.layout
                        && same_texture_owner_desc(&capture_depth.desc, &receiver_depth.desc)
            ))
    {
        return Err(Error::InvalidState);
    }
    Ok(())
}

pub(super) fn import_frame_with_depth_alias(
    graph: &mut GraphBuilder,
    spec: &ForwardGraphSpec,
    aliases: Option<&[Option<crate::graph::definition::TextureHandle>]>,
    shared: &mut SharedResidentImports,
    depth: crate::graph::definition::TextureHandle,
) -> std::result::Result<(CaptureFrameSlots, FrameHandles), ForwardPlanBuildError> {
    let target = graph
        .import_texture_with_state(
            spec.target.desc.clone(),
            spec.target.incoming,
            spec.target.layout,
        )
        .map_err(|_| Error::InvalidState)?;
    import_frame_inner(graph, spec, aliases, shared, Some((target, depth)))
}
