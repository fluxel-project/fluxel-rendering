//! Surface terminal declaration for same-Graph Forward capture.
use super::capture_declaration::{
    FrameHandles, declare_raster, declare_upload, import_frame, same_texture_owner_desc,
    validate_spec,
};
use super::*;
use fluxel_rhi::api::TextureDimension;

pub(crate) struct CaptureForwardSurfaceDeclaration {
    pub(super) graph: GraphBuilder,
    pub(super) lane: LaneId,
    pub(super) capture_upload: GraphPass,
    pub(super) capture_raster: GraphPass,
    pub(super) receiver_upload: GraphPass,
    pub(super) receiver_raster: GraphPass,
    pub(super) composite: GraphPass,
    pub(super) capture: CaptureFrameSlots,
    pub(super) receiver: CaptureFrameSlots,
    pub(super) surface_target: TextureSlot,
    pub(super) capture_topology: FrameTopology,
    pub(super) receiver_topology: FrameTopology,
    pub(super) alignment: u64,
    pub(super) capture_shape: ForwardShape,
    pub(super) receiver_shape: ForwardShape,
    pub(super) extent: [u32; 2],
    pub(super) color_format: TextureFormat,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn declare_capture_surface_forward<B, Capture, Receiver>(
    device: &Device<B>,
    capture: &ForwardFrame<B>,
    receiver: &ForwardFrame<B>,
    image: &SurfaceImage<B>,
    capture_topology: FrameTopology,
    receiver_topology: FrameTopology,
    capture_contribute: Capture,
    receiver_contribute: Receiver,
) -> std::result::Result<CaptureForwardSurfaceDeclaration, ForwardPlanBuildError>
where
    B: ResourceBackend,
    Capture:
        FnOnce(&mut ForwardPhaseContext<'_, '_, '_, B>) -> std::result::Result<(), PipelineError>,
    Receiver:
        FnOnce(&mut ForwardPhaseContext<'_, '_, '_, B>) -> std::result::Result<(), PipelineError>,
{
    if image.device() != device.id() {
        return Err(Error::WrongDevice.into());
    }
    if !image.is_current() {
        return Err(Error::SurfaceOutdated.into());
    }
    if image.extent() != receiver.extent {
        return Err(Error::InvalidRange.into());
    }
    let capture_spec = ForwardGraphSpec::from_frame(capture)?;
    let receiver_spec = ForwardGraphSpec::from_frame(receiver)?;
    if capture_spec.device != device.id()
        || receiver_spec.device != device.id()
        || capture.extent != receiver.extent
        || capture.frame.target_instance == receiver.frame.target_instance
        || capture.frame.target_desc.format != receiver.frame.target_desc.format
        || !capture
            .frame
            .target_desc
            .usage
            .contains(TextureUsage::TEXTURE_BINDING)
        || receiver_spec.textures.len() != 1
        || !receiver_spec.texture_uploads.is_empty()
        || receiver_spec.draws.iter().any(|draw| draw.texture != 0)
    {
        return Err(Error::InvalidState.into());
    }
    let alias = receiver
        .resident
        .textures
        .first()
        .ok_or(Error::InvalidState)?;
    if alias.instance != capture.frame.target_instance
        || alias.texture.device() != capture.frame.target.device()
        || alias.view.device() != capture.frame.target_view.device()
        || !same_texture_owner_desc(&alias.desc, &capture.frame.target_desc)
        || alias.incoming != capture.frame.target_incoming
        || alias.layout != capture.frame.target_layout
    {
        return Err(Error::InvalidState.into());
    }
    validate_spec(device, &capture_spec, capture_topology, false)?;
    validate_spec(device, &receiver_spec, receiver_topology, true)?;
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .map(|lane| lane.id)
        .ok_or(Error::WrongLane)?;
    let mut graph = GraphBuilder::default();
    let (capture_slots, capture_handles) = import_frame(&mut graph, &capture_spec, true)?;
    let (mut receiver_slots, mut receiver_handles) =
        import_frame(&mut graph, &receiver_spec, false)?;
    let (capture_upload, capture_handles) =
        declare_upload(&mut graph, &capture_spec, capture_handles)?;
    let (capture_raster, capture_target) = declare_raster(
        &mut graph,
        &capture_spec,
        capture_topology,
        capture_upload,
        &capture_handles,
        capture_contribute,
    )?;
    receiver_slots.textures = vec![TextureSlot::from(capture_target)];
    receiver_handles.textures = vec![capture_target];
    let (receiver_upload, receiver_handles) =
        declare_upload(&mut graph, &receiver_spec, receiver_handles)?;
    let (receiver_raster, receiver_target) = declare_raster(
        &mut graph,
        &receiver_spec,
        receiver_topology,
        receiver_upload,
        &FrameHandles {
            textures: vec![capture_target],
            ..receiver_handles.clone()
        },
        receiver_contribute,
    )?;
    let surface_target = graph
        .import_surface_texture(TextureDesc {
            label: Some("fluxel forward capture surface lease".into()),
            dimension: TextureDimension::D2,
            extent: [image.extent()[0], image.extent()[1], 1],
            array_layers: 1,
            format: image.format(),
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: Vec::new(),
        })
        .map_err(|_| Error::InvalidState)?;
    let (composite, _) = graph
        .add_pass("forward-capture-surface-composite", |pass| {
            pass.read_texture_with_use(
                receiver_target,
                whole(),
                texture_read_uses().uses,
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )?;
            pass.write_texture_with_use(
                surface_target,
                whole(),
                color_write_uses().uses,
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )?;
            // This is a real surface attachment write. Marking it observed
            // keeps that terminal write (and all of its producer chain) live;
            // it is not a side-effect-only placeholder.
            pass.side_effect();
            Ok(())
        })
        .map_err(|_| Error::InvalidState)?;
    Ok(CaptureForwardSurfaceDeclaration {
        graph,
        lane,
        capture_upload,
        capture_raster,
        receiver_upload,
        receiver_raster,
        composite,
        capture: capture_slots,
        receiver: receiver_slots,
        surface_target: TextureSlot::from(surface_target),
        capture_topology,
        receiver_topology,
        alignment: device.capabilities().limits.uniform_offset_alignment,
        capture_shape: capture_spec.shape,
        receiver_shape: receiver_spec.shape,
        extent: receiver.extent,
        color_format: image.format(),
    })
}
