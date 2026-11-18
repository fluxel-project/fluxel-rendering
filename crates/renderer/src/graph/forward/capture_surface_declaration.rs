//! Surface terminal declaration for same-Graph Forward capture.
use super::super::mip_generation::MipGenerationPass;
use super::capture_declaration::{
    capture_texture_slots, declare_mip_generations, declare_raster, declare_upload, import_frame,
    validate_spec,
};
use super::*;
use fluxel_rhi::api::TextureDimension;

use super::capture_depth_alias::{import_frame_with_depth_alias, validate_depth_alias};
use super::capture_terminal::declare_capture_background_copy;
pub(crate) struct CaptureForwardSurfaceDeclaration {
    pub(super) graph: GraphBuilder,
    pub(super) lane: LaneId,
    pub(super) capture_upload: GraphPass,
    pub(super) capture_mip_generations: Vec<MipGenerationPass>,
    pub(super) capture_raster: GraphPass,
    pub(super) receiver_upload: GraphPass,
    pub(super) receiver_mip_generations: Vec<MipGenerationPass>,
    pub(super) receiver_background_copy: Option<GraphPass>,
    pub(super) receiver_depth_alias: bool,
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
    preserve_capture_background: bool,
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
    validate_resident_texture_devices(&capture.resident, device.id())?;
    validate_resident_texture_devices(&receiver.resident, device.id())?;
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
    {
        return Err(Error::InvalidState.into());
    }
    validate_depth_alias(
        device,
        capture,
        receiver,
        receiver_topology,
        preserve_capture_background,
    )?;
    let receiver_capture = capture_texture_slots(&receiver_spec)?;
    super::capture_bindings::validate_capture_alias(capture, receiver, preserve_capture_background)?;
    validate_spec(device, &capture_spec, capture_topology, false)?;
    validate_spec(device, &receiver_spec, receiver_topology, true)?;
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .map(|lane| lane.id)
        .ok_or(Error::WrongLane)?;
    let mut graph = GraphBuilder::default();
    let mut shared = super::capture_declaration::SharedResidentImports::default();
    let (capture_slots, capture_handles) =
        import_frame(&mut graph, &capture_spec, None, &mut shared)?;
    let (capture_upload, capture_handles) =
        declare_upload(&mut graph, &capture_spec, capture_handles)?;
    let (capture_handles, capture_mip_generations) =
        declare_mip_generations(&mut graph, &capture_spec, capture_handles, &mut shared)?;
    let (capture_raster, capture_target, capture_depth) = declare_raster(
        &mut graph,
        &capture_spec,
        capture_topology,
        capture_upload,
        &capture_handles,
        capture_contribute,
    )?;
    let (aliases, refraction_mips) = super::capture_refraction_mips::prepare_aliases(
        &mut graph,
        &receiver_spec,
        &receiver_capture,
        capture_target,
        preserve_capture_background,
    )?;
    let capture_depth = preserve_capture_background
        .then_some(capture_depth)
        .flatten();
    let (receiver_slots, receiver_handles) = match capture_depth {
        Some(depth) => import_frame_with_depth_alias(
            &mut graph,
            &receiver_spec,
            Some(&aliases),
            &mut shared,
            depth,
        )?,
        None => import_frame(&mut graph, &receiver_spec, Some(&aliases), &mut shared)?,
    };
    let (receiver_upload, receiver_handles) =
        declare_upload(&mut graph, &receiver_spec, receiver_handles)?;
    let (mut receiver_handles, mut receiver_mip_generations) =
        declare_mip_generations(&mut graph, &receiver_spec, receiver_handles, &mut shared)?;
    super::capture_refraction_mips::append_chain(
        &mut graph,
        &mut receiver_handles,
        refraction_mips,
        &mut receiver_mip_generations,
    )?;
    let receiver_background_copy = preserve_capture_background
        .then(|| {
            declare_capture_background_copy(&mut graph, capture_target, receiver_handles.target)
        })
        .transpose()?;
    if let Some((_, target)) = receiver_background_copy {
        receiver_handles.target = target;
    }
    let (receiver_raster, receiver_target, _) = declare_raster(
        &mut graph,
        &receiver_spec,
        receiver_topology,
        receiver_upload,
        &receiver_handles,
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
        capture_mip_generations,
        capture_raster,
        receiver_upload,
        receiver_mip_generations,
        receiver_background_copy: receiver_background_copy.map(|(pass, _)| pass),
        receiver_depth_alias: capture_depth.is_some(),
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
