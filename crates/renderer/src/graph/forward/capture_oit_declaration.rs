//! One same-Graph capture declaration whose producer and receiver may use OIT.
//!
//! The receiver aliases the completed producer version, rather than importing
//! the producer allocation again.  This keeps the OIT final composite, capture
//! sample, and receiver terminal in the one Graph dependency chain.

use super::capture_declaration::{
    CaptureFrameSlots, FrameHandles, SharedResidentImports, capture_texture_slots,
    declare_mip_generations, declare_raster, declare_upload, import_frame, validate_spec,
};
use super::capture_depth_alias::{import_frame_with_depth_alias, validate_depth_alias};
use super::capture_terminal::declare_capture_background_copy;
use super::*;
use fluxel_rhi::api::TextureDimension;

pub(crate) struct CaptureOitDeclaration {
    pub(super) graph: GraphBuilder,
    pub(super) lane: LaneId,
    pub(super) capture_upload: GraphPass,
    pub(super) capture_mips: Vec<super::super::mip_generation::MipGenerationPass>,
    pub(super) capture_raster: GraphPass,
    pub(super) capture_oit: Option<super::oit_plan::ForwardOitPlan>,
    pub(super) receiver_upload: GraphPass,
    pub(super) receiver_mips: Vec<super::super::mip_generation::MipGenerationPass>,
    pub(super) receiver_background_copy: Option<(GraphPass, TextureSlot)>,
    pub(super) receiver_raster: GraphPass,
    pub(super) receiver_oit: Option<super::oit_plan::ForwardOitPlan>,
    pub(super) terminal: CaptureOitTerminal,
    pub(super) capture: CaptureFrameSlots,
    pub(super) receiver: CaptureFrameSlots,
    pub(super) receiver_final: TextureSlot,
    pub(super) alignment: u64,
    pub(super) capture_shape: ForwardShape,
    pub(super) receiver_shape: ForwardShape,
    pub(super) scene_refraction: bool,
}

/// The one actual terminal selected during Graph declaration.
#[derive(Clone, Copy)]
pub(super) enum CaptureOitTerminal {
    Readback {
        pass: GraphPass,
    },
    HdrReadback(super::hdr_output::HdrReadbackTerminal),
    Surface {
        pass: GraphPass,
        slot: TextureSlot,
        extent: [u32; 2],
        format: TextureFormat,
    },
}

#[allow(clippy::too_many_arguments)]
pub(super) fn declare_capture_oit_with_capture_contribution<B, N, C>(
    device: &Device<B>,
    capture: &ForwardFrame<B>,
    receiver: &ForwardFrame<B>,
    capture_oit: Option<&ForwardOitInput<'_, B>>,
    receiver_oit: Option<&ForwardOitInput<'_, B>>,
    scene_refraction: bool,
    capture_ordinary: N,
    capture_contribute: C,
) -> std::result::Result<CaptureOitDeclaration, ForwardPlanBuildError>
where
    B: ResourceBackend,
    N: FnOnce(&mut ForwardPhaseContext<'_, '_, '_, B>) -> std::result::Result<(), PipelineError>,
    C: FnMut(&mut ForwardOitReadContext<'_, '_>) -> std::result::Result<(), PipelineError>,
{
    declare_capture_oit_inner(
        device,
        capture,
        receiver,
        capture_oit,
        receiver_oit,
        scene_refraction,
        None,
        capture_ordinary,
        capture_contribute,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn declare_capture_oit_inner<B, N, C>(
    device: &Device<B>,
    capture: &ForwardFrame<B>,
    receiver: &ForwardFrame<B>,
    capture_oit: Option<&ForwardOitInput<'_, B>>,
    receiver_oit: Option<&ForwardOitInput<'_, B>>,
    scene_refraction: bool,
    image: Option<&SurfaceImage<B>>,
    capture_ordinary: N,
    mut capture_contribute: C,
) -> std::result::Result<CaptureOitDeclaration, ForwardPlanBuildError>
where
    B: ResourceBackend,
    N: FnOnce(&mut ForwardPhaseContext<'_, '_, '_, B>) -> std::result::Result<(), PipelineError>,
    C: FnMut(&mut ForwardOitReadContext<'_, '_>) -> std::result::Result<(), PipelineError>,
{
    validate_resident_texture_devices(&capture.resident, device.id())?;
    validate_resident_texture_devices(&receiver.resident, device.id())?;
    let capture_spec = ForwardGraphSpec::from_frame(capture)?;
    let receiver_spec = ForwardGraphSpec::from_frame(receiver)?;
    if capture_spec.device != device.id()
        || receiver_spec.device != device.id()
        || capture.extent != receiver.extent
        || capture.frame.target_instance == receiver.frame.target_instance
    {
        return Err(Error::InvalidState.into());
    }
    validate_spec(device, &capture_spec, FrameTopology::SortedForward, false)?;
    validate_spec(device, &receiver_spec, FrameTopology::SortedForward, true)?;
    validate_depth_alias(
        device,
        capture,
        receiver,
        FrameTopology::SortedForward,
        scene_refraction,
    )?;
    validate_oit_input(device, capture, &capture_spec, capture_oit)?;
    validate_oit_input(device, receiver, &receiver_spec, receiver_oit)?;
    if let Some(image) = image {
        if image.device() != device.id() {
            return Err(Error::WrongDevice.into());
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated.into());
        }
        if image.extent() != receiver.extent {
            return Err(Error::InvalidRange.into());
        }
    }
    // Verify the receiver's real aliases before importing or allocating any
    // Graph product.  In particular, an OIT capture producer is the final
    // output, never its ordinary opaque attachment or an intermediate side.
    let product = capture_product(capture, capture_oit);
    if scene_refraction && product.desc.format != receiver.frame.target_desc.format {
        return Err(Error::InvalidState.into());
    }
    super::capture_bindings::validate_capture_product_alias(&product, receiver)?;
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .map(|lane| lane.id)
        .ok_or(Error::WrongLane)?;
    let mut graph = GraphBuilder::default();
    let mut shared = SharedResidentImports::default();
    let (capture_slots, capture_handles) =
        import_frame(&mut graph, &capture_spec, None, &mut shared)?;
    let (capture_upload, capture_handles) =
        declare_upload(&mut graph, &capture_spec, capture_handles)?;
    let (capture_handles, capture_mips) =
        declare_mip_generations(&mut graph, &capture_spec, capture_handles, &mut shared)?;
    let capture_dependency = capture_mips
        .last()
        .map_or(capture_upload, |generation| generation.pass);
    let (capture_raster, capture_final, capture_depth, capture_oit_plan) =
        declare_side_with_contribution(
            &mut graph,
            &capture_spec,
            &capture_handles,
            capture_upload,
            capture_dependency,
            capture_oit,
            capture_ordinary,
            &mut capture_contribute,
        )?;
    let capture_texture_slots = capture_texture_slots(&receiver_spec)?;
    let aliases = capture_texture_slots
        .iter()
        .map(|capture_texture| capture_texture.then_some(capture_final))
        .collect::<Vec<_>>();
    let (receiver_slots, receiver_handles) = if scene_refraction {
        import_frame_with_depth_alias(
            &mut graph,
            &receiver_spec,
            Some(&aliases),
            &mut shared,
            capture_depth.ok_or(Error::InvalidState)?,
        )?
    } else {
        import_frame(&mut graph, &receiver_spec, Some(&aliases), &mut shared)?
    };
    let (receiver_upload, receiver_handles) =
        declare_upload(&mut graph, &receiver_spec, receiver_handles)?;
    let (mut receiver_handles, receiver_mips) =
        declare_mip_generations(&mut graph, &receiver_spec, receiver_handles, &mut shared)?;
    let receiver_dependency = receiver_mips
        .last()
        .map_or(receiver_upload, |generation| generation.pass);
    let receiver_background_copy = scene_refraction
        .then(|| {
            declare_capture_background_copy(&mut graph, capture_final, receiver_handles.target)
        })
        .transpose()?;
    if let Some((_, target)) = receiver_background_copy {
        receiver_handles.target = target;
    }
    let (receiver_raster, receiver_final, receiver_oit_plan) = if scene_refraction {
        super::oit_stages::declare_projected_receiver(
            &mut graph,
            &receiver_spec,
            &receiver_handles,
            receiver_dependency,
            receiver_background_copy.ok_or(Error::InvalidState)?.0,
            receiver_oit,
        )?
    } else {
        let mut receiver_reads = |context: &mut ForwardOitReadContext<'_, '_>| context.all_reads();
        let (raster, color, _, oit) = declare_side_with_contribution(
            &mut graph,
            &receiver_spec,
            &receiver_handles,
            receiver_upload,
            receiver_dependency,
            receiver_oit,
            ordinary_reads::<B>,
            &mut receiver_reads,
        )?;
        (raster, color, oit)
    };
    let terminal = match image {
        None if receiver.frame.target_desc.format == TextureFormat::Rgba16Float => {
            CaptureOitTerminal::HdrReadback(super::hdr_output::declare_hdr_readback(
                &mut graph,
                receiver_final,
                receiver.frame.output.as_ref().ok_or(Error::InvalidState)?,
                receiver_handles.readback,
                receiver.extent,
                receiver.readback_row_pitch,
            )?)
        }
        None => CaptureOitTerminal::Readback {
            pass: declare_readback(
                &mut graph,
                &receiver_spec,
                receiver_final,
                receiver_handles.readback,
            )?,
        },
        Some(image) => {
            declare_surface_terminal(&mut graph, receiver_final, receiver.extent, image)?
        }
    };
    Ok(CaptureOitDeclaration {
        graph,
        lane,
        capture_upload,
        capture_mips,
        capture_raster,
        capture_oit: capture_oit_plan,
        receiver_upload,
        receiver_mips,
        receiver_background_copy: receiver_background_copy
            .map(|(pass, _)| (pass, capture_final.into())),
        receiver_raster,
        receiver_oit: receiver_oit_plan,
        terminal,
        capture: capture_slots,
        receiver: receiver_slots,
        receiver_final: receiver_final.into(),
        alignment: device.capabilities().limits.uniform_offset_alignment,
        capture_shape: capture_spec.shape,
        receiver_shape: receiver_spec.shape,
        scene_refraction,
    })
}

fn declare_surface_terminal(
    graph: &mut GraphBuilder,
    receiver_final: crate::graph::definition::TextureHandle,
    extent: [u32; 2],
    image: &SurfaceImage<impl ResourceBackend>,
) -> std::result::Result<CaptureOitTerminal, ForwardPlanBuildError> {
    let surface = graph
        .import_surface_texture(TextureDesc {
            label: Some("fluxel capture OIT surface lease".into()),
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            array_layers: 1,
            format: image.format(),
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: Vec::new(),
        })
        .map_err(|_| Error::InvalidState)?;
    let (pass, ()) = graph
        .add_pass("forward-capture-oit-surface-composite", |pass| {
            pass.read_texture_with_use(
                receiver_final,
                whole(),
                texture_read_uses().uses,
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )?;
            pass.write_texture_with_use(
                surface,
                whole(),
                color_write_uses().uses,
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )?;
            pass.side_effect();
            Ok(())
        })
        .map_err(|_| Error::InvalidState)?;
    Ok(CaptureOitTerminal::Surface {
        pass,
        slot: surface.into(),
        extent,
        format: image.format(),
    })
}

pub(super) fn declare_side_with_contribution<B, N, C>(
    graph: &mut GraphBuilder,
    spec: &ForwardGraphSpec,
    handles: &FrameHandles,
    upload: GraphPass,
    dependency: GraphPass,
    oit: Option<&ForwardOitInput<'_, B>>,
    ordinary: N,
    contribute: &mut C,
) -> std::result::Result<
    (
        GraphPass,
        crate::graph::definition::TextureHandle,
        Option<crate::graph::definition::TextureHandle>,
        Option<super::oit_plan::ForwardOitPlan>,
    ),
    ForwardPlanBuildError,
>
where
    B: ResourceBackend,
    N: FnOnce(&mut ForwardPhaseContext<'_, '_, '_, B>) -> std::result::Result<(), PipelineError>,
    C: FnMut(&mut ForwardOitReadContext<'_, '_>) -> std::result::Result<(), PipelineError>,
{
    match oit {
        Some(input) => {
            let reads = super::oit_draw_reads::OitDrawReads::from_capture_handles(handles, spec);
            let stages = super::oit_stages::declare_stages(
                graph,
                spec,
                &reads,
                input,
                dependency,
                handles.target,
                handles.depth.ok_or(Error::InvalidState)?,
                contribute,
            )?;
            Ok((
                stages.raster_pass,
                stages.output,
                Some(stages.opaque_depth),
                Some(stages.oit),
            ))
        }
        None => {
            let (raster, target, depth) = declare_raster(
                graph,
                spec,
                FrameTopology::SortedForward,
                upload,
                handles,
                ordinary,
            )?;
            Ok((raster, target, depth, None))
        }
    }
}

fn ordinary_reads<B: ResourceBackend>(
    context: &mut ForwardPhaseContext<'_, '_, '_, B>,
) -> std::result::Result<(), PipelineError> {
    context.sorted(|pass| {
        pass.depends_on_uploads()?;
        pass.read_vertex_streams()?;
        pass.read_frame_uniforms()?;
        pass.read_sampled_textures()?;
        pass.write_selected_depth()?;
        pass.read_write_main_color()
    })
}

pub(super) fn capture_product<'a, B: ResourceBackend>(
    frame: &'a ForwardFrame<B>,
    oit: Option<&'a ForwardOitInput<'a, B>>,
) -> super::capture_bindings::CaptureProductOwner<'a, B> {
    match oit {
        Some(input) => super::capture_bindings::CaptureProductOwner {
            texture: &input.output.texture,
            view: &input.output.view,
            desc: &input.output.desc,
            instance: input.output.instance,
            incoming: input.output.incoming,
            layout: input.output.layout,
            extent: frame.extent,
        },
        None => super::capture_bindings::CaptureProductOwner::from_forward_frame(frame),
    }
}

pub(super) fn validate_oit_input<B: ResourceBackend>(
    device: &Device<B>,
    frame: &ForwardFrame<B>,
    spec: &ForwardGraphSpec,
    input: Option<&ForwardOitInput<'_, B>>,
) -> std::result::Result<(), ForwardPlanBuildError> {
    let Some(input) = input else {
        return Ok(());
    };
    input.owners.validate(device.id())?;
    let peel_end = input
        .opaque_count
        .checked_add(input.peel_count)
        .ok_or(Error::InvalidRange)?;
    if input.peel_count == 0
        || peel_end > spec.draws.len()
        || input.owners.extent != frame.extent
        || input.output.device != device.id()
        || input.output.texture.device() != device.id()
        || input.output.view.device() != device.id()
        || input.output.desc.dimension != fluxel_rhi::api::TextureDimension::D2
        || input.output.desc.extent != [frame.extent[0], frame.extent[1], 1]
        || input.output.desc.array_layers != 1
        || input.output.desc.mip_levels != 1
        || input.output.desc.samples != 1
        || input.output.desc.format != frame.frame.target_desc.format
        || !input.output.desc.usage.contains(
            TextureUsage::RENDER_ATTACHMENT
                | TextureUsage::TEXTURE_BINDING
                | TextureUsage::COPY_SRC,
        )
        || input.output.view_desc.format != input.output.desc.format
        || input.output.view_desc.dimension != fluxel_rhi::api::TextureViewDimension::D2
        || input.output.view_desc.aspects != fluxel_rhi::api::TextureAspects::COLOR
        || input.output.view_desc.mips != (0..1)
        || input.output.view_desc.layers != (0..1)
        || input.output.instance == frame.frame.target_instance
        || frame
            .frame
            .depth
            .as_ref()
            .is_some_and(|depth| input.output.instance == depth.instance)
        || oit_internal_instances(input.owners).contains(&input.output.instance)
    {
        return Err(Error::InvalidState.into());
    }
    Ok(())
}

fn oit_internal_instances<B: ResourceBackend>(
    owners: &crate::residency::OitOwners<B>,
) -> [crate::graph::ResourceInstanceId; 7] {
    [
        owners.ping_pong[0].min_max.instance,
        owners.ping_pong[0].front.instance,
        owners.ping_pong[0].back.instance,
        owners.ping_pong[1].min_max.instance,
        owners.ping_pong[1].front.instance,
        owners.ping_pong[1].back.instance,
        owners.back_accumulator.instance,
    ]
}

fn declare_readback(
    graph: &mut GraphBuilder,
    spec: &ForwardGraphSpec,
    target: crate::graph::definition::TextureHandle,
    readback: crate::graph::definition::BufferHandle,
) -> std::result::Result<GraphPass, ForwardPlanBuildError> {
    let length = u64::from(spec.readback_row_pitch)
        .checked_mul(u64::from(spec.extent[1]))
        .ok_or(Error::InvalidRange)?;
    graph
        .add_pass("forward-capture-oit-readback", |pass| {
            pass.read_texture_with_use(
                target,
                whole(),
                texture_copy_read_uses().uses,
                ImageLayout::CopySrc,
                TextureUsage::COPY_SRC,
            )?;
            pass.write_with_use(
                readback,
                0..length,
                copy_write_uses().uses,
                BufferUsage::COPY_DST,
            )?;
            pass.side_effect();
            Ok(())
        })
        .map(|(pass, ())| pass)
        .map_err(|_| Error::InvalidState.into())
}
