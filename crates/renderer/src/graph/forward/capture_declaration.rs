//! One Graph declaration for a captured Forward texture and its receiver.

use fluxel_rhi::{
    api::{Device, Error, ImageLayout, LaneClass, LaneId, Result, TextureUsage},
    backend::ResourceBackend,
};
use std::collections::HashMap;

use crate::{
    FrameTopology,
    pipeline::{ForwardPhaseContext, PipelineError},
    residency::ForwardFrame,
};

use super::super::mip_generation::MipGenerationPass;
use super::capture_depth_alias::{import_frame_with_depth_alias, validate_depth_alias};
pub(in crate::graph) use super::capture_depth_alias::{
    same_buffer_owner_desc, same_texture_owner_desc,
};
use super::capture_terminal::{declare_capture_background_copy, declare_readback};
use super::hdr_output::{HdrReadbackTerminal, declare_hdr_readback};
use super::{
    BufferGraphFact, BufferSlot, ForwardGraphSpec, ForwardPhaseDriver, ForwardPlanBuildError,
    GraphBuilder, GraphPass, TextureGraphFact, TextureSlot, validate_depth_spec,
    validate_depth_topology, validate_forward_topology, validate_graph_draw,
    validate_resident_texture_devices,
};

/// All Graph slots imported for one Forward frame namespace.
pub(super) struct CaptureFrameSlots {
    pub(super) resident_upload: BufferSlot,
    pub(super) frame_upload: BufferSlot,
    pub(super) uniform: BufferSlot,
    pub(super) environment_upload: Option<BufferSlot>,
    pub(super) specular_upload: Option<BufferSlot>,
    pub(super) specular_view_uniform: Option<BufferSlot>,
    pub(super) streams: Vec<BufferSlot>,
    pub(super) textures: Vec<TextureSlot>,
    pub(super) environment: Option<TextureSlot>,
    pub(super) specular_radiance: Option<TextureSlot>,
    pub(super) specular_brdf_lut: Option<TextureSlot>,
    pub(super) depth: Option<TextureSlot>,
    pub(super) target: TextureSlot,
    pub(super) readback: BufferSlot,
}

/// Actual resident imports which may be reused by namespaces in one Graph.
///
/// Only imports whose owner is already complete are retained here.  Resources
/// with this frame's upload work deliberately get a fresh handle/version so a
/// second namespace cannot observe an unrecorded or partially written product.
#[derive(Default)]
pub(in crate::graph) struct SharedResidentImports {
    streams: HashMap<
        crate::graph::ResourceInstanceId,
        (crate::graph::definition::BufferHandle, BufferGraphFact),
    >,
    pub(super) textures: HashMap<
        crate::graph::ResourceInstanceId,
        (crate::graph::definition::TextureHandle, TextureGraphFact),
    >,
}

#[path = "capture_shared_imports.rs"]
mod capture_shared_imports;

/// The five-pass same-submission capture contract.
pub(crate) struct CaptureForwardDeclaration {
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
    pub(super) hdr_terminal: Option<HdrReadbackTerminal>,
    pub(super) receiver_readback: GraphPass,
    pub(super) capture: CaptureFrameSlots,
    pub(super) receiver: CaptureFrameSlots,
    pub(super) capture_topology: FrameTopology,
    pub(super) receiver_topology: FrameTopology,
    pub(super) alignment: u64,
    pub(super) capture_shape: super::ForwardShape,
    pub(super) receiver_shape: super::ForwardShape,
}

/// Declares capture raster -> receiver sampled read -> receiver readback in one graph.
#[allow(clippy::too_many_arguments)]
pub(super) fn declare_capture_forward<B, Capture, Receiver>(
    device: &Device<B>,
    capture: &ForwardFrame<B>,
    receiver: &ForwardFrame<B>,
    capture_topology: FrameTopology,
    receiver_topology: FrameTopology,
    preserve_capture_background: bool,
    capture_contribute: Capture,
    receiver_contribute: Receiver,
) -> std::result::Result<CaptureForwardDeclaration, ForwardPlanBuildError>
where
    B: ResourceBackend,
    Capture:
        FnOnce(&mut ForwardPhaseContext<'_, '_, '_, B>) -> std::result::Result<(), PipelineError>,
    Receiver:
        FnOnce(&mut ForwardPhaseContext<'_, '_, '_, B>) -> std::result::Result<(), PipelineError>,
{
    validate_resident_texture_devices(&capture.resident, device.id())?;
    validate_resident_texture_devices(&receiver.resident, device.id())?;
    let capture_spec = ForwardGraphSpec::from_frame(capture)?;
    let receiver_spec = ForwardGraphSpec::from_frame(receiver)?;
    if capture_spec.device != device.id() || receiver_spec.device != device.id() {
        return Err(Error::WrongDevice.into());
    }
    if capture.extent != receiver.extent
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
    // Validate every named alias against the actual producer before adding
    // either namespace to the Graph, including aliases beyond the first slot.
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
    let mut shared = SharedResidentImports::default();
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
    let hdr_terminal = (receiver.frame.target_desc.format
        == fluxel_rhi::api::TextureFormat::Rgba16Float)
        .then(|| {
            declare_hdr_readback(
                &mut graph,
                receiver_target,
                receiver.frame.output.as_ref().ok_or(Error::InvalidState)?,
                receiver_handles.readback,
                receiver.extent,
                receiver.readback_row_pitch,
            )
        })
        .transpose()?;
    let receiver_readback = match hdr_terminal {
        Some(terminal) => terminal.readback_pass,
        None => declare_readback(
            &mut graph,
            &receiver_spec,
            receiver_target,
            receiver_handles.readback,
        )?,
    };
    Ok(CaptureForwardDeclaration {
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
        hdr_terminal,
        receiver_readback,
        capture: capture_slots,
        receiver: receiver_slots,
        capture_topology,
        receiver_topology,
        alignment: device.capabilities().limits.uniform_offset_alignment,
        capture_shape: capture_spec.shape,
        receiver_shape: receiver_spec.shape,
    })
}

pub(super) fn capture_texture_slots(spec: &ForwardGraphSpec) -> Result<Vec<bool>> {
    let mut slots = vec![false; spec.textures.len()];
    let mut seen = vec![false; spec.textures.len()];
    for draw in &spec.draws {
        if draw.textures.len() != draw.capture_textures.len() {
            return Err(Error::InvalidRange);
        }
        for (&texture, &capture) in draw.textures.iter().zip(&draw.capture_textures) {
            let slot = slots.get_mut(texture).ok_or(Error::InvalidRange)?;
            if seen[texture] && *slot != capture {
                return Err(Error::InvalidState);
            }
            *slot = capture;
            seen[texture] = true;
        }
    }
    slots
        .iter()
        .any(|capture| *capture)
        .then_some(slots)
        .ok_or(Error::InvalidState)
}

#[derive(Clone)]
pub(super) struct FrameHandles {
    pub(super) resident_upload: crate::graph::definition::BufferHandle,
    pub(super) frame_upload: crate::graph::definition::BufferHandle,
    pub(super) uniform: crate::graph::definition::BufferHandle,
    pub(super) environment_upload: Option<crate::graph::definition::BufferHandle>,
    pub(super) specular_upload: Option<crate::graph::definition::BufferHandle>,
    pub(super) specular_view_uniform: Option<crate::graph::definition::BufferHandle>,
    pub(super) streams: Vec<crate::graph::definition::BufferHandle>,
    pub(super) textures: Vec<crate::graph::definition::TextureHandle>,
    pub(super) cold_textures: Vec<bool>,
    pub(super) environment: Option<crate::graph::definition::TextureHandle>,
    pub(super) specular_radiance: Option<crate::graph::definition::TextureHandle>,
    pub(super) specular_brdf_lut: Option<crate::graph::definition::TextureHandle>,
    pub(super) depth: Option<crate::graph::definition::TextureHandle>,
    pub(super) target: crate::graph::definition::TextureHandle,
    pub(super) readback: crate::graph::definition::BufferHandle,
}

pub(super) fn validate_spec<B: ResourceBackend>(
    device: &Device<B>,
    spec: &ForwardGraphSpec,
    topology: FrameTopology,
    receiver: bool,
) -> std::result::Result<(), ForwardPlanBuildError> {
    let alignment = device.capabilities().limits.uniform_offset_alignment;
    if spec.extent.contains(&0) || alignment == 0 {
        return Err(Error::InvalidRange.into());
    }
    validate_forward_topology(topology, spec.draws.len(), spec.shape.batches.len())?;
    validate_depth_spec(spec)?;
    validate_depth_topology(topology, spec.depth.is_some())?;
    for draw in &spec.draws {
        validate_graph_draw(draw, spec.streams.len())?;
        if (!receiver
            && (draw.capture_textures.iter().any(|capture| *capture)
                || draw
                    .textures
                    .iter()
                    .any(|texture| *texture >= spec.textures.len())))
            || (receiver
                && (draw.textures.is_empty()
                    || draw.textures.len() != draw.capture_textures.len()
                    || draw
                        .textures
                        .iter()
                        .any(|texture| *texture >= spec.textures.len())))
            || draw
                .uniform_ranges
                .iter()
                .any(|range| !range.start.is_multiple_of(alignment))
        {
            return Err(Error::InvalidRange.into());
        }
    }
    Ok(())
}

pub(super) fn import_frame(
    graph: &mut GraphBuilder,
    spec: &ForwardGraphSpec,
    aliases: Option<&[Option<crate::graph::definition::TextureHandle>]>,
    shared: &mut SharedResidentImports,
) -> std::result::Result<(CaptureFrameSlots, FrameHandles), ForwardPlanBuildError> {
    import_frame_inner(graph, spec, aliases, shared, None)
}

/// Imports the material namespace while retaining attachments already declared
/// by the enclosing Graph. The caller validates their actual owner identities.
pub(super) fn import_frame_inner(
    graph: &mut GraphBuilder,
    spec: &ForwardGraphSpec,
    aliases: Option<&[Option<crate::graph::definition::TextureHandle>]>,
    shared: &mut SharedResidentImports,
    attachments: Option<(
        crate::graph::definition::TextureHandle,
        crate::graph::definition::TextureHandle,
    )>,
) -> std::result::Result<(CaptureFrameSlots, FrameHandles), ForwardPlanBuildError> {
    let resident_upload = graph
        .import_buffer_desc(
            spec.resident_upload.desc.clone(),
            spec.resident_upload.incoming,
        )
        .map_err(|_| Error::InvalidState)?;
    let frame_upload = graph
        .import_buffer_desc(spec.frame_upload.desc.clone(), spec.frame_upload.incoming)
        .map_err(|_| Error::InvalidState)?;
    let uniform = graph
        .import_buffer_desc(spec.uniform.desc.clone(), spec.uniform.incoming)
        .map_err(|_| Error::InvalidState)?;
    let environment_upload = spec
        .environment_upload
        .as_ref()
        .map(|owner| {
            graph
                .import_buffer_desc(owner.desc.clone(), owner.incoming)
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
    let specular_upload = spec
        .specular_upload
        .as_ref()
        .map(|owner| {
            graph
                .import_buffer_desc(owner.desc.clone(), owner.incoming)
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
    let specular_view_uniform = spec
        .specular_view_uniform
        .as_ref()
        .map(|owner| {
            graph
                .import_buffer_desc(owner.desc.clone(), owner.incoming)
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
    let specular_radiance = spec
        .specular_radiance
        .as_ref()
        .map(|owner| {
            graph
                .import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
    let specular_brdf_lut = spec
        .specular_brdf_lut
        .as_ref()
        .map(|owner| {
            graph
                .import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
    if spec.streams.len() != spec.stream_instances.len()
        || spec.textures.len() != spec.texture_instances.len()
    {
        return Err(Error::InvalidRange.into());
    }
    let streams = spec
        .streams
        .iter()
        .cloned()
        .zip(&spec.stream_instances)
        .enumerate()
        .map(|(index, (owner, instance))| {
            let fresh = spec
                .vertex_uploads
                .iter()
                .any(|upload| upload.stream == index);
            if fresh {
                return graph
                    .import_buffer_desc(owner.desc.clone(), owner.incoming)
                    .map_err(|_| Error::InvalidState);
            }
            if let Some((handle, saved)) = shared.streams.get(instance) {
                if !same_buffer_owner_desc(&saved.desc, &owner.desc)
                    || saved.incoming != owner.incoming
                {
                    return Err(Error::InvalidState);
                }
                return Ok(*handle);
            }
            let handle = graph
                .import_buffer_desc(owner.desc.clone(), owner.incoming)
                .map_err(|_| Error::InvalidState)?;
            shared.streams.insert(*instance, (handle, owner));
            Ok(handle)
        })
        .collect::<Result<Vec<_>>>()?;
    if aliases.is_some_and(|aliases| aliases.len() != spec.textures.len()) {
        return Err(Error::InvalidRange.into());
    }
    let mut uploaded = vec![false; spec.textures.len()];
    for upload in &spec.texture_uploads {
        *uploaded
            .get_mut(upload.texture)
            .ok_or(Error::InvalidRange)? = true;
    }
    let mut cold_textures = vec![false; spec.textures.len()];
    let textures = spec
        .textures
        .iter()
        .enumerate()
        .map(|(index, owner)| {
            aliases
                .and_then(|aliases| aliases[index])
                .map(Ok)
                .unwrap_or_else(|| {
                    let instance = spec.texture_instances[index];
                    if uploaded[index] {
                        if owner.incoming.stages != fluxel_rhi::api::StageMask::EMPTY
                            || owner.incoming.accesses != fluxel_rhi::api::AccessMask::EMPTY
                            || owner.layout != ImageLayout::Undefined
                        {
                            return Err(Error::InvalidState);
                        }
                        cold_textures[index] = true;
                        return graph
                            .create_texture(owner.desc.clone())
                            .map_err(|_| Error::InvalidState);
                    }
                    if let Some((handle, saved)) = shared.textures.get(&instance) {
                        if !same_texture_owner_desc(&saved.desc, &owner.desc)
                            || saved.incoming != owner.incoming
                            || saved.layout != owner.layout
                        {
                            return Err(Error::InvalidState);
                        }
                        return Ok(*handle);
                    }
                    let handle = graph
                        .import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
                        .map_err(|_| Error::InvalidState)?;
                    shared.textures.insert(instance, (handle, owner.clone()));
                    Ok(handle)
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let environment = spec
        .environment
        .as_ref()
        .map(|owner| {
            graph
                .import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
    let (target, depth) = match attachments {
        Some((color, depth)) => (color, Some(depth)),
        None => {
            let target = graph
                .import_texture_with_state(
                    spec.target.desc.clone(),
                    spec.target.incoming,
                    spec.target.layout,
                )
                .map_err(|_| Error::InvalidState)?;
            let depth = spec
                .depth
                .as_ref()
                .map(|owner| {
                    graph
                        .import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
                        .map_err(|_| Error::InvalidState)
                })
                .transpose()?;
            (target, depth)
        }
    };
    let readback = graph
        .import_buffer_desc(spec.readback.desc.clone(), spec.readback.incoming)
        .map_err(|_| Error::InvalidState)?;
    let slots = CaptureFrameSlots {
        resident_upload: BufferSlot::from(resident_upload),
        frame_upload: BufferSlot::from(frame_upload),
        uniform: BufferSlot::from(uniform),
        environment_upload: environment_upload.map(BufferSlot::from),
        specular_upload: specular_upload.map(BufferSlot::from),
        specular_view_uniform: specular_view_uniform.map(BufferSlot::from),
        streams: streams.iter().copied().map(BufferSlot::from).collect(),
        textures: textures.iter().copied().map(TextureSlot::from).collect(),
        environment: environment.map(TextureSlot::from),
        specular_radiance: specular_radiance.map(TextureSlot::from),
        specular_brdf_lut: specular_brdf_lut.map(TextureSlot::from),
        depth: depth.map(TextureSlot::from),
        target: TextureSlot::from(target),
        readback: BufferSlot::from(readback),
    };
    Ok((
        slots,
        FrameHandles {
            resident_upload,
            frame_upload,
            uniform,
            environment_upload,
            specular_upload,
            specular_view_uniform,
            specular_radiance,
            specular_brdf_lut,
            streams,
            textures,
            cold_textures,
            environment,
            depth,
            target,
            readback,
        },
    ))
}

pub(super) use super::capture_upload::{declare_mip_generations, declare_upload};

#[path = "capture_raster_declaration.rs"]
mod raster_declaration;
pub(super) use raster_declaration::declare_raster;
