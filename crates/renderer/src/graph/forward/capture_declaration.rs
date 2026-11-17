//! One Graph declaration for a captured Forward texture and its receiver.

use fluxel_rhi::{
    api::{BufferUsage, Device, Error, ImageLayout, LaneClass, LaneId, Result, TextureUsage},
    backend::ResourceBackend,
};

use crate::{
    FrameTopology,
    pipeline::{ForwardPhaseContext, PipelineError},
    residency::ForwardFrame,
};

use super::{
    BufferSlot, ForwardGraphSpec, ForwardPhaseDriver, ForwardPlanBuildError, GraphBuilder,
    GraphPass, TextureSlot, copy_write_uses, texture_copy_read_uses, validate_depth_spec,
    validate_forward_topology, validate_graph_draw, whole,
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

/// The five-pass same-submission capture contract.
pub(crate) struct CaptureForwardDeclaration {
    pub(super) graph: GraphBuilder,
    pub(super) lane: LaneId,
    pub(super) capture_upload: GraphPass,
    pub(super) capture_raster: GraphPass,
    pub(super) receiver_upload: GraphPass,
    pub(super) receiver_raster: GraphPass,
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
        || receiver_spec.textures.len() != 1
        || !receiver_spec.texture_uploads.is_empty()
        || receiver_spec.draws.iter().any(|draw| draw.texture != 0)
    {
        return Err(Error::InvalidState.into());
    }
    let receiver_texture = receiver
        .resident
        .textures
        .first()
        .ok_or(Error::InvalidState)?;
    // The sampled receiver binding must be the exact producer allocation;
    // matching declarations alone cannot establish resource identity.
    if receiver_texture.instance != capture.frame.target_instance
        || receiver_texture.texture.device() != capture.frame.target.device()
        || receiver_texture.view.device() != capture.frame.target_view.device()
        || !same_texture_owner_desc(&receiver_texture.desc, &capture.frame.target_desc)
        || receiver_texture.incoming != capture.frame.target_incoming
        || receiver_texture.layout != capture.frame.target_layout
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
    // The receiver's prepared texture is an owner alias for capture.target,
    // not a second physical Graph import. Its actual owner facts are checked
    // by the submission path before this declaration is reached.
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
    let receiver_readback = declare_readback(
        &mut graph,
        &receiver_spec,
        receiver_target,
        receiver_handles.readback,
    )?;
    Ok(CaptureForwardDeclaration {
        graph,
        lane,
        capture_upload,
        capture_raster,
        receiver_upload,
        receiver_raster,
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

pub(super) fn same_texture_owner_desc(
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
    if spec.depth.is_some() && topology != FrameTopology::OpaqueForward {
        return Err(Error::InvalidState.into());
    }
    for draw in &spec.draws {
        validate_graph_draw(draw, spec.streams.len())?;
        if (!receiver && draw.texture >= spec.textures.len())
            || (receiver && draw.texture != 0)
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
    import_textures: bool,
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
    let streams = spec
        .streams
        .iter()
        .map(|owner| {
            graph
                .import_buffer_desc(owner.desc.clone(), owner.incoming)
                .map_err(|_| Error::InvalidState)
        })
        .collect::<Result<Vec<_>>>()?;
    let textures = if import_textures {
        spec.textures
            .iter()
            .map(|owner| {
                graph
                    .import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
                    .map_err(|_| Error::InvalidState)
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let environment = spec
        .environment
        .as_ref()
        .map(|owner| {
            graph
                .import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
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
            environment,
            depth,
            target,
            readback,
        },
    ))
}

pub(super) use super::capture_upload::declare_upload;

pub(super) fn declare_raster<B, F>(
    graph: &mut GraphBuilder,
    spec: &ForwardGraphSpec,
    topology: FrameTopology,
    upload: GraphPass,
    handles: &FrameHandles,
    contribute: F,
) -> std::result::Result<(GraphPass, crate::graph::definition::TextureHandle), ForwardPlanBuildError>
where
    B: ResourceBackend,
    F: FnOnce(&mut ForwardPhaseContext<'_, '_, '_, B>) -> std::result::Result<(), PipelineError>,
{
    let mut driver = ForwardPhaseDriver {
        graph: Some(graph),
        marker: std::marker::PhantomData,
        expected: topology,
        upload: Some(upload),
        streams: handles.streams.clone(),
        uniform: Some(handles.uniform),
        textures: handles.textures.clone(),
        environment: handles.environment,
        specular_radiance: handles.specular_radiance,
        specular_brdf_lut: handles.specular_brdf_lut,
        specular_view_uniform: handles.specular_view_uniform,
        specular_mip_levels: spec
            .specular_radiance
            .as_ref()
            .map(|owner| owner.desc.mip_levels),
        depth: handles.depth,
        target: handles.target,
        draws: spec.draws.clone(),
        declared: false,
        output: None,
    };
    let mut context = ForwardPhaseContext::new(&mut driver);
    contribute(&mut context).map_err(ForwardPlanBuildError::Pipeline)?;
    driver
        .finish()
        .map_err(|_| ForwardPlanBuildError::Pipeline(PipelineError::InvalidPhaseContribution))
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
        .add_pass("forward-capture-readback", |pass| {
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
