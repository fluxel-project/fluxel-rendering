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
    GraphPass, TextureSlot, copy_read_uses, copy_write_uses, graph_texture_copy_end,
    texture_copy_read_uses, texture_copy_write_uses, validate_forward_topology,
    validate_graph_draw, whole,
};

/// All Graph slots imported for one Forward frame namespace.
pub(super) struct CaptureFrameSlots {
    pub(super) resident_upload: BufferSlot,
    pub(super) frame_upload: BufferSlot,
    pub(super) uniform: BufferSlot,
    pub(super) environment_upload: Option<BufferSlot>,
    pub(super) streams: Vec<BufferSlot>,
    pub(super) textures: Vec<TextureSlot>,
    pub(super) environment: Option<TextureSlot>,
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
    pub(super) streams: Vec<crate::graph::definition::BufferHandle>,
    pub(super) textures: Vec<crate::graph::definition::TextureHandle>,
    pub(super) environment: Option<crate::graph::definition::TextureHandle>,
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
    let readback = graph
        .import_buffer_desc(spec.readback.desc.clone(), spec.readback.incoming)
        .map_err(|_| Error::InvalidState)?;
    let slots = CaptureFrameSlots {
        resident_upload: BufferSlot::from(resident_upload),
        frame_upload: BufferSlot::from(frame_upload),
        uniform: BufferSlot::from(uniform),
        environment_upload: environment_upload.map(BufferSlot::from),
        streams: streams.iter().copied().map(BufferSlot::from).collect(),
        textures: textures.iter().copied().map(TextureSlot::from).collect(),
        environment: environment.map(TextureSlot::from),
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
            streams,
            textures,
            environment,
            target,
            readback,
        },
    ))
}

pub(super) fn declare_upload(
    graph: &mut GraphBuilder,
    spec: &ForwardGraphSpec,
    handles: FrameHandles,
) -> std::result::Result<(GraphPass, FrameHandles), ForwardPlanBuildError> {
    let ends = spec
        .texture_uploads
        .iter()
        .map(graph_texture_copy_end)
        .collect::<Result<Vec<_>>>()?;
    let (pass, (streams, uniform, textures, environment)) = graph
        .add_pass("forward-capture-upload", |pass| {
            let mut streams = handles.streams.clone();
            for upload in &spec.vertex_uploads {
                let stream = *streams
                    .get(upload.stream)
                    .ok_or(crate::graph::definition::GraphError::InvalidRange)?;
                pass.read_with_use(
                    handles.resident_upload,
                    upload.source.clone(),
                    copy_read_uses().uses,
                    BufferUsage::COPY_SRC,
                )?;
                streams[upload.stream] = pass.write_with_use(
                    stream,
                    upload.destination.clone(),
                    copy_write_uses().uses,
                    BufferUsage::COPY_DST,
                )?;
            }
            pass.read_with_use(
                handles.frame_upload,
                spec.uniform_upload.source.clone(),
                copy_read_uses().uses,
                BufferUsage::COPY_SRC,
            )?;
            let uniform = pass.write_with_use(
                handles.uniform,
                spec.uniform_upload.destination.clone(),
                copy_write_uses().uses,
                BufferUsage::COPY_DST,
            )?;
            let mut textures = handles.textures.clone();
            for (upload, end) in spec.texture_uploads.iter().zip(ends) {
                let texture = *textures
                    .get(upload.texture)
                    .ok_or(crate::graph::definition::GraphError::InvalidRange)?;
                pass.read_with_use(
                    handles.resident_upload,
                    upload.source_offset..end,
                    copy_read_uses().uses,
                    BufferUsage::COPY_SRC,
                )?;
                textures[upload.texture] = pass.write_texture_with_use(
                    texture,
                    whole(),
                    texture_copy_write_uses().uses,
                    ImageLayout::CopyDst,
                    TextureUsage::COPY_DST,
                )?;
            }
            let environment = match (
                handles.environment,
                handles.environment_upload,
                &spec.environment_copy,
            ) {
                (Some(texture), Some(upload), Some(copy)) => {
                    pass.read_with_use(
                        upload,
                        0..u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image),
                        copy_read_uses().uses,
                        BufferUsage::COPY_SRC,
                    )?;
                    Some(pass.write_texture_with_use(
                        texture,
                        whole(),
                        texture_copy_write_uses().uses,
                        ImageLayout::CopyDst,
                        TextureUsage::COPY_DST,
                    )?)
                }
                (None, None, None) => None,
                _ => return Err(crate::graph::definition::GraphError::InvalidRange),
            };
            Ok((streams, uniform, textures, environment))
        })
        .map_err(|_| Error::InvalidState)?;
    Ok((
        pass,
        FrameHandles {
            streams,
            uniform,
            textures,
            environment,
            ..handles
        },
    ))
}

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
