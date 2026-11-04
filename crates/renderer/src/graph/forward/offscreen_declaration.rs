//! Complete private offscreen Forward graph declarations.

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
    BufferSlot, ForwardGraphSpec, ForwardPhaseDriver, ForwardPlanBuildError, ForwardShape,
    GraphBuilder, GraphPass, TextureSlot, copy_read_uses, copy_write_uses, graph_texture_copy_end,
    texture_copy_read_uses, texture_copy_write_uses, validate_forward_topology,
    validate_graph_draw, whole,
};

/// A complete, uncompiled offscreen Forward declaration.
pub(crate) struct OffscreenForwardDeclaration {
    pub(super) graph: GraphBuilder,
    pub(super) lane: LaneId,
    pub(super) upload_pass: GraphPass,
    pub(super) raster_pass: GraphPass,
    pub(super) readback_pass: GraphPass,
    pub(super) resident_upload_slot: BufferSlot,
    pub(super) frame_upload_slot: BufferSlot,
    pub(super) uniform_slot: BufferSlot,
    pub(super) stream_slots: Vec<BufferSlot>,
    pub(super) texture_slots: Vec<TextureSlot>,
    pub(super) target_slot: TextureSlot,
    pub(super) readback_slot: BufferSlot,
    pub(super) draw_count: usize,
    pub(super) stream_count: usize,
    pub(super) texture_count: usize,
    pub(super) extent: [u32; 2],
    pub(super) shape: ForwardShape,
    pub(super) uniform_alignment: u64,
    pub(super) topology: FrameTopology,
}

pub(super) fn declare_offscreen_with<B, F>(
    device: &Device<B>,
    frame: &ForwardFrame<B>,
    expected: FrameTopology,
    contribute: F,
) -> std::result::Result<OffscreenForwardDeclaration, ForwardPlanBuildError>
where
    B: ResourceBackend,
    F: FnOnce(&mut ForwardPhaseContext<'_, '_, '_, B>) -> std::result::Result<(), PipelineError>,
{
    let spec = ForwardGraphSpec::from_frame(frame)?;
    declare_offscreen_spec_with(device, &spec, expected, contribute)
}

pub(super) fn declare_offscreen_spec_with<B, F>(
    device: &Device<B>,
    spec: &ForwardGraphSpec,
    expected: FrameTopology,
    contribute: F,
) -> std::result::Result<OffscreenForwardDeclaration, ForwardPlanBuildError>
where
    B: ResourceBackend,
    F: FnOnce(&mut ForwardPhaseContext<'_, '_, '_, B>) -> std::result::Result<(), PipelineError>,
{
    if spec.device != device.id() {
        return Err(Error::WrongDevice.into());
    }
    let uniform_alignment = device.capabilities().limits.uniform_offset_alignment;
    if spec.extent.contains(&0) || uniform_alignment == 0 {
        return Err(Error::InvalidRange.into());
    }
    validate_forward_topology(expected, spec.draws.len(), spec.shape.batches.len())?;
    for draw in &spec.draws {
        validate_graph_draw(draw, spec.streams.len())?;
        if draw.texture >= spec.textures.len()
            || draw
                .uniform_ranges
                .iter()
                .any(|range| !range.start.is_multiple_of(uniform_alignment))
        {
            return Err(Error::InvalidRange.into());
        }
    }
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .map(|lane| lane.id)
        .ok_or(Error::WrongLane)?;
    let mut graph = GraphBuilder::default();
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
    let streams = spec
        .streams
        .iter()
        .map(|stream| {
            graph
                .import_buffer_desc(stream.desc.clone(), stream.incoming)
                .map_err(|_| Error::InvalidState)
        })
        .collect::<Result<Vec<_>>>()?;
    let textures = spec
        .textures
        .iter()
        .map(|texture| {
            graph
                .import_texture_with_state(texture.desc.clone(), texture.incoming, texture.layout)
                .map_err(|_| Error::InvalidState)
        })
        .collect::<Result<Vec<_>>>()?;
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
    let texture_ends = spec
        .texture_uploads
        .iter()
        .map(graph_texture_copy_end)
        .collect::<Result<Vec<_>>>()?;
    let (upload_pass, (streams, uniform, textures)) = graph
        .add_pass("forward-upload", |pass| {
            let mut outputs = streams.clone();
            for upload in &spec.vertex_uploads {
                let stream = *streams
                    .get(upload.stream)
                    .ok_or(super::super::definition::GraphError::InvalidRange)?;
                let owner = spec
                    .streams
                    .get(upload.stream)
                    .ok_or(super::super::definition::GraphError::InvalidRange)?;
                pass.read_with_use(
                    resident_upload,
                    upload.source.clone(),
                    copy_read_uses().uses,
                    BufferUsage::COPY_SRC,
                )?;
                outputs[upload.stream] = pass.write_with_use(
                    stream,
                    upload.destination.clone(),
                    copy_write_uses().uses,
                    BufferUsage::COPY_DST,
                )?;
                if upload.destination.end > owner.desc.size {
                    return Err(super::super::definition::GraphError::InvalidRange);
                }
            }
            pass.read_with_use(
                frame_upload,
                spec.uniform_upload.source.clone(),
                copy_read_uses().uses,
                BufferUsage::COPY_SRC,
            )?;
            let uniform = pass.write_with_use(
                uniform,
                spec.uniform_upload.destination.clone(),
                copy_write_uses().uses,
                BufferUsage::COPY_DST,
            )?;
            let mut outputs_textures = textures.clone();
            for (upload, end) in spec.texture_uploads.iter().zip(&texture_ends) {
                let texture = *textures
                    .get(upload.texture)
                    .ok_or(super::super::definition::GraphError::InvalidRange)?;
                pass.read_with_use(
                    resident_upload,
                    upload.source_offset..*end,
                    copy_read_uses().uses,
                    BufferUsage::COPY_SRC,
                )?;
                outputs_textures[upload.texture] = pass.write_texture_with_use(
                    texture,
                    whole(),
                    texture_copy_write_uses().uses,
                    ImageLayout::CopyDst,
                    TextureUsage::COPY_DST,
                )?;
            }
            Ok((outputs, uniform, outputs_textures))
        })
        .map_err(|_| Error::InvalidState)?;
    let (raster_pass, target) = {
        let mut driver = ForwardPhaseDriver {
            graph: Some(&mut graph),
            marker: std::marker::PhantomData,
            expected,
            upload: Some(upload_pass),
            streams: streams.clone(),
            uniform: Some(uniform),
            textures: textures.clone(),
            target: Some(target),
            surface_target: false,
            draws: spec.draws.clone(),
            declared: false,
            output: None,
        };
        let mut context = ForwardPhaseContext::new(&mut driver);
        contribute(&mut context).map_err(ForwardPlanBuildError::Pipeline)?;
        driver
            .finish()
            .map_err(|_| ForwardPlanBuildError::Pipeline(PipelineError::InvalidPhaseContribution))?
    };
    let readback_len = u64::from(spec.readback_row_pitch)
        .checked_mul(u64::from(spec.extent[1]))
        .ok_or(Error::InvalidRange)?;
    let (readback_pass, ()) = graph
        .add_pass("forward-readback", |pass| {
            pass.read_texture_with_use(
                target,
                whole(),
                texture_copy_read_uses().uses,
                ImageLayout::CopySrc,
                TextureUsage::COPY_SRC,
            )?;
            pass.write_with_use(
                readback,
                0..readback_len,
                copy_write_uses().uses,
                BufferUsage::COPY_DST,
            )?;
            pass.side_effect();
            Ok(())
        })
        .map_err(|_| Error::InvalidState)?;
    Ok(OffscreenForwardDeclaration {
        graph,
        lane,
        upload_pass,
        raster_pass,
        readback_pass,
        resident_upload_slot: BufferSlot::from(resident_upload),
        frame_upload_slot: BufferSlot::from(frame_upload),
        uniform_slot: BufferSlot::from(uniform),
        stream_slots: streams.into_iter().map(BufferSlot::from).collect(),
        texture_slots: textures.into_iter().map(TextureSlot::from).collect(),
        target_slot: TextureSlot::from(target),
        readback_slot: BufferSlot::from(readback),
        draw_count: spec.draws.len(),
        stream_count: spec.streams.len(),
        texture_count: spec.textures.len(),
        extent: spec.extent,
        shape: spec.shape.clone(),
        uniform_alignment,
        topology: expected,
    })
}
