//! Complete, callback-bearing graph declarations for cached Forward plans.

use super::*;

/// A fully declared direct-surface graph before device-plan compilation.
///
/// It owns no backend resource. Its graph is retained by the compiled plan so
/// later frame declarations can compare the actual dependency/access contract
/// without compiling another execution plan.
pub(crate) struct SurfaceForwardDeclaration {
    pub(super) graph: GraphBuilder,
    pub(super) lane: LaneId,
    pub(super) upload_pass: GraphPass,
    pub(super) raster_pass: GraphPass,
    pub(super) resident_upload_slot: BufferSlot,
    pub(super) frame_upload_slot: BufferSlot,
    pub(super) uniform_slot: BufferSlot,
    pub(super) stream_slots: Vec<BufferSlot>,
    pub(super) texture_slots: Vec<TextureSlot>,
    pub(super) extent: [u32; 2],
    pub(super) draw_count: usize,
    pub(super) stream_count: usize,
    pub(super) texture_count: usize,
    pub(super) color_format: TextureFormat,
    pub(super) shape: SurfaceForwardShape,
    pub(super) topology: FrameTopology,
    pub(super) uniform_alignment: u64,
}

pub(super) fn declare_surface_with<B, F>(
    device: &Device<B>,
    frame: &SurfaceForwardFrame<B>,
    expected: FrameTopology,
    contribute: F,
) -> std::result::Result<SurfaceForwardDeclaration, ForwardPlanBuildError>
where
    B: ResourceBackend,
    F: FnOnce(&mut ForwardPhaseContext<'_, '_, '_, B>) -> std::result::Result<(), PipelineError>,
{
    if frame.resident.device != device.id() {
        return Err(Error::WrongDevice.into());
    }
    let uniform_alignment = device.capabilities().limits.uniform_offset_alignment;
    if frame.extent.contains(&0) || uniform_alignment == 0 {
        return Err(Error::InvalidRange.into());
    }
    validate_surface_topology(expected, frame.draws.len(), frame.draw_batches.len())?;
    validate_surface_uploads(frame)?;
    validate_surface_batches(frame)?;
    for draw in &frame.draws {
        validate_draw(draw, frame.resident.streams.len())?;
        validate_offsets(draw, uniform_alignment)?;
        surface_draw_accesses(frame, draw)?;
        if draw.texture >= frame.resident.textures.len() {
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
            frame.resident.upload_desc.clone(),
            frame.resident.upload_incoming,
        )
        .map_err(|_| Error::InvalidState)?;
    let frame_upload = graph
        .import_buffer_desc(frame.frame.upload_desc.clone(), frame.frame.upload_incoming)
        .map_err(|_| Error::InvalidState)?;
    let uniform = graph
        .import_buffer_desc(
            frame.frame.uniform_desc.clone(),
            frame.frame.uniform_incoming,
        )
        .map_err(|_| Error::InvalidState)?;
    let streams = frame
        .resident
        .streams
        .iter()
        .map(|stream| {
            graph
                .import_buffer_desc(stream.desc.clone(), stream.incoming)
                .map_err(|_| Error::InvalidState)
        })
        .collect::<Result<Vec<_>>>()?;
    let textures = frame
        .resident
        .textures
        .iter()
        .map(|texture| {
            graph
                .import_texture_with_state(texture.desc.clone(), texture.incoming, texture.layout)
                .map_err(|_| Error::InvalidState)
        })
        .collect::<Result<Vec<_>>>()?;
    let texture_ends = frame
        .resident_upload
        .textures
        .iter()
        .map(texture_copy_end)
        .collect::<Result<Vec<_>>>()?;
    let (upload_pass, (streams, uniform, textures)) = graph
        .add_pass("forward-surface-upload", |pass| {
            let mut output_streams = streams.clone();
            for upload in &frame.resident_upload.vertices {
                let stream = *streams
                    .get(upload.stream)
                    .ok_or(super::super::definition::GraphError::InvalidRange)?;
                let owner = frame
                    .resident
                    .streams
                    .get(upload.stream)
                    .ok_or(super::super::definition::GraphError::InvalidRange)?;
                pass.read_with_use(
                    resident_upload,
                    upload.source.clone(),
                    copy_read_uses().uses,
                    BufferUsage::COPY_SRC,
                )?;
                output_streams[upload.stream] = pass.write_with_use(
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
                frame.uniform_upload.source.clone(),
                copy_read_uses().uses,
                BufferUsage::COPY_SRC,
            )?;
            let uniform = pass.write_with_use(
                uniform,
                frame.uniform_upload.destination.clone(),
                copy_write_uses().uses,
                BufferUsage::COPY_DST,
            )?;
            let mut output_textures = textures.clone();
            for (upload, end) in frame.resident_upload.textures.iter().zip(&texture_ends) {
                let texture = *textures
                    .get(upload.texture)
                    .ok_or(super::super::definition::GraphError::InvalidRange)?;
                pass.read_with_use(
                    resident_upload,
                    upload.source_offset..*end,
                    copy_read_uses().uses,
                    BufferUsage::COPY_SRC,
                )?;
                output_textures[upload.texture] = pass.write_texture_with_use(
                    texture,
                    whole(),
                    texture_copy_write_uses().uses,
                    ImageLayout::CopyDst,
                    TextureUsage::COPY_DST,
                )?;
            }
            Ok((output_streams, uniform, output_textures))
        })
        .map_err(|_| Error::InvalidState)?;
    let raster_pass = {
        let draws = frame
            .draws
            .iter()
            .map(|draw| {
                Ok(ForwardGraphDraw {
                    vertex_streams: draw.vertex_streams.clone(),
                    vertex_layouts: draw.vertex_layouts.clone(),
                    vertex_ranges: draw.vertex_ranges.clone(),
                    vertices: draw.vertices.clone(),
                    instances: draw.instances.clone(),
                    uniform_ranges: draw
                        .bind_groups
                        .iter()
                        .flat_map(|group| group.uniform_ranges.iter().cloned())
                        .collect(),
                    texture: draw.texture,
                    accesses: surface_draw_accesses(frame, draw)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut driver = ForwardPhaseDriver {
            graph: Some(&mut graph),
            marker: PhantomData,
            expected,
            upload: Some(upload_pass),
            streams: streams.clone(),
            uniform: Some(uniform),
            textures: textures.clone(),
            target: None,
            surface_target: true,
            draws,
            declared: false,
            output: None,
        };
        let mut context = ForwardPhaseContext::new(&mut driver);
        contribute(&mut context).map_err(ForwardPlanBuildError::Pipeline)?;
        driver
            .finish_surface()
            .map_err(|_| ForwardPlanBuildError::Pipeline(PipelineError::InvalidPhaseContribution))?
    };
    Ok(SurfaceForwardDeclaration {
        graph,
        lane,
        upload_pass,
        raster_pass,
        resident_upload_slot: BufferSlot::from(resident_upload),
        frame_upload_slot: BufferSlot::from(frame_upload),
        uniform_slot: BufferSlot::from(uniform),
        stream_slots: streams.into_iter().map(BufferSlot::from).collect(),
        texture_slots: textures.into_iter().map(TextureSlot::from).collect(),
        extent: frame.extent,
        draw_count: frame.draws.len(),
        stream_count: frame.resident.streams.len(),
        texture_count: frame.resident.textures.len(),
        color_format: frame.color_format,
        shape: surface_forward_shape(frame),
        topology: expected,
        uniform_alignment,
    })
}
