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
    one_color_mip, texture_copy_read_uses, texture_copy_write_uses, validate_depth_spec,
    validate_forward_topology, validate_graph_draw, validate_resident_texture_devices, whole,
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
    pub(super) environment_upload_slot: Option<BufferSlot>,
    pub(super) specular_upload_slot: Option<BufferSlot>,
    pub(super) specular_view_uniform_slot: Option<BufferSlot>,
    pub(super) stream_slots: Vec<BufferSlot>,
    pub(super) texture_slots: Vec<TextureSlot>,
    pub(super) environment_slot: Option<TextureSlot>,
    pub(super) specular_radiance_slot: Option<TextureSlot>,
    pub(super) specular_brdf_lut_slot: Option<TextureSlot>,
    pub(super) depth_slot: Option<TextureSlot>,
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
    validate_resident_texture_devices(&frame.resident, device.id())?;
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
    validate_depth_spec(spec)?;
    if spec.depth.is_some() && expected != FrameTopology::OpaqueForward {
        return Err(Error::InvalidState.into());
    }
    for draw in &spec.draws {
        validate_graph_draw(draw, spec.streams.len())?;
        if draw.textures.is_empty()
            || draw
                .textures
                .iter()
                .any(|texture| *texture >= spec.textures.len())
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
        .map(|fact| {
            graph
                .import_buffer_desc(fact.desc.clone(), fact.incoming)
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
    let specular_view_uniform = spec
        .specular_view_uniform
        .as_ref()
        .map(|fact| {
            graph
                .import_buffer_desc(fact.desc.clone(), fact.incoming)
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
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
    let environment = spec
        .environment
        .as_ref()
        .map(|owner| {
            graph
                .import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
    let specular_radiance = spec
        .specular_radiance
        .as_ref()
        .map(|fact| {
            graph
                .import_texture_with_state(fact.desc.clone(), fact.incoming, fact.layout)
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
    let specular_brdf_lut = spec
        .specular_brdf_lut
        .as_ref()
        .map(|fact| {
            graph
                .import_texture_with_state(fact.desc.clone(), fact.incoming, fact.layout)
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
    let texture_ends = spec
        .texture_uploads
        .iter()
        .map(graph_texture_copy_end)
        .collect::<Result<Vec<_>>>()?;
    let (
        upload_pass,
        (
            streams,
            uniform,
            textures,
            environment,
            specular_radiance,
            specular_brdf_lut,
            specular_view_uniform,
        ),
    ) = graph
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
            let environment = match (environment, environment_upload, &spec.environment_copy) {
                (Some(texture), Some(upload), Some(copy)) => {
                    pass.read_with_use(
                        upload,
                        0..copy.bytes_per_row as u64 * copy.rows_per_image as u64,
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
                _ => return Err(super::super::definition::GraphError::InvalidRange),
            };
            let (radiance, brdf_lut, view_uniform) = match (
                specular_radiance,
                specular_brdf_lut,
                specular_view_uniform,
                specular_upload,
                &spec.specular,
            ) {
                (
                    Some(radiance),
                    Some(brdf_lut),
                    Some(view_uniform),
                    Some(upload),
                    Some(contract),
                ) => {
                    for copy in &contract.radiance_copies {
                        let end = copy
                            .source_offset
                            .checked_add(
                                u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image),
                            )
                            .ok_or(super::super::definition::GraphError::InvalidRange)?;
                        pass.read_with_use(
                            upload,
                            copy.source_offset..end,
                            copy_read_uses().uses,
                            BufferUsage::COPY_SRC,
                        )?;
                    }
                    let copy = &contract.brdf_lut_copy;
                    let end = copy
                        .source_offset
                        .checked_add(u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image))
                        .ok_or(super::super::definition::GraphError::InvalidRange)?;
                    pass.read_with_use(
                        upload,
                        copy.source_offset..end,
                        copy_read_uses().uses,
                        BufferUsage::COPY_SRC,
                    )?;
                    pass.read_with_use(
                        upload,
                        contract.view_uniform_copy.clone(),
                        copy_read_uses().uses,
                        BufferUsage::COPY_SRC,
                    )?;
                    let mut radiance = radiance;
                    for copy in &contract.radiance_copies {
                        radiance = pass.write_texture_with_use(
                            radiance,
                            one_color_mip(copy.mip),
                            texture_copy_write_uses().uses,
                            ImageLayout::CopyDst,
                            TextureUsage::COPY_DST,
                        )?;
                    }
                    let brdf_lut = pass.write_texture_with_use(
                        brdf_lut,
                        whole(),
                        texture_copy_write_uses().uses,
                        ImageLayout::CopyDst,
                        TextureUsage::COPY_DST,
                    )?;
                    let view_uniform = pass.write_with_use(
                        view_uniform,
                        0..112,
                        copy_write_uses().uses,
                        BufferUsage::COPY_DST,
                    )?;
                    (Some(radiance), Some(brdf_lut), Some(view_uniform))
                }
                (None, None, None, None, None) => (None, None, None),
                _ => return Err(super::super::definition::GraphError::InvalidRange),
            };
            Ok((
                outputs,
                uniform,
                outputs_textures,
                environment,
                radiance,
                brdf_lut,
                view_uniform,
            ))
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
            environment,
            specular_radiance,
            specular_brdf_lut,
            specular_view_uniform,
            specular_mip_levels: spec
                .specular_radiance
                .as_ref()
                .map(|value| value.desc.mip_levels),
            target,
            depth,
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
        environment_upload_slot: environment_upload.map(BufferSlot::from),
        specular_upload_slot: specular_upload.map(BufferSlot::from),
        specular_view_uniform_slot: specular_view_uniform.map(BufferSlot::from),
        stream_slots: streams.into_iter().map(BufferSlot::from).collect(),
        texture_slots: textures.into_iter().map(TextureSlot::from).collect(),
        environment_slot: environment.map(TextureSlot::from),
        specular_radiance_slot: specular_radiance.map(TextureSlot::from),
        specular_brdf_lut_slot: specular_brdf_lut.map(TextureSlot::from),
        depth_slot: depth.map(TextureSlot::from),
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
