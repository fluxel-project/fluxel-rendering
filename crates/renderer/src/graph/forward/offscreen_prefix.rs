//! Shared offscreen imports, uploads and mip generation before any raster output.

use super::*;
use crate::frame_data::specular_view::SPECULAR_VIEW_BYTES;

pub(super) struct ForwardDeclarationPrefix {
    pub(super) graph: GraphBuilder,
    pub(super) lane: LaneId,
    pub(super) upload_pass: GraphPass,
    pub(super) mip_passes: Vec<super::super::mip_generation::MipGenerationPass>,
    pub(super) uniform_alignment: u64,
    pub(super) resident_upload: super::super::definition::BufferHandle,
    pub(super) frame_upload: super::super::definition::BufferHandle,
    pub(super) uniform: super::super::definition::BufferHandle,
    pub(super) readback: super::super::definition::BufferHandle,
    pub(super) environment_upload: Option<super::super::definition::BufferHandle>,
    pub(super) specular_upload: Option<super::super::definition::BufferHandle>,
    pub(super) specular_view_uniform: Option<super::super::definition::BufferHandle>,
    pub(super) streams: Vec<super::super::definition::BufferHandle>,
    pub(super) textures: Vec<super::super::definition::TextureHandle>,
    pub(super) environment: Option<super::super::definition::TextureHandle>,
    pub(super) specular_radiance: Option<super::super::definition::TextureHandle>,
    pub(super) specular_brdf_lut: Option<super::super::definition::TextureHandle>,
    pub(super) depth: Option<super::super::definition::TextureHandle>,
    pub(super) target: super::super::definition::TextureHandle,
    pub(super) multisample_color: Option<super::super::definition::TextureHandle>,
}

pub(super) fn declare_prefix<B: ResourceBackend>(
    device: &Device<B>,
    spec: &ForwardGraphSpec,
    expected: FrameTopology,
) -> std::result::Result<ForwardDeclarationPrefix, ForwardPlanBuildError> {
    if spec.device != device.id() {
        return Err(Error::WrongDevice.into());
    }
    let uniform_alignment = device.capabilities().limits.uniform_offset_alignment;
    if spec.extent.contains(&0) || uniform_alignment == 0 {
        return Err(Error::InvalidRange.into());
    }
    validate_forward_topology(expected, spec.draws.len(), spec.shape.batches.len())?;
    validate_depth_spec(spec)?;
    validate_multisample_color_spec(spec)?;
    validate_depth_topology(expected, spec.depth.is_some())?;
    for draw in &spec.draws {
        validate_graph_draw(draw, spec.streams.len())?;
        if draw.capture_textures.iter().any(|capture| *capture)
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
        .enumerate()
        .map(|(index, texture)| {
            if spec
                .texture_uploads
                .iter()
                .any(|upload| upload.texture == index)
                && texture.desc.mip_levels > 1
            {
                graph.create_texture(texture.desc.clone())
            } else {
                graph.import_texture_with_state(
                    texture.desc.clone(),
                    texture.incoming,
                    texture.layout,
                )
            }
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
    let multisample_color = spec
        .multisample_color
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
            mut textures,
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
                let texture = *outputs_textures
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
                    material_mip_layer(
                        &spec.textures[upload.texture].desc,
                        upload.mip_level,
                        upload.array_layer,
                    ),
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
                        0..SPECULAR_VIEW_BYTES,
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
    let mut mip_passes = Vec::new();
    for (index, owner) in spec.textures.iter().enumerate() {
        if spec
            .texture_uploads
            .iter()
            .any(|upload| upload.texture == index)
            && owner.desc.usage.contains(TextureUsage::COPY_SRC)
        {
            let (texture, passes) = super::super::mip_generation::declare_chain(
                &mut graph,
                *textures.get(index).ok_or(Error::InvalidRange)?,
                index,
                owner.desc.mip_levels,
            )
            .map_err(|_| Error::InvalidState)?;
            textures[index] = texture;
            mip_passes.extend(passes);
        }
    }
    Ok(ForwardDeclarationPrefix {
        graph,
        lane,
        upload_pass,
        mip_passes,
        uniform_alignment,
        resident_upload,
        frame_upload,
        uniform,
        readback,
        environment_upload,
        specular_upload,
        specular_view_uniform,
        streams,
        textures,
        environment,
        specular_radiance,
        specular_brdf_lut,
        depth,
        target,
        multisample_color,
    })
}
