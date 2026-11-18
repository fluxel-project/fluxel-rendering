//! Upload declarations for two independent capture-frame namespaces.
use super::super::mip_generation::{MipGenerationPass, declare_chain};
use super::capture_declaration::{FrameHandles, SharedResidentImports};
use super::*;
use crate::frame_data::specular_view::SPECULAR_VIEW_BYTES;

pub(super) fn declare_mip_generations(
    graph: &mut GraphBuilder,
    spec: &ForwardGraphSpec,
    mut handles: FrameHandles,
    shared: &mut SharedResidentImports,
) -> std::result::Result<(FrameHandles, Vec<MipGenerationPass>), ForwardPlanBuildError> {
    if handles.textures.len() != handles.cold_textures.len()
        || handles.textures.len() != spec.textures.len()
        || spec.textures.len() != spec.texture_instances.len()
    {
        return Err(Error::InvalidRange.into());
    }
    let mut generations = Vec::new();
    for (index, cold) in handles.cold_textures.iter().copied().enumerate() {
        if !cold
            || !spec.textures[index]
                .desc
                .usage
                .contains(TextureUsage::COPY_SRC)
        {
            continue;
        }
        let (output, mut chain) = declare_chain(
            graph,
            handles.textures[index],
            index,
            spec.textures[index].desc.mip_levels,
        )
        .map_err(|_| Error::InvalidState)?;
        handles.textures[index] = output;
        generations.append(&mut chain);
        shared
            .textures
            .get_mut(&spec.texture_instances[index])
            .ok_or(Error::InvalidState)?
            .0 = output;
    }
    Ok((handles, generations))
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
    let (
        pass,
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
                if !handles
                    .cold_textures
                    .get(upload.texture)
                    .copied()
                    .ok_or(crate::graph::definition::GraphError::InvalidRange)?
                {
                    continue;
                }
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
                    one_color_mip(upload.mip_level),
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
            let (radiance, brdf_lut, view_uniform) = match (
                handles.specular_radiance,
                handles.specular_brdf_lut,
                handles.specular_view_uniform,
                handles.specular_upload,
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
                streams,
                uniform,
                textures,
                environment,
                radiance,
                brdf_lut,
                view_uniform,
            ))
        })
        .map_err(|_| Error::InvalidState)?;
    Ok((
        pass,
        FrameHandles {
            streams,
            uniform,
            textures,
            environment,
            specular_radiance,
            specular_brdf_lut,
            specular_view_uniform,
            ..handles
        },
    ))
}
