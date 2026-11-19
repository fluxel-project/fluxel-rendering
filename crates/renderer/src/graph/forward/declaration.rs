//! Complete, callback-bearing graph declarations for cached Forward plans.

use super::*;
use crate::frame_data::specular_view::SPECULAR_VIEW_BYTES;

/// A fully declared direct-surface graph before device-plan compilation.
///
/// It owns no backend resource. Its graph is retained by the compiled plan so
/// later frame declarations can compare the actual dependency/access contract
/// without compiling another execution plan.
pub(crate) struct SurfaceForwardDeclaration {
    pub(super) graph: GraphBuilder,
    pub(super) lane: LaneId,
    pub(super) upload_pass: GraphPass,
    pub(super) mip_passes: Vec<super::super::mip_generation::MipGenerationPass>,
    pub(super) indirect: Vec<super::ForwardIndirectGraph>,
    pub(super) particles: Vec<super::ForwardParticleGraph>,
    pub(super) raster_pass: GraphPass,
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
    pub(super) surface_target_slot: TextureSlot,
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
    validate_resident_texture_devices(&frame.resident, device.id())?;
    let uniform_alignment = device.capabilities().limits.uniform_offset_alignment;
    if frame.extent.contains(&0) || uniform_alignment == 0 {
        return Err(Error::InvalidRange.into());
    }
    validate_surface_topology(expected, frame.draws.len(), frame.draw_batches.len())?;
    crate::residency::validate_indirect_draws(&frame.draws, &frame.indirect)?;
    crate::residency::validate_particle_draws(&frame.draws, &frame.particles, &frame.indirect)?;
    validate_surface_depth(frame)?;
    validate_depth_topology(expected, frame.frame.depth.is_some())?;
    validate_surface_uploads(frame)?;
    validate_surface_batches(frame)?;
    for draw in &frame.draws {
        validate_draw(draw, frame.resident.streams.len())?;
        validate_offsets(draw, uniform_alignment)?;
        surface_draw_accesses(frame, draw)?;
        if draw.capture_textures.iter().any(|capture| *capture)
            || draw
                .textures
                .iter()
                .any(|texture| *texture >= frame.resident.textures.len())
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
    let environment_upload = frame
        .environment
        .as_ref()
        .map(|owner| {
            graph
                .import_buffer_desc(owner.upload_desc.clone(), owner.upload_incoming)
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
    let specular_upload = frame
        .specular
        .as_ref()
        .map(|owner| {
            graph
                .import_buffer_desc(owner.upload_desc.clone(), owner.upload_incoming)
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
    let specular_view_uniform = frame
        .specular
        .as_ref()
        .map(|owner| {
            graph
                .import_buffer_desc(owner.view_uniform_desc.clone(), owner.view_uniform_incoming)
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
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
        .enumerate()
        .map(|(index, texture)| {
            if frame
                .resident_upload
                .textures
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
    let environment = frame
        .environment
        .as_ref()
        .map(|owner| {
            graph
                .import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
    let specular_radiance = frame
        .specular
        .as_ref()
        .map(|owner| {
            graph
                .import_texture_with_state(
                    owner.radiance.desc.clone(),
                    owner.radiance.incoming,
                    owner.radiance.layout,
                )
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
    let specular_brdf_lut = frame
        .specular
        .as_ref()
        .map(|owner| {
            graph
                .import_texture_with_state(
                    owner.brdf_lut.desc.clone(),
                    owner.brdf_lut.incoming,
                    owner.brdf_lut.layout,
                )
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
    let surface_target = graph
        .import_surface_texture(TextureDesc {
            label: Some("fluxel forward surface lease".into()),
            dimension: fluxel_rhi::api::TextureDimension::D2,
            extent: [frame.extent[0], frame.extent[1], 1],
            array_layers: 1,
            format: frame.color_format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: Vec::new(),
        })
        .map_err(|_| Error::InvalidState)?;
    let depth = frame
        .frame
        .depth
        .as_ref()
        .map(|owner| {
            graph
                .import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
                .map_err(|_| Error::InvalidState)
        })
        .transpose()?;
    let texture_ends = frame
        .resident_upload
        .textures
        .iter()
        .map(texture_copy_end)
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
                let texture = *output_textures
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
                    material_mip_layer(
                        &frame.resident.textures[upload.texture].desc,
                        upload.mip_level,
                        upload.array_layer,
                    ),
                    texture_copy_write_uses().uses,
                    ImageLayout::CopyDst,
                    TextureUsage::COPY_DST,
                )?;
            }
            let environment = match (environment, environment_upload, frame.environment.as_ref()) {
                (Some(texture), Some(upload), Some(owner)) => {
                    pass.read_with_use(
                        upload,
                        0..owner.upload_desc.size,
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
            let (radiance, lut, view_uniform) = match (
                specular_radiance,
                specular_brdf_lut,
                specular_view_uniform,
                specular_upload,
                frame.specular.as_ref(),
            ) {
                (Some(radiance), Some(lut), Some(view_uniform), Some(upload), Some(owner)) => {
                    for copy in &owner.radiance_copies {
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
                    let copy = &owner.brdf_lut_copy;
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
                        owner.view_uniform_copy.clone(),
                        copy_read_uses().uses,
                        BufferUsage::COPY_SRC,
                    )?;
                    let mut radiance = radiance;
                    for copy in &owner.radiance_copies {
                        radiance = pass.write_texture_with_use(
                            radiance,
                            one_color_mip(copy.mip),
                            texture_copy_write_uses().uses,
                            ImageLayout::CopyDst,
                            TextureUsage::COPY_DST,
                        )?;
                    }
                    let lut = pass.write_texture_with_use(
                        lut,
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
                    (Some(radiance), Some(lut), Some(view_uniform))
                }
                (None, None, None, None, None) => (None, None, None),
                _ => return Err(super::super::definition::GraphError::InvalidRange),
            };
            Ok((
                output_streams,
                uniform,
                output_textures,
                environment,
                radiance,
                lut,
                view_uniform,
            ))
        })
        .map_err(|_| Error::InvalidState)?;
    let mut mip_passes = Vec::new();
    for (index, owner) in frame.resident.textures.iter().enumerate() {
        if frame
            .resident_upload
            .textures
            .iter()
            .any(|upload| upload.texture == index)
            && owner.desc.format != TextureFormat::Depth32Float
            && owner.desc.usage.contains(TextureUsage::COPY_SRC)
        {
            let (texture, passes) = super::super::mip_generation::declare_chain(
                &mut graph,
                textures[index],
                index,
                owner.desc.mip_levels,
            )
            .map_err(|_| Error::InvalidState)?;
            textures[index] = texture;
            mip_passes.extend(passes);
        }
    }
    let raster_dependency = mip_passes.last().map_or(upload_pass, |pass| pass.pass);
    let indirect = super::declare_indirect(&mut graph, &frame.indirect)?;
    let particles = super::declare_particles(&mut graph, &frame.particles)?;
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
                    textures: draw.textures.clone(),
                    capture_textures: draw.capture_textures.clone(),
                    accesses: surface_draw_accesses(frame, draw)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut driver = ForwardPhaseDriver {
            graph: Some(&mut graph),
            marker: PhantomData,
            expected,
            upload: Some(raster_dependency),
            streams: streams.clone(),
            uniform: Some(uniform),
            textures: textures.clone(),
            environment,
            specular_radiance,
            specular_brdf_lut,
            specular_view_uniform,
            specular_mip_levels: frame
                .specular
                .as_ref()
                .map(|owner| owner.radiance.desc.mip_levels),
            target: surface_target,
            resolve_target: None,
            depth,
            draws,
            indirect: indirect.clone(),
            particles: particles.clone(),
            declared: false,
            output: None,
        };
        let mut context = ForwardPhaseContext::new(&mut driver);
        contribute(&mut context).map_err(ForwardPlanBuildError::Pipeline)?;
        let (pass, _, _) = driver.finish().map_err(|_| {
            ForwardPlanBuildError::Pipeline(PipelineError::InvalidPhaseContribution)
        })?;
        pass
    };
    Ok(SurfaceForwardDeclaration {
        graph,
        lane,
        upload_pass,
        mip_passes,
        indirect,
        particles,
        raster_pass,
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
        surface_target_slot: TextureSlot::from(surface_target),
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
