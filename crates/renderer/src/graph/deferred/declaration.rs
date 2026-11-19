//! Logical resource declaration for the bounded Deferred probe.

use super::validation::validate_frame_inputs;
use super::*;
use crate::frame_data::specular_view::SPECULAR_VIEW_BYTES;

impl<B: ResourceBackend> DeferredPlan<B> {
    pub(crate) fn declare_with<F>(
        device: &Device<B>,
        frame: &DeferredFrame<B>,
        declare: F,
    ) -> Result<DeferredDeclaration>
    where
        F: FnOnce(
            &mut crate::DeferredPhaseContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        validate_frame_inputs(device, frame)?;
        Self::declare_body(
            device,
            super::record_view::DeferredRecordView::offscreen(frame),
            declare,
        )
    }

    pub(super) fn declare_body<F>(
        device: &Device<B>,
        frame: super::record_view::DeferredRecordView<'_, B>,
        declare: F,
    ) -> Result<DeferredDeclaration>
    where
        F: FnOnce(
            &mut crate::DeferredPhaseContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        let prefix = Self::declare_prefix(device, frame, declare)?;
        super::terminal::declare_terminal(prefix, None)
    }

    /// Declares the one Deferred Graph through lighting without selecting an
    /// output terminal.  The returned handles are versions in this exact Graph.
    pub(super) fn declare_prefix<F>(
        device: &Device<B>,
        frame: super::record_view::DeferredRecordView<'_, B>,
        declare: F,
    ) -> Result<super::prefix::DeferredGraphPrefix>
    where
        F: FnOnce(
            &mut crate::DeferredPhaseContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        Self::declare_prefix_in_graph(device, frame, GraphBuilder::default(), None, None, declare)
    }

    /// Continues an existing Graph with this Deferred namespace. Capture
    /// aliases refer to the producer's actual final versions in that Graph;
    /// they never create a second import for the producer allocation.
    pub(super) fn declare_prefix_in_graph<F>(
        device: &Device<B>,
        frame: super::record_view::DeferredRecordView<'_, B>,
        mut graph: GraphBuilder,
        aliases: Option<&[Option<super::super::definition::TextureHandle>]>,
        shared: Option<&mut super::imports::SharedDeferredImports>,
        declare: F,
    ) -> Result<super::prefix::DeferredGraphPrefix>
    where
        F: FnOnce(
            &mut crate::DeferredPhaseContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        let lane = device
            .lanes()
            .iter()
            .find(|lane| lane.class == LaneClass::Graphics)
            .map(|lane| lane.id)
            .ok_or(Error::WrongLane)?;
        let fresh_mip_chains = frame
            .resident
            .textures
            .iter()
            .enumerate()
            .map(|(index, texture)| {
                texture.desc.mip_levels > 1
                    && frame
                        .resident_upload
                        .textures
                        .iter()
                        .any(|upload| upload.texture == index)
            })
            .collect::<Vec<_>>();
        let super::imports::DeferredImports {
            resident_upload,
            frame_upload,
            uniform,
            streams,
            materials,
            environment_upload,
            environment,
            specular_upload,
            specular_view_uniform,
            specular_radiance,
            specular_brdf_lut,
            gbuffer,
            effects,
            msaa_gbuffer,
            msaa_effects,
            target,
            scene_color,
            surface_target,
            readback,
            row_pitch,
            deformation: deformation_handles,
        } = super::imports::import(&mut graph, &frame, &fresh_mip_chains, aliases, shared)?;

        let (
            upload_pass,
            (
                streams,
                uniform,
                mut materials,
                environment,
                specular_radiance,
                specular_brdf_lut,
                specular_view_uniform,
                deformation,
            ),
        ) = graph
            .add_pass("deferred-upload", |pass| {
                let mut output_streams = streams.clone();
                for upload in &frame.resident_upload.vertices {
                    let stream = *streams.get(upload.stream).ok_or(GraphError::InvalidRange)?;
                    pass.read_with_use(
                        resident_upload,
                        upload.source.clone(),
                        copy_read().uses,
                        BufferUsage::COPY_SRC,
                    )?;
                    output_streams[upload.stream] = pass.write_with_use(
                        stream,
                        upload.destination.clone(),
                        copy_write().uses,
                        BufferUsage::COPY_DST,
                    )?;
                }
                pass.read_with_use(
                    frame_upload,
                    frame.uniform_upload.source.clone(),
                    copy_read().uses,
                    BufferUsage::COPY_SRC,
                )?;
                let uniform = pass.write_with_use(
                    uniform,
                    frame.uniform_upload.destination.clone(),
                    copy_write().uses,
                    BufferUsage::COPY_DST,
                )?;
                let mut materials = materials;
                for copy in &frame.resident_upload.textures {
                    let texture = *materials
                        .get(copy.texture)
                        .ok_or(GraphError::InvalidRange)?;
                    pass.read_with_use(
                        resident_upload,
                        copy.source_offset
                            ..texture_copy_end(copy).map_err(|_| GraphError::InvalidRange)?,
                        copy_read().uses,
                        BufferUsage::COPY_SRC,
                    )?;
                    materials[copy.texture] = pass.write_texture_with_use(
                        texture,
                        one_color_mip_layer(copy.mip_level, copy.array_layer),
                        texture_copy_write().uses,
                        ImageLayout::CopyDst,
                        TextureUsage::COPY_DST,
                    )?;
                }
                let environment = if let (Some(upload), Some(environment), Some(owner)) =
                    (environment_upload, environment, frame.environment)
                {
                    pass.read_with_use(
                        upload,
                        0..owner.upload_desc.size,
                        copy_read().uses,
                        BufferUsage::COPY_SRC,
                    )?;
                    Some(pass.write_texture_with_use(
                        environment,
                        whole_color(),
                        texture_copy_write().uses,
                        ImageLayout::CopyDst,
                        TextureUsage::COPY_DST,
                    )?)
                } else {
                    None
                };
                let (specular_radiance, specular_brdf_lut, specular_view_uniform) = match (
                    specular_upload,
                    specular_radiance,
                    specular_brdf_lut,
                    specular_view_uniform,
                    frame.specular,
                ) {
                    (
                        Some(upload),
                        Some(radiance),
                        Some(brdf_lut),
                        Some(view_uniform),
                        Some(owner),
                    ) => {
                        let mut radiance = radiance;
                        for copy in &owner.radiance_copies {
                            let end = copy
                                .source_offset
                                .checked_add(
                                    u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image),
                                )
                                .ok_or(GraphError::InvalidRange)?;
                            pass.read_with_use(
                                upload,
                                copy.source_offset..end,
                                copy_read().uses,
                                BufferUsage::COPY_SRC,
                            )?;
                            radiance = pass.write_texture_with_use(
                                radiance,
                                one_color_mip(copy.mip),
                                texture_copy_write().uses,
                                ImageLayout::CopyDst,
                                TextureUsage::COPY_DST,
                            )?;
                        }
                        let copy = &owner.brdf_lut_copy;
                        let end = copy
                            .source_offset
                            .checked_add(
                                u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image),
                            )
                            .ok_or(GraphError::InvalidRange)?;
                        pass.read_with_use(
                            upload,
                            copy.source_offset..end,
                            copy_read().uses,
                            BufferUsage::COPY_SRC,
                        )?;
                        let brdf_lut = pass.write_texture_with_use(
                            brdf_lut,
                            one_color_mip(0),
                            texture_copy_write().uses,
                            ImageLayout::CopyDst,
                            TextureUsage::COPY_DST,
                        )?;
                        pass.read_with_use(
                            upload,
                            owner.view_uniform_copy.clone(),
                            copy_read().uses,
                            BufferUsage::COPY_SRC,
                        )?;
                        let view_uniform = pass.write_with_use(
                            view_uniform,
                            0..SPECULAR_VIEW_BYTES,
                            copy_write().uses,
                            BufferUsage::COPY_DST,
                        )?;
                        (Some(radiance), Some(brdf_lut), Some(view_uniform))
                    }
                    (None, None, None, None, None) => (None, None, None),
                    _ => return Err(GraphError::InvalidRange),
                };
                let deformation = if let (Some(owners), Some(handles)) =
                    (frame.deformation.as_ref(), deformation_handles)
                {
                    let (upload, source, product, parameters) = handles;
                    pass.read_with_use(
                        upload,
                        0..owners.source_bytes,
                        copy_read().uses,
                        BufferUsage::COPY_SRC,
                    )?;
                    let source = pass.write_with_use(
                        source,
                        0..owners.source_bytes,
                        copy_write().uses,
                        BufferUsage::COPY_DST,
                    )?;
                    pass.read_with_use(
                        upload,
                        owners.parameter_source.clone(),
                        copy_read().uses,
                        BufferUsage::COPY_SRC,
                    )?;
                    let parameters = pass.write_with_use(
                        parameters,
                        0..16,
                        copy_write().uses,
                        BufferUsage::COPY_DST,
                    )?;
                    Some(DeferredDeformationGraph {
                        source,
                        parameters,
                        product,
                        source_range: 0..owners.source_bytes,
                        parameter_range: 0..16,
                        product_range: 0..owners.source_bytes,
                        product_stream: 0,
                    })
                } else {
                    None
                };
                Ok((
                    output_streams,
                    uniform,
                    materials,
                    environment,
                    specular_radiance,
                    specular_brdf_lut,
                    specular_view_uniform,
                    deformation,
                ))
            })
            .map_err(|_| Error::InvalidState)?;

        let mut mip_generation = Vec::new();
        for (index, (texture, fresh)) in materials.iter_mut().zip(&fresh_mip_chains).enumerate() {
            let desc = &frame.resident.textures[index].desc;
            if *fresh && desc.usage.contains(TextureUsage::COPY_SRC) {
                let (generated, passes) = super::super::mip_generation::declare_chain(
                    &mut graph,
                    *texture,
                    index,
                    desc.mip_levels,
                )
                .map_err(|_| Error::InvalidState)?;
                *texture = generated;
                mip_generation.extend(passes);
            }
        }

        let draws = frame.draw_graph_data()?;
        let (
            deformation_pass,
            geometry_pass,
            resolve_pass,
            gbuffer,
            effects,
            lighting_pass,
            scene_color,
        ) = {
            let mut driver = DeferredPhaseDriver {
                graph: &mut graph,
                upload: upload_pass,
                streams: streams.clone(),
                draws,
                uniform,
                lighting_view: frame.lighting_view.clone(),
                materials: materials.clone(),
                environment,
                specular_radiance,
                specular_brdf_lut,
                specular_view_uniform,
                specular_mip_levels: frame.specular.map(|owner| owner.radiance.desc.mip_levels),
                gbuffer: [gbuffer[0], gbuffer[1], gbuffer[2]],
                effects,
                msaa_gbuffer: msaa_gbuffer.map(|items| [items[0], items[1], items[2]]),
                msaa_effects,
                target: scene_color,
                deformation,
                deformation_pass: None,
                geometry: None,
                resolve: None,
                lighting: None,
            };
            let mut context = crate::DeferredPhaseContext::new(&mut driver);
            declare(&mut context).map_err(|_| Error::InvalidState)?;
            driver.finish().map_err(|_| Error::InvalidState)?
        };

        let prefix = super::prefix::DeferredGraphPrefix {
            graph,
            lane,
            upload_pass,
            mip_generation,
            deformation_pass,
            geometry_pass,
            resolve_pass,
            lighting_pass,
            scene_color: scene_color.ok_or(Error::InvalidState)?,
            resident_upload,
            frame_upload,
            uniform,
            streams,
            materials,
            environment_upload,
            environment,
            specular_upload,
            specular_view_uniform,
            specular_radiance,
            specular_brdf_lut,
            gbuffer,
            effects,
            msaa_gbuffer,
            msaa_effects,
            surface_target,
            target,
            readback,
            row_pitch,
            deformation_handles,
            extent: frame.extent,
            uniform_alignment: device.capabilities().limits.uniform_offset_alignment,
            transform_bytes: frame
                .draws
                .iter()
                .map(|draw| draw.transform_bytes)
                .collect(),
        };
        Ok(prefix)
    }
}
