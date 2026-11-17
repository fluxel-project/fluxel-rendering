//! Logical resource declaration for the bounded Deferred probe.

use super::validation::validate_frame_inputs;
use super::*;

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
            .map(|stream| graph.import_buffer_desc(stream.desc.clone(), stream.incoming))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| Error::InvalidState)?;
        let material = graph
            .import_texture_with_state(
                frame.resident.textures[0].desc.clone(),
                frame.resident.textures[0].incoming,
                frame.resident.textures[0].layout,
            )
            .map_err(|_| Error::InvalidState)?;
        let environment_upload = frame
            .environment
            .map(|owner| graph.import_buffer_desc(owner.upload_desc.clone(), owner.upload_incoming))
            .transpose()
            .map_err(|_| Error::InvalidState)?;
        let environment = frame
            .environment
            .map(|owner| {
                graph.import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
            })
            .transpose()
            .map_err(|_| Error::InvalidState)?;
        let specular_upload = frame
            .specular
            .map(|owner| graph.import_buffer_desc(owner.upload_desc.clone(), owner.upload_incoming))
            .transpose()
            .map_err(|_| Error::InvalidState)?;
        let specular_view_uniform = frame
            .specular
            .map(|owner| {
                graph.import_buffer_desc(
                    owner.view_uniform_desc.clone(),
                    owner.view_uniform_incoming,
                )
            })
            .transpose()
            .map_err(|_| Error::InvalidState)?;
        let specular_radiance = frame
            .specular
            .map(|owner| {
                graph.import_texture_with_state(
                    owner.radiance.desc.clone(),
                    owner.radiance.incoming,
                    owner.radiance.layout,
                )
            })
            .transpose()
            .map_err(|_| Error::InvalidState)?;
        let specular_brdf_lut = frame
            .specular
            .map(|owner| {
                graph.import_texture_with_state(
                    owner.brdf_lut.desc.clone(),
                    owner.brdf_lut.incoming,
                    owner.brdf_lut.layout,
                )
            })
            .transpose()
            .map_err(|_| Error::InvalidState)?;
        let gbuffer = frame
            .gbuffer
            .iter()
            .map(|texture| {
                graph.import_texture_with_state(
                    texture.desc.clone(),
                    texture.incoming,
                    texture.layout,
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| Error::InvalidState)?;
        let effects = frame
            .effects
            .map(|texture| {
                graph.import_texture_with_state(
                    texture.desc.clone(),
                    texture.incoming,
                    texture.layout,
                )
            })
            .transpose()
            .map_err(|_| Error::InvalidState)?;
        let (target, scene_color, surface_target, readback, row_pitch) = match frame.source {
            super::record_view::RecordSource::Offscreen(owner) => (
                Some(
                    graph
                        .import_texture_with_state(
                            owner.frame.target_desc.clone(),
                            owner.frame.target_incoming,
                            owner.frame.target_layout,
                        )
                        .map_err(|_| Error::InvalidState)?,
                ),
                Some(
                    graph
                        .import_texture_with_state(
                            owner.scene_color.desc.clone(),
                            owner.scene_color.incoming,
                            owner.scene_color.layout,
                        )
                        .map_err(|_| Error::InvalidState)?,
                ),
                None,
                Some(
                    graph
                        .import_buffer_desc(
                            owner.frame.readback_desc.clone(),
                            owner.frame.readback_incoming,
                        )
                        .map_err(|_| Error::InvalidState)?,
                ),
                Some(owner.readback_row_pitch),
            ),
            super::record_view::RecordSource::Surface { frame: owner, .. } => {
                let desc = fluxel_rhi::api::TextureDesc {
                    label: Some("fluxel deferred surface lease".into()),
                    dimension: fluxel_rhi::api::TextureDimension::D2,
                    extent: [owner.extent[0], owner.extent[1], 1],
                    array_layers: 1,
                    format: owner.color_format,
                    mip_levels: 1,
                    samples: 1,
                    usage: TextureUsage::RENDER_ATTACHMENT,
                    view_formats: Vec::new(),
                };
                (
                    None,
                    Some(
                        graph
                            .import_texture_with_state(
                                owner.scene_color.desc.clone(),
                                owner.scene_color.incoming,
                                owner.scene_color.layout,
                            )
                            .map_err(|_| Error::InvalidState)?,
                    ),
                    Some(
                        graph
                            .import_surface_texture(desc)
                            .map_err(|_| Error::InvalidState)?,
                    ),
                    None,
                    None,
                )
            }
        };

        let deformation_handles = frame
            .deformation
            .as_ref()
            .map(|deformation| {
                Ok::<_, Error>((
                    graph
                        .import_buffer_desc(
                            deformation.upload.desc.clone(),
                            deformation.upload.incoming,
                        )
                        .map_err(|_| Error::InvalidState)?,
                    graph
                        .import_buffer_desc(
                            deformation.source.desc.clone(),
                            deformation.source.incoming,
                        )
                        .map_err(|_| Error::InvalidState)?,
                    graph
                        .import_buffer_desc(
                            deformation.product.desc.clone(),
                            deformation.product.incoming,
                        )
                        .map_err(|_| Error::InvalidState)?,
                    graph
                        .import_buffer_desc(
                            deformation.parameters.desc.clone(),
                            deformation.parameters.incoming,
                        )
                        .map_err(|_| Error::InvalidState)?,
                ))
            })
            .transpose()?;

        let (
            upload_pass,
            (
                streams,
                uniform,
                material,
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
                let mut material = material;
                for copy in &frame.resident_upload.textures {
                    if copy.texture != 0 {
                        return Err(GraphError::InvalidRange);
                    }
                    pass.read_with_use(
                        resident_upload,
                        copy.source_offset
                            ..texture_copy_end(copy).map_err(|_| GraphError::InvalidRange)?,
                        copy_read().uses,
                        BufferUsage::COPY_SRC,
                    )?;
                    material = pass.write_texture_with_use(
                        material,
                        whole_color(),
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
                            0..112,
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
                    material,
                    environment,
                    specular_radiance,
                    specular_brdf_lut,
                    specular_view_uniform,
                    deformation,
                ))
            })
            .map_err(|_| Error::InvalidState)?;

        let accesses = frame.draw_accesses()?;
        let (deformation_pass, geometry_pass, gbuffer, lighting_pass, scene_color) = {
            let mut driver = DeferredPhaseDriver {
                graph: &mut graph,
                upload: upload_pass,
                streams: streams.clone(),
                stream_reads: accesses,
                uniform,
                tint: frame.draw.tint_range.clone(),
                transform: frame.draw.transform_range.clone(),
                material,
                environment,
                specular_radiance,
                specular_brdf_lut,
                specular_view_uniform,
                specular_mip_levels: frame.specular.map(|owner| owner.radiance.desc.mip_levels),
                gbuffer: [gbuffer[0], gbuffer[1], gbuffer[2]],
                effects,
                target: scene_color,
                deformation,
                deformation_pass: None,
                geometry: None,
                lighting: None,
            };
            let mut context = crate::DeferredPhaseContext::new(&mut driver);
            declare(&mut context).map_err(|_| Error::InvalidState)?;
            driver.finish().map_err(|_| Error::InvalidState)?
        };

        let (composite_pass, output) = if let Some(readback) = readback {
            let target = target.ok_or(Error::InvalidState)?;
            let scene_color = scene_color.ok_or(Error::InvalidState)?;
            let (composite_pass, target) = graph
                .add_pass("deferred-scene-color-composite", |pass| {
                    pass.depends_on(lighting_pass)?;
                    pass.read_texture_with_use(
                        scene_color,
                        whole_color(),
                        sampled_read().uses,
                        ImageLayout::ShaderReadOnly,
                        TextureUsage::TEXTURE_BINDING,
                    )?;
                    let target = pass.write_texture_with_use(
                        target,
                        whole_color(),
                        color_write().uses,
                        ImageLayout::ColorAttachment,
                        TextureUsage::RENDER_ATTACHMENT,
                    )?;
                    Ok(target)
                })
                .map_err(|_| Error::InvalidState)?;
            let (readback_pass, ()) = graph
                .add_pass("deferred-readback", |pass| {
                    let length = u64::from(row_pitch.ok_or(GraphError::InvalidRange)?)
                        .checked_mul(u64::from(frame.extent[1]))
                        .ok_or(GraphError::InvalidRange)?;
                    pass.read_texture_with_use(
                        target,
                        whole_color(),
                        texture_copy_read().uses,
                        ImageLayout::CopySrc,
                        TextureUsage::COPY_SRC,
                    )?;
                    pass.write_with_use(
                        readback,
                        0..length,
                        copy_write().uses,
                        BufferUsage::COPY_DST,
                    )?;
                    pass.side_effect();
                    Ok(())
                })
                .map_err(|_| Error::InvalidState)?;
            (
                Some(composite_pass),
                DeferredOutputDeclaration::Offscreen { readback_pass },
            )
        } else {
            let scene_color = scene_color.ok_or(Error::InvalidState)?;
            let surface_target = surface_target.ok_or(Error::InvalidState)?;
            let (composite_pass, ()) = graph
                .add_pass("deferred-surface-scene-color-composite", |pass| {
                    pass.depends_on(lighting_pass)?;
                    pass.read_texture_with_use(
                        scene_color,
                        whole_color(),
                        sampled_read().uses,
                        ImageLayout::ShaderReadOnly,
                        TextureUsage::TEXTURE_BINDING,
                    )?;
                    pass.write_texture_with_use(
                        surface_target,
                        whole_color(),
                        color_write().uses,
                        ImageLayout::ColorAttachment,
                        TextureUsage::RENDER_ATTACHMENT,
                    )?;
                    // Presentation consumes this lease after the Graph's
                    // command steps; retain the terminal write in the live
                    // Graph rather than treating it as an untracked side
                    // effect at recording time.
                    pass.side_effect();
                    Ok(())
                })
                .map_err(|_| Error::InvalidState)?;
            (Some(composite_pass), DeferredOutputDeclaration::Surface)
        };

        Ok(DeferredDeclaration {
            graph,
            lane,
            upload_pass,
            deformation_pass,
            geometry_pass,
            lighting_pass,
            composite_pass,
            output,
            resident_upload_slot: BufferSlot::from(resident_upload),
            frame_upload_slot: BufferSlot::from(frame_upload),
            uniform_slot: BufferSlot::from(uniform),
            stream_slots: streams.into_iter().map(BufferSlot::from).collect(),
            texture_slot: TextureSlot::from(material),
            environment_upload_slot: environment_upload.map(BufferSlot::from),
            environment_slot: environment.map(TextureSlot::from),
            specular_upload_slot: specular_upload.map(BufferSlot::from),
            specular_view_uniform_slot: specular_view_uniform.map(BufferSlot::from),
            specular_radiance_slot: specular_radiance.map(TextureSlot::from),
            specular_brdf_lut_slot: specular_brdf_lut.map(TextureSlot::from),
            gbuffer_slots: std::array::from_fn(|index| TextureSlot::from(gbuffer[index])),
            effects_slot: effects.map(TextureSlot::from),
            scene_color_slot: scene_color.map(TextureSlot::from),
            surface_target_slot: surface_target.map(TextureSlot::from),
            target_slot: target.map(TextureSlot::from),
            readback_slot: readback.map(BufferSlot::from),
            deformation_slots: deformation_handles.map(|(upload, source, product, parameters)| {
                [
                    BufferSlot::from(upload),
                    BufferSlot::from(source),
                    BufferSlot::from(product),
                    BufferSlot::from(parameters),
                ]
            }),
            extent: frame.extent,
            uniform_alignment: device.capabilities().limits.uniform_offset_alignment,
            transform_bytes: frame.draw.transform_bytes,
        })
    }
}
