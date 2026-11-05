//! Logical resource declaration for the bounded Deferred probe.

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
        let (target, readback, row_pitch) = match frame.source {
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
                        .import_buffer_desc(
                            owner.frame.readback_desc.clone(),
                            owner.frame.readback_incoming,
                        )
                        .map_err(|_| Error::InvalidState)?,
                ),
                Some(owner.readback_row_pitch),
            ),
            super::record_view::RecordSource::Surface(_) => (None, None, None),
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

        let (upload_pass, (streams, uniform, material, deformation)) = graph
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
                Ok((output_streams, uniform, material, deformation))
            })
            .map_err(|_| Error::InvalidState)?;

        let accesses = frame.draw_accesses()?;
        let (deformation_pass, geometry_pass, gbuffer, lighting_pass, target) = {
            let mut driver = DeferredPhaseDriver {
                graph: &mut graph,
                upload: upload_pass,
                streams: streams.clone(),
                stream_reads: accesses,
                uniform,
                tint: frame.draw.tint_range.clone(),
                transform: frame.draw.transform_range.clone(),
                material,
                gbuffer: [gbuffer[0], gbuffer[1], gbuffer[2]],
                target,
                deformation,
                deformation_pass: None,
                geometry: None,
                lighting: None,
            };
            let mut context = crate::DeferredPhaseContext::new(&mut driver);
            declare(&mut context).map_err(|_| Error::InvalidState)?;
            driver.finish().map_err(|_| Error::InvalidState)?
        };

        let output = if let Some(readback) = readback {
            let target = target.ok_or(Error::InvalidState)?;
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
            DeferredOutputDeclaration::Offscreen { readback_pass }
        } else {
            DeferredOutputDeclaration::Surface
        };

        Ok(DeferredDeclaration {
            graph,
            lane,
            upload_pass,
            deformation_pass,
            geometry_pass,
            lighting_pass,
            output,
            resident_upload_slot: BufferSlot::from(resident_upload),
            frame_upload_slot: BufferSlot::from(frame_upload),
            uniform_slot: BufferSlot::from(uniform),
            stream_slots: streams.into_iter().map(BufferSlot::from).collect(),
            texture_slot: TextureSlot::from(material),
            gbuffer_slots: std::array::from_fn(|index| TextureSlot::from(gbuffer[index])),
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

fn validate_frame_inputs<B: ResourceBackend>(
    device: &Device<B>,
    frame: &DeferredFrame<B>,
) -> Result<()> {
    let required_gbuffer = [
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8Unorm,
        TextureFormat::Depth32Float,
    ];
    if frame.resident.device != device.id()
        || frame.extent.contains(&0)
        || device.capabilities().limits.max_color_attachments < 2
        || device.capabilities().limits.uniform_offset_alignment == 0
        || frame.resident.streams.is_empty()
        || frame.resident.textures.is_empty()
        || frame.draw.vertex_layouts.len() != frame.resident.streams.len()
        || frame.draw.vertex_ranges.len() != frame.resident.streams.len()
        || frame.uniform_upload.source.start >= frame.uniform_upload.source.end
        || frame.uniform_upload.destination.start >= frame.uniform_upload.destination.end
        || frame.uniform_upload.source.end - frame.uniform_upload.source.start
            != frame.uniform_upload.destination.end - frame.uniform_upload.destination.start
        || usize::try_from(frame.uniform_upload.source.end).ok() != Some(frame.uniform_bytes.len())
        || !frame
            .uniform_upload
            .destination
            .start
            .is_multiple_of(device.capabilities().limits.uniform_offset_alignment)
        || validate_draw_uniforms(frame, device.capabilities().limits.uniform_offset_alignment)
            .is_err()
        || frame.readback_row_pitch == 0
        || !frame.readback_row_pitch.is_multiple_of(256)
        || u64::from(frame.readback_row_pitch) < u64::from(frame.extent[0]) * 4
        || !matches_final_target_template(&frame.frame.target_desc, frame.extent)
        || frame.frame.readback_desc.size
            != u64::from(frame.readback_row_pitch) * u64::from(frame.extent[1])
    {
        return Err(Error::InvalidRange);
    }
    if frame.draw.pipeline.device() != device.id()
        || frame.draw.material_group.device() != device.id()
        || frame.draw.transform_group.device() != device.id()
        || frame.lighting_pipeline.device() != device.id()
        || frame.lighting_group.device() != device.id()
    {
        return Err(Error::WrongDevice);
    }
    if let Some(deformation) = &frame.deformation {
        if device.compute().is_none() {
            return Err(Error::Unsupported("compute"));
        }
        validate_deformation(device, &frame.draw, deformation)?;
    }
    for (texture, format) in frame.gbuffer.iter().zip(required_gbuffer) {
        if !matches_gbuffer_template(&texture.desc, frame.extent, format) {
            return Err(Error::InvalidRange);
        }
    }
    validate_uploads(frame)?;
    draw_accesses(frame, &frame.draw)?;
    Ok(())
}

pub(super) fn validate_deformation<B: ResourceBackend>(
    device: &Device<B>,
    draw: &DeferredDraw<B>,
    deformation: &crate::residency::DeferredDeformation<B>,
) -> Result<()> {
    if deformation.upload.buffer.device() != device.id()
        || deformation.source.buffer.device() != device.id()
        || deformation.product.buffer.device() != device.id()
        || deformation.parameters.buffer.device() != device.id()
        || deformation.pipeline.device() != device.id()
        || deformation.group.device() != device.id()
    {
        return Err(Error::WrongDevice);
    }
    if deformation.source.instance != deformation.source_instance
        || deformation.product.instance != deformation.product_instance
        || deformation.parameters.instance != deformation.parameter_instance
        || deformation.source_bytes == 0
        || deformation.source.desc.size != deformation.source_bytes
        || deformation.product.desc.size != deformation.source_bytes
        || deformation
            .parameter_source
            .end
            .checked_sub(deformation.parameter_source.start)
            != Some(16)
        || deformation.parameter_source.end > deformation.upload.desc.size
        || deformation.parameters.desc.size != 16
        || !deformation
            .upload
            .desc
            .usage
            .contains(BufferUsage::COPY_SRC)
        || !deformation
            .source
            .desc
            .usage
            .contains(BufferUsage::COPY_DST | BufferUsage::STORAGE)
        || !deformation
            .product
            .desc
            .usage
            .contains(BufferUsage::STORAGE | BufferUsage::VERTEX)
        || !deformation
            .parameters
            .desc
            .usage
            .contains(BufferUsage::COPY_DST | BufferUsage::UNIFORM)
        || deformation.upload_bytes.len()
            != usize::try_from(deformation.upload.desc.size).unwrap_or(usize::MAX)
        || draw.vertex_layouts.len() != 1
        || draw.vertex_ranges.len() != 1
        || u64::from(deformation.vertex_count).checked_mul(12) != Some(deformation.source_bytes)
        || deformation.groups != [deformation.vertex_count.div_ceil(64), 1, 1]
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

pub(super) fn validate_draw_uniforms<B: ResourceBackend>(
    frame: &DeferredFrame<B>,
    alignment: u64,
) -> Result<()> {
    validate_draw_uniform_ranges(&frame.draw, frame.frame.uniform_desc.size, alignment)
}

pub(super) fn validate_draw_uniform_ranges<B: ResourceBackend>(
    draw: &DeferredDraw<B>,
    uniform_size: u64,
    alignment: u64,
) -> Result<()> {
    if alignment == 0 {
        return Err(Error::InvalidRange);
    }
    if !matches!(draw.transform_bytes, 64 | 128) {
        return Err(Error::InvalidRange);
    }
    for (range, bytes) in [
        (&draw.tint_range, 16_u64),
        (&draw.transform_range, draw.transform_bytes),
    ] {
        if range.start >= range.end
            || range.end - range.start != bytes
            || range.end > uniform_size
            || !range.start.is_multiple_of(alignment)
            || u32::try_from(range.start).is_err()
        {
            return Err(Error::InvalidRange);
        }
    }
    Ok(())
}

pub(super) fn matches_gbuffer_template(
    desc: &fluxel_rhi::api::TextureDesc,
    extent: [u32; 2],
    format: TextureFormat,
) -> bool {
    desc.dimension == fluxel_rhi::api::TextureDimension::D2
        && desc.extent == [extent[0], extent[1], 1]
        && desc.array_layers == 1
        && desc.samples == 1
        && desc.format == format
        && desc
            .usage
            .contains(TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING)
}

pub(super) fn matches_final_target_template(
    desc: &fluxel_rhi::api::TextureDesc,
    extent: [u32; 2],
) -> bool {
    desc.dimension == fluxel_rhi::api::TextureDimension::D2
        && desc.extent == [extent[0], extent[1], 1]
        && desc.array_layers == 1
        && desc.samples == 1
        && desc.format == TextureFormat::Rgba8Unorm
        && desc
            .usage
            .contains(TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC)
}

pub(super) fn validate_uploads<B: ResourceBackend>(frame: &DeferredFrame<B>) -> Result<()> {
    validate_resident_uploads(&frame.resident, &frame.resident_upload)
}

pub(super) fn validate_resident_uploads<B: ResourceBackend>(
    resident: &crate::residency::ForwardResidentBuffers<B>,
    uploads: &crate::residency::ForwardResidentUpload,
) -> Result<()> {
    // A resident cache hit imports the already initialized stream and texture
    // owners in their post-upload incoming states. It deliberately has no
    // static copy declarations and therefore no resident staging bytes.
    if uploads.bytes.is_empty() && (!uploads.vertices.is_empty() || !uploads.textures.is_empty()) {
        return Err(Error::InvalidRange);
    }
    let mut streams = vec![false; resident.streams.len()];
    for upload in &uploads.vertices {
        let owner = resident
            .streams
            .get(upload.stream)
            .ok_or(Error::InvalidRange)?;
        let seen = std::mem::replace(
            streams.get_mut(upload.stream).ok_or(Error::InvalidRange)?,
            true,
        );
        if seen
            || upload.source.start >= upload.source.end
            || upload.destination.start >= upload.destination.end
            || upload.source.end > resident.upload_desc.size
            || upload.source.end > uploads.bytes.len() as u64
            || upload.destination.end > owner.desc.size
            || upload.source.end - upload.source.start
                != upload.destination.end - upload.destination.start
        {
            return Err(Error::InvalidRange);
        }
    }
    let mut texture_uploaded = false;
    for copy in &uploads.textures {
        if texture_uploaded
            || copy.texture != 0
            || !copy.source_offset.is_multiple_of(512)
            || copy.extent != resident.textures[0].desc.extent
            || copy.rows_per_image != copy.extent[1]
            || copy.bytes_per_row == 0
            || !copy.bytes_per_row.is_multiple_of(256)
            || u64::from(copy.bytes_per_row) < u64::from(copy.extent[0]) * 4
            || texture_copy_end(copy)? > resident.upload_desc.size
            || texture_copy_end(copy)? > uploads.bytes.len() as u64
        {
            return Err(Error::InvalidRange);
        }
        texture_uploaded = true;
    }
    let texture = &resident.textures[0];
    let texture_ready = texture.incoming
        == TextureUses {
            stages: StageMask::FRAGMENT,
            accesses: AccessMask::SHADER_READ,
        }
        && texture.layout == ImageLayout::ShaderReadOnly;
    let streams_ready = streams
        .iter()
        .zip(&resident.streams)
        .all(|(uploaded, owner)| {
            *uploaded
                || owner.incoming
                    == BufferUses {
                        stages: StageMask::VERTEX_INPUT,
                        accesses: AccessMask::VERTEX_READ,
                    }
        });
    if (!texture_uploaded && !texture_ready) || !streams_ready {
        return Err(Error::InvalidRange);
    }
    Ok(())
}
