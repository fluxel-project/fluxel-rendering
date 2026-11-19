//! Concrete pre-recording checks for the offscreen Deferred frame.

use super::{
    DeferredFrame, Device, Error, ResourceBackend, Result, TextureFormat,
    declaration_validation::{
        matches_final_target_template, matches_gbuffer_template, validate_deformation,
        validate_draw_uniforms, validate_uploads,
    },
    draw_accesses,
};

pub(super) fn validate_frame_inputs<B: ResourceBackend>(
    device: &Device<B>,
    frame: &DeferredFrame<B>,
) -> Result<()> {
    super::validate_resident_texture_devices(&frame.resident, device.id())?;
    super::declaration_validation::validate_lighting_view(
        frame.pbr && frame.specular.is_none(),
        frame.lighting_view.as_ref(),
        &frame.uniform_upload.destination,
        frame.frame.uniform_desc.size,
        device.capabilities().limits.uniform_offset_alignment,
    )?;
    if frame.resident.device != device.id()
        || frame.extent.contains(&0)
        || device.capabilities().limits.max_color_attachments
            < 2 + u32::from(frame.effects.is_some()) + u32::from(frame.diffuse_roughness.is_some())
        || device.capabilities().limits.uniform_offset_alignment == 0
        || frame.resident.streams.is_empty()
        || frame.draws.is_empty()
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
    if frame.draws.iter().any(|draw| {
        !draw.pipelines_on(device.id())
            || draw.material_group.device() != device.id()
            || draw.transform_group.device() != device.id()
    }) || frame.lighting_pipeline.device() != device.id()
        || frame.lighting_group.device() != device.id()
        || frame.composite_pipeline.device() != device.id()
        || frame.composite_group.device() != device.id()
        || frame.scene_color.texture.device() != device.id()
        || frame.scene_color.view.device() != device.id()
        || frame.scene_color_sampler.device() != device.id()
    {
        return Err(Error::WrongDevice);
    }
    if !frame.scene_color.filterable {
        return Err(Error::Unsupported("filterable Deferred scene-color"));
    }
    crate::renderer::validate_ordinary_sampler(&frame.scene_color_sampler_desc)?;
    match (&frame.environment, &frame.lighting_environment_group) {
        (Some(environment), Some(group)) => {
            if environment.validate_for_sampling(device.id()).is_err()
                || group.device() != device.id()
                || environment.desc.extent.contains(&0)
                || environment.desc.format != TextureFormat::Rgba8Unorm
                || !environment
                    .desc
                    .usage
                    .contains(super::TextureUsage::COPY_DST | super::TextureUsage::TEXTURE_BINDING)
                || !environment
                    .upload_desc
                    .usage
                    .contains(super::BufferUsage::COPY_SRC)
                || environment.upload_desc.size != environment.upload_bytes.len() as u64
                || environment.bytes_per_row == 0
                || !environment.bytes_per_row.is_multiple_of(256)
                || u64::from(environment.bytes_per_row) < u64::from(environment.desc.extent[0]) * 4
            {
                return Err(Error::InvalidRange);
            }
        }
        (None, None) => {}
        // The lighting group can be present solely for specular IBL: it owns
        // the specular texture/sampler/uniform bindings even without the
        // legacy diffuse environment texture.
        (None, Some(group)) if frame.specular.is_some() && group.device() == device.id() => {}
        _ => return Err(Error::InvalidState),
    }
    if let Some(specular) = &frame.specular {
        specular.validate_for_sampling(device.id())?;
    }
    if let Some(deformation) = &frame.deformation {
        if device.compute().is_none() {
            return Err(Error::Unsupported("compute"));
        }
        if frame.draws.len() != 1 {
            return Err(Error::InvalidState);
        }
        validate_deformation(device, &frame.draws[0], deformation)?;
    }
    for (draw_index, product) in &frame.indirect {
        let draw = frame.draws.get(*draw_index).ok_or(Error::InvalidRange)?;
        product.validate_draw(&draw.vertices, &draw.instances)?;
        if product.upload.buffer.device() != device.id()
            || product.parameters.buffer.device() != device.id()
            || product.arguments.buffer.device() != device.id()
            || product.pipeline.device() != device.id()
            || product.group.device() != device.id()
        {
            return Err(Error::WrongDevice);
        }
    }
    for (texture, format) in frame.gbuffer.iter().zip([
        crate::residency::deferred_albedo_format(frame.pbr),
        crate::residency::deferred_normal_format(frame.pbr),
        TextureFormat::Depth32Float,
    ]) {
        if !matches_gbuffer_template(&texture.desc, frame.extent, format) {
            return Err(Error::InvalidRange);
        }
    }
    if let Some(effects) = &frame.effects {
        if effects.texture.device() != device.id()
            || effects.view.device() != device.id()
            || !matches_gbuffer_template(&effects.desc, frame.extent, TextureFormat::Rgba16Float)
        {
            return Err(Error::InvalidRange);
        }
    }
    if let Some(diffuse_roughness) = &frame.diffuse_roughness {
        if diffuse_roughness.texture.device() != device.id()
            || diffuse_roughness.view.device() != device.id()
            || !matches_gbuffer_template(
                &diffuse_roughness.desc,
                frame.extent,
                TextureFormat::R32Float,
            )
        {
            return Err(Error::InvalidRange);
        }
    }
    if let Some(msaa) = &frame.msaa {
        if msaa.pipeline.device() != device.id()
            || msaa.group.device() != device.id()
            || msaa
                .gbuffer
                .iter()
                .zip([
                    crate::residency::deferred_albedo_format(frame.pbr),
                    crate::residency::deferred_normal_format(frame.pbr),
                    TextureFormat::Depth32Float,
                ])
                .any(|(texture, format)| {
                    texture.texture.device() != device.id()
                        || texture.view.device() != device.id()
                        || texture.desc.format != format
                        || texture.desc.extent != [frame.extent[0], frame.extent[1], 1]
                        || texture.desc.samples <= 1
                        || !texture.desc.usage.contains(
                            super::TextureUsage::RENDER_ATTACHMENT
                                | super::TextureUsage::TEXTURE_BINDING,
                        )
                })
            || msaa.gbuffer[0].desc.samples != msaa.gbuffer[1].desc.samples
            || msaa.gbuffer[0].desc.samples != msaa.gbuffer[2].desc.samples
            || matches!(
                (&frame.effects, &msaa.effects),
                (Some(_), None) | (None, Some(_))
            )
            || matches!(
                (&frame.diffuse_roughness, &msaa.diffuse_roughness),
                (Some(_), None) | (None, Some(_))
            )
        {
            return Err(Error::InvalidRange);
        }
        if let Some(effects) = &msaa.effects {
            if effects.texture.device() != device.id()
                || effects.view.device() != device.id()
                || effects.desc.format != TextureFormat::Rgba16Float
                || effects.desc.extent != [frame.extent[0], frame.extent[1], 1]
                || effects.desc.samples != msaa.gbuffer[0].desc.samples
                || !effects.desc.usage.contains(
                    super::TextureUsage::RENDER_ATTACHMENT | super::TextureUsage::TEXTURE_BINDING,
                )
            {
                return Err(Error::InvalidRange);
            }
        }
        if let Some(diffuse_roughness) = &msaa.diffuse_roughness {
            if diffuse_roughness.texture.device() != device.id()
                || diffuse_roughness.view.device() != device.id()
                || diffuse_roughness.desc.format != TextureFormat::R32Float
                || diffuse_roughness.desc.extent != [frame.extent[0], frame.extent[1], 1]
                || diffuse_roughness.desc.samples != msaa.gbuffer[0].desc.samples
                || !diffuse_roughness.desc.usage.contains(
                    super::TextureUsage::RENDER_ATTACHMENT | super::TextureUsage::TEXTURE_BINDING,
                )
            {
                return Err(Error::InvalidRange);
            }
        }
    }
    if !super::declaration_validation::matches_scene_color_template(
        &frame.scene_color.desc,
        frame.extent,
    ) {
        return Err(Error::InvalidRange);
    }
    validate_uploads(frame)?;
    for draw in &frame.draws {
        if draw
            .textures
            .iter()
            .any(|texture| *texture >= frame.resident.textures.len())
        {
            return Err(Error::InvalidRange);
        }
        draw_accesses(frame, draw)?;
    }
    Ok(())
}
