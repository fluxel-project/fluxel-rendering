//! Actual Surface owner validation and geometry footprints.
use super::declaration_validation::{
    matches_gbuffer_template, matches_scene_color_template, validate_deformation, validate_draw_uniform_ranges,
    validate_resident_uploads,
};
use super::*;

pub(super) fn validate_surface_inputs<B: ResourceBackend>(
    device: &Device<B>,
    frame: &DeferredSurfaceFrame<B>,
) -> Result<()> {
    super::validate_resident_texture_devices(&frame.resident, device.id())?;
    super::declaration_validation::validate_lighting_view(
        frame.pbr && frame.specular.is_none(),
        frame.lighting_view.as_ref(),
        &frame.uniform_upload.destination,
        frame.frame.uniform_desc.size,
        device.capabilities().limits.uniform_offset_alignment,
    )?;
    validate_draw_uniform_ranges(
        frame.draws.first().ok_or(Error::InvalidRange)?,
        frame.frame.uniform_desc.size,
        device.capabilities().limits.uniform_offset_alignment,
    )?;
    if frame.resident.device != device.id()
        || frame.extent.contains(&0)
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
        || device.capabilities().limits.max_color_attachments
            < if frame.effects.is_some() { 3 } else { 2 }
        || device.capabilities().limits.uniform_offset_alignment == 0
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
            {
                return Err(Error::InvalidState);
            }
        }
        (None, None) => {}
        (None, Some(group)) if frame.specular.is_some() && group.device() == device.id() => {}
        _ => return Err(Error::InvalidState),
    }
    if let Some(specular) = &frame.specular {
        specular.validate_for_sampling(device.id())?;
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
    if !matches_scene_color_template(&frame.scene_color.desc, frame.extent) {
        return Err(Error::InvalidRange);
    }
    validate_resident_uploads(&frame.resident, &frame.resident_upload)?;
    if let Some(deformation) = &frame.deformation {
        if device.compute().is_none() {
            return Err(Error::Unsupported("compute"));
        }
        if frame.draws.len() != 1 {
            return Err(Error::InvalidState);
        }
        validate_deformation(device, &frame.draws[0], deformation)?;
    }
    for draw in &frame.draws {
        validate_draw_uniform_ranges(
            draw,
            frame.frame.uniform_desc.size,
            device.capabilities().limits.uniform_offset_alignment,
        )?;
        if draw
                .textures
                .iter()
                .any(|texture| *texture >= frame.resident.textures.len())
        {
            return Err(Error::InvalidRange);
        }
        surface_draw_accesses(frame, draw)?;
    }
    Ok(())
}

pub(super) fn surface_draw_accesses<B: ResourceBackend>(
    frame: &DeferredSurfaceFrame<B>,
    draw: &DeferredDraw<B>,
) -> Result<Vec<Option<Range<u64>>>> {
    draw_accesses_parts(&frame.resident, frame.deformation.as_ref(), draw)
}
