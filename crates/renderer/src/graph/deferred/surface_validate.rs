//! Actual Surface owner validation and geometry footprints.
use super::declaration_validation::{
    matches_gbuffer_template, validate_deformation, validate_draw_uniform_ranges,
    validate_resident_uploads,
};
use super::*;

pub(super) fn validate_surface_inputs<B: ResourceBackend>(
    device: &Device<B>,
    frame: &DeferredSurfaceFrame<B>,
) -> Result<()> {
    validate_draw_uniform_ranges(
        &frame.draw,
        frame.frame.uniform_desc.size,
        device.capabilities().limits.uniform_offset_alignment,
    )?;
    if frame.resident.device != device.id()
        || frame.extent.contains(&0)
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
        || device.capabilities().limits.max_color_attachments
            < if frame.effects.is_some() { 3 } else { 2 }
        || device.capabilities().limits.uniform_offset_alignment == 0
    {
        return Err(Error::InvalidRange);
    }
    if frame.draw.pipeline.device() != device.id()
        || frame.draw.material_group.device() != device.id()
        || frame.draw.transform_group.device() != device.id()
        || frame.lighting_pipeline.device() != device.id()
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
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8Unorm,
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
    if !matches_gbuffer_template(
        &frame.scene_color.desc,
        frame.extent,
        TextureFormat::Rgba8Unorm,
    ) {
        return Err(Error::InvalidRange);
    }
    validate_resident_uploads(&frame.resident, &frame.resident_upload)?;
    if let Some(deformation) = &frame.deformation {
        if device.compute().is_none() {
            return Err(Error::Unsupported("compute"));
        }
        validate_deformation(device, &frame.draw, deformation)?;
    }
    surface_draw_accesses(frame, &frame.draw)?;
    Ok(())
}

pub(super) fn surface_draw_accesses<B: ResourceBackend>(
    frame: &DeferredSurfaceFrame<B>,
    draw: &DeferredDraw<B>,
) -> Result<Vec<Option<Range<u64>>>> {
    draw_accesses_parts(&frame.resident, frame.deformation.as_ref(), draw)
}
