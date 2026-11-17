//! Concrete pre-recording checks for the offscreen Deferred frame.

use super::{
    DeferredFrame, Device, Error, ResourceBackend, Result, TextureFormat,
    declaration::{
        matches_final_target_template, matches_gbuffer_template, validate_deformation,
        validate_draw_uniforms, validate_uploads,
    },
    draw_accesses,
};

pub(super) fn validate_frame_inputs<B: ResourceBackend>(
    device: &Device<B>,
    frame: &DeferredFrame<B>,
) -> Result<()> {
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
        _ => return Err(Error::InvalidState),
    }
    if let Some(deformation) = &frame.deformation {
        if device.compute().is_none() {
            return Err(Error::Unsupported("compute"));
        }
        validate_deformation(device, &frame.draw, deformation)?;
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
    if !matches_gbuffer_template(
        &frame.scene_color.desc,
        frame.extent,
        TextureFormat::Rgba8Unorm,
    ) {
        return Err(Error::InvalidRange);
    }
    validate_uploads(frame)?;
    draw_accesses(frame, &frame.draw)?;
    Ok(())
}
