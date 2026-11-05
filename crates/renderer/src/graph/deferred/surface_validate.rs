//! Actual Surface owner validation and geometry footprints.
use super::declaration::{
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
        || device.capabilities().limits.max_color_attachments < 2
        || device.capabilities().limits.uniform_offset_alignment == 0
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
    for (texture, format) in frame.gbuffer.iter().zip([
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8Unorm,
        TextureFormat::Depth32Float,
    ]) {
        if !matches_gbuffer_template(&texture.desc, frame.extent, format) {
            return Err(Error::InvalidRange);
        }
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
