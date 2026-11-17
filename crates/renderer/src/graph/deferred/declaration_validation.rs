//! Deferred declaration input validation helpers.

use super::*;

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
    if !matches!(draw.transform_bytes, 64 | 192) {
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

pub(super) use super::uploads::validate_resident_uploads;
