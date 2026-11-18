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
        || draw.vertex_streams.as_slice() != [0]
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
    if frame.draws.is_empty() {
        return Err(Error::InvalidRange);
    }
    for draw in &frame.draws {
        validate_draw_uniform_ranges(draw, frame.frame.uniform_desc.size, alignment)?;
    }
    Ok(())
}

pub(super) fn validate_draw_uniform_ranges<B: ResourceBackend>(
    draw: &DeferredDraw<B>,
    uniform_size: u64,
    alignment: u64,
) -> Result<()> {
    if alignment == 0 {
        return Err(Error::InvalidRange);
    }
    if !matches!(
        draw.transform_bytes,
        64 | crate::frame_data::forward::PBR_TRANSFORM_BYTES
    ) {
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

/// Deferred lighting stays in a sampleable frame-local owner. The final
/// readback or Surface target has its own fixed output descriptor, so HDR only
/// expands this source format set and never its copy layout.
pub(super) fn matches_scene_color_template(
    desc: &fluxel_rhi::api::TextureDesc,
    extent: [u32; 2],
) -> bool {
    matches!(
        desc.format,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba16Float
    ) && desc.dimension == fluxel_rhi::api::TextureDimension::D2
        && desc.extent == [extent[0], extent[1], 1]
        && desc.array_layers == 1
        && desc.mip_levels == 1
        && desc.samples == 1
        && desc
            .usage
            .contains(TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING)
}

pub(super) fn validate_uploads<B: ResourceBackend>(frame: &DeferredFrame<B>) -> Result<()> {
    validate_resident_uploads(&frame.resident, &frame.resident_upload)
}

pub(super) use super::uploads::validate_resident_uploads;

pub(super) fn validate_lighting_view(
    required: bool,
    range: Option<&std::ops::Range<u64>>,
    upload: &std::ops::Range<u64>,
    buffer_size: u64,
    alignment: u64,
) -> Result<()> {
    let Some(range) = range else {
        return if required {
            Err(Error::InvalidRange)
        } else {
            Ok(())
        };
    };
    if !required
        || alignment == 0
        || !range.start.is_multiple_of(alignment)
        || range.start < upload.start
        || range.end != upload.end
        || range.end > buffer_size
        || range.end.checked_sub(range.start)
            != Some(crate::frame_data::specular_view::BASIC_DEFERRED_VIEW_BYTES)
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}
