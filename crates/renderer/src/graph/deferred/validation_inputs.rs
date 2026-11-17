//! Validation of Deferred draw inputs and upload footprints.
use super::*;

pub(in crate::graph::deferred) fn validate_draw_uniforms<B: ResourceBackend>(
    frame: &DeferredFrame<B>,
    alignment: u64,
) -> Result<()> {
    validate_draw_uniform_ranges(&frame.draw, frame.frame.uniform_desc.size, alignment)
}

pub(in crate::graph::deferred) fn validate_draw_uniform_ranges<B: ResourceBackend>(
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

pub(in crate::graph::deferred) fn matches_gbuffer_template(
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

pub(in crate::graph::deferred) fn matches_final_target_template(
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

pub(in crate::graph::deferred) fn validate_uploads<B: ResourceBackend>(frame: &DeferredFrame<B>) -> Result<()> {
    validate_resident_uploads(&frame.resident, &frame.resident_upload)
}

pub(in crate::graph::deferred) fn validate_resident_uploads<B: ResourceBackend>(
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
