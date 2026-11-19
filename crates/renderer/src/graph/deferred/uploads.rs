//! Validation of actual resident upload owners and ranges.

use super::*;

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
    let mut textures = resident
        .textures
        .iter()
        .map(|texture| {
            vec![vec![false; texture.desc.array_layers as usize]; texture.desc.mip_levels as usize]
        })
        .collect::<Vec<_>>();
    for copy in &uploads.textures {
        let texture = resident
            .textures
            .get(copy.texture)
            .ok_or(Error::InvalidRange)?;
        if !matches!(
            texture.desc.dimension,
            fluxel_rhi::api::TextureDimension::D2 | fluxel_rhi::api::TextureDimension::D3
        ) || (texture.desc.dimension == fluxel_rhi::api::TextureDimension::D2
            && texture.desc.extent[2] != 1)
            || (texture.desc.dimension == fluxel_rhi::api::TextureDimension::D3
                && texture.desc.array_layers != 1)
            || texture.desc.array_layers == 0
        {
            return Err(Error::InvalidRange);
        }
        let uploaded = textures
            .get_mut(copy.texture)
            .and_then(|mips| mips.get_mut(copy.mip_level as usize))
            .and_then(|layers| layers.get_mut(copy.array_layer as usize))
            .ok_or(Error::InvalidRange)?;
        let expected = [
            (texture.desc.extent[0] >> copy.mip_level).max(1),
            (texture.desc.extent[1] >> copy.mip_level).max(1),
            if texture.desc.dimension == fluxel_rhi::api::TextureDimension::D3 {
                (texture.desc.extent[2] >> copy.mip_level).max(1)
            } else {
                1
            },
        ];
        let bytes_per_texel = texture_bytes_per_texel(texture.desc.format)?;
        if std::mem::replace(uploaded, true)
            || !copy.source_offset.is_multiple_of(512)
            || copy.extent != expected
            || copy.rows_per_image != copy.extent[1]
            || copy.bytes_per_row == 0
            || !copy.bytes_per_row.is_multiple_of(256)
            || u64::from(copy.bytes_per_row)
                < u64::from(copy.extent[0]) * u64::from(bytes_per_texel)
            || texture_copy_end(copy)? > resident.upload_desc.size
            || texture_copy_end(copy)? > uploads.bytes.len() as u64
        {
            return Err(Error::InvalidRange);
        }
    }
    let textures_ready = textures
        .iter()
        .zip(&resident.textures)
        .all(|(uploaded, texture)| {
            if !matches!(
                texture.desc.dimension,
                fluxel_rhi::api::TextureDimension::D2 | fluxel_rhi::api::TextureDimension::D3
            ) || (texture.desc.dimension == fluxel_rhi::api::TextureDimension::D2
                && texture.desc.extent[2] != 1)
                || (texture.desc.dimension == fluxel_rhi::api::TextureDimension::D3
                    && texture.desc.array_layers != 1)
                || texture.desc.array_layers == 0
            {
                return false;
            }
            // Mip generation uses the D2 descriptor family: layered textures
            // seed every base layer and let the Graph fill the rest.
            (texture.desc.dimension == fluxel_rhi::api::TextureDimension::D2
                && texture.desc.usage.contains(TextureUsage::COPY_SRC)
                && uploaded
                    .first()
                    .is_some_and(|layers| layers.iter().all(|layer| *layer))
                && uploaded.iter().skip(1).flatten().all(|uploaded| !uploaded))
                || ((texture.desc.dimension != fluxel_rhi::api::TextureDimension::D2
                    || !texture.desc.usage.contains(TextureUsage::COPY_SRC))
                    && uploaded.iter().flatten().all(|uploaded| *uploaded))
                || (texture.incoming
                    == TextureUses {
                        stages: StageMask::FRAGMENT,
                        accesses: AccessMask::SHADER_READ,
                    }
                    && texture.layout == ImageLayout::ShaderReadOnly)
        });
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
    if !textures_ready || !streams_ready {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

fn texture_bytes_per_texel(format: TextureFormat) -> Result<u32> {
    match format {
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => Ok(4),
        TextureFormat::Rgba16Float => Ok(8),
        _ => Err(Error::InvalidRange),
    }
}
