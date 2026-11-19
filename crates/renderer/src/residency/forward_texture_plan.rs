//! Validation and staging facts for resident material-texture uploads.

use super::*;

pub(super) fn plan_texture_uploads(
    bytes: &mut Vec<u8>,
    uploads: &mut Vec<ForwardTextureUpload>,
    index: usize,
    texture: ForwardResidentTextureInput,
) -> Result<ForwardTextureAllocation> {
    match texture {
        ForwardResidentTextureInput::Generated(texture) => {
            let copy = ForwardTextureCopyInput {
                bytes: texture.bytes,
                extent: texture.extent,
                bytes_per_row: texture.bytes_per_row,
            };
            validate_rgba8_copy(&copy)?;
            let max_mips = 1 + copy.extent[0].max(copy.extent[1]).ilog2();
            if texture.mip_levels == 0 || texture.mip_levels > max_mips {
                return Err(Error::InvalidRange);
            }
            let allocation = rgba8_allocation(
                copy.extent,
                texture.encoding,
                texture.mip_levels,
                1,
                TextureViewDimension::D2,
                texture.mip_levels > 1,
            );
            append_texture_upload(bytes, uploads, index, 0, 0, copy)?;
            Ok(allocation)
        }
        ForwardResidentTextureInput::Array(texture) => {
            let extent = texture.copies.first().ok_or(Error::InvalidRange)?.extent;
            let layers = usize::try_from(texture.layers).map_err(|_| Error::InvalidRange)?;
            let mips = usize::try_from(texture.mip_levels).map_err(|_| Error::InvalidRange)?;
            let uploaded_mips = if texture.generate_mips { 1 } else { mips };
            if extent.contains(&0)
                || layers == 0
                || mips == 0
                || (texture.generate_mips && mips == 1)
                || texture.mip_levels > 1 + extent[0].max(extent[1]).ilog2()
                || uploaded_mips.checked_mul(layers) != Some(texture.copies.len())
            {
                return Err(Error::InvalidRange);
            }
            for (mip, copies) in texture.copies.chunks_exact(layers).enumerate() {
                let expected_extent = extent.map(|size| (size >> mip).max(1));
                for copy in copies {
                    if copy.extent != expected_extent {
                        return Err(Error::InvalidRange);
                    }
                    validate_rgba8_copy(copy)?;
                }
            }
            for (copy_index, copy) in texture.copies.into_iter().enumerate() {
                let mip = (copy_index / layers) as u32;
                let layer = (copy_index % layers) as u32;
                append_texture_upload(bytes, uploads, index, mip, layer, copy)?;
            }
            Ok(rgba8_allocation(
                extent,
                texture.encoding,
                texture.mip_levels,
                texture.layers,
                TextureViewDimension::D2Array,
                texture.generate_mips,
            ))
        }
        ForwardResidentTextureInput::AuthoredRgba16Float(texture) => {
            validate_authored_rgba16float_texture(&texture)?;
            let base_extent = texture.mips[0].extent;
            let mip_levels = u32::try_from(texture.mips.len()).map_err(|_| Error::InvalidRange)?;
            for (mip_level, mip) in texture.mips.into_iter().enumerate() {
                append_texture_upload(bytes, uploads, index, mip_level as u32, 0, mip)?;
            }
            Ok(ForwardTextureAllocation {
                desc: TextureDesc {
                    label: Some("fluxel forward authored radiance texture".into()),
                    dimension: TextureDimension::D2,
                    extent: [base_extent[0], base_extent[1], 1],
                    array_layers: 1,
                    format: TextureFormat::Rgba16Float,
                    mip_levels,
                    samples: 1,
                    usage: TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
                    view_formats: Vec::new(),
                },
                view_dimension: TextureViewDimension::D2,
            })
        }
    }
}

fn rgba8_allocation(
    extent: [u32; 2],
    encoding: crate::preparation::TextureEncoding,
    mip_levels: u32,
    array_layers: u32,
    view_dimension: TextureViewDimension,
    generate_mips: bool,
) -> ForwardTextureAllocation {
    let mut usage = TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING;
    if generate_mips {
        usage |= TextureUsage::COPY_SRC;
    }
    ForwardTextureAllocation {
        desc: TextureDesc {
            label: Some("fluxel forward material texture".into()),
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            array_layers,
            format: encoding.rgba8_format(),
            mip_levels,
            samples: 1,
            usage,
            view_formats: Vec::new(),
        },
        view_dimension,
    }
}

fn validate_rgba8_copy(copy: &ForwardTextureCopyInput) -> Result<()> {
    if copy.extent.contains(&0)
        || copy.bytes_per_row == 0
        || copy.bytes_per_row % 256 != 0
        || copy.bytes_per_row < copy.extent[0].checked_mul(4).ok_or(Error::InvalidRange)?
    {
        return Err(Error::InvalidRange);
    }
    validate_texture_copy_bytes(copy.bytes_per_row, copy.extent, &copy.bytes)
}

fn validate_authored_rgba16float_texture(
    texture: &ForwardAuthoredRgba16FloatTextureInput,
) -> Result<()> {
    let Some(first) = texture.mips.first() else {
        return Err(Error::InvalidRange);
    };
    if first.extent.contains(&0) {
        return Err(Error::InvalidRange);
    }
    let complete_mips = 1 + first.extent[0].max(first.extent[1]).ilog2();
    if texture.mips.len() != usize::try_from(complete_mips).map_err(|_| Error::InvalidRange)? {
        return Err(Error::InvalidRange);
    }
    let mut expected = first.extent;
    for mip in &texture.mips {
        if mip.extent != expected
            || mip.bytes_per_row == 0
            || mip.bytes_per_row % 256 != 0
            || mip.bytes_per_row < expected[0].checked_mul(8).ok_or(Error::InvalidRange)?
        {
            return Err(Error::InvalidRange);
        }
        validate_texture_copy_bytes(mip.bytes_per_row, expected, &mip.bytes)?;
        expected = [(expected[0] / 2).max(1), (expected[1] / 2).max(1)];
    }
    Ok(())
}

fn validate_texture_copy_bytes(bytes_per_row: u32, extent: [u32; 2], bytes: &[u8]) -> Result<()> {
    let texture_bytes = u64::from(bytes_per_row)
        .checked_mul(u64::from(extent[1]))
        .ok_or(Error::InvalidRange)?;
    if u64::try_from(bytes.len()).map_err(|_| Error::InvalidRange)? != texture_bytes {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

fn append_texture_upload(
    bytes: &mut Vec<u8>,
    uploads: &mut Vec<ForwardTextureUpload>,
    texture: usize,
    mip_level: u32,
    array_layer: u32,
    source: ForwardTextureCopyInput,
) -> Result<()> {
    pad_to(bytes, 512)?;
    let source_offset = u64::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?;
    bytes
        .try_reserve(source.bytes.len())
        .map_err(|_| Error::OutOfMemory)?;
    bytes.extend_from_slice(&source.bytes);
    uploads.push(ForwardTextureUpload {
        texture,
        mip_level,
        array_layer,
        source_offset,
        bytes_per_row: source.bytes_per_row,
        rows_per_image: source.extent[1],
        extent: [source.extent[0], source.extent[1], 1],
    });
    Ok(())
}
