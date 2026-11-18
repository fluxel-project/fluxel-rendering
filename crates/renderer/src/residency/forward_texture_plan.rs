//! Validation and staging facts for resident material-texture uploads.

use super::*;

pub(super) fn validate_generated_texture(texture: &ForwardTextureInput) -> Result<()> {
    if texture.extent.contains(&0)
        || texture.mip_levels == 0
        || texture.bytes_per_row == 0
        || texture.bytes_per_row % 256 != 0
        || texture.bytes_per_row
            < texture.extent[0]
                .checked_mul(4)
                .ok_or(Error::InvalidRange)?
    {
        return Err(Error::InvalidRange);
    }
    let max_mip_levels = 1 + texture.extent[0].max(texture.extent[1]).ilog2();
    if texture.mip_levels > max_mip_levels {
        return Err(Error::InvalidRange);
    }
    validate_texture_copy_bytes(texture.bytes_per_row, texture.extent, &texture.bytes)
}

pub(super) fn validate_authored_rgba16float_texture(
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

pub(super) fn append_texture_upload(
    bytes: &mut Vec<u8>,
    uploads: &mut Vec<ForwardTextureUpload>,
    texture: usize,
    mip_level: u32,
    source: Vec<u8>,
    bytes_per_row: u32,
    extent: [u32; 2],
) -> Result<()> {
    pad_to(bytes, 512)?;
    let source_offset = u64::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?;
    bytes.extend_from_slice(&source);
    uploads.push(ForwardTextureUpload {
        texture,
        mip_level,
        source_offset,
        bytes_per_row,
        rows_per_image: extent[1],
        extent: [extent[0], extent[1], 1],
    });
    Ok(())
}
