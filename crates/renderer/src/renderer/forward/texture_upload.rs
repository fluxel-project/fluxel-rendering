//! CPU staging for immutable RGBA8 material texture uploads.

use fluxel_rhi::api::Error;

use crate::{ColorTextureData, residency::ForwardTextureInput};

pub(super) fn stage(
    data: &ColorTextureData,
    max_buffer_bytes: u64,
) -> std::result::Result<ForwardTextureInput, Error> {
    let row = u64::from(data.extent[0])
        .checked_mul(4)
        .ok_or(Error::InvalidRange)?;
    let pitch = row.checked_add(255).ok_or(Error::InvalidRange)? / 256 * 256;
    let pitch = u32::try_from(pitch).map_err(|_| Error::InvalidRange)?;
    let size = u64::from(pitch)
        .checked_mul(u64::from(data.extent[1]))
        .ok_or(Error::InvalidRange)?;
    if size > max_buffer_bytes {
        return Err(Error::InvalidRange);
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(usize::try_from(size).map_err(|_| Error::InvalidRange)?)
        .map_err(|_| Error::OutOfMemory)?;
    bytes.resize(usize::try_from(size).map_err(|_| Error::InvalidRange)?, 0);
    let tight = usize::try_from(data.extent[0])
        .map_err(|_| Error::InvalidRange)?
        .checked_mul(4)
        .ok_or(Error::InvalidRange)?;
    for (row, source) in data.rgba8.chunks_exact(tight).enumerate() {
        let start = row.checked_mul(pitch as usize).ok_or(Error::InvalidRange)?;
        bytes[start..start + tight].copy_from_slice(source);
    }
    Ok(ForwardTextureInput {
        bytes,
        extent: data.extent,
        bytes_per_row: pitch,
        encoding: data.encoding,
        mip_levels: data.mip_levels,
    })
}
