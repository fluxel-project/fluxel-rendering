//! CPU staging for immutable material texture uploads.

use fluxel_rhi::api::Error;

use crate::{
    preparation::ResolvedForwardTexture,
    residency::{
        ForwardAuthoredRgba16FloatTextureInput, ForwardAuthoredTextureMipInput,
        ForwardResidentTextureInput, ForwardTextureInput,
    },
};

pub(in crate::renderer) fn stage(
    texture: &ResolvedForwardTexture,
    max_buffer_bytes: u64,
) -> Result<ForwardResidentTextureInput, Error> {
    match texture {
        ResolvedForwardTexture::Asset(texture) => {
            let data = texture.value();
            let (bytes, bytes_per_row) = stage_rows(data.extent, 4, &data.rgba8, max_buffer_bytes)?;
            Ok(ForwardResidentTextureInput::Generated(
                ForwardTextureInput {
                    bytes,
                    extent: data.extent,
                    bytes_per_row,
                    encoding: data.encoding,
                    mip_levels: data.mip_levels,
                },
            ))
        }
        ResolvedForwardTexture::RefractionRadiance(texture) => {
            let mut remaining = max_buffer_bytes;
            let mut mips = Vec::with_capacity(texture.value().mips().len());
            for mip in texture.value().mips() {
                let (bytes, bytes_per_row) = stage_rows(mip.extent, 8, &mip.rgba16f_le, remaining)?;
                remaining -= bytes.len() as u64;
                mips.push(ForwardAuthoredTextureMipInput {
                    bytes,
                    extent: mip.extent,
                    bytes_per_row,
                });
            }
            Ok(ForwardResidentTextureInput::AuthoredRgba16Float(
                ForwardAuthoredRgba16FloatTextureInput { mips },
            ))
        }
        ResolvedForwardTexture::CaptureSlot(_) => Err(Error::InvalidState),
    }
}

fn stage_rows(
    extent: [u32; 2],
    bytes_per_texel: u64,
    source: &[u8],
    max_buffer_bytes: u64,
) -> Result<(Vec<u8>, u32), Error> {
    let row = u64::from(extent[0])
        .checked_mul(bytes_per_texel)
        .ok_or(Error::InvalidRange)?;
    let tight_size = row
        .checked_mul(u64::from(extent[1]))
        .ok_or(Error::InvalidRange)?;
    if tight_size == 0 || tight_size != source.len() as u64 {
        return Err(Error::InvalidRange);
    }
    let pitch = row.checked_add(255).ok_or(Error::InvalidRange)? / 256 * 256;
    let pitch = u32::try_from(pitch).map_err(|_| Error::InvalidRange)?;
    let size = u64::from(pitch)
        .checked_mul(u64::from(extent[1]))
        .ok_or(Error::InvalidRange)?;
    if size > max_buffer_bytes {
        return Err(Error::InvalidRange);
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(usize::try_from(size).map_err(|_| Error::InvalidRange)?)
        .map_err(|_| Error::OutOfMemory)?;
    bytes.resize(usize::try_from(size).map_err(|_| Error::InvalidRange)?, 0);
    let tight = usize::try_from(row).map_err(|_| Error::InvalidRange)?;
    for (dest, source) in bytes
        .chunks_exact_mut(pitch as usize)
        .zip(source.chunks_exact(tight))
    {
        dest[..tight].copy_from_slice(source);
    }
    Ok((bytes, pitch))
}
