//! CPU staging for immutable material texture uploads.

use fluxel_rhi::api::Error;

use crate::{
    preparation::{ColorTextureArrayData, ResolvedForwardTexture},
    residency::{
        ForwardArrayTextureInput, ForwardAuthoredRgba16FloatTextureInput,
        ForwardResidentTextureInput, ForwardTextureCopyInput, ForwardTextureInput,
        LayeredTextureView,
    },
};

pub(in crate::renderer) fn stage(
    texture: &ResolvedForwardTexture,
    max_buffer_bytes: u64,
) -> Result<ForwardResidentTextureInput, Error> {
    match texture {
        ResolvedForwardTexture::Asset(asset) | ResolvedForwardTexture::StorageAsset(asset) => {
            let data = asset.value();
            let extent = [data.extent[0], data.extent[1], 1];
            let (bytes, bytes_per_row) = stage_rows(extent, 4, &data.rgba8, max_buffer_bytes)?;
            if matches!(texture, ResolvedForwardTexture::StorageAsset(_)) {
                return Ok(ForwardResidentTextureInput::Storage(
                    ForwardTextureCopyInput {
                        bytes,
                        extent,
                        bytes_per_row,
                    },
                ));
            }
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
        ResolvedForwardTexture::ArrayAsset(texture) => stage_array(
            texture.value(),
            None,
            LayeredTextureView::Array,
            max_buffer_bytes,
        )
        .map(ForwardResidentTextureInput::Array),
        ResolvedForwardTexture::GeneratedArray { asset, mip_levels } => stage_array(
            asset.value(),
            Some(*mip_levels),
            LayeredTextureView::Array,
            max_buffer_bytes,
        )
        .map(ForwardResidentTextureInput::Array),
        ResolvedForwardTexture::CubeAsset {
            asset,
            generated_mips,
        } => stage_array(
            asset.value(),
            *generated_mips,
            LayeredTextureView::Cube,
            max_buffer_bytes,
        )
        .map(ForwardResidentTextureInput::Array),
        ResolvedForwardTexture::VolumeAsset {
            asset,
            generated_mips,
        } => {
            let data = asset.value();
            let payloads = data.mip_payloads().ok_or(Error::InvalidRange)?;
            let mip_count = data.mip_levels as usize;
            let mut mips = Vec::with_capacity(mip_count);
            let mut remaining = max_buffer_bytes;
            for (extent, source) in payloads {
                let (bytes, bytes_per_row) = stage_rows(extent, 4, source, remaining)?;
                remaining =
                    remaining_after_copy(remaining, bytes.len(), mips.len() + 1 < mip_count)?;
                mips.push(ForwardTextureCopyInput {
                    bytes,
                    extent,
                    bytes_per_row,
                });
            }
            Ok(ForwardResidentTextureInput::Volume {
                mips,
                encoding: data.encoding,
                generated_mips: *generated_mips,
            })
        }
        ResolvedForwardTexture::DepthAsset(texture) => {
            let data = texture.value();
            stage_depth_layer(data.extent, &data.depth32f, max_buffer_bytes)
                .map(ForwardResidentTextureInput::Depth)
        }
        ResolvedForwardTexture::DepthArrayAsset(texture) => {
            let data = texture.value();
            let [width, height] = data.extent;
            let layer_texels = (width as usize)
                .checked_mul(height as usize)
                .filter(|&texels| texels != 0)
                .ok_or(Error::InvalidRange)?;
            let layers = data.layers as usize;
            if layers == 0 || layer_texels.checked_mul(layers) != Some(data.depth32f.len()) {
                return Err(Error::InvalidRange);
            }
            let mut copies = Vec::new();
            copies
                .try_reserve_exact(layers)
                .map_err(|_| Error::OutOfMemory)?;
            let mut remaining = max_buffer_bytes;
            for layer in data.depth32f.chunks_exact(layer_texels) {
                let copy = stage_depth_layer(data.extent, layer, remaining)?;
                remaining =
                    remaining_after_copy(remaining, copy.bytes.len(), copies.len() + 1 < layers)?;
                copies.push(copy);
            }
            Ok(ForwardResidentTextureInput::DepthArray(copies))
        }
        ResolvedForwardTexture::RefractionRadiance(texture) => {
            let mut remaining = max_buffer_bytes;
            let mut mips = Vec::with_capacity(texture.value().mips().len());
            for mip in texture.value().mips() {
                let extent = [mip.extent[0], mip.extent[1], 1];
                let (bytes, bytes_per_row) = stage_rows(extent, 8, &mip.rgba16f_le, remaining)?;
                remaining -= bytes.len() as u64;
                mips.push(ForwardTextureCopyInput {
                    bytes,
                    extent,
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

fn stage_depth_layer(
    [width, height]: [u32; 2],
    depth: &[f32],
    max_buffer_bytes: u64,
) -> Result<ForwardTextureCopyInput, Error> {
    let extent = [width, height, 1];
    let source_size = u64::try_from(depth.len())
        .ok()
        .and_then(|count| count.checked_mul(4))
        .ok_or(Error::InvalidRange)?;
    let (mut bytes, bytes_per_row, _) =
        allocate_staged_rows(extent, 4, source_size, max_buffer_bytes)?;
    for (dest, source) in bytes
        .chunks_exact_mut(bytes_per_row as usize)
        .zip(depth.chunks_exact(width as usize))
    {
        for (texel, depth) in dest.chunks_exact_mut(4).zip(source) {
            texel.copy_from_slice(&depth.to_le_bytes());
        }
    }
    Ok(ForwardTextureCopyInput {
        bytes,
        extent,
        bytes_per_row,
    })
}

fn stage_array(
    data: &ColorTextureArrayData,
    generated_mips: Option<u32>,
    view: LayeredTextureView,
    max_buffer_bytes: u64,
) -> Result<ForwardArrayTextureInput, Error> {
    if generated_mips.is_some() && data.mip_levels != 1 {
        return Err(Error::InvalidRange);
    }
    let mip_payloads = data.mip_payloads().ok_or(Error::InvalidRange)?;
    let layers = usize::try_from(data.layers).map_err(|_| Error::InvalidRange)?;
    let mips = usize::try_from(data.mip_levels).map_err(|_| Error::InvalidRange)?;
    let image_count = mips.checked_mul(layers).ok_or(Error::InvalidRange)?;
    let mut copies = Vec::new();
    copies
        .try_reserve_exact(image_count)
        .map_err(|_| Error::OutOfMemory)?;
    let mut remaining = max_buffer_bytes;
    for ([width, height], payload) in mip_payloads {
        let extent = [width, height, 1];
        for source in payload.chunks_exact(payload.len() / layers) {
            let (bytes, bytes_per_row) = stage_rows(extent, 4, source, remaining)?;
            remaining =
                remaining_after_copy(remaining, bytes.len(), copies.len() + 1 < image_count)?;
            copies.push(ForwardTextureCopyInput {
                bytes,
                extent,
                bytes_per_row,
            });
        }
    }
    Ok(ForwardArrayTextureInput {
        copies,
        layers: data.layers,
        mip_levels: generated_mips.unwrap_or(data.mip_levels),
        generate_mips: generated_mips.is_some(),
        view,
        encoding: data.encoding,
    })
}

// Residency aligns each copy start to 512 bytes within the staging buffer.
fn remaining_after_copy(remaining: u64, size: usize, has_next: bool) -> Result<u64, Error> {
    let size = u64::try_from(size).map_err(|_| Error::InvalidRange)?;
    let span = if has_next {
        size.checked_add((512 - size % 512) % 512)
            .ok_or(Error::InvalidRange)?
    } else {
        size
    };
    remaining.checked_sub(span).ok_or(Error::InvalidRange)
}

fn stage_rows(
    extent: [u32; 3],
    bytes_per_texel: u64,
    source: &[u8],
    max_buffer_bytes: u64,
) -> Result<(Vec<u8>, u32), Error> {
    let (mut bytes, pitch, tight) = allocate_staged_rows(
        extent,
        bytes_per_texel,
        source.len() as u64,
        max_buffer_bytes,
    )?;
    for (dest, source) in bytes
        .chunks_exact_mut(pitch as usize)
        .zip(source.chunks_exact(tight))
    {
        dest[..tight].copy_from_slice(source);
    }
    Ok((bytes, pitch))
}

fn allocate_staged_rows(
    extent: [u32; 3],
    bytes_per_texel: u64,
    source_size: u64,
    max_buffer_bytes: u64,
) -> Result<(Vec<u8>, u32, usize), Error> {
    let row = u64::from(extent[0])
        .checked_mul(bytes_per_texel)
        .ok_or(Error::InvalidRange)?;
    let rows = u64::from(extent[1]) * u64::from(extent[2]);
    let tight_size = row.checked_mul(rows).ok_or(Error::InvalidRange)?;
    if tight_size == 0 || tight_size != source_size {
        return Err(Error::InvalidRange);
    }
    let pitch = row.checked_add(255).ok_or(Error::InvalidRange)? / 256 * 256;
    let pitch = u32::try_from(pitch).map_err(|_| Error::InvalidRange)?;
    let size = u64::from(pitch)
        .checked_mul(rows)
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
    Ok((bytes, pitch, tight))
}
