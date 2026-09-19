//! Private staging allocations, mappings, and texture transfer lowering.
//!
//! Staging buffers stay retained by the submission until completion. This module
//! contains no public upload policy; `upload` owns production stage reporting.

use super::*;

pub(super) fn copy_buffer_to_texture(
    encoder: &mut CopyEncoder,
    source: &OwnedBuffer,
    destination: &OwnedTexture,
    pitch: u32,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let copy = wgpu_hal::BufferTextureCopy {
        buffer_layout: wgt::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(pitch),
            rows_per_image: Some(height),
        },
        texture_base: copy_base(0, [0, 0, 0]),
        size: wgpu_hal::CopyExtent {
            width,
            height,
            depth: 1,
        },
    };
    match (
        encoder.native.as_mut().expect("recording encoder"),
        source.native.as_ref().expect("live buffer"),
        destination.native.as_ref().expect("live texture"),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeEncoder::Dx12(encoder),
            NativeBuffer::Dx12(source),
            NativeTexture::Dx12(destination),
        ) => {
            // SAFETY: the staging and destination belong to this encoder's
            // device, their CopySource/CopyDestination states were transitioned
            // immediately before this call, and checked padded pitch/range cover
            // exactly the closed D2 upload extent. Submission retains both leases.
            unsafe {
                encoder.copy_buffer_to_texture(source, destination, core::iter::once(copy));
            }
        }
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(encoder),
            NativeBuffer::Vulkan(source),
            NativeTexture::Vulkan(destination),
        ) => {
            // SAFETY: the staging and destination belong to this encoder's
            // device, their CopySource/CopyDestination states were transitioned
            // immediately before this call, and checked padded pitch/range cover
            // exactly the closed D2 upload extent. Submission retains both leases.
            unsafe {
                encoder.copy_buffer_to_texture(source, destination, core::iter::once(copy));
            }
        }
        _ => return Err("buffer-to-texture backend mismatch".into()),
    }
    Ok(())
}

#[cfg(any(test, feature = "test-support"))]
pub(super) fn copy_texture_to_buffer(
    encoder: &mut CopyEncoder,
    source: &OwnedTexture,
    destination: &OwnedBuffer,
    pitch: u32,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let copy = wgpu_hal::BufferTextureCopy {
        buffer_layout: wgt::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(pitch),
            rows_per_image: Some(height),
        },
        texture_base: copy_base(0, [0, 0, 0]),
        size: wgpu_hal::CopyExtent {
            width,
            height,
            depth: 1,
        },
    };
    match (
        encoder.native.as_mut().expect("recording encoder"),
        source.native.as_ref().expect("live texture"),
        destination.native.as_ref().expect("live buffer"),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeEncoder::Dx12(encoder),
            NativeTexture::Dx12(source),
            NativeBuffer::Dx12(destination),
        ) => {
            // SAFETY: test-only observation uses the exported incoming state
            // for the source transition, validated CopySource usage, and a
            // checked padded staging range retained through completion.
            unsafe {
                encoder.copy_texture_to_buffer(
                    source,
                    wgt::TextureUses::COPY_SRC,
                    destination,
                    core::iter::once(copy),
                );
            }
        }
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(encoder),
            NativeTexture::Vulkan(source),
            NativeBuffer::Vulkan(destination),
        ) => {
            // SAFETY: test-only observation uses the exported incoming state
            // for the source transition, validated CopySource usage, and a
            // checked padded staging range retained through completion.
            unsafe {
                encoder.copy_texture_to_buffer(
                    source,
                    wgt::TextureUses::COPY_SRC,
                    destination,
                    core::iter::once(copy),
                );
            }
        }
        _ => return Err("texture-to-buffer backend mismatch".into()),
    }
    Ok(())
}

#[cfg(any(test, feature = "test-support"))]
pub(super) fn clear_buffer(
    encoder: &mut CopyEncoder,
    buffer: &OwnedBuffer,
    size: u64,
) -> Result<(), String> {
    match (
        encoder.native.as_mut().expect("recording encoder"),
        buffer.native.as_ref().expect("live buffer"),
    ) {
        #[cfg(feature = "dx12")]
        (NativeEncoder::Dx12(encoder), NativeBuffer::Dx12(buffer)) => unsafe {
            // SAFETY: this test-only range is checked against the live staging
            // buffer size by its caller; encoder and buffer share one device and
            // remain retained until completion.
            encoder.clear_buffer(buffer, 0..size);
        },
        #[cfg(feature = "vulkan")]
        (NativeEncoder::Vulkan(encoder), NativeBuffer::Vulkan(buffer)) => unsafe {
            // SAFETY: the same checked range, same-device, and retained-lifetime
            // proof as the DX12 branch applies.
            encoder.clear_buffer(buffer, 0..size);
        },
        _ => return Err("clear buffer backend mismatch".into()),
    }
    Ok(())
}

pub(super) fn create_staging_buffer(
    owner: &Arc<OpenedDevice>,
    size: u64,
    usage: wgt::BufferUses,
) -> Result<OwnedBuffer, String> {
    let desc = wgpu_hal::BufferDescriptor {
        label: Some("fluxel immutable upload staging"),
        size,
        usage,
        memory_flags: wgpu_hal::MemoryFlags::PREFER_COHERENT,
    };
    let native = match &owner.native {
        #[cfg(feature = "dx12")]
        NativeDevice::Dx12 { device, .. } => NativeBuffer::Dx12(
            // SAFETY: caller validates a non-zero bounded upload size and uses
            // the fixed map-write/copy-source staging recipe.
            unsafe { device.create_buffer(&desc) }.map_err(|e| e.to_string())?,
        ),
        #[cfg(feature = "vulkan")]
        NativeDevice::Vulkan { device, .. } => NativeBuffer::Vulkan(
            // SAFETY: same validated size and fixed staging recipe as DX12.
            unsafe { device.create_buffer(&desc) }.map_err(|e| e.to_string())?,
        ),
    };
    Ok(OwnedBuffer {
        native: Some(native),
        owner: Arc::clone(owner),
        size,
        allowed_usage: buffer_usage_from_native(usage),
    })
}

pub(super) fn with_mapped_write(buffer: &OwnedBuffer, bytes: &[u8]) -> Result<(), String> {
    let range = 0..bytes.len() as u64;
    match (
        &buffer.owner.native,
        buffer.native.as_ref().expect("live staging"),
    ) {
        #[cfg(feature = "dx12")]
        (NativeDevice::Dx12 { device, .. }, NativeBuffer::Dx12(native)) => unsafe {
            // SAFETY: `range` fits this MAP_WRITE staging buffer and no GPU work
            // can observe it while mapped. The mapping covers `bytes.len()` bytes;
            // non-coherent writes are flushed and the buffer is unmapped before use.
            let mapping = device
                .map_buffer(native, range.clone())
                .map_err(|e| e.to_string())?;
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), mapping.ptr.as_ptr(), bytes.len());
            if !mapping.is_coherent {
                device.flush_mapped_ranges(native, core::iter::once(range));
            }
            device.unmap_buffer(native);
        },
        #[cfg(feature = "vulkan")]
        (NativeDevice::Vulkan { device, .. }, NativeBuffer::Vulkan(native)) => unsafe {
            // SAFETY: the same exclusive mapping, bounded copy, flush, and unmap
            // proof as the DX12 branch applies.
            let mapping = device
                .map_buffer(native, range.clone())
                .map_err(|e| e.to_string())?;
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), mapping.ptr.as_ptr(), bytes.len());
            if !mapping.is_coherent {
                device.flush_mapped_ranges(native, core::iter::once(range));
            }
            device.unmap_buffer(native);
        },
        _ => return Err("staging backend mismatch".into()),
    }
    Ok(())
}

#[cfg(any(test, feature = "test-support"))]
pub(super) fn with_mapped_read(buffer: &OwnedBuffer, size: u64) -> Result<Vec<u8>, String> {
    let range = 0..size;
    let mut result = vec![0; size as usize];
    match (
        &buffer.owner.native,
        buffer.native.as_ref().expect("live staging"),
    ) {
        #[cfg(feature = "dx12")]
        (NativeDevice::Dx12 { device, .. }, NativeBuffer::Dx12(native)) => unsafe {
            // SAFETY: completion is observed before this test-only MAP_READ path;
            // `range` fits the live buffer, non-coherent data is invalidated before
            // the bounded CPU copy, and unmap occurs before the buffer can be reused.
            let mapping = device
                .map_buffer(native, range.clone())
                .map_err(|e| e.to_string())?;
            if !mapping.is_coherent {
                device.invalidate_mapped_ranges(native, core::iter::once(range));
            }
            core::ptr::copy_nonoverlapping(mapping.ptr.as_ptr(), result.as_mut_ptr(), result.len());
            device.unmap_buffer(native);
        },
        #[cfg(feature = "vulkan")]
        (NativeDevice::Vulkan { device, .. }, NativeBuffer::Vulkan(native)) => unsafe {
            // SAFETY: the same completed-work, bounded mapping, invalidate, and
            // unmap proof as the DX12 branch applies.
            let mapping = device
                .map_buffer(native, range.clone())
                .map_err(|e| e.to_string())?;
            if !mapping.is_coherent {
                device.invalidate_mapped_ranges(native, core::iter::once(range));
            }
            core::ptr::copy_nonoverlapping(mapping.ptr.as_ptr(), result.as_mut_ptr(), result.len());
            device.unmap_buffer(native);
        },
        _ => return Err("staging backend mismatch".into()),
    }
    Ok(result)
}

pub(crate) fn copy_base(mip_level: u32, origin: [u32; 3]) -> wgpu_hal::TextureCopyBase {
    wgpu_hal::TextureCopyBase {
        mip_level,
        array_layer: 0,
        origin: wgt::Origin3d {
            x: origin[0],
            y: origin[1],
            z: origin[2],
        },
        aspect: wgpu_hal::FormatAspects::COLOR,
    }
}

fn canonical_texture_range(descriptor: TextureDesc, range: TextureRange) -> TextureRange {
    match range {
        TextureRange::Whole => TextureRange::Subresources {
            base_mip_level: 0,
            mip_level_count: descriptor.mip_levels,
            base_array_layer: 0,
            array_layer_count: descriptor.array_layers,
            aspect: match descriptor.format {
                TextureFormat::Depth32Float => TextureAspect::Depth,
                TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => TextureAspect::Color,
                _ => TextureAspect::Color,
            },
        },
        range => range,
    }
}

pub(crate) fn texture_subresources(
    texture: &OwnedTexture,
    descriptor: TextureDesc,
    range: TextureRange,
) -> Result<Vec<(TextureStateKey, wgt::ImageSubresourceRange)>, String> {
    let canonical = canonical_texture_range(descriptor, range);
    let TextureRange::Subresources {
        base_mip_level,
        mip_level_count,
        base_array_layer,
        array_layer_count,
        aspect,
    } = canonical
    else {
        unreachable!("whole ranges are canonicalized")
    };
    let mip_end = base_mip_level
        .checked_add(mip_level_count)
        .ok_or_else(|| "texture mip range overflows".to_owned())?;
    let layer_end = base_array_layer
        .checked_add(array_layer_count)
        .ok_or_else(|| "texture layer range overflows".to_owned())?;
    let compatible_aspect = matches!(
        (descriptor.format, aspect),
        (
            TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb,
            TextureAspect::Color
        ) | (TextureFormat::Depth32Float, TextureAspect::Depth)
    );
    if mip_level_count == 0
        || array_layer_count == 0
        || mip_end > descriptor.mip_levels
        || layer_end > descriptor.array_layers
        || !compatible_aspect
    {
        return Err("invalid texture subresource range".into());
    }
    let allocation = core::ptr::from_ref(texture).addr();
    let mut output = Vec::with_capacity(
        usize::try_from(u64::from(mip_level_count) * u64::from(array_layer_count))
            .map_err(|_| "texture subresource count is too large".to_owned())?,
    );
    for mip_level in base_mip_level..mip_end {
        for array_layer in base_array_layer..layer_end {
            let single = TextureRange::Subresources {
                base_mip_level: mip_level,
                mip_level_count: 1,
                base_array_layer: array_layer,
                array_layer_count: 1,
                aspect,
            };
            output.push((
                TextureStateKey {
                    allocation,
                    mip_level,
                    array_layer,
                    aspect,
                },
                texture_range(descriptor, single)?,
            ));
        }
    }
    Ok(output)
}

fn texture_range(
    descriptor: TextureDesc,
    range: TextureRange,
) -> Result<wgt::ImageSubresourceRange, String> {
    Ok(match range {
        TextureRange::Whole => wgt::ImageSubresourceRange {
            aspect: wgt::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(descriptor.mip_levels),
            base_array_layer: 0,
            array_layer_count: Some(descriptor.array_layers),
        },
        TextureRange::Subresources {
            base_mip_level,
            mip_level_count,
            base_array_layer,
            array_layer_count,
            aspect,
        } => wgt::ImageSubresourceRange {
            aspect: match aspect {
                TextureAspect::Color => wgt::TextureAspect::All,
                TextureAspect::Depth => wgt::TextureAspect::DepthOnly,
                TextureAspect::Stencil => wgt::TextureAspect::StencilOnly,
                _ => return Err("unsupported texture aspect".into()),
            },
            base_mip_level,
            mip_level_count: Some(mip_level_count),
            base_array_layer,
            array_layer_count: Some(array_layer_count),
        },
    })
}
