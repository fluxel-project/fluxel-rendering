//! Native copy command recording.
use super::*;
pub(crate) fn begin_copy_encoder(owner: &Arc<OpenedDevice>) -> Result<CopyEncoder, String> {
    let native = match &owner.native {
        #[cfg(feature = "dx12")]
        NativeDevice::Dx12 { device, queue, .. } => {
            let desc = wgpu_hal::CommandEncoderDescriptor {
                label: Some("fluxel copy graph"),
                queue,
            };
            // SAFETY: queue and device were opened together and remain retained.
            let mut encoder =
                unsafe { device.create_command_encoder(&desc) }.map_err(|e| e.to_string())?;
            // SAFETY: a freshly created encoder is closed.
            unsafe { encoder.begin_encoding(Some("fluxel copy graph")) }
                .map_err(|e| e.to_string())?;
            NativeEncoder::Dx12(encoder)
        }
        #[cfg(feature = "vulkan")]
        NativeDevice::Vulkan { device, queue, .. } => {
            let desc = wgpu_hal::CommandEncoderDescriptor {
                label: Some("fluxel copy graph"),
                queue,
            };
            // SAFETY: queue and device share the retained adapter/device lineage.
            let mut encoder =
                unsafe { device.create_command_encoder(&desc) }.map_err(|e| e.to_string())?;
            // SAFETY: a freshly created encoder is closed.
            unsafe { encoder.begin_encoding(Some("fluxel copy graph")) }
                .map_err(|e| e.to_string())?;
            NativeEncoder::Vulkan(encoder)
        }
    };
    Ok(CopyEncoder {
        native: Some(native),
        owner: Arc::clone(owner),
        buffer_states: HashMap::new(),
        texture_states: HashMap::new(),
        active_render_view: None,
        active_raster_pipeline: None,
        render_views: Vec::new(),
    })
}

pub(crate) fn transition_buffer(
    encoder: &mut CopyEncoder,
    buffer: &OwnedBuffer,
    before: ResourceAccessState,
    after: ResourceAccessState,
) -> Result<(), String> {
    let key = core::ptr::from_ref(buffer).addr();
    let effective_before = match encoder.buffer_states.get(&key).copied() {
        // A range-local plan transition can legitimately name the original
        // state after another range has already changed the whole native
        // buffer. Both ranges now require the same final state; lower this as
        // a same-state barrier, which retains the memory dependency without
        // lying to DX12 about its actual resource state.
        Some(current) if current == after && before != after => current,
        Some(current) if current != before => {
            return Err(format!(
                "planned buffer transition {before:?}->{after:?} conflicts with encoder state {current:?}"
            ));
        }
        _ => before,
    };
    let from = buffer_state(effective_before)?;
    let to = buffer_state(after)?;
    match (
        encoder.native.as_mut().expect("recording encoder"),
        buffer.native.as_ref().expect("live buffer"),
    ) {
        #[cfg(feature = "dx12")]
        (NativeEncoder::Dx12(encoder), NativeBuffer::Dx12(buffer)) => unsafe {
            // SAFETY: the safe execution layer checked device identity and
            // portable states; this live buffer and recording encoder share
            // their retained owner, and encoder state tracks the actual `from`.
            encoder.transition_buffers(core::iter::once(wgpu_hal::BufferBarrier {
                buffer,
                usage: wgpu_hal::StateTransition { from, to },
            }));
        },
        #[cfg(feature = "vulkan")]
        (NativeEncoder::Vulkan(encoder), NativeBuffer::Vulkan(buffer)) => unsafe {
            // SAFETY: same-device ownership, live handles, tracked old state,
            // and retained submission lifetime are identical to the DX12 case.
            encoder.transition_buffers(core::iter::once(wgpu_hal::BufferBarrier {
                buffer,
                usage: wgpu_hal::StateTransition { from, to },
            }));
        },
        _ => return Err("buffer and encoder backend mismatch".into()),
    }
    encoder.buffer_states.insert(key, after);
    Ok(())
}

pub(crate) fn transition_texture(
    encoder: &mut CopyEncoder,
    texture: &OwnedTexture,
    descriptor: TextureDesc,
    range: TextureRange,
    before: ResourceAccessState,
    after: ResourceAccessState,
) -> Result<(), String> {
    let selected = texture_subresources(texture, descriptor, range)?;
    let mut prepared = Vec::with_capacity(selected.len());
    for (key, range) in selected {
        let effective_before = match encoder.texture_states.get(&key).copied() {
            Some(current) if current == after && before != after => current,
            Some(current) if current != before => {
                return Err(format!(
                    "planned texture transition {before:?}->{after:?} conflicts with encoder state {current:?}"
                ));
            }
            _ => before,
        };
        prepared.push((key, texture_state(effective_before)?, range));
    }
    let to = texture_state(after)?;
    match (
        encoder.native.as_mut().expect("recording encoder"),
        texture.native.as_ref().expect("live texture"),
    ) {
        #[cfg(feature = "dx12")]
        (NativeEncoder::Dx12(encoder), NativeTexture::Dx12(texture)) => unsafe {
            // SAFETY: every expanded barrier names a validated live subresource
            // on this same-device texture and uses the tracked actual old state.
            encoder.transition_textures(prepared.iter().map(|(_, from, range)| {
                wgpu_hal::TextureBarrier {
                    texture,
                    range: *range,
                    usage: wgpu_hal::StateTransition { from: *from, to },
                }
            }));
        },
        #[cfg(feature = "vulkan")]
        (NativeEncoder::Vulkan(encoder), NativeTexture::Vulkan(texture)) => unsafe {
            // Vulkan permits an acquired swapchain image to enter from
            // UNDEFINED when its previous contents are discarded. Surface
            // imports are the only resources whose portable incoming state is
            // Present, so lower that first boundary transition conservatively
            // from UNINITIALIZED. This is valid both on first acquisition and
            // after a prior present, and avoids asserting a stale native layout.
            // SAFETY: same validated subresource, same-device, and retained-
            // lifetime proof as DX12; UNINITIALIZED explicitly discards old
            // swapchain contents before the color attachment clear.
            encoder.transition_textures(prepared.iter().map(|(_, from, range)| {
                let from = if *from == wgt::TextureUses::PRESENT {
                    wgt::TextureUses::UNINITIALIZED
                } else {
                    *from
                };
                wgpu_hal::TextureBarrier {
                    texture,
                    range: *range,
                    usage: wgpu_hal::StateTransition { from, to },
                }
            }));
        },
        _ => return Err("texture and encoder backend mismatch".into()),
    }
    for (key, _, _) in prepared {
        encoder.texture_states.insert(key, after);
    }
    Ok(())
}

pub(crate) fn copy_buffer(
    encoder: &mut CopyEncoder,
    source: &OwnedBuffer,
    destination: &OwnedBuffer,
    region: BufferCopyRegion,
) -> Result<(), String> {
    validate_buffer_copy_alignment(region)?;
    let size =
        wgt::BufferSize::new(region.size).ok_or_else(|| "copy size must be non-zero".to_owned())?;
    let copy = wgpu_hal::BufferCopy {
        src_offset: region.source_offset,
        dst_offset: region.destination_offset,
        size,
    };
    match (
        encoder.native.as_mut().expect("recording encoder"),
        source.native.as_ref().expect("live buffer"),
        destination.native.as_ref().expect("live buffer"),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeEncoder::Dx12(encoder),
            NativeBuffer::Dx12(source),
            NativeBuffer::Dx12(destination),
        ) => unsafe {
            // SAFETY: the safe boundary validated non-overflowing aligned ranges
            // within both same-device buffers. Both handles remain leased through
            // submission, and this encoder is actively recording.
            encoder.copy_buffer_to_buffer(source, destination, core::iter::once(copy));
        },
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(encoder),
            NativeBuffer::Vulkan(source),
            NativeBuffer::Vulkan(destination),
        ) => unsafe {
            // SAFETY: the same validated ranges, ownership, recording-state, and
            // lease-retention proof as the DX12 branch applies.
            encoder.copy_buffer_to_buffer(source, destination, core::iter::once(copy));
        },
        _ => return Err("copy buffers and encoder backend mismatch".into()),
    }
    Ok(())
}

pub(crate) fn validate_buffer_copy_alignment(region: BufferCopyRegion) -> Result<(), String> {
    // `wgpu-hal` is an unsafe API and does not provide the safe WebGPU copy
    // contract for us. Preserve `COPY_BUFFER_ALIGNMENT` at this final native
    // boundary too: test-only upload/readback helpers may bypass RenderGraph.
    if region
        .source_offset
        .is_multiple_of(wgt::COPY_BUFFER_ALIGNMENT)
        && region
            .destination_offset
            .is_multiple_of(wgt::COPY_BUFFER_ALIGNMENT)
        && region.size.is_multiple_of(wgt::COPY_BUFFER_ALIGNMENT)
    {
        Ok(())
    } else {
        Err(format!(
            "buffer copy offsets and size must be aligned to COPY_BUFFER_ALIGNMENT ({})",
            wgt::COPY_BUFFER_ALIGNMENT
        ))
    }
}

pub(crate) fn copy_texture(
    encoder: &mut CopyEncoder,
    source: &OwnedTexture,
    destination: &OwnedTexture,
    descriptor: TextureDesc,
    region: TextureCopyRegion,
) -> Result<(), String> {
    let copy = wgpu_hal::TextureCopy {
        src_base: copy_base(region.source_mip_level, region.source_origin),
        dst_base: copy_base(region.destination_mip_level, region.destination_origin),
        size: wgpu_hal::CopyExtent {
            width: region.extent[0],
            height: region.extent[1],
            depth: region.extent[2],
        },
    };
    let source_usage = wgt::TextureUses::COPY_SRC;
    let _ = descriptor;
    match (
        encoder.native.as_mut().expect("recording encoder"),
        source.native.as_ref().expect("live texture"),
        destination.native.as_ref().expect("live texture"),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeEncoder::Dx12(encoder),
            NativeTexture::Dx12(source),
            NativeTexture::Dx12(destination),
        ) => unsafe {
            // SAFETY: descriptor/format/extent and every copied subresource were
            // validated before this boundary; source, destination, and encoder
            // share one retained device and outlive the submitted command buffer.
            encoder.copy_texture_to_texture(
                source,
                source_usage,
                destination,
                core::iter::once(copy),
            );
        },
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(encoder),
            NativeTexture::Vulkan(source),
            NativeTexture::Vulkan(destination),
        ) => unsafe {
            // SAFETY: the same validated texture-copy and retained-lifetime proof
            // as the DX12 branch applies.
            encoder.copy_texture_to_texture(
                source,
                source_usage,
                destination,
                core::iter::once(copy),
            );
        },
        _ => return Err("copy textures and encoder backend mismatch".into()),
    }
    Ok(())
}

pub(crate) fn finish_copy_encoder(mut encoder: CopyEncoder) -> Result<CopyCommandBuffer, String> {
    let native = encoder.native.take().expect("recording encoder");
    let finished = match native {
        #[cfg(feature = "dx12")]
        NativeEncoder::Dx12(mut native) => {
            // SAFETY: this encoder is recording and no prior error was retained.
            let command_buffer = unsafe { native.end_encoding() }.map_err(|e| e.to_string())?;
            NativeFinished::Dx12 {
                owner: Arc::clone(&encoder.owner),
                encoder: native,
                command_buffer,
                render_views: std::mem::take(&mut encoder.render_views),
            }
        }
        #[cfg(feature = "vulkan")]
        NativeEncoder::Vulkan(mut native) => {
            // SAFETY: this encoder is recording and no prior error was retained.
            let command_buffer = unsafe { native.end_encoding() }.map_err(|e| e.to_string())?;
            NativeFinished::Vulkan {
                owner: Arc::clone(&encoder.owner),
                encoder: native,
                command_buffer,
                render_views: std::mem::take(&mut encoder.render_views),
            }
        }
    };
    Ok(CopyCommandBuffer {
        native: Some(finished),
    })
}

pub(crate) fn submit_copy(
    buffer: CopyCommandBuffer,
    leases: Vec<ResourceLease>,
) -> Result<NativeCompletion, String> {
    submit_copy_with_staging(buffer, leases, Vec::new())
}
