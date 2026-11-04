//! Completion-gated reuse helpers for Forward frame allocations.
//!
//! The caller must observe both the completion covering the final GPU use and
//! the associated readback/map result before placing these buffers in a pool.
//! These helpers only normalize owner state and inspect recorded allocation
//! facts; they cannot establish either completion fact themselves.

use fluxel_rhi::{
    api::{
        AccessMask, BufferUsage, BufferUses, DeviceId, ImageLayout, MemoryClass, StageMask,
        TextureDimension, TextureFormat, TextureUsage, TextureUses,
    },
    backend::ResourceBackend,
};

use crate::residency::ForwardFrameBuffers;

/// Records the outgoing owner states expected when a completed frame allocation
/// is reused as the next Forward frame's imported resource set.
pub(crate) fn recycle<B: ResourceBackend>(
    mut buffers: ForwardFrameBuffers<B>,
    had_draws: bool,
) -> ForwardFrameBuffers<B> {
    buffers.upload_incoming = copy_read_uses();
    buffers.uniform_incoming = if had_draws {
        uniform_read_uses()
    } else {
        copy_write_uses()
    };
    buffers.target_incoming = texture_copy_read_uses();
    buffers.target_layout = ImageLayout::CopySrc;
    buffers.readback_incoming = copy_write_uses();
    buffers
}

/// Checks actual owner descriptors and handle provenance before a completed
/// allocation is reused. No Graph declaration is used as an allocation fact.
///
/// This function performs no backend calls and does not prove that GPU work or
/// CPU readback has completed; the pool coordinator establishes those facts.
pub(crate) fn compatible<B: ResourceBackend>(
    buffers: &ForwardFrameBuffers<B>,
    device: DeviceId,
    extent: [u32; 2],
    uniform_bytes: u64,
    row_pitch: u32,
) -> bool {
    let Some(readback_bytes) = u64::from(row_pitch).checked_mul(u64::from(extent[1])) else {
        return false;
    };
    if extent.contains(&0)
        || uniform_bytes == 0
        || row_pitch == 0
        || !row_pitch.is_multiple_of(256)
        || u64::from(row_pitch) < u64::from(extent[0]) * 4
        || buffers.upload.device() != device
        || buffers.uniform.device() != device
        || buffers.target.device() != device
        || buffers.target_view.device() != device
        || buffers.readback.device() != device
    {
        return false;
    }

    let upload = &buffers.upload_desc;
    let uniform = &buffers.uniform_desc;
    let readback = &buffers.readback_desc;
    let target = &buffers.target_desc;
    upload.size >= uniform_bytes
        && upload.memory == MemoryClass::Upload
        && upload.usage.contains(BufferUsage::COPY_SRC)
        && upload.usage.contains(BufferUsage::MAP_WRITE)
        && uniform.size >= uniform_bytes
        && uniform.memory == MemoryClass::Device
        && uniform.usage.contains(BufferUsage::COPY_DST)
        && uniform.usage.contains(BufferUsage::UNIFORM)
        && readback.size >= readback_bytes
        && readback.memory == MemoryClass::Readback
        && readback.usage.contains(BufferUsage::COPY_DST)
        && readback.usage.contains(BufferUsage::MAP_READ)
        && target.dimension == TextureDimension::D2
        && target.extent == [extent[0], extent[1], 1]
        && target.array_layers == 1
        && target.format == TextureFormat::Rgba8Unorm
        && target.mip_levels == 1
        && target.samples == 1
        && target.usage.contains(TextureUsage::RENDER_ATTACHMENT)
        && target.usage.contains(TextureUsage::COPY_SRC)
}

fn copy_read_uses() -> BufferUses {
    BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    }
}

fn copy_write_uses() -> BufferUses {
    BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    }
}

fn uniform_read_uses() -> BufferUses {
    BufferUses {
        stages: StageMask::VERTEX | StageMask::FRAGMENT,
        accesses: AccessMask::UNIFORM_READ,
    }
}

fn texture_copy_read_uses() -> TextureUses {
    TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    }
}
