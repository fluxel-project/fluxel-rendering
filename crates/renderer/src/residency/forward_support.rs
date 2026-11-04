//! Shared allocation support for Forward residency owners.
//!
//! These small helpers keep plan descriptions separate from concrete owner
//! allocation without widening the residency public surface.

use fluxel_rhi::{
    api::{BufferDesc, BufferUses, Device, Error, Result, Support, TextureDesc, TextureUses},
    backend::ResourceBackend,
};

use super::ForwardUniformUpload;

/// Handle-free facts for an offscreen Forward frame allocation.
pub(crate) struct ForwardFramePlan {
    pub(crate) upload_desc: BufferDesc,
    pub(crate) uniform_desc: BufferDesc,
    pub(crate) target_desc: TextureDesc,
    pub(crate) readback_desc: BufferDesc,
    pub(crate) uniform_upload: ForwardUniformUpload,
}

/// Handle-free facts for a direct-surface Forward frame allocation.
pub(crate) struct SurfaceForwardFramePlan {
    pub(crate) upload_desc: BufferDesc,
    pub(crate) uniform_desc: BufferDesc,
    pub(crate) uniform_upload: ForwardUniformUpload,
}

pub(super) fn require_buffer<B: ResourceBackend>(
    device: &Device<B>,
    desc: &BufferDesc,
) -> Result<()> {
    match device.buffer_support(desc) {
        Support::Supported(_) => Ok(()),
        Support::Unsupported(reason) => Err(Error::Unsupported(reason)),
    }
}

pub(super) fn require_texture<B: ResourceBackend>(
    device: &Device<B>,
    desc: &TextureDesc,
) -> Result<()> {
    match device.texture_support(desc) {
        Support::Supported(_) => Ok(()),
        Support::Unsupported(reason) => Err(Error::Unsupported(reason)),
    }
}

pub(super) fn empty_buffer_uses() -> BufferUses {
    BufferUses {
        stages: fluxel_rhi::api::StageMask::EMPTY,
        accesses: fluxel_rhi::api::AccessMask::EMPTY,
    }
}

pub(super) fn empty_texture_uses() -> TextureUses {
    TextureUses {
        stages: fluxel_rhi::api::StageMask::EMPTY,
        accesses: fluxel_rhi::api::AccessMask::EMPTY,
    }
}

pub(super) fn pad_to(bytes: &mut Vec<u8>, alignment: usize) -> Result<()> {
    let remainder = bytes.len() % alignment;
    if remainder != 0 {
        let extra = alignment - remainder;
        bytes.try_reserve(extra).map_err(|_| Error::OutOfMemory)?;
        bytes.resize(bytes.len() + extra, 0);
    }
    Ok(())
}
