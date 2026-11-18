//! Actual Deferred attachments borrowed by the Forward transparent overlay.

use fluxel_rhi::{
    api::{DeviceId, Error, TextureDimension, TextureFormat, TextureUsage},
    backend::ResourceBackend,
};

/// Renderer-private ownership cloned from a Deferred frame for one same-Graph
/// Forward overlay.  The clone keeps the native handles alive through the
/// overlay completion; it never changes their saved descriptor or state.
pub(in crate::renderer) struct BorrowedForwardAttachments<B: ResourceBackend> {
    pub(in crate::renderer) color: crate::residency::ForwardTexture<B>,
    pub(in crate::renderer) depth: crate::residency::ForwardTexture<B>,
}

impl<B: ResourceBackend> BorrowedForwardAttachments<B> {
    pub(in crate::renderer) fn new(
        color: crate::residency::ForwardTexture<B>,
        depth: crate::residency::ForwardTexture<B>,
    ) -> Self {
        Self { color, depth }
    }

    pub(in crate::renderer) fn validate(
        &self,
        device: DeviceId,
        extent: [u32; 2],
    ) -> Result<(), Error> {
        validate_color(&self.color, device, extent)?;
        validate_depth(&self.depth, device, extent)?;
        if self.color.instance == self.depth.instance {
            return Err(Error::InvalidState);
        }
        Ok(())
    }
}

fn validate_color<B: ResourceBackend>(
    owner: &crate::residency::ForwardTexture<B>,
    device: DeviceId,
    extent: [u32; 2],
) -> Result<(), Error> {
    if owner.device != device
        || owner.texture.device() != device
        || owner.view.device() != device
        || owner.desc.dimension != TextureDimension::D2
        || owner.desc.extent != [extent[0], extent[1], 1]
        || owner.desc.array_layers != 1
        || owner.desc.mip_levels != 1
        || owner.desc.samples != 1
        || !matches!(
            owner.desc.format,
            TextureFormat::Rgba8Unorm | TextureFormat::Rgba16Float
        )
        || !owner
            .desc
            .usage
            .contains(TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING)
    {
        return Err(Error::InvalidState);
    }
    Ok(())
}

fn validate_depth<B: ResourceBackend>(
    owner: &crate::residency::ForwardTexture<B>,
    device: DeviceId,
    extent: [u32; 2],
) -> Result<(), Error> {
    if owner.device != device
        || owner.texture.device() != device
        || owner.view.device() != device
        || owner.desc.dimension != TextureDimension::D2
        || owner.desc.extent != [extent[0], extent[1], 1]
        || owner.desc.array_layers != 1
        || owner.desc.mip_levels != 1
        || owner.desc.samples != 1
        || owner.desc.format != TextureFormat::Depth32Float
        || !owner
            .desc
            .usage
            .contains(TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING)
    {
        return Err(Error::InvalidState);
    }
    Ok(())
}
