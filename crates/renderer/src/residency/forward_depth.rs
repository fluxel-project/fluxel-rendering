//! Concrete per-frame depth attachment ownership for Forward.

use fluxel_rhi::{
    api::{
        Device, DeviceId, Error, ImageLayout, Result, Texture, TextureDesc, TextureDimension,
        TextureFormat, TextureUsage, TextureUses, TextureView, TextureViewDesc,
        TextureViewDimension,
    },
    backend::ResourceBackend,
};

use crate::graph::ResourceInstanceId;

/// A per-frame `Depth32Float` attachment owned by Renderer.
pub(crate) struct ForwardDepthAttachment<B: ResourceBackend> {
    pub(crate) texture: Texture<B>,
    pub(crate) view: TextureView<B>,
    pub(crate) desc: TextureDesc,
    pub(crate) instance: ResourceInstanceId,
    pub(crate) incoming: TextureUses,
    pub(crate) layout: ImageLayout,
    pub(crate) device: DeviceId,
}

impl<B: ResourceBackend> ForwardDepthAttachment<B> {
    pub(crate) fn from_texture(depth: &super::ForwardTexture<B>) -> Self {
        Self {
            texture: depth.texture.clone(),
            view: depth.view.clone(),
            desc: depth.desc.clone(),
            instance: depth.instance,
            incoming: depth.incoming,
            layout: depth.layout,
            device: depth.device,
        }
    }

    pub(crate) fn retain(&self) -> Self {
        Self {
            texture: self.texture.clone(),
            view: self.view.clone(),
            desc: self.desc.clone(),
            instance: self.instance,
            incoming: self.incoming,
            layout: self.layout,
            device: self.device,
        }
    }

    /// Allocates a single-sampled depth attachment after capability admission.
    pub(crate) fn allocate(device: &Device<B>, extent: [u32; 2]) -> Result<Self> {
        if extent.contains(&0) {
            return Err(Error::InvalidRange);
        }
        let desc = TextureDesc {
            label: Some("fluxel forward depth".into()),
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            array_layers: 1,
            format: TextureFormat::Depth32Float,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: Vec::new(),
        };
        super::forward_support::require_texture(device, &desc)?;
        let texture = device.create_texture(&desc)?;
        let view = device.create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Depth32Float,
                dimension: TextureViewDimension::D2,
                aspects: fluxel_rhi::api::TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        Ok(Self {
            texture,
            view,
            desc,
            instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
            incoming: TextureUses {
                stages: fluxel_rhi::api::StageMask::EMPTY,
                accesses: fluxel_rhi::api::AccessMask::EMPTY,
            },
            layout: ImageLayout::Undefined,
            device: device.id(),
        })
    }
}
