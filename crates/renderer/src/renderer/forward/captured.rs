//! Explicit logical-slot checks for the provisional captured-color consumer.

use fluxel_rhi::{
    api::{
        AccessMask, Device, DeviceId, Error, ImageLayout, StageMask, Support, Texture,
        TextureAspects, TextureDesc, TextureDimension, TextureUsage, TextureUses, TextureView,
        TextureViewDesc, TextureViewDimension,
    },
    backend::ResourceBackend,
};

use super::super::RendererError;

/// The independent producer name supplied by a same-frame capture product.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct CaptureTextureSlot {
    name: String,
    mipmapped_refraction: bool,
}

impl CaptureTextureSlot {
    /// Creates an explicit logical capture receiver slot.
    pub fn new(name: impl Into<String>) -> Result<Self, RendererError> {
        let name = name.into();
        if name.is_empty() {
            return Err(RendererError::MissingCaptureTextureSlot { name });
        }
        Ok(Self {
            name,
            mipmapped_refraction: false,
        })
    }

    /// Returns the stable material resource name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Opts this producer slot into a mipmapped scene-refraction capture.
    ///
    /// The default remains the producer's sharp base-level color, matching
    /// Babylon render-target refraction unless linear-specular LOD is enabled.
    #[must_use]
    pub fn with_mipmapped_refraction(mut self) -> Self {
        self.mipmapped_refraction = true;
        self
    }

    pub(in crate::renderer) const fn mipmapped_refraction(&self) -> bool {
        self.mipmapped_refraction
    }
}

/// Concrete owner facts for the one capture product admitted into a receiver.
///
/// This is private Renderer plumbing: it retains cloned RHI handles from the
/// producer frame and never exposes a backend object in the public material
/// model.  The combined Graph still imports the producer target once and
/// declares its write before this receiver's sampled read.
pub(crate) struct CaptureTextureOwner<B: ResourceBackend> {
    pub(crate) slot: CaptureTextureSlot,
    pub(crate) texture: Texture<B>,
    pub(crate) view: TextureView<B>,
    pub(crate) desc: fluxel_rhi::api::TextureDesc,
    pub(crate) instance: crate::graph::ResourceInstanceId,
    pub(crate) incoming: TextureUses,
    pub(crate) layout: ImageLayout,
    pub(crate) device: DeviceId,
    pub(crate) filterable: bool,
    pub(super) refraction_depth: Option<crate::residency::ForwardDepthAttachment<B>>,
}

/// Returns the dedicated sampled capture descriptor used by projected scene
/// refraction.
///
/// The producer keeps its original single-level target and its generic capture
/// behavior. This descriptor instead owns the copied color product and its
/// complete mip chain.
pub(crate) fn refraction_capture_desc(source: &TextureDesc) -> TextureDesc {
    let mut desc = source.clone();
    desc.label = Some("Fluxel scene refraction capture".into());
    desc.mip_levels = 1 + source.extent[0].max(source.extent[1]).max(1).ilog2();
    desc.usage = TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING;
    desc
}

impl<B: ResourceBackend> CaptureTextureOwner<B> {
    /// Creates the separate, mipmapped capture product required by projected
    /// scene refraction. A 1×1 source is already its complete mip chain and
    /// therefore remains an alias of the producer product.
    pub(in crate::renderer) fn for_scene_refraction(
        self,
        device: &Device<B>,
    ) -> Result<Self, Error> {
        if !self.slot.mipmapped_refraction() {
            return Ok(self);
        }
        if self.device != device.id()
            || self.texture.device() != device.id()
            || self.view.device() != device.id()
            || self.desc.dimension != TextureDimension::D2
            || self.desc.extent.contains(&0)
            || self.desc.extent[2] != 1
            || self.desc.array_layers != 1
            || self.desc.mip_levels != 1
            || self.desc.samples != 1
            || !Self::is_capture_format(self.desc.format)
            || !self.desc.usage.contains(TextureUsage::COPY_SRC)
            || !self.desc.usage.contains(TextureUsage::TEXTURE_BINDING)
            || !self.filterable
        {
            return Err(Error::InvalidState);
        }
        if self.desc.extent[0] == 1 && self.desc.extent[1] == 1 {
            return Ok(self);
        }
        let desc = refraction_capture_desc(&self.desc);
        let facts = match device.texture_support(&desc) {
            Support::Supported(facts) => facts,
            Support::Unsupported(reason) => return Err(Error::Unsupported(reason)),
        };
        if !facts.filterable || facts.max_mips < desc.mip_levels {
            return Err(Error::Unsupported("filterable scene refraction capture"));
        }
        let texture = device.create_texture(&desc)?;
        let view = device.create_view(
            &texture,
            &TextureViewDesc {
                format: desc.format,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..desc.mip_levels,
                layers: 0..1,
            },
        )?;
        Ok(Self {
            slot: self.slot,
            texture,
            view,
            desc,
            instance: crate::graph::ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
            incoming: TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            layout: ImageLayout::Undefined,
            device: device.id(),
            filterable: facts.filterable,
            refraction_depth: self.refraction_depth,
        })
    }

    /// Borrows the actual Deferred lighting product for same-Graph capture.
    pub(in crate::renderer) fn from_deferred_scene_color(
        device: &fluxel_rhi::api::Device<B>,
        slot: CaptureTextureSlot,
        color: &crate::residency::ForwardTexture<B>,
        extent: [u32; 2],
    ) -> Result<Self, Error> {
        if color.desc.dimension != fluxel_rhi::api::TextureDimension::D2
            || color.desc.extent != [extent[0], extent[1], 1]
            || extent.contains(&0)
            || color.desc.array_layers != 1
            || color.desc.mip_levels != 1
            || color.desc.samples != 1
            || !Self::is_capture_format(color.desc.format)
        {
            return Err(Error::InvalidState);
        }
        Self::from_actual(
            device,
            slot,
            &color.texture,
            &color.view,
            &color.desc,
            color.instance,
            color.incoming,
            color.layout,
            color.device,
        )
    }

    pub(in crate::renderer) fn retain_deferred_refraction_depth(
        &mut self,
        depth: &crate::residency::ForwardTexture<B>,
    ) {
        self.refraction_depth = Some(crate::residency::ForwardDepthAttachment::from_texture(
            depth,
        ));
    }

    pub(crate) fn resident(&self) -> crate::residency::ForwardTexture<B> {
        crate::residency::ForwardTexture {
            texture: self.texture.clone(),
            view: self.view.clone(),
            view_dimension: TextureViewDimension::D2,
            desc: self.desc.clone(),
            instance: self.instance,
            incoming: self.incoming,
            layout: self.layout,
            device: self.device,
            filterable: self.filterable,
        }
    }
    pub(crate) fn from_capture_frame(
        device: &fluxel_rhi::api::Device<B>,
        slot: CaptureTextureSlot,
        frame: &crate::residency::ForwardFrame<B>,
    ) -> Result<Self, Error> {
        if frame.resident.device != device.id()
            || frame.frame.target.device() != device.id()
            || frame.frame.target_view.device() != device.id()
            || !frame
                .frame
                .target_desc
                .usage
                .contains(TextureUsage::TEXTURE_BINDING)
        {
            return Err(Error::WrongDevice);
        }
        Self::from_actual(
            device,
            slot,
            &frame.frame.target,
            &frame.frame.target_view,
            &frame.frame.target_desc,
            frame.frame.target_instance,
            frame.frame.target_incoming,
            frame.frame.target_layout,
            frame.resident.device,
        )
    }

    /// Promotes the actual final OIT output to a capture producer.  OIT's
    /// opaque Forward target is intentionally not accepted at this boundary.
    pub(crate) fn from_oit_output(
        device: &fluxel_rhi::api::Device<B>,
        slot: CaptureTextureSlot,
        output: &crate::residency::OitTextureOwner<B>,
    ) -> Result<Self, Error> {
        if output.device != device.id()
            || output.texture.device() != device.id()
            || output.view.device() != device.id()
        {
            return Err(Error::WrongDevice);
        }
        if output.desc.dimension != fluxel_rhi::api::TextureDimension::D2
            || output.desc.extent.contains(&0)
            || output.desc.extent[2] != 1
            || output.desc.array_layers != 1
            || output.desc.mip_levels != 1
            || output.desc.samples != 1
            || !Self::is_capture_format(output.desc.format)
            || output.view_desc.format != output.desc.format
            || output.view_desc.dimension != fluxel_rhi::api::TextureViewDimension::D2
            || output.view_desc.aspects != fluxel_rhi::api::TextureAspects::COLOR
            || output.view_desc.mips != (0..1)
            || output.view_desc.layers != (0..1)
        {
            return Err(Error::InvalidState);
        }
        Self::from_actual(
            device,
            slot,
            &output.texture,
            &output.view,
            &output.desc,
            output.instance,
            output.incoming,
            output.layout,
            output.device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_actual(
        device: &fluxel_rhi::api::Device<B>,
        slot: CaptureTextureSlot,
        texture: &Texture<B>,
        view: &TextureView<B>,
        desc: &fluxel_rhi::api::TextureDesc,
        instance: crate::graph::ResourceInstanceId,
        incoming: TextureUses,
        layout: ImageLayout,
        owner_device: DeviceId,
    ) -> Result<Self, Error> {
        if owner_device != device.id()
            || texture.device() != device.id()
            || view.device() != device.id()
            || !Self::is_capture_format(desc.format)
            || !desc.usage.contains(TextureUsage::TEXTURE_BINDING)
        {
            return Err(Error::WrongDevice);
        }
        let filterable = match device.texture_support(desc) {
            fluxel_rhi::api::Support::Supported(facts) => facts.filterable,
            fluxel_rhi::api::Support::Unsupported(reason) => {
                return Err(Error::Unsupported(reason));
            }
        };
        Ok(Self {
            slot,
            texture: texture.clone(),
            view: view.clone(),
            desc: desc.clone(),
            instance,
            incoming,
            layout,
            device: owner_device,
            filterable,
            refraction_depth: None,
        })
    }

    pub(crate) fn validate_for_receiver(
        &self,
        device: DeviceId,
        expected_slot: &str,
        require_filterable: bool,
    ) -> Result<(), Error> {
        if self.device != device || self.texture.device() != device || self.view.device() != device
        {
            return Err(Error::WrongDevice);
        }
        if self.slot.name() != expected_slot
            || !Self::is_capture_format(self.desc.format)
            || (require_filterable && !self.filterable)
            || !self.desc.usage.contains(TextureUsage::TEXTURE_BINDING)
        {
            return Err(Error::InvalidState);
        }
        Ok(())
    }

    const fn is_capture_format(format: fluxel_rhi::api::TextureFormat) -> bool {
        matches!(
            format,
            fluxel_rhi::api::TextureFormat::Rgba8Unorm
                | fluxel_rhi::api::TextureFormat::Rgba16Float
        )
    }
}

#[cfg(test)]
#[path = "../../tests/renderer/capture_slot.rs"]
mod tests;
