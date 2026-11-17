//! Explicit logical-slot checks for the provisional captured-color consumer.

use fluxel_rhi::{
    api::{DeviceId, Error, ImageLayout, Texture, TextureUsage, TextureUses, TextureView},
    backend::ResourceBackend,
};

use super::super::RendererError;

/// The independent producer name supplied by a same-frame capture product.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct CaptureTextureSlot(String);

impl CaptureTextureSlot {
    /// Creates an explicit logical capture receiver slot.
    pub fn new(name: impl Into<String>) -> Result<Self, RendererError> {
        let name = name.into();
        if name.is_empty() {
            return Err(RendererError::MissingCaptureTextureSlot { name });
        }
        Ok(Self(name))
    }

    /// Returns the stable material resource name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.0
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
}

impl<B: ResourceBackend> CaptureTextureOwner<B> {
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
            || color.desc.format != fluxel_rhi::api::TextureFormat::Rgba8Unorm
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

    pub(crate) fn resident(&self) -> crate::residency::ForwardTexture<B> {
        crate::residency::ForwardTexture {
            texture: self.texture.clone(),
            view: self.view.clone(),
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
            || output.desc.format != fluxel_rhi::api::TextureFormat::Rgba8Unorm
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
            || self.desc.format != fluxel_rhi::api::TextureFormat::Rgba8Unorm
            || (require_filterable && !self.filterable)
            || !self.desc.usage.contains(TextureUsage::TEXTURE_BINDING)
        {
            return Err(Error::InvalidState);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/renderer/capture_slot.rs"]
mod tests;
