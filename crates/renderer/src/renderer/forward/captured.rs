//! Explicit logical-slot checks for the provisional captured-color consumer.

use fluxel_rhi::{
    api::{DeviceId, Error, ImageLayout, Texture, TextureUsage, TextureUses, TextureView},
    backend::ResourceBackend,
};

use super::super::RendererError;

/// The logical material resource supplied by a same-frame capture product.
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
        let filterable = match device.texture_support(&frame.frame.target_desc) {
            fluxel_rhi::api::Support::Supported(facts) => facts.filterable,
            fluxel_rhi::api::Support::Unsupported(reason) => {
                return Err(Error::Unsupported(reason));
            }
        };
        Ok(Self {
            slot,
            texture: frame.frame.target.clone(),
            view: frame.frame.target_view.clone(),
            desc: frame.frame.target_desc.clone(),
            instance: frame.frame.target_instance,
            incoming: frame.frame.target_incoming,
            layout: frame.frame.target_layout,
            device: frame.resident.device,
            filterable,
        })
    }

    pub(crate) fn validate_for_receiver(
        &self,
        device: DeviceId,
        expected_slot: &str,
    ) -> Result<(), Error> {
        if self.device != device || self.texture.device() != device || self.view.device() != device
        {
            return Err(Error::WrongDevice);
        }
        if self.slot.name() != expected_slot
            || self.desc.format != fluxel_rhi::api::TextureFormat::Rgba8Unorm
            || !self.filterable
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
