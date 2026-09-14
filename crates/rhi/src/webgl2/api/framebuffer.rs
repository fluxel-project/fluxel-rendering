//! Platform-neutral attachment, framebuffer, and render-pass vocabulary.

use super::{FramebufferId, GlError, GlFamilyApi, GlFormat, SurfaceImageId, TextureId};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlAttachmentTarget {
    Texture(TextureId),
    SurfaceImage(SurfaceImageId),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlTextureView {
    pub target: GlAttachmentTarget,
    pub format: GlFormat,
    pub mip_level: u32,
    pub array_layer: u32,
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlLoadOp {
    Load,
    Clear,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlStoreOp {
    Store,
    Discard,
}

/// IEEE values preserved by bit pattern, keeping pass descriptors hashable.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlColorClearValue {
    pub red: u32,
    pub green: u32,
    pub blue: u32,
    pub alpha: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlDepthStencilClearValue {
    pub depth: u32,
    pub stencil: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlColorAttachment {
    pub view: GlTextureView,
    pub resolve_target: Option<GlTextureView>,
    pub load: GlLoadOp,
    pub store: GlStoreOp,
    pub clear: GlColorClearValue,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlDepthStencilAttachment {
    pub view: GlTextureView,
    pub depth_load: GlLoadOp,
    pub depth_store: GlStoreOp,
    pub stencil_load: GlLoadOp,
    pub stencil_store: GlStoreOp,
    pub clear: GlDepthStencilClearValue,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlFramebufferDescriptor {
    pub color_attachments: Vec<GlTextureView>,
    pub depth_stencil_attachment: Option<GlTextureView>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlRenderPassDescriptor {
    pub framebuffer: FramebufferId,
    pub color_attachments: Vec<GlColorAttachment>,
    pub depth_stencil_attachment: Option<GlDepthStencilAttachment>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlFramebufferValidationError {
    NoAttachments,
    TooManyColorAttachments,
    ForeignContext,
    DuplicateAttachment,
    InvalidExtent,
    MismatchedExtent,
    MismatchedSampleCount,
    InvalidColorFormat,
    InvalidDepthStencilFormat,
    InvalidResolve,
    DiscardWithoutResolve,
}

impl GlAttachmentTarget {
    const fn context(self) -> super::ContextStamp {
        match self {
            Self::Texture(texture) => texture.context,
            Self::SurfaceImage(image) => image.context,
        }
    }
}
impl GlTextureView {
    fn is_depth_stencil(self) -> bool {
        matches!(
            self.format,
            GlFormat::Depth16Unorm | GlFormat::Depth24PlusStencil8 | GlFormat::Depth32Float
        )
    }
    fn validate_shape(self) -> Result<(), GlFramebufferValidationError> {
        if self.width == 0 || self.height == 0 || self.sample_count == 0 {
            Err(GlFramebufferValidationError::InvalidExtent)
        } else {
            Ok(())
        }
    }
}
impl GlFramebufferDescriptor {
    pub(crate) fn validate(
        &self,
        max_colors: u32,
        current: super::ContextStamp,
    ) -> Result<(), GlFramebufferValidationError> {
        if self.color_attachments.is_empty() && self.depth_stencil_attachment.is_none() {
            return Err(GlFramebufferValidationError::NoAttachments);
        }
        if self.color_attachments.len() > max_colors as usize {
            return Err(GlFramebufferValidationError::TooManyColorAttachments);
        }
        let mut all = self.color_attachments.clone();
        if let Some(depth) = self.depth_stencil_attachment {
            all.push(depth);
        }
        let Some(first) = all.first().copied() else {
            return Err(GlFramebufferValidationError::NoAttachments);
        };
        for view in all {
            view.validate_shape()?;
            if view.target.context() != current {
                return Err(GlFramebufferValidationError::ForeignContext);
            }
            if view.width != first.width || view.height != first.height {
                return Err(GlFramebufferValidationError::MismatchedExtent);
            }
            if view.sample_count != first.sample_count {
                return Err(GlFramebufferValidationError::MismatchedSampleCount);
            }
        }
        if self
            .color_attachments
            .iter()
            .any(|view| view.is_depth_stencil())
        {
            return Err(GlFramebufferValidationError::InvalidColorFormat);
        }
        if self
            .depth_stencil_attachment
            .is_some_and(|view| !view.is_depth_stencil())
        {
            return Err(GlFramebufferValidationError::InvalidDepthStencilFormat);
        }
        Ok(())
    }
}
impl GlRenderPassDescriptor {
    pub(crate) fn validate(
        &self,
        framebuffer: &GlFramebufferDescriptor,
        max_colors: u32,
        current: super::ContextStamp,
    ) -> Result<(), GlFramebufferValidationError> {
        framebuffer.validate(max_colors, current)?;
        if self.framebuffer.context != current {
            return Err(GlFramebufferValidationError::ForeignContext);
        }
        if self.color_attachments.len() != framebuffer.color_attachments.len() {
            return Err(GlFramebufferValidationError::TooManyColorAttachments);
        }
        for (attachment, view) in self
            .color_attachments
            .iter()
            .zip(&framebuffer.color_attachments)
        {
            if attachment.view != *view {
                return Err(GlFramebufferValidationError::DuplicateAttachment);
            }
            if let Some(resolve) = attachment.resolve_target {
                if resolve.target.context() != current
                    || resolve.format != view.format
                    || resolve.width != view.width
                    || resolve.height != view.height
                    || view.sample_count <= 1
                    || resolve.sample_count != 1
                {
                    return Err(GlFramebufferValidationError::InvalidResolve);
                }
            }
            if attachment.store == GlStoreOp::Discard && attachment.resolve_target.is_none() {
                return Err(GlFramebufferValidationError::DiscardWithoutResolve);
            }
        }
        Ok(())
    }
}

pub(crate) trait GlFramebufferApi: GlFamilyApi {
    fn create_framebuffer(
        &mut self,
        descriptor: &GlFramebufferDescriptor,
    ) -> Result<FramebufferId, GlError>;
    fn destroy_framebuffer(&mut self, framebuffer: FramebufferId) -> Result<(), GlError>;
    fn begin_render_pass(&mut self, descriptor: &GlRenderPassDescriptor) -> Result<(), GlError>;
    fn end_render_pass(&mut self) -> Result<(), GlError>;
}

#[cfg(test)]
mod tests {
    use super::{
        GlColorClearValue, GlFramebufferDescriptor, GlFramebufferValidationError, GlLoadOp,
        GlStoreOp,
    };
    #[test]
    fn clear_value_retains_float_bits() {
        let clear = GlColorClearValue {
            red: 1.0f32.to_bits(),
            green: 0,
            blue: 0,
            alpha: 1.0f32.to_bits(),
        };
        assert_eq!(clear.red, 1.0f32.to_bits());
        assert_eq!(GlLoadOp::Clear, GlLoadOp::Clear);
        assert_eq!(GlStoreOp::Discard, GlStoreOp::Discard);
    }
    #[test]
    fn rejects_empty_framebuffer() {
        let descriptor = GlFramebufferDescriptor {
            color_attachments: vec![],
            depth_stencil_attachment: None,
        };
        let stamp = super::super::ContextStamp::new(
            super::super::DeviceIdentity::new(1).unwrap(),
            super::super::ContextEpoch::INITIAL,
        );
        assert_eq!(
            descriptor.validate(1, stamp),
            Err(GlFramebufferValidationError::NoAttachments)
        );
    }
}
