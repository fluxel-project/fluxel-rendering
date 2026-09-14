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
    /// Explicit `glDrawBuffers` selection over color-attachment indices.
    ///
    /// Empty keeps the driver's default mapping; a nonempty selection must be
    /// in bounds, within the discovered draw-buffer count, and duplicate free.
    pub draw_buffers: Vec<u32>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlRenderPassDescriptor {
    pub framebuffer: FramebufferId,
    pub color_attachments: Vec<GlColorAttachment>,
    pub depth_stencil_attachment: Option<GlDepthStencilAttachment>,
}

/// One source/destination rectangle pair for a framebuffer blit.
///
/// Origins are nonnegative so an out-of-framebuffer rectangle is a plain
/// bounds failure instead of a signed-coordinate edge case.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlBlitRegion {
    pub src_offset: [u32; 2],
    pub src_extent: [u32; 2],
    pub dst_offset: [u32; 2],
    pub dst_extent: [u32; 2],
}

impl GlBlitRegion {
    pub(crate) fn validate(&self) -> Result<(), GlFramebufferValidationError> {
        if self.src_extent.contains(&0) || self.dst_extent.contains(&0) {
            return Err(GlFramebufferValidationError::InvalidBlitRegion);
        }
        Ok(())
    }
}

/// The planes selected by one blit. At least one plane is required.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlBlitMask {
    pub color: bool,
    pub depth: bool,
    pub stencil: bool,
}

impl GlBlitMask {
    pub(crate) const fn is_empty(self) -> bool {
        !self.color && !self.depth && !self.stencil
    }
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
    TooManyDrawBuffers,
    DrawBufferIndexOutOfBounds,
    DuplicateDrawBuffer,
    InvalidBlitRegion,
    EmptyBlitMask,
    BlitFilterIncompatible,
    IdenticalBlitTargets,
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
        max_draw_buffers: u32,
        current: super::ContextStamp,
    ) -> Result<(), GlFramebufferValidationError> {
        if self.color_attachments.is_empty() && self.depth_stencil_attachment.is_none() {
            return Err(GlFramebufferValidationError::NoAttachments);
        }
        if self.color_attachments.len() > max_colors as usize {
            return Err(GlFramebufferValidationError::TooManyColorAttachments);
        }
        if self.draw_buffers.len() > max_draw_buffers as usize {
            return Err(GlFramebufferValidationError::TooManyDrawBuffers);
        }
        for (position, index) in self.draw_buffers.iter().enumerate() {
            if *index as usize >= self.color_attachments.len() {
                return Err(GlFramebufferValidationError::DrawBufferIndexOutOfBounds);
            }
            if self.draw_buffers[..position].contains(index) {
                return Err(GlFramebufferValidationError::DuplicateDrawBuffer);
            }
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
        max_draw_buffers: u32,
        current: super::ContextStamp,
    ) -> Result<(), GlFramebufferValidationError> {
        framebuffer.validate(max_colors, max_draw_buffers, current)?;
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
    /// Copies selected planes from one framebuffer into another.
    ///
    /// This is also the MSAA resolve word: resolving is a blit whose source
    /// is multisampled, so no separate resolve command exists in this contract.
    /// Multisampled targets only accept nearest filtering; providers reject
    /// the clearly incompatible combinations before touching GL state.
    fn blit_framebuffer(
        &mut self,
        source: FramebufferId,
        destination: FramebufferId,
        region: GlBlitRegion,
        filter: super::GlFilterMode,
        masks: GlBlitMask,
    ) -> Result<(), GlError>;
}

#[cfg(test)]
mod tests {
    use super::{
        GlBlitMask, GlBlitRegion, GlColorClearValue, GlFramebufferDescriptor,
        GlFramebufferValidationError, GlLoadOp, GlStoreOp,
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
            draw_buffers: vec![],
        };
        let stamp = super::super::ContextStamp::new(
            super::super::DeviceIdentity::new(1).unwrap(),
            super::super::ContextEpoch::INITIAL,
        );
        assert_eq!(
            descriptor.validate(1, 1, stamp),
            Err(GlFramebufferValidationError::NoAttachments)
        );
    }
    #[test]
    fn draw_buffers_selection_is_bounded_unique_and_in_range() {
        let descriptor = |draw_buffers: Vec<u32>| GlFramebufferDescriptor {
            color_attachments: vec![],
            depth_stencil_attachment: Some(empty_view()),
            draw_buffers,
        };
        let stamp = super::super::ContextStamp::new(
            super::super::DeviceIdentity::new(1).unwrap(),
            super::super::ContextEpoch::INITIAL,
        );
        assert_eq!(
            descriptor(vec![0]).validate(4, 4, stamp),
            Err(GlFramebufferValidationError::DrawBufferIndexOutOfBounds)
        );
        assert_eq!(descriptor(vec![]).validate(4, 4, stamp), Ok(()));
    }
    #[test]
    fn blit_regions_and_masks_fail_closed_before_gl() {
        let region = GlBlitRegion {
            src_offset: [0; 2],
            src_extent: [4, 4],
            dst_offset: [0; 2],
            dst_extent: [4, 4],
        };
        assert_eq!(region.validate(), Ok(()));
        assert_eq!(
            GlBlitRegion {
                dst_extent: [0, 4],
                ..region
            }
            .validate(),
            Err(GlFramebufferValidationError::InvalidBlitRegion)
        );
        assert!(
            GlBlitMask {
                color: false,
                depth: false,
                stencil: false
            }
            .is_empty()
        );
    }
    fn empty_view() -> super::GlTextureView {
        super::GlTextureView {
            target: super::GlAttachmentTarget::Texture(super::TextureId::new(
                super::super::ContextStamp::new(
                    super::super::DeviceIdentity::new(1).unwrap(),
                    super::super::ContextEpoch::INITIAL,
                ),
                0,
                0,
            )),
            format: super::GlFormat::Depth32Float,
            mip_level: 0,
            array_layer: 0,
            width: 1,
            height: 1,
            sample_count: 1,
        }
    }
}
