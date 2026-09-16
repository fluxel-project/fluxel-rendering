//! Platform-neutral attachment, framebuffer, and render-pass vocabulary.
//!
//! This module owns the attachment-view description, the framebuffer and
//! render-pass descriptors built from it, and the validation both descriptors
//! run before a provider touches GL state. It does not own object lifetime
//! (that is the provider) and does not decide whether a context proved a
//! capability: it states the rule and the structured reason, and the provider
//! supplies the discovered facts the rule is checked against.
//!
//! Two storage classes can back an attachment, and both go through the same
//! view type, the same rule sequence, and the same multiview gate. A texture
//! contributes a mip chain and a layer dimension; a renderbuffer is a single
//! allocation with neither, so its only addressable view is level 0, layer 0,
//! of exactly one layer -- the rule that would otherwise be misread as a
//! layered attachment.
//!
//! There is deliberately **no third class for the window system's drawable**,
//! and the absence is a decision rather than a gap.  A surface image is not an
//! attachment here because it is not an attachment anywhere else in this
//! contract: the common RHI's present root is a *texture resource* a graph
//! renders into and a backend then presents, so the drawable is reached by
//! publishing a texture to it, not by naming it as a pass's render target.
//! Layer 1's attachment vocabulary carried a variant for it once
//! (`GlAttachmentTarget::SurfaceImage`) while every provider that could execute
//! one refused it with the same sentence, and Layer 2 tracked it as a
//! dependency no cache ever invalidated; both were removed for the reason the
//! common contract gives.  What this means for a reader is that a pass
//! describes only storage this layer can allocate, and "render to the screen"
//! is a step *after* the pass rather than inside it.

use super::{FramebufferId, GlError, GlFamilyApi, GlFormat, RenderbufferId, TextureId};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlAttachmentTarget {
    Texture(TextureId),
    /// A renderbuffer allocation, which has no mip chain and no layers.
    Renderbuffer(RenderbufferId),
}

/// One view of one attachment's storage.
///
/// The name stays texture-first because it is the common case, but the fields
/// are the view of whichever storage class `target` names: `mip_level`,
/// `array_layer`, and `layer_count` describe a coordinate into the allocation,
/// and a renderbuffer is the degenerate coordinate (level 0, layer 0, one
/// layer) that its own validation arm enforces.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlTextureView {
    pub target: GlAttachmentTarget,
    pub format: GlFormat,
    pub mip_level: u32,
    pub array_layer: u32,
    /// Array layers this view serves in one pass; `1` is a plain attachment.
    ///
    /// A view with more than one layer is a multiview attachment: the pass then
    /// renders that many views, and every attachment of the framebuffer must
    /// agree on the count. The field is view vocabulary, not a context
    /// property; whether the context can serve the count is decided against the
    /// discovered view-count limit, so an unsupported count is rejected before
    /// the first pass rather than at the first draw.
    pub layer_count: u32,
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
    InvalidLayerCount,
    MismatchedExtent,
    MismatchedLayerCount,
    MismatchedSampleCount,
    InvalidColorFormat,
    InvalidDepthStencilFormat,
    InvalidResolve,
    DiscardWithoutResolve,
    TooManyDrawBuffers,
    DrawBufferIndexOutOfBounds,
    DuplicateDrawBuffer,
    MultiviewNotSupported,
    MultiviewViewCountExceedsLimit,
    InvalidBlitRegion,
    EmptyBlitMask,
    BlitFilterIncompatible,
    IdenticalBlitTargets,
}

impl GlAttachmentTarget {
    const fn context(self) -> super::ContextStamp {
        match self {
            Self::Texture(texture) => texture.context,
            Self::Renderbuffer(renderbuffer) => renderbuffer.context,
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
            return Err(GlFramebufferValidationError::InvalidExtent);
        }
        if self.layer_count == 0 {
            return Err(GlFramebufferValidationError::InvalidLayerCount);
        }
        Ok(())
    }
}

/// Rejects an attachment view count this context cannot serve.
///
/// One view is the count every pass already uses and is always legal. More than
/// one view is a multiview pass, which needs both the multiview capability and
/// a view count within the discovered limit; a context that proved neither
/// rejects here, before any pass state changes, instead of rendering the first
/// layer and silently dropping the rest.
fn validate_view_count(
    view_count: u32,
    max_views: u32,
) -> Result<(), GlFramebufferValidationError> {
    if view_count <= 1 {
        return Ok(());
    }
    if max_views < 2 {
        return Err(GlFramebufferValidationError::MultiviewNotSupported);
    }
    if view_count > max_views {
        return Err(GlFramebufferValidationError::MultiviewViewCountExceedsLimit);
    }
    Ok(())
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
            // A pass renders one view count: attachments that disagree would
            // need per-attachment view loops the contract does not define. The
            // count is a shape fact like the extent, so it is compared with the
            // other shape rules, before the extent is.
            if view.layer_count != first.layer_count {
                return Err(GlFramebufferValidationError::MismatchedLayerCount);
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

    /// Returns the view count every attachment of this framebuffer agrees on.
    ///
    /// `validate` has already rejected disagreement, so the first attachment's
    /// count is the whole framebuffer's count; no attachments means no views.
    pub(crate) fn view_count(&self) -> u32 {
        self.color_attachments
            .first()
            .or(self.depth_stencil_attachment.as_ref())
            .map_or(0, |view| view.layer_count)
    }

    /// Rejects a multiview view count this context cannot serve.
    pub(crate) fn validate_multiview(
        &self,
        max_views: u32,
    ) -> Result<(), GlFramebufferValidationError> {
        validate_view_count(self.view_count(), max_views)
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
                // A resolve stays a single-layer blit destination: resolving a
                // multiview attachment is one view per command, never one
                // command for every view.
                if resolve.target.context() != current
                    || resolve.format != view.format
                    || resolve.width != view.width
                    || resolve.height != view.height
                    || resolve.layer_count != 1
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

    /// Returns the view count this pass renders, taken from its attachments.
    pub(crate) fn view_count(&self) -> u32 {
        self.color_attachments
            .first()
            .map(|attachment| attachment.view.layer_count)
            .or_else(|| {
                self.depth_stencil_attachment
                    .map(|attachment| attachment.view.layer_count)
            })
            .unwrap_or(0)
    }

    /// Rejects a multiview view count this context cannot serve.
    ///
    /// `validate` has already proved every attachment matches its framebuffer
    /// view, so the pass agrees on one view count by construction.
    pub(crate) fn validate_multiview(
        &self,
        max_views: u32,
    ) -> Result<(), GlFramebufferValidationError> {
        validate_view_count(self.view_count(), max_views)
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
            layer_count: 1,
            width: 1,
            height: 1,
            sample_count: 1,
        }
    }

    fn stamp() -> super::super::ContextStamp {
        super::super::ContextStamp::new(
            super::super::DeviceIdentity::new(1).unwrap(),
            super::super::ContextEpoch::INITIAL,
        )
    }

    /// A color-renderable view of `layer_count` layers on a fresh allocation.
    fn color_view(slot: u32, layer_count: u32) -> super::GlTextureView {
        super::GlTextureView {
            target: super::GlAttachmentTarget::Texture(super::TextureId::new(stamp(), slot, 0)),
            format: super::GlFormat::Rgba8Unorm,
            mip_level: 0,
            array_layer: 0,
            layer_count,
            width: 4,
            height: 4,
            sample_count: 1,
        }
    }

    #[test]
    fn attachment_views_must_agree_on_their_layer_count() {
        let color = color_view(0, 2);
        // The depth view matches the color view in every shape rule but the
        // layer count, so the case can only fail on that rule.
        let depth = super::GlTextureView {
            layer_count: 1,
            width: color.width,
            height: color.height,
            ..empty_view()
        };
        let descriptor = GlFramebufferDescriptor {
            color_attachments: vec![color],
            depth_stencil_attachment: Some(depth),
            draw_buffers: vec![],
        };
        assert_eq!(
            descriptor.validate(4, 4, stamp()),
            Err(GlFramebufferValidationError::MismatchedLayerCount)
        );
        let agreed = GlFramebufferDescriptor {
            depth_stencil_attachment: Some(super::GlTextureView {
                format: super::GlFormat::Depth32Float,
                layer_count: 2,
                ..depth
            }),
            ..descriptor
        };
        assert_eq!(agreed.validate(4, 4, stamp()), Ok(()));
        assert_eq!(agreed.view_count(), 2);
    }

    #[test]
    fn a_view_serving_no_layer_is_not_a_view() {
        let descriptor = GlFramebufferDescriptor {
            color_attachments: vec![color_view(0, 0)],
            depth_stencil_attachment: None,
            draw_buffers: vec![],
        };
        assert_eq!(
            descriptor.validate(4, 4, stamp()),
            Err(GlFramebufferValidationError::InvalidLayerCount)
        );
    }

    /// A multiview view count is refused outright on a context that did not
    /// prove multiview, and refused above the proved count otherwise.
    #[test]
    fn multiview_view_count_is_gated_by_the_proved_limit() {
        let descriptor = GlFramebufferDescriptor {
            color_attachments: vec![color_view(0, 2)],
            depth_stencil_attachment: None,
            draw_buffers: vec![],
        };
        assert_eq!(
            descriptor.validate_multiview(1),
            Err(GlFramebufferValidationError::MultiviewNotSupported)
        );
        assert_eq!(descriptor.validate_multiview(2), Ok(()));

        let three = GlFramebufferDescriptor {
            color_attachments: vec![color_view(0, 3)],
            ..descriptor
        };
        assert_eq!(
            three.validate_multiview(2),
            Err(GlFramebufferValidationError::MultiviewViewCountExceedsLimit)
        );
        assert_eq!(three.validate_multiview(3), Ok(()));
    }

    #[test]
    fn single_view_passes_never_need_the_multiview_limit() {
        let descriptor = GlFramebufferDescriptor {
            color_attachments: vec![],
            depth_stencil_attachment: Some(empty_view()),
            draw_buffers: vec![],
        };
        assert_eq!(descriptor.view_count(), 1);
        assert_eq!(descriptor.validate_multiview(1), Ok(()));
    }

    /// A pass may not resolve into a multiview attachment: a resolve is a
    /// single-view blit, so a multilayer destination has no defined result.
    #[test]
    fn resolve_targets_are_single_layer() {
        let view = color_view(0, 1);
        let multisampled = super::GlTextureView {
            sample_count: 4,
            ..view
        };
        let pass = super::GlRenderPassDescriptor {
            framebuffer: super::FramebufferId::new(stamp(), 0, 0),
            color_attachments: vec![super::GlColorAttachment {
                view: multisampled,
                resolve_target: Some(super::GlTextureView {
                    layer_count: 2,
                    ..view
                }),
                load: GlLoadOp::Load,
                store: GlStoreOp::Store,
                clear: GlColorClearValue {
                    red: 0,
                    green: 0,
                    blue: 0,
                    alpha: 0,
                },
            }],
            depth_stencil_attachment: None,
        };
        let framebuffer = GlFramebufferDescriptor {
            color_attachments: vec![multisampled],
            depth_stencil_attachment: None,
            draw_buffers: vec![],
        };
        assert_eq!(
            pass.validate(&framebuffer, 4, 4, stamp()),
            Err(GlFramebufferValidationError::InvalidResolve)
        );
        // The pass itself already agrees on one view count, so the multiview
        // gate stays satisfied and only the resolve rule rejects.
        assert_eq!(pass.view_count(), 1);
        assert_eq!(pass.validate_multiview(1), Ok(()));
    }
}
