//! Sections 30 and 31: geometry, clear values, and the attachment set.

use super::*;
use crate::api::command::attachment::validate_raster_scope;
use crate::api::command::geometry::{validate_rect, validate_viewport};
use crate::api::command::{ColorClearValue, Rect, Viewport};

#[test]
fn a_rect_that_overflows_is_refused_at_both_ends() {
    // Section 30's geometry rule: `x + width` must not overflow, and the accessor
    // answers `None` rather than wrapping so that a caller cannot compare a wrapped
    // end against an extent and conclude the rect was inside it.
    let overflowing = Rect::new(u32::MAX, 0, 2, 1);
    assert_eq!(overflowing.right(), None);
    assert_kind(
        validate_rect(overflowing, "the scissor rect"),
        RhiErrorKind::InvalidUsage,
    );

    let fine = Rect::new(3, 3, 1, 1);
    assert_eq!(fine.right(), Some(4));
    assert!(validate_rect(fine, "the scissor rect").is_ok());

    // A zero-area rect is legal: it clips everything, which is a thing a caller may
    // ask for.
    assert!(validate_rect(Rect::new(0, 0, 0, 0), "the scissor rect").is_ok());
}

#[test]
fn a_viewport_that_is_not_a_range_is_refused() {
    // A negative extent is not an extent.
    assert_kind(
        validate_viewport(Viewport::new(0.0, 0.0, -1.0, 4.0, 0.0, 1.0)),
        RhiErrorKind::InvalidUsage,
    );
    // min_depth above max_depth is not a range.
    assert_kind(
        validate_viewport(Viewport::new(0.0, 0.0, 4.0, 4.0, 1.0, 0.0)),
        RhiErrorKind::InvalidUsage,
    );
    // A depth outside 0.0..=1.0 is outside the clip volume's depth convention.
    assert_kind(
        validate_viewport(Viewport::new(0.0, 0.0, 4.0, 4.0, 0.0, 1.5)),
        RhiErrorKind::InvalidUsage,
    );
    // A viewport that is not a number cannot be rasterized to anything.
    assert_kind(
        validate_viewport(Viewport::new(f32::NAN, 0.0, 4.0, 4.0, 0.0, 1.0)),
        RhiErrorKind::InvalidUsage,
    );
    // A zero-width viewport is *legal*: section 30 asks for `width >= 0`, not
    // `width > 0`, and a degenerate viewport draws nothing rather than being
    // malformed. Same for a degenerate depth range.
    assert!(validate_viewport(Viewport::new(0.0, 0.0, 0.0, 4.0, 0.0, 1.0)).is_ok());
    assert!(validate_viewport(Viewport::new(0.0, 0.0, 4.0, 4.0, 0.0, 1.0)).is_ok());
    assert!(validate_viewport(Viewport::new(0.0, 0.0, 4.0, 4.0, 0.5, 0.5)).is_ok());
}

#[test]
fn an_empty_attachment_set_is_refused() {
    assert_kind(
        validate_raster_scope(&RasterScopeDescriptor::new()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_color_clear_must_match_the_formats_clear_class() {
    // Rgba8Sint clears with an integer value; a float clear on it is not a
    // representable clear rather than a rounding question.
    let signed = RasterScopeDescriptor::new().with_color(
        ShaderLocation::new(0),
        ColorAttachment {
            view: ColorAttachmentView::Texture(color_view_of(&renderable_texture(
                TextureFormat::Rgba8Sint,
            ))),
            load: LoadOp::Clear(ColorClearValue::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
            resolve: None,
        },
    );
    assert_kind(validate_raster_scope(&signed), RhiErrorKind::InvalidUsage);

    let matching = RasterScopeDescriptor::new().with_color(
        ShaderLocation::new(0),
        ColorAttachment {
            view: ColorAttachmentView::Texture(color_view_of(&renderable_texture(
                TextureFormat::Rgba8Sint,
            ))),
            load: LoadOp::Clear(ColorClearValue::Sint([0, 0, 0, 1])),
            store: StoreOp::Store,
            resolve: None,
        },
    );
    assert!(validate_raster_scope(&matching).is_ok());
}

#[test]
fn a_color_attachment_needs_color_attachment_usage() {
    let sampled_only = Texture::new(
        object(22),
        device(),
        TextureDescriptor::new_2d(4, 4, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED),
    );
    let scope = RasterScopeDescriptor::new().with_color(
        ShaderLocation::new(0),
        ColorAttachment {
            view: ColorAttachmentView::Texture(color_view_of(&sampled_only)),
            load: LoadOp::Load,
            store: StoreOp::Store,
            resolve: None,
        },
    );
    assert_kind(validate_raster_scope(&scope), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_frame_attachment_must_store() {
    // Section 31.1: a frame is rendered in order to be presented, so discarding it
    // asks for contents that are thrown away.
    let discarded = RasterScopeDescriptor::new().with_color(
        ShaderLocation::new(0),
        ColorAttachment {
            view: ColorAttachmentView::Frame(frame_attachment(TextureFormat::Bgra8Unorm)),
            load: LoadOp::Clear(ColorClearValue::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Discard,
            resolve: None,
        },
    );
    assert_kind(
        validate_raster_scope(&discarded),
        RhiErrorKind::InvalidUsage,
    );

    let stored = RasterScopeDescriptor::new().with_color(
        ShaderLocation::new(0),
        ColorAttachment {
            view: ColorAttachmentView::Frame(frame_attachment(TextureFormat::Bgra8Unorm)),
            load: LoadOp::Clear(ColorClearValue::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
            resolve: None,
        },
    );
    assert!(validate_raster_scope(&stored).is_ok());
}
