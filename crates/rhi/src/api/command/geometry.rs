//! Portable geometry, clear values, and dynamic-state values (specification
//! section 30).
//!
//! This module owns the values a recorded command carries that are neither a
//! resource nor an attachment: a clear color, a scissor rect, a viewport, a load
//! and a store operation, and the color a blend constant is set to.
//!
//! # What this module does not own
//!
//! - Which attachment a value is applied to. A [`ColorClearValue`] does not know
//!   a format, so it cannot know whether it is the right *class* of clear; that
//!   comparison needs the attachment and is
//!   [`crate::api::command::attachment`]'s.
//! - Any backend's dynamic-state limit. A viewport larger than the attachment is
//!   legal here and clipped by the rasterizer, exactly as the native APIs define
//!   it; section 30 does not make it a portable error.
//! - Depth bias, blend factors, and stencil state, which are pipeline values
//!   (section 28) rather than per-command values.
//!
//! # The invariant this module enforces
//!
//! Every value that reaches a recorded command is finite and internally
//! consistent, because section 4 forbids handing a portably-detectable problem
//! to a driver: a `NaN` viewport or an inverted depth range is refused with
//! [`RhiErrorKind::InvalidUsage`] at the verb that takes it, before any backend
//! sees the command.

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};

/// A linear RGBA color, used where a command takes a color rather than a clear.
///
/// Section 30 gives this type to `set_blend_constant`; it is *not* the clear
/// value type. The two are deliberately separate — see [`ColorClearValue`].
///
/// The components are not normalized by the RHI. A value outside `0.0..=1.0` is
/// legal and means what the native API means by it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    /// Red component.
    pub r: f32,
    /// Green component.
    pub g: f32,
    /// Blue component.
    pub b: f32,
    /// Alpha component.
    pub a: f32,
}

impl Color {
    /// States all four components explicitly.
    ///
    /// Section 30 does not declare a constructor, and section 30's struct
    /// literal is legal — but a caller who writes
    /// [`Color::new(0.0, 0.0, 0.0, 0.0)`](Color::new) is stating the blend
    /// constant's default on purpose rather than by leaving fields out.
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
}

/// The value a color attachment is cleared to.
///
/// Three numeric classes rather than one `[f32; 4]`, because a clear value is
/// reinterpreted as the attachment format's bit pattern and not converted: a
/// cleared `R8G8B8A8_UINT` attachment has no float representation of its clear
/// value at all. Section 30 keeps the class in the type so that a mismatched
/// clear is a *portable* error rather than a reinterpretation the driver
/// performs silently.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorClearValue {
    /// A floating-point or normalized clear value.
    Float([f32; 4]),
    /// A signed-integer clear value.
    Sint([i32; 4]),
    /// An unsigned-integer clear value.
    Uint([u32; 4]),
}

/// The numeric class of a color format.
///
/// Section 31.1 requires a clear value's class to match the attachment format's
/// class, and a class is all the comparison needs: the four components are not
/// compared against the format, and the format is not converted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClearValueClass {
    /// Floating-point or normalized formats, cleared with
    /// [`ColorClearValue::Float`].
    Float,
    /// Signed-integer formats, cleared with [`ColorClearValue::Sint`].
    Sint,
    /// Unsigned-integer formats, cleared with [`ColorClearValue::Uint`].
    Uint,
}

impl ClearValueClass {
    /// The name used in refusal messages.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Float => "Float",
            Self::Sint => "Sint",
            Self::Uint => "Uint",
        }
    }
}

impl ColorClearValue {
    /// Which numeric class this value belongs to.
    ///
    /// Section 31.1's comparison reads this against
    /// [`crate::api::format::FormatFacts`]'s class for the attachment format.
    pub(crate) fn class(self) -> ClearValueClass {
        match self {
            Self::Float(_) => ClearValueClass::Float,
            Self::Sint(_) => ClearValueClass::Sint,
            Self::Uint(_) => ClearValueClass::Uint,
        }
    }

    /// The class name, for refusal messages.
    pub(crate) fn class_name(self) -> &'static str {
        self.class().as_str()
    }
}

/// A rectangle in framebuffer coordinates.
///
/// Unsigned, because a scissor rect has no negative extent in P0: section 30
/// notes that a negative viewport height is not used to express a Y flip, and
/// the same reasoning removes the possibility of a negative scissor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    /// Left edge.
    pub x: u32,
    /// Top edge.
    pub y: u32,
    /// Width in pixels. May be zero.
    pub width: u32,
    /// Height in pixels. May be zero.
    pub height: u32,
}

impl Rect {
    /// States the rect.
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// The inclusive-exclusive right edge, or `None` if it would overflow.
    ///
    /// The fallible form is the honest one: section 30 refuses an overflowing
    /// `x + width`, and an accessor that returned `u32` would have to pick an
    /// answer for a rect that must never be constructed.
    pub fn right(&self) -> Option<u32> {
        self.x.checked_add(self.width)
    }

    /// The inclusive-exclusive bottom edge, or `None` if it would overflow.
    pub fn bottom(&self) -> Option<u32> {
        self.y.checked_add(self.height)
    }
}

/// A viewport transform.
///
/// Depth is a range rather than a value because a native viewport maps the
/// clip-space Z range onto `[min_depth, max_depth]`; section 30 fixes the
/// portable constraint that the range is ordered, bounded, and finite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// Left edge in framebuffer coordinates.
    pub x: f32,
    /// Top edge in framebuffer coordinates.
    pub y: f32,
    /// Width in pixels. Must be non-negative and finite.
    pub width: f32,
    /// Height in pixels. Must be non-negative and finite.
    pub height: f32,
    /// The depth the near plane maps to.
    pub min_depth: f32,
    /// The depth the far plane maps to.
    pub max_depth: f32,
}

impl Viewport {
    /// States the viewport.
    pub fn new(x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        }
    }
}

/// What an attachment's contents are set to when a scope begins.
///
/// Generic over the clear value because depth, stencil, and color clears are
/// different types: [`f32`], [`u32`], and [`ColorClearValue`]. One generic
/// `LoadOp` keeps "load or clear" stated once instead of three times, and lets
/// [`crate::api::command::attachment::DepthAttachmentMode`] fix the value type
/// for depth while the color path fixes it for color.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadOp<T> {
    /// Keep the attachment's existing contents.
    Load,
    /// Set the attachment's contents to `T` first.
    Clear(T),
}

impl<T> LoadOp<T> {
    /// Whether this operation discards the attachment's previous contents.
    ///
    /// Read by the actual-use mapping: section 32.4 turns `Clear` into a
    /// scope-begin *write* and `Load` into a scope-begin *read*, and the
    /// distinction decides whether a scope with no draw still declares a write.
    pub(crate) fn is_clear(&self) -> bool {
        match self {
            Self::Load => false,
            Self::Clear(_) => true,
        }
    }

    /// The clear value, if this is a clear.
    pub(crate) fn clear_value(&self) -> Option<&T> {
        match self {
            Self::Load => None,
            Self::Clear(value) => Some(value),
        }
    }
}

/// What happens to an attachment's contents when a scope ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreOp {
    /// The contents remain defined and readable.
    Store,
    /// The contents become undefined after the scope ends.
    ///
    /// Section 32.4 states the consequence that matters to the graph: a
    /// `Discard` attachment's results are not a hazard for a later pass to
    /// wait on, because there are no results to wait for.
    Discard,
}

/// Refuses a rect whose right or bottom edge overflows.
///
/// Section 30's rule, and the only rule it states for a rect: a zero width or
/// height is legal and means an empty region, while an overflowing edge means
/// the rect does not name a region of any framebuffer.
pub(crate) fn validate_rect(rect: Rect, what: &'static str) -> RhiResult<()> {
    if rect.right().is_none() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("{} has x + width overflowing u32", what),
        ));
    }
    if rect.bottom().is_none() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("{} has y + height overflowing u32", what),
        ));
    }
    Ok(())
}

/// Refuses a viewport that is not finite, not non-negative, or not ordered.
///
/// Section 30's four rules, checked in the order it states them so that the
/// message names the first thing wrong with the value.
pub(crate) fn validate_viewport(viewport: Viewport) -> RhiResult<()> {
    for (name, value) in [
        ("x", viewport.x),
        ("y", viewport.y),
        ("width", viewport.width),
        ("height", viewport.height),
        ("min_depth", viewport.min_depth),
        ("max_depth", viewport.max_depth),
    ] {
        if !value.is_finite() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("the viewport's {} is not finite", name),
            ));
        }
    }
    if viewport.width < 0.0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the viewport's width is negative",
        ));
    }
    if viewport.height < 0.0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the viewport's height is negative",
        ));
    }
    if viewport.min_depth < 0.0 || viewport.min_depth > 1.0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the viewport's min_depth is outside 0.0..=1.0",
        ));
    }
    if viewport.max_depth < 0.0 || viewport.max_depth > 1.0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the viewport's max_depth is outside 0.0..=1.0",
        ));
    }
    if viewport.min_depth > viewport.max_depth {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the viewport's min_depth is greater than its max_depth",
        ));
    }
    Ok(())
}
