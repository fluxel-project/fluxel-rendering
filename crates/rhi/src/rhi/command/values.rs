//! Portable command values: geometry, load/store, and copy descriptions.
//!
//! This module owns rhi-design section 30 and the copy descriptors from
//! section 34.
//!
//! # What it is
//!
//! Plain value types a recorder stores. They describe regions in Fluxel's own
//! terms, never in barrier or native-resource terms: there is no layout, no
//! transition, and no native subresource index anywhere in this module.
//!
//! # What it deliberately does not own
//!
//! A negative viewport height is not used to express a Y flip. Coordinate-system
//! adaptation is a toolchain, shader, and backend convention; freezing it as a
//! viewport sign would make the same viewport mean two different things.

use std::ops::Range;

pub use super::super::format::BlitFilter;
pub use super::super::pipeline::IndexFormat;
use super::super::platform::{RhiError, RhiErrorKind, RhiResult};
use super::super::resource::{
    Buffer, Extent3d, Origin3d, Texture, TextureSubresourceLayers,
};

/// A linear RGBA color.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    /// The red component.
    pub r: f32,
    /// The green component.
    pub g: f32,
    /// The blue component.
    pub b: f32,
    /// The alpha component.
    pub a: f32,
}

impl Default for Color {
    /// Transparent black, which is the identity for a blend constant: an unset
    /// constant must not tint a draw that never asked for one.
    fn default() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        }
    }
}

/// A clear value whose numeric class must match the attachment format.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorClearValue {
    /// A floating-point or normalized attachment clear.
    Float([f32; 4]),
    /// A signed-integer attachment clear.
    Sint([i32; 4]),
    /// An unsigned-integer attachment clear.
    Uint([u32; 4]),
}

impl ColorClearValue {
    /// The class this value belongs to.
    pub fn class(&self) -> ClearValueClass {
        match self {
            Self::Float(_) => ClearValueClass::Float,
            Self::Sint(_) => ClearValueClass::Sint,
            Self::Uint(_) => ClearValueClass::Uint,
        }
    }
}

/// The numeric class a clear value must agree with.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClearValueClass {
    /// Floating-point or normalized formats.
    Float,
    /// Signed-integer formats.
    Sint,
    /// Unsigned-integer formats.
    Uint,
}

/// A rectangle in framebuffer coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rect {
    /// The left edge.
    pub x: u32,
    /// The top edge.
    pub y: u32,
    /// The width. Zero is legal.
    pub width: u32,
    /// The height. Zero is legal.
    pub height: u32,
}

impl Rect {
    /// The right edge, or an error when the sum overflows.
    pub fn right(&self) -> RhiResult<u32> {
        self.x.checked_add(self.width).ok_or_else(|| {
            RhiError::new(RhiErrorKind::InvalidUsage, "scissor rectangle overflows")
        })
    }

    /// The bottom edge, or an error when the sum overflows.
    pub fn bottom(&self) -> RhiResult<u32> {
        self.y.checked_add(self.height).ok_or_else(|| {
            RhiError::new(RhiErrorKind::InvalidUsage, "scissor rectangle overflows")
        })
    }
}

/// A viewport in framebuffer coordinates with a depth range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// The left edge.
    pub x: f32,
    /// The top edge.
    pub y: f32,
    /// The width.
    pub width: f32,
    /// The height.
    pub height: f32,
    /// The near depth.
    pub min_depth: f32,
    /// The far depth.
    pub max_depth: f32,
}

impl Viewport {
    /// Validates that every component is finite and the depth range is legal.
    pub fn validate(&self) -> RhiResult<()> {
        if !self.x.is_finite()
            || !self.y.is_finite()
            || !self.width.is_finite()
            || !self.height.is_finite()
            || !self.min_depth.is_finite()
            || !self.max_depth.is_finite()
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "viewport components must be finite",
            ));
        }
        if self.width < 0.0 || self.height < 0.0 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "viewport width and height must not be negative",
            ));
        }
        if !(0.0..=1.0).contains(&self.min_depth) || !(0.0..=1.0).contains(&self.max_depth) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "viewport depth bounds must lie in 0.0..=1.0",
            ));
        }
        if self.min_depth > self.max_depth {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "viewport minimum depth must not exceed maximum depth",
            ));
        }
        Ok(())
    }
}

/// What an attachment does with its contents when a scope begins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadOp<T> {
    /// Preserve the existing contents.
    Load,
    /// Replace the contents with the given value.
    Clear(T),
}

/// What an attachment does with its contents when a scope ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StoreOp {
    /// Keep the contents.
    Store,
    /// Leave the contents undefined.
    Discard,
}

/// A buffer-to-buffer copy.
#[derive(Clone, Debug)]
pub struct BufferCopy {
    /// The source buffer.
    pub src: Buffer,
    /// The source byte offset.
    pub src_offset: u64,
    /// The destination buffer.
    pub dst: Buffer,
    /// The destination byte offset.
    pub dst_offset: u64,
    /// The number of bytes to copy.
    pub size: u64,
}

/// A buffer-to-texture or texture-to-buffer copy.
///
/// One type serves both directions because the fields are the same; the
/// direction is chosen by which recorder method is called.
#[derive(Clone, Debug)]
pub struct BufferTextureCopy {
    /// The buffer side of the copy.
    pub buffer: Buffer,
    /// The buffer byte offset.
    pub buffer_offset: u64,
    /// The byte distance between rows in the buffer.
    pub bytes_per_row: u32,
    /// The number of rows between depth slices in the buffer.
    pub rows_per_image: u32,

    /// The texture side of the copy.
    pub texture: Texture,
    /// The texture subresource on the texture side.
    pub texture_subresource: TextureSubresourceLayers,
    /// The texel origin on the texture side.
    pub texture_origin: Origin3d,
    /// The texel extent of the copy.
    pub extent: Extent3d,
}

/// A texture-to-texture copy.
#[derive(Clone, Debug)]
pub struct TextureCopy {
    /// The source texture.
    pub src: Texture,
    /// The source subresource.
    pub src_subresource: TextureSubresourceLayers,
    /// The source texel origin.
    pub src_origin: Origin3d,
    /// The destination texture.
    pub dst: Texture,
    /// The destination subresource.
    pub dst_subresource: TextureSubresourceLayers,
    /// The destination texel origin.
    pub dst_origin: Origin3d,
    /// The texel extent of the copy.
    pub extent: Extent3d,
}

/// A multisample-to-single-sample resolve.
#[derive(Clone, Debug)]
pub struct TextureResolve {
    /// The multisampled source texture.
    pub src: Texture,
    /// The source subresource.
    pub src_subresource: TextureSubresourceLayers,
    /// The source texel origin.
    pub src_origin: Origin3d,
    /// The single-sample destination texture.
    pub dst: Texture,
    /// The destination subresource.
    pub dst_subresource: TextureSubresourceLayers,
    /// The destination texel origin.
    pub dst_origin: Origin3d,
    /// The texel extent of the resolve.
    pub extent: Extent3d,
}

/// A filtered or unfiltered texture blit.
#[derive(Clone, Debug)]
pub struct TextureBlit {
    /// The source texture.
    pub src: Texture,
    /// The source subresource.
    pub src_subresource: TextureSubresourceLayers,
    /// The source texel origin.
    pub src_origin: Origin3d,
    /// The source texel extent.
    pub src_extent: Extent3d,

    /// The destination texture.
    pub dst: Texture,
    /// The destination subresource.
    pub dst_subresource: TextureSubresourceLayers,
    /// The destination texel origin.
    pub dst_origin: Origin3d,
    /// The destination texel extent.
    pub dst_extent: Extent3d,

    /// The sampling filter.
    pub filter: BlitFilter,
}

/// Whether two byte ranges in the same buffer overlap.
pub(crate) fn byte_ranges_overlap(
    left_offset: u64,
    left_size: u64,
    right_offset: u64,
    right_size: u64,
) -> bool {
    let Some(left_end) = left_offset.checked_add(left_size) else {
        return true;
    };
    let Some(right_end) = right_offset.checked_add(right_size) else {
        return true;
    };
    left_offset < right_end && right_offset < left_end
}

/// Whether two texel boxes overlap in every shared dimension.
pub(crate) fn texel_regions_overlap(
    left_origin: Origin3d,
    left_extent: Extent3d,
    right_origin: Origin3d,
    right_extent: Extent3d,
) -> bool {
    let axis = |left_start: u32, left_len: u32, right_start: u32, right_len: u32| {
        let Some(left_end) = left_start.checked_add(left_len) else {
            return true;
        };
        let Some(right_end) = right_start.checked_add(right_len) else {
            return true;
        };
        left_start < right_end && right_start < left_end
    };
    axis(left_origin.x, left_extent.width, right_origin.x, right_extent.width)
        && axis(
            left_origin.y,
            left_extent.height,
            right_origin.y,
            right_extent.height,
        )
        && axis(
            left_origin.z,
            left_extent.depth,
            right_origin.z,
            right_extent.depth,
        )
}

/// The vertex range and instance range of a non-indexed draw.
pub(crate) fn validate_draw_ranges(
    vertices: &Range<u32>,
    instances: &Range<u32>,
) -> RhiResult<()> {
    if vertices.start > vertices.end {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "draw vertex range is inverted",
        ));
    }
    if instances.start > instances.end {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "draw instance range is inverted",
        ));
    }
    Ok(())
}

/// The index range, base vertex, and instance range of an indexed draw.
pub(crate) fn validate_indexed_draw_ranges(
    indices: &Range<u32>,
    instances: &Range<u32>,
) -> RhiResult<()> {
    if indices.start > indices.end {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "draw index range is inverted",
        ));
    }
    validate_draw_ranges(&(0..1), instances)
}
