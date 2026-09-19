//! Aspects, subresource ranges, origins, and host texel layout (specification
//! sections 14.1 through 14.5).
//!
//! This module owns the vocabulary that says *which part* of a texture an
//! operation touches, and in which byte layout a caller's CPU bytes are stored.
//!
//! # Why tracking and copy use different types
//!
//! Section 14 opens by requiring two shapes rather than one, and the split is
//! load-bearing:
//!
//! ```text
//! TextureSubresourceRange    view / hazard tracking: many mips x many layers
//! TextureSubresourceLayers   copy / upload / readback: one mip x a layer range
//! ```
//!
//! A hazard range and a copy region are not the same set even when they name the
//! same texels: a copy touches one mip level per operation because a native copy
//! region has one, while a tracked hazard may span the whole mip chain. One
//! shared type would have to be legal in the union of both situations, which
//! means it would accept states neither native API can express.
//!
//! # Array layers and Z slices are not the same axis
//!
//! Section 14.4 forbids folding them into one `depth_or_layers` field, and the
//! rules say why: a 2D array layer is addressed by
//! [`TextureSubresourceLayers`], while a 3D texture's Z slice is addressed by
//! [`Origin3d`] plus [`Extent3d`], and a 3D texture's layer is pinned to `0/1`.
//! A single fused field would make "layer 3 of a 3D texture" expressible, and it
//! is not a thing.
//!
//! # What this module does not own
//!
//! - Whether the *format* of a given aspect may be created, sampled, or attached
//!   is [`crate::api::format::FormatFacts`].
//! - Whether the GPU-side copy route exists, and what alignment the native copy
//!   buffer requires, is [`crate::api::resource::route`]. The distinction is
//!   section 14.5's whole point: [`HostTexelLayout`] describes *caller bytes*,
//!   and `RouteSupport::texel_copy_layout` describes the *GPU route*. They are
//!   deliberately not the same numbers, and the upload path may repack one into
//!   the other without telling the caller.

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::{TextureFormat, logical_bytes_per_block};
use crate::api::resource::texture::{Extent3d, TextureDimension};

/// One single aspect of a texture.
///
/// A single aspect, not a set: section 14.3 requires each copy-side value to
/// select exactly one, because a depth/stencil copy is queried and encoded as
/// two routes rather than one. Sets use [`TextureAspects`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureAspect {
    /// Color planes.
    Color,
    /// Depth plane.
    Depth,
    /// Stencil plane.
    Stencil,
}

/// A set of texture aspects.
///
/// Unlike [`TextureAspect`], this *is* a set, because a view may legitimately
/// cover both planes of a depth-stencil texture (section 15.3 allows a
/// depth/stencil attachment view to select depth, stencil, or both).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureAspects(u8);

impl TextureAspects {
    /// The color aspect.
    pub const COLOR: Self = Self(1 << 0);
    /// The depth aspect.
    pub const DEPTH: Self = Self(1 << 1);
    /// The stencil aspect.
    pub const STENCIL: Self = Self(1 << 2);

    /// Whether every bit set in `other` is set in `self`.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two aspect sets.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether no aspect bit is set.
    ///
    /// An empty aspect set names no part of a texture, so every validation that
    /// accepts one of these refuses it first.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// View / hazard tracking: may cover multiple mips + multiple array layers.
///
/// The tracking-side shape. Section 14.2 pins a 3D texture to `base_layer = 0`
/// and `layer_count = 1`, because a Z slice of a 3D texture is not an
/// independent array subresource — its Z range is expressed by a copy origin and
/// extent instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureSubresourceRange {
    /// Which aspects the range covers. Must not be empty.
    pub aspects: TextureAspects,
    /// First mip level covered.
    pub base_mip: u32,
    /// Number of mip levels covered. Must be at least 1.
    pub mip_count: u32,
    /// First array layer covered. Must be 0 for a 3D texture.
    pub base_layer: u32,
    /// Number of array layers covered. Must be 1 for a 3D texture.
    pub layer_count: u32,
}

/// One mip + a range of array layers.
///
/// Semantically analogous to Vulkan ImageSubresourceLayers, but not a native struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureSubresourceLayers {
    /// The single aspect copied. Depth and stencil are never combined here.
    pub aspect: TextureAspect,
    /// The mip level copied. One per operation, because a native copy region
    /// addresses one level.
    pub mip_level: u32,
    /// First array layer copied. Must be 0 for 1D and 3D textures.
    pub base_layer: u32,
    /// Number of array layers copied. Must be 1 for 1D and 3D textures.
    pub layer_count: u32,
}

/// A texel coordinate at which a copy starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Origin3d {
    /// X coordinate.
    pub x: u32,
    /// Y coordinate. Must be 0 for a 1D texture.
    pub y: u32,
    /// Z coordinate. Must be 0 except for a 3D texture, where it selects the
    /// first Z slice.
    pub z: u32,
}

/// The byte layout of a caller's CPU source bytes.
///
/// An upload source is ordinary CPU bytes; its layout **is not the GPU
/// copy-buffer layout requirement**. Section 14.5 keeps the two apart so that a
/// native 256-byte row pitch is never imposed on an asset loader, and so that
/// the RHI may repack a normal CPU layout into private staging without that
/// repack counting as a hidden fallback of a copy command.
///
/// The rules this must satisfy, and the ones it must **not** be asked to
/// satisfy, are enforced by `validate_host_texel_layout`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostTexelLayout {
    /// Byte distance between starts of adjacent rows in the CPU source.
    pub bytes_per_row: u32,

    /// Number of rows between starts of adjacent images/layers/depth slices in the CPU source.
    pub rows_per_image: u32,
}

/// The single-aspect set a single aspect denotes.
///
/// The bridge between the copy-side [`TextureAspect`] and the set-valued
/// [`TextureAspects`] that a view or a hazard range uses. Kept as the one
/// conversion so that "is this aspect one the format has" has a single
/// implementation.
pub(crate) fn aspect_bits(aspect: TextureAspect) -> TextureAspects {
    match aspect {
        TextureAspect::Color => TextureAspects::COLOR,
        TextureAspect::Depth => TextureAspects::DEPTH,
        TextureAspect::Stencil => TextureAspects::STENCIL,
    }
}

/// Checks a tracking range against the texture it refers to.
///
/// Section 14.2's pinned 3D case:
///
/// ```text
/// base_layer = 0
/// layer_count = 1
/// ```
///
/// Plus the two rules that make a range name something: at least one mip, at
/// least one layer, and at least one aspect.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the view and hazard-tracking paths call this once api::binding is declared"
    )
)]
pub(crate) fn validate_subresource_range(
    range: TextureSubresourceRange,
    dimension: TextureDimension,
) -> RhiResult<()> {
    if range.aspects.is_empty() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a subresource range must select at least one aspect",
        ));
    }
    if range.mip_count == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a subresource range must cover at least one mip level",
        ));
    }
    if range.layer_count == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a subresource range must cover at least one array layer",
        ));
    }
    if dimension == TextureDimension::D3 && (range.base_layer != 0 || range.layer_count != 1) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a 3D texture Z slice is not an array subresource, so a subresource \
             range over one must use base_layer = 0 and layer_count = 1",
        ));
    }
    Ok(())
}

/// Checks a copy-side subresource against the texture it refers to.
///
/// Section 14.3's frozen rules:
///
/// ```text
/// Each layers value may select only one aspect
///
/// D1: base_layer = 0, layer_count = 1
/// D2: base_layer/layer_count select array layers
/// D3: base_layer = 0, layer_count = 1
/// ```
///
/// "Only one aspect" is enforced structurally: [`TextureSubresourceLayers`]
/// carries a single [`TextureAspect`], not a set, so a `Depth | Stencil` copy
/// cannot be spelled. That is why this function has no rule for it — the type
/// makes the state unconstructible rather than refusing it later, which section
/// 4 prefers.
pub(crate) fn validate_subresource_layers(
    layers: TextureSubresourceLayers,
    dimension: TextureDimension,
) -> RhiResult<()> {
    if layers.layer_count == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a subresource layer range must cover at least one array layer",
        ));
    }
    match dimension {
        TextureDimension::D1 | TextureDimension::D3 => {
            if layers.base_layer != 0 || layers.layer_count != 1 {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "a {:?} texture has no array layers, so a copy must use \
                         base_layer = 0 and layer_count = 1, not {} and {}",
                        dimension, layers.base_layer, layers.layer_count
                    ),
                ));
            }
        }
        TextureDimension::D2 => {}
    }
    Ok(())
}

/// Checks an origin/extent pair against the texture it refers to.
///
/// Section 14.4:
///
/// ```text
/// D1:       origin.y = origin.z = 0, extent.height = extent.depth = 1
/// D2:       origin.z = 0, extent.depth = 1
///           (the array range is expressed by TextureSubresourceLayers)
/// D3:       the array layer is fixed at 0/1, and
///           origin.z + extent.depth express the Z slice range
/// ```
///
/// The extent check runs for every dimension: a copy of zero texels is not a
/// copy, and refusing it here keeps it from reaching a native API that would
/// answer differently on each backend.
pub(crate) fn validate_origin_extent(
    origin: Origin3d,
    extent: Extent3d,
    dimension: TextureDimension,
) -> RhiResult<()> {
    if extent.width == 0 || extent.height == 0 || extent.depth == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "copy extent {}x{}x{} must have every component greater than zero",
                extent.width, extent.height, extent.depth
            ),
        ));
    }

    match dimension {
        TextureDimension::D1 => {
            if origin.y != 0 || origin.z != 0 {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a 1D texture has no Y or Z axis, so origin.y and origin.z must be 0",
                ));
            }
            if extent.height != 1 || extent.depth != 1 {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "a 1D copy must have extent.height = extent.depth = 1, not {} and {}",
                        extent.height, extent.depth
                    ),
                ));
            }
        }
        TextureDimension::D2 => {
            if origin.z != 0 {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a 2D texture addressed by layer rather than by Z, so origin.z must be 0",
                ));
            }
            if extent.depth != 1 {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "a 2D copy must have extent.depth = 1, not {}; the array range is \
                         expressed by the subresource instead",
                        extent.depth
                    ),
                ));
            }
        }
        TextureDimension::D3 => {}
    }
    Ok(())
}

/// Checks a caller's CPU layout against the texels it must cover.
///
/// Section 14.5's four rules, minus the one that needs the byte count:
///
/// ```text
/// bytes_per_row >= logical row bytes
/// bytes_per_row is aligned to format block bytes
/// rows_per_image >= logical block-row count
/// the source byte range covers the final copied texel/block
/// ```
///
/// The last rule needs `bytes.len()`, so it belongs to upload validation
/// ([`crate::api::resource::transfer`]) and is checked there through
/// [`source_bytes_required`].
///
/// The three byte-based rules are skipped when
/// [`logical_bytes_per_block`] returns `None` — that is, for the depth formats
/// whose entry size the format name does not fix. A portable layer that refused
/// a layout for a backend-chosen block size would be inventing a constraint, the
/// same mistake as reporting a byte count it cannot know.
///
/// Note what is *absent*: there is no check against 256-byte row pitch or any
/// other native copy alignment. Section 14.5 assigns that number to the GPU
/// route, and an asset loader's bytes must not be held to it.
pub(crate) fn validate_host_texel_layout(
    layout: HostTexelLayout,
    extent: Extent3d,
    format: TextureFormat,
) -> RhiResult<()> {
    if layout.rows_per_image < extent.height {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "host rows_per_image {} is smaller than the {} block rows of the copied \
                 extent",
                layout.rows_per_image, extent.height
            ),
        ));
    }

    if let Some(block_bytes) = logical_bytes_per_block(format) {
        let logical_row = u64::from(extent.width) * u64::from(block_bytes);
        if u64::from(layout.bytes_per_row) < logical_row {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "host bytes_per_row {} is smaller than the {logical_row} bytes of one \
                     row of {:?}",
                    layout.bytes_per_row, format
                ),
            ));
        }
        if !layout.bytes_per_row.is_multiple_of(block_bytes) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "host bytes_per_row {} is not a multiple of the {block_bytes}-byte \
                     block of {:?}",
                    layout.bytes_per_row, format
                ),
            ));
        }
    }
    Ok(())
}

/// The number of source bytes a texture upload must provide.
///
/// Section 14.5's last rule: "the source byte range covers the final copied
/// texel/block". The last copied row starts after every preceding row — with
/// `rows_per_image` separating images (array layers or depth slices) and
/// `bytes_per_row` separating rows — and must be followed by one full logical
/// row of texels.
///
/// `None` when the format's entry size is not fixed by its name, for the reason
/// given on [`validate_host_texel_layout`].
pub(crate) fn source_bytes_required(
    layout: HostTexelLayout,
    extent: Extent3d,
    image_count: u32,
    format: TextureFormat,
) -> Option<u64> {
    let block_bytes = u64::from(logical_bytes_per_block(format)?);
    let logical_row = u64::from(extent.width) * block_bytes;
    let rows_per_image = u64::from(layout.rows_per_image);
    let bytes_per_row = u64::from(layout.bytes_per_row);

    let images_before_last = u64::from(image_count.saturating_sub(1));
    let rows_before_last = u64::from(extent.height.saturating_sub(1));

    Some(
        images_before_last * rows_per_image * bytes_per_row
            + rows_before_last * bytes_per_row
            + logical_row,
    )
}
