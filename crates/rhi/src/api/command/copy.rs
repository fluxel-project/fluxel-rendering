//! Copy, resolve, and blit descriptors and their portable validation
//! (specification sections 34.1 through 34.5), plus the upload and readback
//! encoding rules (section 35).
//!
//! This module owns the six copy-family descriptors, the blit filter, and every
//! check section 34 states that is decidable **without a device**. The verbs
//! themselves are on [`crate::api::command::CommandRecorder`], because section
//! 34.6 records them and only a recorder knows whether it is open.
//!
//! # What is checked here, and what is not
//!
//! Each descriptor has two halves of its rule list:
//!
//! ```text
//! portable   usage bits, identity, ranges, extents, subresource agreement,
//!            same-texture overlap, region-vs-buffer footprint
//! device     RouteQuery::... Support, BufferCopyLayoutLimits /
//!            TexelCopyLayoutLimits alignment, "does this route exist at all"
//! ```
//!
//! The device half is not checked here, and cannot be: a route answer and a
//! native alignment come from a device that has not been created. Section 9.4
//! forbids the backend from quietly substituting a shader or CPU fallback for an
//! unsupported route, so the refusal has to come back as
//! [`crate::api::error::RhiErrorKind::Unsupported`] from the verb — which is why
//! the copy verbs are in the unbuilt tier and the checks below are not.
//!
//! # What this module does not own
//!
//! - Job-level upload and readback validation, which is
//!   [`crate::api::resource::transfer`]'s, performed when a device creates the
//!   job. What section 35 adds is the *actual use* a recorded job produces, and
//!   that mapping lives in [`crate::api::command`].
//! - Host texel layout. Section 34.2 is emphatic that a copy buffer's layout is
//!   GPU copy-buffer layout and not [`crate::api::resource::subresource::HostTexelLayout`];
//!   the upload path may repack one into the other without telling the caller.
//!
//! # The invariant this module enforces
//!
//! **A copy names a real region on both sides, and never aliases itself.** A
//! copy whose source and destination overlap in the same texture is refused
//! rather than lowered, because P0 defines no memmove (section 34.1): a native
//! copy command given overlapping regions has undefined results, and section 4
//! forbids letting a driver discover that.

use crate::api::command::require_device;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::{TextureFormat, logical_bytes_per_block};
use crate::api::identity::DeviceIdentity;
use crate::api::resource::buffer::{Buffer, BufferRange, BufferUsage, validate_buffer_range};
use crate::api::resource::route::RouteQuery;
use crate::api::resource::subresource::{
    Origin3d, TextureAspect, TextureSubresourceLayers, validate_origin_extent,
    validate_subresource_layers,
};
use crate::api::resource::texture::{Extent3d, Texture, TextureDimension, TextureUsage};

/// A byte-range copy between two buffers.
///
/// The two offsets are separate fields rather than one `BufferRange`, because the
/// source and destination ranges live in different buffers and a single range
/// would have to be validated against both.
#[derive(Clone)]
pub struct BufferCopy {
    /// The buffer read from.
    pub src: Buffer,
    /// The byte offset read from.
    pub src_offset: u64,
    /// The buffer written to.
    pub dst: Buffer,
    /// The byte offset written to.
    pub dst_offset: u64,
    /// How many bytes are copied.
    pub size: u64,
}

/// A copy between a buffer and a texture, or a texture and a buffer.
///
/// One descriptor for both directions, as section 34.2 writes it: the fields are
/// the same and only the direction differs, so two types would be two names for
/// one shape. The direction is stated by which verb is called —
/// `copy_buffer_to_texture` or `copy_texture_to_buffer` — and by the usage bits
/// the two resources carry.
#[derive(Clone)]
pub struct BufferTextureCopy {
    /// The buffer side.
    pub buffer: Buffer,
    /// The byte offset of the buffer region.
    pub buffer_offset: u64,
    /// The byte pitch between rows of the copied image.
    pub bytes_per_row: u32,
    /// The number of rows in one image of the copied region.
    pub rows_per_image: u32,

    /// The texture side.
    pub texture: Texture,
    /// Which mip and layers of the texture are copied.
    pub texture_subresource: TextureSubresourceLayers,
    /// Where in the texture the region begins.
    pub texture_origin: Origin3d,
    /// How much of the texture is copied.
    pub extent: Extent3d,
}

/// A texel-region copy between two textures.
#[derive(Clone)]
pub struct TextureCopy {
    /// The texture read from.
    pub src: Texture,
    /// Which mip and layers are read.
    pub src_subresource: TextureSubresourceLayers,
    /// Where the source region begins.
    pub src_origin: Origin3d,

    /// The texture written to.
    pub dst: Texture,
    /// Which mip and layers are written.
    pub dst_subresource: TextureSubresourceLayers,
    /// Where the destination region begins.
    pub dst_origin: Origin3d,

    /// How much of each texture is copied.
    pub extent: Extent3d,
}

/// A multisampled-to-single-sampled resolve.
///
/// Deliberately the same field list as [`TextureCopy`] rather than a shared type
/// with a flag: section 34.4's rules are *not* section 34.3's — a resolve
/// requires a multisampled source, a single-sampled destination, and a matching
/// format, while a copy requires neither — so one type with a mode would have to
/// accept the union of both descriptors.
#[derive(Clone)]
pub struct TextureResolve {
    /// The multisampled texture read from.
    pub src: Texture,
    /// Which mip and layers are resolved.
    pub src_subresource: TextureSubresourceLayers,
    /// Where the source region begins.
    pub src_origin: Origin3d,

    /// The single-sampled texture written to.
    pub dst: Texture,
    /// Which mip and layers receive the result.
    pub dst_subresource: TextureSubresourceLayers,
    /// Where the destination region begins.
    pub dst_origin: Origin3d,

    /// How much of each texture is resolved.
    pub extent: Extent3d,
}

/// How a blit samples when the source and destination regions differ in size.
///
/// `Linear` is a question rather than a promise: section 34.5 requires the route
/// to say whether it is supported, because a blit with filtering is a different
/// native operation from a scaled copy on some backends and absent on others.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlitFilter {
    /// Each destination texel takes its nearest source texel.
    Nearest,
    /// Each destination texel is a weighted average of source texels.
    Linear,
}

/// A filtered blit between two single-sampled color textures.
///
/// Separate source and destination extents, because scaling is what makes this a
/// blit: a 1:1 region transfer is [`TextureCopy`].
#[derive(Clone)]
pub struct TextureBlit {
    /// The texture read from.
    pub src: Texture,
    /// Which mip and layers are read.
    pub src_subresource: TextureSubresourceLayers,
    /// Where the source region begins.
    pub src_origin: Origin3d,
    /// How much of the source is read.
    pub src_extent: Extent3d,

    /// The texture written to.
    pub dst: Texture,
    /// Which mip and layers are written.
    pub dst_subresource: TextureSubresourceLayers,
    /// Where the destination region begins.
    pub dst_origin: Origin3d,
    /// How much of the destination is written.
    pub dst_extent: Extent3d,

    /// How the source is sampled.
    pub filter: BlitFilter,
}

/// The four facts a copy route key is built from.
///
/// A struct rather than four arguments, because every route key in this module
/// needs exactly these and the call sites read better with them named.
#[derive(Clone, Copy)]
struct TextureCopyShape {
    /// Dimensionality.
    dimension: TextureDimension,
    /// Format.
    format: TextureFormat,
    /// Aspect.
    aspect: TextureAspect,
    /// Sample count.
    sample_count: u32,
}

impl TextureCopyShape {
    /// Reads the four facts off a texture and one of its subresources.
    fn of(texture: &Texture, layers: &TextureSubresourceLayers) -> Self {
        Self {
            dimension: texture.descriptor().dimension,
            format: texture.descriptor().format,
            aspect: layers.aspect,
            sample_count: texture.descriptor().sample_count,
        }
    }
}

/// Refuses a buffer without the usage bit a copy needs.
fn require_buffer_usage(buffer: &Buffer, usage: BufferUsage, what: &'static str) -> RhiResult<()> {
    if !buffer.descriptor().usage.contains(usage) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("{} was not created with {} usage", what, usage_name(usage)),
        ));
    }
    Ok(())
}

/// Refuses a texture without the usage bit a copy needs.
fn require_texture_usage(
    texture: &Texture,
    usage: TextureUsage,
    what: &'static str,
) -> RhiResult<()> {
    if !texture.descriptor().usage.contains(usage) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "{} was not created with {} usage",
                what,
                texture_usage_name(usage)
            ),
        ));
    }
    Ok(())
}

/// The name of one buffer usage bit, for a refusal message.
///
/// A chain of comparisons rather than a `match`, because a usage set may carry
/// several bits at once: a `match` on the six single-bit constants would need a
/// catch-all arm for `VERTEX | INDEX`, and a catch-all arm on this crate's own
/// type is exactly what the house style forbids. The comparison is against one
/// bit at a time, so every arm is a real single-bit answer.
fn usage_name(usage: BufferUsage) -> &'static str {
    if usage == BufferUsage::COPY_SRC {
        "COPY_SRC"
    } else if usage == BufferUsage::COPY_DST {
        "COPY_DST"
    } else if usage == BufferUsage::VERTEX {
        "VERTEX"
    } else if usage == BufferUsage::INDEX {
        "INDEX"
    } else if usage == BufferUsage::UNIFORM {
        "UNIFORM"
    } else {
        "STORAGE"
    }
}

/// The name of one texture usage bit, for a refusal message.
///
/// The same shape as [`usage_name`], for the same reason: one bit at a time, and
/// no catch-all arm.
fn texture_usage_name(usage: TextureUsage) -> &'static str {
    if usage == TextureUsage::COPY_SRC {
        "COPY_SRC"
    } else if usage == TextureUsage::COPY_DST {
        "COPY_DST"
    } else if usage == TextureUsage::SAMPLED {
        "SAMPLED"
    } else if usage == TextureUsage::STORAGE {
        "STORAGE"
    } else if usage == TextureUsage::COLOR_ATTACHMENT {
        "COLOR_ATTACHMENT"
    } else {
        "DEPTH_STENCIL_ATTACHMENT"
    }
}

/// Whether two byte ranges in one buffer overlap.
///
/// Section 34.1's rule. An empty range overlaps nothing, which cannot arise for
/// a copy because `size > 0` is checked first.
fn byte_ranges_overlap(a: u64, a_size: u64, b: u64, b_size: u64) -> bool {
    a < b + b_size && b < a + a_size
}

/// One texel region: the mip, the layers, and the box within the plane.
struct TexelRegion<'a> {
    mip_level: u32,
    layers: &'a TextureSubresourceLayers,
    origin: Origin3d,
    extent: Extent3d,
}

/// Whether two texel regions of one texture overlap.
///
/// Three conditions, all necessary: the same mip level (different mips are
/// different memory), intersecting layer ranges, and intersecting boxes. The box
/// comparison includes `z`, which is what makes a 3D texture's overlapping slices
/// detectable even though a 3D copy's layer range is pinned to one layer.
fn texel_regions_overlap(a: &TexelRegion<'_>, b: &TexelRegion<'_>) -> bool {
    if a.mip_level != b.mip_level {
        return false;
    }
    let a_layers = a.layers.base_layer..a.layers.base_layer.saturating_add(a.layers.layer_count);
    let b_layers = b.layers.base_layer..b.layers.base_layer.saturating_add(b.layers.layer_count);
    if a_layers.start >= b_layers.end || b_layers.start >= a_layers.end {
        return false;
    }
    let overlaps_axis = |a_origin: u32, a_len: u32, b_origin: u32, b_len: u32| {
        let a_end = a_origin as u64 + a_len as u64;
        let b_end = b_origin as u64 + b_len as u64;
        (a_origin as u64) < b_end && (b_origin as u64) < a_end
    };
    overlaps_axis(a.origin.x, a.extent.width, b.origin.x, b.extent.width)
        && overlaps_axis(a.origin.y, a.extent.height, b.origin.y, b.extent.height)
        && overlaps_axis(a.origin.z, a.extent.depth, b.origin.z, b.extent.depth)
}

/// Checks a buffer-to-buffer copy against section 34.1's portable half.
///
/// The route question and the copy-layout alignment are not checked here; see
/// the module documentation.
pub(crate) fn validate_buffer_copy(copy: &BufferCopy, device: DeviceIdentity) -> RhiResult<()> {
    require_device(copy.src.device_identity(), device, "the copy source")?;
    require_device(copy.dst.device_identity(), device, "the copy destination")?;
    require_buffer_usage(&copy.src, BufferUsage::COPY_SRC, "the copy source")?;
    require_buffer_usage(&copy.dst, BufferUsage::COPY_DST, "the copy destination")?;

    if copy.size == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a buffer copy of zero bytes copies nothing",
        ));
    }
    validate_buffer_range(
        BufferRange::new(copy.src_offset, copy.size),
        copy.src.descriptor().size,
    )?;
    validate_buffer_range(
        BufferRange::new(copy.dst_offset, copy.size),
        copy.dst.descriptor().size,
    )?;

    if copy.src.id() == copy.dst.id()
        && byte_ranges_overlap(copy.src_offset, copy.size, copy.dst_offset, copy.size)
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a buffer copy's source and destination ranges overlap, and P0 defines no memmove",
        ));
    }

    Ok(())
}

/// Checks a buffer-to-texture or texture-to-buffer copy against section 34.2.
///
/// `to_texture` selects the direction, which decides which usage bits are
/// required and which route key is built. Everything else is direction
/// independent, which is why one function serves both verbs.
pub(crate) fn validate_buffer_texture_copy(
    copy: &BufferTextureCopy,
    device: DeviceIdentity,
    to_texture: bool,
) -> RhiResult<()> {
    require_device(copy.buffer.device_identity(), device, "the copy buffer")?;
    require_device(copy.texture.device_identity(), device, "the copy texture")?;

    if to_texture {
        require_buffer_usage(&copy.buffer, BufferUsage::COPY_SRC, "the copy buffer")?;
        require_texture_usage(&copy.texture, TextureUsage::COPY_DST, "the copy texture")?;
    } else {
        require_buffer_usage(&copy.buffer, BufferUsage::COPY_DST, "the copy buffer")?;
        require_texture_usage(&copy.texture, TextureUsage::COPY_SRC, "the copy texture")?;
    }

    if copy.texture.descriptor().sample_count != 1 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a buffer/texture copy requires a single-sampled texture; a multisampled texture is \
             resolved or copied within a texture instead",
        ));
    }
    if copy.bytes_per_row == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a buffer/texture copy needs a non-zero bytes_per_row",
        ));
    }
    if copy.rows_per_image == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a buffer/texture copy needs a non-zero rows_per_image",
        ));
    }

    let descriptor = copy.texture.descriptor();
    validate_subresource_layers(copy.texture_subresource, descriptor.dimension)?;
    validate_origin_extent(copy.texture_origin, copy.extent, descriptor.dimension)?;
    validate_texel_layout_footprint(copy, descriptor.dimension)?;

    Ok(())
}

/// Checks that a buffer/texture copy's buffer layout covers its texel region.
///
/// Section 34.2's "layout footprint sufficiently covers region", computed from
/// the descriptor alone. The three parts:
///
/// ```text
/// bytes_per_row >= one row of the region, in whole blocks
/// rows_per_image >= one image of the region, in whole blocks
/// buffer_offset + bytes_per_row * rows_per_image * images <= buffer size
/// ```
///
/// `images` is the region's depth for a 3D texture and one otherwise, because a
/// 2D array's layers are addressed by the subresource rather than by the layout.
///
/// The *alignment* half of section 34.2 — whether the route accepts this
/// `buffer_offset` and this `bytes_per_row` — is [`TexelCopyLayoutLimits`] and
/// needs a device, so it is not checked here.
///
/// One row is `extent.width * bytes_per_block`, which is exact because P0
/// declares no block-compressed format (section 8 lists none). A compressed
/// format would make the row count `ceil(width / block_width)` and the image
/// count `ceil(height / block_height)`, and that is where the change would go.
fn validate_texel_layout_footprint(
    copy: &BufferTextureCopy,
    dimension: TextureDimension,
) -> RhiResult<()> {
    let extent = copy.extent;

    if let Some(block_bytes) = logical_bytes_per_block(copy.texture.descriptor().format) {
        let minimum_row_bytes = u64::from(extent.width) * u64::from(block_bytes);
        if u64::from(copy.bytes_per_row) < minimum_row_bytes {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "a buffer/texture copy of a {}-texel-wide region needs at least {} bytes per \
                     row, but bytes_per_row is {}",
                    extent.width, minimum_row_bytes, copy.bytes_per_row
                ),
            ));
        }
        if u64::from(copy.rows_per_image) < u64::from(extent.height) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "a buffer/texture copy of a {}-texel-high region needs at least {} rows per \
                     image, but rows_per_image is {}",
                    extent.height, extent.height, copy.rows_per_image
                ),
            ));
        }
    }

    let images = match dimension {
        TextureDimension::D3 => u64::from(extent.depth),
        TextureDimension::D1 | TextureDimension::D2 => 1,
    };
    let footprint = u64::from(copy.bytes_per_row)
        .checked_mul(u64::from(copy.rows_per_image))
        .and_then(|row_bytes| row_bytes.checked_mul(images))
        .and_then(|total| total.checked_add(copy.buffer_offset));
    match footprint {
        Some(end) if end <= copy.buffer.descriptor().size => Ok(()),
        Some(end) => Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "a buffer/texture copy's layout reaches byte {} but the buffer is {} bytes",
                end,
                copy.buffer.descriptor().size
            ),
        )),
        None => Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a buffer/texture copy's layout footprint overflows u64",
        )),
    }
}

/// Checks a texture-to-texture copy against section 34.3.
pub(crate) fn validate_texture_copy(copy: &TextureCopy, device: DeviceIdentity) -> RhiResult<()> {
    require_device(copy.src.device_identity(), device, "the copy source")?;
    require_device(copy.dst.device_identity(), device, "the copy destination")?;
    require_texture_usage(&copy.src, TextureUsage::COPY_SRC, "the copy source")?;
    require_texture_usage(&copy.dst, TextureUsage::COPY_DST, "the copy destination")?;

    let src = copy.src.descriptor();
    let dst = copy.dst.descriptor();

    if src.format != dst.format {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a texture copy requires both textures to have the same format",
        ));
    }
    if copy.src_subresource.aspect != copy.dst_subresource.aspect {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a texture copy requires both sides to name the same aspect",
        ));
    }
    if copy.src_subresource.layer_count != copy.dst_subresource.layer_count {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a texture copy requires both sides to cover the same number of layers",
        ));
    }

    validate_subresource_layers(copy.src_subresource, src.dimension)?;
    validate_subresource_layers(copy.dst_subresource, dst.dimension)?;
    validate_origin_extent(copy.src_origin, copy.extent, src.dimension)?;
    validate_origin_extent(copy.dst_origin, copy.extent, dst.dimension)?;

    if copy.src.id() == copy.dst.id() {
        let a = TexelRegion {
            mip_level: copy.src_subresource.mip_level,
            layers: &copy.src_subresource,
            origin: copy.src_origin,
            extent: copy.extent,
        };
        let b = TexelRegion {
            mip_level: copy.dst_subresource.mip_level,
            layers: &copy.dst_subresource,
            origin: copy.dst_origin,
            extent: copy.extent,
        };
        if texel_regions_overlap(&a, &b) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a texture copy's source and destination regions overlap",
            ));
        }
    }

    Ok(())
}

/// Checks a direct resolve against section 34.4.
pub(crate) fn validate_texture_resolve(
    resolve: &TextureResolve,
    device: DeviceIdentity,
) -> RhiResult<()> {
    require_device(resolve.src.device_identity(), device, "the resolve source")?;
    require_device(
        resolve.dst.device_identity(),
        device,
        "the resolve destination",
    )?;
    require_texture_usage(&resolve.src, TextureUsage::COPY_SRC, "the resolve source")?;
    require_texture_usage(
        &resolve.dst,
        TextureUsage::COPY_DST,
        "the resolve destination",
    )?;

    let src = resolve.src.descriptor();
    let dst = resolve.dst.descriptor();

    if resolve.src_subresource.aspect != TextureAspect::Color
        || resolve.dst_subresource.aspect != TextureAspect::Color
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "P0 freezes a direct resolve to the Color aspect; a depth or stencil resolve is not \
             part of the frozen surface",
        ));
    }
    if src.format != dst.format {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a resolve requires both textures to have the same format",
        ));
    }
    if src.sample_count <= 1 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a resolve source must be multisampled",
        ));
    }
    if dst.sample_count != 1 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a resolve destination must be single-sampled",
        ));
    }
    if resolve.src_subresource.layer_count != resolve.dst_subresource.layer_count {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a resolve requires both sides to cover the same number of layers",
        ));
    }

    validate_subresource_layers(resolve.src_subresource, src.dimension)?;
    validate_subresource_layers(resolve.dst_subresource, dst.dimension)?;
    validate_origin_extent(resolve.src_origin, resolve.extent, src.dimension)?;
    validate_origin_extent(resolve.dst_origin, resolve.extent, dst.dimension)?;

    if resolve.src.id() == resolve.dst.id() {
        let a = TexelRegion {
            mip_level: resolve.src_subresource.mip_level,
            layers: &resolve.src_subresource,
            origin: resolve.src_origin,
            extent: resolve.extent,
        };
        let b = TexelRegion {
            mip_level: resolve.dst_subresource.mip_level,
            layers: &resolve.dst_subresource,
            origin: resolve.dst_origin,
            extent: resolve.extent,
        };
        if texel_regions_overlap(&a, &b) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a resolve's source and destination regions overlap",
            ));
        }
    }

    Ok(())
}

/// Checks a blit against section 34.5.
pub(crate) fn validate_texture_blit(blit: &TextureBlit, device: DeviceIdentity) -> RhiResult<()> {
    require_device(blit.src.device_identity(), device, "the blit source")?;
    require_device(blit.dst.device_identity(), device, "the blit destination")?;
    require_texture_usage(&blit.src, TextureUsage::COPY_SRC, "the blit source")?;
    require_texture_usage(&blit.dst, TextureUsage::COPY_DST, "the blit destination")?;

    let src = blit.src.descriptor();
    let dst = blit.dst.descriptor();

    if blit.src_subresource.aspect != TextureAspect::Color
        || blit.dst_subresource.aspect != TextureAspect::Color
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "P0 freezes a direct blit to the Color aspect",
        ));
    }
    if src.sample_count != 1 || dst.sample_count != 1 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a blit requires both textures to be single-sampled; a multisampled source is \
             resolved, not blitted",
        ));
    }
    if blit.src_subresource.layer_count != blit.dst_subresource.layer_count {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a blit requires both sides to cover the same number of layers",
        ));
    }

    validate_subresource_layers(blit.src_subresource, src.dimension)?;
    validate_subresource_layers(blit.dst_subresource, dst.dimension)?;
    validate_origin_extent(blit.src_origin, blit.src_extent, src.dimension)?;
    validate_origin_extent(blit.dst_origin, blit.dst_extent, dst.dimension)?;

    if blit.src.id() == blit.dst.id() {
        let a = TexelRegion {
            mip_level: blit.src_subresource.mip_level,
            layers: &blit.src_subresource,
            origin: blit.src_origin,
            extent: blit.src_extent,
        };
        let b = TexelRegion {
            mip_level: blit.dst_subresource.mip_level,
            layers: &blit.dst_subresource,
            origin: blit.dst_origin,
            extent: blit.dst_extent,
        };
        if texel_regions_overlap(&a, &b) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a blit's source and destination regions overlap",
            ));
        }
    }

    Ok(())
}

/// The route key a buffer-to-buffer copy asks about.
pub(crate) fn buffer_copy_route() -> RouteQuery {
    RouteQuery::BufferToBuffer
}

/// The route key a buffer/texture copy asks about, in the given direction.
pub(crate) fn buffer_texture_route(copy: &BufferTextureCopy, to_texture: bool) -> RouteQuery {
    if to_texture {
        buffer_to_texture_route(&copy.texture, &copy.texture_subresource)
    } else {
        texture_to_buffer_route(&copy.texture, &copy.texture_subresource)
    }
}

/// The route key a buffer-to-texture copy asks about.
///
/// Split out from [`buffer_texture_route`] because a readback asks the same
/// question without holding a [`BufferTextureCopy`]: it carries the texture and
/// its subresource and nothing about a buffer. Two callers, one mapping from a
/// texture's shape to a route key — a second copy of those three fields is how
/// the two would come to disagree about which question they are asking.
pub(crate) fn buffer_to_texture_route(
    texture: &Texture,
    layers: &TextureSubresourceLayers,
) -> RouteQuery {
    let shape = TextureCopyShape::of(texture, layers);
    RouteQuery::BufferToTexture {
        dimension: shape.dimension,
        format: shape.format,
        aspect: shape.aspect,
    }
}

/// The route key a texture-to-buffer copy or a texture readback asks about.
///
/// The counterpart of [`buffer_to_texture_route`], and shared with
/// [`ReadbackRequest::Texture`](crate::api::resource::transfer::ReadbackRequest)
/// for the same reason.
pub(crate) fn texture_to_buffer_route(
    texture: &Texture,
    layers: &TextureSubresourceLayers,
) -> RouteQuery {
    let shape = TextureCopyShape::of(texture, layers);
    RouteQuery::TextureToBuffer {
        dimension: shape.dimension,
        format: shape.format,
        aspect: shape.aspect,
    }
}

/// The route key a texture-to-texture copy asks about.
pub(crate) fn texture_copy_route(copy: &TextureCopy) -> RouteQuery {
    let src = TextureCopyShape::of(&copy.src, &copy.src_subresource);
    let dst = TextureCopyShape::of(&copy.dst, &copy.dst_subresource);
    RouteQuery::TextureToTexture {
        src_dimension: src.dimension,
        src_format: src.format,
        src_aspect: src.aspect,
        src_sample_count: src.sample_count,

        dst_dimension: dst.dimension,
        dst_format: dst.format,
        dst_aspect: dst.aspect,
        dst_sample_count: dst.sample_count,
    }
}

/// The route key a direct resolve asks about.
pub(crate) fn resolve_route(resolve: &TextureResolve) -> RouteQuery {
    RouteQuery::Resolve {
        format: resolve.src.descriptor().format,
        src_sample_count: resolve.src.descriptor().sample_count,
    }
}

/// The route key a blit asks about.
pub(crate) fn blit_route(blit: &TextureBlit) -> RouteQuery {
    RouteQuery::Blit {
        src_dimension: blit.src.descriptor().dimension,
        src_format: blit.src.descriptor().format,
        dst_dimension: blit.dst.descriptor().dimension,
        dst_format: blit.dst.descriptor().format,
        filter: blit.filter,
    }
}

// ---------------------------------------------------------------------------
// Canonical capability encoding
// ---------------------------------------------------------------------------
//
// The rules of the encoding, and what it is for, are stated once in
// `api::capability::CapabilityFacts`. It lives here because this type is declared
// here, and `RouteQuery::Blit` carries it as a key field.

impl BlitFilter {
    /// Writes this filter's canonical byte.
    ///
    /// A fieldless enum encodes as its discriminant. The two members are not
    /// interchangeable — section 34.5 makes a filtered blit a different native
    /// operation from a nearest one — so they must not encode alike, which is
    /// exactly what a discriminant guarantees and what a `bool` would not have.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        out.push(*self as u8);
    }
}
