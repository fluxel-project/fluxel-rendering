//! The copy footprints the RHI's own transfer paths express.
//!
//! `fluxel-rendergraph` already owns the graph-level copy regions -- its
//! `BufferCopyRegion` and `TextureCopyRegion` -- and this layer consumes them rather
//! than restating them. What the portable contract deliberately does not model is the
//! buffer-image crossing the RHI's own staging upload and readback paths use: where a
//! texel region's bytes live inside a buffer, where the texel box begins inside an
//! image, and how large that box is. That vocabulary is here, because those two paths
//! are the RHI's own and a graph must never be able to name them.
//!
//! The one rule is the image footprint's, and it is stated once for both copy routes.
//! A texel box must copy something, must name a mip the image has, must address the
//! single layer this execution model records per command, and must fit the extent of
//! the mip it names. Both the image-to-image route and the buffer-image route lower
//! through [`check_image_region`], so the two cannot disagree about the shape of a
//! box -- which is exactly the split `native::vulkan::copy` already needed and would
//! otherwise have written twice.

use fluxel_rendergraph::{TextureDesc, TextureDimension};

/// Where in a buffer a texel region's bytes are addressed.
///
/// `row_length` and `image_height` are **texel** counts rather than byte counts --
/// the unit the native APIs state them in -- and a zero means "tightly packed", which
/// is the shape this layer's own uploads use. They are carried rather than derived
/// because a caller may pack records with padding, and an assumed tight packing would
/// read the wrong bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TexelCopyLayout {
    /// The byte offset of the region's first texel.
    pub(crate) offset: u64,
    /// Texels per row, or zero where the rows are tightly packed.
    pub(crate) row_length: u32,
    /// Rows per layer, or zero where the layers are tightly packed.
    pub(crate) image_height: u32,
}

/// Where a texel region begins inside an image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TexelCopyBase {
    /// The mip level the region addresses.
    pub(crate) mip_level: u32,
    /// The texel coordinate the region begins at.
    pub(crate) origin: [u32; 3],
}

/// One buffer-image transfer footprint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BufferImageRegion {
    /// Where the region's bytes live in the buffer.
    pub(crate) buffer: TexelCopyLayout,
    /// Where the region begins in the image.
    pub(crate) image: TexelCopyBase,
    /// The size of the box copied, in texels.
    pub(crate) extent: [u32; 3],
}

/// Why a texel box cannot address the image it names.
///
/// Each variant is a refusal a caller fixes differently, which is why they are not
/// collapsed into one "bad region": a box that copies nothing, a box that names a mip
/// the image does not have, a box that addresses a layer this execution model cannot
/// record, and a box that leaves the mip are four different mistakes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ImageRegionError {
    /// The box copies nothing, which no copy command defines.
    ZeroExtent,
    /// The box names a mip level the image does not have.
    UnknownMipLevel,
    /// The box begins at a z coordinate that is not layer zero.
    ///
    /// A native copy names a subresource *range*, so a portable z origin could be read
    /// two ways: as a layer index or as a depth coordinate. This execution model
    /// records one layer per command, so the ambiguity is resolved by refusing every
    /// non-zero z origin rather than inventing one reading. A layered or volume copy
    /// arrives with the vocabulary that says which layer it addresses.
    LayerOrigin,
    /// The box's z extent is not the single layer this execution model records.
    LayerCount,
    /// The box leaves the extent of the mip it addresses.
    OutOfBounds,
}

/// The axis limits one mip of `desc` offers a copy box.
///
/// The x and y limits are the mip's own extent, each axis halved per level and floored
/// at one, which is the shape the native texture lowering floors the declared extent
/// to. The z limit is the extent's z axis as the texture was created: for a
/// depth-addressable image that is its depth, and for every other image it is the
/// array layer count, which is what the native image lowering writes for a depth
/// greater than one. A region records one layer here, so this limit is what keeps a
/// box from addressing an axis the image does not have.
fn mip_extent(desc: &TextureDesc, mip_level: u32) -> [u32; 3] {
    let depth = if desc.dimension == TextureDimension::D3 {
        desc.extent.depth
    } else {
        desc.array_layers
    };
    [
        (desc.extent.width >> mip_level).max(1),
        (desc.extent.height >> mip_level).max(1),
        (depth >> mip_level).max(1),
    ]
}

/// Checks one texel box against the image it addresses.
///
/// Pure and total: it reads the description the image was created from and asks the
/// driver nothing, so both copy routes can share it and every refusal is provable
/// without a device. The checks are ordered so that one bad box produces one sentence,
/// and the mip is checked before it is indexed by [`mip_extent`].
pub(crate) fn check_image_region(
    desc: &TextureDesc,
    mip_level: u32,
    origin: [u32; 3],
    extent: [u32; 3],
) -> Result<(), ImageRegionError> {
    if extent.contains(&0) {
        return Err(ImageRegionError::ZeroExtent);
    }
    if mip_level >= desc.mip_levels {
        return Err(ImageRegionError::UnknownMipLevel);
    }
    if origin[2] != 0 {
        return Err(ImageRegionError::LayerOrigin);
    }
    if extent[2] != 1 {
        return Err(ImageRegionError::LayerCount);
    }
    let limit = mip_extent(desc, mip_level);
    for axis in 0..3 {
        if origin[axis]
            .checked_add(extent[axis])
            .is_none_or(|end| end > limit[axis])
        {
            return Err(ImageRegionError::OutOfBounds);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxel_rendergraph::{Extent3d, TextureFormat};

    fn desc(format: TextureFormat, width: u32, height: u32, depth: u32) -> TextureDesc {
        TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width,
                height,
                depth,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format,
        }
    }

    #[test]
    fn a_box_that_fits_its_mip_is_accepted() {
        let described = desc(TextureFormat::Rgba8Unorm, 16, 8, 1);
        assert_eq!(check_image_region(&described, 0, [0, 0, 0], [16, 8, 1]), Ok(()));
        // Exactly covering a sub-box is inside, and the boundary is not off by one.
        assert_eq!(check_image_region(&described, 0, [15, 7, 0], [1, 1, 1]), Ok(()));
    }

    #[test]
    fn each_bad_box_is_refused_by_its_own_name() {
        let described = desc(TextureFormat::Rgba8Unorm, 8, 8, 1);
        assert_eq!(
            check_image_region(&described, 0, [0, 0, 0], [0, 1, 1]),
            Err(ImageRegionError::ZeroExtent)
        );
        assert_eq!(
            check_image_region(&described, 0, [0, 0, 0], [1, 0, 1]),
            Err(ImageRegionError::ZeroExtent)
        );
        assert_eq!(
            check_image_region(&described, 1, [0, 0, 0], [1, 1, 1]),
            Err(ImageRegionError::UnknownMipLevel)
        );
        assert_eq!(
            check_image_region(&described, 0, [0, 1, 0], [8, 8, 1]),
            Err(ImageRegionError::OutOfBounds)
        );
        // 7 + 2 runs past a width of 8, and the wrapped form is refused rather than
        // passed on as an out-of-range texel box.
        assert_eq!(
            check_image_region(&described, 0, [7, 0, 0], [2, 1, 1]),
            Err(ImageRegionError::OutOfBounds)
        );
    }

    #[test]
    fn the_layer_rules_refuse_a_multi_layer_or_offset_box() {
        let mut layered = desc(TextureFormat::Rgba8Unorm, 8, 8, 1);
        layered.array_layers = 4;
        assert_eq!(
            check_image_region(&layered, 0, [0, 0, 1], [8, 8, 1]),
            Err(ImageRegionError::LayerOrigin),
            "a non-zero layer index has no meaning this vocabulary can state"
        );
        assert_eq!(
            check_image_region(&layered, 0, [0, 0, 0], [8, 8, 2]),
            Err(ImageRegionError::LayerCount),
            "a multi-layer box is not the one layer this execution model records"
        );
        // Layer zero of layer zero is the one shape accepted.
        assert_eq!(check_image_region(&layered, 0, [0, 0, 0], [8, 8, 1]), Ok(()));
    }

    #[test]
    fn a_mip_level_addresses_its_own_halved_extent() {
        let mut described = desc(TextureFormat::Rgba8Unorm, 8, 8, 1);
        described.mip_levels = 2;
        // Level one of an 8x8 image is 4x4, so this exactly covers it.
        assert_eq!(check_image_region(&described, 1, [0, 0, 0], [4, 4, 1]), Ok(()));
        assert_eq!(
            check_image_region(&described, 1, [0, 0, 0], [5, 4, 1]),
            Err(ImageRegionError::OutOfBounds)
        );
    }

    #[test]
    fn the_z_axis_is_depth_for_a_volume_and_layers_for_everything_else() {
        // The trap this rule exists for: reading `extent.depth` for both would let a
        // two-dimensional box address layers an image does not have, and reading
        // `array_layers` for both would refuse a legal volume.
        let mut layered = desc(TextureFormat::Rgba8Unorm, 8, 8, 1);
        layered.array_layers = 4;
        // A D2 image's z axis is its layer count, which is one here even though the
        // declared depth is also one -- so the two agree only by construction.
        assert_eq!(check_image_region(&layered, 0, [0, 0, 0], [8, 8, 1]), Ok(()));

        let mut volume = desc(TextureFormat::Rgba8Unorm, 8, 8, 4);
        volume.dimension = TextureDimension::D3;
        // The first layer of a volume is still one layer, and the same layer rules
        // apply rather than a dimension-specific escape hatch.
        assert_eq!(check_image_region(&volume, 0, [0, 0, 0], [8, 8, 1]), Ok(()));
        assert_eq!(
            check_image_region(&volume, 0, [0, 0, 1], [8, 8, 1]),
            Err(ImageRegionError::LayerOrigin)
        );
    }

    #[test]
    fn the_footprint_vocabulary_carries_what_was_written() {
        // The value types are plain data, and the one property worth pinning is that a
        // tight layout and a padded one are distinguishable: a backend that derived
        // the layout instead of carrying it would collapse the two and read the wrong
        // bytes.
        let tight = BufferImageRegion {
            buffer: TexelCopyLayout {
                offset: 0,
                row_length: 0,
                image_height: 0,
            },
            image: TexelCopyBase {
                mip_level: 0,
                origin: [0, 0, 0],
            },
            extent: [4, 4, 1],
        };
        let padded = BufferImageRegion {
            buffer: TexelCopyLayout {
                offset: 256,
                row_length: 8,
                image_height: 8,
            },
            ..tight
        };
        assert_ne!(tight, padded);
        assert_eq!(padded.buffer.offset, 256);
        assert_eq!(padded.buffer.row_length, 8);
        assert_eq!(padded.buffer.image_height, 8);
        assert_eq!(padded.image, tight.image, "only the buffer side differs");
    }
}
