//! Step 8's pure half: the portable copy regions lowered onto `Vulkan` copy
//! records, with the boundary checks repeated where the driver is reached.
//!
//! # Two routes, and only the two this layer owns
//!
//! `Vulkan` spells the two routes this family uses as `vkCmdCopyBuffer` and
//! `vkCmdCopyImage`. The buffer/image routes (`vkCmdCopyBufferToImage` and
//! `vkCmdCopyImageToBuffer`) belong to the RHI's own staging upload and readback
//! path, and neither one has vocabulary in
//! [`CopyApi`](crate::common::api::families::CopyApi) yet, so a record for them
//! here would be a second truth about a call nobody can make. They land with the
//! step that first needs them.
//!
//! # Why the checks live here rather than only in `imp`
//!
//! The borrowed execution layer already validates a copy region before it reaches
//! the native boundary (plan step 8's wording). This module does not replace that:
//! it repeats the same checks at the boundary that actually reaches the driver,
//! because the new path can be called from the upload and readback helpers that
//! bypass RenderGraph, and a driver validation error is a worse answer than a
//! region refused by value. The rules are the same ones the safe layer states, not
//! new ones:
//!
//! - a buffer copy's offsets and size are aligned to [`COPY_BUFFER_ALIGNMENT`] and
//!   the region fits both declared sizes, with the end computed in checked
//!   arithmetic so an overflow is a refusal rather than a wrapped range;
//! - a texture copy's extent is non-zero on every axis, both mips exist, the
//!   origin plus extent fits the extent of the mip it addresses, and neither side
//!   names a mip count the other does not have;
//! - the two textures carry the same format, because a `Vulkan` image copy is only
//!   defined between compatible formats and the portable layer already requires
//!   equality.
//!
//! # The two places this module decides something the shared vocabulary cannot
//!
//! [`TextureCopyRegion`] carries no aspect, and that is right: the aspect a copy
//! must touch follows from the image's format, which the resource table owns. The
//! aspect is therefore asked of the **mapped** `Vulkan` format through
//! [`super::texture::aspect`], so the depth fact keeps the single source of truth
//! [`super::format::is_depth`] already gives it.
//!
//! The second is layers. The borrowed execution layer already refuses a region
//! whose origin z is non-zero or whose extent z is not one, for every dimension
//! (`rendergraph/src/execution/recording/copy.rs`), and the borrowed `Vulkan` path
//! records one layer per command (`map_subresource_layers` writes `layer_count =
//! 1`). This module keeps both halves of that shape and refuses the two cases the
//! shared layer refuses -- [`CopyRegionError::LayerOrigin`] and
//! [`CopyRegionError::LayerCount`] -- by name, rather than giving a portable region
//! the graph could not have built a meaning it never had. A layered or volume copy
//! arrives with the consumer that needs it, together with the vocabulary to say
//! which layer it addresses.

use ash::vk;
use fluxel_rendergraph::{BufferCopyRegion, TextureCopyRegion, TextureDesc, TextureDimension};

/// The alignment a buffer copy's offsets and size share.
///
/// This is the portable `COPY_BUFFER_ALIGNMENT` the safe layer enforces, repeated
/// where the driver is reached. A test in [`super::command`] pins it against that
/// constant, since the two live in different modules and a copy-backend change
/// would otherwise move them apart silently.
pub(crate) const COPY_BUFFER_ALIGNMENT: u64 = 4;

/// Why a portable copy region was refused before the driver was reached.
///
/// Every variant is a rule the shared layer already states, so a caller that
/// satisfies the contract never sees one; they exist so the boundary that reaches
/// the driver cannot rely on a caller having run the shared check.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CopyRegionError {
    /// The copied size is zero, which `Vulkan` does not define.
    ZeroSize,
    /// A buffer offset or the size is not a multiple of
    /// [`COPY_BUFFER_ALIGNMENT`].
    Misaligned,
    /// A buffer range's end overflows the `u64` byte space.
    Overflow,
    /// A buffer range leaves the buffer the region named.
    OutOfBounds,
    /// The source and destination texture formats differ.
    ///
    /// `Vulkan` defines a copy between *compatible* formats, and the portable layer
    /// already requires equality; accepting a compatible pair here would be a
    /// capability claim the graph's own check never made.
    FormatMismatch,
    /// The copied extent is zero on an axis, which `Vulkan` does not define.
    ZeroExtent,
    /// A mip level named by the region does not exist on that texture.
    UnknownMipLevel,
    /// A layered or volume copy names a z origin this backend's record cannot
    /// express.
    ///
    /// A `Vulkan` copy names a subresource *range* -- `base_array_layer` plus
    /// `layer_count` -- so the portable region's z origin and z extent could each be
    /// read two ways: as a layer index and a layer count, or as a depth coordinate
    /// and a box depth. The portable layer and the borrowed native path both resolve
    /// the ambiguity by refusing every non-zero z origin, and this module does the
    /// same rather than inventing one reading for the graph.
    LayerOrigin,
    /// A copy's z extent is not the single layer this backend records per command.
    ///
    /// See [`Self::LayerOrigin`]: a multi-layer or volume copy needs vocabulary the
    /// portable region does not carry, and one layer per command is what the path
    /// being replaced records.
    LayerCount,
}

/// The axis limits one mip of `desc` offers a copy box.
///
/// The x and y limits are the mip's own extent, each axis halved per level and
/// floored at one, which is the shape the texture lowering already floors the
/// declared extent to. The z limit is the extent's z axis as the texture was
/// created: for a depth-addressable image that is its depth, and for every other
/// image it is the array layer count, which is what
/// [`super::texture::image_create_info`] writes for a depth greater than one. A
/// region only ever names one layer here, so this limit is what keeps a box from
/// addressing an axis the image does not have.
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

/// Lowers one portable buffer copy region, or refuses it by value.
///
/// The end of each range is computed in checked arithmetic: an offset near
/// `u64::MAX` must be a refusal, not a wrapped range that passes the bounds test.
pub(crate) fn buffer_copy(
    source_size: u64,
    destination_size: u64,
    region: BufferCopyRegion,
) -> Result<vk::BufferCopy, CopyRegionError> {
    if region.size == 0 {
        return Err(CopyRegionError::ZeroSize);
    }
    if !region.source_offset.is_multiple_of(COPY_BUFFER_ALIGNMENT)
        || !region
            .destination_offset
            .is_multiple_of(COPY_BUFFER_ALIGNMENT)
        || !region.size.is_multiple_of(COPY_BUFFER_ALIGNMENT)
    {
        return Err(CopyRegionError::Misaligned);
    }
    for (offset, declared) in [
        (region.source_offset, source_size),
        (region.destination_offset, destination_size),
    ] {
        let end = offset
            .checked_add(region.size)
            .ok_or(CopyRegionError::Overflow)?;
        if end > declared {
            return Err(CopyRegionError::OutOfBounds);
        }
    }
    Ok(vk::BufferCopy {
        src_offset: region.source_offset,
        dst_offset: region.destination_offset,
        size: region.size,
    })
}

/// Lowers one portable texture copy region, or refuses it by value.
///
/// `format` is the **mapped** `Vulkan` format both images were created with; the
/// aspect is derived from it in [`super::texture::aspect`] rather than guessed
/// from the portable format. `source` and `destination` are the descriptions the
/// two images were created from, which is what the bounds are checked against.
pub(crate) fn image_copy(
    source: &TextureDesc,
    destination: &TextureDesc,
    region: TextureCopyRegion,
    format: vk::Format,
) -> Result<vk::ImageCopy, CopyRegionError> {
    if source.format != destination.format {
        return Err(CopyRegionError::FormatMismatch);
    }
    if region.extent.contains(&0) {
        return Err(CopyRegionError::ZeroExtent);
    }
    if region.source_mip_level >= source.mip_levels
        || region.destination_mip_level >= destination.mip_levels
    {
        return Err(CopyRegionError::UnknownMipLevel);
    }

    // Both z halves of the region are refused unless they are the single layer at
    // layer zero this backend records, which is the same refusal the shared layer
    // states and the shape the borrowed native path writes.
    for origin in [region.source_origin, region.destination_origin] {
        if origin[2] != 0 {
            return Err(CopyRegionError::LayerOrigin);
        }
    }
    if region.extent[2] != 1 {
        return Err(CopyRegionError::LayerCount);
    }

    for (origin, extent, limit) in [
        (
            region.source_origin,
            region.extent,
            mip_extent(source, region.source_mip_level),
        ),
        (
            region.destination_origin,
            region.extent,
            mip_extent(destination, region.destination_mip_level),
        ),
    ] {
        for axis in 0..3 {
            if origin[axis]
                .checked_add(extent[axis])
                .is_none_or(|end| end > limit[axis])
            {
                return Err(CopyRegionError::OutOfBounds);
            }
        }
    }

    let aspect = super::texture::aspect(format);
    let subresource = |mip_level: u32, origin: [u32; 3]| vk::ImageSubresourceLayers {
        aspect_mask: aspect,
        mip_level,
        base_array_layer: origin[2],
        layer_count: 1,
    };
    Ok(vk::ImageCopy {
        src_subresource: subresource(region.source_mip_level, region.source_origin),
        src_offset: offset(region.source_origin),
        dst_subresource: subresource(region.destination_mip_level, region.destination_origin),
        dst_offset: offset(region.destination_origin),
        extent: vk::Extent3D {
            width: region.extent[0],
            height: region.extent[1],
            depth: region.extent[2],
        },
    })
}

/// A portable texel origin as a `Vulkan` offset.
///
/// The conversion reports the same value: a portable origin is a non-negative
/// texel coordinate bounded by the mip extent checked above, and `Vulkan`'s offset
/// is signed. The saturating form is a total function rather than a panic, and a
/// coordinate large enough to saturate is already outside every mip the check
/// admits.
fn offset(origin: [u32; 3]) -> vk::Offset3D {
    let component = |value: u32| i32::try_from(value).unwrap_or(i32::MAX);
    vk::Offset3D {
        x: component(origin[0]),
        y: component(origin[1]),
        z: component(origin[2]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxel_rendergraph::{Extent3d, TextureFormat};

    fn buffer_region(source_offset: u64, destination_offset: u64, size: u64) -> BufferCopyRegion {
        BufferCopyRegion {
            source_offset,
            destination_offset,
            size,
        }
    }

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

    fn region(origin: [u32; 3], destination: [u32; 3], extent: [u32; 3]) -> TextureCopyRegion {
        TextureCopyRegion {
            source_origin: origin,
            destination_origin: destination,
            extent,
            source_mip_level: 0,
            destination_mip_level: 0,
        }
    }

    #[test]
    fn a_valid_buffer_region_lowers_field_for_field() {
        let copy = buffer_copy(256, 512, buffer_region(16, 32, 64)).expect("a valid region");
        assert_eq!(copy.src_offset, 16);
        assert_eq!(copy.dst_offset, 32);
        assert_eq!(copy.size, 64);
    }

    #[test]
    fn a_zero_misaligned_or_overflowing_buffer_region_is_refused_by_name() {
        assert_eq!(
            buffer_copy(256, 256, buffer_region(0, 0, 0)).err(),
            Some(CopyRegionError::ZeroSize)
        );
        assert_eq!(
            buffer_copy(256, 256, buffer_region(2, 0, 4)).err(),
            Some(CopyRegionError::Misaligned)
        );
        assert_eq!(
            buffer_copy(256, 256, buffer_region(0, 2, 4)).err(),
            Some(CopyRegionError::Misaligned)
        );
        assert_eq!(
            buffer_copy(256, 256, buffer_region(0, 0, 6)).err(),
            Some(CopyRegionError::Misaligned)
        );
        // The wrapped-end trap: `u64::MAX` would pass a plain `end > size` test if
        // the addition were allowed to wrap. The size is a multiple of the
        // alignment, so the overflow is what this case isolates.
        assert_eq!(
            buffer_copy(u64::MAX, 256, buffer_region(u64::MAX - 3, 0, 4)).err(),
            Some(CopyRegionError::Overflow)
        );
    }

    #[test]
    fn a_buffer_region_that_leaves_either_buffer_is_refused() {
        assert_eq!(
            buffer_copy(64, 256, buffer_region(64, 0, 4)).err(),
            Some(CopyRegionError::OutOfBounds)
        );
        assert_eq!(
            buffer_copy(256, 64, buffer_region(0, 64, 4)).err(),
            Some(CopyRegionError::OutOfBounds)
        );
        // Exactly at the end is inside.
        assert!(buffer_copy(64, 64, buffer_region(60, 60, 4)).is_ok());
    }

    #[test]
    fn a_valid_texture_region_lowers_the_box_and_one_layer() {
        let described = desc(TextureFormat::Rgba8Unorm, 16, 8, 1);
        let copy = image_copy(
            &described,
            &described,
            region([4, 2, 0], [0, 0, 0], [8, 4, 1]),
            vk::Format::R8G8B8A8_UNORM,
        )
        .expect("a valid region");
        assert_eq!(copy.src_offset.x, 4);
        assert_eq!(copy.src_offset.y, 2);
        assert_eq!(copy.src_offset.z, 0);
        assert_eq!(copy.dst_offset.x, 0);
        assert_eq!(copy.dst_offset.y, 0);
        assert_eq!(copy.extent.width, 8);
        assert_eq!(copy.extent.height, 4);
        assert_eq!(copy.extent.depth, 1);
        assert_eq!(copy.src_subresource.mip_level, 0);
        assert_eq!(copy.dst_subresource.mip_level, 0);
        assert_eq!(copy.src_subresource.aspect_mask, vk::ImageAspectFlags::COLOR);
        assert_eq!(copy.dst_subresource.aspect_mask, vk::ImageAspectFlags::COLOR);
        // One layer per command is written, not derived from an array count that is
        // not part of the portable region.
        assert_eq!(copy.src_subresource.layer_count, 1);
        assert_eq!(copy.dst_subresource.layer_count, 1);
        assert_eq!(copy.src_subresource.base_array_layer, 0);
        assert_eq!(copy.dst_subresource.base_array_layer, 0);
    }

    #[test]
    fn a_depth_texture_copy_takes_the_depth_aspect() {
        let described = desc(TextureFormat::Depth32Float, 16, 8, 1);
        let copy = image_copy(
            &described,
            &described,
            region([0, 0, 0], [0, 0, 0], [1, 1, 1]),
            vk::Format::D32_SFLOAT,
        )
        .expect("a valid region");
        assert_eq!(copy.src_subresource.aspect_mask, vk::ImageAspectFlags::DEPTH);
        assert_eq!(
            copy.dst_subresource.aspect_mask,
            vk::ImageAspectFlags::DEPTH
        );
    }

    #[test]
    fn differing_texture_formats_are_refused() {
        let source = desc(TextureFormat::Rgba8Unorm, 8, 8, 1);
        let destination = desc(TextureFormat::Bgra8Unorm, 8, 8, 1);
        assert_eq!(
            image_copy(
                &source,
                &destination,
                region([0, 0, 0], [0, 0, 0], [1, 1, 1]),
                vk::Format::R8G8B8A8_UNORM,
            )
            .err(),
            Some(CopyRegionError::FormatMismatch)
        );
    }

    #[test]
    fn a_zero_extent_or_unknown_mip_is_refused_by_name() {
        let described = desc(TextureFormat::Rgba8Unorm, 8, 8, 1);
        assert_eq!(
            image_copy(
                &described,
                &described,
                region([0, 0, 0], [0, 0, 0], [0, 1, 1]),
                vk::Format::R8G8B8A8_UNORM,
            )
            .err(),
            Some(CopyRegionError::ZeroExtent)
        );

        let mut two_levels = described;
        two_levels.mip_levels = 2;
        let mut beyond = region([0, 0, 0], [0, 0, 0], [1, 1, 1]);
        beyond.source_mip_level = 2;
        assert_eq!(
            image_copy(&two_levels, &described, beyond, vk::Format::R8G8B8A8_UNORM).err(),
            Some(CopyRegionError::UnknownMipLevel)
        );

        let mut destination_beyond = region([0, 0, 0], [0, 0, 0], [1, 1, 1]);
        destination_beyond.destination_mip_level = 2;
        assert_eq!(
            image_copy(
                &described,
                &two_levels,
                destination_beyond,
                vk::Format::R8G8B8A8_UNORM
            )
            .err(),
            Some(CopyRegionError::UnknownMipLevel)
        );
    }

    #[test]
    fn a_region_that_leaves_the_mip_extent_is_refused() {
        let described = desc(TextureFormat::Rgba8Unorm, 8, 8, 1);
        // 7 + 2 runs past the mip's width of 8, and the wrapped form is refused
        // rather than passed on as an out-of-range texel box.
        assert_eq!(
            image_copy(
                &described,
                &described,
                region([7, 0, 0], [0, 0, 0], [2, 1, 1]),
                vk::Format::R8G8B8A8_UNORM,
            )
            .err(),
            Some(CopyRegionError::OutOfBounds)
        );
        assert_eq!(
            image_copy(
                &described,
                &described,
                region([0, 0, 0], [0, 8, 0], [1, 1, 1]),
                vk::Format::R8G8B8A8_UNORM,
            )
            .err(),
            Some(CopyRegionError::OutOfBounds)
        );
        // Exactly covering the mip is inside.
        assert!(
            image_copy(
                &described,
                &described,
                region([0, 0, 0], [0, 0, 0], [8, 8, 1]),
                vk::Format::R8G8B8A8_UNORM,
            )
            .is_ok()
        );
    }

    #[test]
    fn a_mip_level_addresses_its_own_halved_extent() {
        let mut described = desc(TextureFormat::Rgba8Unorm, 8, 8, 1);
        described.mip_levels = 2;
        let mut second = region([0, 0, 0], [0, 0, 0], [4, 4, 1]);
        second.source_mip_level = 1;
        second.destination_mip_level = 1;
        // Level one of an 8x8 image is 4x4, so this exactly covers it.
        assert!(
            image_copy(&described, &described, second, vk::Format::R8G8B8A8_UNORM).is_ok()
        );
        let mut too_wide = second;
        too_wide.extent = [5, 4, 1];
        assert_eq!(
            image_copy(
                &described,
                &described,
                too_wide,
                vk::Format::R8G8B8A8_UNORM
            )
            .err(),
            Some(CopyRegionError::OutOfBounds)
        );
    }

    #[test]
    fn a_layered_or_volume_copy_is_refused_by_name() {
        // The portable region carries no layer range, so a non-zero z origin and a
        // multi-layer z extent are the two shapes this backend cannot record. The
        // shared layer refuses both already; the names here keep the refusal
        // answerable at the boundary that reaches the driver.
        let mut layered = desc(TextureFormat::Rgba8Unorm, 8, 8, 1);
        layered.array_layers = 4;
        assert_eq!(
            image_copy(
                &layered,
                &layered,
                region([0, 0, 1], [0, 0, 0], [8, 8, 1]),
                vk::Format::R8G8B8A8_UNORM,
            )
            .err(),
            Some(CopyRegionError::LayerOrigin),
            "a non-zero layer index has no meaning the region can state"
        );
        assert_eq!(
            image_copy(
                &layered,
                &layered,
                region([0, 0, 0], [0, 0, 2], [8, 8, 1]),
                vk::Format::R8G8B8A8_UNORM,
            )
            .err(),
            Some(CopyRegionError::LayerOrigin),
            "and the destination is checked too"
        );
        assert_eq!(
            image_copy(
                &layered,
                &layered,
                region([0, 0, 0], [0, 0, 0], [8, 8, 2]),
                vk::Format::R8G8B8A8_UNORM,
            )
            .err(),
            Some(CopyRegionError::LayerCount),
            "a multi-layer box is not one layer"
        );

        // Layer zero of layer zero is the one shape that records, and its record
        // names exactly one layer.
        let copy = image_copy(
            &layered,
            &layered,
            region([0, 0, 0], [0, 0, 0], [8, 8, 1]),
            vk::Format::R8G8B8A8_UNORM,
        )
        .expect("one whole layer of a layered image");
        assert_eq!(copy.src_subresource.base_array_layer, 0);
        assert_eq!(copy.src_subresource.layer_count, 1);
    }

    #[test]
    fn a_volume_copy_is_refused_by_the_same_two_names() {
        // A depth-addressable image is not an escape hatch: its z axis is a depth
        // box, which the portable region cannot state either, so the refusal is the
        // same and is not special-cased by dimension.
        let mut volume = desc(TextureFormat::Rgba8Unorm, 8, 8, 4);
        volume.dimension = TextureDimension::D3;
        assert_eq!(
            image_copy(
                &volume,
                &volume,
                region([0, 0, 1], [0, 0, 0], [8, 8, 1]),
                vk::Format::R8G8B8A8_UNORM,
            )
            .err(),
            Some(CopyRegionError::LayerOrigin)
        );
        assert_eq!(
            image_copy(
                &volume,
                &volume,
                region([0, 0, 0], [0, 0, 0], [8, 8, 3]),
                vk::Format::R8G8B8A8_UNORM,
            )
            .err(),
            Some(CopyRegionError::LayerCount)
        );
        assert!(
            image_copy(
                &volume,
                &volume,
                region([0, 0, 0], [0, 0, 0], [8, 8, 1]),
                vk::Format::R8G8B8A8_UNORM,
            )
            .is_ok(),
            "the first layer of a volume still copies"
        );
    }
}
