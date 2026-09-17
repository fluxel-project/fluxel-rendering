//! Step 4's pure half: the portable texture description lowered onto `Vulkan`.
//!
//! Three mappings, and one rule shared with the buffer side: **nothing is widened**.
//! Each portable usage kind maps to exactly the flag that serves it, so a texture
//! is never more capable than the graph declared. The graph's capability check is
//! what decides whether an operation is legal, and a texture that quietly acquired
//! `TRANSFER_DST` would pass a check it should have failed.
//!
//! The dimension mapping returns an `Option` for the same reason the format mapping
//! does (see [`super::format`]): the portable enum is foreign and may be
//! `#[non_exhaustive]`, so no backend can have a compile-checked total mapping. An
//! unrecognized dimension is refused here rather than becoming an `UNDEFINED` image
//! the driver rejects later with a worse message.

use ash::vk;
use fluxel_rendergraph::{TextureDimension, TextureUsage, TextureUsageKind};

use super::format::is_depth;

/// Every portable texture usage kind, in `TextureUsageKind`'s declaration order.
///
/// Written in the enum's own order so a kind added upstream is visible in review.
/// `TextureUsageKind` offers no variant iterator, so no test can prove this list
/// complete; the test below asserts what is checkable, that every flag this backend
/// needs is reachable.
const KINDS: [(TextureUsageKind, vk::ImageUsageFlags); 7] = [
    (TextureUsageKind::Sampled, vk::ImageUsageFlags::SAMPLED),
    (
        TextureUsageKind::StorageRead,
        vk::ImageUsageFlags::STORAGE,
    ),
    (
        TextureUsageKind::StorageWrite,
        vk::ImageUsageFlags::STORAGE,
    ),
    (
        TextureUsageKind::ColorAttachment,
        vk::ImageUsageFlags::COLOR_ATTACHMENT,
    ),
    (
        TextureUsageKind::DepthStencilAttachment,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
    ),
    (
        TextureUsageKind::CopySource,
        vk::ImageUsageFlags::TRANSFER_SRC,
    ),
    (
        TextureUsageKind::CopyDestination,
        vk::ImageUsageFlags::TRANSFER_DST,
    ),
];

/// The `Vulkan` image type for one portable dimension, or `None` where this backend
/// has no equivalent.
pub(crate) fn image_type(dimension: TextureDimension) -> Option<vk::ImageType> {
    Some(match dimension {
        TextureDimension::D1 => vk::ImageType::TYPE_1D,
        TextureDimension::D2 => vk::ImageType::TYPE_2D,
        TextureDimension::D3 => vk::ImageType::TYPE_3D,
        // A dimension added upstream, which this backend has not been taught.
        _ => return None,
    })
}

/// Lowers the requested usages to the flags that serve exactly them.
pub(crate) fn usage_flags(usage: TextureUsage) -> vk::ImageUsageFlags {
    KINDS
        .iter()
        .filter(|(kind, _)| usage.contains(*kind))
        .fold(vk::ImageUsageFlags::empty(), |flags, (_, flag)| {
            flags | *flag
        })
}

/// The aspect flags a view of `format` selects.
///
/// Asked of the mapped `Vulkan` format rather than the portable one, so the depth
/// question has one source of truth: the format actually used to create the image.
pub(crate) fn aspect(format: vk::Format) -> vk::ImageAspectFlags {
    if is_depth(format) {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxel_rendergraph::TextureFormat;

    fn of(kinds: &[TextureUsageKind]) -> TextureUsage {
        TextureUsage::from_kinds(kinds.iter().copied())
    }

    #[test]
    fn each_dimension_maps_to_its_own_image_type() {
        assert_eq!(
            image_type(TextureDimension::D1),
            Some(vk::ImageType::TYPE_1D)
        );
        assert_eq!(
            image_type(TextureDimension::D2),
            Some(vk::ImageType::TYPE_2D)
        );
        assert_eq!(
            image_type(TextureDimension::D3),
            Some(vk::ImageType::TYPE_3D)
        );
    }

    #[test]
    fn both_storage_directions_need_the_storage_flag() {
        for kind in [
            TextureUsageKind::StorageRead,
            TextureUsageKind::StorageWrite,
        ] {
            assert_eq!(usage_flags(of(&[kind])), vk::ImageUsageFlags::STORAGE);
        }
    }

    #[test]
    fn sampled_and_attachment_usages_are_distinct_flags() {
        let sampled = usage_flags(of(&[TextureUsageKind::Sampled]));
        assert!(sampled.contains(vk::ImageUsageFlags::SAMPLED));
        assert!(!sampled.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT));
        assert!(!sampled.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT));

        let colour = usage_flags(of(&[TextureUsageKind::ColorAttachment]));
        assert!(colour.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT));
        assert!(!colour.contains(vk::ImageUsageFlags::SAMPLED));
    }

    #[test]
    fn the_mapping_never_widens_to_a_transfer_destination() {
        let declared = of(&[
            TextureUsageKind::Sampled,
            TextureUsageKind::ColorAttachment,
        ]);
        assert!(!usage_flags(declared).contains(vk::ImageUsageFlags::TRANSFER_DST));
    }

    #[test]
    fn a_depth_image_gets_the_depth_aspect_and_a_colour_image_the_colour_one() {
        use super::super::format::image_format;
        let depth = image_format(TextureFormat::Depth32Float).expect("mapped");
        let colour = image_format(TextureFormat::Rgba8Unorm).expect("mapped");
        assert_eq!(aspect(depth), vk::ImageAspectFlags::DEPTH);
        assert_eq!(aspect(colour), vk::ImageAspectFlags::COLOR);
    }

    #[test]
    fn every_flag_this_backend_needs_is_reachable() {
        let all = usage_flags(TextureUsage::from_kinds(KINDS.iter().map(|(kind, _)| *kind)));
        for expected in [
            vk::ImageUsageFlags::SAMPLED,
            vk::ImageUsageFlags::STORAGE,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageUsageFlags::TRANSFER_DST,
        ] {
            assert!(all.contains(expected), "{expected:?} is unreachable");
        }
    }
}
