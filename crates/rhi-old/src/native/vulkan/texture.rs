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
use fluxel_rendergraph::{TextureDesc, TextureDimension, TextureUsage, TextureUsageKind};

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

/// The `Vulkan` image-view type for one portable description.
///
/// A two-dimensional description with more than one layer is an *array* view rather
/// than a single slice: the caller declared layers, and a `TYPE_2D` view of a layered
/// image silently selects layer zero, which is a different resource than the one the
/// graph described. A dimension this backend has not been taught is refused for the
/// same reason [`image_type`] refuses it.
pub(crate) fn view_type(desc: &TextureDesc) -> Option<vk::ImageViewType> {
    Some(match desc.dimension {
        TextureDimension::D1 => vk::ImageViewType::TYPE_1D,
        TextureDimension::D2 => {
            if desc.array_layers.max(1) > 1 {
                vk::ImageViewType::TYPE_2D_ARRAY
            } else {
                vk::ImageViewType::TYPE_2D
            }
        }
        TextureDimension::D3 => vk::ImageViewType::TYPE_3D,
        // A dimension added upstream, which this backend has not been taught.
        _ => return None,
    })
}

/// The create-info for the view a caller samples an image through.
///
/// The view spans exactly the levels and layers the description declared, so a
/// caller cannot sample a mip the graph did not name, and its aspect is asked of the
/// mapped `Vulkan` format rather than the portable one, so the depth question keeps
/// the single source of truth [`aspect`] exists for. The component mapping is left at
/// its default identity swizzle, which is what "the image's own channels" means.
///
/// Returns `None` wherever this backend cannot build the view, which is the same set
/// of descriptions [`image_create_info`] refuses: a view of an image this backend
/// cannot create is not a thing to build.
pub(crate) fn view_create_info(
    desc: &TextureDesc,
    image: vk::Image,
) -> Option<vk::ImageViewCreateInfo<'static>> {
    let format = super::format::image_format(desc.format)?;
    Some(
        vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type(desc)?)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect(format),
                base_mip_level: 0,
                level_count: desc.mip_levels.max(1),
                base_array_layer: 0,
                layer_count: desc.array_layers.max(1),
            }),
    )
}

/// The create-info for an image described by `desc` and used exactly as `usage`
/// states.
///
/// Returns `None` for every description this backend cannot honour, and the reasons
/// are deliberately collected in one place rather than reached one at a time:
///
/// - a dimension with no `Vulkan` equivalent, or a format with none;
/// - a zero extent, because `Vulkan` requires a non-zero image;
/// - a sample count other than one. Multisampling is a capability row that nothing
///   has proved and no retained recipe asks for, so a multisampled image is refused
///   here rather than created on the assumption that the format supports the count.
///
/// Tiling is `OPTIMAL`, which is what a device-local image the GPU both samples and
/// writes wants; a linear image exists for host access, and the upload path uses a
/// buffer for that rather than a linear image.
pub(crate) fn image_create_info(
    desc: &TextureDesc,
    usage: TextureUsage,
) -> Option<vk::ImageCreateInfo<'static>> {
    if desc.sample_count != 1 {
        return None;
    }
    if desc.extent.width == 0 || desc.extent.height == 0 || desc.extent.depth == 0 {
        return None;
    }
    Some(
        vk::ImageCreateInfo::default()
            .image_type(image_type(desc.dimension)?)
            .format(super::format::image_format(desc.format)?)
            .extent(vk::Extent3D {
                width: desc.extent.width,
                height: desc.extent.height,
                depth: desc.extent.depth,
            })
            .mip_levels(desc.mip_levels.max(1))
            .array_layers(desc.array_layers.max(1))
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage_flags(usage))
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            // `Vulkan` requires the initial layout to be `UNDEFINED`: the image's
            // contents are undefined until something writes them, and the graph's
            // first use transitions it from there.
            .initial_layout(vk::ImageLayout::UNDEFINED),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxel_rendergraph::TextureFormat;
    use fluxel_rendergraph::{Extent3d, TextureDesc};

    fn desc(format: TextureFormat, sample_count: u32) -> TextureDesc {
        TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: 16,
                height: 8,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count,
            format,
        }
    }

    #[test]
    fn a_two_dimensional_image_carries_its_extent_format_and_usage() {
        let declared = of(&[TextureUsageKind::Sampled, TextureUsageKind::CopyDestination]);
        let info = image_create_info(&desc(TextureFormat::Rgba8Unorm, 1), declared)
            .expect("a supported description");
        assert_eq!(info.image_type, vk::ImageType::TYPE_2D);
        assert_eq!(info.format, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(info.extent.width, 16);
        assert_eq!(info.extent.height, 8);
        assert_eq!(info.extent.depth, 1);
        assert_eq!(info.samples, vk::SampleCountFlags::TYPE_1);
        assert_eq!(info.tiling, vk::ImageTiling::OPTIMAL);
        assert_eq!(info.sharing_mode, vk::SharingMode::EXCLUSIVE);
        assert_eq!(info.initial_layout, vk::ImageLayout::UNDEFINED);
        assert!(info.usage.contains(vk::ImageUsageFlags::SAMPLED));
        assert!(info.usage.contains(vk::ImageUsageFlags::TRANSFER_DST));
        assert!(!info.usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT));
    }

    #[test]
    fn a_multisampled_image_is_refused_rather_than_assumed_supported() {
        // Nothing has proved a multisample row and no recipe asks for one, so the
        // refusal happens here instead of at the driver.
        assert!(image_create_info(&desc(TextureFormat::Rgba8Unorm, 4), TextureUsage::empty()).is_none());
    }

    #[test]
    fn a_zero_extent_is_refused() {
        let mut zero = desc(TextureFormat::Rgba8Unorm, 1);
        zero.extent.width = 0;
        assert!(image_create_info(&zero, TextureUsage::empty()).is_none());
    }

    #[test]
    fn mip_levels_and_array_layers_are_at_least_one() {
        let mut sparse = desc(TextureFormat::Rgba8Unorm, 1);
        sparse.mip_levels = 0;
        sparse.array_layers = 0;
        let info = image_create_info(&sparse, TextureUsage::empty()).expect("supported");
        assert_eq!(info.mip_levels, 1);
        assert_eq!(info.array_layers, 1);
    }

    #[test]
    fn a_real_image_is_created_and_destroyed_on_this_machine() {
        // The texture half of step 4 against the real driver, again without an
        // allocator: an image handle is independent of the memory bound to it.
        use crate::Validation;
        use crate::native::vulkan::open;

        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let declared = of(&[TextureUsageKind::Sampled, TextureUsageKind::CopyDestination]);
        let info =
            image_create_info(&desc(TextureFormat::Rgba8Unorm, 1), declared).expect("supported");
        // SAFETY: the device is live and owns the create/destroy entry points; the
        // handle is destroyed exactly once below and never stored.
        let handle = unsafe { opened.device.device().create_image(&info, None) }
            .expect("a valid image description");
        // SAFETY: the handle came from this device and is destroyed here once,
        // before any memory was bound to it, which is the legal order.
        unsafe { opened.device.device().destroy_image(handle, None) };
    }

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
    fn a_layered_two_dimensional_texture_gets_an_array_view() {
        // A `TYPE_2D` view of a layered image silently selects layer zero, which is
        // a different resource than the one the description named.
        let single = desc(TextureFormat::Rgba8Unorm, 1);
        assert_eq!(view_type(&single), Some(vk::ImageViewType::TYPE_2D));

        let mut layered = single;
        layered.array_layers = 4;
        assert_eq!(view_type(&layered), Some(vk::ImageViewType::TYPE_2D_ARRAY));
    }

    #[test]
    fn each_dimension_maps_to_its_own_view_type() {
        let mut one = desc(TextureFormat::Rgba8Unorm, 1);
        one.dimension = TextureDimension::D1;
        assert_eq!(view_type(&one), Some(vk::ImageViewType::TYPE_1D));

        let mut three = desc(TextureFormat::Rgba8Unorm, 1);
        three.dimension = TextureDimension::D3;
        assert_eq!(view_type(&three), Some(vk::ImageViewType::TYPE_3D));
    }

    #[test]
    fn a_view_spans_the_declared_levels_and_layers_and_the_colour_aspect() {
        let mut described = desc(TextureFormat::Rgba8Unorm, 1);
        described.mip_levels = 3;
        described.array_layers = 2;
        let info = view_create_info(&described, vk::Image::null()).expect("a supported view");
        assert_eq!(info.view_type, vk::ImageViewType::TYPE_2D_ARRAY);
        assert_eq!(info.format, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(
            info.subresource_range.aspect_mask,
            vk::ImageAspectFlags::COLOR
        );
        assert_eq!(info.subresource_range.base_mip_level, 0);
        assert_eq!(info.subresource_range.level_count, 3);
        assert_eq!(info.subresource_range.base_array_layer, 0);
        assert_eq!(info.subresource_range.layer_count, 2);
        // The default component mapping is the identity swizzle, which is what "the
        // image's own channels" means. Compared field by field because ash's
        // `ComponentMapping` derives no `PartialEq`.
        assert!(info.components.r == vk::ComponentSwizzle::IDENTITY);
        assert!(info.components.g == vk::ComponentSwizzle::IDENTITY);
        assert!(info.components.b == vk::ComponentSwizzle::IDENTITY);
        assert!(info.components.a == vk::ComponentSwizzle::IDENTITY);
    }

    #[test]
    fn a_depth_texture_view_takes_the_depth_aspect() {
        let info = view_create_info(&desc(TextureFormat::Depth32Float, 1), vk::Image::null())
            .expect("a supported view");
        assert_eq!(
            info.subresource_range.aspect_mask,
            vk::ImageAspectFlags::DEPTH
        );
    }

    #[test]
    fn zero_levels_or_layers_become_the_one_the_image_was_created_with() {
        // The image lowering floors both at one, so the view has to span the same
        // range or it would name levels the image does not have.
        let mut sparse = desc(TextureFormat::Rgba8Unorm, 1);
        sparse.mip_levels = 0;
        sparse.array_layers = 0;
        let info = view_create_info(&sparse, vk::Image::null()).expect("a supported view");
        assert_eq!(info.subresource_range.level_count, 1);
        assert_eq!(info.subresource_range.layer_count, 1);
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
