//! The portable texture format lowered onto a `Vulkan` image format.
//!
//! Five formats exist in the portable contract, and every one of them maps to
//! exactly one `Vulkan` format. Two properties of that mapping are worth stating
//! because they are what the rest of the backend relies on:
//!
//! 1. **It has an explicit unsupported answer.** `TextureFormat` is
//!    `#[non_exhaustive]`, so this backend cannot match it exhaustively and the
//!    compiler cannot prove the mapping complete. That is not a defect to work
//!    around with a wildcard that invents a format: the mapping returns `None` for
//!    a portable format this backend has no equivalent for, so an unrecognized
//!    format is refused here, before any driver call, rather than becoming an
//!    `UNDEFINED` image the driver rejects later with a worse message.
//! 2. **It is injective.** No two portable formats may share one `Vulkan` format.
//!    `Rgba8Unorm` and `Rgba8UnormSrgb` carry the same bytes and differ only in
//!    whether sampling decodes them; collapsing the two would silently apply or
//!    drop an sRGB decode, which is the double-gamma mistake the surface format
//!    rules already refuse to make.
//!
//! A test pins the second property, since it is the one a copy-paste error in the
//! match would break without any other signal.

use ash::vk;
use fluxel_rendergraph::TextureFormat;

/// The portable formats this backend maps, in the portable enum's declaration
/// order.
///
/// [`TextureFormat`] is `#[non_exhaustive]` and offers no variant iterator, so the
/// list has to be written out. Keeping it here, beside the mapping it describes,
/// means the format-evidence query iterates the same set this module lowers -- a
/// format added upstream is simply not in the list and is refused by
/// [`image_format`]'s `None` rather than guessed at. A test pins what *is*
/// checkable: that every entry maps, and that the list has no duplicate.
pub(crate) const MAPPED: [TextureFormat; 5] = [
    TextureFormat::Rgba8Unorm,
    TextureFormat::Rgba8UnormSrgb,
    TextureFormat::Bgra8Unorm,
    TextureFormat::Rgba16Float,
    TextureFormat::Depth32Float,
];

/// The `Vulkan` format for one portable format, or `None` where this backend has
/// no equivalent.
pub(crate) fn image_format(format: TextureFormat) -> Option<vk::Format> {
    Some(match format {
        TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        // Kept distinct from the UNORM spelling above: encoded bytes and a decode
        // at sample time are not the same resource.
        TextureFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
        // A portable format added upstream, which this backend has not been taught.
        _ => return None,
    })
}

/// Whether a `Vulkan` image format carries depth rather than colour.
///
/// Asked of the *`Vulkan`* format rather than the portable one on purpose: the
/// portable enum is non-exhaustive, so answering there would need a second list to
/// keep in step with this one. The mapped format is the single source of truth, and
/// a caller has it by the time it asks.
pub(crate) fn is_depth(format: vk::Format) -> bool {
    format == vk::Format::D32_SFLOAT
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapped(format: TextureFormat) -> vk::Format {
        image_format(format).expect("a portable format this backend maps")
    }

    #[test]
    fn each_portable_format_maps_to_its_own_vulkan_format() {
        assert_eq!(mapped(TextureFormat::Rgba8Unorm), vk::Format::R8G8B8A8_UNORM);
        assert_eq!(
            mapped(TextureFormat::Rgba8UnormSrgb),
            vk::Format::R8G8B8A8_SRGB
        );
        assert_eq!(mapped(TextureFormat::Bgra8Unorm), vk::Format::B8G8R8A8_UNORM);
        assert_eq!(
            mapped(TextureFormat::Rgba16Float),
            vk::Format::R16G16B16A16_SFLOAT
        );
        assert_eq!(mapped(TextureFormat::Depth32Float), vk::Format::D32_SFLOAT);
    }

    #[test]
    fn every_mapped_format_is_taught_and_listed_once() {
        // The list is what the format-evidence query iterates, so a format missing
        // from it would be silently unexamined, and a duplicate would be queried
        // twice. Neither is expressible in the type, so both are asserted.
        for (index, format) in MAPPED.iter().enumerate() {
            assert!(
                image_format(*format).is_some(),
                "{format:?} is listed but not mapped"
            );
            for other in &MAPPED[index + 1..] {
                assert_ne!(format, other, "{format:?} is listed twice");
            }
        }
    }

    #[test]
    fn the_mapping_is_injective() {
        // A copy-paste in the match above would break this and nothing else.
        let mapped_all: Vec<vk::Format> = MAPPED.iter().copied().map(mapped).collect();
        for (index, format) in mapped_all.iter().enumerate() {
            for (other_index, other) in mapped_all.iter().enumerate() {
                if index != other_index {
                    assert_ne!(
                        format, other,
                        "{:?} and {:?} share a Vulkan format",
                        MAPPED[index], MAPPED[other_index]
                    );
                }
            }
        }
    }

    #[test]
    fn the_unorm_and_srgb_spellings_stay_distinct() {
        // The specific collapse the injectivity test guards, named so a reader
        // knows why it matters: sampling decodes sRGB and does not decode UNORM.
        assert_ne!(
            mapped(TextureFormat::Rgba8Unorm),
            mapped(TextureFormat::Rgba8UnormSrgb)
        );
    }

    #[test]
    fn only_the_depth_format_is_depth() {
        for format in MAPPED {
            assert_eq!(
                is_depth(mapped(format)),
                format == TextureFormat::Depth32Float,
                "{format:?}"
            );
        }
    }
}
