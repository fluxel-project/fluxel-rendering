//! Step 11's per-format evidence: what one physical device proved about each
//! portable format this backend was taught.
//!
//! # The query is the evidence
//!
//! Step 11's wording is exact: **a format fact is recorded only where
//! `vkGetPhysicalDeviceFormatProperties` proved it.** So there is no profile table
//! here and no default: [`capabilities`] turns one driver answer into the common
//! facts, and [`record_mapped`] asks the driver once per format this backend maps.
//! A format the backend has not been taught has no query at all and is refused by
//! [`FormatError::Unsupported`], which is the same refusal
//! [`super::format::image_format`]'s `None` already makes for a texture.
//!
//! # Optimal tiling is the tiling this backend creates
//!
//! `vkGetPhysicalDeviceFormatProperties` answers three flag sets -- linear tiling,
//! optimal tiling and buffer. Every image this backend creates is `OPTIMAL`
//! (`texture::image_create_info`), and no format this layer knows is a vertex or
//! texel-buffer format, so `optimal_tiling_features` is the only one that describes
//! a resource that can exist. The other two are deliberately not folded in: a
//! union of the three would claim capabilities for an image shape the graph cannot
//! name.
//!
//! # One sample count, and why
//!
//! The portable table keys on `(format, sample count)` because the GL family's
//! renderbuffer rows differ per count. This backend refuses every multisampled
//! image description ([`texture::image_create_info`] returns `None` for
//! `sample_count != 1`), and `vkGetPhysicalDeviceFormatProperties` answers per
//! format rather than per sample count, so every fact read here belongs to
//! [`SINGLE_SAMPLE`]. The multisampled rows arrive with the query that proves
//! them (`vkGetPhysicalDeviceImageFormatProperties`, per usage) when a
//! multisampled resource first exists -- recording a count this backend cannot
//! create would be a capability claim with no object behind it.

use ash::vk;
use fluxel_rendergraph::TextureFormat;

use crate::common::formats::{
    FormatCapabilities, FormatEvidence, FormatTable, FormatTableError,
};

use super::format::{MAPPED, image_format};

/// The sample count every fact this module reads describes.
///
/// Not a default: it is the only count this backend creates an image with, and
/// `vkGetPhysicalDeviceFormatProperties` has no per-count dimension to read.
pub(crate) const SINGLE_SAMPLE: u32 = 1;

/// Why a format's facts could not be read or recorded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FormatError {
    /// The portable format has no `Vulkan` image format in this backend, so there
    /// is no format to ask about.
    ///
    /// Unreachable from this crate today, because a `#[non_exhaustive]` foreign enum
    /// cannot be given a variant its defining crate does not declare. It is a real
    /// path across crate versions -- a `TextureFormat` added upstream must not
    /// become a silently absent row -- which is why it is a value rather than an
    /// `expect` on the mapping.
    Unsupported(TextureFormat),
    /// The facts contradicted a row the table already held.
    Table(FormatTableError),
}

/// Lowers one `vkGetPhysicalDeviceFormatProperties` answer into the common facts.
///
/// Every fact comes from `optimal_tiling_features` and from nothing else, and the
/// flags the portable table does not model (`BLIT_SRC`, `BLIT_DST`,
/// `UNIFORM_TEXEL_BUFFER`, the texel-buffer and chroma bits) are deliberately
/// dropped rather than mapped onto a neighbouring fact.
pub(crate) fn capabilities(
    format: TextureFormat,
    properties: &vk::FormatProperties,
) -> FormatCapabilities {
    let features = properties.optimal_tiling_features;
    FormatCapabilities {
        format,
        sample_count: SINGLE_SAMPLE,
        // The driver was asked about exactly this format, which is the evidence an
        // explicit API offers and the reason the table may record a storage fact
        // from here at all.
        evidence: FormatEvidence::OperationProbed,
        sampled: features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE),
        // The linear-filter bit, not the sampled bit: a format can be sampled
        // without being filterable, and the retained linear-clamp recipe's legality
        // depends on the difference.
        filterable: features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR),
        // One fact for both attachment kinds, because the portable vocabulary asks
        // one question. Which kind a format belongs to is asked of the mapped
        // `Vulkan` format (`format::is_depth`), never inferred from this flag.
        renderable: features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT)
            || features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT),
        blendable: features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND),
        // `Vulkan` spells one bit for both storage directions, so both facts are
        // that one proof. They stay separate fields because the portable table
        // refuses to make either imply the other.
        storage_read: features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE),
        storage_write: features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE),
        copy_source: features.contains(vk::FormatFeatureFlags::TRANSFER_SRC),
        copy_destination: features.contains(vk::FormatFeatureFlags::TRANSFER_DST),
    }
}

/// Reads one portable format's facts from the driver.
///
/// The format is mapped first, so a format this backend cannot name is refused
/// before the driver is reached -- the same order `texture::image_create_info`
/// already uses.
pub(crate) fn query(
    instance: &ash::Instance,
    adapter: vk::PhysicalDevice,
    format: TextureFormat,
) -> Result<FormatCapabilities, FormatError> {
    let image_format = image_format(format).ok_or(FormatError::Unsupported(format))?;
    // SAFETY: the adapter was enumerated from this instance, which is still live;
    // the call creates nothing and only reports the driver's own answer.
    let properties =
        unsafe { instance.get_physical_device_format_properties(adapter, image_format) };
    Ok(capabilities(format, &properties))
}

/// Reads and records the facts of every format this backend maps.
///
/// The set is [`MAPPED`], which is the same list [`image_format`] lowers, so the
/// table cannot hold a format no query asked about and cannot omit one the backend
/// can create.
pub(crate) fn record_mapped(
    instance: &ash::Instance,
    adapter: vk::PhysicalDevice,
    table: &mut FormatTable,
) -> Result<(), FormatError> {
    for format in MAPPED {
        let facts = query(instance, adapter, format)?;
        table.record(facts).map_err(FormatError::Table)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn properties(features: vk::FormatFeatureFlags) -> vk::FormatProperties {
        vk::FormatProperties {
            optimal_tiling_features: features,
            ..Default::default()
        }
    }

    #[test]
    fn each_optimal_tiling_flag_lowers_to_its_own_fact() {
        let facts = capabilities(
            TextureFormat::Rgba8Unorm,
            &properties(
                vk::FormatFeatureFlags::SAMPLED_IMAGE
                    | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
                    | vk::FormatFeatureFlags::COLOR_ATTACHMENT
                    | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
                    | vk::FormatFeatureFlags::TRANSFER_SRC
                    | vk::FormatFeatureFlags::TRANSFER_DST,
            ),
        );
        assert!(facts.sampled);
        assert!(facts.filterable);
        assert!(facts.renderable);
        assert!(facts.blendable);
        assert!(facts.copy_source);
        assert!(facts.copy_destination);
        // Storage is the one fact those flags did not carry, so it stays false
        // rather than being inferred from the others.
        assert!(!facts.storage_read);
        assert!(!facts.storage_write);
        assert_eq!(facts.evidence, FormatEvidence::OperationProbed);
        assert_eq!(facts.sample_count, SINGLE_SAMPLE);
        assert_eq!(facts.format, TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn a_flag_the_table_does_not_model_does_not_lower_to_one() {
        // `BLIT_SRC` is a real `Vulkan` capability the portable table deliberately
        // does not carry. It must not leak into a neighbouring fact such as
        // `copy_source`, which is a different operation.
        let facts = capabilities(
            TextureFormat::Rgba8Unorm,
            &properties(vk::FormatFeatureFlags::BLIT_SRC),
        );
        assert!(!facts.sampled);
        assert!(!facts.copy_source);
        assert!(!facts.copy_destination);
        assert!(!facts.renderable);
    }

    #[test]
    fn the_renderable_fact_covers_both_attachment_kinds() {
        let colour = capabilities(
            TextureFormat::Rgba8Unorm,
            &properties(vk::FormatFeatureFlags::COLOR_ATTACHMENT),
        );
        assert!(colour.renderable);
        assert!(!colour.blendable);

        let depth = capabilities(
            TextureFormat::Depth32Float,
            &properties(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT),
        );
        assert!(depth.renderable);
        // A depth attachment is not a colour blend target, and the flag says so.
        assert!(!depth.blendable);
    }

    #[test]
    fn the_storage_direction_is_one_proof_for_both() {
        let with = capabilities(
            TextureFormat::Rgba8Unorm,
            &properties(vk::FormatFeatureFlags::STORAGE_IMAGE),
        );
        assert!(with.storage_read);
        assert!(with.storage_write);

        let without = capabilities(
            TextureFormat::Rgba8Unorm,
            &properties(vk::FormatFeatureFlags::empty()),
        );
        assert!(!without.storage_read);
        assert!(!without.storage_write);
    }

    #[test]
    fn a_storage_fact_the_driver_proved_passes_the_table_rule() {
        // The lowering's evidence has to be the evidence the table demands, or the
        // one fact no profile guarantees could never be recorded at all.
        let mut table = FormatTable::default();
        let facts = capabilities(
            TextureFormat::Rgba8Unorm,
            &properties(vk::FormatFeatureFlags::STORAGE_IMAGE),
        );
        assert_eq!(table.record(facts), Ok(()));
        assert_eq!(
            table.get(TextureFormat::Rgba8Unorm, SINGLE_SAMPLE),
            Some(facts)
        );
    }

    #[test]
    fn an_empty_answer_is_a_proved_negative_rather_than_an_absent_row() {
        let mut table = FormatTable::default();
        let nothing = capabilities(
            TextureFormat::Rgba8UnormSrgb,
            &properties(vk::FormatFeatureFlags::empty()),
        );
        assert_eq!(table.record(nothing), Ok(()));
        let read_back = table
            .get(TextureFormat::Rgba8UnormSrgb, SINGLE_SAMPLE)
            .expect("the pair was examined");
        assert_eq!(read_back, nothing);
        assert!(!read_back.sampled);
        // A pair nothing asked about is still absent, so the two answers stay
        // different sentences.
        assert_eq!(table.get(TextureFormat::Rgba8Unorm, SINGLE_SAMPLE), None);
    }

    #[test]
    fn a_real_adapter_proves_every_format_this_backend_maps() {
        // Step 11 against the real driver. A machine with no Vulkan adapter returns
        // before asserting, because having no GPU is not this test's subject.
        use crate::Validation;
        use crate::native::vulkan::{adapter, instance};

        let Ok(instance) = instance::open(Validation::Disabled) else {
            return;
        };
        let Ok(adapters) = adapter::enumerate(instance.instance()) else {
            return;
        };
        if adapters.is_empty() {
            return;
        }
        let index = adapter::select(adapters.len(), 0).expect("index zero exists");

        let mut table = FormatTable::default();
        record_mapped(instance.instance(), adapters[index], &mut table)
            .expect("every mapped format is one the driver can report on");

        let formats: Vec<TextureFormat> = table.iter().map(|facts| facts.format).collect();
        assert_eq!(
            formats,
            MAPPED.to_vec(),
            "the table holds exactly the formats this backend asked about, in its own order"
        );
        for facts in table.iter() {
            assert_eq!(facts.sample_count, SINGLE_SAMPLE);
            assert_eq!(facts.evidence, FormatEvidence::OperationProbed);
        }

        // The facts asserted below are the ones `Vulkan` makes mandatory for these
        // formats with optimal tiling, so a failure is a real disagreement rather
        // than an optional capability this board happens to lack.
        let colour = table
            .get(TextureFormat::Rgba8Unorm, SINGLE_SAMPLE)
            .expect("the backend asked about it");
        assert!(colour.sampled, "R8G8B8A8_UNORM is a mandatory sampled format");
        assert!(colour.renderable, "and a mandatory colour attachment");
        assert!(colour.blendable);
        assert!(colour.copy_source);
        assert!(colour.copy_destination);

        let srgb = table
            .get(TextureFormat::Rgba8UnormSrgb, SINGLE_SAMPLE)
            .expect("the backend asked about it");
        assert!(srgb.sampled);
        assert!(srgb.renderable);

        let depth = table
            .get(TextureFormat::Depth32Float, SINGLE_SAMPLE)
            .expect("the backend asked about it");
        assert!(depth.renderable, "D32_SFLOAT is a mandatory depth attachment");
        assert!(!depth.blendable, "a depth format is not a colour blend target");
        assert!(depth.copy_source);
        assert!(depth.copy_destination);

        // A repeated discovery is accepted and changes nothing, which exercises the
        // identical-repeat rule against real driver answers.
        record_mapped(instance.instance(), adapters[index], &mut table)
            .expect("an identical repeat is not a contradiction");
        assert_eq!(table.iter().count(), MAPPED.len());
    }
}
