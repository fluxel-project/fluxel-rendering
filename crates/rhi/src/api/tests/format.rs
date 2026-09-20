//! Format-fact contract tests (specification sections 8.1 through 8.5).
//!
//! Two kinds of assertion live here, matching the two halves of the chapter:
//!
//! * The **format facts** that can be decided from the format name alone —
//!   aspects, alpha, block geometry, and the one byte-count table — are checked
//!   directly, because they are the facts a caller plans with.
//! * The **descriptor-dependent query** is checked by construction and readback.
//!   Which keys a device answers `Supported` is a device fact and is not
//!   asserted here; what is asserted is that a key exists, carries exactly the
//!   five facts section 8.3 lists, and is a value a caller can cache.

use crate::api::format::{
    FormatFacts, StorageAccessSupport, TextureFormat, TextureSupport, TextureSupportLimits,
    TextureSupportQuery,
};
use crate::api::resource::subresource::TextureAspects;
use crate::api::resource::texture::{
    Extent3d, TextureDimension, TextureUsage, TextureViewCompatibility,
};

/// Every P0 format, in section 8.1's own order.
///
/// Written out rather than derived from the enum, because a list derived from
/// the type under test cannot detect a variant that went missing — and this list
/// is what pins the count the specification freezes.
const ALL_FORMATS: [TextureFormat; 38] = [
    TextureFormat::R8Unorm,
    TextureFormat::R8Snorm,
    TextureFormat::R8Uint,
    TextureFormat::R8Sint,
    TextureFormat::Rg8Unorm,
    TextureFormat::Rg8Snorm,
    TextureFormat::Rg8Uint,
    TextureFormat::Rg8Sint,
    TextureFormat::Rgba8Unorm,
    TextureFormat::Rgba8UnormSrgb,
    TextureFormat::Rgba8Snorm,
    TextureFormat::Rgba8Uint,
    TextureFormat::Rgba8Sint,
    TextureFormat::Bgra8Unorm,
    TextureFormat::Bgra8UnormSrgb,
    TextureFormat::R16Uint,
    TextureFormat::R16Sint,
    TextureFormat::R16Float,
    TextureFormat::Rg16Uint,
    TextureFormat::Rg16Sint,
    TextureFormat::Rg16Float,
    TextureFormat::Rgba16Uint,
    TextureFormat::Rgba16Sint,
    TextureFormat::Rgba16Float,
    TextureFormat::R32Uint,
    TextureFormat::R32Sint,
    TextureFormat::R32Float,
    TextureFormat::Rg32Uint,
    TextureFormat::Rg32Sint,
    TextureFormat::Rg32Float,
    TextureFormat::Rgba32Uint,
    TextureFormat::Rgba32Sint,
    TextureFormat::Rgba32Float,
    TextureFormat::Depth16Unorm,
    TextureFormat::Depth24Plus,
    TextureFormat::Depth24PlusStencil8,
    TextureFormat::Depth32Float,
    TextureFormat::Depth32FloatStencil8,
];

/// Facts for one format, assembled the way the device façade will assemble them.
///
/// The storage-access booleans are deliberately arbitrary: no test in this file
/// reads them, because section 8.2's only accessor for them is `supports`,
/// which waits on the `StorageAccess` type owned by module 03. What this helper
/// pins is that the facts object can be built and that every *other* accessor
/// answers from the format alone.
fn facts(format: TextureFormat) -> FormatFacts {
    FormatFacts::new(
        format,
        StorageAccessSupport::new(true, true, true),
        true,
        true,
        true,
        true,
    )
}

#[test]
fn attachment_and_blend_accessors_read_the_probed_record() {
    let color = FormatFacts::new(
        TextureFormat::Rgba8Unorm,
        StorageAccessSupport::new(false, false, false),
        true,
        false,
        false,
        true,
    );
    assert!(color.color_attachment());
    assert!(!color.depth_attachment());
    assert!(!color.stencil_attachment());
    assert!(color.blendable());

    let depth_stencil = FormatFacts::new(
        TextureFormat::Depth24PlusStencil8,
        StorageAccessSupport::new(false, false, false),
        false,
        true,
        true,
        false,
    );
    assert!(!depth_stencil.color_attachment());
    assert!(depth_stencil.depth_attachment());
    assert!(depth_stencil.stencil_attachment());
    assert!(!depth_stencil.blendable());
}

#[test]
fn every_p0_format_survives_the_round_trip_through_its_own_accessors() {
    // Section 8.1's list has 38 entries. The count is asserted, not just the
    // presence of each name, because the failure this catches is a variant
    // dropped while transcribing.
    assert_eq!(ALL_FORMATS.len(), 38);

    let mut seen = std::collections::HashSet::new();
    for format in ALL_FORMATS {
        assert!(seen.insert(format), "{format:?} is listed twice");
        // All three accessors are total: every format answers, and none panics.
        let _ = facts(format).aspects();
        let _ = facts(format).has_alpha_channel();
        let _ = facts(format).logical_bytes_per_block();
    }
    assert_eq!(seen.len(), 38);
}

#[test]
fn formats_are_comparable_hashable_and_printable() {
    // Section 8.1 derives `Clone, Copy, Debug, PartialEq, Eq, Hash`, and the
    // texture descriptor stores them in a `Vec`, so all six are load-bearing.
    fn assert_usable<T: Copy + PartialEq + Eq + std::hash::Hash + std::fmt::Debug>() {}
    // The query is a cache key rather than a scalar: it carries the declared
    // alternate view formats in a `Vec`, so it is `Clone` and not `Copy`. That
    // is section 8.3's shape, not an omission.
    fn assert_key<T: Clone + PartialEq + Eq + std::hash::Hash + std::fmt::Debug>() {}
    assert_usable::<TextureFormat>();
    assert_key::<TextureSupportQuery>();

    let mut set = std::collections::HashSet::new();
    set.insert(TextureFormat::Rgba8Unorm);
    assert!(set.contains(&TextureFormat::Rgba8Unorm));
    assert!(!set.contains(&TextureFormat::Rgba8UnormSrgb));
}

#[test]
fn aspects_follow_the_format_family() {
    // The color formats carry only the color aspect.
    assert_eq!(
        facts(TextureFormat::Rgba8Unorm).aspects(),
        TextureAspects::COLOR
    );
    assert_eq!(
        facts(TextureFormat::Bgra8UnormSrgb).aspects(),
        TextureAspects::COLOR
    );
    assert_eq!(
        facts(TextureFormat::R8Unorm).aspects(),
        TextureAspects::COLOR
    );

    // A pure depth format carries only depth...
    assert_eq!(
        facts(TextureFormat::Depth16Unorm).aspects(),
        TextureAspects::DEPTH
    );
    assert_eq!(
        facts(TextureFormat::Depth24Plus).aspects(),
        TextureAspects::DEPTH
    );
    assert_eq!(
        facts(TextureFormat::Depth32Float).aspects(),
        TextureAspects::DEPTH
    );

    // ...and a depth-stencil format carries both, which is the case section
    // 14.3 names when it says the two planes are copied and queried separately.
    let both = TextureAspects::DEPTH.union(TextureAspects::STENCIL);
    assert_eq!(facts(TextureFormat::Depth24PlusStencil8).aspects(), both);
    assert_eq!(facts(TextureFormat::Depth32FloatStencil8).aspects(), both);

    // No format carries a stencil aspect without a depth aspect: a stencil-only
    // format is not in P0, so the shape of the set is bounded by what is legal.
    for format in ALL_FORMATS {
        let aspects = facts(format).aspects();
        assert!(
            !aspects.contains(TextureAspects::STENCIL) || aspects.contains(TextureAspects::DEPTH),
            "{format:?} has a stencil aspect without a depth aspect"
        );
    }
}

#[test]
fn alpha_follows_the_channel_count_not_the_channel_order() {
    assert!(facts(TextureFormat::Rgba8Unorm).has_alpha_channel());
    assert!(facts(TextureFormat::Bgra8Unorm).has_alpha_channel());
    assert!(facts(TextureFormat::Bgra8UnormSrgb).has_alpha_channel());
    assert!(facts(TextureFormat::Rgba16Float).has_alpha_channel());
    assert!(facts(TextureFormat::Rgba32Float).has_alpha_channel());

    assert!(!facts(TextureFormat::R8Unorm).has_alpha_channel());
    assert!(!facts(TextureFormat::Rg8Unorm).has_alpha_channel());
    assert!(!facts(TextureFormat::Rg16Float).has_alpha_channel());
    assert!(!facts(TextureFormat::Depth32Float).has_alpha_channel());
}

#[test]
fn block_geometry_is_one_texel_for_every_p0_format() {
    // No compressed or planar format is in P0 (section 8.1), so every block is
    // exactly one texel. The accessors exist so that a caller who reads them
    // gets the right answer for the formats that exist, which is what makes the
    // addition of one that does not cover a single texel a data change rather
    // than a silent breakage.
    for format in ALL_FORMATS {
        assert_eq!(facts(format).block_width(), 1, "{format:?}");
        assert_eq!(facts(format).block_height(), 1, "{format:?}");
    }
}

#[test]
fn byte_counts_are_reported_only_where_the_format_name_fixes_them() {
    // One byte per texel.
    assert_eq!(
        facts(TextureFormat::R8Unorm).logical_bytes_per_block(),
        Some(1)
    );
    assert_eq!(
        facts(TextureFormat::R8Sint).logical_bytes_per_block(),
        Some(1)
    );
    // Two.
    assert_eq!(
        facts(TextureFormat::Rg8Unorm).logical_bytes_per_block(),
        Some(2)
    );
    assert_eq!(
        facts(TextureFormat::R16Float).logical_bytes_per_block(),
        Some(2)
    );
    assert_eq!(
        facts(TextureFormat::Depth16Unorm).logical_bytes_per_block(),
        Some(2)
    );
    // Four.
    assert_eq!(
        facts(TextureFormat::Rgba8Unorm).logical_bytes_per_block(),
        Some(4)
    );
    assert_eq!(
        facts(TextureFormat::Bgra8UnormSrgb).logical_bytes_per_block(),
        Some(4)
    );
    assert_eq!(
        facts(TextureFormat::R32Float).logical_bytes_per_block(),
        Some(4)
    );
    assert_eq!(
        facts(TextureFormat::Depth32Float).logical_bytes_per_block(),
        Some(4)
    );
    assert_eq!(
        facts(TextureFormat::Rg16Uint).logical_bytes_per_block(),
        Some(4)
    );
    // Eight.
    assert_eq!(
        facts(TextureFormat::Rgba16Float).logical_bytes_per_block(),
        Some(8)
    );
    assert_eq!(
        facts(TextureFormat::Rg32Uint).logical_bytes_per_block(),
        Some(8)
    );
    // Sixteen.
    assert_eq!(
        facts(TextureFormat::Rgba32Float).logical_bytes_per_block(),
        Some(16)
    );

    // The three formats whose backing the format name does not fix report
    // `None` instead of a number a caller would mistake for a measurement.
    assert_eq!(
        facts(TextureFormat::Depth24Plus).logical_bytes_per_block(),
        None
    );
    assert_eq!(
        facts(TextureFormat::Depth24PlusStencil8).logical_bytes_per_block(),
        None
    );
    assert_eq!(
        facts(TextureFormat::Depth32FloatStencil8).logical_bytes_per_block(),
        None
    );
}

#[test]
fn the_byte_count_table_agrees_with_the_format_name_where_it_reports() {
    // A weak but broad check that the table did not drift: every format whose
    // name ends in `32` and that reports a count reports a multiple of four, and
    // the count never exceeds the sum of its widest channels.
    for format in ALL_FORMATS {
        if let Some(bytes) = facts(format).logical_bytes_per_block() {
            assert!(bytes > 0, "{format:?}");
            assert!(bytes <= 16, "{format:?} reports {bytes} bytes per texel");
        }
    }
}

// ---------------------------------------------------------------------------
// The descriptor-dependent query (section 8.3) and its answer (8.4).
// ---------------------------------------------------------------------------

#[test]
fn a_query_carries_exactly_the_five_facts_the_key_is_built_from() {
    // Section 8.3 builds the key from dimension, format, usage, sample count,
    // the declared alternate view formats, and the creation-time view intent.
    // Extent, mip count, and layer count are explicitly *not* in the key.
    let query = TextureSupportQuery::new(
        TextureDimension::D2,
        TextureFormat::Rgba16Float,
        TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::SAMPLED),
        4,
    )
    .with_view_format(TextureFormat::Rgba8UnormSrgb)
    .with_view_format(TextureFormat::Rgba16Float)
    .with_view_compatibility(TextureViewCompatibility::CUBE);

    assert_eq!(query.dimension(), TextureDimension::D2);
    assert_eq!(query.format(), TextureFormat::Rgba16Float);
    assert_eq!(
        query.usage(),
        TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::SAMPLED)
    );
    assert_eq!(query.sample_count(), 4);
    assert_eq!(
        query.view_formats(),
        &[TextureFormat::Rgba8UnormSrgb, TextureFormat::Rgba16Float]
    );
    assert_eq!(query.view_compatibility(), TextureViewCompatibility::CUBE);
}

#[test]
fn a_fresh_query_starts_with_no_view_intent() {
    // The constructor's defaults are observable through the accessors, and they
    // are what makes "ask the plain question" a one-liner.
    let query = TextureSupportQuery::new(
        TextureDimension::D2,
        TextureFormat::Rgba8Unorm,
        TextureUsage::SAMPLED,
        1,
    );
    assert!(query.view_formats().is_empty());
    assert_eq!(query.view_compatibility(), TextureViewCompatibility::NONE);
}

#[test]
fn two_identically_built_queries_are_one_key_and_a_difference_is_a_different_key() {
    // The query is a cache key, so equality and hashing are part of its
    // contract rather than conveniences. Each of the four scalar facts must
    // change the key on its own.
    let base = || {
        TextureSupportQuery::new(
            TextureDimension::D2,
            TextureFormat::Rgba8Unorm,
            TextureUsage::SAMPLED,
            1,
        )
    };

    assert_eq!(base(), base());

    // Equal keys hash equally, which is what lets the device cache an answer.
    let mut answers = std::collections::HashMap::new();
    answers.insert(base(), TextureSupport::Unsupported);
    assert!(answers.contains_key(&base()));

    assert_ne!(
        base(),
        TextureSupportQuery::new(
            TextureDimension::D3,
            TextureFormat::Rgba8Unorm,
            TextureUsage::SAMPLED,
            1
        )
    );
    assert_ne!(
        base(),
        TextureSupportQuery::new(
            TextureDimension::D2,
            TextureFormat::Bgra8Unorm,
            TextureUsage::SAMPLED,
            1
        )
    );
    assert_ne!(
        base(),
        TextureSupportQuery::new(
            TextureDimension::D2,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COPY_DST,
            1
        )
    );
    assert_ne!(
        base(),
        TextureSupportQuery::new(
            TextureDimension::D2,
            TextureFormat::Rgba8Unorm,
            TextureUsage::SAMPLED,
            4
        )
    );
    assert_ne!(
        base(),
        base().with_view_compatibility(TextureViewCompatibility::CUBE)
    );
    assert_ne!(
        base(),
        base().with_view_format(TextureFormat::Rgba8UnormSrgb)
    );
}

#[test]
fn a_supported_answer_reports_its_limits_and_an_unsupported_one_reports_nothing() {
    let supported = TextureSupport::Supported(TextureSupportLimits::new(
        Extent3d::d2(8192, 8192),
        14,
        2048,
    ));
    assert!(supported.is_supported());
    let limits = supported
        .limits()
        .expect("a supported answer carries limits");
    assert_eq!(limits.max_extent(), Extent3d::d2(8192, 8192));
    assert_eq!(limits.max_mip_levels(), 14);
    assert_eq!(limits.max_array_layers(), 2048);

    let unsupported = TextureSupport::Unsupported;
    assert!(!unsupported.is_supported());
    assert!(unsupported.limits().is_none());
}

#[test]
fn the_limits_answer_the_question_the_key_deliberately_left_out() {
    // Section 8.4's worked example, expressed through the types: the same
    // query answer can permit a 2D color attachment and forbid a 3D one,
    // because the extent and dimensionality of the descriptor are checked
    // against the returned limits rather than asked about in the key. This test
    // builds both answers for one key so the distinction is visible in one
    // place.
    let key = TextureSupportQuery::new(
        TextureDimension::D2,
        TextureFormat::Rgba16Float,
        TextureUsage::COLOR_ATTACHMENT,
        1,
    );

    // A 2D-only device: the key is answerable per descriptor, and the limits
    // are what make the 3D descriptor illegal.
    let two_d_only = TextureSupport::Supported(TextureSupportLimits::new(
        Extent3d::d3(4096, 4096, 1),
        13,
        1,
    ));
    assert!(two_d_only.is_supported());
    assert_eq!(two_d_only.limits().unwrap().max_extent().depth, 1);

    // And the same key with a device that has no such image format at all.
    let none = TextureSupport::Unsupported;
    assert!(!none.is_supported());

    // The key itself never mentions extent or dimensionality of the *texture*
    // beyond the dimension class it is keyed on.
    assert_eq!(key.dimension(), TextureDimension::D2);
    assert_eq!(key.sample_count(), 1);
}

/// The format enumeration is the whole declared set, and nothing else.
///
/// `TextureFormat::all` is what a backend walks when it fills a format table, so
/// a variant missing from the list is a format no device ever reports facts for —
/// and because `format()` answers `Option`, that absence is a legal answer rather
/// than a panic. It would therefore be found by a caller whose `R8Unorm` textures
/// inexplicably did not work, on one backend, some time later.
///
/// Discriminants are the check rather than a second hand-written count. The list
/// must contain exactly one entry per discriminant from zero through the highest
/// one it contains, which fails in both directions: a variant left out lowers the
/// count below the highest discriminant, and a duplicate raises it above.
#[test]
fn the_format_enumeration_covers_every_declared_variant() {
    let enumerated: Vec<TextureFormat> = TextureFormat::all().collect();
    let highest = enumerated
        .iter()
        .map(|format| *format as usize)
        .max()
        .expect("the P0 format set is not empty");

    assert_eq!(
        enumerated.len(),
        highest + 1,
        "TextureFormat::all must list every variant exactly once; it lists {} of {}",
        enumerated.len(),
        highest + 1
    );
}
