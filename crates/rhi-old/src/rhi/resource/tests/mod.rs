//! Contract tests for texture and buffer descriptors.
//!
//! The subject here is rhi-design section 48.1: a descriptor that participates
//! in a capability query, a fingerprint, a capture definition, or a statistics
//! identity comparison must be canonicalized first.

use super::super::format::TextureFormat;
use super::{TextureDescriptor, TextureUsage};

/// A texture with a non-trivial view-format set.
fn texture() -> TextureDescriptor {
    TextureDescriptor::new_2d(4, 4, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED)
}

#[test]
fn view_formats_are_sorted_and_deduplicated() {
    let descriptor = texture()
        .with_view_format(TextureFormat::Rgba8UnormSrgb)
        .with_view_format(TextureFormat::Bgra8Unorm)
        .with_view_format(TextureFormat::Rgba8UnormSrgb);

    // The raw descriptor keeps what the caller declared, including the repeat.
    assert_eq!(descriptor.view_formats.len(), 3);

    let canonical = descriptor.canonicalized();

    assert_eq!(
        canonical.view_formats.len(),
        2,
        "the repeat is deduplicated: {:?}",
        canonical.view_formats
    );
    // Which of two formats sorts first is the format type's own ordering, not
    // something section 48.1 fixes, so this asserts the shape the contract does
    // fix — that the result is sorted and holds exactly what was declared —
    // rather than pinning an order a future variant could legitimately change.
    // That the order is independent of declaration order is what
    // `declaration_order_does_not_change_the_canonical_form` proves.
    assert!(
        canonical.view_formats.is_sorted(),
        "the canonical form is sorted, not merely deduplicated: {:?}",
        canonical.view_formats
    );
    for declared in [TextureFormat::Rgba8UnormSrgb, TextureFormat::Bgra8Unorm] {
        assert!(
            canonical.view_formats.contains(&declared),
            "{declared:?} was declared and must survive canonicalization: {:?}",
            canonical.view_formats
        );
    }
}

#[test]
fn declaration_order_does_not_change_the_canonical_form() {
    let ascending = texture()
        .with_view_format(TextureFormat::Bgra8Unorm)
        .with_view_format(TextureFormat::Rgba8UnormSrgb);
    let descending = texture()
        .with_view_format(TextureFormat::Rgba8UnormSrgb)
        .with_view_format(TextureFormat::Bgra8Unorm);

    assert_ne!(ascending, descending, "the raw descriptors do differ");
    assert_eq!(
        ascending.canonicalized(),
        descending.canonicalized(),
        "two spellings of one texture must canonicalize to one descriptor"
    );
}

#[test]
fn a_texture_with_no_view_formats_canonicalizes_to_itself() {
    let descriptor = texture();

    assert_eq!(descriptor.canonicalized(), descriptor);
}

#[test]
fn a_repeated_view_format_is_deduplicated_rather_than_rejected() {
    let descriptor = texture()
        .with_view_format(TextureFormat::Bgra8Unorm)
        .with_view_format(TextureFormat::Bgra8Unorm);

    let canonical = descriptor.canonicalized();

    assert_eq!(canonical.view_formats, vec![TextureFormat::Bgra8Unorm]);
    // A set with a repeat is still a legal texture: the repeat carries no
    // extra meaning, unlike a view format equal to the base format below.
    assert!(canonical.normalize_and_validate().is_ok());
}

#[test]
fn a_view_format_equal_to_the_base_format_is_a_contradiction_not_a_duplicate() {
    let descriptor = texture().with_view_format(TextureFormat::Rgba8Unorm);

    // Canonicalization does not repair it, because dropping it would silently
    // accept a request the caller did not make.
    let canonical = descriptor.canonicalized();
    assert_eq!(canonical.view_formats, vec![TextureFormat::Rgba8Unorm]);

    let error = canonical
        .normalize_and_validate()
        .expect_err("a view format repeating the base format is rejected");
    assert!(error.message().contains("base format"), "{error}");
}

#[test]
fn usage_must_be_non_empty() {
    // An empty usage set has no named constant: it is the absence of every
    // flag, which is why this test lives inside the module that owns the bits.
    let descriptor =
        TextureDescriptor::new_2d(4, 4, TextureFormat::Rgba8Unorm, TextureUsage(0));

    let error = descriptor
        .normalize_and_validate()
        .expect_err("a texture with no usage is rejected");
    assert!(error.message().contains("usage"), "{error}");
}

#[test]
fn a_zero_extent_is_rejected() {
    let descriptor =
        TextureDescriptor::new_2d(0, 4, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED);

    let error = descriptor
        .normalize_and_validate()
        .expect_err("a zero extent is rejected before any backend sees it");
    assert!(error.message().contains("extent"), "{error}");
}
