//! Section 14: subresources, origins and host texel layout.

use super::*;
use crate::api::error::RhiErrorKind;
use crate::api::format::TextureFormat;
use crate::api::resource::subresource::{
    HostTexelLayout, Origin3d, TextureAspect, TextureAspects, TextureSubresourceLayers,
    TextureSubresourceRange, aspect_bits, source_bytes_required, validate_host_texel_layout,
    validate_origin_extent, validate_subresource_layers, validate_subresource_range,
};
use crate::api::resource::texture::{Extent3d, TextureDimension};

#[test]
fn aspect_bits_bridge_the_single_and_set_forms() {
    assert_eq!(aspect_bits(TextureAspect::Color), TextureAspects::COLOR);
    assert_eq!(aspect_bits(TextureAspect::Depth), TextureAspects::DEPTH);
    assert_eq!(aspect_bits(TextureAspect::Stencil), TextureAspects::STENCIL);

    let both = TextureAspects::DEPTH.union(TextureAspects::STENCIL);
    assert!(both.contains(aspect_bits(TextureAspect::Depth)));
    assert!(both.contains(aspect_bits(TextureAspect::Stencil)));
    assert!(!both.contains(TextureAspects::COLOR));
    assert!(!both.is_empty());
}

#[test]
fn a_tracking_range_must_name_something() {
    let legal = TextureSubresourceRange {
        aspects: TextureAspects::COLOR,
        base_mip: 0,
        mip_count: 1,
        base_layer: 0,
        layer_count: 1,
    };
    assert!(validate_subresource_range(legal, TextureDimension::D2).is_ok());

    let empty_aspects = TextureSubresourceRange {
        aspects: TextureAspects::COLOR.union(TextureAspects::COLOR),
        ..legal
    };
    // Not empty, so accepted — see the usage tests for why the empty set is
    // unreachable from the public API.
    assert!(validate_subresource_range(empty_aspects, TextureDimension::D2).is_ok());

    let no_mips = TextureSubresourceRange {
        mip_count: 0,
        ..legal
    };
    assert_kind(
        validate_subresource_range(no_mips, TextureDimension::D2),
        RhiErrorKind::InvalidUsage,
    );

    let no_layers = TextureSubresourceRange {
        layer_count: 0,
        ..legal
    };
    assert_kind(
        validate_subresource_range(no_layers, TextureDimension::D2),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_3d_tracking_range_is_pinned_to_one_layer() {
    // Section 14.2: a Z slice of a 3D texture is not an independent array
    // subresource.
    let legal = TextureSubresourceRange {
        aspects: TextureAspects::COLOR,
        base_mip: 0,
        mip_count: 1,
        base_layer: 0,
        layer_count: 1,
    };
    assert!(validate_subresource_range(legal, TextureDimension::D3).is_ok());

    let with_a_layer = TextureSubresourceRange {
        base_layer: 1,
        ..legal
    };
    assert_kind(
        validate_subresource_range(with_a_layer, TextureDimension::D3),
        RhiErrorKind::InvalidUsage,
    );

    let with_layers = TextureSubresourceRange {
        layer_count: 2,
        ..legal
    };
    assert_kind(
        validate_subresource_range(with_layers, TextureDimension::D3),
        RhiErrorKind::InvalidUsage,
    );

    // The same range is legal for a 2D texture, so the refusal above is about
    // the dimension and not about the numbers.
    assert!(validate_subresource_range(with_a_layer, TextureDimension::D2).is_ok());
}

#[test]
fn a_copy_subresource_covers_one_mip_and_a_layer_range() {
    let legal = TextureSubresourceLayers {
        aspect: TextureAspect::Color,
        mip_level: 0,
        base_layer: 0,
        layer_count: 2,
    };
    assert!(validate_subresource_layers(legal, TextureDimension::D2).is_ok());

    let no_layers = TextureSubresourceLayers {
        layer_count: 0,
        ..legal
    };
    assert_kind(
        validate_subresource_layers(no_layers, TextureDimension::D2),
        RhiErrorKind::InvalidUsage,
    );

    // 1D and 3D have no array layers at all.
    assert_kind(
        validate_subresource_layers(legal, TextureDimension::D1),
        RhiErrorKind::InvalidUsage,
    );
    assert_kind(
        validate_subresource_layers(legal, TextureDimension::D3),
        RhiErrorKind::InvalidUsage,
    );

    let pinned = TextureSubresourceLayers {
        layer_count: 1,
        ..legal
    };
    assert!(validate_subresource_layers(pinned, TextureDimension::D1).is_ok());
    assert!(validate_subresource_layers(pinned, TextureDimension::D3).is_ok());
}

#[test]
fn origin_and_extent_follow_the_dimension() {
    // D1: no Y or Z at all.
    let d1 = Origin3d { x: 0, y: 0, z: 0 };
    assert!(validate_origin_extent(d1, Extent3d::d1(4), TextureDimension::D1).is_ok());
    assert_kind(
        validate_origin_extent(
            Origin3d { x: 0, y: 1, z: 0 },
            Extent3d::d1(4),
            TextureDimension::D1,
        ),
        RhiErrorKind::InvalidUsage,
    );
    assert_kind(
        validate_origin_extent(
            d1,
            Extent3d {
                width: 4,
                height: 2,
                depth: 1,
            },
            TextureDimension::D1,
        ),
        RhiErrorKind::InvalidUsage,
    );

    // D2: array layers, not Z.
    assert!(validate_origin_extent(d1, Extent3d::d2(4, 4), TextureDimension::D2).is_ok());
    assert_kind(
        validate_origin_extent(
            Origin3d { x: 0, y: 0, z: 1 },
            Extent3d::d2(4, 4),
            TextureDimension::D2,
        ),
        RhiErrorKind::InvalidUsage,
    );
    assert_kind(
        validate_origin_extent(
            d1,
            Extent3d {
                width: 4,
                height: 4,
                depth: 2,
            },
            TextureDimension::D2,
        ),
        RhiErrorKind::InvalidUsage,
    );

    // D3: Z is exactly how a slice range is expressed, so it is not refused.
    assert!(
        validate_origin_extent(
            Origin3d { x: 0, y: 0, z: 2 },
            Extent3d::d3(4, 4, 2),
            TextureDimension::D3
        )
        .is_ok()
    );

    // Every dimension refuses a zero-sized copy.
    for dimension in [
        TextureDimension::D1,
        TextureDimension::D2,
        TextureDimension::D3,
    ] {
        assert_kind(
            validate_origin_extent(d1, Extent3d::d3(0, 1, 1), dimension),
            RhiErrorKind::InvalidUsage,
        );
    }
}

#[test]
fn a_host_layout_is_checked_against_the_rows_it_must_cover() {
    let extent = Extent3d::d2(4, 2);
    let format = TextureFormat::Rgba8Unorm; // four bytes per texel, so 16 per row

    let legal = HostTexelLayout {
        bytes_per_row: 16,
        rows_per_image: 2,
    };
    assert!(validate_host_texel_layout(legal, extent, format).is_ok());

    // Spare space per row is allowed: the CPU source may be padded.
    let padded = HostTexelLayout {
        bytes_per_row: 32,
        rows_per_image: 2,
    };
    assert!(validate_host_texel_layout(padded, extent, format).is_ok());

    let too_short = HostTexelLayout {
        bytes_per_row: 12,
        rows_per_image: 2,
    };
    assert_kind(
        validate_host_texel_layout(too_short, extent, format),
        RhiErrorKind::InvalidUsage,
    );

    let misaligned = HostTexelLayout {
        bytes_per_row: 18,
        rows_per_image: 2,
    };
    assert_kind(
        validate_host_texel_layout(misaligned, extent, format),
        RhiErrorKind::InvalidUsage,
    );

    let too_few_rows = HostTexelLayout {
        bytes_per_row: 16,
        rows_per_image: 1,
    };
    assert_kind(
        validate_host_texel_layout(too_few_rows, extent, format),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_host_layout_is_not_held_to_a_native_copy_pitch() {
    // Section 14.5's central promise: the caller's bytes need not meet the
    // 256-byte row pitch a native copy footprint wants. A row pitch of 16 for a
    // four-texel RGBA row is deliberately far below that, and must be accepted.
    let layout = HostTexelLayout {
        bytes_per_row: 16,
        rows_per_image: 1,
    };
    assert!(
        validate_host_texel_layout(layout, Extent3d::d2(4, 1), TextureFormat::Rgba8Unorm).is_ok()
    );
}

#[test]
fn a_format_whose_entry_size_is_unknown_imposes_no_byte_rules() {
    // `Depth24Plus` names a precision rather than a layout, so the portable
    // layer has no byte count to check against. It must not invent one.
    let layout = HostTexelLayout {
        bytes_per_row: 3,
        rows_per_image: 2,
    };
    assert!(
        validate_host_texel_layout(layout, Extent3d::d2(4, 2), TextureFormat::Depth24Plus).is_ok()
    );

    // The row-count rule is not byte-based, so it still applies.
    let too_few_rows = HostTexelLayout {
        bytes_per_row: 3,
        rows_per_image: 1,
    };
    assert_kind(
        validate_host_texel_layout(too_few_rows, Extent3d::d2(4, 2), TextureFormat::Depth24Plus),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn the_source_byte_requirement_covers_the_last_copied_texel() {
    // Section 14.5's fourth rule. For a 4x2 RGBA8 region with a 16-byte row
    // pitch, the last copied row starts at byte 16 and needs 16 more bytes.
    let layout = HostTexelLayout {
        bytes_per_row: 16,
        rows_per_image: 2,
    };
    assert_eq!(
        source_bytes_required(layout, Extent3d::d2(4, 2), 1, TextureFormat::Rgba8Unorm),
        Some(32)
    );

    // Two images separated by rows_per_image rows: one whole image, then the
    // second image's last row.
    assert_eq!(
        source_bytes_required(layout, Extent3d::d2(4, 2), 2, TextureFormat::Rgba8Unorm),
        Some(64)
    );

    // A padded row pitch widens the requirement, which is exactly why the rule
    // is computed rather than assumed to be tightly packed: one row of 64 bytes
    // before the last one, then the 16 logical bytes of the last row.
    let padded = HostTexelLayout {
        bytes_per_row: 64,
        rows_per_image: 2,
    };
    assert_eq!(
        source_bytes_required(padded, Extent3d::d2(4, 2), 1, TextureFormat::Rgba8Unorm),
        Some(80)
    );

    // And for a format with no fixed entry size there is nothing to compute.
    assert_eq!(
        source_bytes_required(layout, Extent3d::d2(4, 2), 1, TextureFormat::Depth24Plus),
        None
    );
}
