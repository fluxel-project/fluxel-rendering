//! Section 9: route facts.

use super::*;
use crate::api::error::RhiErrorKind;
use crate::api::format::TextureFormat;
use crate::api::resource::route::{
    BufferCopyLayoutLimits, RouteCapabilities, RouteQuery, RouteSupport, TexelCopyLayoutLimits,
};
use crate::api::resource::subresource::TextureAspect;
use crate::api::resource::texture::TextureDimension;

#[test]
fn every_route_key_carries_the_shape_facts_that_change_legality() {
    // Section 9.1's reason for putting shape in the key: a key without it would
    // answer `Supported` for a descriptor the route cannot execute. The test
    // builds every variant so that a key that lost a field would not compile.
    let keys = [
        RouteQuery::BufferToBuffer,
        RouteQuery::BufferToTexture {
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            aspect: TextureAspect::Color,
        },
        RouteQuery::TextureToBuffer {
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            aspect: TextureAspect::Color,
        },
        RouteQuery::TextureToTexture {
            src_dimension: TextureDimension::D2,
            src_format: TextureFormat::Rgba8Unorm,
            src_aspect: TextureAspect::Color,
            src_sample_count: 1,
            dst_dimension: TextureDimension::D2,
            dst_format: TextureFormat::Rgba8Unorm,
            dst_aspect: TextureAspect::Color,
            dst_sample_count: 1,
        },
        RouteQuery::Resolve {
            format: TextureFormat::Rgba8Unorm,
            src_sample_count: 4,
        },
    ];

    let mut set = std::collections::HashSet::new();
    for key in keys {
        assert!(set.insert(key), "{key:?} is not a distinct key");
    }
    assert_eq!(set.len(), 5);

    // The same key built twice is one key, and a sample count is part of it.
    assert!(set.contains(&RouteQuery::BufferToBuffer));
    assert!(!set.contains(&RouteQuery::Resolve {
        format: TextureFormat::Rgba8Unorm,
        src_sample_count: 8,
    }));
}

#[test]
fn a_route_answer_reports_its_layouts_only_when_supported() {
    let supported = RouteSupport::Supported(RouteCapabilities::new(Some(copy_limits()), None));
    assert!(supported.is_supported());
    let capabilities = supported
        .capabilities()
        .expect("supported carries capabilities");
    assert_eq!(capabilities.buffer_copy_layout(), Some(copy_limits()));
    assert_eq!(capabilities.texel_copy_layout(), None);

    let unsupported = RouteSupport::Unsupported;
    assert!(!unsupported.is_supported());
    assert!(unsupported.capabilities().is_none());
}

#[test]
fn copy_alignment_is_checked_on_both_sides() {
    let limits = BufferCopyLayoutLimits::new(4, 4);
    assert_eq!(limits.offset_alignment(), 4);
    assert_eq!(limits.size_alignment(), 4);

    assert!(limits.validate(0, 4).is_ok());
    assert!(limits.validate(4, 1024).is_ok());
    assert_kind(limits.validate(1, 4), RhiErrorKind::InvalidUsage);
    assert_kind(limits.validate(4, 6), RhiErrorKind::InvalidUsage);
}

#[test]
fn texel_copy_alignment_is_checked_on_both_sides() {
    let limits = TexelCopyLayoutLimits::new(256, 256);
    assert_eq!(limits.buffer_offset_alignment(), 256);
    assert_eq!(limits.bytes_per_row_alignment(), 256);

    assert!(limits.validate(0, 256).is_ok());
    assert!(limits.validate(512, 1024).is_ok());
    assert_kind(limits.validate(128, 256), RhiErrorKind::InvalidUsage);
    assert_kind(limits.validate(0, 512 + 1), RhiErrorKind::InvalidUsage);
}

#[test]
fn texel_copy_image_stride_distinguishes_array_images_from_3d_slices() {
    let limits = TexelCopyLayoutLimits::new(512, 256).with_image_layout(512, true);
    assert_eq!(limits.image_stride_alignment(), 512);
    assert!(limits.tightly_packed_3d_slices());

    // Two D2 array layers name two independently placed D3D12 footprints.
    assert!(
        limits
            .validate_image_layout(256, 2, 1, TextureDimension::D2, 2)
            .is_ok()
    );
    assert_kind(
        limits.validate_image_layout(256, 1, 1, TextureDimension::D2, 2),
        RhiErrorKind::InvalidUsage,
    );

    // A D3 region is one footprint. Its slices do not each need a 512-byte
    // start, but the backend that requested this fact cannot represent padding
    // rows between consecutive Z slices.
    assert!(
        limits
            .validate_image_layout(256, 2, 2, TextureDimension::D3, 3)
            .is_ok()
    );
    assert_kind(
        limits.validate_image_layout(256, 3, 2, TextureDimension::D3, 3),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_zero_alignment_is_no_constraint_rather_than_a_panic() {
    // A malformed capability must not be able to turn validation into an abort.
    let limits = BufferCopyLayoutLimits::new(0, 0);
    assert!(limits.validate(1, 3).is_ok());
}
