//! Section 13: texture creation.

use super::*;
use crate::api::error::RhiErrorKind;
use crate::api::format::{TextureFormat, TextureSupport, TextureSupportLimits};
use crate::api::resource::texture::{
    Extent3d, Texture, TextureDescriptor, TextureUsage, TextureViewCompatibility,
    validate_texture_descriptor,
};

#[test]
fn a_legal_texture_descriptor_is_accepted() {
    let mut descriptor = simple_texture_descriptor(TextureUsage::SAMPLED);
    assert!(validate_texture_descriptor(&mut descriptor, &generous_texture_support()).is_ok());

    // A combined usage set, a mip chain, and array layers are each legal on their
    // own and together: none of the shape invariants is about usage.
    let mut layered =
        simple_texture_descriptor(TextureUsage::SAMPLED.union(TextureUsage::COLOR_ATTACHMENT))
            .with_mip_levels(3)
            .with_array_layers(4);
    assert!(validate_texture_descriptor(&mut layered, &generous_texture_support()).is_ok());
}

#[test]
fn an_extent_with_a_zero_component_is_refused() {
    for extent in [
        Extent3d::d3(0, 4, 4),
        Extent3d::d3(4, 0, 4),
        Extent3d::d3(4, 4, 0),
    ] {
        let mut descriptor = TextureDescriptor::new_3d(
            extent.width,
            extent.height,
            extent.depth,
            TextureFormat::R8Unorm,
            TextureUsage::SAMPLED,
        );
        assert_kind(
            validate_texture_descriptor(&mut descriptor, &generous_texture_support()),
            RhiErrorKind::InvalidUsage,
        );
    }
}

#[test]
fn a_descriptor_without_alternate_view_formats_needs_no_declaration() {
    // The complement of the "base format may not be declared" rule: declaring
    // nothing is legal, because the base format is always viewable as itself.
    let mut descriptor = simple_texture_descriptor(TextureUsage::SAMPLED);
    assert!(descriptor.view_formats.is_empty());
    assert!(validate_texture_descriptor(&mut descriptor, &generous_texture_support()).is_ok());
    assert!(descriptor.view_formats.is_empty());
}

#[test]
fn zero_mips_layers_or_samples_are_refused() {
    for mut descriptor in [
        simple_texture_descriptor(TextureUsage::SAMPLED).with_mip_levels(0),
        simple_texture_descriptor(TextureUsage::SAMPLED).with_array_layers(0),
        simple_texture_descriptor(TextureUsage::SAMPLED).with_sample_count(0),
    ] {
        assert_kind(
            validate_texture_descriptor(&mut descriptor, &generous_texture_support()),
            RhiErrorKind::InvalidUsage,
        );
    }
}

#[test]
fn a_d1_texture_is_pinned_to_one_row_one_layer_one_sample() {
    // Section 13.1's D1 block, each clause separately.
    let mut wrong_height =
        TextureDescriptor::new_1d(4, TextureFormat::R8Unorm, TextureUsage::SAMPLED);
    wrong_height.extent.height = 2;
    assert_kind(
        validate_texture_descriptor(&mut wrong_height, &generous_texture_support()),
        RhiErrorKind::InvalidUsage,
    );

    let mut wrong_depth =
        TextureDescriptor::new_1d(4, TextureFormat::R8Unorm, TextureUsage::SAMPLED);
    wrong_depth.extent.depth = 2;
    assert_kind(
        validate_texture_descriptor(&mut wrong_depth, &generous_texture_support()),
        RhiErrorKind::InvalidUsage,
    );

    let mut wrong_layers =
        TextureDescriptor::new_1d(4, TextureFormat::R8Unorm, TextureUsage::SAMPLED)
            .with_array_layers(2);
    assert_kind(
        validate_texture_descriptor(&mut wrong_layers, &generous_texture_support()),
        RhiErrorKind::InvalidUsage,
    );

    let mut wrong_samples =
        TextureDescriptor::new_1d(4, TextureFormat::R8Unorm, TextureUsage::SAMPLED)
            .with_sample_count(2);
    assert_kind(
        validate_texture_descriptor(&mut wrong_samples, &generous_texture_support()),
        RhiErrorKind::InvalidUsage,
    );

    // And the legal case, so the four refusals above are not refusals of
    // everything.
    let mut legal = TextureDescriptor::new_1d(4, TextureFormat::R8Unorm, TextureUsage::SAMPLED);
    assert!(validate_texture_descriptor(&mut legal, &generous_texture_support()).is_ok());
}

#[test]
fn a_d2_texture_is_pinned_to_one_depth() {
    let mut wrong_depth = simple_texture_descriptor(TextureUsage::SAMPLED);
    wrong_depth.extent.depth = 2;
    assert_kind(
        validate_texture_descriptor(&mut wrong_depth, &generous_texture_support()),
        RhiErrorKind::InvalidUsage,
    );

    // Array layers are legal for a 2D texture, so that is not a refusal.
    let mut layered = simple_texture_descriptor(TextureUsage::SAMPLED).with_array_layers(6);
    assert!(validate_texture_descriptor(&mut layered, &generous_texture_support()).is_ok());
}

#[test]
fn a_d3_texture_is_pinned_to_one_layer_and_one_sample() {
    let mut wrong_layers =
        TextureDescriptor::new_3d(4, 4, 4, TextureFormat::R8Unorm, TextureUsage::SAMPLED)
            .with_array_layers(2);
    assert_kind(
        validate_texture_descriptor(&mut wrong_layers, &generous_texture_support()),
        RhiErrorKind::InvalidUsage,
    );

    let mut wrong_samples =
        TextureDescriptor::new_3d(4, 4, 4, TextureFormat::R8Unorm, TextureUsage::SAMPLED)
            .with_sample_count(2);
    assert_kind(
        validate_texture_descriptor(&mut wrong_samples, &generous_texture_support()),
        RhiErrorKind::InvalidUsage,
    );

    // Depth is what a 3D texture uses instead of layers, so several slices are
    // legal and are not a refusal.
    let mut sliced =
        TextureDescriptor::new_3d(4, 4, 4, TextureFormat::R8Unorm, TextureUsage::SAMPLED);
    assert!(validate_texture_descriptor(&mut sliced, &generous_texture_support()).is_ok());
}

#[test]
fn multisampling_requires_two_dimensions_and_one_mip() {
    // Section 13.1: "sample_count > 1: dimension = D2, mip_levels = 1".
    let mut legal = simple_texture_descriptor(TextureUsage::COLOR_ATTACHMENT).with_sample_count(4);
    assert!(validate_texture_descriptor(&mut legal, &generous_texture_support()).is_ok());

    let mut mipmapped = simple_texture_descriptor(TextureUsage::COLOR_ATTACHMENT)
        .with_sample_count(4)
        .with_mip_levels(2);
    assert_kind(
        validate_texture_descriptor(&mut mipmapped, &generous_texture_support()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn too_many_mip_levels_are_refused_on_both_sides_of_the_ceiling() {
    // A 4x4 texture permits three levels: 4, 2, 1. The boundary is tested from
    // both sides, because an off-by-one here silently truncates a mip chain.
    let mut exactly = simple_texture_descriptor(TextureUsage::SAMPLED).with_mip_levels(3);
    assert!(validate_texture_descriptor(&mut exactly, &generous_texture_support()).is_ok());

    let mut one_too_many = simple_texture_descriptor(TextureUsage::SAMPLED).with_mip_levels(4);
    assert_kind(
        validate_texture_descriptor(&mut one_too_many, &generous_texture_support()),
        RhiErrorKind::InvalidUsage,
    );

    // The ceiling follows the largest axis, not the width: 1x8 permits four.
    let mut tall = TextureDescriptor::new_2d(1, 8, TextureFormat::R8Unorm, TextureUsage::SAMPLED)
        .with_mip_levels(4);
    assert!(validate_texture_descriptor(&mut tall, &generous_texture_support()).is_ok());
}

#[test]
fn the_base_format_may_not_appear_among_the_view_formats() {
    // Section 13.1: "view_formats ... may not include the base format itself".
    let mut descriptor = simple_texture_descriptor(TextureUsage::SAMPLED)
        .with_view_format(TextureFormat::Rgba8Unorm);
    assert_kind(
        validate_texture_descriptor(&mut descriptor, &generous_texture_support()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_validated_descriptor_adopts_the_canonical_view_format_set() {
    // The canonical set is the descriptor's, not only the query's: section 13.3
    // builds the query from this list, so a descriptor that kept a duplicate
    // would make the query a different key than the same descriptor built in a
    // different order.
    let mut descriptor = simple_texture_descriptor(TextureUsage::SAMPLED);
    descriptor.view_formats = vec![
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm,
    ];
    assert!(validate_texture_descriptor(&mut descriptor, &generous_texture_support()).is_ok());
    assert_eq!(
        descriptor.view_formats,
        vec![TextureFormat::Rgba8UnormSrgb, TextureFormat::Bgra8Unorm]
    );
}

#[test]
fn a_refused_descriptor_is_left_untouched() {
    // A rejected call must not have a side effect the caller did not ask for.
    let mut descriptor = simple_texture_descriptor(TextureUsage::SAMPLED);
    descriptor.view_formats = vec![TextureFormat::Bgra8Unorm, TextureFormat::Bgra8Unorm];
    descriptor.mip_levels = 9; // 4x4 permits three
    let before = descriptor.view_formats.clone();

    assert_kind(
        validate_texture_descriptor(&mut descriptor, &generous_texture_support()),
        RhiErrorKind::InvalidUsage,
    );
    assert_eq!(descriptor.view_formats, before);
}

#[test]
fn cube_compatibility_is_validated_at_creation() {
    // Section 13.2's P0 rules: D2, square, at least six layers, one sample. Each
    // clause is refused on its own so that the message a caller sees names the
    // clause they broke.
    let mut legal = simple_texture_descriptor(TextureUsage::SAMPLED)
        .with_array_layers(6)
        .with_view_compatibility(TextureViewCompatibility::CUBE);
    assert!(validate_texture_descriptor(&mut legal, &generous_texture_support()).is_ok());

    let mut not_square =
        TextureDescriptor::new_2d(8, 4, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED)
            .with_array_layers(6)
            .with_view_compatibility(TextureViewCompatibility::CUBE);
    assert_kind(
        validate_texture_descriptor(&mut not_square, &generous_texture_support()),
        RhiErrorKind::InvalidUsage,
    );

    let mut too_few_layers = simple_texture_descriptor(TextureUsage::SAMPLED)
        .with_array_layers(4)
        .with_view_compatibility(TextureViewCompatibility::CUBE);
    assert_kind(
        validate_texture_descriptor(&mut too_few_layers, &generous_texture_support()),
        RhiErrorKind::InvalidUsage,
    );

    let mut multisampled = simple_texture_descriptor(TextureUsage::SAMPLED)
        .with_array_layers(6)
        .with_sample_count(4)
        .with_view_compatibility(TextureViewCompatibility::CUBE);
    assert_kind(
        validate_texture_descriptor(&mut multisampled, &generous_texture_support()),
        RhiErrorKind::InvalidUsage,
    );

    // A cube intent on a 1D texture cannot be honoured by any backend.
    let mut one_d = TextureDescriptor::new_1d(8, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED)
        .with_view_compatibility(TextureViewCompatibility::CUBE);
    assert_kind(
        validate_texture_descriptor(&mut one_d, &generous_texture_support()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_descriptor_the_device_cannot_create_is_unsupported() {
    // Section 13.4's last check, and the case it exists for: `Rgba16Float` as a
    // 3D color attachment. The key is unsupported, which is a device refusal
    // rather than a descriptor mistake.
    let mut descriptor = TextureDescriptor::new_3d(
        64,
        64,
        64,
        TextureFormat::Rgba16Float,
        TextureUsage::COLOR_ATTACHMENT,
    );
    assert_kind(
        validate_texture_descriptor(&mut descriptor, &TextureSupport::Unsupported),
        RhiErrorKind::Unsupported,
    );
}

#[test]
fn the_devices_limits_bound_the_extent_mips_and_layers() {
    // Section 8.4's second half: the query says whether the *key* is creatable,
    // and the limits decide whether this particular extent, mip count, and layer
    // count are. Both are tested on both sides of each limit.
    let support =
        TextureSupport::Supported(TextureSupportLimits::new(Extent3d::d3(64, 32, 8), 4, 2));

    let mut at_the_limit =
        TextureDescriptor::new_2d(64, 32, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED)
            .with_mip_levels(4)
            .with_array_layers(2);
    assert!(validate_texture_descriptor(&mut at_the_limit, &support).is_ok());

    let mut too_wide =
        TextureDescriptor::new_2d(65, 32, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED);
    assert_kind(
        validate_texture_descriptor(&mut too_wide, &support),
        RhiErrorKind::InvalidUsage,
    );

    let mut too_tall =
        TextureDescriptor::new_2d(64, 33, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED);
    assert_kind(
        validate_texture_descriptor(&mut too_tall, &support),
        RhiErrorKind::InvalidUsage,
    );

    let mut too_many_mips =
        TextureDescriptor::new_2d(64, 32, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED)
            .with_mip_levels(5);
    assert_kind(
        validate_texture_descriptor(&mut too_many_mips, &support),
        RhiErrorKind::InvalidUsage,
    );

    let mut too_many_layers =
        TextureDescriptor::new_2d(64, 32, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED)
            .with_array_layers(3);
    assert_kind(
        validate_texture_descriptor(&mut too_many_layers, &support),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_texture_reports_its_own_id_device_and_descriptor() {
    let descriptor = simple_texture_descriptor(TextureUsage::SAMPLED).with_label("albedo");
    let texture = Texture::new(object(11), identity(2, 3), descriptor);

    assert_eq!(texture.id(), object(11));
    assert_eq!(texture.device_identity(), identity(2, 3));
    assert_eq!(texture.descriptor().label.as_deref(), Some("albedo"));
    assert_eq!(texture.descriptor().extent, Extent3d::d2(4, 4));
    assert_eq!(texture.clone().id(), texture.id());
}

#[test]
fn the_view_compatibility_set_composes_like_a_bitset() {
    assert!(TextureViewCompatibility::NONE.contains(TextureViewCompatibility::NONE));
    assert!(!TextureViewCompatibility::NONE.contains(TextureViewCompatibility::CUBE));
    assert!(TextureViewCompatibility::CUBE.contains(TextureViewCompatibility::CUBE));
    assert_eq!(
        TextureViewCompatibility::NONE.union(TextureViewCompatibility::CUBE),
        TextureViewCompatibility::CUBE
    );
}
