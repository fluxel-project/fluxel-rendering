//! Section 15: texture views.

use super::*;
use crate::api::error::RhiErrorKind;
use crate::api::format::TextureFormat;
use crate::api::resource::subresource::TextureAspects;
use crate::api::resource::texture::{
    Extent3d, TextureDescriptor, TextureUsage, TextureViewCompatibility,
};
use crate::api::resource::view::{
    TextureView, TextureViewDescriptor, TextureViewDimension, validate_texture_view_descriptor,
};

#[test]
fn a_view_of_the_whole_texture_resolves_from_the_texture_descriptor() {
    // The replacement for the removed `whole_2d`: "complete" comes from the
    // texture's own descriptor, so the result is right for every dimension.
    let texture = texture_with(
        TextureDescriptor::new_2d(8, 8, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED)
            .with_mip_levels(3)
            .with_array_layers(4),
    );

    let descriptor = TextureViewDescriptor::whole(&texture, TextureViewDimension::D2Array)
        .expect("a whole view of a plain texture is legal");
    assert_eq!(descriptor.dimension, TextureViewDimension::D2Array);
    assert_eq!(descriptor.format, None);
    assert_eq!(descriptor.aspects, TextureAspects::COLOR);
    assert_eq!(descriptor.base_mip, 0);
    assert_eq!(descriptor.mip_count, 3);
    assert_eq!(descriptor.base_layer, 0);
    assert_eq!(descriptor.layer_count, 4);

    // The same call for a 3D texture covers its slices rather than layers.
    let volume = texture_with(TextureDescriptor::new_3d(
        4,
        4,
        4,
        TextureFormat::R8Unorm,
        TextureUsage::SAMPLED,
    ));
    let whole_volume = TextureViewDescriptor::whole(&volume, TextureViewDimension::D3).unwrap();
    assert_eq!(whole_volume.layer_count, 1);
    assert_eq!(whole_volume.aspects, TextureAspects::COLOR);
}

#[test]
fn a_whole_cube_view_still_has_to_satisfy_the_cube_rules() {
    // Section 15.2: "Cube/CubeArray compatibility validation is still
    // performed".
    let plain = texture_with(simple_texture_descriptor(TextureUsage::SAMPLED).with_array_layers(6));
    assert_kind(
        TextureViewDescriptor::whole(&plain, TextureViewDimension::Cube).map(|_| ()),
        RhiErrorKind::InvalidUsage,
    );

    let cube = texture_with(
        simple_texture_descriptor(TextureUsage::SAMPLED)
            .with_array_layers(6)
            .with_view_compatibility(TextureViewCompatibility::CUBE),
    );
    let descriptor = TextureViewDescriptor::whole(&cube, TextureViewDimension::Cube)
        .expect("six layers with the CUBE intent is a cube");
    assert_eq!(descriptor.layer_count, 6);

    let cube_array = TextureViewDescriptor::whole(&cube, TextureViewDimension::CubeArray)
        .expect("six layers is also one cube of an array");
    assert_eq!(cube_array.layer_count, 6);

    // Six is the only legal whole-cube layer count, and a whole cube-array view
    // needs a multiple of six.
    let four = texture_with(
        simple_texture_descriptor(TextureUsage::SAMPLED)
            .with_array_layers(4)
            .with_view_compatibility(TextureViewCompatibility::CUBE),
    );
    assert_kind(
        TextureViewDescriptor::whole(&four, TextureViewDimension::CubeArray).map(|_| ()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_view_descriptor_must_name_a_range_the_texture_has() {
    let base = simple_texture_descriptor(TextureUsage::SAMPLED)
        .with_mip_levels(3)
        .with_array_layers(4);
    let legal = TextureViewDescriptor::new(
        TextureViewDimension::D2Array,
        TextureAspects::COLOR,
        1,
        2,
        2,
        2,
    );
    assert!(validate_texture_view_descriptor(&legal, &base).is_ok());

    let no_mips = TextureViewDescriptor::new(
        TextureViewDimension::D2Array,
        TextureAspects::COLOR,
        0,
        0,
        0,
        1,
    );
    assert_kind(
        validate_texture_view_descriptor(&no_mips, &base),
        RhiErrorKind::InvalidUsage,
    );

    let no_layers = TextureViewDescriptor::new(
        TextureViewDimension::D2Array,
        TextureAspects::COLOR,
        0,
        1,
        0,
        0,
    );
    assert_kind(
        validate_texture_view_descriptor(&no_layers, &base),
        RhiErrorKind::InvalidUsage,
    );

    // Exactly at the end of the mip chain is legal; one past it is not.
    let exactly =
        TextureViewDescriptor::new(TextureViewDimension::D2, TextureAspects::COLOR, 2, 1, 0, 1);
    assert!(validate_texture_view_descriptor(&exactly, &base).is_ok());
    let past_the_end =
        TextureViewDescriptor::new(TextureViewDimension::D2, TextureAspects::COLOR, 2, 2, 0, 1);
    assert_kind(
        validate_texture_view_descriptor(&past_the_end, &base),
        RhiErrorKind::InvalidUsage,
    );

    // Same for layers.
    let last_layer = TextureViewDescriptor::new(
        TextureViewDimension::D2Array,
        TextureAspects::COLOR,
        0,
        1,
        3,
        1,
    );
    assert!(validate_texture_view_descriptor(&last_layer, &base).is_ok());
    let past_the_layers = TextureViewDescriptor::new(
        TextureViewDimension::D2Array,
        TextureAspects::COLOR,
        0,
        1,
        3,
        2,
    );
    assert_kind(
        validate_texture_view_descriptor(&past_the_layers, &base),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_view_may_not_select_an_aspect_the_format_lacks() {
    let depth = texture_with(TextureDescriptor::new_2d(
        8,
        8,
        TextureFormat::Depth32Float,
        TextureUsage::SAMPLED,
    ));

    let depth_view =
        TextureViewDescriptor::new(TextureViewDimension::D2, TextureAspects::DEPTH, 0, 1, 0, 1);
    assert!(validate_texture_view_descriptor(&depth_view, depth.descriptor()).is_ok());

    let color_view =
        TextureViewDescriptor::new(TextureViewDimension::D2, TextureAspects::COLOR, 0, 1, 0, 1);
    assert_kind(
        validate_texture_view_descriptor(&color_view, depth.descriptor()),
        RhiErrorKind::InvalidUsage,
    );

    // A depth-stencil format carries both planes, and a view may take either.
    let both = texture_with(TextureDescriptor::new_2d(
        8,
        8,
        TextureFormat::Depth24PlusStencil8,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT,
    ));
    let stencil = TextureViewDescriptor::new(
        TextureViewDimension::D2,
        TextureAspects::STENCIL,
        0,
        1,
        0,
        1,
    );
    assert!(validate_texture_view_descriptor(&stencil, both.descriptor()).is_ok());
}

#[test]
fn an_undeclared_alternate_view_format_is_refused() {
    // Section 13.1's creation-time declaration, enforced at view creation: a
    // format the texture did not declare may not be viewed as.
    let base = texture_with(simple_texture_descriptor(TextureUsage::SAMPLED));

    let undeclared =
        TextureViewDescriptor::new(TextureViewDimension::D2, TextureAspects::COLOR, 0, 1, 0, 1)
            .with_format(TextureFormat::Bgra8Unorm);
    assert_kind(
        validate_texture_view_descriptor(&undeclared, base.descriptor()),
        RhiErrorKind::InvalidUsage,
    );

    // Declared, so accepted — and the base format itself needs no declaration.
    let declared = texture_with(
        simple_texture_descriptor(TextureUsage::SAMPLED)
            .with_view_format(TextureFormat::Bgra8Unorm),
    );
    assert!(validate_texture_view_descriptor(&undeclared, declared.descriptor()).is_ok());

    let same_as_base =
        TextureViewDescriptor::new(TextureViewDimension::D2, TextureAspects::COLOR, 0, 1, 0, 1)
            .with_format(TextureFormat::Rgba8Unorm);
    assert!(validate_texture_view_descriptor(&same_as_base, base.descriptor()).is_ok());
}

#[test]
fn a_view_dimension_must_match_the_texture_dimension() {
    let base = simple_texture_descriptor(TextureUsage::SAMPLED);
    let volume = texture_with(TextureDescriptor::new_3d(
        4,
        4,
        4,
        TextureFormat::R8Unorm,
        TextureUsage::SAMPLED,
    ));

    for dimension in [TextureViewDimension::D1, TextureViewDimension::D3] {
        let view = TextureViewDescriptor::new(dimension, TextureAspects::COLOR, 0, 1, 0, 1);
        assert_kind(
            validate_texture_view_descriptor(&view, &base),
            RhiErrorKind::InvalidUsage,
        );
    }

    // The reverse: a 2D view of a 3D texture is the sliced view P0 defers.
    let flat =
        TextureViewDescriptor::new(TextureViewDimension::D2, TextureAspects::COLOR, 0, 1, 0, 1);
    assert_kind(
        validate_texture_view_descriptor(&flat, volume.descriptor()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_cube_view_needs_the_intent_and_exactly_six_faces() {
    let without_intent =
        texture_with(simple_texture_descriptor(TextureUsage::SAMPLED).with_array_layers(6));
    let cube = TextureViewDescriptor::new(
        TextureViewDimension::Cube,
        TextureAspects::COLOR,
        0,
        1,
        0,
        6,
    );
    assert_kind(
        validate_texture_view_descriptor(&cube, without_intent.descriptor()),
        RhiErrorKind::InvalidUsage,
    );

    let with_intent = texture_with(
        simple_texture_descriptor(TextureUsage::SAMPLED)
            .with_array_layers(6)
            .with_view_compatibility(TextureViewCompatibility::CUBE),
    );
    assert!(validate_texture_view_descriptor(&cube, with_intent.descriptor()).is_ok());

    let wrong_count = TextureViewDescriptor::new(
        TextureViewDimension::Cube,
        TextureAspects::COLOR,
        0,
        1,
        0,
        4,
    );
    assert_kind(
        validate_texture_view_descriptor(&wrong_count, with_intent.descriptor()),
        RhiErrorKind::InvalidUsage,
    );

    // A cube array covers a multiple of six, and twelve is legal.
    let twelve_layers = texture_with(
        simple_texture_descriptor(TextureUsage::SAMPLED)
            .with_array_layers(12)
            .with_view_compatibility(TextureViewCompatibility::CUBE),
    );
    let cube_array = TextureViewDescriptor::new(
        TextureViewDimension::CubeArray,
        TextureAspects::COLOR,
        0,
        1,
        0,
        12,
    );
    assert!(validate_texture_view_descriptor(&cube_array, twelve_layers.descriptor()).is_ok());

    let eight_layers = texture_with(
        simple_texture_descriptor(TextureUsage::SAMPLED)
            .with_array_layers(8)
            .with_view_compatibility(TextureViewCompatibility::CUBE),
    );
    let not_a_multiple = TextureViewDescriptor::new(
        TextureViewDimension::CubeArray,
        TextureAspects::COLOR,
        0,
        1,
        0,
        8,
    );
    assert_kind(
        validate_texture_view_descriptor(&not_a_multiple, eight_layers.descriptor()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn a_view_reports_its_resolved_format_and_its_own_range() {
    let texture = texture_with(
        TextureDescriptor::new_2d(8, 4, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED)
            .with_mip_levels(3)
            .with_array_layers(4)
            .with_view_format(TextureFormat::Bgra8Unorm),
    );

    // format() resolves the descriptor's `None` to the base format.
    let plain = TextureViewDescriptor::new(
        TextureViewDimension::D2Array,
        TextureAspects::COLOR,
        1,
        2,
        1,
        3,
    );
    let view = TextureView::new(object(21), device(), texture.clone(), plain);
    assert_eq!(view.id(), object(21));
    assert_eq!(view.device_identity(), device());
    assert_eq!(view.format(), TextureFormat::Rgba8Unorm);
    assert_eq!(view.aspects(), TextureAspects::COLOR);
    assert_eq!(view.sample_count(), 1);
    assert_eq!(view.layer_count(), 3);
    assert_eq!(view.texture().id(), texture.id());

    // extent() is the base_mip level, floored at one texel, and array layers do
    // not contribute to depth.
    assert_eq!(view.extent(), Extent3d::d2(4, 2));

    // An explicit alternate format is reported as itself.
    let alternate =
        TextureViewDescriptor::new(TextureViewDimension::D2, TextureAspects::COLOR, 0, 1, 0, 1)
            .with_format(TextureFormat::Bgra8Unorm);
    let alternate_view = TextureView::new(object(22), device(), texture.clone(), alternate);
    assert_eq!(alternate_view.format(), TextureFormat::Bgra8Unorm);

    // Cloning a view is the same view.
    let clone = alternate_view.clone();
    assert_eq!(clone.id(), alternate_view.id());
    assert!(std::ptr::eq(clone.native(), alternate_view.native()));
}

#[test]
fn a_views_extent_halves_per_level_and_never_reaches_zero() {
    // The standard mip reduction, at the boundary where a naive shift would
    // report a zero-sized extent.
    let texture = texture_with(
        TextureDescriptor::new_2d(8, 2, TextureFormat::R8Unorm, TextureUsage::SAMPLED)
            .with_mip_levels(4),
    );

    let mut expected = [
        Extent3d::d2(8, 2),
        Extent3d::d2(4, 1),
        Extent3d::d2(2, 1),
        Extent3d::d2(1, 1),
    ];
    for (level, want) in expected.iter_mut().enumerate() {
        let descriptor = TextureViewDescriptor::new(
            TextureViewDimension::D2,
            TextureAspects::COLOR,
            level as u32,
            1,
            0,
            1,
        );
        let view = TextureView::new(
            object(30 + level as u64),
            device(),
            texture.clone(),
            descriptor,
        );
        assert_eq!(view.extent(), *want, "level {level}");
    }

    // A 3D texture reduces its Z axis the same way, and reports depth = 1 when
    // it is not 3D even though it has layers.
    let volume = texture_with(
        TextureDescriptor::new_3d(8, 8, 8, TextureFormat::R8Unorm, TextureUsage::SAMPLED)
            .with_mip_levels(4),
    );
    let deep =
        TextureViewDescriptor::new(TextureViewDimension::D3, TextureAspects::COLOR, 1, 1, 0, 1);
    let deep_view = TextureView::new(object(40), device(), volume, deep);
    assert_eq!(deep_view.extent(), Extent3d::d3(4, 4, 4));

    let layered = texture_with(
        TextureDescriptor::new_2d(8, 8, TextureFormat::R8Unorm, TextureUsage::SAMPLED)
            .with_array_layers(6)
            .with_view_compatibility(TextureViewCompatibility::CUBE),
    );
    let cube = TextureViewDescriptor::new(
        TextureViewDimension::Cube,
        TextureAspects::COLOR,
        0,
        1,
        0,
        6,
    );
    let cube_view = TextureView::new(object(41), device(), layered, cube);
    assert_eq!(cube_view.extent(), Extent3d::d2(8, 8));
    assert_eq!(cube_view.extent().depth, 1);
}

#[test]
fn view_usage_is_a_nonempty_subset_of_texture_usage() {
    let texture = TextureDescriptor::new_2d(
        4,
        4,
        TextureFormat::Rgba8Unorm,
        TextureUsage::SAMPLED.union(TextureUsage::COPY_DST),
    );
    let legal =
        TextureViewDescriptor::new(TextureViewDimension::D2, TextureAspects::COLOR, 0, 1, 0, 1)
            .with_usage(TextureUsage::SAMPLED);
    assert!(validate_texture_view_descriptor(&legal, &texture).is_ok());

    let unavailable = legal.clone().with_usage(TextureUsage::COLOR_ATTACHMENT);
    assert_kind(
        validate_texture_view_descriptor(&unavailable, &texture),
        RhiErrorKind::InvalidUsage,
    );
}
