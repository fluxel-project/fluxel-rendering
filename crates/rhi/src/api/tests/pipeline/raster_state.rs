//! Sections 25 and 26: the raster fixed state and the target signature.
//!
//! The parts of the fixed state that are pure value rules (topology, write-mask
//! bits) and the target signature's trailing-`None` canonicalization, including
//! the canonical form a created pipeline stores. `use super::*` brings in the
//! fixtures and the vocabulary the whole chapter's tests share; the banner below
//! is the original section banner.

use super::*;
// ---------------------------------------------------------------------------
// Section 25 / 26: fixed state and the target signature.
// ---------------------------------------------------------------------------

#[test]
fn only_the_strip_topologies_are_strips() {
    assert!(!PrimitiveTopology::PointList.is_strip());
    assert!(!PrimitiveTopology::LineList.is_strip());
    assert!(!PrimitiveTopology::TriangleList.is_strip());
    assert!(PrimitiveTopology::LineStrip.is_strip());
    assert!(PrimitiveTopology::TriangleStrip.is_strip());
}

#[test]
fn color_write_mask_bits_are_distinct_and_compose() {
    let channels = [
        ColorWriteMask::RED,
        ColorWriteMask::GREEN,
        ColorWriteMask::BLUE,
        ColorWriteMask::ALPHA,
    ];
    for (index, first) in channels.iter().enumerate() {
        assert!(ColorWriteMask::ALL.contains(*first));
        // `contains` is the subset query, so an empty mask is contained in every
        // mask — including itself. The interesting direction is the other one:
        // nothing but the empty mask is contained in `NONE`.
        assert!(first.contains(ColorWriteMask::NONE));
        for second in channels.iter().skip(index + 1) {
            assert_ne!(first, second);
            assert!(!first.contains(*second));
        }
    }
    assert!(!ColorWriteMask::NONE.contains(ColorWriteMask::RED));
    assert!(!ColorWriteMask::NONE.contains(ColorWriteMask::GREEN));
    assert_eq!(
        ColorWriteMask::RED.union(ColorWriteMask::GREEN),
        ColorWriteMask::RED.union(ColorWriteMask::GREEN)
    );
    assert!(
        !ColorWriteMask::RED
            .union(ColorWriteMask::GREEN)
            .contains(ColorWriteMask::BLUE)
    );
}

#[test]
fn the_target_signature_drops_trailing_holes_and_keeps_interior_ones() {
    let mut desc = raster_with(vertex_module(1, Vec::new()));
    desc.color_targets = vec![Some(ColorTargetState::new(TARGET)), None, None];

    let signature = desc.target_signature();
    assert_eq!(signature.color_formats, vec![Some(TARGET)]);
    assert_eq!(signature.depth_stencil_format, None);
    assert_eq!(signature.sample_count, 1);

    // An interior hole is a real location and stays.
    let mut desc = raster_with(vertex_module(1, Vec::new()));
    desc.color_targets = vec![
        Some(ColorTargetState::new(TARGET)),
        None,
        Some(ColorTargetState::new(TARGET)),
    ];
    assert_eq!(
        desc.target_signature().color_formats,
        vec![Some(TARGET), None, Some(TARGET)]
    );
}

#[test]
fn adding_a_color_target_fills_the_locations_before_it_with_nothing() {
    let desc = raster_with(vertex_module(1, Vec::new()))
        .with_color_target(ShaderLocation::new(2), ColorTargetState::new(TARGET));
    assert_eq!(desc.color_targets.len(), 3);
    assert!(desc.color_targets[0].is_none());
    assert!(desc.color_targets[1].is_none());
    assert!(desc.color_targets[2].is_some());
    assert_eq!(
        desc.target_signature().color_formats,
        vec![None, None, Some(TARGET)]
    );
}

#[test]
fn a_created_pipeline_stores_the_canonical_signature() {
    let mut desc = raster_with(vertex_module(1, Vec::new()));
    desc.color_targets = vec![Some(ColorTargetState::new(TARGET)), None];
    let pipeline = RasterPipeline::new(
        object(50),
        device(),
        desc,
        crate::api::tests::mock::raster_pipeline_backend_for_test(),
    );

    assert_eq!(
        pipeline.target_signature().color_formats,
        vec![Some(TARGET)]
    );
    assert_eq!(pipeline.interface().id(), object(40));
    assert_eq!(pipeline.device_identity(), device());
}

#[test]
fn multiview_masks_distinguish_baseline_selective_and_the_32_view_boundary() {
    // Positive: baseline multiview is a compact prefix of views and does not
    // need the stronger selective capability.
    let baseline = raster_with(vertex_module(51, Vec::new())).with_multiview_mask(0b111);
    assert!(
        check_raster(
            &baseline,
            &Facts::new().without_feature(OptionalFeature::SelectiveMultiview),
        )
        .is_ok()
    );

    // Negative: a hole is observable semantics, not merely an alternate bit
    // spelling for baseline multiview.
    let sparse = raster_with(vertex_module(52, Vec::new())).with_multiview_mask(0b101);
    assert_kind(
        check_raster(
            &sparse,
            &Facts::new().without_feature(OptionalFeature::SelectiveMultiview),
        ),
        RhiErrorKind::Unsupported,
    );
    assert!(check_raster(&sparse, &Facts::new()).is_ok());

    // Boundary: all 32 view bits are still a contiguous low mask. `u32::MAX`
    // must not overflow into a false sparse-mask rejection.
    let all_views = raster_with(vertex_module(53, Vec::new())).with_multiview_mask(u32::MAX);
    assert!(
        check_raster(
            &all_views,
            &Facts::new()
                .without_feature(OptionalFeature::SelectiveMultiview)
                .limit(LimitKey::MaxMultiviewViewCount, 32),
        )
        .is_ok()
    );
    assert_kind(
        check_raster(
            &all_views,
            &Facts::new().limit(LimitKey::MaxMultiviewViewCount, 31),
        ),
        RhiErrorKind::InvalidUsage,
    );

    // Zero is neither baseline nor selective multiview.
    let zero = raster_with(vertex_module(54, Vec::new())).with_multiview_mask(0);
    assert_kind(
        check_raster(&zero, &Facts::new()),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn non_default_multisample_masks_are_an_explicit_capability_request() {
    // The all-open mask is baseline and must not make an otherwise ordinary
    // pipeline depend on the optional native fixed-function state.
    let baseline = raster_with(vertex_module(55, Vec::new()));
    assert!(
        check_raster(
            &baseline,
            &Facts::new().without_feature(OptionalFeature::MultisampleMask),
        )
        .is_ok()
    );

    // A partially enabled mask and the all-disabled boundary are both real
    // state requests, so neither may silently lower as the baseline mask.
    for mask in [0x0000_00ff, 0] {
        let narrowed = raster_with(vertex_module(56, Vec::new()))
            .with_multisample(MultisampleState::new(4).with_mask(mask));
        assert_kind(
            check_raster(
                &narrowed,
                &Facts::new().without_feature(OptionalFeature::MultisampleMask),
            ),
            RhiErrorKind::Unsupported,
        );
        assert!(check_raster(&narrowed, &Facts::new()).is_ok());
    }
}
