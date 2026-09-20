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
