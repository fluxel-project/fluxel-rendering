//! Query-bracket lifecycle contracts for raster and compute scopes.
//!
//! The tests intentionally drive the public `QuerySet` creation verb first and
//! then the public scope verbs.  A manually assembled query handle would only
//! prove private packet bookkeeping, while these prove the contract a caller
//! actually receives: a query is a single non-nestable bracket with a matching
//! `(QuerySet, index)` terminator.

use super::*;
use crate::api::command::RecorderDescriptor;
use crate::api::query::{PipelineStatistics, QuerySetDescriptor, QueryType};
use crate::api::tests::mock::query_device_for_test;

fn query_device() -> crate::api::platform::Device {
    query_device_for_test(device())
}

fn occlusion_set(device: &crate::api::platform::Device, count: u32) -> crate::api::query::QuerySet {
    device
        .create_query_set(&QuerySetDescriptor::new(QueryType::Occlusion, count))
        .expect("the query mock enables occlusion")
}

fn statistics_set(
    device: &crate::api::platform::Device,
    count: u32,
) -> crate::api::query::QuerySet {
    device
        .create_query_set(&QuerySetDescriptor::new(
            QueryType::PipelineStatistics(PipelineStatistics::VERTEX_SHADER_INVOCATIONS),
            count,
        ))
        .expect("the query mock enables the selected statistic")
}

#[test]
fn raster_query_bracket_accepts_a_matching_final_slot_boundary() {
    let device = query_device();
    let set = occlusion_set(&device, 2);
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    let mut scope = recorder
        .begin_raster(&color_scope("query bracket"))
        .unwrap();

    // `count - 1` is the addressable upper boundary and the exact matching end
    // closes it, allowing the scope and recording to finish normally.
    scope.begin_query(&set, 1).unwrap();
    scope.end_query(&set, 1).unwrap();
    scope.end().unwrap();
    assert!(recorder.finish().is_ok());
}

#[test]
fn raster_query_bracket_refuses_end_before_begin_nesting_and_mismatched_end() {
    let device = query_device();
    let first = occlusion_set(&device, 2);
    let second = occlusion_set(&device, 2);
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    let mut scope = recorder
        .begin_raster(&color_scope("query lifecycle"))
        .unwrap();

    assert_kind(scope.end_query(&first, 0), RhiErrorKind::InvalidUsage);
    scope.begin_query(&first, 0).unwrap();
    assert_kind(scope.begin_query(&first, 0), RhiErrorKind::InvalidUsage);
    assert_kind(scope.begin_query(&second, 0), RhiErrorKind::InvalidUsage);
    assert_kind(scope.end_query(&first, 1), RhiErrorKind::InvalidUsage);
    assert_kind(scope.end_query(&second, 0), RhiErrorKind::InvalidUsage);
    scope.end_query(&first, 0).unwrap();
    scope.end().unwrap();
}

#[test]
fn compute_query_bracket_accepts_a_matching_final_slot_boundary() {
    let device = query_device();
    let set = statistics_set(&device, 2);
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    let mut scope = recorder.begin_compute(&Default::default()).unwrap();

    scope.begin_query(&set, 1).unwrap();
    scope.end_query(&set, 1).unwrap();
    scope.end().unwrap();
    assert!(recorder.finish().is_ok());
}

#[test]
fn compute_query_bracket_refuses_end_before_begin_nesting_and_mismatched_end() {
    let device = query_device();
    let first = statistics_set(&device, 2);
    let second = statistics_set(&device, 2);
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    let mut scope = recorder.begin_compute(&Default::default()).unwrap();

    assert_kind(scope.end_query(&first, 0), RhiErrorKind::InvalidUsage);
    scope.begin_query(&first, 0).unwrap();
    assert_kind(scope.begin_query(&first, 0), RhiErrorKind::InvalidUsage);
    assert_kind(scope.begin_query(&second, 0), RhiErrorKind::InvalidUsage);
    assert_kind(scope.end_query(&first, 1), RhiErrorKind::InvalidUsage);
    assert_kind(scope.end_query(&second, 0), RhiErrorKind::InvalidUsage);
    scope.end_query(&first, 0).unwrap();
    scope.end().unwrap();
}

/// A query result slot is produced once per `RecordedWork`, even if a caller
/// closes the first bracket before attempting a second one.  This is distinct
/// from the active-bracket nesting rule above and prevents backend-dependent
/// last-write-wins behaviour.
#[test]
fn closed_query_slot_cannot_be_written_again_in_one_recording() {
    let device = query_device();
    let raster_set = occlusion_set(&device, 1);
    let mut raster = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    let mut raster_scope = raster
        .begin_raster(&color_scope("single query writer"))
        .unwrap();
    raster_scope.begin_query(&raster_set, 0).unwrap();
    raster_scope.end_query(&raster_set, 0).unwrap();
    assert_kind(
        raster_scope.begin_query(&raster_set, 0),
        RhiErrorKind::InvalidUsage,
    );
    raster_scope.end().unwrap();
    assert!(raster.finish().is_ok());

    let compute_set = statistics_set(&device, 1);
    let mut compute = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    let mut compute_scope = compute.begin_compute(&Default::default()).unwrap();
    compute_scope.begin_query(&compute_set, 0).unwrap();
    compute_scope.end_query(&compute_set, 0).unwrap();
    assert_kind(
        compute_scope.begin_query(&compute_set, 0),
        RhiErrorKind::InvalidUsage,
    );
    compute_scope.end().unwrap();
    assert!(compute.finish().is_ok());
}

#[test]
fn active_query_refuses_scope_end_and_drop_poison_prevents_recording_reuse() {
    let device = query_device();
    let raster_set = occlusion_set(&device, 1);
    let mut raster_recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    {
        let mut scope = raster_recorder
            .begin_raster(&color_scope("unclosed raster query"))
            .unwrap();
        scope.begin_query(&raster_set, 0).unwrap();
        assert_kind(scope.end(), RhiErrorKind::InvalidUsage);
    }
    assert_kind(
        raster_recorder.finish().map(|_| ()),
        RhiErrorKind::InvalidUsage,
    );

    let compute_set = statistics_set(&device, 1);
    let mut compute_recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    {
        let mut scope = compute_recorder.begin_compute(&Default::default()).unwrap();
        scope.begin_query(&compute_set, 0).unwrap();
        // Dropping without `end` cannot emit a synthetic end query.  It poisons
        // instead, so no half-open native query can ever be submitted.
    }
    assert_kind(
        compute_recorder.finish().map(|_| ()),
        RhiErrorKind::InvalidUsage,
    );
}
