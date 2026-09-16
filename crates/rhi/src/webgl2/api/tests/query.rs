//! Contract tests for surface lease and query/sync lifetime recording.

use super::*;

#[test]
fn mock_records_surface_lease_and_query_lifetimes() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let lease = match api.acquire_surface_image().expect("acquire") {
        GlSurfaceAcquire::Lease(lease) => lease,
        GlSurfaceAcquire::Suspended => panic!("active surface"),
    };
    api.present_surface(lease).expect("present");
    let query = api.create_query().expect("query");
    api.begin_occlusion_query(query).expect("begin");
    api.end_occlusion_query().expect("end");
    assert_eq!(
        api.query_result(query).expect("result"),
        GlQueryResult::Pending
    );
    api.destroy_query(query).expect("destroy");
    assert!(api.calls().ends_with(&[
        MockCall::AcquireSurface(lease),
        MockCall::PresentSurface(lease),
        MockCall::CreateQuery(query),
        MockCall::BeginOcclusion(query),
        MockCall::EndOcclusion,
        MockCall::QueryResult(query),
        MockCall::DestroyQuery(query),
    ]));
}

#[test]
fn mock_elapsed_and_timestamp_queries_record_and_inject_results() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let query = api.create_query().expect("query");
    assert_eq!(api.query_result(query), Ok(GlQueryResult::Pending));

    api.inject_query_result(query, GlQueryResult::Available(77));
    api.begin_elapsed_query(query).expect("begin elapsed");
    api.end_elapsed_query().expect("end elapsed");
    api.query_timestamp(query).expect("timestamp");
    assert_eq!(api.query_result(query), Ok(GlQueryResult::Available(77)));

    let trace = api.calls();
    assert!(
        trace
            .iter()
            .any(|call| matches!(call, MockCall::BeginElapsed(q) if *q == query))
    );
    assert!(
        trace
            .iter()
            .any(|call| matches!(call, MockCall::EndElapsed))
    );
    assert!(
        trace
            .iter()
            .any(|call| matches!(call, MockCall::QueryTimestamp(q) if *q == query))
    );
}
