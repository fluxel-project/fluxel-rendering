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

/// One acquisition against the recorder's default 1x1 drawable.
fn acquire_one_frame(api: &mut MockGlFamilyApi) -> GlSurfaceLease {
    match api.acquire_surface_image().expect("acquire") {
        GlSurfaceAcquire::Lease(lease) => lease,
        GlSurfaceAcquire::Suspended => panic!("active surface"),
    }
}

#[test]
fn mock_records_publishing_one_frame_into_the_drawable() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let source = api
        .create_texture_resource(mock_texture_desc())
        .expect("source texture");
    let lease = acquire_one_frame(&mut api);

    api.publish_surface_image(lease, source).expect("publish");
    assert!(
        api.calls()
            .ends_with(&[MockCall::PublishSurface { lease, source }])
    );

    // Publishing is what ends the acquisition, so it cannot also be presented:
    // one lease is one frame, and the second verb has nothing left to act on.
    assert!(api.present_surface(lease).is_err());
}

#[test]
fn a_refused_publish_leaves_the_acquisition_to_present() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    // The drawable is 1x1, so a 2x2 source is refused by the shared extent
    // rule rather than being scaled onto it by whatever the driver keeps.
    let source = api
        .create_texture_resource(GlTextureDesc {
            extent: GlExtent3d {
                width: 2,
                height: 2,
                depth_or_layers: 1,
            },
            ..mock_texture_desc()
        })
        .expect("source texture");
    let lease = acquire_one_frame(&mut api);

    assert!(api.publish_surface_image(lease, source).is_err());
    assert!(
        !api.calls()
            .iter()
            .any(|call| matches!(call, MockCall::PublishSurface { .. })),
        "a refused publish must not be recorded as one"
    );
    // The refusal happens before the consume, so the frame the caller already
    // has is still presentable.
    api.present_surface(lease).expect("present");
}
