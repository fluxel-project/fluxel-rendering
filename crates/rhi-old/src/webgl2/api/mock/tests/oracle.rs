//! Evidence that the recorder's own oracle answers only what it was told.
//!
//! Completion and compressed-upload sizes are the two facts a test must be able
//! to constrain before any other domain assertion means anything: a recorder
//! that invented completion would make every release path pass, and one that
//! measured a compressed slice with the client layout would accept an upload no
//! backend accepts.

use super::*;

#[test]
fn a_compressed_upload_must_be_one_complete_mip_of_the_exact_encoded_size() {
    let mut api = compressed_recorder();
    let texture = api
        .create_texture_resource(GlTextureDesc {
            dimension: GlTextureDimension::D2,
            extent: GlExtent3d {
                width: 4,
                height: 4,
                depth_or_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            format: GlFormat::Bc1RgbaUnorm,
            usage: GlTextureUsage::SAMPLED,
        })
        .expect("compressed texture");
    let region = |origin: [u32; 3], extent: GlExtent3d| GlTextureRegion {
        subresource: GlTextureSubresource {
            texture,
            aspect: GlTextureAspect::Color,
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
        },
        origin,
        extent,
    };
    let full = GlExtent3d {
        width: 4,
        height: 4,
        depth_or_layers: 1,
    };
    // One 4x4 BC1 block encodes to exactly 8 bytes, and that is the only
    // accepted length. Measuring the slice with the client pixel layout would
    // have demanded 64 bytes from the same region and 16 from a 2x2 one.
    assert_eq!(
        api.upload_texture(region([0; 3], full), rgba8_layout(), &[0; 8]),
        Ok(())
    );
    api.clear_calls();
    assert!(
        api.upload_texture(region([0; 3], full), rgba8_layout(), &[0; 64])
            .is_err(),
        "an RGBA8-sized slice is not one encoded BC1 mip"
    );
    assert!(
        api.upload_texture(region([0; 3], full), rgba8_layout(), &[0; 7])
            .is_err(),
        "a truncated block is rejected rather than padded"
    );
    // A region that does not cover its mip is rejected before any size check:
    // compressed storage is undefined until a whole mip defines it, which is
    // exactly what both executable backends enforce.
    assert!(
        api.upload_texture(
            region(
                [0; 3],
                GlExtent3d {
                    width: 2,
                    height: 2,
                    depth_or_layers: 1,
                }
            ),
            rgba8_layout(),
            &[0; 8]
        )
        .is_err(),
        "a compressed sub-rectangle is not a complete mip"
    );
    // Each rejection reached the trace as an error and none of them recorded an
    // upload, so the guard provably precedes the side effect rather than being
    // reported after one.
    assert!(
        api.calls()
            .iter()
            .all(|call| !matches!(call, MockCall::UploadTexture(_))),
        "a rejected compressed upload must not be recorded as an upload"
    );
    assert_eq!(
        api.calls()
            .iter()
            .filter(|call| matches!(call, MockCall::Error(_)))
            .count(),
        3,
        "each rejection is recorded once as an error"
    );
}

#[test]
fn an_uninjected_fence_never_reports_completion() {
    let mut api = recorder();
    let lease = api.create_fence().expect("fence is created");
    // `Pending` is the only honest default: the recorder owns no submission
    // queue, so nothing it recorded can have finished.  A `Complete` default
    // would let a completion-safe release test pass against a mock that never
    // modelled the wait at all.
    assert_eq!(
        api.poll_fence(lease).expect("poll is answered"),
        GlFenceStatus::Pending
    );
    assert_eq!(
        api.wait_fence(lease, GlWaitBound { nanoseconds: 1_000 })
            .expect("wait is answered"),
        GlFenceStatus::Pending
    );
}

#[test]
fn an_injected_completion_is_observable_through_poll_and_wait() {
    let mut api = recorder();
    let lease = api.create_fence().expect("fence is created");
    api.inject_fence_status(lease.fence, GlFenceStatus::Complete);
    assert_eq!(
        api.poll_fence(lease).expect("poll is answered"),
        GlFenceStatus::Complete
    );
    // A bounded wait reports the injected answer rather than inventing
    // progress.  The bound is deliberately not modelled: the recorder has no
    // queue to wait on, and pretending a zero bound differs from a non-zero one
    // would let a caller "prove" bounded-progress behaviour against no clock.
    assert_eq!(
        api.wait_fence(lease, GlWaitBound::POLL)
            .expect("wait is answered"),
        GlFenceStatus::Complete
    );
}

#[test]
fn a_failed_fence_is_reported_distinctly_from_completion() {
    let mut api = recorder();
    let lease = api.create_fence().expect("fence is created");
    api.inject_fence_status(lease.fence, GlFenceStatus::Failed);
    // `Failed` must not collapse into `Complete`: a release path that treats
    // any non-pending answer as safe retirement would retire work whose
    // submission actually failed.
    assert_eq!(
        api.poll_fence(lease).expect("poll is answered"),
        GlFenceStatus::Failed
    );
}

#[test]
fn a_destroyed_fence_can_no_longer_be_observed_at_all() {
    let mut api = recorder();
    let lease = api.create_fence().expect("fence is created");
    api.inject_fence_status(lease.fence, GlFenceStatus::Complete);
    api.destroy_fence(lease).expect("fence is destroyed");
    // The injected answer is removed with the fence, so the oracle describes
    // exactly the live fence set.  The stale lease is rejected before the
    // answer is consulted, which is why removal is hygiene for the map rather
    // than the only guard: `slot()` is monotonic within an epoch and a
    // restoration changes the epoch, so a fence `SyncId` cannot in fact recur.
    assert!(api.poll_fence(lease).is_err());
    assert!(api.wait_fence(lease, GlWaitBound::POLL).is_err());
}
