//! Contract tests for the mock recorder's own deterministic oracle.
//!
//! These assert what the recorder is allowed to *claim* about work it never
//! executed, which is a property of the recorder rather than of the Layer 1
//! contract, so they live next to the recorder.
//!
//! The discovery fixture below is deliberately a copy of the API-level one:
//! `api/tests.rs` is currently over the 1500-line ceiling in the ecosystem
//! module rule and is scheduled to split, at which point both suites share one
//! `pub(crate)` fixture.  Duplicating it for now keeps this evidence
//! independent of a file another work package is editing.

use super::*;

fn limits() -> GlLimits {
    GlLimits {
        max_texture_size: 2_048,
        max_3d_texture_size: 256,
        max_array_texture_layers: 256,
        max_cube_map_texture_size: 2_048,
        max_renderbuffer_size: 4_096,
        max_color_attachments: 4,
        max_draw_buffers: 4,
        max_vertex_attributes: 16,
        max_viewport_dimensions: [2_048; 2],
        max_viewports: 1,
        max_vertex_texture_image_units: 16,
        max_fragment_texture_image_units: 16,
        max_combined_texture_image_units: 16,
        max_uniform_buffer_bindings: 24,
        max_uniform_block_size: 16_384,
        uniform_buffer_offset_alignment: 256,
        max_vertex_uniform_blocks: 12,
        max_fragment_uniform_blocks: 12,
        max_compute_uniform_blocks: 12,
        max_combined_uniform_blocks: 24,
        max_storage_buffer_bindings: 8,
        max_storage_block_size: 1 << 27,
        storage_buffer_offset_alignment: 256,
        max_vertex_storage_blocks: 4,
        max_fragment_storage_blocks: 4,
        max_compute_storage_blocks: 4,
        max_combined_storage_blocks: 8,
        max_image_units: 4,
        max_combined_image_units: 4,
        max_samples: 4,
        max_color_texture_samples: 4,
        max_depth_texture_samples: 4,
        max_integer_samples: 4,
        max_compute_work_group_count: [65_535; 3],
        max_compute_work_group_size: [1_024, 1_024, 64],
        max_compute_work_group_invocations: 1_024,
        max_multiview_view_count: 2,
        max_multi_draw_indirect_count: Some(1),
        query_counter_bits: 32,
        max_texture_anisotropy: GlFiniteF32::new(16.0),
    }
}

fn formats() -> GlFormatTable {
    let mut table = GlFormatTable::default();
    for format in [
        GlFormat::Rgba8Unorm,
        GlFormat::Rgba8Srgb,
        GlFormat::Depth32Float,
    ] {
        table
            .record(GlFormatCapabilities {
                format,
                resource_kind: GlFormatResourceKind::Texture,
                sample_count: 1,
                evidence: GlFormatEvidence::CoreGuaranteed,
                sampled: true,
                filterable: true,
                renderable: true,
                blendable: true,
                storage_read: false,
                storage_write: false,
                copy_source: true,
                copy_destination: true,
            })
            .expect("unique fact");
    }
    table
}

/// A recorder bound to a WebGL2 snapshot at exactly the profile minimums.
fn recorder() -> MockGlFamilyApi {
    let stamp = ContextStamp::new(
        DeviceIdentity::new(7).expect("identity"),
        ContextEpoch::INITIAL,
    );
    let snapshot = GlDiscoveryBuilder::new(
        stamp,
        GlContextInfo::new(
            GlFamilyProfile::WebGl2,
            "version",
            "glsl",
            "vendor",
            "renderer",
            "driver",
            GlContextFlags::default(),
        ),
        GlExtensionSet::default(),
        limits(),
        formats(),
    )
    .expect("test discovery")
    .build();
    MockGlFamilyApi::from_discovery(snapshot)
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
