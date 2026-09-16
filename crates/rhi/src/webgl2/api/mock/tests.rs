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

/// A recorder whose WebGL2 ledger acquired S3TC and whose format table
/// therefore carries one exact compressed fact.
fn compressed_recorder() -> MockGlFamilyApi {
    let extension = GlKnownExtension::CompressedTextureS3tc;
    let mut extensions = GlExtensionSet::default();
    extensions.report_raw("WEBGL_compressed_texture_s3tc");
    assert!(extensions.acquire(extension), "the ledger acquires S3TC");
    let mut table = formats();
    table
        .record(GlFormatCapabilities {
            format: GlFormat::Bc1RgbaUnorm,
            resource_kind: GlFormatResourceKind::Texture,
            sample_count: 1,
            evidence: GlFormatEvidence::ExtensionAcquired(extension),
            sampled: true,
            filterable: true,
            renderable: false,
            blendable: false,
            storage_read: false,
            storage_write: false,
            copy_source: false,
            copy_destination: false,
        })
        .expect("exact compressed fact");
    let stamp = ContextStamp::new(
        DeviceIdentity::new(7).expect("identity"),
        ContextEpoch::INITIAL,
    );
    MockGlFamilyApi::from_discovery(
        GlDiscoveryBuilder::new(
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
            extensions,
            limits(),
            table,
        )
        .expect("test discovery")
        .build(),
    )
}

/// The client layout a compressed upload is *not* measured by.
fn rgba8_layout() -> GlPixelLayout {
    GlPixelLayout {
        format: GlPixelFormat::Rgba8,
        bytes_per_row: 64,
        rows_per_image: 4,
        offset: 0,
        alignment: 4,
        repack: GlRepackPolicy::Disallow,
    }
}

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
            region([0; 3], GlExtent3d {
                width: 2,
                height: 2,
                depth_or_layers: 1,
            }),
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
