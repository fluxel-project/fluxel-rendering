//! Query API positive, refusal, and boundary conformance cases.

use crate::api::command::RecorderDescriptor;
use crate::api::error::RhiErrorKind;
use crate::api::format::TextureFormat;
use crate::api::identity::ObjectId;
use crate::api::identity::{DeviceIdentity, DeviceInstanceId};
use crate::api::platform::LimitKey;
use crate::api::query::{
    PipelineStatistics, QuerySetDescriptor, QueryType, TimestampQueryCapabilities,
};
use crate::api::resource::{BufferDescriptor, BufferUsage};
use crate::api::resource::{TextureAspects, TextureSubresourceRange, TextureUsage};
use crate::api::tests::fixture;
use crate::api::tests::mock::query_device_for_test;

fn device(instance: u64) -> DeviceIdentity {
    DeviceIdentity::new(DeviceInstanceId::new(instance))
}

#[test]
fn query_set_creation_and_encoder_timestamp_are_positive() {
    let device = query_device_for_test(device(1));
    let set = device
        .create_query_set(&QuerySetDescriptor::new(QueryType::Timestamp, 2))
        .expect("timestamp query set");
    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .expect("recorder");
    recorder
        .write_timestamp(&set, 1)
        .expect("timestamp records");
    assert!(recorder.finish().is_ok());
}

#[test]
fn query_rejects_zero_count_and_out_of_range_index() {
    let device = query_device_for_test(device(1));
    let zero = device
        .create_query_set(&QuerySetDescriptor::new(QueryType::Timestamp, 0))
        .unwrap_err();
    assert_eq!(zero.kind(), RhiErrorKind::InvalidUsage);
    let set = device
        .create_query_set(&QuerySetDescriptor::new(QueryType::Timestamp, 1))
        .unwrap();
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    let error = recorder.write_timestamp(&set, 1).unwrap_err();
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
}

/// Query allocation is bounded by an explicit device fact, not a library
/// constant. The exact ceiling is accepted; its successor is rejected before
/// native allocation, which pins both sides of the public contract.
#[test]
fn query_set_capacity_is_a_reported_limit_with_exact_boundary() {
    let device = query_device_for_test(device(3));
    assert_eq!(
        device.capabilities().limit(LimitKey::MaxQueriesPerQuerySet),
        Some(8)
    );
    assert!(
        device
            .create_query_set(&QuerySetDescriptor::new(QueryType::Timestamp, 8))
            .is_ok()
    );
    assert_eq!(
        device
            .create_query_set(&QuerySetDescriptor::new(QueryType::Timestamp, 9))
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
}

#[test]
fn pipeline_statistics_selects_only_the_supported_counter_subset() {
    let device = query_device_for_test(device(5));
    let selected = PipelineStatistics::VERTEX_SHADER_INVOCATIONS
        .union(PipelineStatistics::FRAGMENT_SHADER_INVOCATIONS);
    assert!(
        device
            .create_query_set(&QuerySetDescriptor::new(
                QueryType::PipelineStatistics(selected),
                1
            ))
            .is_ok()
    );
    assert_eq!(
        device
            .create_query_set(&QuerySetDescriptor::new(
                QueryType::PipelineStatistics(PipelineStatistics::NONE),
                1,
            ))
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
}

#[test]
fn timestamp_facts_expose_conversion_valid_bits_and_nonblocking_boundary() {
    assert!(TimestampQueryCapabilities::new(0.0, None, false).is_none());
    assert!(TimestampQueryCapabilities::new(f64::NAN, None, false).is_none());
    assert!(TimestampQueryCapabilities::new(1.0, Some(0), false).is_none());
    let facts = TimestampQueryCapabilities::new(0.5, Some(36), true).unwrap();
    assert_eq!(facts.ticks_to_nanos(8), Some(4.0));
    assert_eq!(facts.valid_bits, Some(36));
    assert!(facts.non_blocking_resolve);

    let device = query_device_for_test(device(6));
    let reported = device.capabilities().timestamp_queries();
    assert_eq!(reported.period_nanos, Some(1.0));
    assert_eq!(reported.ticks_to_nanos(42), Some(42.0));
    assert!(reported.non_blocking_resolve);
}

#[test]
fn resolve_checks_usage_alignment_range_and_foreign_set() {
    let first = query_device_for_test(device(1));
    let second = query_device_for_test(device(2));
    let set = first
        .create_query_set(&QuerySetDescriptor::new(QueryType::Timestamp, 2))
        .unwrap();
    let foreign = second
        .create_query_set(&QuerySetDescriptor::new(QueryType::Timestamp, 2))
        .unwrap();
    let bad_usage = first
        .create_buffer(&BufferDescriptor::new(32, BufferUsage::COPY_DST))
        .unwrap();
    let target = first
        .create_buffer(&BufferDescriptor::new(32, BufferUsage::QUERY_RESOLVE))
        .unwrap();
    let mut recorder = first.create_recorder(&RecorderDescriptor::new()).unwrap();
    assert_eq!(
        recorder
            .resolve_query_set(&set, 0, 1, &bad_usage, 0)
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
    assert_eq!(
        recorder
            .resolve_query_set(&set, 0, 1, &target, 4)
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
    assert_eq!(
        recorder
            .resolve_query_set(&foreign, 0, 1, &target, 0)
            .unwrap_err()
            .kind(),
        RhiErrorKind::WrongDevice
    );
    assert_eq!(
        recorder
            .resolve_query_set(&set, 1, 2, &target, 0)
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
}

#[test]
fn query_resolve_uses_reported_alignment_not_a_hidden_eight_byte_rule() {
    let device = query_device_for_test(device(4));
    assert_eq!(
        device
            .capabilities()
            .limit(LimitKey::QueryResolveBufferAlignment),
        Some(8)
    );
    let set = device
        .create_query_set(&QuerySetDescriptor::new(QueryType::Timestamp, 1))
        .unwrap();
    let destination = device
        .create_buffer(&BufferDescriptor::new(32, BufferUsage::QUERY_RESOLVE))
        .unwrap();
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    recorder
        .resolve_query_set(&set, 0, 1, &destination, 8)
        .expect("an exactly aligned resolve is recordable");
    assert_eq!(
        recorder
            .resolve_query_set(&set, 0, 1, &destination, 4)
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
}

#[test]
fn one_recorded_work_cannot_write_the_same_timestamp_slot_twice() {
    let device = query_device_for_test(device(7));
    let set = device
        .create_query_set(&QuerySetDescriptor::new(QueryType::Timestamp, 1))
        .unwrap();
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    recorder.write_timestamp(&set, 0).unwrap();
    assert_eq!(
        recorder.write_timestamp(&set, 0).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );
}

#[test]
fn pipeline_statistics_resolve_uses_exact_selected_counter_layout() {
    let device = query_device_for_test(device(6));
    let selected = PipelineStatistics::VERTEX_SHADER_INVOCATIONS
        .union(PipelineStatistics::FRAGMENT_SHADER_INVOCATIONS);
    assert_eq!(selected.result_words(), 2);
    assert_eq!(QueryType::PipelineStatistics(selected).result_words(), 2);
    let set = device
        .create_query_set(&QuerySetDescriptor::new(
            QueryType::PipelineStatistics(selected),
            2,
        ))
        .unwrap();
    // Two queries x two selected u64 counters ends exactly at byte 32.
    let exact = device
        .create_buffer(&BufferDescriptor::new(32, BufferUsage::QUERY_RESOLVE))
        .unwrap();
    let short = device
        .create_buffer(&BufferDescriptor::new(24, BufferUsage::QUERY_RESOLVE))
        .unwrap();
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    recorder.resolve_query_set(&set, 0, 2, &exact, 0).unwrap();
    assert_eq!(
        recorder
            .resolve_query_set(&set, 0, 2, &short, 0)
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage,
        "four result words must not fit in a three-word destination"
    );
}

#[test]
fn clear_buffer_records_only_with_declared_usage_and_range() {
    let device = query_device_for_test(device(1));
    let valid = device
        .create_buffer(&BufferDescriptor::new(16, BufferUsage::COPY_DST))
        .unwrap();
    let invalid = device
        .create_buffer(&BufferDescriptor::new(16, BufferUsage::COPY_SRC))
        .unwrap();
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    assert_eq!(
        recorder
            .clear_buffer(&invalid, crate::api::resource::BufferRange::new(0, 4))
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
    assert_eq!(
        recorder
            .clear_buffer(&valid, crate::api::resource::BufferRange::new(12, 8))
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
    for range in [
        crate::api::resource::BufferRange::new(2, 4),
        crate::api::resource::BufferRange::new(0, 6),
    ] {
        assert_eq!(
            recorder.clear_buffer(&valid, range).unwrap_err().kind(),
            RhiErrorKind::InvalidUsage,
            "native fill routes require four-byte offset and size alignment"
        );
    }
    recorder
        .clear_buffer(&valid, crate::api::resource::BufferRange::new(12, 4))
        .unwrap();
    assert!(recorder.finish().is_ok());
}

#[test]
fn dispatch_indirect_checks_usage_alignment_and_extent_before_lowering() {
    let device = query_device_for_test(device(1));
    let wrong = device
        .create_buffer(&BufferDescriptor::new(16, BufferUsage::COPY_SRC))
        .unwrap();
    let indirect = device
        .create_buffer(&BufferDescriptor::new(16, BufferUsage::INDIRECT))
        .unwrap();
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    let mut scope = recorder
        .begin_compute(&crate::api::command::ComputeScopeDescriptor::new())
        .unwrap();
    assert_eq!(
        scope.dispatch_indirect(&wrong, 0).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );
    assert_eq!(
        scope.dispatch_indirect(&indirect, 2).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );
    assert_eq!(
        scope.dispatch_indirect(&indirect, 8).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );
    // Argument validation has passed; the remaining refusal is the ordinary
    // state-machine rule that dispatches (including indirect ones) need a pipeline.
    assert_eq!(
        scope.dispatch_indirect(&indirect, 0).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );
}

#[test]
fn clear_texture_checks_usage_and_subresource_bounds() {
    let device = query_device_for_test(device(1));
    let valid = fixture::texture(
        ObjectId::next(),
        device.identity(),
        crate::api::resource::TextureDescriptor::new_2d(
            4,
            4,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COPY_DST,
        )
        .with_mip_levels(2),
    );
    let invalid = fixture::texture(
        ObjectId::next(),
        device.identity(),
        crate::api::resource::TextureDescriptor::new_2d(
            4,
            4,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COPY_SRC,
        ),
    );
    let one = TextureSubresourceRange {
        aspects: TextureAspects::COLOR,
        base_mip: 0,
        mip_count: 1,
        base_layer: 0,
        layer_count: 1,
    };
    let over_mip = TextureSubresourceRange {
        base_mip: 1,
        mip_count: 2,
        ..one
    };
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    assert_eq!(
        recorder.clear_texture(&invalid, one).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );
    assert_eq!(
        recorder.clear_texture(&valid, over_mip).unwrap_err().kind(),
        RhiErrorKind::InvalidUsage
    );
    recorder.clear_texture(&valid, one).unwrap();
    assert!(recorder.finish().is_ok());
}

#[test]
fn depth_clear_requires_attachment_usage_before_recording() {
    let device = query_device_for_test(device(1));
    let copy_only = fixture::texture(
        ObjectId::next(),
        device.identity(),
        crate::api::resource::TextureDescriptor::new_2d(
            4,
            4,
            TextureFormat::Depth32Float,
            TextureUsage::COPY_DST,
        ),
    );
    let clearable = fixture::texture(
        ObjectId::next(),
        device.identity(),
        crate::api::resource::TextureDescriptor::new_2d(
            4,
            4,
            TextureFormat::Depth32Float,
            TextureUsage::COPY_DST.union(TextureUsage::DEPTH_STENCIL_ATTACHMENT),
        ),
    );
    let depth = TextureSubresourceRange {
        aspects: TextureAspects::DEPTH,
        base_mip: 0,
        mip_count: 1,
        base_layer: 0,
        layer_count: 1,
    };
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    assert_eq!(
        recorder
            .clear_texture(&copy_only, depth)
            .unwrap_err()
            .kind(),
        RhiErrorKind::InvalidUsage
    );
    recorder.clear_texture(&clearable, depth).unwrap();
    assert!(recorder.finish().is_ok());
}
