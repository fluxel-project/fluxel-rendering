//! Contract tests for the statistics module (§47).
//!
//! These pin the properties the rest of the RHI depends on: a collection epoch
//! is a real boundary, a counter saturates instead of wrapping, a snapshot is
//! never half of an event, a level does not collect what it does not name, the
//! estimate is descriptor arithmetic, and a delta never silently mixes two
//! devices or two epochs.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    CommandKind, CommandStatistics, CreatedObject, DeviceStatistics, LaneWork, MemoryEstimate,
    MemoryEstimateQuality, ObjectKind, RecorderCounters, ScopeKind, StatisticsClock,
    StatisticsConfig, StatisticsDetail, SubmissionEvent, UsedObject,
};
use crate::rhi::format::{SubmissionLaneId, TextureFormat};
use crate::rhi::platform::{
    DeviceIdentity, ObjectId, RhiErrorKind, next_identity, next_object_id,
};
use crate::rhi::presentation::{AcquireErrorKind, AcquiredFrameId, PresentFailure, PresentState};
use crate::rhi::resource::{
    Buffer, BufferBackend, BufferDescriptor, BufferUsage, Texture, TextureBackend,
    TextureDescriptor, TextureUsage,
};

/// A statistics domain with no device behind it.
///
/// A backend is the only production source of a domain, but a contract test only
/// needs the domain itself, so it mints its own identity.
fn domain() -> DeviceStatistics {
    DeviceStatistics::new(next_identity())
}

/// A monotonic clock the test drives by hand.
struct TestClock {
    now_ns: AtomicU64,
}

impl TestClock {
    fn new() -> Self {
        Self {
            now_ns: AtomicU64::new(0),
        }
    }

    fn advance(&self, ns: u64) {
        self.now_ns.fetch_add(ns, Ordering::Relaxed);
    }
}

impl StatisticsClock for TestClock {
    fn now_ns(&self) -> u64 {
        self.now_ns.load(Ordering::Relaxed)
    }
}

/// A domain measured on `clock`.
fn clocked_domain(clock: &Arc<TestClock>) -> DeviceStatistics {
    let clock: Arc<TestClock> = Arc::clone(clock);
    let clock: Arc<dyn StatisticsClock> = clock;
    DeviceStatistics::with_clock(next_identity(), clock)
}

/// A buffer handle reporting `descriptor` for `device`.
struct TestBuffer {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: BufferDescriptor,
}

impl BufferBackend for TestBuffer {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn descriptor(&self) -> &BufferDescriptor {
        &self.descriptor
    }
}

fn test_buffer(device: DeviceIdentity, descriptor: BufferDescriptor) -> Buffer {
    Buffer::new(Arc::new(TestBuffer {
        id: next_object_id(),
        device,
        descriptor,
    }))
}

/// A texture handle reporting `descriptor` for `device`.
struct TestTexture {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: TextureDescriptor,
}

impl TextureBackend for TestTexture {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn descriptor(&self) -> &TextureDescriptor {
        &self.descriptor
    }
}

fn test_texture(device: DeviceIdentity, descriptor: TextureDescriptor) -> Texture {
    Texture::new(Arc::new(TestTexture {
        id: next_object_id(),
        device,
        descriptor,
    }))
}

/// Reports one texture creation, so the inventory sees it.
fn create_texture(stats: &DeviceStatistics, descriptor: &TextureDescriptor) {
    stats.record_object_created(&CreatedObject::Texture {
        id: next_object_id(),
        descriptor,
    });
}

/// The estimate of a texture built from `descriptor`, as a plain byte count.
fn texture_bytes(stats: &DeviceStatistics, descriptor: TextureDescriptor) -> Option<u64> {
    let texture = test_texture(stats.inner.identity, descriptor);
    stats
        .estimate_texture_memory(&texture)
        .expect("the texture belongs to this domain")
        .logical_estimated_bytes
}

#[test]
fn a_fresh_domain_collects_nothing_and_owes_nothing() {
    let stats = domain();

    assert_eq!(stats.collection_epoch(), 0);
    assert_eq!(stats.config().detail(), StatisticsDetail::Minimal);

    let snapshot = stats.snapshot();
    assert_eq!(snapshot.sequence(), 1);
    assert_eq!(snapshot.collection_epoch(), 0);
    assert_eq!(snapshot.device_identity(), stats.inner.identity);
    assert_eq!(snapshot.cumulative().commands.recorders_finished, 0);

    let interval = snapshot.delta_since(&snapshot).unwrap();
    assert_eq!(interval.elapsed_cpu_ns, 0);
    assert!(interval.lanes.is_empty());
    assert!(
        interval.working_set.is_none(),
        "the working set is not collected below Detailed"
    );

    let inventory = stats.inventory().unwrap();
    assert_eq!(inventory.objects.buffers, 0);
    assert_eq!(inventory.objects.textures, 0);
    assert_eq!(inventory.memory.total_resources.logical_estimated_bytes, Some(0));
}

#[test]
fn configure_starts_a_new_epoch_and_restarts_event_counters() {
    let stats = domain();
    stats.record_recorder_finished();
    stats.record_acquire_succeeded();
    assert_eq!(stats.snapshot().cumulative().commands.recorders_finished, 1);

    stats.configure(StatisticsConfig::basic()).unwrap();

    assert_eq!(stats.collection_epoch(), 1);
    assert_eq!(stats.config().detail(), StatisticsDetail::Basic);
    let snapshot = stats.snapshot();
    assert_eq!(snapshot.collection_epoch(), 1);
    assert_eq!(
        snapshot.cumulative().commands.recorders_finished,
        0,
        "an event counter restarts with the epoch that defines its rule"
    );
    assert_eq!(snapshot.cumulative().presentation.acquires_succeeded, 0);

    stats.configure(StatisticsConfig::minimal()).unwrap();
    assert_eq!(stats.collection_epoch(), 2);
    assert_eq!(stats.config().detail(), StatisticsDetail::Minimal);
}

#[test]
fn configure_keeps_the_live_inventory() {
    let stats = domain();
    let buffer = test_buffer(
        stats.inner.identity,
        BufferDescriptor::new(1024, BufferUsage::UNIFORM),
    );
    stats.record_object_created(&CreatedObject::Buffer {
        id: buffer.id(),
        descriptor: buffer.descriptor(),
    });
    stats.record_recorder_finished();

    stats.configure(StatisticsConfig::minimal()).unwrap();

    let inventory = stats.inventory().unwrap();
    assert_eq!(
        inventory.objects.buffers, 1,
        "an existing object is not a property of the collection rule"
    );
    assert_eq!(inventory.memory.buffers.logical_estimated_bytes, Some(1024));
    assert_eq!(
        stats.snapshot().cumulative().commands.recorders_finished,
        0,
        "the event counter did restart"
    );
}

#[test]
fn snapshots_of_different_devices_cannot_be_differenced() {
    let first = domain();
    let second = domain();

    let error = second
        .snapshot()
        .delta_since(&first.snapshot())
        .expect_err("two devices are two domains");

    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.operation(), Some("delta_since"));
}

#[test]
fn snapshots_of_different_epochs_cannot_be_differenced() {
    let stats = domain();
    let before = stats.snapshot();
    stats.configure(StatisticsConfig::basic()).unwrap();

    let error = stats
        .snapshot()
        .delta_since(&before)
        .expect_err("two collection rules cannot be differenced");

    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.operation(), Some("delta_since"));
}

#[test]
fn reversed_snapshots_cannot_be_differenced() {
    let stats = domain();
    let first = stats.snapshot();
    let second = stats.snapshot();

    assert!(second.delta_since(&first).is_ok());

    let error = first
        .delta_since(&second)
        .expect_err("an older snapshot cannot be the newer end of an interval");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.operation(), Some("delta_since"));
}

#[test]
fn delta_reports_exactly_what_the_interval_recorded() {
    let clock = Arc::new(TestClock::new());
    let stats = clocked_domain(&clock);
    let identity = stats.inner.identity;

    let before = stats.snapshot();
    clock.advance(5_000_000);
    stats.record_recorder_finished();
    stats.record_acquire_succeeded();
    stats.record_present_planned();
    stats.record_present_terminal(&PresentState::Accepted);
    let buffer = test_buffer(identity, BufferDescriptor::new(4096, BufferUsage::UNIFORM));
    stats.record_object_created(&CreatedObject::Buffer {
        id: buffer.id(),
        descriptor: buffer.descriptor(),
    });
    stats.record_submission(&SubmissionEvent {
        plans_accepted: 1,
        batches_planned: 2,
        batches_accepted: 2,
        recorded_work_items: 4,
        ..SubmissionEvent::default()
    });
    let after = stats.snapshot();

    assert_eq!(before.sequence(), 1);
    assert_eq!(after.sequence(), 2);
    assert_eq!(after.cpu_time_ns(), 5_000_000);

    let interval = after.delta_since(&before).unwrap();
    assert_eq!(interval.device, identity);
    assert_eq!(interval.collection_epoch, 0);
    assert_eq!(interval.elapsed_cpu_ns, 5_000_000);
    assert_eq!(interval.commands.recorders_finished, 1);
    assert_eq!(interval.submissions.submission_calls, 1);
    assert_eq!(interval.submissions.plans_accepted, 1);
    assert_eq!(interval.submissions.batches_planned, 2);
    assert_eq!(interval.submissions.batches_accepted, 2);
    assert_eq!(interval.submissions.recorded_work_items_accepted, 4);
    assert_eq!(interval.presentation.acquires_succeeded, 1);
    assert_eq!(interval.presentation.presents_planned, 1);
    assert_eq!(interval.presentation.presents_accepted, 1);
    assert_eq!(interval.resources.buffers_created, 1);
    assert!(interval.working_set.is_none());
}

#[test]
fn minimal_collects_no_command_or_binding_counters() {
    let stats = domain();
    let mut recorder = RecorderCounters::new(StatisticsDetail::Detailed);
    recorder.note_scope(ScopeKind::Raster);
    recorder.note_command(CommandKind::Draw);
    recorder.note_pipeline_bind();
    recorder.note_pipeline_change();

    assert_eq!(recorder.commands().draw_calls, 1, "the recorder did collect");
    assert_eq!(recorder.bindings().pipeline_changes, 1);

    stats.merge_recorder(&recorder, &[UsedObject::Buffer(next_object_id())]);

    let snapshot = stats.snapshot();
    assert_eq!(snapshot.cumulative().commands.raster_scopes, 0);
    assert_eq!(snapshot.cumulative().commands.draw_calls, 0);
    assert_eq!(snapshot.cumulative().bindings.pipeline_bind_calls, 0);
    assert_eq!(snapshot.cumulative().bindings.pipeline_changes, 0);
}

#[test]
fn merge_is_gated_by_the_current_level_not_the_recorder_level() {
    let stats = domain();
    let mut recorder = RecorderCounters::new(StatisticsDetail::Detailed);
    recorder.note_command(CommandKind::Draw);
    recorder.note_pipeline_change();
    assert_eq!(recorder.commands().draw_calls, 1);

    // The level changed while the recorder was still open, so the counters it
    // collected belong to a rule the current epoch does not apply.
    stats.configure(StatisticsConfig::minimal()).unwrap();
    stats.merge_recorder(&recorder, &[]);

    let snapshot = stats.snapshot();
    assert_eq!(snapshot.cumulative().commands.draw_calls, 0);
    assert_eq!(snapshot.cumulative().bindings.pipeline_changes, 0);
}

#[test]
fn basic_collects_command_and_bind_calls_but_not_state_changes() {
    let stats = domain();
    stats.configure(StatisticsConfig::basic()).unwrap();

    let mut recorder = RecorderCounters::new(StatisticsDetail::Basic);
    recorder.note_scope(ScopeKind::Raster);
    recorder.note_scope(ScopeKind::Compute);
    recorder.note_command(CommandKind::Draw);
    recorder.note_command(CommandKind::DrawIndexed);
    recorder.note_command(CommandKind::Dispatch);
    recorder.note_pipeline_bind();
    recorder.note_bind_group_bind();
    recorder.note_buffer_bind();
    recorder.note_pipeline_change();
    stats.merge_recorder(&recorder, &[]);
    stats.record_recorder_finished();

    let snapshot = stats.snapshot();
    let commands = &snapshot.cumulative().commands;
    assert_eq!(commands.raster_scopes, 1);
    assert_eq!(commands.compute_scopes, 1);
    assert_eq!(commands.draw_calls, 1);
    assert_eq!(commands.draw_indexed_calls, 1);
    assert_eq!(commands.dispatch_calls, 1);
    assert_eq!(commands.recorders_finished, 1);

    let bindings = &snapshot.cumulative().bindings;
    assert_eq!(bindings.pipeline_bind_calls, 1);
    assert_eq!(bindings.bind_group_bind_calls, 1);
    assert_eq!(bindings.buffer_bind_calls, 1);
    assert_eq!(
        bindings.pipeline_changes, 0,
        "effective state changes are a Detailed counter"
    );
}

#[test]
fn detailed_collects_state_changes_that_basic_left_at_zero() {
    let stats = domain();
    stats.configure(StatisticsConfig::basic()).unwrap();
    let mut basic = RecorderCounters::new(StatisticsDetail::Basic);
    basic.note_pipeline_change();
    basic.note_texture_binding_change();
    stats.merge_recorder(&basic, &[]);
    assert_eq!(stats.snapshot().cumulative().bindings.pipeline_changes, 0);

    stats.configure(StatisticsConfig::detailed()).unwrap();
    let mut detailed = RecorderCounters::new(StatisticsDetail::Detailed);
    detailed.note_pipeline_change();
    detailed.note_shader_set_change();
    detailed.note_bind_group_change();
    detailed.note_buffer_binding_change();
    detailed.note_texture_binding_change();
    detailed.note_sampler_binding_change();
    detailed.note_render_target_set_change();
    stats.merge_recorder(&detailed, &[]);

    let snapshot = stats.snapshot();
    let bindings = &snapshot.cumulative().bindings;
    assert_eq!(bindings.pipeline_changes, 1);
    assert_eq!(bindings.shader_set_changes, 1);
    assert_eq!(bindings.bind_group_changes, 1);
    assert_eq!(bindings.buffer_binding_changes, 1);
    assert_eq!(bindings.texture_binding_changes, 1);
    assert_eq!(bindings.sampler_binding_changes, 1);
    assert_eq!(bindings.render_target_set_changes, 1);
}

#[test]
fn counters_saturate_instead_of_wrapping() {
    let stats = domain();
    let saturated = SubmissionEvent {
        batches_accepted: u64::MAX,
        recorded_work_items: u64::MAX,
        ..SubmissionEvent::default()
    };
    stats.record_submission(&saturated);
    stats.record_submission(&saturated);

    let snapshot = stats.snapshot();
    let submissions = &snapshot.cumulative().submissions;
    assert_eq!(submissions.batches_accepted, u64::MAX);
    assert_eq!(submissions.recorded_work_items_accepted, u64::MAX);
    assert_eq!(submissions.submission_calls, 2);

    // The command path saturates the same way.
    let mut total = CommandStatistics::default();
    let full = CommandStatistics {
        draw_calls: u64::MAX,
        debug_markers: u64::MAX,
        ..CommandStatistics::default()
    };
    total.saturating_add_gated(&full, StatisticsDetail::Detailed);
    total.saturating_add_gated(&full, StatisticsDetail::Detailed);
    assert_eq!(total.draw_calls, u64::MAX);
    assert_eq!(total.debug_markers, u64::MAX);
}

#[test]
fn working_set_counts_distinct_objects_per_interval() {
    let stats = domain();
    stats.configure(StatisticsConfig::detailed()).unwrap();
    let first = next_object_id();
    let second = next_object_id();

    let before = stats.snapshot();
    stats.record_use(UsedObject::Buffer(first));
    stats.record_use(UsedObject::Buffer(second));
    stats.record_use(UsedObject::Buffer(first));
    stats.record_use(UsedObject::Texture(first));
    let after = stats.snapshot();

    let working_set = after.delta_since(&before).unwrap().working_set.unwrap();
    assert_eq!(working_set.unique_buffers, 2, "a repeat use is not a new object");
    assert_eq!(working_set.unique_textures, 1);
    assert_eq!(working_set.unique_samplers, 0);

    // The next interval used nothing.
    let later = stats.snapshot();
    let next = later.delta_since(&after).unwrap().working_set.unwrap();
    assert_eq!(next.unique_buffers, 0);
    assert_eq!(next.unique_textures, 0);
}

#[test]
fn recorder_uses_reach_the_working_set_at_detailed_only() {
    let stats = domain();
    let buffer = next_object_id();
    let mut recorder = RecorderCounters::new(StatisticsDetail::Detailed);
    recorder.note_command(CommandKind::Draw);

    stats.merge_recorder(&recorder, &[UsedObject::Buffer(buffer)]);
    let before = stats.snapshot();
    assert!(before.delta_since(&before).unwrap().working_set.is_none());

    stats.configure(StatisticsConfig::detailed()).unwrap();
    let before = stats.snapshot();
    stats.merge_recorder(&recorder, &[UsedObject::Buffer(buffer)]);
    let after = stats.snapshot();
    let working_set = after.delta_since(&before).unwrap().working_set.unwrap();
    assert_eq!(working_set.unique_buffers, 1);
}

#[test]
fn lane_intervals_keep_only_lanes_used_in_the_interval_in_canonical_order() {
    let stats = domain();
    let lane_zero = SubmissionLaneId::new(0);
    let lane_one = SubmissionLaneId::new(1);
    let before = stats.snapshot();

    stats.record_submission(&SubmissionEvent {
        plans_accepted: 1,
        batches_planned: 5,
        batches_accepted: 3,
        recorded_work_items: 3,
        cross_lane_dependencies: 1,
        gpu_dependency_routes: 1,
        lanes: vec![
            LaneWork {
                lane: lane_one,
                batches_accepted: 2,
                recorded_work_items: 2,
            },
            LaneWork {
                lane: lane_zero,
                batches_accepted: 1,
                recorded_work_items: 1,
            },
        ],
        ..SubmissionEvent::default()
    });
    let after = stats.snapshot();

    let interval = after.delta_since(&before).unwrap();
    assert_eq!(interval.lanes.len(), 2);
    assert_eq!(interval.lanes[0].lane, lane_zero, "lanes are canonically ordered");
    assert_eq!(interval.lanes[0].batches_accepted, 1);
    assert_eq!(interval.lanes[0].recorded_work_items, 1);
    assert_eq!(interval.lanes[1].lane, lane_one);
    assert_eq!(interval.lanes[1].batches_accepted, 2);
    assert_eq!(interval.lanes[1].recorded_work_items, 2);
    assert_eq!(interval.submissions.batches_accepted, 3);
    assert_eq!(interval.submissions.submission_calls, 1);

    stats.record_submission(&SubmissionEvent {
        lanes: vec![LaneWork {
            lane: lane_one,
            batches_accepted: 1,
            recorded_work_items: 1,
        }],
        ..SubmissionEvent::default()
    });
    let later = stats.snapshot();
    let second = later.delta_since(&after).unwrap();
    assert_eq!(second.lanes.len(), 1, "an unused lane is not reported");
    assert_eq!(second.lanes[0].lane, lane_one);
    assert_eq!(second.lanes[0].batches_accepted, 1);
}

#[test]
fn reclaim_removes_an_object_from_the_inventory_and_the_working_set() {
    let stats = domain();
    stats.configure(StatisticsConfig::detailed()).unwrap();
    let id = next_object_id();
    let descriptor = BufferDescriptor::new(2048, BufferUsage::STORAGE);
    stats.record_object_created(&CreatedObject::Buffer {
        id,
        descriptor: &descriptor,
    });

    let before = stats.snapshot();
    stats.record_use(UsedObject::Buffer(id));
    let after = stats.snapshot();
    assert_eq!(
        after.delta_since(&before).unwrap().working_set.unwrap().unique_buffers,
        1
    );

    stats.record_object_reclaimed(ObjectKind::Buffer, id);

    let inventory = stats.inventory().unwrap();
    assert_eq!(inventory.objects.buffers, 0);
    assert_eq!(inventory.memory.buffers.logical_estimated_bytes, Some(0));
    let snapshot = stats.snapshot();
    assert_eq!(snapshot.cumulative().resources.buffers_created, 1);
    assert_eq!(snapshot.cumulative().resources.buffers_reclaimed, 1);

    let later = stats.snapshot();
    assert_eq!(
        later.delta_since(&snapshot).unwrap().working_set.unwrap().unique_buffers,
        0,
        "a reclaimed object cannot be the working set of a later interval"
    );
}

#[test]
fn inventory_counts_texture_usages_as_overlapping_subsets() {
    let stats = domain();
    let color = TextureDescriptor::new_2d(
        4,
        4,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT,
    );
    let depth = TextureDescriptor::new_2d(
        4,
        4,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT,
    );
    let sampled = TextureDescriptor::new_2d(
        4,
        4,
        TextureFormat::Rgba8Unorm,
        TextureUsage::SAMPLED.union(TextureUsage::COPY_DST),
    );
    create_texture(&stats, &color);
    create_texture(&stats, &depth);
    create_texture(&stats, &sampled);
    let view = next_object_id();
    stats.record_object_created(&CreatedObject::TextureView(view));

    let inventory = stats.inventory().unwrap();
    assert_eq!(inventory.objects.textures, 3);
    assert_eq!(inventory.objects.render_target_textures, 2);
    assert_eq!(inventory.objects.color_attachment_textures, 1);
    assert_eq!(inventory.objects.depth_stencil_textures, 1);
    assert_eq!(inventory.objects.texture_views, 1);
    assert_eq!(inventory.memory.textures.logical_estimated_bytes, Some(192));
    assert_eq!(
        inventory.memory.render_target_textures.logical_estimated_bytes,
        Some(128)
    );
    assert_eq!(inventory.memory.total_resources.logical_estimated_bytes, Some(192));
}

#[test]
fn outstanding_frames_track_acquisition_and_terminal_states() {
    let stats = domain();
    let frame = AcquiredFrameId::new(stats.inner.identity, 1);

    assert_eq!(stats.inventory().unwrap().objects.outstanding_frames, 0);
    stats.record_frame_acquired(frame);
    assert_eq!(stats.inventory().unwrap().objects.outstanding_frames, 1);
    stats.record_frame_terminal(frame);
    assert_eq!(stats.inventory().unwrap().objects.outstanding_frames, 0);
}

#[test]
fn frame_attachments_leave_the_working_set_when_the_frame_ends() {
    let stats = domain();
    stats.configure(StatisticsConfig::detailed()).unwrap();
    let frame = AcquiredFrameId::new(stats.inner.identity, 7);
    stats.record_frame_acquired(frame);

    let before = stats.snapshot();
    stats.record_use(UsedObject::FrameAttachment(frame));
    let after = stats.snapshot();
    let working_set = after.delta_since(&before).unwrap().working_set.unwrap();
    assert_eq!(working_set.unique_frame_attachments, 1);

    stats.record_frame_terminal(frame);
    let snapshot = stats.snapshot();
    let later = stats.snapshot();
    let next = later.delta_since(&snapshot).unwrap().working_set.unwrap();
    assert_eq!(next.unique_frame_attachments, 0);
}

#[test]
fn presentation_refusals_and_outcomes_stay_disjoint() {
    let stats = domain();
    let before = stats.snapshot();

    stats.record_acquire_refused(AcquireErrorKind::NotReady);
    stats.record_acquire_refused(AcquireErrorKind::Timeout);
    stats.record_acquire_refused(AcquireErrorKind::Outdated);
    stats.record_acquire_refused(AcquireErrorKind::TargetLost);
    // These four produced no frame, so they planned no present and belong to no
    // present category.
    stats.record_acquire_refused(AcquireErrorKind::FrameOutstanding);
    stats.record_acquire_refused(AcquireErrorKind::ZeroSizeOrSuspended);
    stats.record_acquire_refused(AcquireErrorKind::DeviceLost);
    stats.record_acquire_refused(AcquireErrorKind::OutOfMemory);

    stats.record_acquire_succeeded();
    stats.record_present_planned();
    stats.record_present_terminal(&PresentState::Pending);
    stats.record_present_terminal(&PresentState::Accepted);
    stats.record_present_terminal(&PresentState::Outdated);
    stats.record_present_terminal(&PresentState::TargetLost);
    stats.record_present_terminal(&PresentState::Failed(PresentFailure::new("terminal")));
    stats.record_frame_abandoned();

    let after = stats.snapshot();
    let presentation = &after.delta_since(&before).unwrap().presentation;
    assert_eq!(presentation.acquires_succeeded, 1);
    assert_eq!(presentation.acquire_not_ready, 1);
    assert_eq!(presentation.acquire_timeout, 1);
    assert_eq!(presentation.acquire_outdated, 1);
    assert_eq!(presentation.acquire_target_lost, 1);
    assert_eq!(
        presentation.acquires_succeeded
            + presentation.acquire_not_ready
            + presentation.acquire_timeout
            + presentation.acquire_outdated
            + presentation.acquire_target_lost,
        5,
        "the four uncounted refusals moved no acquire counter"
    );

    assert_eq!(presentation.presents_planned, 1);
    assert_eq!(presentation.presents_accepted, 1);
    assert_eq!(presentation.presents_outdated, 1);
    assert_eq!(presentation.presents_target_lost, 1);
    assert_eq!(presentation.presents_failed, 1);
    assert_eq!(
        presentation.presents_accepted
            + presentation.presents_outdated
            + presentation.presents_target_lost
            + presentation.presents_failed,
        4,
        "Pending is not terminal and device loss is not a present outcome"
    );
    assert_eq!(presentation.frames_abandoned, 1);
}

#[test]
fn buffer_estimate_is_the_descriptor_size() {
    let stats = domain();
    let buffer = test_buffer(
        stats.inner.identity,
        BufferDescriptor::new(4096, BufferUsage::UNIFORM),
    );

    let estimate = stats.estimate_buffer_memory(&buffer).unwrap();
    assert_eq!(estimate.logical_estimated_bytes, Some(4096));
    assert_eq!(estimate.quality, MemoryEstimateQuality::LogicalEstimate);

    let absent = MemoryEstimate::default();
    assert_eq!(absent.logical_estimated_bytes, None);
    assert_eq!(absent.quality, MemoryEstimateQuality::Unknown);
}

#[test]
fn texture_estimate_sums_mips_layers_and_samples() {
    let stats = domain();

    // 4x4 Rgba8: 64 + 16 + 4.
    let mipped = TextureDescriptor::new_2d(
        4,
        4,
        TextureFormat::Rgba8Unorm,
        TextureUsage::SAMPLED,
    )
    .with_mip_levels(3);
    assert_eq!(texture_bytes(&stats, mipped), Some(84));

    // 8x8 Rgba8 is 256 bytes per layer.
    let layered = TextureDescriptor::new_2d(
        8,
        8,
        TextureFormat::Rgba8Unorm,
        TextureUsage::SAMPLED,
    )
    .with_array_layers(4);
    assert_eq!(texture_bytes(&stats, layered), Some(1024));

    // 8x8 Rgba8 is 256 bytes per sample of a multisampled attachment.
    let multisampled = TextureDescriptor::new_2d(
        8,
        8,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT,
    )
    .with_sample_count(4);
    assert_eq!(texture_bytes(&stats, multisampled), Some(1024));

    // 4x4x2 R8 with two mips: 32 + 4.
    let volume = TextureDescriptor::new_3d(
        4,
        4,
        2,
        TextureFormat::R8Unorm,
        TextureUsage::SAMPLED,
    )
    .with_mip_levels(2);
    assert_eq!(texture_bytes(&stats, volume), Some(36));

    // 8x8 depth32float is 256 bytes.
    let depth = TextureDescriptor::new_2d(
        8,
        8,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT,
    );
    assert_eq!(texture_bytes(&stats, depth), Some(256));
}

#[test]
fn texture_estimate_sums_a_mip_tail_analytically() {
    let stats = domain();
    // Every level of a 1x1 texture is one texel, so forty declared levels are
    // forty bytes: the tail past the 32nd level is summed, not iterated.
    let tail = TextureDescriptor::new_2d(1, 1, TextureFormat::R8Unorm, TextureUsage::SAMPLED)
        .with_mip_levels(40);

    assert_eq!(texture_bytes(&stats, tail), Some(40));
}

#[test]
fn texture_estimate_is_unknown_for_implementation_defined_backing() {
    let stats = domain();
    let descriptor = TextureDescriptor::new_2d(
        8,
        8,
        TextureFormat::Depth24Plus,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT,
    );
    let texture = test_texture(stats.inner.identity, descriptor);

    let estimate = stats.estimate_texture_memory(&texture).unwrap();
    assert_eq!(estimate.logical_estimated_bytes, None);
    assert_eq!(estimate.quality, MemoryEstimateQuality::Unknown);
}

#[test]
fn inventory_memory_is_unknown_when_one_class_is_unknown() {
    let stats = domain();
    let buffer = test_buffer(
        stats.inner.identity,
        BufferDescriptor::new(1024, BufferUsage::UNIFORM),
    );
    stats.record_object_created(&CreatedObject::Buffer {
        id: buffer.id(),
        descriptor: buffer.descriptor(),
    });
    create_texture(
        &stats,
        &TextureDescriptor::new_2d(
            8,
            8,
            TextureFormat::Depth24Plus,
            TextureUsage::DEPTH_STENCIL_ATTACHMENT,
        ),
    );

    let inventory = stats.inventory().unwrap();
    assert_eq!(inventory.memory.buffers.logical_estimated_bytes, Some(1024));
    assert_eq!(inventory.memory.textures.logical_estimated_bytes, None);
    assert_eq!(
        inventory.memory.total_resources.logical_estimated_bytes,
        None,
        "an unknown class is not zero"
    );
    assert_eq!(
        inventory.memory.total_resources.quality,
        MemoryEstimateQuality::Unknown
    );
}

#[test]
fn estimating_a_foreign_resource_is_refused() {
    let stats = domain();
    let buffer = test_buffer(
        next_identity(),
        BufferDescriptor::new(64, BufferUsage::UNIFORM),
    );
    let error = stats.estimate_buffer_memory(&buffer).unwrap_err();
    assert_eq!(error.kind(), RhiErrorKind::WrongDevice);
    assert_eq!(error.operation(), Some("estimate_buffer_memory"));
    assert_eq!(error.object(), Some(buffer.id()));

    let texture = test_texture(
        next_identity(),
        TextureDescriptor::new_2d(4, 4, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED),
    );
    let error = stats.estimate_texture_memory(&texture).unwrap_err();
    assert_eq!(error.kind(), RhiErrorKind::WrongDevice);
    assert_eq!(error.operation(), Some("estimate_texture_memory"));
    assert_eq!(error.object(), Some(texture.id()));
}

#[test]
fn sampler_reports_no_rate_before_time_elapses() {
    let clock = Arc::new(TestClock::new());
    let stats = clocked_domain(&clock);
    let mut sampler = stats.frame_sampler();

    let frame = sampler.sample_frame().unwrap();
    assert_eq!(frame.interval().elapsed_cpu_ns, 0);
    assert_eq!(
        frame.fps(),
        None,
        "two samples in one clock tick have no measurable rate"
    );
}

#[test]
fn sampler_reports_the_calling_rate_after_time_elapses() {
    let clock = Arc::new(TestClock::new());
    let stats = clocked_domain(&clock);
    let mut sampler = stats.frame_sampler();
    clock.advance(16_666_667);

    let frame = sampler.sample_frame().unwrap();
    assert_eq!(frame.interval().elapsed_cpu_ns, 16_666_667);
    let fps = frame.fps().unwrap();
    assert!(
        (fps - 60.0).abs() < 0.01,
        "one frame per 16.67 ms is about 60 fps, got {fps}"
    );

    clock.advance(8_333_333);
    let frame = sampler.sample_frame().unwrap();
    assert_eq!(frame.interval().elapsed_cpu_ns, 8_333_333);
    assert!((frame.fps().unwrap() - 120.0).abs() < 0.01);
}

#[test]
fn sampler_refuses_to_span_a_collection_epoch() {
    let clock = Arc::new(TestClock::new());
    let stats = clocked_domain(&clock);
    let mut sampler = stats.frame_sampler();
    clock.advance(1_000_000);
    stats.configure(StatisticsConfig::basic()).unwrap();

    let error = sampler.sample_frame().unwrap_err();
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.operation(), Some("sample_frame"));
}

#[test]
fn concurrent_events_are_never_observed_half_applied() {
    let stats = domain();
    stats.configure(StatisticsConfig::basic()).unwrap();
    const EVENTS: u64 = 2_000;

    let writer = {
        let stats = stats.clone();
        std::thread::spawn(move || {
            for _ in 0..EVENTS {
                // One logical event moves both fields, so a snapshot that sees
                // one must see the other.
                stats.record_submission(&SubmissionEvent {
                    plans_accepted: 1,
                    batches_accepted: 1,
                    recorded_work_items: 1,
                    ..SubmissionEvent::default()
                });
            }
        })
    };

    let mut torn = false;
    for _ in 0..EVENTS {
        let snapshot = stats.snapshot();
        let submissions = &snapshot.cumulative().submissions;
        if submissions.batches_accepted != submissions.recorded_work_items_accepted {
            torn = true;
        }
        if submissions.batches_accepted > EVENTS {
            torn = true;
        }
    }
    writer.join().unwrap();

    assert!(!torn, "a snapshot observed half of a logical event");
    let snapshot = stats.snapshot();
    assert_eq!(snapshot.cumulative().submissions.submission_calls, EVENTS);
    assert_eq!(snapshot.cumulative().submissions.batches_accepted, EVENTS);
    assert_eq!(
        snapshot.cumulative().submissions.recorded_work_items_accepted,
        EVENTS
    );
}
