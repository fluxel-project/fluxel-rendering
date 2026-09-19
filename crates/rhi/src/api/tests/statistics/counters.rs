//! Counter-record tests (specification 47.5–47.8 and 47.10–47.12).
//!
//! Every type in this file is a plain record of named `u64` fields with no
//! methods, so there is no behaviour to drive. What is worth testing is therefore
//! not what the counters *do* — the recorder that increments them does not exist
//! — but what the records *say*: that a caller can read a named field without
//! going through a map, that the groups stay separate rather than flattening into
//! one struct of sixty fields, and that the boundaries the chapter draws (calls
//! versus changes, planned versus accepted, acquire refusal versus present
//! outcome) are drawn in the type rather than left to a reader's memory.
//!
//! Section 47.21's note on scope applies to all of it: a counter that reads zero
//! here reads zero because nothing was ever recorded. None of these tests is
//! evidence that anything happened, and none of them is GPU evidence.

use crate::api::statistics::{
    BindingStatistics, CommandStatistics, CumulativeStatistics, LaneIntervalStatistics,
    PresentationStatistics, ResourceLifecycleStatistics, SubmissionStatistics,
    WorkingSetStatistics,
};
use crate::api::submission::SubmissionLaneId;

/// Every group is `Default`, which is what lets `CumulativeStatistics` be
/// `Default` and lets a backend start a collection epoch without assembling
/// sixty fields by hand.
///
/// The assertion is that the default claims *nothing happened*: every field is
/// zero. A group that defaulted to a sentinel, or to `u64::MAX` for a saturating
/// counter, would make a freshly restarted epoch report a number it never
/// measured.
#[test]
fn every_counter_group_defaults_to_a_record_of_nothing() {
    let commands = CommandStatistics::default();
    assert_eq!(commands.recorders_finished, 0);
    assert_eq!(commands.raster_scopes, 0);
    assert_eq!(commands.compute_scopes, 0);
    assert_eq!(commands.draw_calls, 0);
    assert_eq!(commands.draw_indexed_calls, 0);
    assert_eq!(commands.dispatch_calls, 0);
    assert_eq!(commands.buffer_copies, 0);
    assert_eq!(commands.buffer_to_texture_copies, 0);
    assert_eq!(commands.texture_to_buffer_copies, 0);
    assert_eq!(commands.texture_copies, 0);
    assert_eq!(commands.resolves, 0);
    assert_eq!(commands.blits, 0);
    assert_eq!(commands.upload_commands, 0);
    assert_eq!(commands.readback_commands, 0);
    assert_eq!(commands.debug_markers, 0);

    let bindings = BindingStatistics::default();
    assert_eq!(bindings.pipeline_bind_calls, 0);
    assert_eq!(bindings.pipeline_changes, 0);
    assert_eq!(bindings.shader_set_changes, 0);
    assert_eq!(bindings.bind_group_bind_calls, 0);
    assert_eq!(bindings.bind_group_changes, 0);
    assert_eq!(bindings.buffer_bind_calls, 0);
    assert_eq!(bindings.buffer_binding_changes, 0);
    assert_eq!(bindings.texture_binding_changes, 0);
    assert_eq!(bindings.sampler_binding_changes, 0);
    assert_eq!(bindings.render_target_set_changes, 0);

    let submissions = SubmissionStatistics::default();
    assert_eq!(submissions.submission_calls, 0);
    assert_eq!(submissions.plans_accepted, 0);
    assert_eq!(submissions.plans_rejected, 0);
    assert_eq!(submissions.batches_planned, 0);
    assert_eq!(submissions.batches_accepted, 0);
    assert_eq!(submissions.recorded_work_items_accepted, 0);
    assert_eq!(submissions.cross_lane_dependencies, 0);
    assert_eq!(submissions.external_dependencies, 0);
    assert_eq!(submissions.gpu_dependency_routes, 0);
    assert_eq!(submissions.collapsed_dependency_routes, 0);

    let lifecycle = ResourceLifecycleStatistics::default();
    assert_eq!(lifecycle.buffers_created, 0);
    assert_eq!(lifecycle.buffers_reclaimed, 0);
    assert_eq!(lifecycle.textures_created, 0);
    assert_eq!(lifecycle.textures_reclaimed, 0);
    assert_eq!(lifecycle.texture_views_created, 0);
    assert_eq!(lifecycle.texture_views_reclaimed, 0);
    assert_eq!(lifecycle.samplers_created, 0);
    assert_eq!(lifecycle.samplers_reclaimed, 0);
    assert_eq!(lifecycle.shader_modules_created, 0);
    assert_eq!(lifecycle.shader_modules_reclaimed, 0);
    assert_eq!(lifecycle.bind_group_layouts_created, 0);
    assert_eq!(lifecycle.bind_group_layouts_reclaimed, 0);
    assert_eq!(lifecycle.bind_groups_created, 0);
    assert_eq!(lifecycle.bind_groups_reclaimed, 0);
    assert_eq!(lifecycle.pipeline_interfaces_created, 0);
    assert_eq!(lifecycle.pipeline_interfaces_reclaimed, 0);
    assert_eq!(lifecycle.raster_pipelines_created, 0);
    assert_eq!(lifecycle.raster_pipelines_reclaimed, 0);
    assert_eq!(lifecycle.compute_pipelines_created, 0);
    assert_eq!(lifecycle.compute_pipelines_reclaimed, 0);

    let working_set = WorkingSetStatistics::default();
    assert_eq!(working_set.unique_buffers, 0);
    assert_eq!(working_set.unique_textures, 0);
    assert_eq!(working_set.unique_frame_attachments, 0);
    assert_eq!(working_set.unique_samplers, 0);
    assert_eq!(working_set.unique_bind_groups, 0);
    assert_eq!(working_set.unique_shader_modules, 0);
    assert_eq!(working_set.unique_raster_pipelines, 0);
    assert_eq!(working_set.unique_compute_pipelines, 0);
    assert_eq!(working_set.unique_submission_lanes, 0);
}

/// The five groups are five subjects, and a snapshot has to be able to hold one
/// without disturbing the others.
///
/// This is the reason the record is not one flat struct of sixty fields: a caller
/// that displays "commands" should be able to pass [`CommandStatistics`] alone,
/// and a backend that fills one group should not be able to overwrite another by
/// mistake.
#[test]
fn the_cumulative_record_keeps_its_five_subjects_apart() {
    let cumulative = CumulativeStatistics {
        commands: CommandStatistics {
            draw_calls: 7,
            ..CommandStatistics::default()
        },
        bindings: BindingStatistics::default(),
        submissions: SubmissionStatistics::default(),
        presentation: PresentationStatistics::default(),
        resources: ResourceLifecycleStatistics::default(),
    };

    assert_eq!(cumulative.commands.draw_calls, 7);

    // Setting one field of one group left every other group's every field alone.
    assert_eq!(cumulative.commands.dispatch_calls, 0);
    assert_eq!(cumulative.bindings.pipeline_bind_calls, 0);
    assert_eq!(cumulative.submissions.submission_calls, 0);
    assert_eq!(cumulative.presentation.acquires_succeeded, 0);
    assert_eq!(cumulative.resources.buffers_created, 0);

    // And the whole record is `Default`, so a backend starts an epoch with one
    // call rather than five.
    let restarted = CumulativeStatistics::default();
    assert_eq!(restarted.commands.draw_calls, 0);
}

/// Section 47.7 splits every binding family into a *call* counter and a *change*
/// counter, and the split is the interesting part: a redundant bind is a real CPU
/// cost and an effective-state no-op, so a renderer optimising a draw loop needs
/// both numbers to tell which one it has.
///
/// The test writes that example down: the same pipeline bound twice is two calls
/// and one change, and both are readable side by side.
#[test]
fn a_redundant_bind_is_two_calls_and_one_change() {
    let bindings = BindingStatistics {
        pipeline_bind_calls: 2,
        pipeline_changes: 1,
        ..BindingStatistics::default()
    };

    assert_eq!(bindings.pipeline_bind_calls, 2);
    assert_eq!(bindings.pipeline_changes, 1);
    assert_ne!(bindings.pipeline_bind_calls, bindings.pipeline_changes);

    // The same split exists for bind groups and for buffer bindings, which is
    // what makes the rule a rule rather than one counter's quirk.
    assert_eq!(bindings.bind_group_bind_calls, 0);
    assert_eq!(bindings.bind_group_changes, 0);
    assert_eq!(bindings.buffer_bind_calls, 0);
    assert_eq!(bindings.buffer_binding_changes, 0);
}

/// Section 47.8's worked example, and the reason `planned` and `accepted` are
/// separate at every level.
///
/// A plan whose first batch is accepted and whose second fails immediately is
/// *accepted* — some native work was — while two batches were planned and one was
/// accepted. The old model recorded the whole plan as rejected, which is the case
/// this shape exists to make unsayable: there is no field a reader could set that
/// would make the plan both accepted and rejected.
#[test]
fn a_partially_accepted_plan_is_neither_wholly_accepted_nor_rejected() {
    let partial = SubmissionStatistics {
        submission_calls: 1,
        plans_accepted: 1,
        plans_rejected: 0,
        batches_planned: 2,
        batches_accepted: 1,
        ..SubmissionStatistics::default()
    };

    assert_eq!(partial.plans_accepted + partial.plans_rejected, 1);
    assert_eq!(partial.batches_planned, 2);
    assert_eq!(partial.batches_accepted, 1);
    assert!(partial.batches_accepted < partial.batches_planned);

    // And a plan rejected before submission is the other case: no batch entered
    // acceptance, so the plan counts as rejected and its batches are not counted
    // as accepted.
    let wholly_rejected = SubmissionStatistics {
        submission_calls: 1,
        plans_accepted: 0,
        plans_rejected: 1,
        batches_planned: 0,
        batches_accepted: 0,
        ..SubmissionStatistics::default()
    };
    assert_eq!(wholly_rejected.plans_accepted, 0);
    assert_eq!(wholly_rejected.batches_accepted, 0);
}

/// Section 47.10's two accounting domains stay disjoint, and the test states the
/// rule as an assertion a caller can reproduce: an interval that refused to
/// acquire twice increments the acquire counters and *no* `presents_*` field,
/// because no present was planned or accepted.
///
/// The distinction is what stops a statistics reader from concluding that two
/// presents failed. Nothing was presented, and a record that folded a `NotReady`
/// into `presents_failed` would report a present that never existed.
#[test]
fn an_acquire_refusal_is_not_a_present_outcome() {
    let refused = PresentationStatistics {
        acquires_succeeded: 0,
        acquire_not_ready: 2,
        acquire_timeout: 0,
        acquire_outdated: 0,
        acquire_target_lost: 0,
        presents_planned: 0,
        presents_accepted: 0,
        presents_outdated: 0,
        presents_target_lost: 0,
        presents_failed: 0,
        frames_abandoned: 0,
    };

    assert_eq!(refused.acquire_not_ready, 2);
    assert_eq!(refused.presents_planned, 0);
    assert_eq!(refused.presents_accepted, 0);
    assert_eq!(refused.presents_failed, 0);
    assert_eq!(refused.presents_outdated, 0);
    assert_eq!(refused.presents_target_lost, 0);

    // The other domain: a present that failed for an unspecified reason
    // increments exactly one `presents_*` category and no acquire field.
    let failed = PresentationStatistics {
        presents_planned: 1,
        presents_failed: 1,
        ..PresentationStatistics::default()
    };
    assert_eq!(failed.presents_planned, 1);
    assert_eq!(failed.presents_failed, 1);
    assert_eq!(failed.acquire_not_ready, 0);
    assert_eq!(failed.acquire_timeout, 0);
    assert_eq!(failed.acquires_succeeded, 0);

    // `acquires_succeeded` and `presents_planned` are different numbers for a
    // reason: a frame may be acquired and abandoned without ever being presented.
    let abandoned = PresentationStatistics {
        acquires_succeeded: 1,
        frames_abandoned: 1,
        ..PresentationStatistics::default()
    };
    assert_eq!(abandoned.acquires_succeeded, 1);
    assert_eq!(abandoned.frames_abandoned, 1);
    assert_eq!(abandoned.presents_planned, 0);
}

/// A lane row names one logical lane and its two counts, and it is readable
/// without a reader having to know which lane is which by number.
///
/// Section 47.9 says the rows list only lanes actually used and are sorted by
/// lane identity. A row that did nothing is therefore absent rather than zero,
/// which is why the row carries a lane at all: the *presence* of the row is the
/// statement "this lane was used".
#[test]
fn a_lane_row_names_its_lane_and_its_two_counts() {
    let graphics = LaneIntervalStatistics {
        lane: SubmissionLaneId::new(0),
        batches_accepted: 5,
        recorded_work_items: 11,
    };
    let transfer = LaneIntervalStatistics {
        lane: SubmissionLaneId::new(1),
        batches_accepted: 2,
        recorded_work_items: 2,
    };

    assert_ne!(graphics.lane, transfer.lane);
    assert_eq!(graphics.batches_accepted, 5);
    assert_eq!(graphics.recorded_work_items, 11);

    // Section 47.9's own display example, which the fields have to be able to
    // produce without inventing queue switches: "Graphics lane: 5 batches / 11
    // work items".
    assert_eq!(graphics.batches_accepted, 5);
    assert_eq!(graphics.recorded_work_items, 11);

    // A lane with a row is a lane that was used; the counts can still be zero
    // for a batch that carried no work items, and those are different statements.
    let empty_batch = LaneIntervalStatistics {
        lane: SubmissionLaneId::new(2),
        batches_accepted: 1,
        recorded_work_items: 0,
    };
    assert_eq!(empty_batch.batches_accepted, 1);
    assert_eq!(empty_batch.recorded_work_items, 0);
}

/// Section 47.11's distinction, which is the whole reason the reclaim counters
/// are worth having: `reclaimed` is not "a public handle was dropped".
///
/// A caller dropping a handle early and a caller holding one for the life of the
/// frame produce the same drop count and very different reclaim counts. The test
/// pins the shape that makes both readable — a created/reclaimed pair per class,
/// with no field for drops — because a record with a `*_dropped` counter would
/// invite exactly the reading the chapter forbids.
#[test]
fn the_lifecycle_record_pairs_creation_with_reclaim_for_every_class() {
    let busy = ResourceLifecycleStatistics {
        buffers_created: 4,
        textures_created: 2,
        // Nothing has been reclaimed yet: every handle is still held, which is a
        // state the record can express and a drop count could not distinguish
        // from "the handles were dropped but retirement has not run".
        ..ResourceLifecycleStatistics::default()
    };

    assert_eq!(busy.buffers_created, 4);
    assert_eq!(busy.buffers_reclaimed, 0);
    assert_eq!(busy.textures_created, 2);
    assert_eq!(busy.textures_reclaimed, 0);

    // The ten classes each have both counters, so no class can be counted on one
    // side only.
    let retiring = ResourceLifecycleStatistics {
        buffers_created: 4,
        buffers_reclaimed: 4,
        textures_created: 2,
        textures_reclaimed: 1,
        texture_views_created: 2,
        texture_views_reclaimed: 2,
        samplers_created: 1,
        samplers_reclaimed: 0,
        shader_modules_created: 1,
        shader_modules_reclaimed: 0,
        bind_group_layouts_created: 1,
        bind_group_layouts_reclaimed: 0,
        bind_groups_created: 3,
        bind_groups_reclaimed: 1,
        pipeline_interfaces_created: 1,
        pipeline_interfaces_reclaimed: 0,
        raster_pipelines_created: 1,
        raster_pipelines_reclaimed: 0,
        compute_pipelines_created: 1,
        compute_pipelines_reclaimed: 0,
    };

    assert_eq!(retiring.buffers_created, retiring.buffers_reclaimed);
    assert!(retiring.textures_reclaimed < retiring.textures_created);
    assert!(retiring.bind_groups_reclaimed < retiring.bind_groups_created);
}

/// Section 47.12 counts logical identities and never native handles, and a frame
/// is an identity for this purpose.
///
/// The test is small on purpose: the only property a default record can exhibit
/// is that the *field* exists. That is the point of asserting it — section 47.12
/// lists frames among the things a working set counts, so a caller displaying
/// "this pass touched 3 frames" has somewhere to read that from, and the counter
/// is a count of `AcquiredFrameId`s rather than of native swapchain images.
#[test]
fn the_working_set_counts_logical_identities_including_frames() {
    let working_set = WorkingSetStatistics {
        unique_buffers: 3,
        unique_textures: 2,
        unique_frame_attachments: 1,
        ..WorkingSetStatistics::default()
    };

    assert_eq!(working_set.unique_buffers, 3);
    assert_eq!(working_set.unique_textures, 2);
    assert_eq!(working_set.unique_frame_attachments, 1);

    // The six other classes exist and are untouched by the three above, which is
    // what "the working set is per class" means.
    assert_eq!(working_set.unique_samplers, 0);
    assert_eq!(working_set.unique_bind_groups, 0);
    assert_eq!(working_set.unique_shader_modules, 0);
    assert_eq!(working_set.unique_raster_pipelines, 0);
    assert_eq!(working_set.unique_compute_pipelines, 0);
    assert_eq!(working_set.unique_submission_lanes, 0);
}
