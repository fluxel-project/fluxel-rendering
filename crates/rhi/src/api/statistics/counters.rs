//! The cumulative and per-interval counter records (specification 47.5–47.8 and
//! 47.10–47.12).
//!
//! Every type here is a plain record of `u64` counters with public fields, and
//! that is deliberate rather than lazy: the metric *definitions* are frozen
//! across backends, so a caller must be able to read a named field, and a
//! backend must be unable to invent a counter of its own. An opaque map would
//! let two backends both report "draw calls" while meaning different things.
//!
//! Three rules apply to every counter in this file, and they come from section
//! 47 rather than from this module:
//!
//! - **Saturating, never wrapping.** Section 47.5 forbids wraparound: a counter
//!   that reached `u64::MAX` stays there. Silent wraparound would turn a
//!   long-running system's numbers into plausible-looking garbage.
//! - **Semantic, not native.** A command count is a count of Fluxel commands,
//!   and section 47.6 is explicit that it is not a `VkCmd` count, a D3D12
//!   command count, a Metal encoder-call count, or a WebGPU internal command
//!   count. The numbers are comparable across backends precisely because none of
//!   them is a driver's number.
//! - **Effective, not called.** A *change* counter counts a change in effective
//!   state, not an API call — except where the name says `_calls`, which counts
//!   calls. Section 47.21 freezes both definitions, and the distinction is why
//!   binding a pipeline twice in a row increments `pipeline_bind_calls` by two
//!   and `pipeline_changes` by one.
//!
//! Section 47.7 adds the scope rule that makes the change counters well-defined:
//! command state such as pipeline and bind group is considered unbound at the
//! start of every raster or compute scope, so setting the same pipeline in two
//! scopes is two changes.

use crate::api::resource::transient::TransientMemoryStatistics;
use crate::api::submission::SubmissionLaneId;

/// Everything a device has counted since the current collection epoch began.
///
/// The five groups are the five things section 47 measures, and they are kept
/// separate rather than flattened into one struct of sixty fields so that a
/// caller can hold, pass, or display the group it cares about.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct CumulativeStatistics {
    /// Command and scope counts.
    pub commands: CommandStatistics,
    /// Bind and effective-state-change counts.
    pub bindings: BindingStatistics,
    /// Submission structure counts.
    pub submissions: SubmissionStatistics,
    /// Presentation lifecycle counts.
    pub presentation: PresentationStatistics,
    /// Logical object lifecycle counts.
    pub resources: ResourceLifecycleStatistics,
    /// Transient logical/backing/aliasing observations.
    pub transient: TransientMemoryStatistics,
}

/// Commands and scopes the RHI was asked to record (specification 47.6).
///
/// Scope counts and command counts are both here because they answer different
/// questions: `raster_scopes` says how the work was organised, `draw_calls` says
/// how much of it there was. Section 47.8 removed the corresponding
/// `logical_lane_changes` counter for a different reason — lanes have no
/// portable global order — and no equivalent was put in its place.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct CommandStatistics {
    /// Recorders that reached `finish()` and produced recorded work.
    pub recorders_finished: u64,

    /// Raster scopes begun.
    pub raster_scopes: u64,
    /// Compute scopes begun.
    pub compute_scopes: u64,

    /// Non-indexed draw commands recorded.
    pub draw_calls: u64,
    /// Indexed draw commands recorded.
    pub draw_indexed_calls: u64,
    /// Dispatch commands recorded.
    pub dispatch_calls: u64,

    /// Buffer-to-buffer copies.
    pub buffer_copies: u64,
    /// Buffer-to-texture copies.
    pub buffer_to_texture_copies: u64,
    /// Texture-to-buffer copies.
    pub texture_to_buffer_copies: u64,
    /// Texture-to-texture copies.
    pub texture_copies: u64,

    /// Multisample resolves.
    pub resolves: u64,
    /// Blits, which section 9 makes a distinct route from a copy because it
    /// filters and rescales.
    pub blits: u64,

    /// Upload commands recorded.
    pub upload_commands: u64,
    /// Readback commands recorded.
    pub readback_commands: u64,

    /// Debug markers pushed.
    pub debug_markers: u64,
}

/// Binding calls and effective binding-state changes (specification 47.7).
///
/// The chapter splits each binding family into a *call* counter and a *change*
/// counter, and the split is the interesting part: a redundant bind is a real
/// CPU cost and an effective-state no-op, and a caller optimising a renderer
/// needs to see both numbers to tell which one it has.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct BindingStatistics {
    /// Pipeline bind calls issued.
    pub pipeline_bind_calls: u64,

    /// Effective `Pipeline` [`crate::api::identity::ObjectId`] changes within a
    /// scope.
    ///
    /// Scope-local: state does not carry across a scope boundary, so the first
    /// bind in a scope is always a change (section 47.7).
    pub pipeline_changes: u64,

    /// Effective executable-shader-set changes.
    ///
    /// The set is `(vertex ShaderModule, optional fragment ShaderModule)` for a
    /// raster pipeline and `(compute ShaderModule)` for a compute pipeline.
    /// Section 47.7 requires the comparison to be over those identities and
    /// forbids using a hash or fingerprint of them to decide it: a fingerprint
    /// is a cache key, and section 21.1 states that equal fingerprints cannot
    /// alone replace correctness validation.
    pub shader_set_changes: u64,

    /// Bind-group bind calls issued.
    pub bind_group_bind_calls: u64,

    /// Effective binding-tuple changes: `(BindGroup ObjectId, dynamic_offsets[])`.
    ///
    /// The dynamic offsets are part of the tuple because changing an offset
    /// changes which bytes the binding reads while leaving the bind group
    /// identity alone.
    pub bind_group_changes: u64,

    /// `set_vertex_buffer` and `set_index_buffer` calls.
    pub buffer_bind_calls: u64,

    /// Effective buffer-binding element changes.
    ///
    /// Covers vertex and index bindings, buffer bindings inside a bind group,
    /// and effective range changes caused by dynamic offsets.
    pub buffer_binding_changes: u64,

    /// Effective texture-element changes caused by bind-group state changes.
    ///
    /// Section 47.7's boundary: a pipeline change that makes a shader start or
    /// stop *reading* an already-bound texture does not increment this, because
    /// no bind state changed. Actual use is [`super::WorkingSetStatistics`]'s
    /// subject, not this counter's.
    pub texture_binding_changes: u64,

    /// Effective sampler-element changes caused by bind-group state changes.
    ///
    /// A sampler generates no memory hazard and is still counted, because it is
    /// part of the command's semantics and of what a statistics reader is trying
    /// to explain (section 47.2).
    pub sampler_binding_changes: u64,

    /// Adjacent raster-scope render-target-set changes within one recorder.
    ///
    /// Defined over *one recorder's* raster-scope sequence and not over the
    /// device: several recorders may record in parallel and later execute on
    /// different lanes, so no reliable system-wide "previous render target"
    /// exists. The first set, going from "no target" to a target, counts as one
    /// change.
    pub render_target_set_changes: u64,
}

/// The structure of what was submitted (specification 47.8).
///
/// The counts distinguish *planned* from *accepted* at every level, because a
/// submit that partially succeeds is the case the old model got wrong: section
/// 47.8's worked example is a plan whose first batch is accepted and whose
/// second fails immediately, and the rule is that the plan counts as accepted,
/// `batches_planned` counts both batches, and `batches_accepted` counts one. It
/// is not correct to record the whole plan as rejected.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct SubmissionStatistics {
    /// `Device::submit()` calls.
    pub submission_calls: u64,

    /// Plans for which at least some native work was accepted.
    pub plans_accepted: u64,

    /// Plans fully rejected before submission, with no native work accepted.
    pub plans_rejected: u64,

    /// Logical batches declared in accepted plans.
    pub batches_planned: u64,

    /// Logical batches that entered backend acceptance successfully.
    pub batches_accepted: u64,

    /// Recorded work items inside accepted batches.
    pub recorded_work_items_accepted: u64,

    /// Explicit cross-lane dependencies declared within the plan.
    ///
    /// Counted where the dependency is declared and the lanes differ, which is
    /// section 47.21's definition. It is a count of declared relations, not of
    /// the lowering routes chosen for them — those are the next two counters.
    pub cross_lane_dependencies: u64,

    /// Prior `CompletionPoint` to current `PlanPoint` dependencies in accepted
    /// plans.
    pub external_dependencies: u64,

    /// Final lowerings that chose a GPU-side dependency.
    pub gpu_dependency_routes: u64,

    /// Final lowerings that required collapsing two logical lanes into one
    /// ordered execution domain.
    ///
    /// The counter exists because a collapse is the expensive answer to a
    /// dependency the target cannot express as a GPU-side wait, and a caller
    /// that sees it climb is being told what its lane assignment is costing.
    pub collapsed_dependency_routes: u64,
}

/// What one logical lane did during one interval (specification 47.9).
///
/// Section 47.21 defines this as accepted batch and work-item counts per logical
/// lane, and section 47.8 removed `logical_lane_changes` because there is no
/// portable definition of "how many times adjacent execution batches switched
/// from lane A to lane B". This record replaces it with something that *is*
/// definable, and an upper layer can total the rows to display "graphics lane: 5
/// batches / 11 work items" without inventing queue switches.
///
/// The rows in an [`super::snapshot::IntervalStatistics`] are canonically sorted
/// by [`Self::lane`] and list only lanes that were actually used during the
/// interval, so an absent lane is a lane that did nothing rather than a lane
/// that reported zero.
///
/// # `Default` is deliberately absent
///
/// Section 47.9 writes `#[derive(Clone, Debug, Default)]` here, and `Default`
/// cannot be honoured: it would have to produce a [`SubmissionLaneId`], and that
/// type has no `Default` because a lane identity exists only because a device
/// enumerated it. A defaulted row would name lane zero and assert a lane that
/// may not exist, which contradicts both that type's own invariant and this
/// record's rule that it lists only lanes actually used. The derivable half is
/// kept and the unsatisfiable derive is dropped, as a reported defect rather
/// than a silent repair.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct LaneIntervalStatistics {
    /// The logical lane this row describes.
    pub lane: SubmissionLaneId,
    /// Logical batches accepted on this lane during the interval.
    pub batches_accepted: u64,
    /// Recorded work items accepted on this lane during the interval.
    pub recorded_work_items: u64,
}

/// The presentation lifecycle (specification 47.10).
///
/// Two disjoint accounting domains live in this one record, and section 47.10 is
/// emphatic that they stay disjoint:
///
/// ```text
/// acquire refusal            -> the acquire_* fields, and NOTHING else
/// submitted present outcome  -> exactly one presents_* field
/// ```
///
/// `NotReady`, `Timeout`, `FrameOutstanding`, zero-size suspension, and an
/// acquire-time out-of-memory increment no `presents_*` field, because no present
/// was planned or accepted. Acquire `TargetOutdated` and `TargetLost` increment
/// only their own acquire fields. Device loss is reported by the device and loss
/// statistics and must not be relabelled as a present outcome. A terminal
/// present state increments exactly one applicable `presents_*` category.
///
/// Not counted, and deliberately: scan-out count, displayed frame count, and
/// vsync count. Those are presentation *timing*, which section 47.1 leaves to a
/// future extension rather than approximating here with a number the RHI cannot
/// observe.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct PresentationStatistics {
    /// Frames acquired successfully.
    pub acquires_succeeded: u64,

    /// Acquire refusals because a frame was not ready.
    pub acquire_not_ready: u64,
    /// Acquire refusals because the target's timeout elapsed.
    pub acquire_timeout: u64,
    /// Acquire refusals because the target's configuration no longer matches it.
    pub acquire_outdated: u64,
    /// Acquire refusals because the target was lost.
    pub acquire_target_lost: u64,

    /// Frames consumed by `present_after()`.
    pub presents_planned: u64,

    /// Presents the target accepted.
    pub presents_accepted: u64,
    /// Presents that failed because the target's configuration no longer
    /// matches it.
    pub presents_outdated: u64,
    /// Presents that failed because the target was lost.
    pub presents_target_lost: u64,
    /// Presents that failed for any other reason.
    pub presents_failed: u64,

    /// Frames abandoned, counting both an explicit abandon and the safety
    /// abandonment a dropped frame performs.
    pub frames_abandoned: u64,
}

/// The logical object lifecycle (specification 47.11).
///
/// Creation and reclaim are counted per object class, and section 47.11 freezes
/// what the second word means:
///
/// ```text
/// reclaimed != a public handle was dropped
/// reclaimed  = the backing satisfied completion-safe reclaim conditions
/// ```
///
/// The distinction is the whole reason the counter is worth having. A caller
/// dropping a handle early and a caller holding one for the life of the frame
/// produce the same drop count and very different reclaim counts, and only the
/// reclaim count says whether retirement is keeping up.
///
/// Spec 02 §18.8 is what requires these counts to exist at all: they are part of
/// the resource layer's promise that create and reclaim are observable.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct ResourceLifecycleStatistics {
    /// Buffers created.
    pub buffers_created: u64,
    /// Textures created.
    pub textures_created: u64,
    /// Texture views created.
    pub texture_views_created: u64,
    /// Samplers created.
    pub samplers_created: u64,

    /// Shader modules created.
    pub shader_modules_created: u64,
    /// Bind group layouts created.
    pub bind_group_layouts_created: u64,
    /// Bind groups created.
    pub bind_groups_created: u64,
    /// Pipeline interfaces created.
    pub pipeline_interfaces_created: u64,
    /// Raster pipelines created.
    pub raster_pipelines_created: u64,
    /// Compute pipelines created.
    pub compute_pipelines_created: u64,

    /// Buffers reclaimed.
    pub buffers_reclaimed: u64,
    /// Textures reclaimed.
    pub textures_reclaimed: u64,
    /// Texture views reclaimed.
    pub texture_views_reclaimed: u64,
    /// Samplers reclaimed.
    pub samplers_reclaimed: u64,

    /// Shader modules reclaimed.
    pub shader_modules_reclaimed: u64,
    /// Bind group layouts reclaimed.
    pub bind_group_layouts_reclaimed: u64,
    /// Bind groups reclaimed.
    pub bind_groups_reclaimed: u64,
    /// Pipeline interfaces reclaimed.
    pub pipeline_interfaces_reclaimed: u64,
    /// Raster pipelines reclaimed.
    pub raster_pipelines_reclaimed: u64,
    /// Compute pipelines reclaimed.
    pub compute_pipelines_reclaimed: u64,
}

/// The unique objects an interval actually used (specification 47.12).
///
/// Collected only at [`crate::api::statistics::StatisticsDetail::Detailed`],
/// because it needs the recorder to keep a set rather than a counter. This is
/// the counter group that answers "what did this frame actually touch", which is
/// the question a pipeline-change count cannot answer: a change count says how
/// often effective state moved, and this says how much distinct state there was
/// to move between.
///
/// Every value is a count of logical identities — `ObjectId`, `AcquiredFrameId`,
/// `SubmissionLaneId` — and never of native handles. A renderer that reads the
/// same texture through four views still reports one texture.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct WorkingSetStatistics {
    /// Distinct buffers used.
    pub unique_buffers: u64,
    /// Distinct textures used.
    pub unique_textures: u64,
    /// Distinct acquired frames used.
    pub unique_frame_attachments: u64,

    /// Distinct samplers used.
    pub unique_samplers: u64,
    /// Distinct bind groups used.
    pub unique_bind_groups: u64,

    /// Distinct shader modules used.
    pub unique_shader_modules: u64,
    /// Distinct raster pipelines used.
    pub unique_raster_pipelines: u64,
    /// Distinct compute pipelines used.
    pub unique_compute_pipelines: u64,

    /// Distinct submission lanes used.
    pub unique_submission_lanes: u64,
}
