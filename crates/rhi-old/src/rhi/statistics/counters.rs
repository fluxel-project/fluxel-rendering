//! Cumulative counters, the recorder-local accumulator, and their deltas
//! (§47.5 - §47.8).
//!
//! Every counter here is a Fluxel semantic count of a portable event. None of
//! them promises a native count: `buffer_copies` is not a backend's command
//! count, and no field exists for a native barrier, descriptor, encoder, queue
//! switch, or timestamp (§47.1).

use super::config::StatisticsDetail;

/// Adds `delta` to `counter`, saturating at [`u64::MAX`] (§47.5).
///
/// Every statistics counter is a `u64` and every sink addition goes through
/// this function: an exceptionally long-running system must report `u64::MAX`
/// rather than wrap to a small value that reads like a fresh counter.
#[inline]
pub(crate) fn saturating_add(counter: &mut u64, delta: u64) {
    *counter = counter.saturating_add(delta);
}

/// Counts one observation of `counter`, saturating.
#[inline]
pub(crate) fn count_one(counter: &mut u64) {
    saturating_add(counter, 1);
}

/// Field-wise saturating addition for one counter group.
macro_rules! add_fields {
    ($target:expr, $source:expr, $($field:ident),+ $(,)?) => {
        $(
            saturating_add(&mut $target.$field, $source.$field);
        )+
    };
}

/// Field-wise saturating subtraction for one counter group.
///
/// Every field of `$target` becomes `target - previous`, so `$target` must start
/// as the current value: `delta_since` clones `self` into it and subtracts
/// `previous`. An interval is the difference of two cumulative values taken in
/// the same collection epoch, and a counter is monotone within an epoch, so the
/// subtraction cannot underflow. `saturating_sub` states that invariant
/// defensively instead of relying on a debug assertion.
macro_rules! delta_fields {
    ($target:expr, $previous:expr, $($field:ident),+ $(,)?) => {
        $(
            $target.$field = $target.$field.saturating_sub($previous.$field);
        )+
    };
}

/// One Fluxel semantic command an actual-use observation can report.
///
/// The variants map one-to-one onto [`CommandStatistics`] fields, so a backend
/// never has to decide how one native command "maps" onto several counters.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CommandKind {
    /// A non-indexed draw.
    Draw,
    /// An indexed draw.
    DrawIndexed,
    /// A compute dispatch.
    Dispatch,
    /// A buffer-to-buffer copy.
    BufferCopy,
    /// A buffer-to-texture copy.
    BufferToTextureCopy,
    /// A texture-to-buffer copy.
    TextureToBufferCopy,
    /// A texture-to-texture copy.
    TextureCopy,
    /// A multisample resolve.
    Resolve,
    /// A blit.
    Blit,
    /// A host-to-device upload command that is not a plain copy route.
    Upload,
    /// A device-to-host readback command that is not a plain copy route.
    Readback,
    /// A debug marker inserted into a command stream.
    DebugMarker,
}

/// The kind of scope a recorder opened.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ScopeKind {
    /// A raster scope.
    Raster,
    /// A compute scope.
    Compute,
}

/// Command and scope counters (§47.6).
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct CommandStatistics {
    /// Recorders that reached `finish()`.
    ///
    /// Low-frequency lifecycle data, so it is collected at every detail level.
    pub recorders_finished: u64,

    /// Raster scopes opened.
    pub raster_scopes: u64,
    /// Compute scopes opened.
    pub compute_scopes: u64,

    /// Non-indexed draw calls recorded.
    pub draw_calls: u64,
    /// Indexed draw calls recorded.
    pub draw_indexed_calls: u64,
    /// Compute dispatches recorded.
    pub dispatch_calls: u64,

    /// Buffer-to-buffer copies recorded.
    pub buffer_copies: u64,
    /// Buffer-to-texture copies recorded.
    pub buffer_to_texture_copies: u64,
    /// Texture-to-buffer copies recorded.
    pub texture_to_buffer_copies: u64,
    /// Texture-to-texture copies recorded.
    pub texture_copies: u64,

    /// Multisample resolves recorded.
    pub resolves: u64,
    /// Blits recorded.
    pub blits: u64,

    /// Host-to-device upload commands recorded.
    pub upload_commands: u64,
    /// Device-to-host readback commands recorded.
    pub readback_commands: u64,

    /// Debug markers recorded.
    pub debug_markers: u64,
}

impl CommandStatistics {
    /// Adds every counter `detail` collects.
    pub(crate) fn saturating_add_gated(&mut self, other: &Self, detail: StatisticsDetail) {
        saturating_add(&mut self.recorders_finished, other.recorders_finished);
        if !detail.collects_commands() {
            return;
        }
        add_fields!(
            self,
            other,
            raster_scopes,
            compute_scopes,
            draw_calls,
            draw_indexed_calls,
            dispatch_calls,
            buffer_copies,
            buffer_to_texture_copies,
            texture_to_buffer_copies,
            texture_copies,
            resolves,
            blits,
            upload_commands,
            readback_commands,
            debug_markers,
        );
    }

    /// The per-field difference from `previous`.
    pub(crate) fn delta_since(&self, previous: &Self) -> Self {
        let mut delta = self.clone();
        delta_fields!(
            delta,
            previous,
            recorders_finished,
            raster_scopes,
            compute_scopes,
            draw_calls,
            draw_indexed_calls,
            dispatch_calls,
            buffer_copies,
            buffer_to_texture_copies,
            texture_to_buffer_copies,
            texture_copies,
            resolves,
            blits,
            upload_commands,
            readback_commands,
            debug_markers,
        );
        delta
    }
}

/// Binding and effective-state-change counters (§47.7).
///
/// The `*_calls` counters count API calls; the change counters count *effective*
/// state changes. They are different questions: binding the same pipeline twice
/// in one scope is two calls and one change.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct BindingStatistics {
    /// Pipeline bind calls.
    pub pipeline_bind_calls: u64,

    /// Effective Pipeline [`ObjectId`] changes within a scope.
    ///
    /// Command state starts unbound in every scope, so setting the same
    /// pipeline in two scopes is two changes.
    ///
    /// [`ObjectId`]: crate::rhi::platform::ObjectId
    pub pipeline_changes: u64,

    /// Effective executable shader set changes.
    ///
    /// Raster is `(vertex ShaderModule ObjectId, optional fragment ObjectId)`
    /// and compute is the compute `ShaderModule ObjectId`. No hash or
    /// fingerprint participates: this is an identity change, not a content
    /// comparison.
    pub shader_set_changes: u64,

    /// Bind group bind calls.
    pub bind_group_bind_calls: u64,

    /// Effective binding tuple changes: `(BindGroup ObjectId, dynamic_offsets)`.
    pub bind_group_changes: u64,

    /// `set_vertex_buffer` and `set_index_buffer` calls.
    pub buffer_bind_calls: u64,

    /// Effective buffer binding element changes.
    ///
    /// Counts vertex/index binding, bind-group buffer binding, and the effective
    /// range changes a dynamic offset produces.
    pub buffer_binding_changes: u64,

    /// Effective texture element changes caused by bind-group state changes.
    pub texture_binding_changes: u64,

    /// Effective sampler element changes caused by bind-group state changes.
    pub sampler_binding_changes: u64,

    /// Adjacent raster scopes of one recorder that name a different attachment
    /// set.
    ///
    /// The transition from "no target" to a target counts as one. The sequence
    /// is per recorder on purpose: parallel recorders that later execute on
    /// different lanes have no portable system-wide "previous target".
    pub render_target_set_changes: u64,
}

impl BindingStatistics {
    /// Adds every counter `detail` collects.
    pub(crate) fn saturating_add_gated(&mut self, other: &Self, detail: StatisticsDetail) {
        if detail.collects_bind_calls() {
            add_fields!(
                self,
                other,
                pipeline_bind_calls,
                bind_group_bind_calls,
                buffer_bind_calls,
            );
        }
        if detail.collects_state_changes() {
            add_fields!(
                self,
                other,
                pipeline_changes,
                shader_set_changes,
                bind_group_changes,
                buffer_binding_changes,
                texture_binding_changes,
                sampler_binding_changes,
                render_target_set_changes,
            );
        }
    }

    /// The per-field difference from `previous`.
    pub(crate) fn delta_since(&self, previous: &Self) -> Self {
        let mut delta = self.clone();
        delta_fields!(
            delta,
            previous,
            pipeline_bind_calls,
            pipeline_changes,
            shader_set_changes,
            bind_group_bind_calls,
            bind_group_changes,
            buffer_bind_calls,
            buffer_binding_changes,
            texture_binding_changes,
            sampler_binding_changes,
            render_target_set_changes,
        );
        delta
    }
}

/// The per-lane part of one submission event.
///
/// There is no `Default`: a lane with no work to report is not part of an event,
/// and a defaulted `LaneWork` would carry a lane the submission never used.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaneWork {
    /// The lane the work was accepted on.
    pub(crate) lane: crate::rhi::format::SubmissionLaneId,
    /// Batches this plan accepted on that lane.
    pub(crate) batches_accepted: u64,
    /// RecordedWork items this plan accepted on that lane.
    pub(crate) recorded_work_items: u64,
}

/// One `Device::submit()` observation (§47.8).
///
/// A submission is the one place where several counters move together, so it is
/// reported as one event rather than as independent increments: a snapshot must
/// never see a plan's accepted batches without its accepted work items.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub(crate) struct SubmissionEvent {
    /// Plans for which at least some native work was accepted.
    pub(crate) plans_accepted: u64,
    /// Plans fully rejected before submission, with no native work accepted.
    pub(crate) plans_rejected: u64,
    /// Logical batches declared in accepted plans.
    pub(crate) batches_planned: u64,
    /// Logical batches that entered backend acceptance successfully.
    pub(crate) batches_accepted: u64,
    /// RecordedWork items in accepted batches.
    pub(crate) recorded_work_items: u64,
    /// Explicit cross-lane dependencies within these plans.
    pub(crate) cross_lane_dependencies: u64,
    /// Prior CompletionPoint to current PlanPoint dependencies in accepted
    /// plans.
    pub(crate) external_dependencies: u64,
    /// Final lowerings that chose a GPU-side dependency.
    pub(crate) gpu_dependency_routes: u64,
    /// Final lowerings that required lane collapse.
    pub(crate) collapsed_dependency_routes: u64,
    /// Accepted batch and work-item counts per lane this plan used.
    pub(crate) lanes: Vec<LaneWork>,
}

/// Submission statistics (§47.8).
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct SubmissionStatistics {
    /// `Device::submit()` calls.
    pub submission_calls: u64,

    /// Number of plans for which at least some native work was accepted.
    pub plans_accepted: u64,

    /// Number of plans fully rejected before submission, with no native work
    /// accepted.
    pub plans_rejected: u64,

    /// Total logical batches declared in accepted plans.
    pub batches_planned: u64,

    /// Number of logical batches that actually entered backend acceptance.
    ///
    /// A plan that accepts batch A and fails on batch B contributes one accepted
    /// batch here and is still counted as accepted: the whole plan is never
    /// relabeled as rejected.
    pub batches_accepted: u64,

    /// Number of RecordedWork items in accepted batches.
    pub recorded_work_items_accepted: u64,

    /// Number of explicit cross-lane dependencies within accepted plans.
    pub cross_lane_dependencies: u64,

    /// Number of prior CompletionPoint to current PlanPoint dependencies in
    /// accepted plans.
    pub external_dependencies: u64,

    /// Number of final lowerings that chose GPU-side dependencies.
    pub gpu_dependency_routes: u64,

    /// Number of final lowerings that required lane collapse.
    pub collapsed_dependency_routes: u64,
}

impl SubmissionStatistics {
    /// Folds one submit call and its plan accounting into this group.
    pub(crate) fn saturating_add_event(&mut self, event: &SubmissionEvent) {
        count_one(&mut self.submission_calls);
        // The event names the count as the plan reports it and the cumulative
        // counter names it as the specification freezes it, so this one field
        // pair is spelled out rather than macro-generated.
        saturating_add(
            &mut self.recorded_work_items_accepted,
            event.recorded_work_items,
        );
        add_fields!(
            self,
            event,
            plans_accepted,
            plans_rejected,
            batches_planned,
            batches_accepted,
            cross_lane_dependencies,
            external_dependencies,
            gpu_dependency_routes,
            collapsed_dependency_routes,
        );
    }

    /// The per-field difference from `previous`.
    pub(crate) fn delta_since(&self, previous: &Self) -> Self {
        let mut delta = self.clone();
        delta_fields!(
            delta,
            previous,
            submission_calls,
            plans_accepted,
            plans_rejected,
            batches_planned,
            batches_accepted,
            recorded_work_items_accepted,
            cross_lane_dependencies,
            external_dependencies,
            gpu_dependency_routes,
            collapsed_dependency_routes,
        );
        delta
    }
}

/// Presentation statistics (§47.10).
///
/// Acquire refusal and submitted-present outcome are disjoint domains: an
/// acquire that produced no frame increments no `presents_*` field, because no
/// present was planned or accepted. Device loss is reported by device state and
/// is never relabeled as a present outcome.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct PresentationStatistics {
    /// Acquires that produced a frame.
    pub acquires_succeeded: u64,

    /// Acquires refused because no frame was available yet.
    pub acquire_not_ready: u64,
    /// Acquires refused because the wait budget elapsed.
    pub acquire_timeout: u64,
    /// Acquires refused because the target needs reconfiguration.
    pub acquire_outdated: u64,
    /// Acquires refused because the target is lost.
    pub acquire_target_lost: u64,

    /// Frames consumed by `present_after()` planning.
    pub presents_planned: u64,

    /// Presents the presentation system or host lifecycle accepted.
    ///
    /// This never means the frame reached scan-out.
    pub presents_accepted: u64,
    /// Presents that failed because the target needs reconfiguration.
    pub presents_outdated: u64,
    /// Presents that failed because the target is lost.
    pub presents_target_lost: u64,
    /// Presents the backend reported as a terminal failure.
    pub presents_failed: u64,

    /// Explicit abandon plus Drop-safety abandonment of a consumed frame.
    pub frames_abandoned: u64,
}

impl PresentationStatistics {
    /// The per-field difference from `previous`.
    pub(crate) fn delta_since(&self, previous: &Self) -> Self {
        let mut delta = self.clone();
        delta_fields!(
            delta,
            previous,
            acquires_succeeded,
            acquire_not_ready,
            acquire_timeout,
            acquire_outdated,
            acquire_target_lost,
            presents_planned,
            presents_accepted,
            presents_outdated,
            presents_target_lost,
            presents_failed,
            frames_abandoned,
        );
        delta
    }
}

/// Logical object lifecycle statistics (§47.11).
///
/// `reclaimed` is not a public-handle drop. It counts objects whose backing
/// satisfied completion-safe reclaim conditions, which is what makes it usable
/// as a leak signal: creating N objects and reclaiming N means the backings are
/// gone, whatever the handles did.
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

    /// Buffers reclaimed after completion-safe conditions were met.
    pub buffers_reclaimed: u64,
    /// Textures reclaimed after completion-safe conditions were met.
    pub textures_reclaimed: u64,
    /// Texture views reclaimed after completion-safe conditions were met.
    pub texture_views_reclaimed: u64,
    /// Samplers reclaimed after completion-safe conditions were met.
    pub samplers_reclaimed: u64,

    /// Shader modules reclaimed after completion-safe conditions were met.
    pub shader_modules_reclaimed: u64,
    /// Bind group layouts reclaimed after completion-safe conditions were met.
    pub bind_group_layouts_reclaimed: u64,
    /// Bind groups reclaimed after completion-safe conditions were met.
    pub bind_groups_reclaimed: u64,
    /// Pipeline interfaces reclaimed after completion-safe conditions were met.
    pub pipeline_interfaces_reclaimed: u64,
    /// Raster pipelines reclaimed after completion-safe conditions were met.
    pub raster_pipelines_reclaimed: u64,
    /// Compute pipelines reclaimed after completion-safe conditions were met.
    pub compute_pipelines_reclaimed: u64,
}

impl ResourceLifecycleStatistics {
    /// The per-field difference from `previous`.
    pub(crate) fn delta_since(&self, previous: &Self) -> Self {
        let mut delta = self.clone();
        delta_fields!(
            delta,
            previous,
            buffers_created,
            textures_created,
            texture_views_created,
            samplers_created,
            shader_modules_created,
            bind_group_layouts_created,
            bind_groups_created,
            pipeline_interfaces_created,
            raster_pipelines_created,
            compute_pipelines_created,
            buffers_reclaimed,
            textures_reclaimed,
            texture_views_reclaimed,
            samplers_reclaimed,
            shader_modules_reclaimed,
            bind_group_layouts_reclaimed,
            bind_groups_reclaimed,
            pipeline_interfaces_reclaimed,
            raster_pipelines_reclaimed,
            compute_pipelines_reclaimed,
        );
        delta
    }
}

/// Every cumulative counter of one statistics domain (§47.5).
///
/// The groups are separated by the question they answer, not by where the data
/// comes from, so a renderer, a graph, and a debug HUD can each read the one
/// group they need.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct CumulativeStatistics {
    /// Command and scope counts.
    pub commands: CommandStatistics,
    /// Bind-call and effective-state-change counts.
    pub bindings: BindingStatistics,
    /// Submission structure counts.
    pub submissions: SubmissionStatistics,
    /// Presentation lifecycle counts.
    pub presentation: PresentationStatistics,
    /// Logical object lifecycle counts.
    pub resources: ResourceLifecycleStatistics,
}

impl CumulativeStatistics {
    /// The per-group difference from `previous`.
    pub(crate) fn delta_since(&self, previous: &Self) -> Self {
        Self {
            commands: self.commands.delta_since(&previous.commands),
            bindings: self.bindings.delta_since(&previous.bindings),
            submissions: self.submissions.delta_since(&previous.submissions),
            presentation: self.presentation.delta_since(&previous.presentation),
            resources: self.resources.delta_since(&previous.resources),
        }
    }
}

/// Recorder-local command and binding counters (§47.4).
///
/// One recorder owns one of these and mutates it with no lock at all: recording
/// a draw must never serialize two parallel recorders (§47.19), and the
/// recommended merge point is `finish()`. [`DeviceStatistics::merge_recorder`]
/// is the only shared write.
///
/// A counter the recorder's detail level does not collect is not merely ignored
/// at merge time: `note_*` returns before touching it, so a `Minimal` recorder
/// pays nothing for the command path.
///
/// [`DeviceStatistics::merge_recorder`]: super::DeviceStatistics::merge_recorder
#[derive(Clone, Debug)]
pub(crate) struct RecorderCounters {
    detail: StatisticsDetail,
    commands: CommandStatistics,
    bindings: BindingStatistics,
}

impl RecorderCounters {
    /// A local accumulator for a recorder collecting `detail`.
    pub(crate) fn new(detail: StatisticsDetail) -> Self {
        Self {
            detail,
            commands: CommandStatistics::default(),
            bindings: BindingStatistics::default(),
        }
    }

    /// The detail level this accumulator collects.
    pub(crate) fn detail(&self) -> StatisticsDetail {
        self.detail
    }

    /// The command counters collected so far.
    pub(crate) fn commands(&self) -> &CommandStatistics {
        &self.commands
    }

    /// The binding counters collected so far.
    pub(crate) fn bindings(&self) -> &BindingStatistics {
        &self.bindings
    }

    /// Records one opened scope.
    pub(crate) fn note_scope(&mut self, scope: ScopeKind) {
        if !self.detail.collects_commands() {
            return;
        }
        match scope {
            ScopeKind::Raster => count_one(&mut self.commands.raster_scopes),
            ScopeKind::Compute => count_one(&mut self.commands.compute_scopes),
        }
    }

    /// Records one Fluxel semantic command.
    pub(crate) fn note_command(&mut self, command: CommandKind) {
        if !self.detail.collects_commands() {
            return;
        }
        let counter = match command {
            CommandKind::Draw => &mut self.commands.draw_calls,
            CommandKind::DrawIndexed => &mut self.commands.draw_indexed_calls,
            CommandKind::Dispatch => &mut self.commands.dispatch_calls,
            CommandKind::BufferCopy => &mut self.commands.buffer_copies,
            CommandKind::BufferToTextureCopy => &mut self.commands.buffer_to_texture_copies,
            CommandKind::TextureToBufferCopy => &mut self.commands.texture_to_buffer_copies,
            CommandKind::TextureCopy => &mut self.commands.texture_copies,
            CommandKind::Resolve => &mut self.commands.resolves,
            CommandKind::Blit => &mut self.commands.blits,
            CommandKind::Upload => &mut self.commands.upload_commands,
            CommandKind::Readback => &mut self.commands.readback_commands,
            CommandKind::DebugMarker => &mut self.commands.debug_markers,
        };
        count_one(counter);
    }

    /// Records one pipeline bind call.
    pub(crate) fn note_pipeline_bind(&mut self) {
        if self.detail.collects_bind_calls() {
            count_one(&mut self.bindings.pipeline_bind_calls);
        }
    }

    /// Records one bind group bind call.
    pub(crate) fn note_bind_group_bind(&mut self) {
        if self.detail.collects_bind_calls() {
            count_one(&mut self.bindings.bind_group_bind_calls);
        }
    }

    /// Records one vertex/index buffer bind call.
    pub(crate) fn note_buffer_bind(&mut self) {
        if self.detail.collects_bind_calls() {
            count_one(&mut self.bindings.buffer_bind_calls);
        }
    }

    /// Records one effective pipeline change.
    pub(crate) fn note_pipeline_change(&mut self) {
        if self.detail.collects_state_changes() {
            count_one(&mut self.bindings.pipeline_changes);
        }
    }

    /// Records one effective executable shader set change.
    pub(crate) fn note_shader_set_change(&mut self) {
        if self.detail.collects_state_changes() {
            count_one(&mut self.bindings.shader_set_changes);
        }
    }

    /// Records one effective bind group tuple change.
    pub(crate) fn note_bind_group_change(&mut self) {
        if self.detail.collects_state_changes() {
            count_one(&mut self.bindings.bind_group_changes);
        }
    }

    /// Records one effective buffer binding element change.
    pub(crate) fn note_buffer_binding_change(&mut self) {
        if self.detail.collects_state_changes() {
            count_one(&mut self.bindings.buffer_binding_changes);
        }
    }

    /// Records one effective texture element change.
    pub(crate) fn note_texture_binding_change(&mut self) {
        if self.detail.collects_state_changes() {
            count_one(&mut self.bindings.texture_binding_changes);
        }
    }

    /// Records one effective sampler element change.
    pub(crate) fn note_sampler_binding_change(&mut self) {
        if self.detail.collects_state_changes() {
            count_one(&mut self.bindings.sampler_binding_changes);
        }
    }

    /// Records one adjacent-raster-scope render target set change.
    pub(crate) fn note_render_target_set_change(&mut self) {
        if self.detail.collects_state_changes() {
            count_one(&mut self.bindings.render_target_set_changes);
        }
    }
}
