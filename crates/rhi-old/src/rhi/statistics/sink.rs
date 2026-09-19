//! The crate-private sink a backend records statistics through (§47.4).
//!
//! Nothing here is public. The sink is the whole write side of the module:
//! a backend reports logical Fluxel events and this file decides which counter
//! group each one moves, whether the current detail level collects it, and how
//! the event reaches the live inventory and the working set.
//!
//! Two write paths exist on purpose:
//!
//! * [`DeviceStatistics::merge_recorder`] is the bulk merge of a recorder's
//!   local counters at `finish()`. It is the only path that carries command and
//!   binding counts, so recording a draw never touches a shared lock.
//! * every other method is one low-frequency device-level event: a submission, a
//!   present, an acquire, or an object lifecycle transition.

use crate::rhi::platform::ObjectId;
use crate::rhi::presentation::{AcquireErrorKind, AcquiredFrameId, PresentState};

use super::counters::count_one;
use super::inventory::used_object;
use super::{CreatedObject, DeviceStatistics, LaneWork, ObjectKind, RecorderCounters};
use super::{SubmissionEvent, UsedObject};

impl DeviceStatistics {
    /// Bulk-merges one finished recorder's local counters (§47.4).
    ///
    /// `uses` is the recorder's actual-use list, which is the only honest source
    /// of "which objects did this interval really use": a binding change is not
    /// a use, and a pipeline change that starts reading an already-bound
    /// resource is not a binding change (§47.7).
    ///
    /// The merge is gated by the *current* detail level, not by the level the
    /// recorder was created with. A `configure()` that lands between recording
    /// and `finish()` must not inject counters of a level the new epoch says are
    /// not collected.
    pub(crate) fn merge_recorder(&self, counters: &RecorderCounters, uses: &[UsedObject]) {
        let mut state = self.inner.lock();
        let detail = state.config.detail;
        state
            .cumulative
            .commands
            .saturating_add_gated(counters.commands(), detail);
        state
            .cumulative
            .bindings
            .saturating_add_gated(counters.bindings(), detail);
        if detail.collects_working_set() {
            let sequence = state.next_sequence;
            for used in uses {
                state.working_set.note_use(*used, sequence);
            }
        }
    }

    /// Records one recorder reaching `finish()`.
    pub(crate) fn record_recorder_finished(&self) {
        count_one(&mut self.inner.lock().cumulative.commands.recorders_finished);
    }

    /// Records one `Device::submit()` call and its plan accounting (§47.8).
    ///
    /// A partially accepted plan reports one event carrying its accepted batch
    /// and work-item counts: it is accepted, not rejected, and the whole plan is
    /// never relabeled.
    pub(crate) fn record_submission(&self, event: &SubmissionEvent) {
        let mut state = self.inner.lock();
        state.cumulative.submissions.saturating_add_event(event);
        for lane in &event.lanes {
            let LaneWork {
                lane,
                batches_accepted,
                recorded_work_items,
            } = *lane;
            state
                .lanes
                .entry(lane)
                .or_default()
                .record(batches_accepted, recorded_work_items);
            if state.config.detail.collects_working_set() {
                let sequence = state.next_sequence;
                state
                    .working_set
                    .note_use(UsedObject::SubmissionLane(lane), sequence);
            }
        }
    }

    /// Records an acquire that produced a frame.
    pub(crate) fn record_acquire_succeeded(&self) {
        count_one(&mut self.inner.lock().cumulative.presentation.acquires_succeeded);
    }

    /// Records an acquire refusal (§47.10).
    ///
    /// Only the four refusal kinds the frozen counters name are counted.
    /// `FrameOutstanding`, a zero-size or suspended target, a lost device, and an
    /// acquire-time out-of-memory increment nothing: they planned no present, and
    /// relabeling them as a present outcome would corrupt the accounting the
    /// present categories exist for.
    pub(crate) fn record_acquire_refused(&self, kind: AcquireErrorKind) {
        let mut state = self.inner.lock();
        let counter = match kind {
            AcquireErrorKind::NotReady => &mut state.cumulative.presentation.acquire_not_ready,
            AcquireErrorKind::Timeout => &mut state.cumulative.presentation.acquire_timeout,
            AcquireErrorKind::Outdated => &mut state.cumulative.presentation.acquire_outdated,
            AcquireErrorKind::TargetLost => &mut state.cumulative.presentation.acquire_target_lost,
            AcquireErrorKind::FrameOutstanding
            | AcquireErrorKind::ZeroSizeOrSuspended
            | AcquireErrorKind::DeviceLost
            | AcquireErrorKind::OutOfMemory => return,
        };
        count_one(counter);
    }

    /// Records a frame entering `Acquired`.
    pub(crate) fn record_frame_acquired(&self, frame: AcquiredFrameId) {
        self.inner.lock().inventory.insert_frame(frame);
    }

    /// Records a frame leaving `Acquired`/`PlannedForPresent`.
    pub(crate) fn record_frame_terminal(&self, frame: AcquiredFrameId) {
        let mut state = self.inner.lock();
        state.inventory.remove_frame(frame);
        state.working_set.forget(UsedObject::FrameAttachment(frame));
    }

    /// Records a frame consumed by present planning.
    pub(crate) fn record_present_planned(&self) {
        count_one(&mut self.inner.lock().cumulative.presentation.presents_planned);
    }

    /// Records a terminal present outcome (§47.10).
    ///
    /// A terminal state increments exactly one applicable category. `Pending` is
    /// not terminal, and device loss is reported by device state rather than
    /// relabeled as a present outcome.
    pub(crate) fn record_present_terminal(&self, state: &PresentState) {
        let mut domain = self.inner.lock();
        let counter = match state {
            PresentState::Accepted => &mut domain.cumulative.presentation.presents_accepted,
            PresentState::Outdated => &mut domain.cumulative.presentation.presents_outdated,
            PresentState::TargetLost => &mut domain.cumulative.presentation.presents_target_lost,
            PresentState::Failed(_) => &mut domain.cumulative.presentation.presents_failed,
            PresentState::Pending | PresentState::DeviceLost(_) => return,
        };
        count_one(counter);
    }

    /// Records explicit abandon or Drop-safety abandonment of a frame.
    pub(crate) fn record_frame_abandoned(&self) {
        count_one(&mut self.inner.lock().cumulative.presentation.frames_abandoned);
    }

    /// Records an object entering the live inventory (§47.11, §47.14).
    pub(crate) fn record_object_created(&self, object: &CreatedObject<'_>) {
        let mut state = self.inner.lock();
        let created = match object.kind() {
            ObjectKind::Buffer => &mut state.cumulative.resources.buffers_created,
            ObjectKind::Texture => &mut state.cumulative.resources.textures_created,
            ObjectKind::TextureView => &mut state.cumulative.resources.texture_views_created,
            ObjectKind::Sampler => &mut state.cumulative.resources.samplers_created,
            ObjectKind::ShaderModule => &mut state.cumulative.resources.shader_modules_created,
            ObjectKind::BindGroupLayout => {
                &mut state.cumulative.resources.bind_group_layouts_created
            }
            ObjectKind::BindGroup => &mut state.cumulative.resources.bind_groups_created,
            ObjectKind::PipelineInterface => {
                &mut state.cumulative.resources.pipeline_interfaces_created
            }
            ObjectKind::RasterPipeline => &mut state.cumulative.resources.raster_pipelines_created,
            ObjectKind::ComputePipeline => &mut state.cumulative.resources.compute_pipelines_created,
        };
        count_one(created);
        state.inventory.insert(object);
    }

    /// Records an object whose backing satisfied completion-safe reclaim
    /// conditions.
    ///
    /// This is not a handle drop. The lifecycle counter and the live table move
    /// together under one lock, so a snapshot can never see a reclaimed object
    /// still counted as live.
    pub(crate) fn record_object_reclaimed(&self, kind: ObjectKind, id: ObjectId) {
        let mut state = self.inner.lock();
        let reclaimed = match kind {
            ObjectKind::Buffer => &mut state.cumulative.resources.buffers_reclaimed,
            ObjectKind::Texture => &mut state.cumulative.resources.textures_reclaimed,
            ObjectKind::TextureView => &mut state.cumulative.resources.texture_views_reclaimed,
            ObjectKind::Sampler => &mut state.cumulative.resources.samplers_reclaimed,
            ObjectKind::ShaderModule => &mut state.cumulative.resources.shader_modules_reclaimed,
            ObjectKind::BindGroupLayout => {
                &mut state.cumulative.resources.bind_group_layouts_reclaimed
            }
            ObjectKind::BindGroup => &mut state.cumulative.resources.bind_groups_reclaimed,
            ObjectKind::PipelineInterface => {
                &mut state.cumulative.resources.pipeline_interfaces_reclaimed
            }
            ObjectKind::RasterPipeline => &mut state.cumulative.resources.raster_pipelines_reclaimed,
            ObjectKind::ComputePipeline => {
                &mut state.cumulative.resources.compute_pipelines_reclaimed
            }
        };
        count_one(reclaimed);
        state.inventory.remove(kind, id);
        if let Some(used) = used_object(kind, id) {
            state.working_set.forget(used);
        }
    }

    /// Records one actual use that a device-level path observed outside a
    /// recorder merge.
    ///
    /// A recorder reports its uses in bulk at `finish()`; this entry point is for
    /// the few uses a device path knows on its own.
    pub(crate) fn record_use(&self, used: UsedObject) {
        let mut state = self.inner.lock();
        if !state.config.detail.collects_working_set() {
            return;
        }
        let sequence = state.next_sequence;
        state.working_set.note_use(used, sequence);
    }
}
