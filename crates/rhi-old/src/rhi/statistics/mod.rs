//! Portable logical statistics (rhi-design §47).
//!
//! Statistics answer one question with the same definitions on every backend:
//! *what logical work did this device do, and what does it still own?* They are
//! for a renderer, a graph, a debug HUD, and a benchmark. They are not a
//! replacement for PIX, RenderDoc, Xcode, or a vendor profiler, and no backend
//! difference is allowed to change a count.
//!
//! # What is observed
//!
//! Portable commands, portable state changes, logical submission structure,
//! presentation lifecycle, logical object lifecycle, live inventory,
//! descriptor-based logical resource bytes, and a caller-defined frame interval
//! with its sampling rate.
//!
//! # What is deliberately not observed
//!
//! No native barrier, descriptor bind, command buffer, or queue switch count; no
//! actual queue overlap; no driver PSO cache fact; no hardware counter,
//! occupancy, or cache statistic; no actual VRAM allocation, residency, or
//! fragmentation; no GPU timestamp or frame time; and no display scan-out rate.
//! Those belong to a future query, backend tooling, allocator telemetry, or
//! present-timing surface. A field that would imply one of them does not exist
//! here, and a snapshot never promises one.
//!
//! # One domain per device identity
//!
//! A [`DeviceStatistics`] handle is device-identity scoped and there is no
//! global singleton. Two devices are two domains; a caller that wants a combined
//! view combines two snapshots itself, because the RHI has no portable way to
//! add a DX12 device's counts to a Vulkan device's counts and call it one
//! number. Handles are cheap clones of one shared domain, so every clone of the
//! same device's handle reports the same counters.
//!
//! # Collection epochs
//!
//! Detail changes which counters are *collected*, so [`DeviceStatistics::configure`]
//! starts a new collection epoch: the epoch increments, event cumulative
//! counters restart at zero, and the live inventory is untouched. Without that,
//! a `Detailed` cumulative total would silently be read as "since device
//! creation" even though the first half of its life was collected at `Minimal`.
//! `configure()` is thread-safe and never requires the GPU to be idle: it takes
//! this domain's own lock and touches no device state.
//!
//! # Why a snapshot is consistent (§47.4)
//!
//! Everything a snapshot reports lives in one state behind one mutex, and every
//! write path applies a whole logical event inside one critical section:
//!
//! * a recorder's command and binding counters are accumulated *locally* and
//!   merged in bulk at `finish()`, so a completed recorder arrives as one event;
//! * a submission arrives as one event carrying every counter it moves, so a
//!   plan's accepted batches can never be visible without its accepted work
//!   items;
//! * an object lifecycle transition moves its counter and the live table
//!   together, so an object is never counted as created but not live.
//!
//! [`DeviceStatistics::snapshot`] clones that state while holding the same lock,
//! so the clone is a serialization point in the domain's total write order: it
//! sees every event that completed before it and no event that completes after
//! it. There is no window in which a logical event is half applied, which is
//! exactly the property a sequence of independent atomic reads cannot give -
//! thirty relaxed loads, each individually atomic, still belong to thirty
//! different moments, and a multi-field event would tear between them.
//!
//! Two further properties come from the same design:
//!
//! * **No GPU wait.** Nothing under the lock talks to a device, waits for
//!   completion, or polls a fence. A snapshot never blocks on GPU work, and it
//!   always reports the counters as of the moment it took the lock, so a caller
//!   may snapshot at any time.
//! * **No recorder serialization.** A recorder never takes this lock per
//!   command; it owns a `RecorderCounters` and merges once. Parallel recorders
//!   therefore never contend, and enabling statistics cannot reorder or
//!   serialize recording (§47.19).
//!
//! # Detail gating
//!
//! `Minimal` collects session structure, `Basic` adds per-command and bind-call
//! counts, and `Detailed` adds effective state changes and the interval working
//! set:
//!
//! | data | Minimal | Basic | Detailed |
//! | --- | --- | --- | --- |
//! | resource lifecycle, live inventory | yes | yes | yes |
//! | submissions, presentation, per-lane usage | yes | yes | yes |
//! | finished recorders | yes | yes | yes |
//! | scopes, draws, dispatches, copies, resolves, blits, uploads, readbacks, markers | - | yes | yes |
//! | pipeline/bind-group/buffer bind calls | - | yes | yes |
//! | effective pipeline, shader set, bind group, buffer/texture/sampler binding, render-target set changes | - | - | yes |
//! | interval working set | - | - | yes |
//!
//! A counter a level does not collect stays at zero, and the sink returns before
//! touching it, so a gated counter costs nothing. [`IntervalStatistics::working_set`]
//! is `None` rather than empty whenever the level does not collect it.
//!
//! # Memory estimates
//!
//! [`MemoryEstimate`] is descriptor arithmetic: a buffer is its descriptor size
//! and a texture is summed mip/block/array/sample arithmetic over declared format
//! facts. It excludes tiling, row alignment, driver metadata, compression, the
//! mip tail, fragmentation, aliasing, and residency, so it is named
//! `logical_estimated_bytes` and is never a VRAM or physical-byte number. A
//! format with implementation-defined backing reports `Unknown` rather than an
//! invented value.
//!
//! # The sink
//!
//! The write side is crate-private. A backend obtains a [`DeviceStatistics`]
//! handle from its device, keeps one `RecorderCounters` per recorder, and
//! reports device-level events (submission, acquire, present, object lifecycle)
//! through the sink methods on this type.
//!
//! The sink is complete but not yet driven. No production call site feeds it
//! today, so every counter reads zero until the owners of the recording,
//! submission, presentation, and creation/reclaim paths report their events. An
//! unwired counter that reads zero is indistinguishable from a device that did
//! nothing, so the wiring is a required follow-up, not an optional one: the
//! recorder reports its `RecorderCounters` and its actual uses at `finish()`,
//! a submission reports one `SubmissionEvent`, a presentation reports its
//! acquire and present outcomes, and every object creation or reclaim reports
//! through `CreatedObject`.

mod config;
mod counters;
mod interval;
mod inventory;
mod sampler;
mod sink;
mod state;

#[cfg(test)]
mod tests;

pub use config::{StatisticsConfig, StatisticsDetail};
pub use counters::{
    BindingStatistics, CommandStatistics, CumulativeStatistics, PresentationStatistics,
    ResourceLifecycleStatistics, SubmissionStatistics,
};
pub use interval::{IntervalStatistics, LaneIntervalStatistics, WorkingSetStatistics};
pub use inventory::{
    InventoryStatistics, LiveObjectCounts, MemoryEstimate, MemoryEstimateQuality,
    ResourceMemoryStatistics,
};
pub use sampler::{FrameStatistics, FrameStatisticsSampler};

pub(crate) use counters::{LaneWork, RecorderCounters, SubmissionEvent};
// The sink vocabulary a recorder reports commands with. It has no in-crate
// consumer yet - the recorder that implements `RecorderBackend` is the next one
// - so it is imported here as deliberate surface rather than left in a private
// module nothing outside statistics could name.
#[allow(
    unused_imports,
    reason = "sink vocabulary frozen for the recorder that will report commands"
)]
pub(crate) use counters::{CommandKind, ScopeKind};
pub(crate) use interval::UsedObject;
pub(crate) use inventory::{CreatedObject, ObjectKind};
pub(crate) use state::{Domain, StatisticsClock, SystemClock};

use std::sync::Arc;

use crate::rhi::platform::{
    DeviceIdentity, ObjectId, RhiError, RhiErrorKind, RhiResult,
};
use crate::rhi::resource::{Buffer, Texture};

use interval::IntervalBasis;

/// The statistics domain of one device identity (§47.2).
///
/// The handle is opaque and cheap to clone: every clone refers to the same
/// domain, and no clone can drift from another. It is the only entry point to
/// this module's data, and it is scoped to one [`DeviceIdentity`] so two devices
/// can never be summed into one counter by accident.
#[derive(Clone)]
pub struct DeviceStatistics {
    pub(super) inner: Arc<Domain>,
}

impl DeviceStatistics {
    /// A domain for `identity`, measured on the process monotonic clock.
    ///
    /// A backend calls this once per device identity; the device then hands out
    /// clones.
    pub(crate) fn new(identity: DeviceIdentity) -> Self {
        Self::with_clock(identity, Arc::new(SystemClock::new()))
    }

    /// A domain for `identity` measured on `clock`.
    ///
    /// The clock is injectable so an interval's arithmetic - and therefore FPS -
    /// can be pinned exactly instead of being read from the host's timer.
    pub(crate) fn with_clock(identity: DeviceIdentity, clock: Arc<dyn StatisticsClock>) -> Self {
        Self {
            inner: Arc::new(Domain::new(identity, clock)),
        }
    }

    /// Starts a new collection epoch with `config` (§47.3).
    ///
    /// The epoch increments, every *event* cumulative counter restarts at zero,
    /// and the live inventory is not reset: objects that exist are not a
    /// property of the collection rule. Counters that the new level does not
    /// collect are simply not written afterwards.
    ///
    /// This is thread-safe and does not require the GPU to be idle. It takes
    /// this domain's lock, writes configuration, and returns; it issues no
    /// device call and inserts no command or barrier.
    ///
    /// The `RhiResult` is reserved for a config a device cannot collect. Every
    /// configuration expressible today is portable, so this returns `Ok`.
    pub fn configure(&self, config: StatisticsConfig) -> RhiResult<()> {
        self.inner.lock().start_epoch(config);
        Ok(())
    }

    /// The configuration in force.
    pub fn config(&self) -> StatisticsConfig {
        self.inner.lock().config
    }

    /// The current collection epoch.
    ///
    /// A snapshot may only be differenced against a snapshot of the same epoch.
    pub fn collection_epoch(&self) -> u64 {
        self.inner.lock().epoch
    }

    /// A consistent snapshot of every counter, taken without waiting for the
    /// GPU (§47.4).
    ///
    /// See the module documentation for why the result cannot contain half of a
    /// logical event. The timestamp is read before the counters, so it is a lower
    /// bound on the moment they describe.
    ///
    /// In `Detailed` mode the snapshot also copies the last-use table, so a
    /// snapshot costs one pass over the objects that have been used since the
    /// current epoch began. Copying it is what makes a snapshot immutable and
    /// still valid after later uses, reclaims, or a new epoch.
    pub fn snapshot(&self) -> StatisticsSnapshot {
        let cpu_time_ns = self.inner.clock.now_ns();
        let mut state = self.inner.lock();
        let sequence = state.next_sequence;
        state.next_sequence = state.next_sequence.saturating_add(1);
        let basis = IntervalBasis {
            lanes: state
                .lanes
                .iter()
                .map(|(lane, counters)| (*lane, *counters))
                .collect(),
            working_set: if state.config.detail.collects_working_set() {
                Some(state.working_set.clone())
            } else {
                None
            },
        };
        StatisticsSnapshot {
            device: self.inner.identity,
            collection_epoch: state.epoch,
            sequence,
            cpu_time_ns,
            cumulative: state.cumulative.clone(),
            basis,
        }
    }

    /// A sampler whose baseline is a snapshot taken now (§47.13).
    pub fn frame_sampler(&self) -> FrameStatisticsSampler {
        FrameStatisticsSampler::new(self.clone(), self.snapshot())
    }

    /// The live inventory and its logical memory estimate (§47.15).
    ///
    /// The result is computed from the object lifecycle table, not from
    /// counters, so a clone of a handle is invisible to it and a reclaimed
    /// object is gone.
    ///
    /// The `RhiResult` is reserved for an identity that can no longer answer.
    /// A live domain always reports its inventory, so this returns `Ok`.
    pub fn inventory(&self) -> RhiResult<InventoryStatistics> {
        let state = self.inner.lock();
        Ok(InventoryStatistics {
            device: self.inner.identity,
            objects: state.inventory.counts(),
            memory: state.inventory.memory(),
        })
    }

    /// The logical memory estimate of one buffer (§47.16, §47.18).
    ///
    /// The answer comes from the buffer's canonical descriptor alone: this does
    /// not wait for the GPU and does not query a native heap.
    pub fn estimate_buffer_memory(&self, buffer: &Buffer) -> RhiResult<MemoryEstimate> {
        self.check_device(buffer.device_identity(), buffer.id(), "estimate_buffer_memory")?;
        Ok(inventory::estimate_buffer_bytes(buffer.descriptor()))
    }

    /// The logical memory estimate of one texture (§47.17, §47.18).
    ///
    /// The answer comes from the texture's canonical descriptor and declared
    /// format facts alone: this does not wait for the GPU and does not query a
    /// native heap.
    pub fn estimate_texture_memory(&self, texture: &Texture) -> RhiResult<MemoryEstimate> {
        self.check_device(
            texture.device_identity(),
            texture.id(),
            "estimate_texture_memory",
        )?;
        Ok(inventory::estimate_texture_bytes(texture.descriptor()))
    }

    /// Refuses a resource that belongs to another device identity.
    fn check_device(
        &self,
        owner: DeviceIdentity,
        object: ObjectId,
        operation: &'static str,
    ) -> RhiResult<()> {
        if owner == self.inner.identity {
            return Ok(());
        }
        Err(
            RhiError::new(
                RhiErrorKind::WrongDevice,
                "resource belongs to a different device identity than this statistics domain",
            )
            .at(operation)
            .on(object),
        )
    }
}

/// One consistent set of counters at one moment (§47.4).
///
/// A snapshot is immutable and self-contained: everything
/// [`StatisticsSnapshot::delta_since`] needs is inside it, so two snapshots of
/// the same epoch stay a valid pair no matter what happens to the device
/// afterwards.
#[derive(Clone, Debug)]
pub struct StatisticsSnapshot {
    device: DeviceIdentity,
    collection_epoch: u64,
    sequence: u64,
    cpu_time_ns: u64,
    cumulative: CumulativeStatistics,
    basis: IntervalBasis,
}

impl StatisticsSnapshot {
    /// The device identity this snapshot describes.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The collection epoch this snapshot belongs to.
    pub fn collection_epoch(&self) -> u64 {
        self.collection_epoch
    }

    /// This snapshot's sequence number, which orders snapshots of one epoch.
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Monotonic CPU nanoseconds on this device's statistics clock.
    pub fn cpu_time_ns(&self) -> u64 {
        self.cpu_time_ns
    }

    /// The cumulative counters, as of this snapshot's moment.
    pub fn cumulative(&self) -> &CumulativeStatistics {
        &self.cumulative
    }

    /// The counters recorded between `previous` and this snapshot (§47.4).
    ///
    /// Refuses with [`RhiErrorKind::InvalidUsage`] when the two snapshots belong
    /// to different device identities, to different collection epochs, or when
    /// `previous` is newer than this snapshot. Each of those would produce a
    /// number that looks like an interval but is not one: crossing identities or
    /// epochs mixes two definitions, and a reversed pair subtracts in the wrong
    /// direction.
    pub fn delta_since(&self, previous: &StatisticsSnapshot) -> RhiResult<IntervalStatistics> {
        if self.device != previous.device {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "snapshots from different device identities cannot be differenced",
            )
            .at("delta_since"));
        }
        if self.collection_epoch != previous.collection_epoch {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "snapshots from different collection epochs cannot be differenced",
            )
            .at("delta_since"));
        }
        if self.sequence < previous.sequence {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "snapshots must be differenced oldest first",
            )
            .at("delta_since"));
        }
        Ok(IntervalStatistics {
            device: self.device,
            collection_epoch: self.collection_epoch,
            elapsed_cpu_ns: self.cpu_time_ns.saturating_sub(previous.cpu_time_ns),
            commands: self
                .cumulative
                .commands
                .delta_since(&previous.cumulative.commands),
            bindings: self
                .cumulative
                .bindings
                .delta_since(&previous.cumulative.bindings),
            submissions: self
                .cumulative
                .submissions
                .delta_since(&previous.cumulative.submissions),
            presentation: self
                .cumulative
                .presentation
                .delta_since(&previous.cumulative.presentation),
            resources: self
                .cumulative
                .resources
                .delta_since(&previous.cumulative.resources),
            lanes: self.basis.lane_delta(&previous.basis),
            working_set: self
                .basis
                .working_set_delta(previous.sequence, self.sequence),
        })
    }
}
