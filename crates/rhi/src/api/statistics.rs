//! Statistics — portable logical statistics (specification section 47).
//!
//! **This file is being written in specification order** and is the root of the
//! statistics chapter. It owns the device-scoped service, the collection-detail
//! and collection-epoch model, and the two device verbs that turn a collected
//! observation into a number a caller can read. The frozen metric definitions
//! themselves are split along section seams into the submodules below:
//!
//! ```text
//! counters   47.5 - 47.8, 47.10 - 47.12   the cumulative and interval counters
//! snapshot   47.4, 47.9, 47.13            consistency, the interval delta, the
//!                                         frame sampler
//! inventory  47.14 - 47.18                live inventory and the logical memory
//!                                         estimate
//! ```
//!
//! The split is a size decision, not a responsibility decision: section 47 is
//! one chapter with one subject, and the seams are the section boundaries the
//! specification already draws.
//!
//! # What statistics are
//!
//! Counts and durations the RHI observed about **its own portable work**: the
//! commands it was asked to record, the state changes that were effective, the
//! structure of what was submitted, the presentation lifecycle, the logical
//! object lifecycle, the live inventory, a descriptor-based byte estimate, and a
//! caller-defined sampling interval.
//!
//! # What statistics are not
//!
//! Section 47.1 freezes the boundary, and the boundary is the point of the
//! chapter. Statistics are **not**:
//!
//! ```text
//! native barrier count            actual queue overlap
//! native descriptor bind count    driver PSO cache hit/miss
//! native command buffer count     hardware counters / occupancy / cache misses
//! native queue switch             actual VRAM allocation / residency / fragmentation
//!                                 GPU timestamp / frame time
//!                                 display scan-out FPS
//! ```
//!
//! Those arrive later, independently, from a query surface, backend tooling, an
//! allocator telemetry channel, and present timing. A backend that reported a
//! native number through this chapter would make two backends incomparable while
//! looking comparable, which is worse than reporting nothing.
//!
//! Two consequences a reader should keep in mind:
//!
//! - A statistic that reads zero is not evidence that nothing happened. A
//!   counter is only collected at the detail level that includes it, and only
//!   for work the RHI was told about.
//! - Statistics never change RHI semantics. Section 47.19 lists what an
//!   implementation may do (recorder-local counters, merge at `finish()`, merge
//!   at submission, versioned lifecycle inventory, local unique sets) and what
//!   it may not (insert GPU commands or barriers, wait for the GPU, change lane
//!   assignment or graph culling or aliasing or present behaviour, serialize
//!   parallel recorders). Turning statistics on may cost CPU time and memory and
//!   may cost nothing else.
//!
//! # Device scope
//!
//! Each [`DeviceIdentity`] has an independent statistics domain and there is no
//! global singleton. A caller running two devices aggregates the two snapshots
//! itself; the RHI does not combine them into one counter, because a combined
//! number would have no owner and no epoch.

use crate::api::error::RhiResult;
use crate::api::identity::DeviceIdentity;
use crate::api::platform::device::Device;
use crate::api::resource::buffer::Buffer;
use crate::api::resource::texture::Texture;

pub mod counters;
pub mod inventory;
pub mod snapshot;

pub use counters::{
    BindingStatistics, CommandStatistics, CumulativeStatistics, LaneIntervalStatistics,
    PresentationStatistics, ResourceLifecycleStatistics, SubmissionStatistics,
    WorkingSetStatistics,
};
pub use inventory::{
    InventoryStatistics, LiveObjectCounts, MemoryEstimate, MemoryEstimateQuality,
    ResourceMemoryStatistics,
};
pub use snapshot::{
    FrameStatistics, FrameStatisticsSampler, IntervalStatistics, StatisticsSnapshot,
};

/// How much of section 47's counter set is collected.
///
/// A level, not a set of flags: the levels nest, and each one adds the cost of
/// the level below it. Switching level restarts the cumulative counters, which
/// is what the collection epoch exists to make unambiguous.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatisticsDetail {
    /// Low-frequency data such as inventory, lifecycle, submission, and
    /// presentation.
    ///
    /// The default. Per-command counters are not collected, so this level costs
    /// no work in a hot recording loop.
    Minimal,
    /// Adds draw, dispatch, copy, scope, and bind-call counts.
    ///
    /// This is the level at which the recorder has to update counters per
    /// command.
    Basic,
    /// Adds effective state changes and the interval working set.
    ///
    /// The working set additionally requires the recorder to maintain a set of
    /// the unique objects it actually used, which is the most expensive level.
    Detailed,
}

/// What a device's statistics service is collecting.
///
/// A one-field struct rather than a bare enum, and `#[non_exhaustive]`: section
/// 47.3 gives today's shape and section 47.20.1 permits a future setting to be
/// added without breaking a caller that already stores a config. The field is
/// `pub` because the specification writes it that way.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatisticsConfig {
    /// How much of the counter set to collect.
    pub detail: StatisticsDetail,
}

impl StatisticsConfig {
    /// Collects inventory, lifecycle, submission, and presentation only.
    pub fn minimal() -> Self {
        Self {
            detail: StatisticsDetail::Minimal,
        }
    }

    /// Adds per-command counts.
    pub fn basic() -> Self {
        Self {
            detail: StatisticsDetail::Basic,
        }
    }

    /// Adds effective state changes and the interval working set.
    pub fn detailed() -> Self {
        Self {
            detail: StatisticsDetail::Detailed,
        }
    }
}

impl Default for StatisticsConfig {
    /// The cheapest level, which is also the one that cannot surprise a caller
    /// by adding per-command work to a recording loop they already wrote.
    fn default() -> Self {
        Self::minimal()
    }
}

/// A device's statistics service.
///
/// Obtained from [`Device::statistics`]. Cloneable, and every clone refers to
/// the **same** collection domain: the domain belongs to the device, not to the
/// handle, so `configure()` through one handle is visible through another. That
/// is section 47.2's rule that each `DeviceIdentity` has one independent
/// statistics domain, and it is why this type is an opaque handle rather than a
/// value a caller owns.
///
/// There is no global statistics singleton, and no way to aggregate two devices'
/// domains here: section 47.2 puts that combination in the caller's hands.
///
/// # What is built and what is not
///
/// Two of the verbs below are answered from data already in hand — the
/// descriptor-based memory estimates of section 47.15 — and they are
/// implemented. The rest read counters that the RHI increments while it records,
/// submits, and presents, and that code does not exist yet, so they panic with a
/// message naming what is missing.
pub struct DeviceStatistics {
    device: DeviceIdentity,
}

// Written by hand rather than derived, per adjudication A16 in the 0.16 plan:
// the backend port will add a handle to the collection domain here, that handle
// has no reason to be `Debug`, and printing a native object into a log is a
// leak. Only the portable identity is shown.
impl core::fmt::Debug for DeviceStatistics {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("DeviceStatistics")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl Clone for DeviceStatistics {
    /// Yields another handle to the same device's domain, not a second copy of
    /// the counters.
    fn clone(&self) -> Self {
        Self {
            device: self.device,
        }
    }
}

impl DeviceStatistics {
    /// Opens a handle to a device's statistics domain.
    ///
    /// Crate-private: section 47.2 scopes a domain to one `DeviceIdentity`, so
    /// only the device façade may hand one out. The domain itself lives on the
    /// device; until the backend port stores it there, this handle carries the
    /// identity it is scoped to and the verbs that would read the domain panic.
    pub(crate) fn new(device: DeviceIdentity) -> Self {
        Self { device }
    }

    /// Switches the collection level and starts a new collection epoch.
    ///
    /// The freeze rule of section 47.3, which is why this verb is not merely a
    /// setter:
    ///
    /// ```text
    /// configure()
    ///     -> collection_epoch += 1
    ///     -> event cumulative counters restart at 0
    ///     -> live inventory is not reset
    /// ```
    ///
    /// Restarting the cumulative counters is what removes the ambiguity of
    /// running `Minimal` for an hour and `Detailed` afterwards: without an
    /// epoch, a "since device creation" number would silently mix two collection
    /// regimes. The live inventory is not reset because it describes what
    /// exists, and what exists did not change.
    ///
    /// Thread-safe, and it does not require the GPU to be idle: no command is
    /// inserted and nothing waits (section 47.19).
    pub fn configure(&self, config: StatisticsConfig) -> RhiResult<()> {
        let _ = config;
        unimplemented!(
            "switching the collection level restarts the cumulative counters on \
             the device's collection domain; the contract is fixed, the \
             collection domain is not built"
        )
    }

    /// The collection level in effect.
    pub fn config(&self) -> StatisticsConfig {
        unimplemented!(
            "the collection level lives on the device's collection domain; the \
             contract is fixed, the collection domain is not built"
        )
    }

    /// The current collection epoch.
    ///
    /// Incremented once per successful [`Self::configure`]. A snapshot carries
    /// the epoch it was taken under, and two snapshots from different epochs
    /// cannot be subtracted ([`StatisticsSnapshot::delta_since`]).
    pub fn collection_epoch(&self) -> u64 {
        unimplemented!(
            "the epoch lives on the device's collection domain; the contract is \
             fixed, the collection domain is not built"
        )
    }

    /// Reads a consistent set of counters.
    ///
    /// Does not wait for the GPU. A concurrent event may land before or after
    /// the read, but one logical event is never torn in half: the returned set
    /// is internally consistent (section 47.4). That is a promise about the
    /// reader, not about time — two snapshots taken around a submission may
    /// both report it or neither may, and only the ordering of `sequence` says
    /// which reads are comparable.
    pub fn snapshot(&self) -> StatisticsSnapshot {
        unimplemented!(
            "a snapshot reads counters the RHI increments while it records, \
             submits, and presents; the contract is fixed, the counters are not \
             built"
        )
    }

    /// Starts a sampler for a caller-defined frame boundary.
    ///
    /// The RHI does not know what a frame is, so it does not decide where one
    /// starts: the caller chooses a boundary, calls
    /// [`FrameStatisticsSampler::sample_frame`] there, and gets the interval
    /// statistics and a sampling rate between two such boundaries.
    pub fn frame_sampler(&self) -> FrameStatisticsSampler {
        let _ = self;
        unimplemented!(
            "a sampler starts from a snapshot, and a snapshot reads counters the \
             RHI increments while it works; the contract is fixed, the counters \
             are not built"
        )
    }

    /// The live inventory and logical memory estimate.
    ///
    /// Answers "what exists right now", which is a different question from
    /// "what has happened": the counts here are not affected by a collection
    /// epoch, and do not restart when [`Self::configure`] runs.
    pub fn inventory(&self) -> RhiResult<InventoryStatistics> {
        unimplemented!(
            "the live inventory reads the object lifecycle table the RHI \
             maintains as it creates and reclaims objects; the contract is \
             fixed, the table is not built"
        )
    }

    /// The logical bytes a buffer is estimated to occupy.
    ///
    /// Section 47.16's rule is exact and needs no backend:
    ///
    /// ```text
    /// logical_estimated_bytes = BufferDescriptor.size
    /// ```
    ///
    /// It excludes native allocation padding, page granularity, and metadata, so
    /// it is a logical estimate and not a residency measurement — see
    /// [`MemoryEstimate`] for why the name is frozen that way.
    ///
    /// # Refusals
    ///
    /// A buffer created by a different device is [`crate::api::error::RhiErrorKind::WrongDevice`],
    /// checked before the descriptor is read. Section 3.3 makes that the only
    /// answer to cross-device use, and section 4 requires portable validation to
    /// reach it rather than letting a backend discover it.
    pub fn estimate_buffer_memory(&self, buffer: &Buffer) -> RhiResult<MemoryEstimate> {
        if buffer.device_identity() != self.device {
            return Err(crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::WrongDevice,
                "the buffer belongs to a different device than this statistics service",
            )
            .with_object(buffer.id()));
        }
        Ok(MemoryEstimate::logical(buffer.descriptor().size))
    }

    /// The logical bytes a texture is estimated to occupy.
    ///
    /// Section 47.17's rule sums the per-mip block count over every level and
    /// layer with checked arithmetic, and answers [`MemoryEstimate::unknown`]
    /// for any overflow and for a format whose backing the format name does not
    /// fix. The arithmetic itself is
    /// `inventory::estimate_texture_bytes`, which takes the format facts as a
    /// parameter so that the rule is testable without a device.
    ///
    /// Here it panics: the call needs the device's `FormatFacts` for the
    /// texture's format, and those come from the capability snapshot the backend
    /// port builds. It does not wait for the GPU and does not query a native
    /// heap — section 47.18 forbids both, because a number that arrives after a
    /// wait is a different kind of number and is not comparable across backends.
    ///
    /// # Refusals
    ///
    /// A texture created by a different device is [`crate::api::error::RhiErrorKind::WrongDevice`],
    /// checked before anything panics, for the reason given on
    /// [`Self::estimate_buffer_memory`].
    pub fn estimate_texture_memory(&self, texture: &Texture) -> RhiResult<MemoryEstimate> {
        if texture.device_identity() != self.device {
            return Err(crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::WrongDevice,
                "the texture belongs to a different device than this statistics service",
            )
            .with_object(texture.id()));
        }
        unimplemented!(
            "the estimate needs the device's FormatFacts for the texture's \
             format, which the backend port produces with the capability \
             snapshot; the contract is fixed, the facts are not built"
        )
    }
}

impl Device {
    /// This device's statistics service.
    ///
    /// Each call returns another handle to the same domain, because the domain
    /// belongs to the device identity rather than to the handle (section 47.2).
    /// That property is not visible until the domain exists: the backend port is
    /// what stores it on the device and threads it through here, and until then
    /// every handle is scoped to the identity and the collection verbs panic.
    ///
    /// The device façade owns that field and this module cannot reach it, so the
    /// port adds a crate-private accessor on `Device` rather than a public one:
    /// a statistics domain is not something a caller may pass around.
    pub fn statistics(&self) -> DeviceStatistics {
        DeviceStatistics::new(self.identity())
    }
}
