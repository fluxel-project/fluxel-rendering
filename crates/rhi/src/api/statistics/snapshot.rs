//! Snapshot consistency, the interval delta, and the frame sampler
//! (specification 47.4, 47.9, and 47.13).
//!
//! A snapshot is a consistent read of a device's cumulative counters; an
//! interval is the difference between two snapshots; a frame sample is an
//! interval whose boundaries the *caller* chose. The three are one subject —
//! "what happened between two points I picked" — which is why they share a
//! module.
//!
//! The one thing this module is emphatic about is what a snapshot is not. It
//! does not wait for the GPU (section 47.4), it does not observe a driver, and
//! its `cpu_time_ns` is the *statistics clock*, a monotonic CPU time, not a frame
//! time. A caller that reads a GPU duration out of these numbers is reading a
//! number this chapter never claimed to produce.

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::DeviceIdentity;

use super::counters::{
    BindingStatistics, CommandStatistics, CumulativeStatistics, LaneIntervalStatistics,
    PresentationStatistics, ResourceLifecycleStatistics, SubmissionStatistics,
    WorkingSetStatistics,
};

/// A consistent read of one device's cumulative counters.
///
/// `sequence` orders snapshots taken under the same epoch, and a snapshot is
/// only comparable to an earlier one from the same device and epoch
/// ([`Self::delta_since`]). The counters themselves are counts since
/// [`super::DeviceStatistics::configure`] last started an epoch, so a number in
/// here is never "since device creation" unless no reconfigure ever happened.
///
/// Fields are private, unlike the counter records: a caller may hold and
/// compare a snapshot, and there is nothing useful it could build one out of.
#[derive(Clone, Debug)]
pub struct StatisticsSnapshot {
    device: DeviceIdentity,
    collection_epoch: u64,
    sequence: u64,
    cpu_time_ns: u64,
    cumulative: CumulativeStatistics,
}

impl StatisticsSnapshot {
    /// Assembles a snapshot.
    ///
    /// Crate-private: only the collection domain that observed the counters may
    /// say what they were. A caller-built snapshot would be a fabricated
    /// observation, and [`Self::delta_since`] would happily subtract it.
    pub(crate) fn new(
        device: DeviceIdentity,
        collection_epoch: u64,
        sequence: u64,
        cpu_time_ns: u64,
        cumulative: CumulativeStatistics,
    ) -> Self {
        Self {
            device,
            collection_epoch,
            sequence,
            cpu_time_ns,
            cumulative,
        }
    }

    /// The device this snapshot observed.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The collection epoch this snapshot was taken under.
    pub fn collection_epoch(&self) -> u64 {
        self.collection_epoch
    }

    /// A monotonic counter ordering snapshots within an epoch.
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Monotonic CPU time on the device statistics clock, in nanoseconds.
    ///
    /// Not wall-clock time, not GPU time, and not comparable across processes or
    /// devices. Its only defined use is as the difference between two snapshots
    /// of the same device, which is what an interval's `elapsed_cpu_ns` is and
    /// what the sampler's rate is computed from.
    pub fn cpu_time_ns(&self) -> u64 {
        self.cpu_time_ns
    }

    /// The cumulative counters as read.
    pub fn cumulative(&self) -> &CumulativeStatistics {
        &self.cumulative
    }

    /// The difference between this snapshot and an earlier one.
    ///
    /// # Refusals
    ///
    /// Three portable preconditions, all
    /// [`RhiErrorKind::InvalidUsage`], all checked before anything else:
    ///
    /// ```text
    /// same DeviceIdentity
    /// same collection_epoch
    /// self.sequence >= previous.sequence
    /// ```
    ///
    /// The epoch rule is the one that matters most. Section 47.3 restarts the
    /// cumulative counters on every reconfigure, so subtracting across epochs
    /// would produce a negative or absurd interval rather than a refusal, and a
    /// caller would have no way to notice. Section 49 lists this comparison —
    /// "statistics snapshot DeviceIdentity / collection epoch compatibility" —
    /// among the checks that must happen before anything else.
    ///
    pub fn delta_since(&self, previous: &StatisticsSnapshot) -> RhiResult<IntervalStatistics> {
        if self.device != previous.device {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "cannot take the difference of snapshots from two different devices",
            ));
        }
        if self.collection_epoch != previous.collection_epoch {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "cannot take the difference of snapshots from two collection epochs; \
                 reconfigure restarts the cumulative counters, so the difference would \
                 not be an interval",
            ));
        }
        if self.sequence < previous.sequence {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "cannot take the difference of a snapshot and a later one; the interval \
                 would run backwards",
            ));
        }
        // The default runtime service currently has no per-lane or working-set
        // instrumentation. Its cumulative source is initialized to zero and is
        // reset atomically on every epoch, therefore this is an exact empty
        // interval rather than a guessed native metric.
        Ok(IntervalStatistics {
            device: self.device,
            collection_epoch: self.collection_epoch,
            elapsed_cpu_ns: self.cpu_time_ns.saturating_sub(previous.cpu_time_ns),
            commands: CommandStatistics::default(),
            bindings: BindingStatistics::default(),
            submissions: SubmissionStatistics::default(),
            presentation: PresentationStatistics::default(),
            resources: ResourceLifecycleStatistics::default(),
            lanes: Vec::new(),
            working_set: None,
        })
    }
}

/// What happened between two snapshots of one device.
///
/// The interval counterpart of [`CumulativeStatistics`], plus the two things
/// that are meaningful only over an interval: the per-lane breakdown and the
/// optional working set. Section 47.9 lists only lanes actually used during the
/// interval, sorted canonically by lane identity, so that two devices' intervals
/// can be compared row by row.
///
/// `elapsed_cpu_ns` is the statistics-clock difference and is what
/// [`FrameStatistics::fps`] divides into. It is a CPU sampling interval, which is
/// why it can be chosen by the caller at all.
///
/// # `Clone` and `Debug` are forced additions
///
/// Section 47.9 writes `#[non_exhaustive]` on this struct and no derives at all.
/// The two here are not an embellishment: section 47.13 writes
/// `#[derive(Clone, Debug)]` on [`FrameStatistics`], which owns one of these in a
/// private field, so the derives are pulled through rather than chosen. Where the
/// specification lists no derives and nothing forces one, none is added.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct IntervalStatistics {
    /// The device this interval observed.
    pub device: DeviceIdentity,
    /// The collection epoch both endpoints were taken under.
    pub collection_epoch: u64,

    /// Statistics-clock nanoseconds between the two snapshots.
    pub elapsed_cpu_ns: u64,

    /// Command and scope counts during the interval.
    pub commands: CommandStatistics,
    /// Bind and effective-state-change counts during the interval.
    pub bindings: BindingStatistics,
    /// Submission structure counts during the interval.
    pub submissions: SubmissionStatistics,
    /// Presentation lifecycle counts during the interval.
    pub presentation: PresentationStatistics,
    /// Logical object lifecycle counts during the interval.
    pub resources: ResourceLifecycleStatistics,

    /// Per-lane usage, sorted by lane identity.
    ///
    /// Lists only lanes actually used during this interval.
    pub lanes: Vec<LaneIntervalStatistics>,

    /// The unique objects used during the interval, when the collection level
    /// includes them.
    ///
    /// `None` at [`super::StatisticsDetail::Minimal`] and
    /// [`super::StatisticsDetail::Basic`], where the working set is not
    /// collected. `None` means "not collected", which is a different statement
    /// from a `Some` whose counts are zero.
    pub working_set: Option<WorkingSetStatistics>,
}

/// A sampler for a caller-defined frame boundary.
///
/// Returned by [`super::DeviceStatistics::frame_sampler`], and the reason the
/// RHI has no notion of a frame: the caller picks the boundary, the sampler
/// remembers the snapshot from the previous boundary, and each
/// [`Self::sample_frame`] reports what happened between them.
///
/// No derives, matching section 47.13, which lists none here while it lists
/// `Clone` and `Debug` on [`FrameStatistics`] two blocks below. A sampler is a
/// piece of caller-held state that reports through
/// [`Self::sample_frame`]; the deliberate absence of `Clone` means a caller
/// cannot fork one boundary into two divergent histories, and the absence of
/// `Debug` keeps whatever the port stores alongside the previous snapshot out of
/// a log.
pub struct FrameStatisticsSampler {
    statistics: DeviceStatisticsHandle,
    previous: StatisticsSnapshot,
}

/// The service handle a sampler holds.
///
/// A type alias of convenience: a sampler owns a handle to the same device's
/// statistics domain rather than a snapshot, because it has to take a new
/// snapshot at each boundary.
type DeviceStatisticsHandle = super::DeviceStatistics;

impl FrameStatisticsSampler {
    /// Starts a sampler whose first sample covers the turn since `previous`.
    ///
    /// Crate-private: a sampler is handed out by
    /// [`super::DeviceStatistics::frame_sampler`], which is where the opening
    /// snapshot is taken.
    pub(crate) fn new(statistics: super::DeviceStatistics, previous: StatisticsSnapshot) -> Self {
        Self {
            statistics,
            previous,
        }
    }

    /// Samples the interval since the previous call.
    ///
    /// # Refusals
    ///
    /// If the collection epoch changed since the sampler was created, this is
    /// [`RhiErrorKind::InvalidUsage`] and the caller recreates the sampler
    /// (section 47.13). The refusal comes from
    /// [`StatisticsSnapshot::delta_since`]'s epoch rule, which is the same rule:
    /// an interval may not straddle a reconfigure, because the counters on the
    /// far side start from zero and the difference would be meaningless rather
    /// than merely wrong.
    ///
    /// A caller that wants to accept a reconfigure does not need a new API: it
    /// drops this sampler and asks for another one.
    ///
    /// The sampler takes a fresh synchronous snapshot on each call. The epoch
    /// refusal is reachable whenever a concurrent caller reconfigures the
    /// device statistics service between two samples.
    pub fn sample_frame(&mut self) -> RhiResult<FrameStatistics> {
        let current = self.statistics.snapshot();
        let interval = current.delta_since(&self.previous)?;
        let fps = frame_rate(interval.elapsed_cpu_ns);
        self.previous = current;
        Ok(FrameStatistics::new(interval, fps))
    }
}

/// One caller-defined frame sample.
///
/// Holds the interval and the sampling rate side by side rather than deriving
/// the rate at the call site, so that a caller displaying it does not have to
/// remember the formula — and, more importantly, so that the answer for an empty
/// interval is decided in one place.
#[derive(Clone, Debug)]
pub struct FrameStatistics {
    interval: IntervalStatistics,
    fps: Option<f64>,
}

impl FrameStatistics {
    /// Pairs an interval with its computed rate.
    ///
    /// Crate-private: the rate is defined as a function of the interval, and a
    /// caller that could pass an arbitrary one would be able to record a number
    /// the RHI never computed.
    pub(crate) fn new(interval: IntervalStatistics, fps: Option<f64>) -> Self {
        Self { interval, fps }
    }

    /// The interval this sample covers.
    pub fn interval(&self) -> &IntervalStatistics {
        &self.interval
    }

    /// The sampling rate, or `None` when the interval was empty.
    ///
    /// Section 47.13 fixes the formula:
    ///
    /// ```text
    /// elapsed_cpu_ns > 0   -> 1e9 / elapsed_cpu_ns
    /// elapsed_cpu_ns == 0  -> None
    /// ```
    ///
    /// `None` rather than an infinity, because an interval of zero says the two
    /// boundaries fell inside one statistics clock tick and there is no rate to
    /// report. This is a **caller-defined render-loop CPU sampling rate**. It is
    /// not a GPU frame rate and not a display scan-out rate, and section 47.1
    /// leaves those to a presentation timing extension rather than deriving them
    /// from this number.
    pub fn fps(&self) -> Option<f64> {
        self.fps
    }
}

/// The sampling rate of an interval measured on the statistics clock.
///
/// Section 47.13's formula, as a function so that it can be checked without a
/// device: the rule is arithmetic on one number and needs nothing from a
/// backend, which is exactly the kind of portable rule section 4 requires to be
/// decided above the backend rather than below it.
///
/// Returns `None` for an empty interval, and never an infinity or a NaN.
pub(crate) fn frame_rate(elapsed_cpu_ns: u64) -> Option<f64> {
    if elapsed_cpu_ns == 0 {
        None
    } else {
        Some(1e9 / elapsed_cpu_ns as f64)
    }
}
