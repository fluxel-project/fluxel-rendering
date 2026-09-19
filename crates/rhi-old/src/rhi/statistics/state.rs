//! The mutable state behind one statistics domain, and its clock.
//!
//! Everything a snapshot reports lives in one [`State`] behind one lock. That
//! single lock is the whole consistency argument: a logical event is applied as
//! one critical section, so a snapshot cannot observe half of one, and the
//! snapshot's clone is a serialization point in the domain's write order rather
//! than a sequence of independent reads that a writer could interleave.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use crate::rhi::format::SubmissionLaneId;
use crate::rhi::platform::DeviceIdentity;

use super::config::StatisticsConfig;
use super::counters::CumulativeStatistics;
use super::interval::{LaneCounters, WorkingSetTable};
use super::inventory::InventoryTable;

/// The monotonic CPU clock a statistics domain measures intervals with.
///
/// The domain owns its clock so that "CPU time" has one meaning per domain: the
/// nanoseconds since that domain was created, independent of wall-clock changes
/// and of every other domain.
pub(crate) trait StatisticsClock: Send + Sync {
    /// Monotonic nanoseconds since the domain was created.
    fn now_ns(&self) -> u64;
}

/// The process monotonic clock.
pub(crate) struct SystemClock {
    start: Instant,
}

impl SystemClock {
    /// A clock whose zero is now.
    pub(crate) fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}

impl StatisticsClock for SystemClock {
    fn now_ns(&self) -> u64 {
        // Saturates rather than wraps. Reaching `u64::MAX` nanoseconds needs
        // about 584 years of uptime, so this is a formality, but a wrapped clock
        // would silently invert every elapsed-time delta.
        u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }
}

/// Everything one [`DeviceStatistics`] handle shares.
///
/// [`DeviceStatistics`]: super::DeviceStatistics
pub(crate) struct Domain {
    /// The identity this domain belongs to.
    pub(crate) identity: DeviceIdentity,
    /// This domain's monotonic clock.
    pub(crate) clock: Arc<dyn StatisticsClock>,
    state: Mutex<State>,
}

impl Domain {
    /// A fresh domain for `identity`.
    pub(crate) fn new(identity: DeviceIdentity, clock: Arc<dyn StatisticsClock>) -> Self {
        Self {
            identity,
            clock,
            state: Mutex::new(State::new()),
        }
    }

    /// The domain state, recovering a poisoned lock.
    ///
    /// A panic inside a `note_*`/`record_*` call can leave a counter group part
    /// way through an update. Statistics are observations, so the worst case of
    /// continuing is a slightly wrong counter; refusing every later read would
    /// turn a diagnostic nuisance into a dead device.
    pub(crate) fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Everything a snapshot copies, all under one lock.
pub(crate) struct State {
    /// The collection configuration in force.
    pub(crate) config: StatisticsConfig,
    /// The current collection epoch.
    pub(crate) epoch: u64,
    /// The sequence number the next snapshot will carry.
    ///
    /// A use recorded now is stamped with this value, which places it in the
    /// interval that starts at that snapshot. Because both the stamp and the
    /// snapshot happen under this state's lock, a use is never lost between two
    /// intervals and never counted in both.
    pub(crate) next_sequence: u64,
    /// The cumulative counters of the current epoch.
    pub(crate) cumulative: CumulativeStatistics,
    /// Per-lane cumulative counts, kept in canonical lane order.
    pub(crate) lanes: BTreeMap<SubmissionLaneId, LaneCounters>,
    /// The detailed-mode last-use table.
    pub(crate) working_set: WorkingSetTable,
    /// The live object lifecycle table, which no epoch reset touches.
    pub(crate) inventory: InventoryTable,
}

impl State {
    /// The state of a domain that has never been configured.
    ///
    /// The first epoch is 0 with the default (minimal) configuration, so the
    /// first `configure()` produces epoch 1 and a caller can always tell "never
    /// configured" from "configured once".
    fn new() -> Self {
        Self {
            config: StatisticsConfig::default(),
            epoch: 0,
            next_sequence: 1,
            cumulative: CumulativeStatistics::default(),
            lanes: BTreeMap::new(),
            working_set: WorkingSetTable::default(),
            inventory: InventoryTable::default(),
        }
    }

    /// Starts a new collection epoch (§47.3).
    ///
    /// Event cumulative counters restart because they were collected under the
    /// old rule; the live inventory does not change, because the objects that
    /// exist are not a property of the collection rule.
    pub(crate) fn start_epoch(&mut self, config: StatisticsConfig) {
        self.config = config;
        self.epoch = self.epoch.saturating_add(1);
        self.cumulative = CumulativeStatistics::default();
        self.lanes.clear();
        self.working_set = WorkingSetTable::default();
    }
}
