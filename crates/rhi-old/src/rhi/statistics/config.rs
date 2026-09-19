//! Collection detail and configuration (§47.3).
//!
//! Detail is not a display filter: it decides which counters a domain
//! *collects*. A counter a level does not collect is never written, which is
//! what makes the gate cost nothing and what makes a new collection epoch
//! necessary whenever the level changes.

/// How much detail a statistics domain collects (§47.3).
///
/// The levels are cumulative. Each one is named for the data it adds, so a
/// caller can read a `Minimal` snapshot and a `Detailed` snapshot with the same
/// field names and still know which zeros mean "not collected".
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StatisticsDetail {
    /// Low-frequency data: live inventory, object lifecycle, submission, and
    /// presentation.
    ///
    /// Command, binding, and working-set counters stay at zero.
    ///
    /// This is the default level: a device that was never configured collects
    /// the low-frequency data and nothing else.
    #[default]
    Minimal,

    /// Adds the command counters (scope/draw/dispatch/copy/resolve/blit/
    /// upload/readback/marker) and the bind-call counters.
    Basic,

    /// Adds the effective state changes (pipeline, shader set, bind group,
    /// buffer binding, texture binding, sampler binding, render-target set) and
    /// the interval working set.
    Detailed,
}

impl StatisticsDetail {
    /// Whether scope and command counters are collected.
    pub(crate) fn collects_commands(self) -> bool {
        matches!(self, Self::Basic | Self::Detailed)
    }

    /// Whether bind-call counters are collected.
    pub(crate) fn collects_bind_calls(self) -> bool {
        matches!(self, Self::Basic | Self::Detailed)
    }

    /// Whether effective-state-change counters are collected.
    pub(crate) fn collects_state_changes(self) -> bool {
        matches!(self, Self::Detailed)
    }

    /// Whether the interval working set is collected.
    pub(crate) fn collects_working_set(self) -> bool {
        matches!(self, Self::Detailed)
    }
}

/// The collection configuration of one statistics domain (§47.3).
///
/// `#[non_exhaustive]`, so a later collection policy can be added without
/// breaking a caller that constructs one of the constructors below.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatisticsConfig {
    /// The detail level this domain collects.
    pub detail: StatisticsDetail,
}

impl StatisticsConfig {
    /// Inventory, lifecycle, submission, and presentation only.
    pub fn minimal() -> Self {
        Self {
            detail: StatisticsDetail::Minimal,
        }
    }

    /// `minimal()` plus command and bind-call counters.
    pub fn basic() -> Self {
        Self {
            detail: StatisticsDetail::Basic,
        }
    }

    /// `basic()` plus effective state changes and the interval working set.
    pub fn detailed() -> Self {
        Self {
            detail: StatisticsDetail::Detailed,
        }
    }

    /// The detail level this configuration collects.
    pub fn detail(&self) -> StatisticsDetail {
        self.detail
    }
}

impl Default for StatisticsConfig {
    fn default() -> Self {
        Self::minimal()
    }
}
