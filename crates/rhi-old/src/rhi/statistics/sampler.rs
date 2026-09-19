//! The caller-defined frame sampler and its FPS (§47.13).
//!
//! The RHI does not know what a "game frame" is, so it does not define one: a
//! sampler is created by a caller and sampled by that caller, and the rate it
//! reports is a CPU sampling rate, not a GPU or scan-out rate.

use crate::rhi::platform::{RhiError, RhiErrorKind, RhiResult};

use super::interval::IntervalStatistics;
use super::{DeviceStatistics, StatisticsSnapshot};

/// A caller-bounded frame sampler (§47.13).
///
/// It holds a baseline snapshot taken when the sampler was created, so the first
/// `sample_frame()` already reports a real interval - the elapsed time since the
/// caller asked for the sampler, which is normally the previous frame boundary.
pub struct FrameStatisticsSampler {
    statistics: DeviceStatistics,
    previous: StatisticsSnapshot,
}

impl FrameStatisticsSampler {
    /// A sampler whose baseline is `previous`.
    pub(crate) fn new(statistics: DeviceStatistics, previous: StatisticsSnapshot) -> Self {
        Self {
            statistics,
            previous,
        }
    }

    /// Samples the interval since the previous sample.
    ///
    /// Returns `InvalidUsage` when the collection epoch changed since the
    /// sampler was created: a delta across two detail levels would mix two
    /// collection rules, so the caller recreates the sampler instead (§47.13).
    pub fn sample_frame(&mut self) -> RhiResult<FrameStatistics> {
        let current = self.statistics.snapshot();
        if current.collection_epoch() != self.previous.collection_epoch() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "frame statistics span more than one collection epoch",
            )
            .at("sample_frame"));
        }
        let interval = current.delta_since(&self.previous)?;
        let fps = frames_per_second(interval.elapsed_cpu_ns);
        self.previous = current;
        Ok(FrameStatistics { interval, fps })
    }
}

/// One sampled frame interval (§47.13).
#[derive(Clone, Debug)]
pub struct FrameStatistics {
    interval: IntervalStatistics,
    fps: Option<f64>,
}

impl FrameStatistics {
    /// The counters recorded during this frame interval.
    pub fn interval(&self) -> &IntervalStatistics {
        &self.interval
    }

    /// The caller-defined sampling rate, absent when no time elapsed.
    ///
    /// This is the rate at which the caller sampled, not a GPU frame rate and
    /// not a display scan-out rate.
    pub fn fps(&self) -> Option<f64> {
        self.fps
    }
}

/// `1e9 / elapsed_cpu_ns`, or `None` when no time elapsed (§47.13).
///
/// Two samples taken in the same clock tick have no measurable rate, and
/// reporting infinity or a saturating large number would misreport the sampling
/// interval the caller actually used.
fn frames_per_second(elapsed_cpu_ns: u64) -> Option<f64> {
    if elapsed_cpu_ns == 0 {
        return None;
    }
    Some(1_000_000_000.0 / elapsed_cpu_ns as f64)
}
