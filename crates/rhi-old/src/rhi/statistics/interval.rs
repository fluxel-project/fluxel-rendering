//! Interval data: per-lane usage and the detailed-mode working set
//! (§47.9, §47.12).
//!
//! An interval is the difference of two snapshots of one collection epoch. Two
//! of its parts cannot be produced by subtracting cumulative counters, so the
//! snapshots carry the raw material for them instead:
//!
//! * per-lane counts are cumulative per lane and differenced per lane;
//! * the working set is a *distinct-object* question per interval, so every
//!   actual use is stamped with the sequence number of the snapshot it precedes
//!   and an interval counts the objects whose last use falls inside its window.

use std::collections::HashMap;

use crate::rhi::format::SubmissionLaneId;
use crate::rhi::platform::ObjectId;
use crate::rhi::presentation::AcquiredFrameId;

use super::counters::count_one;

/// Cumulative accepted batch and work-item counts for one lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LaneCounters {
    /// Batches accepted on this lane.
    pub(crate) batches_accepted: u64,
    /// RecordedWork items accepted on this lane.
    pub(crate) recorded_work_items: u64,
}

impl LaneCounters {
    /// Counts one accepted batch and its work items on this lane, saturating.
    pub(crate) fn record(&mut self, batches: u64, work_items: u64) {
        super::counters::saturating_add(&mut self.batches_accepted, batches);
        super::counters::saturating_add(&mut self.recorded_work_items, work_items);
    }

    /// The per-field difference from `previous`, or `None` when this lane was
    /// not used during the interval.
    pub(crate) fn delta_since(&self, previous: Self) -> Option<Self> {
        let batches_accepted = self.batches_accepted.saturating_sub(previous.batches_accepted);
        let recorded_work_items = self
            .recorded_work_items
            .saturating_sub(previous.recorded_work_items);
        if batches_accepted == 0 && recorded_work_items == 0 {
            return None;
        }
        Some(Self {
            batches_accepted,
            recorded_work_items,
        })
    }
}

/// How much one submission lane was used during an interval (§47.9).
///
/// There is no `Default`: an interval reports only the lanes it used, so a
/// defaulted entry would name a lane the interval did not touch.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct LaneIntervalStatistics {
    /// The lane.
    pub lane: SubmissionLaneId,
    /// Logical batches accepted on it during this interval.
    pub batches_accepted: u64,
    /// RecordedWork items accepted on it during this interval.
    pub recorded_work_items: u64,
}

/// The distinct logical objects an interval actually used (§47.12).
///
/// Every "unique" value counts logical IDs - [`ObjectId`], [`AcquiredFrameId`],
/// [`SubmissionLaneId`] - never a native handle.
///
/// [`ObjectId`]: crate::rhi::platform::ObjectId
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct WorkingSetStatistics {
    /// Distinct buffers used.
    pub unique_buffers: u64,
    /// Distinct textures used.
    pub unique_textures: u64,
    /// Distinct acquired frame attachments used.
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

    /// Distinct submission lanes submitted to.
    pub unique_submission_lanes: u64,
}

/// One logical object an actual-use observation can refer to.
///
/// This is the working-set key domain: it is deliberately logical, so a
/// backend reports what Fluxel used rather than what it lowered.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum UsedObject {
    /// A buffer.
    Buffer(ObjectId),
    /// A texture.
    Texture(ObjectId),
    /// An acquired frame attachment.
    FrameAttachment(AcquiredFrameId),
    /// A sampler.
    Sampler(ObjectId),
    /// A bind group.
    BindGroup(ObjectId),
    /// A shader module.
    ShaderModule(ObjectId),
    /// A raster pipeline.
    RasterPipeline(ObjectId),
    /// A compute pipeline.
    ComputePipeline(ObjectId),
    /// A submission lane.
    SubmissionLane(SubmissionLaneId),
}

/// The last-use table of a detailed-mode domain.
///
/// Each entry records the sequence number of the snapshot that follows the
/// object's most recent use, so "the working set of the interval between
/// snapshots `p` and `s`" is one pass over entries whose stamp lies in
/// `(p, s]`. This is collected only at [`StatisticsDetail::Detailed`], where
/// maintaining local unique sets is explicitly allowed (§47.19).
///
/// Entries are dropped when an object is reclaimed, which bounds the table by
/// the live inventory plus whatever a backend used but has not reclaimed yet.
///
/// [`StatisticsDetail::Detailed`]: super::StatisticsDetail
#[derive(Clone, Debug, Default)]
pub(crate) struct WorkingSetTable {
    last_use: HashMap<UsedObject, u64>,
}

impl WorkingSetTable {
    /// Stamps `object` as used by the interval that starts at `sequence`.
    pub(crate) fn note_use(&mut self, object: UsedObject, sequence: u64) {
        self.last_use.insert(object, sequence);
    }

    /// Drops `object`, because it no longer exists.
    pub(crate) fn forget(&mut self, object: UsedObject) {
        self.last_use.remove(&object);
    }

    /// The distinct objects whose last use falls in `(after, at_or_before]`.
    pub(crate) fn interval(&self, after: u64, at_or_before: u64) -> WorkingSetStatistics {
        let mut working_set = WorkingSetStatistics::default();
        for (object, sequence) in &self.last_use {
            if *sequence <= after || *sequence > at_or_before {
                continue;
            }
            match object {
                UsedObject::Buffer(_) => count_one(&mut working_set.unique_buffers),
                UsedObject::Texture(_) => count_one(&mut working_set.unique_textures),
                UsedObject::FrameAttachment(_) => {
                    count_one(&mut working_set.unique_frame_attachments);
                }
                UsedObject::Sampler(_) => count_one(&mut working_set.unique_samplers),
                UsedObject::BindGroup(_) => count_one(&mut working_set.unique_bind_groups),
                UsedObject::ShaderModule(_) => count_one(&mut working_set.unique_shader_modules),
                UsedObject::RasterPipeline(_) => count_one(&mut working_set.unique_raster_pipelines),
                UsedObject::ComputePipeline(_) => {
                    count_one(&mut working_set.unique_compute_pipelines);
                }
                UsedObject::SubmissionLane(_) => {
                    count_one(&mut working_set.unique_submission_lanes);
                }
            }
        }
        working_set
    }
}

/// What one snapshot must remember for a later [`IntervalStatistics`].
///
/// It is stored by value in the snapshot, not shared with the live domain, so a
/// snapshot stays a complete and immutable record of its own moment.
#[derive(Clone, Debug)]
pub(crate) struct IntervalBasis {
    /// Per-lane cumulative counts, in canonical lane order.
    pub(crate) lanes: Vec<(SubmissionLaneId, LaneCounters)>,
    /// The last-use table, present exactly at `Detailed`.
    pub(crate) working_set: Option<WorkingSetTable>,
}

impl IntervalBasis {
    /// The per-lane difference from `previous`, keeping only lanes this
    /// interval actually used and sorting them canonically.
    pub(crate) fn lane_delta(&self, previous: &Self) -> Vec<LaneIntervalStatistics> {
        let mut lanes = Vec::new();
        for (lane, counters) in &self.lanes {
            let before = previous
                .lanes
                .binary_search_by_key(lane, |(candidate, _)| *candidate)
                .map(|index| previous.lanes[index].1)
                .unwrap_or_default();
            if let Some(delta) = counters.delta_since(before) {
                lanes.push(LaneIntervalStatistics {
                    lane: *lane,
                    batches_accepted: delta.batches_accepted,
                    recorded_work_items: delta.recorded_work_items,
                });
            }
        }
        lanes
    }

    /// The working set of `(after, at_or_before]`, absent when this basis was
    /// collected at a level that does not track it.
    pub(crate) fn working_set_delta(
        &self,
        after: u64,
        at_or_before: u64,
    ) -> Option<WorkingSetStatistics> {
        let working_set = self.working_set.as_ref()?;
        Some(working_set.interval(after, at_or_before))
    }
}

/// One interval of counters, measured between two snapshots (§47.9).
///
/// The interval carries the most recent snapshot's device and epoch, because
/// `delta_since` refuses to compare across either.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct IntervalStatistics {
    /// The device identity the two snapshots belong to.
    pub device: crate::rhi::platform::DeviceIdentity,
    /// The collection epoch the two snapshots belong to.
    pub collection_epoch: u64,

    /// Monotonic CPU nanoseconds between the two snapshots.
    pub elapsed_cpu_ns: u64,

    /// Command counters recorded during the interval.
    pub commands: super::counters::CommandStatistics,
    /// Binding counters recorded during the interval.
    pub bindings: super::counters::BindingStatistics,
    /// Submission counters recorded during the interval.
    pub submissions: super::counters::SubmissionStatistics,
    /// Presentation counters recorded during the interval.
    pub presentation: super::counters::PresentationStatistics,
    /// Resource lifecycle counters recorded during the interval.
    pub resources: super::counters::ResourceLifecycleStatistics,

    /// Lanes actually used during this interval, sorted by lane.
    pub lanes: Vec<LaneIntervalStatistics>,

    /// The distinct objects used during this interval, present exactly when the
    /// collection level tracks it.
    pub working_set: Option<WorkingSetStatistics>,
}
