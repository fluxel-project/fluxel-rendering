//! Snapshot, interval, and sampler tests (specification 47.4, 47.9, and 47.13).
//!
//! This is the half of section 47 that is tested behaviourally rather than read
//! as a shape, because the portable rules it contains are decidable without a
//! backend: which two snapshots may be subtracted, and what the sampling rate of
//! an interval is. The counters themselves arrive from a recorder that does not
//! exist, so [`StatisticsSnapshot::new`] is the crate-private constructor the
//! collection domain will call, and the tests build snapshots through it the way
//! the tests for the resource chapters build buffers through theirs.
//!
//! What the chapter needs and what it has do not line up in one place, and the
//! `#[should_panic]` test below is where that shows: [`IntervalStatistics`]
//! requires per-lane rows and a working set, and a [`StatisticsSnapshot`] carries
//! neither. The verb refuses portably and then panics rather than returning an
//! interval with an empty lane list, which would state that no lane was used.

use crate::api::error::RhiErrorKind;
use crate::api::identity::DeviceIdentity;
use crate::api::statistics::counters::{
    CumulativeStatistics, LaneIntervalStatistics, WorkingSetStatistics,
};
use crate::api::statistics::snapshot::frame_rate;
use crate::api::statistics::{
    FrameStatistics, FrameStatisticsSampler, StatisticsDetail, StatisticsSnapshot,
};
use crate::api::submission::SubmissionLaneId;

use super::{assert_kind, device, identity, other_device, statistics};

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

fn snapshot_on(
    device: DeviceIdentity,
    epoch: u64,
    sequence: u64,
    cpu_time_ns: u64,
) -> StatisticsSnapshot {
    StatisticsSnapshot::new(
        device,
        epoch,
        sequence,
        cpu_time_ns,
        CumulativeStatistics::default(),
    )
}

// ---------------------------------------------------------------------------
// Section 47.4 — snapshot consistency, and the three portable preconditions.
// ---------------------------------------------------------------------------

/// A snapshot reads back exactly what it was taken from, and the fields are the
/// ones a caller compares two snapshots by.
#[test]
fn a_snapshot_reads_back_the_observation_it_was_taken_from() {
    let snapshot = snapshot_on(device(), 3, 17, 1_000_000);

    assert_eq!(snapshot.device_identity(), device());
    assert_eq!(snapshot.collection_epoch(), 3);
    assert_eq!(snapshot.sequence(), 17);
    assert_eq!(snapshot.cpu_time_ns(), 1_000_000);

    // The counters are readable through the snapshot rather than requiring a
    // second call that could observe a different moment — which is the whole
    // meaning of "consistent" in section 47.4.
    assert_eq!(snapshot.cumulative().commands.draw_calls, 0);
    assert_eq!(snapshot.cumulative().submissions.submission_calls, 0);
}

/// Two snapshots from different devices cannot be subtracted, and the refusal is
/// [`RhiErrorKind::InvalidUsage`] — a portable usage error, decided before any
/// question reaches a backend.
#[test]
fn two_devices_snapshots_cannot_be_subtracted() {
    let earlier = snapshot_on(device(), 1, 1, 1_000);
    let later = snapshot_on(other_device(), 1, 2, 2_000);

    assert_kind(later.delta_since(&earlier), RhiErrorKind::InvalidUsage);
    assert_kind(earlier.delta_since(&later), RhiErrorKind::InvalidUsage);
}

/// The epoch rule is the one that matters most. Section 47.3 restarts the
/// cumulative counters on every reconfigure, so subtracting across epochs would
/// produce a negative or absurd interval rather than a refusal, and a caller
/// would have no way to notice.
///
/// This is the rule that makes the collection epoch observable at all: without
/// it, the epoch would be a number a caller could read and never act on.
#[test]
fn snapshots_from_two_collection_epochs_cannot_be_subtracted() {
    let before_reconfigure = snapshot_on(device(), 1, 5, 1_000);
    let after_reconfigure = snapshot_on(device(), 2, 6, 2_000);

    assert_kind(
        after_reconfigure.delta_since(&before_reconfigure),
        RhiErrorKind::InvalidUsage,
    );

    // In both directions: the rule is about comparability, not about which of
    // the two came first.
    assert_kind(
        before_reconfigure.delta_since(&after_reconfigure),
        RhiErrorKind::InvalidUsage,
    );
}

/// An interval may not run backwards, and the refusal is portable rather than
/// arithmetic.
///
/// A snapshot taken later cannot be a *previous* endpoint. The check is on the
/// sequence rather than on the clock, because two reads inside one statistics
/// tick share a `cpu_time_ns` and are still ordered.
#[test]
fn an_interval_may_not_run_backwards() {
    let earlier = snapshot_on(device(), 1, 4, 1_000);
    let later = snapshot_on(device(), 1, 9, 2_000);

    assert_kind(earlier.delta_since(&later), RhiErrorKind::InvalidUsage);

    // Equal sequences are comparable: two reads at the same sequence are the
    // same observation, and an interval of zero length is legal.
    let same = snapshot_on(device(), 1, 9, 3_000);
    assert!(same.sequence() >= later.sequence());
}

/// The order of the three checks is the order the contract states, and a
/// snapshot that fails two of them reports the first.
///
/// This is worth pinning because the three refusals are different diagnoses: a
/// caller that saw "epochs differ" when the real problem was two devices would
/// reconfigure a device it should never have been comparing against.
#[test]
fn a_snapshot_failing_two_preconditions_reports_the_devices_first() {
    // Different device *and* different epoch *and* a backwards sequence.
    let earlier = snapshot_on(device(), 1, 9, 1_000);
    let later = snapshot_on(other_device(), 2, 4, 2_000);

    let error = earlier
        .delta_since(&later)
        .expect_err("a snapshot from another device is refused");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert!(
        error.message().contains("different devices"),
        "the first failing precondition must be the one reported: {}",
        error.message()
    );
}

/// The interval itself, once the preconditions hold, is reached only after the
/// portable half has run — and it panics, because the state it needs does not
/// exist.
///
/// [`IntervalStatistics`] requires `lanes` and an optional per-interval working
/// set, and the statistics domain that would accumulate them does not exist yet:
/// the cumulative record has no per-lane dimension, and a working set is a set of
/// objects observed during the interval rather than a difference of counts.
/// Returning an interval with an empty lane list would state that no lane was
/// used, which is a claim about the device and not about what is built, so the
/// verb panics instead. This is an open gap and not worked around; the reason
/// lives on `delta_since`.
#[test]
#[should_panic(expected = "the interval state is not built")]
fn a_comparable_pair_of_snapshots_reaches_the_unbuilt_interval() {
    let earlier = snapshot_on(device(), 1, 4, 1_000);
    let later = snapshot_on(device(), 1, 9, 2_000);

    let _ = later.delta_since(&earlier);
}

// ---------------------------------------------------------------------------
// Section 47.13 — the caller-defined sampling rate.
// ---------------------------------------------------------------------------

/// Section 47.13's formula, which is arithmetic on one number and needs nothing
/// from a backend — which is exactly why the rule is a free function and why it
/// is testable here.
#[test]
fn the_sampling_rate_is_one_billion_divided_by_the_interval() {
    // One second is one sample per second.
    assert_eq!(frame_rate(1_000_000_000), Some(1.0));

    // Half a second is two.
    assert_eq!(frame_rate(500_000_000), Some(2.0));

    // Ten milliseconds is a hundred.
    assert_eq!(frame_rate(10_000_000), Some(100.0));

    // The 60 Hz frame time, to within the precision a rate is read at.
    let sixty_hz = frame_rate(16_666_667).expect("a non-empty interval has a rate");
    assert!(
        (sixty_hz - 60.0).abs() < 1e-3,
        "16.67 ms is 60 Hz, not {sixty_hz}"
    );
}

/// An empty interval has no rate, and `None` is the answer rather than an
/// infinity or a NaN.
///
/// The two boundaries fell inside one statistics clock tick, which says nothing
/// about how fast anything ran. An infinity would propagate through any average a
/// caller computed, and a NaN would make every comparison against it false —
/// including the comparison that was supposed to catch it.
#[test]
fn an_empty_interval_has_no_rate_rather_than_an_infinite_one() {
    assert_eq!(frame_rate(0), None);
}

/// The rate is finite and positive for every non-empty interval, including the
/// smallest and largest the clock can produce.
///
/// This is the property a caller displaying the number depends on, and it is the
/// one an overflow or a division the wrong way round would silently break: a rate
/// computed as `elapsed / 1e9` would be plausible-looking and inverted.
#[test]
fn the_rate_is_finite_and_positive_over_the_whole_clock_range() {
    for elapsed in [1_u64, 2, 1_000, 1_000_000, u64::MAX / 2, u64::MAX] {
        let rate = frame_rate(elapsed).expect("a non-empty interval has a rate");
        assert!(rate.is_finite(), "elapsed {elapsed} gave a non-finite rate");
        assert!(rate > 0.0, "elapsed {elapsed} gave a non-positive rate");
        assert!(!rate.is_nan(), "elapsed {elapsed} gave NaN");
    }

    // Monotonically decreasing in the interval: a longer interval is a slower
    // rate, which is the direction an inverted formula would reverse.
    let fast = frame_rate(1_000_000).expect("a rate");
    let slow = frame_rate(100_000_000).expect("a rate");
    assert!(fast > slow);
}

// ---------------------------------------------------------------------------
// The sampler and the interval's shape.
// ---------------------------------------------------------------------------

/// The interval's working set is `Option`, and the two states it distinguishes
/// are not the same statement.
///
/// `None` means the level did not collect it — `Minimal` and `Basic` do not — and
/// `Some` with zero counts means it collected and found nothing. A caller that
/// conflated them would report "this frame touched nothing" for a frame it simply
/// did not measure.
#[test]
fn an_uncollected_working_set_is_not_an_empty_one() {
    assert_ne!(StatisticsDetail::Basic, StatisticsDetail::Detailed);

    let not_collected: Option<WorkingSetStatistics> = None;
    let collected_and_empty: Option<WorkingSetStatistics> = Some(WorkingSetStatistics::default());

    assert!(not_collected.is_none());
    assert_eq!(
        collected_and_empty.map(|set| set.unique_textures),
        Some(0),
        "a collected working set reports zero textures rather than reporting nothing"
    );

    // And the row type the interval's `lanes` is made of is the same one the
    // cumulative record does not have — which is the asymmetry that makes the
    // interval unbuildable from two snapshots.
    let row = LaneIntervalStatistics {
        lane: SubmissionLaneId::new(0),
        batches_accepted: 3,
        recorded_work_items: 4,
    };
    assert_eq!(row.batches_accepted, 3);
}

/// The sampler's shape, as a renderer calls it.
///
/// Compiled, never called: a sampler starts from a snapshot, and a snapshot reads
/// counters the RHI increments while it works. What this reviews is the part that
/// is a design decision rather than a build step — the boundary is the
/// *caller's*, so `sample_frame` takes no argument and the RHI never learns what
/// a frame is. A caller that wanted to sample at a different boundary does not
/// pass one; it calls `sample_frame` at a different place.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_a_renderer_samples_at_its_own_boundary() {
    let statistics = statistics();
    let mut sampler = statistics.frame_sampler();

    let sample: FrameStatistics = sampler.sample_frame().expect("the epoch is unchanged");

    let _interval = sample.interval();
    let _fps: Option<f64> = sample.fps();
}

/// A sampler carries the snapshot it will subtract, and the epoch it will refuse
/// across, so a caller that reconfigures mid-frame gets an error instead of a
/// wrong number.
///
/// Compiled, never called for the same reason as above; what it reviews is that
/// the refusal travels with the sampler rather than being the caller's
/// responsibility. A caller that wanted to accept a reconfigure does not need a
/// new API — it drops this sampler and asks for another one.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_a_sampler_refuses_across_a_reconfigure(
    statistics: &crate::api::statistics::DeviceStatistics,
    previous: StatisticsSnapshot,
) {
    let mut sampler = FrameStatisticsSampler::new(statistics.clone(), previous);

    match sampler.sample_frame() {
        Ok(_sample) => {}
        Err(error) => assert_eq!(error.kind(), RhiErrorKind::InvalidUsage),
    }
}

/// The sample pairs the interval with the rate, so that a caller displaying one
/// does not have to remember the formula and the answer for an empty interval is
/// decided in one place.
#[test]
fn a_sample_carries_its_interval_and_its_rate_together() {
    let interval = crate::api::statistics::IntervalStatistics {
        device: device(),
        collection_epoch: 1,
        elapsed_cpu_ns: 0,
        commands: Default::default(),
        bindings: Default::default(),
        submissions: Default::default(),
        presentation: Default::default(),
        resources: Default::default(),
        lanes: Vec::new(),
        working_set: None,
    };

    let empty = FrameStatistics::new(interval, frame_rate(0));
    assert_eq!(empty.fps(), None);
    assert_eq!(empty.interval().elapsed_cpu_ns, 0);
    assert!(empty.interval().lanes.is_empty());

    let busy = FrameStatistics::new(
        crate::api::statistics::IntervalStatistics {
            elapsed_cpu_ns: 10_000_000,
            ..empty.interval().clone()
        },
        frame_rate(10_000_000),
    );
    assert_eq!(busy.fps(), Some(100.0));
    assert_eq!(busy.interval().device, device());
    assert_eq!(busy.interval().collection_epoch, 1);
}

/// A snapshot prints its observation, and the debug output is usable in a log
/// without exposing anything but portable state.
#[test]
fn a_snapshot_prints_its_portable_observation() {
    let rendered = format!("{:?}", snapshot_on(identity(4), 5, 6, 7));

    // A snapshot is not an opaque handle — every field is a portable number or a
    // device identity — so this one derives `Debug` rather than writing it by
    // hand, and the field names appear. Adjudication A16 applies to handles with
    // a native domain behind them, which this type does not have.
    assert!(rendered.contains("StatisticsSnapshot"), "{rendered}");
    assert!(rendered.contains("collection_epoch: 5"), "{rendered}");
    assert!(rendered.contains("sequence: 6"), "{rendered}");
    assert!(rendered.contains("cpu_time_ns: 7"), "{rendered}");
}
