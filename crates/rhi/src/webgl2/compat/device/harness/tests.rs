//! The tests of the drawing entry, in their own module.
//!
//! Responsibility: assert what [`super::drive_desktop_gl4_draws`] reports when
//! the workload runs, and nothing about a driver -- the mock backend is the
//! vehicle because the claim under test is *which number goes where*, which is
//! a fact about the projection rather than about hardware.
//!
//! Not owned here: the entry and its report types ([`super`]), the adapter the
//! real path builds, and the real-context evidence, which no in-crate test can
//! produce (`CLAUDE.md` §4.5).
//!
//! # Why the file, and not the head of `super`
//!
//! It was the tail of `super::harness.rs`, and that file crossed §8's 600-line
//! ceiling with its tests counted -- 794 lines, of which these were 220.  The
//! subject of `super` is one thing (drive the workload, report what it cost) and
//! does not need splitting; what it did not need was its own tests inside it.
//! Nothing here is a narrower subject than `super`, so this is a move rather
//! than a division.

use super::*;
use crate::webgl2::api::tests::snapshot;
use crate::webgl2::api::{GlFamilyProfile, MockGlFamilyApi, OwnerThreadIdentity};
use crate::webgl2::state::StateDomain;

/// One run of the workload over the mock backend this crate's own suites use.
///
/// The mock is the right vehicle for exactly the claim under test: what this
/// module has to get right is *which number goes where*, and that is a fact
/// about the projection rather than about a driver.  A real context would make
/// the same assertions harder to read and no more true.
///
/// The reading is projected from the same snapshot the mock is built from,
/// through the same function the real entry calls, so the mock run exercises
/// the whole of the workload rather than a reduced version of it.
fn drive(mode: ExecutionMode, draws: u32) -> DesktopGl4DrawReport {
    let discovery = snapshot(GlFamilyProfile::WebGl2);
    // The test thread owns the mock, so the identity it reports is this
    // thread's -- which is the same fact the real entry records.
    let reading = conformance::report(&discovery, [4, 4], &OwnerThreadIdentity::current());
    let cost = workload::drive(
        MockGlFamilyApi::from_discovery(discovery),
        mode,
        draws,
        [4, 4],
        stepping_clock(),
    )
    .unwrap_or_else(|error| panic!("the workload runs over the mock context in {mode:?}: {error}"));
    DesktopGl4DrawReport::from_cost(reading, cost)
}

/// A clock that is not a clock: it advances one fixed step per read.
///
/// The workload takes its clock from its caller, because the two surfaces that
/// drive it have different ones and `Instant` does not exist on
/// `wasm32-unknown-unknown` at all.  That makes the durations a *wiring* fact
/// rather than a timing fact, and this is the vehicle that can assert wiring:
/// with a real clock the assertion would be a race it could lose, and the
/// numbers below would say nothing a second run would repeat.
fn stepping_clock() -> impl Fn() -> u64 {
    let tick = std::cell::Cell::new(0u64);
    move || {
        tick.set(tick.get() + STEP_NANOS);
        tick.get()
    }
}

/// One step of [`stepping_clock`], in nanoseconds.
const STEP_NANOS: u64 = 1_000;

/// A report's per-domain rows, as `(name, requests, emitted, skipped)`.
fn rows(report: &DesktopGl4DrawReport) -> Vec<(String, u64, u64, u64)> {
    report
        .domains
        .iter()
        .map(|domain| {
            (
                domain.domain.clone(),
                domain.requests,
                domain.emitted,
                domain.skipped,
            )
        })
        .collect()
}

fn sum(rows: &[(String, u64, u64, u64)], pick: fn(&(String, u64, u64, u64)) -> u64) -> u64 {
    rows.iter().map(pick).sum()
}

#[test]
fn the_workload_runs_as_one_pass_whatever_the_mode() {
    for mode in [ExecutionMode::Optimized, ExecutionMode::Oracle] {
        let report = drive(mode, 8);
        assert_eq!(report.passes, 1, "the workload is one pass, in {mode:?}");
        assert_eq!(report.pass_loads, 1, "with one attachment loaded");
        assert_eq!(report.pass_stores, 1, "and stored at the end");
        assert_eq!(report.draws_requested, 8);
        assert_eq!(report.drawable_extent, [4, 4]);
        assert_eq!(
            report.domains.len(),
            StateDomain::COUNT,
            "the report names every domain, not only the ones this frame touched"
        );
    }
}

/// The two durations are the gaps between the clock reads the run makes.
///
/// The exact numbers are the assertion rather than a coincidence of the stepping
/// clock: they say the run reads its clock *four* times -- once to open the run,
/// once either side of the executor call, once to close the run -- so
/// `submit_nanos` is one step and `total_nanos` is three, and the second contains
/// the first.  A run that read its clock per draw would also produce two
/// plausible-looking numbers, and would be reporting the wrong interval.
///
/// This is the regression guard for the defect that put the clock in the
/// signature: the durations used to come from `std::time::Instant` inside the
/// workload, which panics on `wasm32-unknown-unknown`, so the browser surface
/// could not drive the workload at all.  Nothing *here* would have caught that --
/// a native-only test cannot -- which is why the browser test exists beside it.
#[test]
fn the_durations_are_the_gaps_between_the_clock_reads_the_run_makes() {
    let report = drive(ExecutionMode::Optimized, 2);
    assert_eq!(
        report.submit_nanos, STEP_NANOS,
        "one step: the clock is read either side of the executor call, and nowhere inside it"
    );
    assert_eq!(
        report.total_nanos,
        3 * STEP_NANOS,
        "three steps: open the run, open the executor call, close it, close the run"
    );
}

/// A differential is only a differential if both halves complete.
///
/// Found on hardware, not here: the optimized path was the only one that
/// preserved an invariant the draw verb asserts.  A pipeline install binds
/// the pipeline's vertex array and records it; under the oracle the next
/// geometry request re-derives that array and destroys the one it replaced,
/// so the recorded id is dead by the time the draw re-resolves it, and the
/// run refuses with `draw-raster / vertex array is not live` -- at *one*
/// draw, not at some count.  The recorder has to model that refusal or this
/// test passes for the wrong reason, which is what it did before the
/// recorder was taught the rule.
#[test]
fn the_uncached_path_completes_a_frame_with_more_than_one_draw() {
    for draws in [1, 2, 8] {
        let oracle = drive(ExecutionMode::Oracle, draws);
        assert_eq!(oracle.draws_requested, draws);
        assert_eq!(oracle.passes, 1);
    }
}

/// The differential this whole entry exists for.
///
/// The identity a reader might expect -- `oracle.emitted ==
/// optimized.emitted + optimized.skipped` -- does **not** hold, and this test
/// says so rather than asserting it.  Measured with eight draws: `optimized`
/// emits 8 and skips 21, `oracle` emits 51.  The reason is a property of the
/// layer and not of this harness -- a domain may emit for a reason that is not
/// a request, so `requests` is not `emitted + skipped` in either mode:
/// `pipeline` is asked eight times and answers with two emits and seven skips
/// under one mode and sixteen emits under the other.
///
/// What *is* mode-independent is `requests` itself, and that is the fact
/// pinned here: the mode decides what the layer does about a request, never
/// what the frame asks for.  A differential whose two halves were asked
/// different things would be measuring two workloads.
#[test]
fn the_two_modes_are_asked_the_same_thing_and_answer_differently() {
    let optimized = drive(ExecutionMode::Optimized, 8);
    let oracle = drive(ExecutionMode::Oracle, 8);
    let (optimized_rows, oracle_rows) = (rows(&optimized), rows(&oracle));

    let requests = |rows: &[(String, u64, u64, u64)]| -> Vec<(String, u64)> {
        rows.iter()
            .map(|(name, requests, _, _)| (name.clone(), *requests))
            .collect()
    };
    assert_eq!(
        requests(&optimized_rows),
        requests(&oracle_rows),
        "the mode decides what the layer does about a request, never what the frame asks for"
    );

    let skipped = |rows: &[(String, u64, u64, u64)]| sum(rows, |row| row.3);
    let emitted = |rows: &[(String, u64, u64, u64)]| sum(rows, |row| row.2);

    assert!(
        skipped(&optimized_rows) > 0,
        "a repeated draw loop dirties nothing between iterations, so an optimized layer \
         proves some calls redundant -- a zero here would mean the workload had stopped \
         being the steady state the funnel screens against"
    );
    assert_eq!(
        skipped(&oracle_rows),
        0,
        "and the oracle proves nothing redundant, by definition"
    );
    assert!(
        emitted(&oracle_rows) > emitted(&optimized_rows),
        "so the oracle emits strictly more: {} against {}",
        emitted(&oracle_rows),
        emitted(&optimized_rows)
    );
    for (name, _, optimized_emitted, _) in &optimized_rows {
        let oracle_emitted = oracle_rows
            .iter()
            .find(|row| row.0 == *name)
            .map(|row| row.2)
            .expect("both modes report the same domains");
        assert!(
            oracle_emitted >= *optimized_emitted,
            "{name}: a layer that skips nothing cannot emit fewer calls than one that does"
        );
    }
}

/// The oracle keeps no derived-cache traffic, which bounds what it can
/// baseline.
///
/// Pinned rather than discovered later: a candidate judged on cache hits or
/// created entries has no oracle number to compare against, because the
/// uncached path never consults those caches.  Such a candidate is screened
/// against the optimized run's own earlier revision instead, and the funnel
/// has to say so before it screens one.
#[test]
fn the_oracle_reports_no_derived_cache_traffic() {
    let oracle = drive(ExecutionMode::Oracle, 8);
    assert_eq!(
        (
            oracle.cache_hits,
            oracle.cache_misses,
            oracle.cache_created,
            oracle.cache_live_entries
        ),
        (0, 0, 0, 0),
        "the uncached path consults no derived cache, so it has nothing to report"
    );
    let optimized = drive(ExecutionMode::Optimized, 8);
    assert!(
        optimized.cache_hits + optimized.cache_misses + optimized.cache_created > 0,
        "while the optimized path does consult them, which is the asymmetry"
    );
}

#[test]
fn a_mode_is_parsed_or_refused_rather_than_defaulted() {
    assert!(matches!(
        parse_mode("optimized"),
        Ok(ExecutionMode::Optimized)
    ));
    assert!(matches!(parse_mode("oracle"), Ok(ExecutionMode::Oracle)));
    let refused = parse_mode("fast").expect_err("a spelling that is neither is refused");
    assert!(
        refused.contains("fast"),
        "and the refusal names what it got"
    );
}

/// Both refusals happen while the request is still a request.
///
/// Checked on `parse_request` rather than the entry because the entry's next
/// act is to open a context over a real drawable, and a window is the one
/// thing these two refusals exist to avoid needing: they are decided before
/// the host is consulted at all.
#[test]
fn a_workload_is_refused_before_a_context_is_opened() {
    assert_eq!(
        parse_request("optimized", 1, 0).expect_err("a zero-draw workload measures only setup"),
        "a workload of zero draws measures only the setup, so it is refused"
    );
    assert_eq!(
        parse_request("optimized", 0, 8).expect_err("a zero identity names no device"),
        "a device identity has to be nonzero"
    );
    assert!(
        parse_request("oracle", 7, 8).is_ok(),
        "and a well-formed request is not refused"
    );
}
