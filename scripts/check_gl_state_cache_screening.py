#!/usr/bin/env python3
"""Screen the state cache against its own uncached baseline on real hardware.

Why this exists
---------------
``0.15-plan.md``'s experiment funnel is explicit that a candidate is retained
only when representative latency improves *beyond the measured noise*, emitted
calls fall as predicted, and the pixels stay identical -- and that a single run
per mode is a reading rather than a measurement.  The tree could produce the
reading (``examples/windows-gl4 --draws N --mode oracle|optimized``) but nothing
produced the measurement, so this checker is the funnel's steps 1--4 for the one
candidate this series can screen on hardware: interleaved, adjacent rounds of the
uncached baseline and the cached implementation over the same workload, with the
correctness gate asserted on every round rather than once.

What it measures, and what it deliberately does not
---------------------------------------------------
The clock belongs to the fixture's own run: ``submit_nanos`` brackets the
executor call and excludes context creation, discovery, buffer upload and graph
compilation, which is exactly "owner-thread CPU submission latency".  ``total_nanos``
brackets the whole run and *includes* the pixel readback this checker asks for, so
it is reported as a secondary reading and never used as the decision metric.

The picture gate is byte identity across every round of both modes, and it is a
gate rather than a statistic: a mode that rendered a different frame would make
the two halves of the differential incomparable, which is the confound the
differential exists to remove.  A run whose readback is missing, unreadable or
different fails the whole screening closed.

The guard, and why it is the same script
----------------------------------------
The funnel names guard workloads after the representative one, and the guard this
candidate most needs is the frame that has *nothing to skip*: a workload that
skips nothing is one where the cache is pure bookkeeping, so it is the frame in
which the candidate can only cost more than the calls it saves.  ``--guard`` runs
exactly that comparison -- and refuses any round that skipped something, so a run
that quietly became a screening again cannot be reported as its own guard.  The
arithmetic is shared with the screening because the question is the same
comparison read in the other direction; what differs is what the answer is called.

The guard varies the **draw count**, which varies how much of the frame is
redundant while holding the state the frame touches fixed.  That is one axis of
the guard set the funnel names and not the whole of it: the guards that vary the
*state* -- deliberate alternation, resource churn, resize, context recovery --
need a workload with a variety knob it does not have, and a screening that reads
this file's report has to say so rather than call the guard set covered.

What "beyond the measured noise" means here
-------------------------------------------
Stated in the report rather than left to the reader, and strict in both
directions.  The baseline's own rounds define the band: the candidate is retained
only when its **median** is below the baseline's **minimum**, and on a guard it
has regressed only when its median is above the baseline's **maximum**.  Between
those two the verdict is that the two modes are not distinguishable at this
workload -- reported as ``baseline_retained`` on a screening (a candidate that did
not clear the bar has not earned its place) and as ``no_regression`` on a guard.

Interleaving is what makes the band mean anything: drift over the session hits
both modes over the set, so a gap smaller than each mode's own spread is a gap the
measurement cannot resolve.  The report always states both the medians and the
ranges, so a reader can see how far apart the two modes actually were rather than
only which side of the threshold they landed on.

Platform
--------
Windows only, because the surface is WGL: ``examples/windows-gl4`` opens a real
window through ``fluxel-host`` and this checker drives that fixture.  On any
other platform the screening is *skipped* with exit 0 rather than failed, since a
missing platform is not a defect in the thing being screened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys

# Every script in this directory is run both as a program and as a module by its
# test, which loads it by path; neither puts the directory itself on `sys.path`,
# so the one sibling import this file makes is set up here rather than left to
# whichever caller happened to be first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gl_raster_picture import extract_json  # noqa: E402  (after the path it needs)

DEFAULT_MANIFEST = Path("examples/windows-gl4/Cargo.toml")
DEFAULT_OUT = Path("target/gl-state-cache-screening")

# The workload the plan names as representative, and the one the counters were
# built to describe: one pass, this many resident indexed draws over the same
# pipeline and binding set.
DEFAULT_DRAWS = 2000

# Ten scored pairs after one discarded pair.  The warm-up pair is dropped rather
# than averaged in because the first run in a process pays for a cold driver, and
# a screening whose baseline carried that cost and whose candidate did not would
# measure the wrong thing.
DEFAULT_PAIRS = 10
DEFAULT_WARMUP_PAIRS = 1

# Fixed so a rerun reproduces the same order; the point of shuffling is that a
# monotonic drift over the session cannot line up with the mode, not that the
# order is secret.
DEFAULT_SEED = 1517

MODES = ("oracle", "optimized")


class ScreeningError(RuntimeError):
    """A run that produced no usable evidence, as opposed to a bad result."""


def percentile(sorted_values: list[int], fraction: float) -> int:
    """Nearest-rank percentile over an ascending list, without interpolation.

    Nearest rank rather than an interpolated one because every value here is a
    real measured round: a p95 that is the average of two rounds is a number no
    run ever produced, and this report's whole claim is that its numbers came off
    a clock.
    """
    if not sorted_values:
        raise ValueError("no values to take a percentile of")
    rank = max(1, round(fraction * len(sorted_values) + 0.5))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def summarize(values: list[int]) -> dict:
    """The five readings the funnel's step 1 asks a run to record."""
    ordered = sorted(values)
    return {
        "rounds": len(ordered),
        "min": ordered[0],
        "median": int(statistics.median(ordered)),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def run_fixture(manifest: Path, draws: int, mode: str, colour: Path,
                timeout: float) -> dict:
    """Runs one round over a real drawable and returns the report it printed.

    Each round is a fresh process because it has to be: ``SetPixelFormat`` may be
    called once per window, so a process gets exactly one WGL context over a given
    drawable and the mode is fixed when that context is adopted.  Interleaving the
    modes therefore means alternating processes, which is also what keeps a
    long-lived process's own warm-up from being attributed to a mode.
    """
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--offline",
            "--quiet",
            "--locked",
            "--manifest-path",
            str(manifest),
            "--",
            "--draws",
            str(draws),
            "--mode",
            mode,
            "--readback",
            str(colour),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise ScreeningError(
            f"the {mode} round exited {completed.returncode}: "
            f"{completed.stderr.strip()[-400:]}"
        )
    try:
        report = extract_json(completed.stdout)
    except ValueError as error:
        raise ScreeningError(f"the {mode} round printed no report: {error}") from error
    if not colour.exists():
        raise ScreeningError(f"the {mode} round did not write {colour}")
    report["_colour_sha256"] = hashlib.sha256(colour.read_bytes()).hexdigest()
    report["_colour_bytes"] = colour.stat().st_size
    return report


def workload_of(report: dict) -> dict:
    """The one nested object a driven run prints, or a stated refusal."""
    workload = report.get("workload")
    if not isinstance(workload, dict):
        raise ScreeningError(
            "a round printed no workload: a run that drove nothing cannot be screened"
        )
    return workload


def emitted(workload: dict) -> int:
    """Every domain's emitted calls, summed."""
    return sum(domain["emitted"] for domain in workload["domains"])


def skipped(workload: dict) -> int:
    """Every domain's skipped calls, summed."""
    return sum(domain["skipped"] for domain in workload["domains"])


def prediction_problems(rounds: list[dict], guard: bool) -> list[str]:
    """Whether the rounds are the comparison the caller says they are.

    Two-sided on purpose, and the two kinds of run are checked on opposite sides
    of the same fact.  A *screening* pair is only a screening if the candidate
    skipped work the baseline emitted -- a candidate that skipped nothing has not
    been applied, and a baseline that skipped anything is not an uncached baseline.
    A *guard* pair is only a guard if there was nothing to skip at all, which is
    the frame the candidate is supposed to cost nothing on.

    Either way the checks are per round rather than trusted from one pair: a
    single round that was not the run it claims to be would make the latency
    comparison below a comparison of the wrong two things.
    """
    problems = []
    for round_ in rounds:
        workload = round_["workload"]
        mode = round_["mode"]
        if round_["requested_mode"] != mode:
            problems.append(
                f"a round asked for {round_['requested_mode']} and reported {mode}"
            )
        if mode == "oracle":
            if skipped(workload) != 0:
                problems.append(
                    f"an oracle round skipped {skipped(workload)} call(s); "
                    "the baseline is defined as the uncached run"
                )
            if workload["cache_hits"] != 0:
                problems.append(
                    f"an oracle round reported {workload['cache_hits']} cache hit(s)"
                )
        elif guard:
            if skipped(workload) != 0:
                problems.append(
                    f"a guard round skipped {skipped(workload)} call(s), so there "
                    "was redundancy for the candidate to remove and this is a "
                    "screening round rather than a guard"
                )
        else:
            if skipped(workload) == 0:
                problems.append(
                    "an optimized round skipped nothing, so the cache was not "
                    "exercised and its cost is not what this round measured"
                )
    if guard:
        return problems
    optimized = [round_ for round_ in rounds if round_["mode"] == "optimized"]
    oracle = [round_ for round_ in rounds if round_["mode"] == "oracle"]
    if optimized and oracle:
        if min(emitted(r["workload"]) for r in optimized) >= max(
            emitted(r["workload"]) for r in oracle
        ):
            problems.append(
                "the optimized rounds did not emit fewer calls than every oracle "
                "round, so the prediction the candidate rests on is not visible"
            )
    return problems


def picture_problems(rounds: list[dict]) -> list[str]:
    """Whether every round rendered the same frame, both modes included."""
    digests = {round_["_colour_sha256"] for round_ in rounds}
    if len(digests) == 1:
        return []
    detail = ", ".join(
        sorted(
            f"{round_['mode']}#{round_['round']}={round_['_colour_sha256'][:12]}"
            for round_ in rounds
        )
    )
    return [
        "the rounds did not all render the same frame, so the two halves of the "
        f"differential are not comparable: {detail}"
    ]


def decide(rounds: list[dict], guard: bool) -> dict:
    """The funnel's decision, and the numbers behind it.

    Both metrics are reported for the reader; the decision uses ``submit_nanos``
    for the reason the module doc gives -- it is the frame's submission and
    nothing else, and the readback this checker asks for rides in ``total_nanos``
    alone.

    The arithmetic is the same for both kinds of run and only its reading differs.
    On a screening the candidate has to *clear* the baseline's best round to be
    retained; on a guard, where there is nothing to skip, the same comparison
    asked backwards is the question that matters -- whether the candidate's
    typical round is worse than the baseline's *worst*, which is the only way the
    bookkeeping can be said to cost more than the calls it saves.

    Two thresholds rather than one, and a band between them, because one threshold
    is the wrong shape for a guard: the first guard run put the candidate's median
    2.7% above the baseline's with the two ranges overlapping almost completely,
    and a rule that calls that a regression would be reporting the noise band as a
    finding.  Inside the band the answer is that the two modes are not
    distinguishable at this workload, which is a result; what is *not* claimed is
    that the candidate is faster.
    """
    stats = {
        mode: {
            "submit_nanos": summarize(
                [r["workload"]["submit_nanos"] for r in rounds if r["mode"] == mode]
            ),
            "total_nanos": summarize(
                [r["workload"]["total_nanos"] for r in rounds if r["mode"] == mode]
            ),
            "emitted": summarize(
                [emitted(r["workload"]) for r in rounds if r["mode"] == mode]
            ),
            "skipped": summarize(
                [skipped(r["workload"]) for r in rounds if r["mode"] == mode]
            ),
        }
        for mode in MODES
    }
    candidate = stats["optimized"]["submit_nanos"]
    baseline = stats["oracle"]["submit_nanos"]
    clears = candidate["median"] < baseline["min"]
    breaks = candidate["median"] > baseline["max"]
    return {
        "kind": "guard" if guard else "screening",
        "metric": "submit_nanos",
        "criterion": "candidate median outside the baseline's whole range, "
                     "on the side this kind of run asks about",
        "candidate_median_nanos": candidate["median"],
        "baseline_min_nanos": baseline["min"],
        "baseline_max_nanos": baseline["max"],
        "beyond_noise": clears or breaks,
        # Who was faster *typically*, which is a fact even when the gap is inside
        # the band and the verdict below declines to call it a difference.
        "winner": "optimized" if candidate["median"] < baseline["median"] else "oracle",
        # The same comparison, named for what it means in each kind of run: a
        # screening asks whether the candidate earned its place, a guard asks
        # whether it costs anything where it has nothing to earn.  Inside the band
        # both answers are "no": a candidate that did not clear the bar is not
        # retained, and one that did not break it has not regressed.
        "verdict": ("regression" if breaks else "no_regression") if guard
        else ("retained" if clears else "baseline_retained"),
        "stats": stats,
    }


def replay(report: dict) -> list[dict]:
    """Rebuilds the rounds a retained report recorded, in the shape ``decide`` reads.

    A report is durable evidence only if the decision inside it can be re-derived
    from the numbers inside it, so this is what re-adjudicating a committed report
    runs: everything ``decide`` reads is a field the report keeps, and nothing here
    consults the hardware again.  A report whose verdict was edited, or whose rows
    were collected under a rule that has since changed, fails when it is replayed
    -- which is the property that makes committing the JSON worth anything.
    """
    return [
        {
            "round": row["round"],
            # The mode was asked for and reported; a retained report records the
            # pair as one name, and `decide` reads the reported one.
            "requested_mode": row["mode"],
            "mode": row["mode"],
            "workload": {
                "mode": row["mode"],
                "submit_nanos": row["submit_nanos"],
                "total_nanos": row["total_nanos"],
                # Kept because the baseline's definition includes it: an oracle
                # round that hit its own cache is not an uncached baseline, and a
                # replay that dropped this field could not tell.
                "cache_hits": row["cache_hits"],
                # The report keeps the totals rather than the eight domain rows,
                # because the totals are what the statistics are taken over.  They
                # are restated here as one row so that `emitted` and `skipped` stay
                # the single definition of those sums instead of gaining a second
                # one for replayed reports; nothing downstream reads a domain name.
                "domains": [
                    {"domain": "replayed", "emitted": row["emitted"],
                     "skipped": row["skipped"]}
                ],
            },
            "_colour_sha256": row["colour_sha256"],
        }
        for row in report["rounds"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=None,
                        help=f"fixture manifest, default {DEFAULT_MANIFEST}")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--warmup-pairs", type=int, default=DEFAULT_WARMUP_PAIRS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--guard", action="store_true",
                        help="run the guard instead: the frame must have nothing to skip")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"where to keep the artifacts, default {DEFAULT_OUT}")
    arguments = parser.parse_args()

    if sys.platform != "win32":
        kind = "guard" if arguments.guard else "screening"
        print(f"the {kind} needs WGL, so it is skipped on {sys.platform}")
        return 0

    root = arguments.root.resolve()
    manifest = arguments.manifest or (root / DEFAULT_MANIFEST)
    out = arguments.out or (root / DEFAULT_OUT)
    rounds_dir = out / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)

    # One discarded pair first -- one run of each mode, so that neither mode is
    # the only one paying for a cold driver -- and then the scored rounds in a
    # shuffled order.  The warm-ups are not shuffled because they are discarded
    # either way, and the scored rounds are, because that is what stops a
    # monotonic drift over the session from lining up with a mode.
    warmup_order = list(MODES) * arguments.warmup_pairs
    scored_order = list(MODES) * arguments.pairs
    random.Random(arguments.seed).shuffle(scored_order)
    order = warmup_order + scored_order

    rounds: list[dict] = []
    warmups: list[dict] = []
    for index, mode in enumerate(order):
        scored = index >= len(warmup_order)
        colour = rounds_dir / f"{index:02d}-{mode}.rgba"
        try:
            report = run_fixture(manifest, arguments.draws, mode, colour, arguments.timeout)
            report["workload"] = workload_of(report)
            # The mode is lifted out of the workload because the rest of this
            # file asks about it by name, and a round whose reported mode did not
            # match the one asked for is a finding rather than a lookup failure:
            # the keyword is what was *requested*, and the workload object is what
            # the run says it actually used.
            report["requested_mode"] = mode
            report["mode"] = report["workload"]["mode"]
        except (ScreeningError, ValueError, subprocess.TimeoutExpired) as error:
            print(f"the screening produced no evidence: {error}", file=sys.stderr)
            return 1
        report["round"] = index
        (rounds if scored else warmups).append(report)

    problems = picture_problems(rounds) + prediction_problems(rounds, arguments.guard)
    if problems:
        print(
            f"the {'guard' if arguments.guard else 'screening'} is not the comparison "
            "it claims to be:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    decision = decide(rounds, arguments.guard)
    report = {
        "draws": arguments.draws,
        "pairs": arguments.pairs,
        "warmup_pairs": arguments.warmup_pairs,
        "seed": arguments.seed,
        "order": order,
        "scored_order": scored_order,
        "renderer": rounds[0].get("renderer"),
        "driver_or_browser": rounds[0].get("driver_or_browser"),
        "profile": rounds[0].get("profile"),
        "colour_sha256": rounds[0]["_colour_sha256"],
        "decision": decision,
        "warmups": [
            {
                "round": r["round"],
                "mode": r["mode"],
                "submit_nanos": r["workload"]["submit_nanos"],
                "total_nanos": r["workload"]["total_nanos"],
            }
            for r in warmups
        ],
        "rounds": [
            {
                "round": r["round"],
                "mode": r["mode"],
                "submit_nanos": r["workload"]["submit_nanos"],
                "total_nanos": r["workload"]["total_nanos"],
                "emitted": emitted(r["workload"]),
                "skipped": skipped(r["workload"]),
                "cache_hits": r["workload"]["cache_hits"],
                "cache_misses": r["workload"]["cache_misses"],
                "steady_state_allocations": r["workload"]["steady_state_allocations"],
                "colour_sha256": r["_colour_sha256"],
            }
            for r in rounds
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    stats = decision["stats"]
    print(f"artifacts in {out}")
    print(
        f"  {report['renderer']} ({report['driver_or_browser']}), profile {report['profile']}"
    )
    print(
        f"  {decision['kind']}: {arguments.draws} draws, {arguments.pairs} interleaved "
        f"pairs, seed {arguments.seed}, warm-up pairs dropped: {arguments.warmup_pairs}"
    )
    for mode in MODES:
        submit = stats[mode]["submit_nanos"]
        total = stats[mode]["total_nanos"]
        print(
            f"  {mode:<9} submit_nanos median {submit['median']:>10} "
            f"p95 {submit['p95']:>10} max {submit['max']:>10} min {submit['min']:>10}"
        )
        print(
            f"  {'':<9} total_nanos  median {total['median']:>10}  "
            f"emitted {stats[mode]['emitted']['median']:>6} "
            f"skipped {stats[mode]['skipped']['median']:>6}"
        )
    print(
        f"\nthe frame is identical across all {len(rounds)} scored rounds: "
        f"{report['colour_sha256'][:16]}"
    )
    print(
        f"\ndecision: candidate median {decision['candidate_median_nanos']} against the "
        f"baseline's {decision['baseline_min_nanos']}..{decision['baseline_max_nanos']} -- "
        f"{'outside' if decision['beyond_noise'] else 'inside'} the noise band"
    )
    print(f"verdict: {decision['verdict']}, typically faster: {decision['winner']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
