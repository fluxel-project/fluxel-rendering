from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).parents[1] / "check_gl_state_cache_screening.py"
SPEC = importlib.util.spec_from_file_location("check_gl_state_cache_screening", SCRIPT)
assert SPEC and SPEC.loader
SCREENING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCREENING
SPEC.loader.exec_module(SCREENING)

# The workload block a real context produced, kept so that the rounds this test
# feeds the screening are shaped like a run rather than like a test's idea of
# one: every domain list, counter and duration below comes from an AMD Radeon
# 780M through the desktop driver.
HARDWARE_REPORT = Path(__file__).parent / "data/gl4_raster_readback_radeon_780m.json"

# The two runs this machine actually produced, kept the way the GL-family
# conformance reports are kept: the numbers are hardware's answer, so a machine
# with no GPU can still run this gate's logic against a real reading rather than
# against one written to pass.
SCREENING_REPORT = Path(__file__).parent / "data/gl_state_cache_screening_radeon_780m.json"
GUARD_REPORT = Path(__file__).parent / "data/gl_state_cache_guard_1draw_radeon_780m.json"

# Two sha256s that differ in one nibble, so that a test saying "these rounds
# rendered different frames" is saying it about two distinct digests rather than
# about a missing field.
FRAME = "0d18605583c84d0f98752488f8eacd189567be721076e7e6e486ba0006f75276"
OTHER_FRAME = "1d18605583c84d0f98752488f8eacd189567be721076e7e6e486ba0006f75276"


def hardware_workload() -> dict:
    """The real workload block, as one driven run reported it."""
    report = json.loads(HARDWARE_REPORT.read_text(encoding="utf-8"))
    return copy.deepcopy(report["workload"])


def round_(mode: str, submit_nanos: int, skipped: int, emitted: int,
           cache_hits: int = 0, digest: str = FRAME, index: int = 0) -> dict:
    """One scored round, shaped the way ``main`` hands it to the checks.

    The per-domain numbers are collapsed onto the session domain, which is what
    makes ``emitted`` and ``skipped`` the totals the checks read: a round whose
    totals came from eight domains or from one is the same round to everything
    downstream of the sum.
    """
    workload = hardware_workload()
    workload["mode"] = mode
    workload["submit_nanos"] = submit_nanos
    workload["total_nanos"] = submit_nanos * 3
    workload["cache_hits"] = cache_hits
    for domain in workload["domains"]:
        domain["emitted"] = 0
        domain["skipped"] = 0
    workload["domains"][0]["emitted"] = emitted
    workload["domains"][0]["skipped"] = skipped
    return {
        "round": index,
        "requested_mode": mode,
        "mode": mode,
        "workload": workload,
        "_colour_sha256": digest,
    }


def pair(submit_nanos: int, index: int) -> list[dict]:
    """One adjacent baseline/candidate pair with the same frame and prediction."""
    return [
        round_("oracle", submit_nanos, skipped=0, emitted=12000, index=index),
        round_("optimized", submit_nanos, skipped=11990, emitted=10,
               cache_hits=1999, index=index),
    ]


class PercentileTests(unittest.TestCase):
    def test_nearest_rank_returns_a_value_a_round_actually_produced(self) -> None:
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.assertEqual(SCREENING.percentile(values, 0.95), 100)
        self.assertEqual(SCREENING.percentile(values, 0.5), 60)

    def test_a_single_round_is_its_own_percentile(self) -> None:
        self.assertEqual(SCREENING.percentile([7], 0.95), 7)

    def test_no_rounds_is_refused_rather_than_answered(self) -> None:
        with self.assertRaises(ValueError):
            SCREENING.percentile([], 0.5)


class SummarizeTests(unittest.TestCase):
    def test_the_five_readings_come_off_the_rounds(self) -> None:
        self.assertEqual(
            SCREENING.summarize([5, 1, 9, 3, 7]),
            {"rounds": 5, "min": 1, "median": 5, "p95": 9, "max": 9},
        )


class PictureTests(unittest.TestCase):
    def test_one_frame_across_every_round_is_no_problem(self) -> None:
        rounds = pair(1000, 0) + pair(900, 1)
        self.assertEqual(SCREENING.picture_problems(rounds), [])

    def test_a_mode_that_rendered_a_different_frame_is_refused(self) -> None:
        rounds = pair(1000, 0)
        rounds[1]["_colour_sha256"] = OTHER_FRAME
        problems = SCREENING.picture_problems(rounds)
        self.assertEqual(len(problems), 1)
        self.assertIn("not comparable", problems[0])


class PredictionTests(unittest.TestCase):
    def test_a_baseline_that_skipped_nothing_and_a_candidate_that_skipped_is_quiet(
        self,
    ) -> None:
        self.assertEqual(
            SCREENING.prediction_problems(pair(1000, 0) + pair(900, 1), guard=False), []
        )

    def test_a_baseline_that_skipped_anything_is_not_a_baseline(self) -> None:
        rounds = pair(1000, 0)
        rounds[0]["workload"]["domains"][0]["skipped"] = 1
        problems = SCREENING.prediction_problems(rounds, guard=False)
        self.assertTrue(any("uncached run" in problem for problem in problems))

    def test_a_baseline_that_hit_its_own_cache_is_not_a_baseline(self) -> None:
        rounds = pair(1000, 0)
        rounds[0]["workload"]["cache_hits"] = 1
        problems = SCREENING.prediction_problems(rounds, guard=False)
        self.assertTrue(any("cache hit" in problem for problem in problems))

    def test_a_candidate_that_skipped_nothing_was_not_exercised(self) -> None:
        rounds = pair(1000, 0)
        rounds[1]["workload"]["domains"][0]["skipped"] = 0
        problems = SCREENING.prediction_problems(rounds, guard=False)
        self.assertTrue(any("skipped nothing" in problem for problem in problems))

    def test_a_round_whose_mode_is_not_the_one_asked_for_is_refused(self) -> None:
        rounds = pair(1000, 0)
        rounds[0]["workload"]["mode"] = "optimized"
        rounds[0]["mode"] = "optimized"
        problems = SCREENING.prediction_problems(rounds, guard=False)
        self.assertTrue(any("asked for oracle" in problem for problem in problems))

    def test_a_candidate_that_emits_no_fewer_calls_has_not_shown_its_prediction(
        self,
    ) -> None:
        rounds = pair(1000, 0)
        rounds[1]["workload"]["domains"][0]["emitted"] = 12000
        problems = SCREENING.prediction_problems(rounds, guard=False)
        self.assertTrue(any("emit fewer calls" in problem for problem in problems))


class GuardTests(unittest.TestCase):
    """The guard is the screening's checks read on the other side of the same fact."""

    def guard_pair(self, submit_nanos: int, index: int = 0) -> list[dict]:
        return [
            round_("oracle", submit_nanos, skipped=0, emitted=9, index=index),
            round_("optimized", submit_nanos, skipped=0, emitted=8, cache_hits=0,
                   index=index),
        ]

    def test_a_frame_with_nothing_to_skip_is_the_guard(self) -> None:
        self.assertEqual(
            SCREENING.prediction_problems(self.guard_pair(1000), guard=True), []
        )

    def test_a_guard_round_that_skipped_something_was_a_screening(self) -> None:
        rounds = self.guard_pair(1000)
        rounds[1]["workload"]["domains"][0]["skipped"] = 1
        problems = SCREENING.prediction_problems(rounds, guard=True)
        self.assertTrue(any("rather than a guard" in problem for problem in problems))

    def test_the_screening_refuses_the_frame_the_guard_requires(self) -> None:
        # The one case that says the two kinds of run are actually distinguished:
        # the same rounds are a clean guard and a refused screening.
        rounds = self.guard_pair(1000)
        self.assertEqual(SCREENING.prediction_problems(rounds, guard=True), [])
        self.assertNotEqual(SCREENING.prediction_problems(rounds, guard=False), [])

    def test_a_candidate_that_costs_more_with_nothing_to_skip_is_a_regression(self) -> None:
        rounds = [round_("oracle", value, 0, 9, index=index)
                  for index, value in enumerate([1000, 2000, 3000])]
        rounds += [round_("optimized", value, 0, 8, index=index)
                   for index, value in enumerate([4000, 5000, 6000])]
        decision = SCREENING.decide(rounds, guard=True)
        self.assertEqual(decision["kind"], "guard")
        self.assertEqual(decision["verdict"], "regression")
        self.assertEqual(decision["winner"], "oracle")

    def test_a_candidate_that_does_not_cost_more_is_no_regression(self) -> None:
        rounds = [round_("oracle", value, 0, 9, index=index)
                  for index, value in enumerate([3000, 5000, 7000])]
        rounds += [round_("optimized", value, 0, 8, index=index)
                   for index, value in enumerate([1000, 2000, 2900])]
        decision = SCREENING.decide(rounds, guard=True)
        self.assertEqual(decision["verdict"], "no_regression")
        self.assertEqual(decision["winner"], "optimized")

    def test_a_candidate_inside_the_bands_overlap_is_not_called_a_regression(self) -> None:
        # The case the first guard run actually produced: the candidate's median
        # sits above the baseline's median and above its minimum, but the two
        # ranges overlap almost completely.  Reporting that as a regression would
        # be reporting the noise band as a finding, which is what the upper
        # threshold exists to refuse.
        rounds = [round_("oracle", value, 0, 9, index=index)
                  for index, value in enumerate([1000, 2000, 3000])]
        rounds += [round_("optimized", value, 0, 8, index=index)
                   for index, value in enumerate([2100, 2900, 3100])]
        decision = SCREENING.decide(rounds, guard=True)
        self.assertFalse(decision["beyond_noise"])
        self.assertEqual(decision["verdict"], "no_regression")
        # The typical winner is still reported, because it is a fact about the
        # rounds even where the verdict declines to call it a difference.
        self.assertEqual(decision["winner"], "oracle")

    def test_the_same_rounded_band_is_not_a_retention_either(self) -> None:
        rounds = [round_("oracle", value, 0, 9, index=index)
                  for index, value in enumerate([3000, 4000, 5000])]
        rounds += [round_("optimized", value, 11990, 10, cache_hits=1999, index=index)
                   for index, value in enumerate([3100, 3900, 4900])]
        decision = SCREENING.decide(rounds, guard=False)
        self.assertFalse(decision["beyond_noise"])
        self.assertEqual(decision["verdict"], "baseline_retained")
        self.assertEqual(decision["winner"], "optimized")


class DecisionTests(unittest.TestCase):
    def test_a_candidate_whose_typical_round_beats_the_baselines_best_is_retained(
        self,
    ) -> None:
        rounds = [round_("oracle", value, 0, 12000, index=index)
                  for index, value in enumerate([3000, 5000, 7000])]
        rounds += [round_("optimized", value, 11990, 10, cache_hits=1999, index=index)
                   for index, value in enumerate([1000, 2000, 2900])]
        decision = SCREENING.decide(rounds, guard=False)
        self.assertEqual(decision["winner"], "optimized")
        self.assertEqual(decision["candidate_median_nanos"], 2000)
        self.assertEqual(decision["baseline_min_nanos"], 3000)

    def test_a_candidate_inside_the_noise_band_leaves_the_baseline_standing(self) -> None:
        rounds = [round_("oracle", value, 0, 12000, index=index)
                  for index, value in enumerate([1000, 5000, 7000])]
        rounds += [round_("optimized", value, 11990, 10, cache_hits=1999, index=index)
                   for index, value in enumerate([4000, 6000, 8000])]
        decision = SCREENING.decide(rounds, guard=False)
        self.assertEqual(decision["verdict"], "baseline_retained")
        self.assertFalse(decision["beyond_noise"])

    def test_a_tie_is_not_an_improvement(self) -> None:
        rounds = [round_("oracle", 1000, 0, 12000, index=0)]
        rounds += [round_("optimized", 1000, 11990, 10, cache_hits=1999, index=1)]
        decision = SCREENING.decide(rounds, guard=False)
        self.assertEqual(decision["winner"], "oracle")
        self.assertEqual(decision["verdict"], "baseline_retained")
        self.assertEqual(decision["candidate_median_nanos"], decision["baseline_min_nanos"])

    def test_the_decision_metric_is_the_submission_and_not_the_whole_run(self) -> None:
        rounds = [round_("oracle", 1000, 0, 12000, index=0)]
        candidate = round_("optimized", 900, 11990, 10, cache_hits=1999, index=1)
        # The readback this checker asks for rides in `total_nanos` alone, so a
        # candidate whose whole run is slower but whose submission is faster is
        # still the winner -- and this is the case that says the report knows it.
        candidate["workload"]["total_nanos"] = 5000
        rounds.append(candidate)
        decision = SCREENING.decide(rounds, guard=False)
        self.assertEqual(decision["metric"], "submit_nanos")
        self.assertEqual(decision["winner"], "optimized")
        self.assertLess(
            decision["stats"]["oracle"]["total_nanos"]["median"],
            decision["stats"]["optimized"]["total_nanos"]["median"],
        )


class RetainedReportTests(unittest.TestCase):
    """The committed reports have to re-derive the verdicts they record.

    This is the half of the gate that runs on a machine with no GPU: the evidence
    is a real 780M reading, and what is checked here is that the decision printed
    beside those numbers is the one the numbers still give.
    """

    def replay(self, path: Path, guard: bool) -> tuple[dict, dict]:
        report = json.loads(path.read_text(encoding="utf-8"))
        return report, SCREENING.decide(SCREENING.replay(report), guard=guard)

    def test_the_screening_re_derives_its_own_decision(self) -> None:
        report, decision = self.replay(SCREENING_REPORT, guard=False)
        self.assertEqual(decision, report["decision"])

    def test_the_guard_re_derives_its_own_decision(self) -> None:
        report, decision = self.replay(GUARD_REPORT, guard=True)
        self.assertEqual(decision, report["decision"])

    def test_the_screening_retained_the_candidate_beyond_the_noise(self) -> None:
        report, decision = self.replay(SCREENING_REPORT, guard=False)
        self.assertEqual(decision["verdict"], "retained")
        # The claim is not "faster on average": it is that the candidate's typical
        # round beat the baseline's best one, which is what the two thresholds are
        # for and what a report edited to say "retained" would have to satisfy.
        self.assertLess(decision["candidate_median_nanos"], decision["baseline_min_nanos"])

    def test_the_guard_found_no_material_cost(self) -> None:
        report, decision = self.replay(GUARD_REPORT, guard=True)
        self.assertEqual(decision["verdict"], "no_regression")
        self.assertFalse(decision["beyond_noise"])

    def test_the_replayed_rounds_keep_the_frame_and_the_prediction(self) -> None:
        report = json.loads(SCREENING_REPORT.read_text(encoding="utf-8"))
        rounds = SCREENING.replay(report)
        self.assertEqual(SCREENING.picture_problems(rounds), [])
        self.assertEqual(SCREENING.prediction_problems(rounds, guard=False), [])

    def test_the_replayed_guard_rounds_skipped_nothing(self) -> None:
        report = json.loads(GUARD_REPORT.read_text(encoding="utf-8"))
        rounds = SCREENING.replay(report)
        self.assertEqual(SCREENING.prediction_problems(rounds, guard=True), [])


if __name__ == "__main__":
    unittest.main()
