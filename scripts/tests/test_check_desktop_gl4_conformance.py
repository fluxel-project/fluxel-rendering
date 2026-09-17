from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).parents[1] / "check_desktop_gl4_conformance.py"
SPEC = importlib.util.spec_from_file_location("check_desktop_gl4_conformance", SCRIPT)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)

# The report a real context produced, kept so that a machine without a GPU can
# still run this gate's logic and so that the passing case is hardware's answer
# rather than one written to pass.
HARDWARE_REPORT = Path(__file__).parent / "data/desktop_gl4_radeon_780m.json"


def set_capability(report: dict, name: str, value: bool) -> None:
    for row in report["capabilities"]:
        if row[0] == name:
            row[1] = value
            return
    report["capabilities"].append([name, value])


def set_limit(report: dict, name: str, value: str) -> None:
    for row in report["limits"]:
        if row[0] == name:
            row[1] = value
            return
    report["limits"].append([name, value])


class CheckDesktopGl4ConformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report = json.loads(HARDWARE_REPORT.read_text(encoding="utf-8"))

    def mutate(self, **changes):
        report = copy.deepcopy(self.report)
        for path, value in changes.items():
            head, _, tail = path.partition("__")
            if head == "capability":
                set_capability(report, tail, value)
            elif head == "limit":
                set_limit(report, tail, value)
            else:
                report[path] = value
        return CHECKER.check(report)

    def test_the_retained_hardware_report_passes(self) -> None:
        self.assertEqual(CHECKER.check(self.report), [])
        self.assertEqual(CHECKER.check(self.report, (640, 480)), [])

    def test_the_three_rows_this_gate_was_written_for_are_required(self) -> None:
        # The probes that asked for a draw and a dispatch with no program
        # current.  If any of these stops being required, this gate stops
        # catching the defect that motivated it.
        for name in ("compute", "indirect-draw", "indirect-dispatch"):
            with self.subTest(capability=name):
                problems = self.mutate(**{f"capability__{name}": False})
                self.assertTrue(
                    any(name in problem for problem in problems),
                    f"{name} false was accepted: {problems}",
                )

    def test_every_required_capability_is_actually_required(self) -> None:
        for name in CHECKER.REQUIRED_CAPABILITIES:
            with self.subTest(capability=name):
                problems = self.mutate(**{f"capability__{name}": False})
                self.assertTrue(problems, f"{name} false was accepted")
        self.assertEqual(
            set(CHECKER.REQUIRED_CAPABILITIES),
            {"compute", "storage-buffer", "storage-image", "indirect-draw",
             "indirect-dispatch", "timer-query"},
        )

    def test_a_capability_absent_from_the_report_is_not_a_pass(self) -> None:
        report = copy.deepcopy(self.report)
        report["capabilities"] = [
            row for row in report["capabilities"] if row[0] != "compute"
        ]
        problems = CHECKER.check(report)
        self.assertTrue(any("compute" in problem for problem in problems), problems)

    def test_a_recorded_open_row_that_changes_fails(self) -> None:
        for name, (recorded, row) in CHECKER.RECORDED_OPEN.items():
            with self.subTest(capability=name):
                problems = self.mutate(**{f"capability__{name}": not recorded})
                self.assertTrue(
                    any(row in problem for problem in problems),
                    f"{name} changing was accepted without naming {row}: {problems}",
                )

    def test_a_context_that_did_not_open_fails(self) -> None:
        self.assertTrue(self.mutate(opened=False))

    def test_a_profile_below_the_desktop_core_floor_fails(self) -> None:
        problems = self.mutate(profile="Desktop { major: 3, minor: 3 }")
        self.assertTrue(any("floor" in problem for problem in problems), problems)

    def test_a_non_desktop_profile_fails(self) -> None:
        self.assertTrue(self.mutate(profile="WebGl2"))

    def test_a_profile_without_a_version_fails(self) -> None:
        self.assertTrue(self.mutate(profile="Desktop"))

    def test_missing_required_facts_fail(self) -> None:
        for name in (*CHECKER.REQUIRED_SCALARS, *CHECKER.REQUIRED_LISTS, *CHECKER.REQUIRED_FLAGS):
            with self.subTest(field=name):
                report = copy.deepcopy(self.report)
                del report[name]
                self.assertTrue(CHECKER.check(report), f"a missing {name} was accepted")

    def test_a_short_extension_list_fails(self) -> None:
        problems = self.mutate(
            reported_extensions=["GL_ARB_compute_shader"], reported_extension_count=1
        )
        self.assertTrue(any("below the" in problem for problem in problems), problems)

    def test_a_short_typed_extension_ledger_fails(self) -> None:
        problems = self.mutate(typed_extensions=[["GL_ARB_compute_shader", "probed"]])
        self.assertTrue(any("typed" in problem for problem in problems), problems)

    def test_a_count_that_disagrees_with_the_list_fails(self) -> None:
        problems = self.mutate(reported_extension_count=7)
        self.assertTrue(any("the list holds" in problem for problem in problems), problems)

    def test_an_alignment_that_is_not_a_modulus_fails(self) -> None:
        for value in ("0", "not a number"):
            with self.subTest(value=value):
                problems = self.mutate(limit__uniform_buffer_offset_alignment=value)
                self.assertTrue(
                    any("uniform_buffer_offset_alignment" in problem for problem in problems),
                    problems,
                )

    def test_the_desktop_texture_floor_is_enforced(self) -> None:
        problems = self.mutate(limit__max_texture_size="2048")
        self.assertTrue(any("desktop floor" in problem for problem in problems), problems)

    def test_a_missing_limit_row_fails(self) -> None:
        report = copy.deepcopy(self.report)
        report["limits"] = [
            row for row in report["limits"] if row[0] != "max_multiview_view_count"
        ]
        problems = CHECKER.check(report)
        self.assertTrue(any("max_multiview_view_count" in p for p in problems), problems)

    def test_the_probe_bound_surface_facts_are_recorded_not_blessed(self) -> None:
        problems = self.mutate(surface_facts="default_framebuffer")
        self.assertTrue(
            any(CHECKER.RECORDED_SURFACE_FACTS[1] in problem for problem in problems), problems
        )

    def test_the_requested_extent_is_compared_when_one_is_given(self) -> None:
        self.assertTrue(CHECKER.check(self.report, (800, 600)))
        self.assertEqual(CHECKER.check(self.report, (640, 480)), [])

    def test_a_malformed_row_fails_instead_of_being_skipped(self) -> None:
        report = copy.deepcopy(self.report)
        report["capabilities"] = [["compute"], "storage-buffer"]
        problems = CHECKER.check(report)
        self.assertTrue(any("not a pair" in problem for problem in problems), problems)

    def test_extract_json_takes_the_object_out_of_noise(self) -> None:
        stdout = '   Compiling something\n{"opened": true}\n    Finished\n'
        self.assertEqual(CHECKER.extract_json(stdout), {"opened": True})

    def test_extract_json_refuses_a_stream_without_an_object(self) -> None:
        with self.assertRaises(ValueError):
            CHECKER.extract_json("no object here")
        with self.assertRaises(ValueError):
            CHECKER.extract_json("")


if __name__ == "__main__":
    unittest.main()
