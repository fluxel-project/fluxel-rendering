from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).parents[1] / "check_gl4_raster_readback.py"
SPEC = importlib.util.spec_from_file_location("check_gl4_raster_readback", SCRIPT)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)

# The report a real context produced while driving the workload -- an AMD Radeon
# 780M through the desktop driver, not through an emulation layer -- kept so that
# a machine without a GPU can still run this gate's logic and so that the passing
# case is hardware's answer rather than one written to pass.
HARDWARE_REPORT = Path(__file__).parent / "data/gl4_raster_readback_radeon_780m.json"


def hardware() -> tuple[dict, bytes]:
    """Returns the hardware report and the bytes its own hex grid stands for.

    The bytes are the report's, and that is the point rather than a shortcut: the
    gate requires the two statements of the picture to agree, so a fixture whose
    file disagreed with its grid would be testing a run that failed for a reason
    no expected picture can diagnose.
    """
    report = json.loads(HARDWARE_REPORT.read_text(encoding="utf-8"))
    return report, CHECKER.pixels_from_hex(report["workload"]["colour"]["pixels_rgba8"])


def flip_rows(raw: bytes) -> bytes:
    """Returns the same picture with its rows reversed, which is the trap.

    A readback whose rows came back top-down would carry the same bytes in a
    different order for this workload's geometry only because the covered pixels
    sit one row above the region's origin -- which is exactly why the gate has to
    assert where they are rather than only that some pixel is green.
    """
    extent = CHECKER.EXPECTED_EXTENT
    rows = [raw[row * extent[0] * 4 : (row + 1) * extent[0] * 4] for row in range(extent[1])]
    return b"".join(reversed(rows))


class CheckGl4RasterReadbackTests(unittest.TestCase):
    def test_the_hardware_frame_is_the_picture_the_workload_requires(self) -> None:
        report, raw = hardware()
        self.assertEqual(CHECKER.check(report, raw), [])

    def test_a_run_with_no_frame_is_refused(self) -> None:
        report, raw = hardware()
        del report["workload"]
        problems = CHECKER.check(report, raw)
        self.assertEqual(len(problems), 1)
        self.assertIn("no frame was driven", problems[0])

    def test_a_run_that_read_no_picture_is_refused(self) -> None:
        report, raw = hardware()
        del report["workload"]["colour"]
        problems = CHECKER.check(report, raw)
        self.assertEqual(len(problems), 1)
        self.assertIn("no colour target", problems[0])

    def test_a_report_that_disagrees_with_its_own_file_is_refused(self) -> None:
        report, raw = hardware()
        # The two statements of one picture are checked against each other before
        # any expected value is consulted, so this failure is reported alone.
        problems = CHECKER.check(report, flip_rows(raw))
        self.assertEqual(len(problems), 1)
        self.assertIn("different pictures", problems[0])

    def test_a_frame_whose_rows_were_flipped_is_refused(self) -> None:
        report, raw = hardware()
        flipped = flip_rows(raw)
        # Told to state what it produced, so the grid and the file agree and the
        # only thing left wrong is where the covered pixels are.
        colour = report["workload"]["colour"]
        colour["pixels_rgba8"] = [
            flipped[offset : offset + 4].hex()
            for offset in range(0, len(flipped), 4)
        ]
        problems = CHECKER.check(report, flipped)
        self.assertTrue(
            any("row 1, column 1" in problem for problem in problems),
            f"the covered pixel is looked for where the geometry puts it: {problems}",
        )
        self.assertTrue(
            any("row 2, column 1" in problem and "the clear" in problem for problem in problems),
            f"and the row it was moved to is not allowed to carry it: {problems}",
        )

    def test_a_frame_that_was_never_drawn_into_is_refused(self) -> None:
        report, raw = hardware()
        blank = bytes(CHECKER.EXPECTED_CLEAR) * (len(raw) // 4)
        colour = report["workload"]["colour"]
        colour["pixels_rgba8"] = [
            blank[offset : offset + 4].hex() for offset in range(0, len(blank), 4)
        ]
        problems = CHECKER.check(report, blank)
        self.assertTrue(
            any("kernel's colour" in problem for problem in problems),
            f"a frame that is the clear and nothing else is the failure this gate exists for: "
            f"{problems}",
        )

    def test_a_truncated_file_is_refused(self) -> None:
        report, raw = hardware()
        problems = CHECKER.check(report, raw[:-4])
        self.assertTrue(
            any("the file holds" in problem for problem in problems),
            f"a short file is caught before any pixel is read: {problems}",
        )

    def test_a_wrong_row_order_statement_is_refused(self) -> None:
        report, raw = hardware()
        report["workload"]["colour"]["row_order"] = "top-left"
        problems = CHECKER.check(report, raw)
        self.assertTrue(
            any("row order" in problem for problem in problems),
            f"the consumer that flips the image is told which order it has: {problems}",
        )

    def test_the_magnified_png_is_top_down(self) -> None:
        _, raw = hardware()
        width, height, rows = CHECKER.magnify(raw, CHECKER.EXPECTED_EXTENT, 2)
        self.assertEqual((width, height), (8, 8))
        # The covered pixels are in the second row from the region's own origin,
        # which is the second row from the *bottom* of the family's bytes and so
        # the third of four rows from the top of the reversed image -- magnified
        # rows four and five, with the pair starting at byte six -- one magnified
        # pixel is three bytes repeated, so column one begins there.
        covered = bytes(CHECKER.EXPECTED_COLOUR[:3]) * 2 * 2
        self.assertEqual(rows[4][6:18], covered)
        self.assertEqual(rows[5][6:18], covered)
        self.assertEqual(rows[2], bytes(CHECKER.EXPECTED_CLEAR[:3]) * 8)
        self.assertEqual(rows[0], bytes(CHECKER.EXPECTED_CLEAR[:3]) * 8)


if __name__ == "__main__":
    unittest.main()
