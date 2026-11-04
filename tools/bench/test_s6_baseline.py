"""Focused S6 baseline fact-contract tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("s6_baseline", Path(__file__).with_name("s6_baseline.py"))
assert SPEC and SPEC.loader
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


def offscreen_fixture(run: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "s6_offscreen_fixture", Path(baseline.INTEGRATION_TOOLS) / "test_s6_offscreen.py"
    )
    assert spec and spec.loader
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    fixture_module.S6OffscreenTests().fixture(run)


class S6BaselineTests(unittest.TestCase):
    def measurement(self) -> dict[str, int]:
        return {field: 0 for field in baseline.REQUIRED_MEASUREMENT_FACTS} | {"acceptedSubmissions": 11}

    def test_missing_measurement_fact_rejects_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            path = run / "s6-composed"
            path.mkdir()
            (path / "s6-composed-measurements.json").write_text(json.dumps({}), encoding="utf-8")
            with mock.patch.object(baseline.offscreen, "review", return_value=[]):
                with self.assertRaisesRegex(ValueError, "missing integer fields"):
                    baseline.facts(run)

    def test_review_failure_rejects_sample_before_measurement_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with mock.patch.object(baseline.offscreen, "review", return_value=["tampered input"]):
                with self.assertRaisesRegex(ValueError, "tampered input"):
                    baseline.facts(run)

    def test_non_eleven_submission_fact_rejects_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            path = run / "s6-composed"
            path.mkdir()
            measurement = self.measurement() | {"acceptedSubmissions": 10}
            (path / "s6-composed-measurements.json").write_text(json.dumps(measurement), encoding="utf-8")
            with mock.patch.object(baseline.offscreen, "review", return_value=[]):
                with self.assertRaisesRegex(ValueError, "exactly eleven"):
                    baseline.facts(run)

    def test_real_offscreen_fixture_is_accepted_from_process_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            offscreen_fixture(root / "s6-composed")
            self.assertEqual(baseline.facts(root)["plan_compilations"], 1)

    def test_real_readback_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "s6-composed"
            offscreen_fixture(run)
            actual = bytearray((run / "s6-frame-0-actual.rgba").read_bytes())
            actual[0] ^= 1
            (run / "s6-frame-0-actual.rgba").write_bytes(actual)
            with self.assertRaisesRegex(ValueError, "offscreen review failed"):
                baseline.facts(root)


if __name__ == "__main__":
    unittest.main()
