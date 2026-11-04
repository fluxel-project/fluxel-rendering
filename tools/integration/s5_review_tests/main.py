"""End-to-end tests for the S5 review command using its real evidence paths."""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from .fixture import build


def _review():
    path = Path(__file__).parents[1] / "s5_review.py"
    spec = importlib.util.spec_from_file_location("s5_review_fixture_target", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class S5ReviewMainTests(unittest.TestCase):
    def _run(self, review, fixture: dict) -> tuple[int, dict]:
        argv = ["s5_review.py", "--human-review", str(fixture["human"]), "--output", str(fixture["output"]), *fixture["args"]]
        with mock.patch.object(review, "ROOT", fixture["source"]), mock.patch.object(sys, "argv", argv):
            result = review.main()
        return result, json.loads(Path(fixture["output"]).read_text(encoding="utf-8"))

    def test_main_accepts_complete_six_attempt_ledger(self) -> None:
        review = _review()
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build(Path(temporary), review); result, report = self._run(review, fixture)
            self.assertEqual(result, 0); self.assertEqual(report["evidence_status"], "verified")
            self.assertEqual(len(report["checks"]), 12)

    def test_main_rejects_tampered_evidence(self) -> None:
        review = _review()
        for name in ("command_exit", "diagnostic", "source_hash", "required_raw", "input", "actual"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = build(Path(temporary), review); attempt, closure, run = fixture["attempts"][0]
                if name == "command_exit": attempt["commands"][0]["result"]["exit_status"] = 1
                elif name == "diagnostic": attempt["validation"]["native_diagnostic_count"] = 1
                elif name == "source_hash": attempt["source_inputs"][0]["sha256"] = "forged"
                elif name == "required_raw": (run / "s5-forward-geometry-0.txt").unlink()
                elif name == "input": (run / "s5-forward-frame-1-input.json").write_text("{}", encoding="utf-8")
                else: (run / "s5-forward-readback.rgba").write_bytes(bytes(96 * 96 * 4))
                _write = (closure / "manifest.json").write_text
                _write(json.dumps({"schema_version": 4, "stage": "S5 Forward integration evidence", "attempts": [attempt]}), encoding="utf-8")
                result, report = self._run(review, fixture)
                self.assertEqual(result, 1); self.assertEqual(report["evidence_status"], "open")
