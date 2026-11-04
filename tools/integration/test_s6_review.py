"""Exercise the complete S6 reviewer entry point and evidence provenance."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import s6_review
from s6_review_parts import common, oracle
from s6_review_parts.ledger_fixture import build, write_manifest
import test_s6_offscreen
from test_s6_surface import fixture as surface_fixture


class S6ReviewTests(unittest.TestCase):
    def fixture(self, root: Path) -> dict:
        return build(root, s6_review.base, test_s6_offscreen.S6OffscreenTests().fixture, surface_fixture)

    def run_review(self, fixture: dict) -> tuple[int, dict]:
        argv = ["s6_review.py", "--visual-review", str(fixture["visual"]),
                "--output", str(fixture["output"]), *fixture["args"]]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(s6_review.base, "ROOT", fixture["source"]):
            result = s6_review.main()
        return result, json.loads(fixture["output"].read_text(encoding="utf-8"))

    def test_complete_matrix_runs_actual_review_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            result, report = self.run_review(fixture)
            self.assertEqual(result, 0, report)
            self.assertEqual(report["evidence_status"], "verified")
            self.assertEqual(report["stage_closure_decision"], "not_made_by_this_tool")

    def test_rejects_matrix_provenance_and_visual_tampering(self) -> None:
        for mutation in ("source", "command", "raw", "input", "visual", "tests", "matrix", "schema"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(Path(temporary))
                attempt, closure, run = fixture["attempts"][0]
                if mutation == "source":
                    attempt["source_inputs"][0]["sha256"] = "forged"
                elif mutation == "command":
                    attempt["commands"][0]["result"]["exit_status"] = 1
                elif mutation == "raw":
                    (run / "s6-frame-0-actual.rgba").write_bytes(bytes(96 * 96 * 4))
                elif mutation == "input":
                    (run / "s6-frame-0-input.json").write_text("{}", encoding="utf-8")
                elif mutation == "visual":
                    visual = json.loads(fixture["visual"].read_text(encoding="utf-8"))
                    # A valid hash from a different scenario is not evidence for this frame.
                    frames = visual["evidence"][0]["frames"]
                    frames[0]["raw"] = copy.deepcopy(frames[1]["raw"])
                    fixture["visual"].write_text(json.dumps(visual), encoding="utf-8")
                elif mutation == "tests":
                    attempt["commands"] = [item for item in attempt["commands"] if item["kind"] != "test"]
                elif mutation == "matrix":
                    fixture["args"] = fixture["args"][:-5]
                write_manifest(closure, attempt)
                if mutation == "schema":
                    value = json.loads((closure / "manifest.json").read_text(encoding="utf-8"))
                    value["schema_version"] = 4
                    (closure / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
                result, report = self.run_review(fixture)
                self.assertEqual(result, 1, report)
                self.assertEqual(report["evidence_status"], "open")

    def test_oracle_distinguishes_mesh_order_and_feature_replacement(self) -> None:
        def at(case: str, x: int, y: int) -> bytes:
            raw, _ = oracle.image(case)
            return raw[(y * 96 + x) * 4:(y * 96 + x + 1) * 4]

        self.assertEqual(at("frame-0", 48, 48), bytes((0, 255, 0, 255)))
        self.assertEqual(at("frame-1", 48, 48), bytes((32, 32, 223, 255)))
        self.assertEqual(at("pending-new", 48, 48), bytes((191, 64, 0, 255)))
        self.assertEqual(at("pending-new", 48, 31), bytes((0, 255, 0, 255)))
        self.assertEqual(at("pending-old", 48, 31), bytes((32, 32, 223, 255)))
        self.assertEqual(at("provider-mesh-only", 28, 28), bytes(4))

    def test_malformed_visual_indices_are_rejected_without_sorting_mixed_types(self) -> None:
        value = {"status": "reviewed", "attempt_ids": ["a"], "evidence": [{
            "attempt_id": "a", "frames": [{"index": None, "visual_verdict": "passed"}]}]}
        self.assertTrue(common.visual_surface_failures(value, Path("."), "a", {}, Path("."), Path(".")))


if __name__ == "__main__":
    unittest.main()
