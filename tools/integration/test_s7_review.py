from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import s7_review
from s7_review_parts import common, facts


class S7ReviewTests(unittest.TestCase):
    def test_relocated_raw_paths_use_retained_bytes_and_reject_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "new" / "attempt-123"
            root.mkdir(parents=True)
            root = root.resolve()
            raw = root / "raw.bin"
            raw.write_bytes(b"retained")
            original = Path(temporary) / "old" / "attempt-123" / "raw.bin"
            self.assertEqual(common.retained_path(root, str(original)), raw)
            record = {"path": str(original), "sha256": common.sha256(raw)}
            self.assertEqual(common.hash_records(root, [record]), [])
            raw.write_bytes(b"tampered")
            self.assertTrue(common.hash_records(root, [record]))
            self.assertIsNone(common.retained_path(root, str(original.parent / ".." / "escape")))

    def test_surface_facts_accept_actual_shapes_and_reject_wrong_retirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events = ("configure", "acquire", "complete", "resize", "configure", "acquire", "complete")
            (root / "s7-deferred-surface-lifecycle.jsonl").write_text(
                "".join(json.dumps({"event": event}) + "\n" for event in events), encoding="utf-8")
            for name in ("frame-0", "pbr-metal"):
                (root / f"s7-deferred-surface-{name}-facts.json").write_text(
                    '{"graphPasses":3,"pendingAfterComplete":0}', encoding="utf-8")
            path = root / "s7-deferred-surface-deformation-first-facts.json"
            path.write_text('{"graphPasses":4,"computeBeforeGeometry":true,"pending":[0,1,0,1,0]}', encoding="utf-8")
            self.assertEqual(facts.review_surface(root), [])
            path.write_text('{"graphPasses":4,"computeBeforeGeometry":true,"pending":[0,1,1,1,0]}', encoding="utf-8")
            self.assertTrue(facts.review_surface(root))

    def test_open_when_matrix_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "manifest.json").write_text(json.dumps({
                "schemaVersion": 1, "stage": "S7 deferred integration evidence", "attempts": []}), encoding="utf-8")
            output = root / "review.json"
            with mock.patch.object(sys, "argv", ["s7_review.py", "--closure", str(root), "--output", str(output),
                                                   "--attempt", "missing", "dx12", "s7-deferred"]):
                self.assertEqual(s7_review.main(), 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["evidence_status"], "open")
            self.assertEqual(report["stage_closure_decision"], "not_made_by_this_tool")

    def test_unknown_selector_cannot_be_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "manifest.json").write_text("{}", encoding="utf-8")
            output = root / "review.json"
            with mock.patch.object(sys, "argv", ["s7_review.py", "--closure", str(root), "--output", str(output),
                                                   "--attempt", "x", "vulkan", "s7-deferred"]):
                self.assertEqual(s7_review.main(), 1)
            self.assertIn("unknown", output.read_text(encoding="utf-8"))

    def test_device_facts_malformed_inputs_fail_closed_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "device-isolation-facts.json").write_text("{}", encoding="utf-8")
            (root / "foreign-submission-rejected-facts.json").write_text("{}", encoding="utf-8")
            self.assertTrue(facts.review_device(root))


if __name__ == "__main__":
    unittest.main()
