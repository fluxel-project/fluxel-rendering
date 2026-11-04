"""Fast source-identity and fact-contract tests for the S5 baseline tool."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "s5_baseline", Path(__file__).with_name("s5_baseline.py")
)
assert SPEC and SPEC.loader
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


class S5BaselineTests(unittest.TestCase):
    def make_root(self, temporary: str) -> tuple[Path, Path]:
        root = Path(temporary)
        source = root / "dirty-source.rs"
        source.write_bytes(b"uncommitted source\n")
        tool = root / "tools" / "bench" / "helper.py"
        tool.parent.mkdir(parents=True)
        tool.write_text("# untracked benchmark helper\n", encoding="utf-8")
        return root, source

    def test_dirty_stable_inputs_are_archived_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, source = self.make_root(temporary)
            output = root / "output"
            output.mkdir()
            with mock.patch.object(baseline, "source_input_paths", return_value=[source]):
                snapshot = baseline.capture_source_snapshot(output, root)
            with zipfile.ZipFile(output / "source-inputs.zip") as archive:
                self.assertEqual(archive.read("dirty-source.rs"), b"uncommitted source\n")
                self.assertIn("tools/bench/helper.py", archive.namelist())
                self.assertEqual(json.loads(archive.read("snapshot-inputs.json")), snapshot["inputs"])
            baseline.require_stable_source(snapshot["inputs"], root)

    def test_edit_after_freeze_rejects_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, source = self.make_root(temporary)
            output = root / "output"
            output.mkdir()
            actual_freeze = baseline.freeze_source_inputs

            def freeze_then_edit(inputs, destination, frozen_root):
                result = actual_freeze(inputs, destination, frozen_root)
                source.write_bytes(b"changed during capture\n")
                return result

            with (mock.patch.object(baseline, "source_input_paths", return_value=[source]),
                  mock.patch.object(baseline, "freeze_source_inputs", side_effect=freeze_then_edit)):
                with self.assertRaisesRegex(RuntimeError, "source changed during capture"):
                    baseline.capture_source_snapshot(output, root)
            self.assertTrue((output / "source-inputs.zip").is_file())

    def test_missing_measurement_fact_rejects_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run" / "s5-forward"
            run.mkdir(parents=True)
            cache = {name: 0 for name in baseline.REQUIRED_CACHE_FACTS}
            (run / "s5-forward-cache-facts.json").write_text(json.dumps(cache), encoding="utf-8")
            (run / "s5-forward-measurements.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "measurement facts are missing"):
                baseline.facts(run.parent)


if __name__ == "__main__":
    unittest.main()
