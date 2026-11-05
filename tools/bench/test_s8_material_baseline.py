"""Isolated fact-contract tests for the S8 measurement runner."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("s8_material_baseline", Path(__file__).with_name("s8_material_baseline.py"))
assert SPEC and SPEC.loader
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


def valid_facts(scenario: str = "cold", smoke: bool = False) -> dict[str, object]:
    return {**{name: 0 for name in baseline.REQUIRED_FACTS}, "workload": "s8-material-baseline", "role": "representative",
            "primaryMetric": "cpu-preparation-submit", "gpuTiming": "unobserved", "scenario": scenario, "smoke": smoke,
            "draws": baseline.ACTUAL_DRAWS, "textures": baseline.ACTUAL_TEXTURES, "allTexturesReferenced": True}


class S8MaterialBaselineTests(unittest.TestCase):
    def write_run(self, root: Path, data: dict[str, object], raw: bool = True) -> None:
        output = root / "s8-material-baseline"
        output.mkdir(parents=True, exist_ok=True)
        (output / "s8-material-baseline.json").write_text(json.dumps(data), encoding="utf-8")
        (output / "s8-material-input.json").write_bytes(b"input")
        if raw:
            (output / "s8-material-cold-raw.rgba").write_bytes(b"raw")

    def test_actual_contract_accepts_raw_non_boolean_integers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.write_run(root, valid_facts())
            self.assertEqual(baseline.facts(root, "cold", False)["draws"], baseline.ACTUAL_DRAWS)
            hashes = baseline.raw_input_readback_hashes(root)
            self.assertEqual(len(hashes["raw_input_artifacts"]), 1)
            self.assertEqual(len(hashes["raw_readback_artifacts"]), 1)

    def test_boolean_integer_and_wrong_actual_shape_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data = valid_facts(); data["draws"] = True; self.write_run(root, data)
            with self.assertRaisesRegex(ValueError, "non-boolean integer draws"):
                baseline.facts(root, "cold", False)
            data = valid_facts(); data["textures"] = 10; self.write_run(root, data)
            with self.assertRaisesRegex(ValueError, "10000 draws, 1000 textures"):
                baseline.facts(root, "cold", False)

    def test_missing_raw_and_scenario_mismatch_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.write_run(root, valid_facts("warm"), raw=False)
            with self.assertRaisesRegex(ValueError, "scenario/GPU"):
                baseline.facts(root, "cold", False)
            with self.assertRaisesRegex(ValueError, "missing raw input or raw/readback"):
                baseline.raw_input_readback_hashes(root)

    def test_workload_command_uses_required_cli(self) -> None:
        command = baseline.workload_command(Path("consumer.exe"), Path("output"), "replacement-10pct", True)
        self.assertEqual(command[-4:], ["--s8-material-only", "--s8-scenario", "replacement-10pct", "--s8-smoke"])


if __name__ == "__main__":
    unittest.main()
