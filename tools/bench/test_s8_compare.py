from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("s8_compare", Path(__file__).with_name("s8_compare.py"))
assert SPEC and SPEC.loader
compare = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(compare)


def sample(label: str, scenario: str, elapsed: int, bump: int = 0, pair: int = 0) -> dict:
    facts = {name: 1 for name in compare.COUNTERS}; facts["submitElapsedNs"] = elapsed; facts["acceptedTextureAllocations"] += bump
    return {"label": label, "scenario": scenario, "facts": facts, "pair": pair}


class S8CompareTests(unittest.TestCase):
    def test_formal_threshold_requires_every_scene_to_clear_ten_percent(self) -> None:
        rows = []
        for scene, candidate in (("cold", 89), ("warm", 90), ("replacement-10pct", 91)):
            rows += [entry for pair in range(7) for entry in (sample("baseline", scene, 100, pair=pair), sample("candidate", scene, candidate, pair=pair))]
        verdict = compare.decision(rows, compare.SCENARIOS, 7, False)
        self.assertFalse(verdict["primary_acceptance"])
        self.assertEqual(verdict["classification"], "formal")

    def test_exact_ten_percent_passes_with_integer_threshold(self) -> None:
        rows = [entry for scene in compare.SCENARIOS for pair in range(7) for entry in (sample("baseline", scene, 100, pair=pair), sample("candidate", scene, 90, pair=pair))]
        self.assertTrue(compare.decision(rows, compare.SCENARIOS, 7, False)["primary_acceptance"])

    def test_screen_is_not_formal_even_when_fast(self) -> None:
        rows = [sample(label, scene, 100 if label == "baseline" else 50) for scene in compare.SCENARIOS for label in ("baseline", "candidate")]
        self.assertFalse(compare.decision(rows, compare.SCENARIOS, 2, True)["primary_acceptance"])

    def test_missing_actual_pair_is_incomplete(self) -> None:
        rows = [entry for scene in compare.SCENARIOS for pair in range(6) for entry in (sample("baseline", scene, 100, pair=pair), sample("candidate", scene, 80, pair=pair))]
        verdict = compare.decision(rows, compare.SCENARIOS, 7, False)
        self.assertFalse(verdict["complete_unique_pairs"])
        self.assertEqual(verdict["classification"], "incomplete")

    def test_counter_tampering_rejects_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for suffix in ("a", "b"):
                folder = root / suffix / "s8-material-baseline"; folder.mkdir(parents=True)
                for name in compare.INPUTS[:-1]: (folder / name).write_bytes(b"same")
            a = {**sample("baseline", "cold", 1), "directory": str(root / "a")}; b = {**sample("candidate", "cold", 1, 1), "directory": str(root / "b")}
            with self.assertRaisesRegex(ValueError, "counter changed"): compare.compare_pair(a, b)

    def test_changed_required_input_rejects_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for suffix, value in (("a", b"one"), ("b", b"two")):
                folder = root / suffix / "s8-material-baseline"; folder.mkdir(parents=True)
                for name in compare.INPUTS[:-1]: (folder / name).write_bytes(value)
            a = {**sample("baseline", "cold", 1), "directory": str(root / "a")}; b = {**sample("candidate", "cold", 1), "directory": str(root / "b")}
            with self.assertRaisesRegex(ValueError, "input bytes differ"): compare.compare_pair(a, b)

    def test_small_guard_rejects_large_api_regression(self) -> None:
        rows = [sample("baseline", scene, 1_000_000) for scene in compare.SCENARIOS] + [sample("candidate", scene, 2_000_000) for scene in compare.SCENARIOS]
        self.assertFalse(compare.guard_decision(rows)["accepted"])


if __name__ == "__main__": unittest.main()
