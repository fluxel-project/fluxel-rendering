"""Immutable S7 runner selection and archive-contract tests."""

import unittest
import json
from pathlib import Path

try:
    from . import s7_smoke, smoke_s7
except ImportError:
    import s7_smoke
    import smoke_s7


class S7InventoryTests(unittest.TestCase):
    def focused_attempt(self, inputs: list[dict], result: dict | None = None) -> dict:
        passed = {"exit_status": 0, "test_command_recognized": True, "executed": 1}
        return {
            "source_inputs": inputs,
            "source_snapshot": {"kind": "exact_worktree_inputs"},
            "commands": [
                {"kind": "test", "command": ["cargo", "test", "-p", "fluxel-renderer", "--lib", "deferred"], "result": result or passed},
                {"kind": "test", "command": ["cargo", "test", "--manifest-path", str(s7_smoke.ARTIFACT / "Cargo.toml"), "--bin", "rendering-integration"], "result": result or passed},
                {"kind": "test", "command": ["cargo", "test", "--manifest-path", str(s7_smoke.ROOT / "crates/renderer/examples/external-renderer" / "Cargo.toml")], "result": result or passed},
                {"kind": "test", "command": ["python", "-m", "unittest", "discover", "--start-directory", "tools/integration", "--pattern", "test_s7*.py"], "result": result or passed},
            ],
        }

    def test_formal_matrix_has_three_distinct_native_workloads(self) -> None:
        self.assertEqual(s7_smoke.BACKENDS, ("dx12", "native-gl-wgl"))
        self.assertEqual(set(s7_smoke.WORKLOADS), set(smoke_s7._CONTRACTS))
        arguments = [smoke_s7.s7_workload_contract(item)["entry_argument"] for item in s7_smoke.WORKLOADS]
        self.assertEqual(arguments, ["--s7-deferred-only", "--s7-deferred-surface-only", "--s7-device-isolation-only"])
        self.assertEqual(s7_smoke.WINDOW_CAPTURE_POLICY, "schema2-nondisturbing")

    def test_schema_and_stage_are_formal_s7_only(self) -> None:
        self.assertEqual(smoke_s7.S7_SCHEMA_VERSION, 6)
        self.assertEqual(smoke_s7.S7_STAGE, "S7 Deferred integration evidence")
        self.assertEqual(len(smoke_s7.s7_workload_contract("s7-deferred")["expected_cases"]), 11)
        self.assertEqual(len(smoke_s7.s7_workload_contract("s7-deferred-surface")["subworkloads"]), 2)
        self.assertEqual(len(smoke_s7.s7_workload_contract("s7-deferred-device-isolation")["expected_cases"]), 8)

    def test_schema_selects_only_s7_attempts(self) -> None:
        schema = json.loads((Path(__file__).parent / "manifest.schema.json").read_text(encoding="utf-8"))
        self.assertIn(6, schema["properties"]["schema_version"]["enum"])
        clauses = {item["if"]["properties"]["schema_version"]["const"]: item["then"] for item in schema["allOf"]}
        self.assertEqual(clauses[6]["properties"]["attempts"]["items"]["$ref"], "#/$defs/s7_attempt")
        options = schema["$defs"]["s7_attempt"]["allOf"][1]["properties"]["workload"]["oneOf"]
        self.assertEqual({item["properties"]["selection"]["const"] for item in options}, {contract["selection"] for contract in map(smoke_s7.s7_workload_contract, s7_smoke.WORKLOADS)})

    def test_only_complete_successful_group_for_exact_source_suppresses_rerun(self) -> None:
        inputs = [{"path": "one.rs", "sha256": "a" * 64}]
        self.assertTrue(s7_smoke._completed_focused_group(self.focused_attempt(inputs), inputs))
        self.assertFalse(s7_smoke._completed_focused_group(self.focused_attempt(inputs, {"exit_status": 1, "test_command_recognized": True, "executed": 1}), inputs))
        self.assertFalse(s7_smoke._completed_focused_group(self.focused_attempt(inputs), [{"path": "one.rs", "sha256": "b" * 64}]))


if __name__ == "__main__":
    unittest.main()
