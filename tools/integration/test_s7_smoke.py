"""Negative and closure tests for the formal S7 nested-workload inventory."""

import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import smoke_s7
except ImportError:
    import smoke_s7


class S7SmokeTests(unittest.TestCase):
    def write_workload(self, root: Path, workload: str) -> Path:
        contract = smoke_s7.s7_workload_contract(workload)
        directory = root / contract["directory"]
        directory.mkdir(parents=True)
        manifests = [(Path(contract["required_nested_workload"]).name, contract["selection"], contract["expected_cases"])]
        for name, identity, cases in smoke_s7._CONTRACTS[workload]["subworkloads"]:
            manifests.append((name, identity, list(cases)))
        for manifest_name, identity, names in manifests:
            prefix = Path(manifest_name).stem
            shared = [f"{prefix}-shared.bin"]
            cases_value = [{"name": name, "artifacts": [f"{prefix}-{name}.raw"]} for name in names]
            value = {"schemaVersion": 1, "workload": identity, "sharedArtifacts": shared, "cases": cases_value}
            if manifest_name == Path(contract["required_nested_workload"]).name:
                value["subworkloads"] = contract["subworkloads"]
            (directory / manifest_name).write_text(json.dumps(value), encoding="utf-8")
            (directory / shared[0]).write_bytes(b"shared")
            for case in cases_value:
                (directory / case["artifacts"][0]).write_bytes(b"raw")
        return directory

    def test_every_contract_accepts_complete_exact_inventory(self) -> None:
        for workload in smoke_s7._CONTRACTS:
            with self.subTest(workload=workload), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.write_workload(root, workload)
                self.assertEqual(smoke_s7.s7_nested_workload_evidence(root, workload)["status"], "verified")

    def test_unlisted_file_or_unsafe_alias_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.write_workload(root, "s7-deferred")
            (directory / "unlisted.bin").write_bytes(b"bad")
            evidence = smoke_s7.s7_nested_workload_evidence(root, "s7-deferred")
            self.assertEqual(evidence["status"], "incomplete")
            self.assertTrue(any("not manifest-enumerated" in item for item in evidence["failures"]))
            (directory / "unlisted.bin").unlink()
            manifest = directory / "s7-deferred-workload.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["cases"][0]["artifacts"].append("./s7-deferred-workload-shared.bin")
            manifest.write_text(json.dumps(value), encoding="utf-8")
            evidence = smoke_s7.s7_nested_workload_evidence(root, "s7-deferred")
            self.assertEqual(evidence["status"], "incomplete")
            self.assertTrue(any("duplicated or aliased" in item for item in evidence["failures"]))

    def test_missing_subworkload_and_wrong_case_set_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.write_workload(root, "s7-deferred-surface")
            (directory / "s7-deferred-surface-pbr-workload.json").unlink()
            evidence = smoke_s7.s7_nested_workload_evidence(root, "s7-deferred-surface")
            self.assertEqual(evidence["status"], "incomplete")
            self.assertTrue(any("missing required nested workload" in item for item in evidence["failures"]))


if __name__ == "__main__":
    unittest.main()
