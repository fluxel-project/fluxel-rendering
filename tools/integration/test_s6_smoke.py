"""Focused S6 manifest-contract regression tests."""

import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import smoke_s6
except ImportError:
    import smoke_s6


class S6SmokeTests(unittest.TestCase):
    def write_manifest(self, root: Path, workload: str, cases: list[dict[str, object]]) -> Path:
        contract = smoke_s6.s6_workload_contract(workload)
        directory = root / contract["required_nested_workload"].rsplit("/", 1)[0]
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / Path(contract["required_nested_workload"]).name
        manifest.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "workload": contract["selection"],
                    "cases": cases,
                }
            ),
            encoding="utf-8",
        )
        return directory

    def test_composed_requires_each_named_case_and_every_listed_artifact(self) -> None:
        workload = "s6-composed"
        contract = smoke_s6.s6_workload_contract(workload)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = [
                {"name": name, "artifacts": [f"{index}.raw"]}
                for index, name in enumerate(contract["expected_cases"])
            ]
            directory = self.write_manifest(root, workload, cases)
            for index in range(len(cases)):
                (directory / f"{index}.raw").write_bytes(b"raw")
            self.assertEqual(
                smoke_s6.s6_nested_workload_evidence(root, workload)["status"], "verified"
            )
            (directory / "2.raw").unlink()
            evidence = smoke_s6.s6_nested_workload_evidence(root, workload)
            self.assertEqual(evidence["status"], "incomplete")
            self.assertTrue(any("2.raw" in failure for failure in evidence["failures"]))

    def test_surface_rejects_missing_case_and_duplicate_artifact_claims(self) -> None:
        workload = "s6-surface"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.write_manifest(
                root,
                workload,
                [{"name": "mesh-after-quad-initial", "artifacts": ["shared.raw"]}],
            )
            (directory / "shared.raw").write_bytes(b"raw")
            evidence = smoke_s6.s6_nested_workload_evidence(root, workload)
            self.assertEqual(evidence["status"], "incomplete")
            self.assertTrue(any("expected named cases" in failure for failure in evidence["failures"]))
            directory = self.write_manifest(
                root,
                workload,
                [
                    {"name": "mesh-after-quad-initial", "artifacts": ["shared.raw"]},
                    {"name": "quad-after-mesh-resized", "artifacts": ["shared.raw"]},
                ],
            )
            (directory / "shared.raw").write_bytes(b"raw")
            evidence = smoke_s6.s6_nested_workload_evidence(root, workload)
            self.assertEqual(evidence["status"], "incomplete")
            self.assertTrue(any("duplicated" in failure for failure in evidence["failures"]))

    def test_manifest_rejects_unlisted_files_and_path_aliases_or_escapes(self) -> None:
        workload = "s6-surface"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.write_manifest(
                root,
                workload,
                [
                    {"name": "mesh-after-quad-initial", "artifacts": ["one.raw"]},
                    {"name": "quad-after-mesh-resized", "artifacts": ["two.raw"]},
                ],
            )
            (directory / "one.raw").write_bytes(b"one")
            (directory / "two.raw").write_bytes(b"two")
            (directory / "unlisted.raw").write_bytes(b"unlisted")
            evidence = smoke_s6.s6_nested_workload_evidence(root, workload)
            self.assertEqual(evidence["status"], "incomplete")
            self.assertTrue(any("not manifest-enumerated" in failure for failure in evidence["failures"]))

            self.write_manifest(
                root,
                workload,
                [
                    {"name": "mesh-after-quad-initial", "artifacts": ["one.raw", "./one.raw"]},
                    {
                        "name": "quad-after-mesh-resized",
                        "artifacts": ["../escape.raw", r"C:\\escape.raw"],
                    },
                ],
            )
            (directory / "unlisted.raw").unlink()
            evidence = smoke_s6.s6_nested_workload_evidence(root, workload)
            self.assertEqual(evidence["status"], "incomplete")
            self.assertTrue(any("aliased" in failure for failure in evidence["failures"]))
            self.assertTrue(any("unsafe path" in failure for failure in evidence["failures"]))

    def test_s6_contracts_keep_distinct_schema_stage_and_invocations(self) -> None:
        self.assertEqual(smoke_s6.S6_SCHEMA_VERSION, 5)
        self.assertEqual(smoke_s6.S6_STAGE, "S6 typed Forward integration evidence")
        self.assertEqual(
            smoke_s6.s6_workload_contract("s6-composed")["entry_argument"], "--s6-only"
        )
        self.assertEqual(
            smoke_s6.s6_workload_contract("s6-surface")["entry_argument"], "--s6-surface-only"
        )

    def test_schema_v5_selects_only_s6_attempt_contracts(self) -> None:
        schema = json.loads(
            (Path(__file__).parents[2] / "documents" / "integration-manifest.schema.json")
            .read_text(encoding="utf-8")
        )
        self.assertIn(5, schema["properties"]["schema_version"]["enum"])
        self.assertIn("S6 typed Forward integration evidence", schema["properties"]["stage"]["enum"])
        clauses = {
            clause["if"]["properties"]["schema_version"]["const"]: clause["then"]
            for clause in schema["allOf"]
        }
        self.assertEqual(clauses[5]["properties"]["attempts"]["items"]["$ref"], "#/$defs/s6_attempt")
        self.assertEqual(clauses[4]["properties"]["attempts"]["items"]["$ref"], "#/$defs/s5_attempt")
        s6 = schema["$defs"]["s6_attempt"]["allOf"][1]["properties"]["workload"]
        selections = {
            option["properties"]["selection"]["const"] for option in s6["oneOf"]
        }
        self.assertEqual(selections, {"s6-composed-offscreen-typed-quad", "s6-composed-surface-resize"})
        self.assertNotIn("s5-scene-forward-four-layouts", selections)
