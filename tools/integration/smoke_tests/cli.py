"""Split smoke runner regression tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest import mock
from .support import smoke

class SmokeRunnerTests(unittest.TestCase):

    def test_command_output_records_a_real_child_and_handles_failure(self) -> None:
        self.assertEqual(smoke.command_output([sys.executable, "-c", "print('toolchain-probe')"]), "toolchain-probe")
        self.assertIsNone(smoke.command_output([sys.executable, "-c", "raise SystemExit(7)"]))

    def test_workload_schema_versions_do_not_cross_stage_boundaries(self) -> None:
        self.assertEqual(smoke.schema_version_for_workload("s0-s3"), 3)
        self.assertEqual(smoke.schema_version_for_workload("s5-forward"), 4)
        self.assertEqual(smoke.schema_version_for_workload("s5-surface"), 4)
        self.assertEqual(smoke.schema_version_for_workload("s5-device-isolation"), 4)
        self.assertEqual(smoke.stage_for_workload("s0-s3"), "S0-S3 integration evidence")
        self.assertEqual(smoke.stage_for_workload("s5-forward"), "S5 Forward integration evidence")
        self.assertEqual(smoke.stage_for_workload("s5-surface"), "S5 Forward integration evidence")
        self.assertEqual(smoke.stage_for_workload("s5-device-isolation"), "S5 Forward integration evidence")


    def test_schema_v4_requires_the_s5_workload_contract(self) -> None:
        schema = json.loads(
            (Path(__file__).parents[1] / "manifest.schema.json")
            .read_text(encoding="utf-8")
        )
        s5_clause = next(
            clause for clause in schema["allOf"]
            if clause["if"]["properties"]["schema_version"].get("const") == 4
        )
        self.assertEqual(
            s5_clause["then"]["properties"]["attempts"]["items"]["$ref"],
            "#/$defs/s5_attempt",
        )
        contract = schema["$defs"]["s5_attempt"]["allOf"][1]["properties"]["workload"]
        choices = contract["oneOf"]
        self.assertEqual(choices[0]["properties"]["entry_argument"]["const"], "--s5-only")
        self.assertEqual(choices[1]["properties"]["entry_argument"]["const"], "--s5-device-only")
        self.assertEqual(choices[2]["properties"]["entry_argument"]["const"], "--s5-surface-only")


    def test_build_mode_records_recognized_focused_test_counts(self) -> None:
        """The real CLI path must retain tests without replacing the manifest path."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            artifact = root / "crates" / "renderer" / "examples" / "rendering-integration"
            artifact.mkdir(parents=True)
            (root / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            (root / "Cargo.lock").write_text("version = 4\n", encoding="utf-8")
            (artifact / "Cargo.toml").write_text(
                '[package]\nname = "rendering-integration"\nversion = "0.0.0"\nedition = "2021"\n',
                encoding="utf-8",
            )
            (artifact / "Cargo.lock").write_text("version = 4\n", encoding="utf-8")

            def successful_command(command, _cwd, _timeout, directory, _environment=None):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "stdout.log").write_text(
                    "Ran 3 tests in 0.001s\nOK\n"
                    if command[:3] == [sys.executable, "-m", "unittest"] else "",
                    encoding="utf-8",
                )
                (directory / "stderr.log").write_text("", encoding="utf-8")
                return smoke.log_result(directory, 0, False, "parent_exited")

            checkout = {"path": str(root), "revision": "fixture", "dirty": False,
                        "role": "local_checkout_inventory"}
            with (mock.patch.object(smoke, "ROOT", root),
                  mock.patch.object(smoke, "ARTIFACT", artifact),
                  mock.patch.object(smoke, "checkout_inventory", return_value=checkout),
                  mock.patch.object(smoke, "build_environment", return_value={"status": "auditable"}),
                  mock.patch.object(smoke, "command_output", return_value="fixture"),
                  mock.patch.object(smoke, "built_fluxel_packages", return_value=[]),
                  mock.patch.object(smoke, "verify_built_sources", return_value={"status": "verified"}),
                  mock.patch.object(smoke, "verify_git_checkouts", return_value={"status": "verified"}),
                  mock.patch.object(smoke, "source_consistency", return_value={"status": "stable", "changed_inputs": []}),
                  mock.patch.object(smoke, "run_command", side_effect=successful_command),
                  mock.patch.object(sys, "argv", [
                      "smoke.py", "--backend", "dx12", "--mode", "build",
                      "--test-command-json", json.dumps([sys.executable, "-m", "unittest"]),
                  ])):
                self.assertEqual(smoke.main(), 0)

            manifest = root / "target/integration/s0-s3/fixture/manifest.json"
            recorded = json.loads(manifest.read_text(encoding="utf-8"))
            focused = [entry for entry in recorded["attempts"][-1]["commands"] if entry["kind"] == "test"]
            self.assertEqual(len(focused), 1)
            self.assertTrue(focused[0]["result"]["test_command_recognized"])
            self.assertEqual(focused[0]["result"]["executed"], 3)


    def test_build_mode_rejects_focused_test_command_that_runs_zero_tests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            artifact = root / "crates" / "renderer" / "examples" / "rendering-integration"
            artifact.mkdir(parents=True)
            (root / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            (root / "Cargo.lock").write_text("version = 4\n", encoding="utf-8")
            (artifact / "Cargo.toml").write_text(
                '[package]\nname = "rendering-integration"\nversion = "0.0.0"\nedition = "2021"\n',
                encoding="utf-8",
            )
            (artifact / "Cargo.lock").write_text("version = 4\n", encoding="utf-8")

            def zero_test_command(command, _cwd, _timeout, directory, _environment=None):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "stdout.log").write_text(
                    "Ran 0 tests in 0.001s\nOK\n"
                    if command[:3] == [sys.executable, "-m", "unittest"] else "",
                    encoding="utf-8",
                )
                (directory / "stderr.log").write_text("", encoding="utf-8")
                return smoke.log_result(directory, 0, False, "parent_exited")

            checkout = {"path": str(root), "revision": "fixture", "dirty": False,
                        "role": "local_checkout_inventory"}
            with (mock.patch.object(smoke, "ROOT", root),
                  mock.patch.object(smoke, "ARTIFACT", artifact),
                  mock.patch.object(smoke, "checkout_inventory", return_value=checkout),
                  mock.patch.object(smoke, "build_environment", return_value={"status": "auditable"}),
                  mock.patch.object(smoke, "command_output", return_value="fixture"),
                  mock.patch.object(smoke, "built_fluxel_packages", return_value=[]),
                  mock.patch.object(smoke, "verify_built_sources", return_value={"status": "verified"}),
                  mock.patch.object(smoke, "verify_git_checkouts", return_value={"status": "verified"}),
                  mock.patch.object(smoke, "source_consistency", return_value={"status": "stable", "changed_inputs": []}),
                  mock.patch.object(smoke, "run_command", side_effect=zero_test_command),
                  mock.patch.object(sys, "argv", [
                      "smoke.py", "--backend", "dx12", "--mode", "build",
                      "--test-command-json", json.dumps([sys.executable, "-m", "unittest"]),
                  ])):
                self.assertEqual(smoke.main(), 1)

            manifest = root / "target/integration/s0-s3/fixture/manifest.json"
            recorded = json.loads(manifest.read_text(encoding="utf-8"))
            focused = [entry for entry in recorded["attempts"][-1]["commands"] if entry["kind"] == "test"]
            self.assertEqual(focused[0]["result"]["executed"], 0)
            self.assertTrue(focused[0]["result"]["test_command_recognized"])


    def test_attempt_binary_survives_the_next_backend_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            built = root / "integration.exe"
            attempt = root / "attempt"
            attempt.mkdir()
            built.write_bytes(b"first-backend-artifact")
            retained = smoke.retain_executable(built, attempt)
            built.write_bytes(b"second-backend-artifact")
            self.assertEqual(Path(retained["path"]).read_bytes(), b"first-backend-artifact")
            self.assertEqual(smoke.sha256(Path(retained["path"])), retained["sha256"])


    def test_s5_nested_workload_requires_all_raw_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run" / "s5-forward"
            run.mkdir(parents=True)
            (run / "s5-forward-workload.json").write_text(json.dumps({
                "workload": "s5-scene-forward-four-layouts"
            }), encoding="utf-8")
            for name in smoke.s5_required_raw_artifacts():
                (run / name).write_bytes(b"raw")
            evidence = smoke.nested_workload_evidence(run.parent, "s5-forward")
            self.assertEqual(evidence["status"], "verified")
            (run / "s5-forward-geometry-1-stream-2.bin").unlink()
            evidence = smoke.nested_workload_evidence(run.parent, "s5-forward")
            self.assertEqual(evidence["status"], "incomplete")
            self.assertTrue(any("geometry-1-stream-2" in item for item in evidence["failures"]))


    def test_s5_device_isolation_requires_shared_inputs_and_both_readbacks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run" / "s5-forward-device-isolation"
            run.mkdir(parents=True)
            (run / "s5-device-isolation-workload.json").write_text(json.dumps({
                "workload": "s5-forward-two-device-cache-isolation",
                "sharedAssets": "s5-device-isolation-shared-assets.json",
                "oldReuseReadback": "s5-device-isolation-old-reuse-actual.rgba",
            }), encoding="utf-8")
            for name in smoke.s5_device_isolation_required_raw_artifacts():
                (run / name).write_bytes(b"raw")
            evidence = smoke.nested_workload_evidence(run.parent, "s5-device-isolation")
            self.assertEqual(evidence["status"], "verified")
            (run / "s5-device-isolation-new-actual.rgba").unlink()
            evidence = smoke.nested_workload_evidence(run.parent, "s5-device-isolation")
            self.assertEqual(evidence["status"], "incomplete")
            self.assertTrue(any("new-actual" in item for item in evidence["failures"]))
