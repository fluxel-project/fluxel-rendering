"""Stdlib regression checks for immutable smoke-runner attempt evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

SPEC = importlib.util.spec_from_file_location("smoke", Path(__file__).with_name("smoke.py"))
assert SPEC and SPEC.loader
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class SmokeRunnerTests(unittest.TestCase):
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

    def test_modified_cargo_git_checkout_cannot_claim_the_locked_commit(self) -> None:
        inventory = [{"name": "fluxel-rhi", "manifest_path": "checkout/core/Cargo.toml"}]
        locked = [{"name": "fluxel-rhi", "source": "git+https://example.test/rhi#exact"}]
        def git_status(changes: str):
            def probe(directory, *arguments):
                if arguments == ("rev-parse", "--show-toplevel"):
                    return "checkout"
                if arguments == ("rev-parse", "HEAD"):
                    return "exact"
                return changes
            return probe
        with mock.patch.object(smoke, "git_value", side_effect=git_status("?? .cargo-ok")):
            self.assertEqual(smoke.verify_git_checkouts(inventory, locked)["status"], "verified")
        for changed in (" M core/src/lib.rs\n?? .cargo-ok", "?? injected.rs\n?? .cargo-ok"):
            with mock.patch.object(smoke, "git_value", side_effect=git_status(changed)):
                self.assertEqual(smoke.verify_git_checkouts(inventory, locked)["status"], "mismatch")

    def test_frozen_source_preserves_dirty_and_new_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "new.rs"
            source.write_bytes(b"fn main() {}\n")
            inputs = [{"path": "new.rs", "sha256": smoke.sha256(source)}]
            artifact = root / "inputs.zip"
            frozen = smoke.freeze_source_inputs(inputs, artifact, root)
            source.write_bytes(b"changed after capture\n")
            with zipfile.ZipFile(artifact) as archive:
                self.assertEqual(archive.read("new.rs"), b"fn main() {}\n")
                self.assertEqual(json.loads(archive.read("snapshot-inputs.json")), inputs)
            self.assertEqual(frozen["sha256"], smoke.sha256(artifact))

    def test_source_change_during_freeze_is_not_silently_attributed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "main.rs"
            source.write_bytes(b"one")
            inputs = [{"path": "main.rs", "sha256": smoke.sha256(source)}]
            source.write_bytes(b"two")
            with self.assertRaisesRegex(ValueError, "source changed while freezing"):
                smoke.freeze_source_inputs(inputs, root / "inputs.zip", root)

    def test_missing_manifest_retains_a_failed_attempt_without_building(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            check_sources = smoke.source_consistency
            checkout = {"path": str(root), "revision": "review", "dirty": True,
                        "role": "local_checkout_inventory"}
            with (mock.patch.object(smoke, "ROOT", root),
                  mock.patch.object(smoke, "ARTIFACT", root / "missing-consumer"),
                  mock.patch.object(smoke, "__file__", str(root / "tools/integration/smoke.py")),
                  mock.patch.object(smoke, "source_consistency",
                                    side_effect=lambda inputs: check_sources(inputs, root)),
                  mock.patch.object(smoke, "checkout_inventory", return_value=checkout),
                  mock.patch.object(smoke, "command_output", return_value=None),
                  mock.patch.object(smoke, "run_command") as run,
                  mock.patch.object(sys, "argv", ["smoke.py", "--backend", "dx12"])):
                self.assertEqual(smoke.main(), 2)
                run.assert_not_called()
            manifest = root / "target/integration/s0-s1/review/manifest.json"
            attempt = json.loads(manifest.read_text(encoding="utf-8"))["attempts"][0]
            self.assertEqual(attempt["runner_exit_status"], 2)
            result = attempt["commands"][0]["result"]
            self.assertEqual(result["child_process_lifecycle"], "not_started")
            stderr = Path(result["raw_logs"][1]["path"])
            self.assertIn("independent artifact manifest is missing", stderr.read_text())

    def test_compiled_package_inventory_preserves_cargo_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "cargo.jsonl"
            report.write_text(json.dumps({
                "reason": "compiler-artifact",
                "package_id": "git+https://github.com/fluxel-project/fluxel-rhi?rev=abc#fluxel-rhi-core@0.1.0",
                "manifest_path": "checkout/core/Cargo.toml",
                "target": {"name": "fluxel_rhi_core", "kind": ["lib"]},
                "features": ["feature-b", "feature-a"], "fresh": True,
            }) + "\n", encoding="utf-8")
            package = smoke.built_fluxel_packages(report)[0]
            self.assertEqual(package["name"], "fluxel-rhi-core")
            self.assertEqual(package["features"], ["feature-a", "feature-b"])
            self.assertTrue(package["fresh"])

    def test_actual_build_path_override_cannot_reuse_git_lock_proof(self) -> None:
        source = "git+https://example.test/pinned?rev=abc"
        names = ["fluxel-assets", "fluxel-rhi", "fluxel-host"]
        locked = [{"name": name, "source": source + "#abc"} for name in names]
        inventory = [{"name": name, "package_id": source + "#" + name + "@0.1.0"} for name in names]
        inventory.append({"name": "fluxel-renderer", "package_id": "path+file:///renderer", "manifest_path": str(smoke.ROOT / "crates/renderer/Cargo.toml")})
        self.assertEqual(smoke.verify_built_sources(inventory, locked)["status"], "verified")
        inventory[1]["package_id"] = "path+file:///override#fluxel-rhi@0.1.0"
        report = smoke.verify_built_sources(inventory, locked)
        self.assertEqual(report["status"], "mismatch")
        self.assertEqual(report["mismatches"][0]["name"], "fluxel-rhi")

    def test_source_edits_or_deletion_invalidate_attempt_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.rs"
            source.write_text("original\n", encoding="utf-8")
            inputs = [{"path": "source.rs", "sha256": smoke.sha256(source)}]
            self.assertEqual(smoke.source_consistency(inputs, root)["status"], "stable")
            extra = root / "added.rs"
            extra.write_text("new module\n", encoding="utf-8")
            self.assertEqual(smoke.source_consistency(inputs, root)["status"], "changed")
            extra.unlink()
            source.write_text("changed\n", encoding="utf-8")
            self.assertEqual(smoke.source_consistency(inputs, root)["status"], "changed")
            source.unlink()
            report = smoke.source_consistency(inputs, root)
            self.assertEqual(report["status"], "changed")
            self.assertIsNone(report["changed_inputs"][0]["after_sha256"])

    def test_both_surface_backends_require_host_to_be_compiled(self) -> None:
        source = "git+https://example.test/pinned?rev=abc"
        names = ["fluxel-assets", "fluxel-rhi", "fluxel-host"]
        locked = [{"name": name, "source": source + "#abc"} for name in names]
        inventory = [{"name": name, "package_id": source + "#" + name + "@0.1.0"} for name in names]
        inventory.append({"name": "fluxel-renderer", "package_id": "path+file:///renderer",
                          "manifest_path": str(smoke.ROOT / "crates/renderer/Cargo.toml")})
        for backend in smoke.BACKENDS:
            required = smoke.required_built_packages(backend)
            self.assertIn("fluxel-host", required)
            self.assertEqual(smoke.verify_built_sources(inventory, locked, required)["status"], "verified")
            without_host = [item for item in inventory if item["name"] != "fluxel-host"]
            report = smoke.verify_built_sources(without_host, locked, required)
            self.assertEqual(report["missing_required_packages"], ["fluxel-host"])

    def test_shader_edits_and_new_shader_units_invalidate_source_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shader = root / "raster.wgsl"
            shader.write_text("original", encoding="utf-8")
            inputs = [{"path": "raster.wgsl", "sha256": smoke.sha256(shader)}]
            self.assertEqual(smoke.source_consistency(inputs, root)["status"], "stable")
            extra = root / "extra.wgsl"
            extra.write_text("new", encoding="utf-8")
            self.assertEqual(smoke.source_consistency(inputs, root)["status"], "changed")
            extra.unlink()
            shader.write_text("modified", encoding="utf-8")
            self.assertEqual(smoke.source_consistency(inputs, root)["status"], "changed")

    def test_rebuilt_workload_does_not_inherit_historical_diagnostic_waivers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stderr = Path(temporary) / "stderr.log"
            unknown = "D3D12 [D3D12_MESSAGE_SEVERITY(1)/D3D12_MESSAGE_ID(820)]: unexpected failure"
            historical = "D3D12 [D3D12_MESSAGE_SEVERITY(2)/D3D12_MESSAGE_ID(820)]: clear optimization warning"
            stderr.write_text(historical + "\n" + unknown + "\n", encoding="utf-8")
            report = smoke.classify_dx12_diagnostics(stderr)
            self.assertEqual(report["unreviewed_native_diagnostics"], [historical, unknown])
            self.assertEqual(report["policy"], "tools/integration/README.md")
            self.assertIsNotNone(report["policy_sha256"])

    def test_validation_evidence_requires_current_attempt_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            missing = smoke.validation_evidence("dx12", run, "enabled")
            self.assertEqual(missing["status"], "unverified")
            marker = run / "dx12-debug-request.json"
            marker.write_text('{"requested": true, "deviceOpenCompleted": true}\n', encoding="utf-8")
            self.assertEqual(smoke.validation_evidence("dx12", run, "enabled")["status"], "unverified")
            marker.write_text(json.dumps({"requested": True, "deviceOpenCompleted": True,
                                          "activationContract": f"{smoke.RHI_REVISION} EnableDebugLayer"}), encoding="utf-8")
            observed = smoke.validation_evidence("dx12", run, "enabled")
            self.assertEqual(observed["status"], "observed")
            self.assertEqual(observed["artifact"]["sha256"], smoke.sha256(marker))
            (run / "wgl-debug-observer.json").write_text(
                json.dumps({"installed": True, "logSaturated": False, "activationMarkerObserved": True,
                            "activationMessages": ["fluxel-rendering WGL diagnostic observer active"],
                            "nativeDiagnostics": ["driver note"]}), encoding="utf-8"
            )
            wgl = smoke.validation_evidence("native-gl-wgl", run, "enabled")
            self.assertEqual(wgl["status"], "observed")
            self.assertEqual(wgl["native_diagnostic_count"], 1)
            replacement = run / "replacement-diagnostics"
            replacement.mkdir()
            (replacement / "wgl-debug-observer.json").write_text(json.dumps({
                "installed": True, "logSaturated": False, "activationMarkerObserved": True,
                "activationMessages": ["fluxel-rendering WGL diagnostic observer active"],
                "nativeDiagnostics": ["replacement driver failure"],
            }), encoding="utf-8")
            contexts = smoke.validation_evidence("native-gl-wgl", run, "enabled")
            self.assertEqual(contexts["native_diagnostic_count"], 2)
            self.assertEqual(len(contexts["additional_artifacts"]), 1)
            marker = replacement / "wgl-debug-observer.json"
            saturated = json.loads(marker.read_text(encoding="utf-8"))
            saturated["logSaturated"] = True
            marker.write_text(json.dumps(saturated), encoding="utf-8")
            self.assertEqual(smoke.validation_evidence("native-gl-wgl", run, "enabled")["status"], "unverified")

    def test_cargo_uri_package_id_and_hyphenated_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = directory / "Cargo.toml"
            manifest.write_text('[package]\nname = "rendering-integration"\n', encoding="utf-8")
            executable = str(directory / "rendering-integration.exe")
            report = directory / "build.log"
            report.write_text(json.dumps({
                "reason": "compiler-artifact",
                "package_id": "path+file:///example#rendering-integration@0.0.0",
                "manifest_path": str(manifest),
                "target": {"name": "rendering-integration", "kind": ["bin"]},
                "executable": executable,
            }) + "\n", encoding="utf-8")
            self.assertEqual(smoke.find_built_executable(report, manifest), executable)

    def test_repeated_backend_attempts_preserve_failed_raw_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / smoke.attempt_name("dx12")
            second = Path(temporary) / smoke.attempt_name("dx12")
            self.assertNotEqual(first, second)
            result = smoke.run_command(
                [sys.executable, "-c", "print('first failure'); raise SystemExit(7)"],
                Path(temporary), 5, first,
            )
            initial_hash = smoke.sha256(first / "stdout.log")
            smoke.run_command(
                [sys.executable, "-c", "print('second attempt')"],
                Path(temporary), 5, second,
            )
            self.assertEqual(result["exit_status"], 7)
            self.assertEqual(smoke.sha256(first / "stdout.log"), initial_hash)
            self.assertIn("first failure", (first / "stdout.log").read_text())

    def test_timeout_preserves_logs_and_marks_child_lifecycle_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = smoke.run_command([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"], directory, 0.5, directory)
            self.assertTrue(result["timed_out"])
            self.assertEqual(result["child_process_lifecycle"], "unverified_after_parent_termination")
            self.assertIn("started", (directory / "stdout.log").read_text(encoding="utf-8"))
            self.assertIsNotNone(result["raw_logs"][0]["sha256"])


if __name__ == "__main__":
    unittest.main()
