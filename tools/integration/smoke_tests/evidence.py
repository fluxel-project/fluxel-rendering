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

    def test_s5_surface_requires_confirmed_resize_and_passed_capture_verdicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run" / "s5-forward-surface"
            run.mkdir(parents=True)
            (run / "s5-forward-surface-workload.json").write_text(json.dumps({
                "workload": "s5-forward-surface-resize",
                "initialExtent": [320, 180],
                "resizedExtent": [256, 144],
                "presentedFrames": 2,
                "completionBeforeReconfigure": True,
                "visualEvidence": "native-client-capture",
            }), encoding="utf-8")
            for name in smoke.s5_surface_required_raw_artifacts():
                (run / name).write_bytes(b"raw")
            (run / "forward-surface-expected.json").write_text(json.dumps({
                "triangleRgba": [255, 0, 0, 255], "backgroundRgb": [0, 0, 0], "nativeCapture": True,
            }), encoding="utf-8")
            (run / "surface-events.jsonl").write_text(json.dumps({
                "phase": "forward-resize", "requestedExtent": [256, 144],
                "clientExtent": [256, 144], "selectedEvent": "resized",
            }) + "\n", encoding="utf-8")
            (run / "surface-configuration.jsonl").write_text("\n".join(json.dumps({
                "extent": extent, "configurationSupport": True, "requiredUsage": "render_attachment",
            }) for extent in ([320, 180], [256, 144])) + "\n", encoding="utf-8")
            for frame, extent in enumerate(([320, 180], [256, 144])):
                image = f"surface-client-frame-{frame}.bmp"
                (run / f"surface-client-frame-{frame}.capture.json").write_text(json.dumps({
                    "schemaVersion": 1, "frame": frame, "clientExtent": extent, "pixelFormat": "BGRA8-top-down",
                    "image": image, "automaticVisualPass": False,
                }), encoding="utf-8")
                (run / f"surface-client-frame-{frame}.verdict.json").write_text(json.dumps({
                    "frame": frame, "image": image,
                    "oracle": "desktop-capture-corners-and-stable-triangle-interior", "status": "passed",
                }), encoding="utf-8")
            evidence = smoke.nested_workload_evidence(run.parent, "s5-surface")
            self.assertEqual(evidence["status"], "verified")
            (run / "surface-client-frame-1.verdict.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
            evidence = smoke.nested_workload_evidence(run.parent, "s5-surface")
            self.assertEqual(evidence["status"], "incomplete")
            self.assertTrue(any("capture verdict for frame 1" in item for item in evidence["failures"]))


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
            manifest = root / "target/integration/s0-s3/review/manifest.json"
            recorded = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(recorded["schema_version"], 3)
            self.assertEqual(recorded["stage"], "S0-S3 integration evidence")
            historical = recorded["historical_foundation_support_decision"]
            self.assertEqual(historical["status"], "historical_reference_only")
            self.assertTrue(historical["cannot_approve_current_support"])
            self.assertNotIn("support_decision", recorded)
            self.assertIn("s2-texture", recorded["attempts"][0]["workload"]["selection"])
            self.assertIn("s3-material-replacement", recorded["attempts"][0]["workload"]["selection"])
            attempt = recorded["attempts"][0]
            self.assertEqual(attempt["runner_exit_status"], 2)
            result = attempt["commands"][0]["result"]
            self.assertEqual(result["child_process_lifecycle"], "not_started")
            stderr = Path(result["raw_logs"][1]["path"])
            self.assertIn("independent artifact manifest is missing", stderr.read_text())
            retained = {Path(item["path"]).name for item in attempt["retained_attempt_files"]}
            self.assertIn("stderr.log", retained)
            self.assertIn("stdout.log", retained)


    def test_s5_missing_manifest_retains_a_schema_contract_without_run_artifacts(self) -> None:
        """A pre-build S5 failure has a contract but cannot have nested output."""
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
                  mock.patch.object(sys, "argv", [
                      "smoke.py", "--backend", "dx12", "--workload", "s5-forward",
                  ])):
                self.assertEqual(smoke.main(), 2)
                run.assert_not_called()

            manifest = root / "target/integration/s5-forward/review/manifest.json"
            recorded = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(recorded["schema_version"], 4)
            attempt = recorded["attempts"][0]
            self.assertEqual(attempt["runner_exit_status"], 2)
            self.assertEqual(attempt["workload"]["entry_argument"], "--s5-only")
            self.assertNotIn("nested_workloads", attempt["commands"][0]["result"])
            self.assertIn(
                "independent artifact manifest is missing",
                Path(attempt["commands"][0]["result"]["raw_logs"][1]["path"])
                .read_text(encoding="utf-8"),
            )


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
