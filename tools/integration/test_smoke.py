"""Stdlib regression checks for immutable S0-S3 smoke-runner attempt evidence."""
import importlib.util
import json
import os
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
    def test_build_mode_records_recognized_focused_test_counts(self) -> None:
        """The real CLI path must retain tests without replacing the manifest path."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            artifact = root / "examples" / "rendering-integration"
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
            artifact = root / "examples" / "rendering-integration"
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
        inventory.extend([
            {"name": "fluxel-shader", "package_id": "path+file:///shader", "manifest_path": str(smoke.ROOT / "crates/shader/Cargo.toml")},
            {"name": "fluxel-material", "package_id": "path+file:///material", "manifest_path": str(smoke.ROOT / "crates/material/Cargo.toml")},
        ])
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

    def test_build_scripts_and_cargo_config_are_closed_without_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative, content in {
                "Cargo.toml": "[workspace]\n",
                "crates/shader/Cargo.toml": "[package]\nname='shader'\n",
                "crates/shader/build.rs": "fn main() {}\n",
                "examples/rendering-integration/Cargo.toml": "[package]\nname='consumer'\n",
                ".cargo/config.toml": "[build]\n",
                ".cargo/credentials.toml": "token = 'secret'\n",
                "crates/shader/draft/private.rs": "secret\n",
                "crates/shader/AGENTS.md": "local\n",
            }.items():
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            resolved_root = root.resolve()
            paths = {path.relative_to(resolved_root).as_posix() for path in smoke.source_input_paths(root)}
            self.assertIn("crates/shader/build.rs", paths)
            self.assertIn(".cargo/config.toml", paths)
            self.assertNotIn(".cargo/credentials.toml", paths)
            self.assertNotIn("crates/shader/draft/private.rs", paths)
            self.assertNotIn("crates/shader/AGENTS.md", paths)
            inputs = [{"path": name, "sha256": smoke.sha256(root / name)} for name in sorted(paths)]
            self.assertEqual(smoke.source_consistency(inputs, root)["status"], "stable")
            (root / "crates/shader/build.rs").write_text("changed\n", encoding="utf-8")
            self.assertEqual(smoke.source_consistency(inputs, root)["status"], "changed")
            (root / "crates/shader/build.rs").write_text("fn main() {}\n", encoding="utf-8")
            (root / ".cargo/config.toml").unlink()
            without_config = [{"path": name, "sha256": smoke.sha256(root / name)}
                              for name in sorted(paths - {".cargo/config.toml"})]
            (root / ".cargo/config.toml").write_text("[target]\n", encoding="utf-8")
            self.assertEqual(smoke.source_consistency(without_config, root)["status"], "changed")

    def test_build_environment_never_records_override_values(self) -> None:
        with mock.patch.dict(smoke.os.environ, {"RUSTFLAGS": "--cfg secret_value"}, clear=False):
            evidence = smoke.build_environment()
        self.assertEqual(evidence["status"], "rejected")
        self.assertNotIn("--cfg secret_value", json.dumps(evidence))
        self.assertIn("sha256", evidence["overrides"]["RUSTFLAGS"])

    def test_probable_secret_or_default_cargo_home_config_rejects_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); secret = root / "crates/shader/local.key"
            secret.parent.mkdir(parents=True); secret.write_text("not archived", encoding="utf-8")
            self.assertEqual(smoke.sensitive_build_inputs(root), 1)
            self.assertEqual(smoke.build_environment(root)["status"], "rejected")
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary); (home / ".cargo").mkdir()
            (home / ".cargo/config.toml").write_text("[build]\n", encoding="utf-8")
            with mock.patch.object(smoke.Path, "home", return_value=home):
                self.assertEqual(smoke.build_environment(Path(temporary) / "workspace")["status"], "rejected")

    def test_parent_cargo_config_and_target_rustflags_are_rejected_without_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary); root = parent / "workspace"
            (root / "examples/rendering-integration").mkdir(parents=True)
            config = parent / ".cargo/config.toml"; config.parent.mkdir()
            config.write_text("[build]\n", encoding="utf-8")
            with mock.patch.dict(smoke.os.environ, {"CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS": "-C link-arg=secret"}, clear=False):
                evidence = smoke.build_environment(root)
            self.assertEqual(evidence["status"], "rejected")
            self.assertNotIn("-C link-arg=secret", json.dumps(evidence))
            self.assertTrue(any("outside" in reason for reason in evidence["reasons"]))

    def test_cargo_config_with_auth_assignment_is_rejected_and_not_archived(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); config = root / ".cargo/config.toml"
            config.parent.mkdir(); config.write_text("token = 'private'\n", encoding="utf-8")
            paths = {path.relative_to(root.resolve()).as_posix() for path in smoke.source_input_paths(root)}
            self.assertNotIn(".cargo/config.toml", paths)
            evidence = smoke.build_environment(root)
            self.assertEqual(evidence["status"], "rejected")
            self.assertNotIn("private", json.dumps(evidence))

    def test_external_link_and_dynamic_rust_include_are_rejected_without_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "crates/shader/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text('const DATA: &str = include_str!(concat!("../", "outside.txt"));\n', encoding="utf-8")
            self.assertGreater(smoke.closure_escape_count(root), 0)
            evidence = smoke.build_environment(root)
            self.assertEqual(evidence["status"], "rejected")
            self.assertNotIn("outside.txt", json.dumps(evidence))

    def test_build_script_is_snapshotted_but_rejected_without_a_closed_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); build = root / "crates/shader/build.rs"
            build.parent.mkdir(parents=True); build.write_text("fn main() {}\n", encoding="utf-8")
            self.assertEqual(smoke.uncontrolled_build_script_count(root), 1)
            self.assertEqual(smoke.build_environment(root)["status"], "rejected")

    def test_target_escape_link_is_ignored_before_link_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); target = root / "crates/shader/target/outside"
            target.parent.mkdir(parents=True)
            outside = root.parent / "outside-rust-include.txt"; outside.write_text("outside", encoding="utf-8")
            try:
                os.symlink(outside, target)
            except OSError as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            self.assertEqual(smoke.closure_escape_count(root), 0)

    @unittest.skipUnless(os.name == "nt", "junctions are Windows-only")
    def test_external_junction_is_rejected_but_target_junction_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"; root.mkdir()
            outside = Path(temporary) / "outside"; outside.mkdir()
            external = root / "crates/shader/external-junction"; external.parent.mkdir(parents=True)
            ignored = root / "crates/shader/target/ignored-junction"; ignored.parent.mkdir(parents=True)
            def junction(link: Path) -> None:
                command = (f"$link = '{str(link).replace("'", "''")}'; "
                           f"$target = '{str(outside).replace("'", "''")}'; "
                           "New-Item -ItemType Junction -Path $link -Target $target -ErrorAction Stop | Out-Null")
                result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                        capture_output=True, text=True)
                if result.returncode != 0:
                    self.skipTest("junction creation is unavailable on this platform or account")
            def remove_junction(link: Path) -> None:
                command = f"Remove-Item -LiteralPath '{str(link).replace("'", "''")}' -Force -ErrorAction Stop"
                subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                               capture_output=True, text=True, check=True)
            try:
                junction(external)
                self.assertGreater(smoke.closure_escape_count(root), 0)
                remove_junction(external)
                junction(ignored)
                self.assertEqual(smoke.closure_escape_count(root), 0)
            finally:
                for link in (external, ignored):
                    if link.exists() or getattr(link, "is_junction", lambda: False)():
                        remove_junction(link)

    def test_actual_test_count_rejects_zero_or_non_test_and_counts_runner_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); log = root / "stdout.log"; err = root / "stderr.log"
            err.write_text("", encoding="utf-8")
            result = {"raw_logs": [{"path": str(log)}, {"path": str(err)}]}
            log.write_text("Ran 0 tests in 0.000s\nOK\n", encoding="utf-8")
            self.assertEqual(smoke.actual_test_count([sys.executable, "-m", "unittest"], result), 0)
            log.write_text("Ran 3 tests in 0.010s\nOK\n", encoding="utf-8")
            self.assertEqual(smoke.actual_test_count([sys.executable, "-m", "unittest"], result), 3)
            log.write_text("test result: ok. 2 passed; 0 failed;\ntest result: ok. 4 passed; 0 failed;\n", encoding="utf-8")
            self.assertEqual(smoke.actual_test_count(["cargo", "test"], result), 6)
            self.assertIsNone(smoke.actual_test_count(["echo", "Ran 9 tests"], result))

    def test_controlled_child_environment_excludes_arbitrary_fluxel_and_secret_values(self) -> None:
        with mock.patch.dict(smoke.os.environ, {"PATH": "toolchain", "API_KEY": "secret",
                                                "FLUXEL_UNTRUSTED": "leak"}, clear=True):
            environment, evidence = smoke.controlled_child_environment({"FLUXEL_DX12_DEBUG": "1"})
        self.assertEqual(environment, {"PATH": "toolchain", "FLUXEL_DX12_DEBUG": "1"})
        self.assertNotIn("API_KEY", evidence)
        self.assertNotIn("FLUXEL_UNTRUSTED", evidence)
        self.assertNotIn("toolchain", json.dumps(evidence))

    def test_both_surface_backends_require_host_to_be_compiled(self) -> None:
        source = "git+https://example.test/pinned?rev=abc"
        names = ["fluxel-assets", "fluxel-rhi", "fluxel-host"]
        locked = [{"name": name, "source": source + "#abc"} for name in names]
        inventory = [{"name": name, "package_id": source + "#" + name + "@0.1.0"} for name in names]
        inventory.append({"name": "fluxel-renderer", "package_id": "path+file:///renderer",
                          "manifest_path": str(smoke.ROOT / "crates/renderer/Cargo.toml")})
        inventory.extend([
            {"name": "fluxel-shader", "package_id": "path+file:///shader", "manifest_path": str(smoke.ROOT / "crates/shader/Cargo.toml")},
            {"name": "fluxel-material", "package_id": "path+file:///material", "manifest_path": str(smoke.ROOT / "crates/material/Cargo.toml")},
        ])
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
