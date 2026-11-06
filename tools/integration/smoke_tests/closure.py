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

    def test_actual_build_path_override_cannot_reuse_git_lock_proof(self) -> None:
        source = "git+https://example.test/pinned?rev=abc"
        names = ["fluxel-assets", "fluxel-rhi", "fluxel-host"]
        locked = [{"name": name, "source": source + "#abc"} for name in names]
        inventory = [{"name": name, "package_id": source + "#" + name + "@0.1.0"} for name in names]
        inventory.append({"name": "external-renderer", "package_id": "path+file:///external-renderer", "manifest_path": str(smoke.ROOT / "crates/renderer/examples/external-renderer/Cargo.toml")})
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


    def test_built_package_inventory_includes_the_external_consumer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "build.log"
            report.write_text(json.dumps({
                "reason": "compiler-artifact",
                "package_id": "path+file:///external-renderer#external-renderer@0.0.0",
                "manifest_path": str(smoke.ROOT / "crates/renderer/examples/external-renderer/Cargo.toml"),
                "target": {"name": "external_renderer", "kind": ["lib"]},
                "features": [],
            }) + "\n", encoding="utf-8")
            packages = smoke.built_fluxel_packages(report)
            self.assertEqual([package["name"] for package in packages], ["external-renderer"])


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
                "crates/renderer/examples/rendering-integration/Cargo.toml": "[package]\nname='consumer'\n",
                ".cargo/config.toml": "[build]\n",
                "crates/renderer/examples/.cargo/config.toml": "[build]\n",
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
            self.assertIn("crates/renderer/examples/.cargo/config.toml", paths)
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


    def test_source_closure_covers_nested_runner_helpers_and_external_consumer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {
                "tools/integration/reviewer/helpers.py": "def review(): pass\n",
                "crates/renderer/examples/external-renderer/Cargo.toml": "[package]\nname='external-renderer'\n",
                "crates/renderer/examples/external-renderer/src/lib.rs": "pub fn render() {}\n",
            }
            for relative, content in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            paths = {path.relative_to(root.resolve()).as_posix()
                     for path in smoke.source_input_paths(root)}
            self.assertIn("tools/integration/reviewer/helpers.py", paths)
            self.assertIn("crates/renderer/examples/external-renderer/Cargo.toml", paths)
            self.assertIn("crates/renderer/examples/external-renderer/src/lib.rs", paths)
            inputs = [{"path": name, "sha256": smoke.sha256(root / name)} for name in sorted(paths)]
            self.assertEqual(smoke.source_consistency(inputs, root)["status"], "stable")
            (root / "tools/integration/reviewer/helpers.py").write_text("def changed(): pass\n", encoding="utf-8")
            self.assertEqual(smoke.source_consistency(inputs, root)["status"], "changed")


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
            (root / "crates/renderer/examples/rendering-integration").mkdir(parents=True)
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
        inventory.append({"name": "external-renderer", "package_id": "path+file:///external-renderer",
                          "manifest_path": str(smoke.ROOT / "crates/renderer/examples/external-renderer/Cargo.toml")})
        inventory.append({"name": "fluxel-renderer", "package_id": "path+file:///renderer",
                          "manifest_path": str(smoke.ROOT / "crates/renderer/Cargo.toml")})
        inventory.extend([
            {"name": "fluxel-shader", "package_id": "path+file:///shader", "manifest_path": str(smoke.ROOT / "crates/shader/Cargo.toml")},
            {"name": "fluxel-material", "package_id": "path+file:///material", "manifest_path": str(smoke.ROOT / "crates/material/Cargo.toml")},
        ])
        for backend in smoke.BACKENDS:
            required = smoke.required_built_packages(backend)
            self.assertIn("fluxel-host", required)
            self.assertIn("external-renderer", required)
            self.assertEqual(smoke.verify_built_sources(inventory, locked, required)["status"], "verified")
            without_host = [item for item in inventory if item["name"] != "fluxel-host"]
            report = smoke.verify_built_sources(without_host, locked, required)
            self.assertEqual(report["missing_required_packages"], ["fluxel-host"])
            without_external = [item for item in inventory if item["name"] != "external-renderer"]
            report = smoke.verify_built_sources(without_external, locked, required)
            self.assertEqual(report["missing_required_packages"], ["external-renderer"])


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
            self.assertEqual(report["policy"], "tools/integration/policy.md")
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

    def test_attempt_names_remain_distinct_when_the_clock_does_not_advance(self) -> None:
        fixed = smoke.dt.datetime(2026, 11, 5, 11, 31, 33, 534252, tzinfo=smoke.dt.timezone.utc)
        class FrozenDateTime:
            @staticmethod
            def now(_timezone: object) -> object:
                return fixed

        with mock.patch.object(smoke.dt, "datetime", FrozenDateTime):
            first = smoke.attempt_name("dx12")
            second = smoke.attempt_name("dx12")
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("dx12-20261105T113133534252Z-"))


    def test_timeout_preserves_logs_and_marks_child_lifecycle_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = smoke.run_command([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"], directory, 0.5, directory)
            self.assertTrue(result["timed_out"])
            self.assertEqual(result["child_process_lifecycle"], "unverified_after_parent_termination")
            self.assertIn("started", (directory / "stdout.log").read_text(encoding="utf-8"))
            self.assertIsNotNone(result["raw_logs"][0]["sha256"])
