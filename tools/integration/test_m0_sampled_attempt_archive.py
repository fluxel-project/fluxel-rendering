"""Archival and runner-boundary tests for M0 sampled-material attempts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

import m0_sampled_attempt as attempt


class M0SampledAttemptArchiveTests(unittest.TestCase):
    def test_archive_records_identify_retained_bytes_instead_of_live_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.rs"
            source.write_bytes(b"original source")
            snapshot = root / "source.zip"
            with mock.patch.object(attempt, "ROOT", root):
                expected = attempt.source_records([Path("source.rs")])
                attempt.source_archive(snapshot, [Path("source.rs")])
                source.write_bytes(b"changed source")
                self.assertEqual(attempt.archived_source_records(snapshot), expected)
                self.assertNotEqual(
                    attempt.archived_source_records(snapshot),
                    attempt.source_records([Path("source.rs")]),
                )

    def test_archive_race_is_rejected_before_gpu_even_if_live_source_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.rs"
            source.write_bytes(b"original source")
            attempt_dir = root / "attempt"

            def inconsistent_archive(destination: Path, _: list[Path]) -> str:
                with zipfile.ZipFile(destination, "w") as archive:
                    archive.writestr("source.rs", b"transient source")
                return attempt.sha256(destination)

            with mock.patch.object(attempt, "ROOT", root), mock.patch.object(
                attempt, "source_files", return_value=[Path("source.rs")]
            ), mock.patch.object(attempt, "working_tree", return_value={"head": "head"}), mock.patch.object(
                attempt, "source_archive", side_effect=inconsistent_archive
            ), mock.patch.object(attempt.subprocess, "run") as run, mock.patch.object(
                sys, "argv", ["attempt", "--backend", "dx12", "--attempt-dir", str(attempt_dir)]
            ):
                self.assertEqual(attempt.main(), 2)
                run.assert_not_called()
            manifest = json.loads((attempt_dir / "attempt.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["gpuStarted"])
            self.assertNotEqual(manifest["sourceInputs"], manifest["archivedSourceInputs"])

    def test_actual_lock_is_the_integration_workspace_lock(self) -> None:
        self.assertEqual(
            attempt.INTEGRATION_LOCK,
            attempt.MANIFEST.parent / "Cargo.lock",
        )
        packages = attempt.resolved_git_packages()
        self.assertTrue(packages)
        self.assertTrue(all(item["source"].startswith("git+") for item in packages))

    def test_source_snapshot_detects_added_files_as_well_as_changed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "before.rs").write_text("before", encoding="utf-8")
            with mock.patch.object(attempt, "ROOT", root), mock.patch.object(
                attempt, "SOURCE_ROOTS", (Path("source"),)
            ):
                before = attempt.source_records(attempt.source_files())
                (source / "after.rs").write_text("after", encoding="utf-8")
                after = attempt.source_records(attempt.source_files())
            self.assertNotEqual(before, after)

    def test_uv0_pbr_uses_its_own_cli_selector(self) -> None:
        self.assertEqual(
            attempt.WORKLOAD_SELECTORS["m1-uv0-pbr-forward-deferred"],
            "--m1-uv0-pbr-only",
        )

    def test_environment_irradiance_uses_its_own_forward_selector(self) -> None:
        self.assertEqual(
            attempt.WORKLOAD_SELECTORS["m1-environment-irradiance-forward-deferred"],
            "--m1-environment-irradiance-forward-only",
        )

    def test_surface_effects_uses_its_own_dual_pipeline_selector(self) -> None:
        self.assertEqual(
            attempt.WORKLOAD_SELECTORS["m1-surface-effects-forward-deferred"],
            "--m1-surface-effects-only",
        )

    def test_uv0_pbr_input_fixture_is_valid_json_with_two_dimensional_probes(self) -> None:
        fixture = attempt.ROOT / "crates/renderer/examples/rendering-integration/assets/m1-uv0-pbr-input.json"
        value = json.loads(fixture.read_text(encoding="utf-8"))
        self.assertEqual(value["workload"], "m1-uv0-node-texture-pbr-forward-deferred")
        self.assertEqual(value["readbackOrigin"], "top-left Core row zero")
        self.assertEqual(value["probes"], [[8, 8], [40, 8], [8, 56], [40, 56]])

    def test_main_records_post_gpu_deletion_race_as_unstable_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.rs"
            source.write_text("before", encoding="utf-8")
            manifest_path = root / "Cargo.toml"
            manifest_path.write_text("[package]\nname = \"test-consumer\"\n", encoding="utf-8")
            lock = root / "Cargo.lock"
            lock.write_text("version = 4\npackage = []\n", encoding="utf-8")
            attempt_dir = root / "attempt"

            def fake_output(_: list[str]) -> str:
                return "head\n"

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                if command[:2] == ["cargo", "build"]:
                    executable = root / "target" / "test-consumer.exe"
                    executable.parent.mkdir(parents=True, exist_ok=True)
                    executable.write_bytes(b"compiled")
                    record = {
                        "reason": "compiler-artifact",
                        "manifest_path": str(manifest_path),
                        "executable": str(executable),
                        "target": {"name": "test-consumer", "kind": ["bin"]},
                    }
                    return subprocess.CompletedProcess(command, 0, json.dumps(record) + "\n", "")
                if command[0].endswith(".exe"):
                    source.unlink()
                    return subprocess.CompletedProcess(command, 0, "gpu output", "")
                if command[:3] == ["git", "diff", "--binary"]:
                    return subprocess.CompletedProcess(command, 0, b"", b"")
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(attempt, "ROOT", root), mock.patch.object(
                attempt, "MANIFEST", root / "Cargo.toml"
            ), mock.patch.object(attempt, "INTEGRATION_LOCK", lock), mock.patch.object(
                attempt, "source_files", return_value=[Path("source.rs")]
            ), mock.patch.object(attempt, "output", side_effect=fake_output), mock.patch.object(
                attempt, "resolved_git_packages", return_value=[]
            ), mock.patch.object(attempt.subprocess, "run", side_effect=fake_run), mock.patch.object(
                sys, "argv", ["attempt", "--backend", "dx12", "--attempt-dir", str(attempt_dir)]
            ):
                self.assertEqual(attempt.main(), 2)

            manifest = json.loads((attempt_dir / "attempt.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["sourceStable"])
            self.assertEqual(manifest["runnerExitStatus"], 0)
            self.assertEqual(
                manifest["sourceErrors"]["after"],
                [{"path": "source.rs", "error": "missing-during-hash"}],
            )

    def test_runner_rejects_retained_executable_replacement_after_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.rs"
            source.write_bytes(b"source")
            manifest_path = root / "Cargo.toml"
            manifest_path.write_text("[package]\nname = \"test-consumer\"\n", encoding="utf-8")
            (root / "Cargo.lock").write_text("version = 4\npackage = []\n", encoding="utf-8")
            binary = root / "target" / "test-consumer.exe"
            attempt_dir = root / "attempt"

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                if command[:2] == ["cargo", "build"]:
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    binary.write_bytes(b"compiled")
                    record = {
                        "reason": "compiler-artifact",
                        "manifest_path": str(manifest_path),
                        "executable": str(binary),
                        "target": {"name": "test-consumer", "kind": ["bin"]},
                    }
                    return subprocess.CompletedProcess(command, 0, json.dumps(record) + "\n", "")
                Path(command[0]).write_bytes(b"tampered retained binary")
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(attempt, "ROOT", root), mock.patch.object(
                attempt, "MANIFEST", manifest_path
            ), mock.patch.object(attempt, "INTEGRATION_LOCK", root / "Cargo.lock"), mock.patch.object(
                attempt, "source_files", return_value=[Path("source.rs")]
            ), mock.patch.object(attempt, "working_tree", return_value={"head": "head"}), mock.patch.object(
                attempt, "resolved_git_packages", return_value=[]
            ), mock.patch.object(attempt.subprocess, "run", side_effect=fake_run), mock.patch.object(
                sys, "argv", ["attempt", "--backend", "dx12", "--attempt-dir", str(attempt_dir)]
            ):
                self.assertEqual(attempt.main(), 2)
            record = json.loads((attempt_dir / "attempt.json").read_text(encoding="utf-8"))
            self.assertNotEqual(record["executable"]["sha256"], record["executable"]["afterRunSha256"])
            self.assertEqual(record["commands"][1]["kind"], "directExecutable")

    def test_missing_cargo_executable_keeps_a_failed_attempt_without_running_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source.rs").write_bytes(b"source")
            manifest_path = root / "Cargo.toml"
            manifest_path.write_text("[package]\nname = \"test-consumer\"\n", encoding="utf-8")
            (root / "Cargo.lock").write_text("version = 4\npackage = []\n", encoding="utf-8")
            attempt_dir = root / "attempt"
            with mock.patch.object(attempt, "ROOT", root), mock.patch.object(
                attempt, "MANIFEST", manifest_path
            ), mock.patch.object(attempt, "INTEGRATION_LOCK", root / "Cargo.lock"), mock.patch.object(
                attempt, "source_files", return_value=[Path("source.rs")]
            ), mock.patch.object(attempt, "working_tree", return_value={"head": "head"}), mock.patch.object(
                attempt, "resolved_git_packages", return_value=[]
            ), mock.patch.object(
                attempt.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, "not a Cargo artifact\n", ""),
            ) as run, mock.patch.object(
                sys, "argv", ["attempt", "--backend", "dx12", "--attempt-dir", str(attempt_dir)]
            ):
                self.assertEqual(attempt.main(), 2)
            run.assert_called_once()
            record = json.loads((attempt_dir / "attempt.json").read_text(encoding="utf-8"))
            self.assertFalse(record["gpuStarted"])
            self.assertIsNone(record["executable"])
            self.assertIn("did not report an executable", record["commands"][1]["runnerError"])
            self.assertTrue((attempt_dir / "build-stdout.log").is_file())
            self.assertTrue((attempt_dir / "stderr.log").is_file())
