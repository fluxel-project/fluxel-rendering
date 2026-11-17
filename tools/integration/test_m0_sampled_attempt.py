"""Focused contract tests for M0 sampled-material attempt evidence."""

from __future__ import annotations

import sys
import tempfile
import unittest
import json
import subprocess
import zipfile
from pathlib import Path
from unittest import mock


TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

import m0_sampled_attempt as attempt


class M0SampledAttemptTests(unittest.TestCase):
    def test_find_built_executable_accepts_only_selected_binary_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "Cargo.toml"
            manifest.write_text("[package]\nname = \"selected-consumer\"\n", encoding="utf-8")
            expected = root / "selected.exe"
            expected.write_bytes(b"selected")
            other = root / "other.exe"
            other.write_bytes(b"other")
            records = [
                {
                    "reason": "compiler-artifact",
                    "manifest_path": str(root / "other.toml"),
                    "executable": str(other),
                    "target": {"name": "selected-consumer", "kind": ["bin"]},
                },
                {
                    "reason": "compiler-artifact",
                    "manifest_path": str(manifest),
                    "executable": str(expected),
                    "target": {"name": "selected-consumer", "kind": ["bin"]},
                },
            ]
            self.assertEqual(
                attempt.find_built_executable("\n".join(json.dumps(item) for item in records), manifest),
                expected,
            )

    def test_retain_executable_copies_the_reported_binary_with_its_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "cargo-output.exe"
            source.write_bytes(b"compiled binary")
            evidence = attempt.retain_executable(source, root / "attempt")
            retained = root / "attempt" / evidence["path"]
            self.assertEqual(retained.read_bytes(), b"compiled binary")
            self.assertEqual(evidence["sha256"], attempt.sha256(source))
            self.assertEqual(evidence["cargoReportedPath"], str(source))

    def test_e0_uses_its_own_manifest_lock_and_asset_argument(self) -> None:
        for backend, workload in ((b, w) for b in ("dx12", "native-gl-wgl")
                                  for w in ("e0-texture-material-forward-deferred", "e0-texture-material-surface",
                                            "m1-srgb-filter-forward-deferred", "m1-srgb-filter-surface",
                                            "m1-material-bindings-forward-deferred", "m1-material-bindings-surface")):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                example = root / "crates/renderer/examples/02-texture-material"
                example.mkdir(parents=True)
                manifest_path = example / "Cargo.toml"
                manifest_path.write_text("[package]\nname = \"renderer-02-texture-material\"\n", encoding="utf-8")
                lock = example / "Cargo.lock"
                lock.write_text("version = 4\npackage = []\n", encoding="utf-8")
                (root / "Cargo.lock").write_text("context only", encoding="utf-8")
                asset = root / "texture.ktx"
                asset.write_bytes(b"input")
                source = root / "source.rs"
                source.write_bytes(b"fixed source")
                destination = root / "attempt"
                binary = root / "target" / "renderer-02-texture-material.exe"

                def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                    if command[:2] == ["cargo", "build"]:
                        binary.parent.mkdir(parents=True, exist_ok=True)
                        binary.write_bytes(b"compiled")
                        record = {
                            "reason": "compiler-artifact",
                            "manifest_path": str(manifest_path),
                            "executable": str(binary),
                            "target": {"name": "renderer-02-texture-material", "kind": ["bin"]},
                        }
                        return subprocess.CompletedProcess(command, 0, json.dumps(record) + "\n", "")
                    return subprocess.CompletedProcess(command, 0, "ran retained binary", "")

                with mock.patch.object(attempt, "ROOT", root), mock.patch.object(
                    attempt, "working_tree", return_value={"head": "head"}
                ), mock.patch.object(attempt, "source_files", return_value=[Path("source.rs")]), mock.patch.object(
                    attempt.subprocess, "run", side_effect=fake_run
                ) as run, mock.patch.object(
                    sys, "argv", ["attempt", "--backend", backend, "--workload",
                                  workload,
                                  *(["--asset", str(asset)] if workload.startswith("e0-") else []),
                                  "--attempt-dir", str(destination)]
                ):
                    self.assertEqual(attempt.main(), 0)
                build_command = run.call_args_list[0].args[0]
                command = run.call_args_list[1].args[0]
                self.assertEqual(Path(build_command[build_command.index("--manifest-path") + 1]).as_posix(),
                                 "crates/renderer/examples/02-texture-material/Cargo.toml")
                self.assertEqual(build_command[build_command.index("--features") + 1], backend)
                self.assertIn("--visible" if workload.endswith("surface") else "--offscreen", command)
                self.assertEqual("--exit-after-capture" in command, workload.endswith("surface"))
                if workload.startswith("e0-"):
                    self.assertEqual(command[command.index("--asset") + 1], str(asset.resolve()))
                    self.assertNotIn("--srgb-filter", command)
                else:
                    self.assertIn("--srgb-filter", command)
                    self.assertNotIn("--asset", command)
                self.assertEqual("--permuted-bindings" in command, workload.startswith("m1-material-bindings-"))
                manifest = json.loads((destination / "attempt.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["locks"][0]["sha256"], attempt.sha256(lock))
                self.assertEqual(manifest["locks"][0]["path"],
                                 "crates/renderer/examples/02-texture-material/Cargo.lock")
                self.assertEqual(manifest["executable"]["sha256"], attempt.sha256(binary))
                self.assertEqual(manifest["executable"]["afterRunSha256"], attempt.sha256(binary))
                self.assertTrue((destination / manifest["executable"]["path"]).is_file())

    def test_dynamic_uniform_uses_its_own_lock_and_bounded_surface_selector(self) -> None:
        for backend, workload in ((b, w) for b in ("dx12", "native-gl-wgl")
                                  for w in ("e0-dynamic-uniform-material-forward", "e0-dynamic-uniform-material-surface")):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                example = root / "crates/renderer/examples/04-dynamic-uniform-material"
                example.mkdir(parents=True)
                manifest_path = example / "Cargo.toml"
                manifest_path.write_text("[package]\nname = \"renderer-04-dynamic-uniform-material\"\n", encoding="utf-8")
                lock = example / "Cargo.lock"
                lock.write_text("version = 4\npackage = []\n", encoding="utf-8")
                (root / "Cargo.lock").write_text("context only", encoding="utf-8")
                asset = root / "texture.ktx"
                asset.write_bytes(b"input")
                source = root / "source.rs"
                source.write_bytes(b"fixed source")
                destination = root / "attempt"
                binary = root / "target" / "renderer-04-dynamic-uniform-material.exe"

                def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                    if command[:2] == ["cargo", "build"]:
                        binary.parent.mkdir(parents=True, exist_ok=True)
                        binary.write_bytes(b"compiled")
                        record = {
                            "reason": "compiler-artifact",
                            "manifest_path": str(manifest_path),
                            "executable": str(binary),
                            "target": {"name": "renderer-04-dynamic-uniform-material", "kind": ["bin"]},
                        }
                        return subprocess.CompletedProcess(command, 0, json.dumps(record) + "\n", "")
                    return subprocess.CompletedProcess(command, 0, "ran retained binary", "")

                with mock.patch.object(attempt, "ROOT", root), mock.patch.object(
                    attempt, "working_tree", return_value={"head": "head"}
                ), mock.patch.object(attempt, "source_files", return_value=[Path("source.rs")]), mock.patch.object(
                    attempt.subprocess, "run", side_effect=fake_run
                ) as run, mock.patch.object(
                    sys, "argv", ["attempt", "--backend", backend, "--workload",
                                  workload,
                                  "--attempt-dir", str(destination)]
                ):
                    self.assertEqual(attempt.main(), 0)
                build_command = run.call_args_list[0].args[0]
                command = run.call_args_list[1].args[0]
                self.assertEqual(Path(build_command[build_command.index("--manifest-path") + 1]).as_posix(),
                                 "crates/renderer/examples/04-dynamic-uniform-material/Cargo.toml")
                self.assertEqual(build_command[build_command.index("--features") + 1], backend)
                self.assertIn("--visible" if workload.endswith("surface") else "--offscreen", command)
                self.assertEqual("--exit-after-capture" in command, workload.endswith("surface"))
                self.assertNotIn("--asset", command)
                self.assertNotIn("--dragon", command)
                manifest = json.loads((destination / "attempt.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["locks"][0]["sha256"], attempt.sha256(lock))
                self.assertEqual(manifest["locks"][0]["path"],
                                 "crates/renderer/examples/04-dynamic-uniform-material/Cargo.lock")
                self.assertEqual(manifest["executable"]["sha256"], attempt.sha256(binary))
                self.assertEqual(manifest["executable"]["afterRunSha256"], attempt.sha256(binary))
                self.assertTrue((destination / manifest["executable"]["path"]).is_file())
    def test_standalone_material_examples_use_their_own_lock_and_bounded_surface_selector(self) -> None:
        for backend, workload in ((b, w) for b in ("dx12", "native-gl-wgl")
                                  for w in ("m1-specular-ibl-forward-deferred", "m1-specular-ibl-surface",
                                            "m1-orm-ibl-sphere-forward-deferred", "m1-orm-ibl-sphere-surface",
                                            "m1-orm-ibl-multi-object-forward-deferred", "m1-orm-ibl-multi-object-surface",
                                            "m1-four-texture-forward-deferred", "m1-four-texture-surface",
                                            "m1-mixed-capture-forward", "m1-mixed-capture-surface",
                                            "m1-material-mips-forward-deferred", "m1-material-mips-surface",
                                            "m1-mipmap-tunnel-forward-deferred", "m1-mipmap-tunnel-surface")):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                tunnel = workload.startswith("m1-mipmap-tunnel-")
                example_name = ("09-mipmap-tunnel-material" if tunnel else "pbr-four-texture-material" if workload.startswith(("m1-four-texture-", "m1-mixed-capture-", "m1-material-mips-")) else
                                "pbr-sphere-material" if workload.startswith("m1-orm-ibl-") else "pbr-environment-material")
                package_name = f"renderer-{example_name}"
                example = root / "crates/renderer/examples" / example_name
                example.mkdir(parents=True)
                manifest_path = example / "Cargo.toml"
                manifest_path.write_text(f'[package]\nname = "{package_name}"\n', encoding="utf-8")
                lock = example / "Cargo.lock"
                lock.write_text("version = 4\npackage = []\n", encoding="utf-8")
                (root / "Cargo.lock").write_text("context only", encoding="utf-8")
                asset = root / "texture.ktx"
                asset.write_bytes(b"input")
                model = root / "tunnel.gltf"
                model.write_bytes(b"model")
                source = root / "source.rs"
                source.write_bytes(b"fixed source")
                destination = root / "attempt"
                binary = root / "target" / f"{package_name}.exe"

                def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                    if command[:2] == ["cargo", "build"]:
                        binary.parent.mkdir(parents=True, exist_ok=True)
                        binary.write_bytes(b"compiled")
                        record = {
                            "reason": "compiler-artifact",
                            "manifest_path": str(manifest_path),
                            "executable": str(binary),
                            "target": {"name": package_name, "kind": ["bin"]},
                        }
                        return subprocess.CompletedProcess(command, 0, json.dumps(record) + "\n", "")
                    return subprocess.CompletedProcess(command, 0, "ran retained binary", "")

                with mock.patch.object(attempt, "ROOT", root), mock.patch.object(
                    attempt, "working_tree", return_value={"head": "head"}
                ), mock.patch.object(attempt, "source_files", return_value=[Path("source.rs")]), mock.patch.object(
                    attempt.subprocess, "run", side_effect=fake_run
                ) as run, mock.patch.object(
                    sys, "argv", ["attempt", "--backend", backend, "--workload",
                                  workload,
                                  "--attempt-dir", str(destination)] +
                                 (["--asset", str(asset), "--model", str(model)] if tunnel else [])
                ):
                    self.assertEqual(attempt.main(), 0)
                build_command = run.call_args_list[0].args[0]
                command = run.call_args_list[1].args[0]
                self.assertEqual(Path(build_command[build_command.index("--manifest-path") + 1]).as_posix(),
                                 f"crates/renderer/examples/{example_name}/Cargo.toml")
                self.assertEqual(build_command[build_command.index("--features") + 1], backend)
                self.assertIn("--visible" if workload.endswith("surface") else "--offscreen", command)
                self.assertEqual("--exit-after-capture" in command, workload.endswith("surface"))
                self.assertEqual("--multi-records" in command, workload.startswith("m1-orm-ibl-multi-object-"))
                self.assertEqual("--mixed-capture" in command, workload.startswith("m1-mixed-capture-"))
                self.assertEqual("--mip-generation" in command, workload.startswith("m1-material-mips-"))
                self.assertEqual("--asset" in command, tunnel)
                self.assertEqual("--model" in command, tunnel)
                if tunnel:
                    self.assertEqual(command[command.index("--asset") + 1], str(asset.resolve()))
                    self.assertEqual(command[command.index("--model") + 1], str(model.resolve()))
                self.assertNotIn("--dragon", command)
                manifest = json.loads((destination / "attempt.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["locks"][0]["sha256"], attempt.sha256(lock))
                self.assertEqual(manifest["locks"][0]["path"],
                                 f"crates/renderer/examples/{example_name}/Cargo.lock")
                self.assertEqual(manifest["executable"]["sha256"], attempt.sha256(binary))
                self.assertEqual(manifest["executable"]["afterRunSha256"], attempt.sha256(binary))
                self.assertTrue((destination / manifest["executable"]["path"]).is_file())

    def test_capture_uses_its_own_lock_and_both_model_arguments(self) -> None:
        for backend in ("dx12", "native-gl-wgl"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                example = root / "crates/renderer/examples/07-offscreen-material"
                example.mkdir(parents=True)
                manifest_path = example / "Cargo.toml"
                manifest_path.write_text("[package]\nname = \"renderer-07-offscreen-material\"\n", encoding="utf-8")
                lock = example / "Cargo.lock"
                lock.write_text("version = 4\npackage = []\n", encoding="utf-8")
                (root / "Cargo.lock").write_text("context only", encoding="utf-8")
                asset = root / "dragon.gltf"
                asset.write_bytes(b"input")
                source = root / "source.rs"
                source.write_bytes(b"fixed source")
                destination = root / "attempt"
                binary = root / "target" / "renderer-07-offscreen-material.exe"

                def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                    if command[:2] == ["cargo", "build"]:
                        binary.parent.mkdir(parents=True, exist_ok=True)
                        binary.write_bytes(b"compiled")
                        record = {
                            "reason": "compiler-artifact",
                            "manifest_path": str(manifest_path),
                            "executable": str(binary),
                            "target": {"name": "renderer-07-offscreen-material", "kind": ["bin"]},
                        }
                        return subprocess.CompletedProcess(command, 0, json.dumps(record) + "\n", "")
                    return subprocess.CompletedProcess(command, 0, "ran retained binary", "")

                with mock.patch.object(attempt, "ROOT", root), mock.patch.object(
                    attempt, "working_tree", return_value={"head": "head"}
                ), mock.patch.object(attempt, "source_files", return_value=[Path("source.rs")]), mock.patch.object(
                    attempt.subprocess, "run", side_effect=fake_run
                ) as run, mock.patch.object(
                    sys, "argv", ["attempt", "--backend", backend, "--workload",
                                  "e0-capture-material-forward", "--dragon", str(asset), "--plane", str(asset),
                                  "--attempt-dir", str(destination)]
                ):
                    self.assertEqual(attempt.main(), 0)
                build_command = run.call_args_list[0].args[0]
                command = run.call_args_list[1].args[0]
                self.assertEqual(Path(build_command[build_command.index("--manifest-path") + 1]).as_posix(),
                                 "crates/renderer/examples/07-offscreen-material/Cargo.toml")
                self.assertEqual(build_command[build_command.index("--features") + 1], backend)
                self.assertIn("--offscreen", command)
                self.assertEqual(command[command.index("--dragon") + 1], str(asset.resolve()))
                manifest = json.loads((destination / "attempt.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["locks"][0]["sha256"], attempt.sha256(lock))
                self.assertEqual(manifest["locks"][0]["path"],
                                 "crates/renderer/examples/07-offscreen-material/Cargo.lock")
                self.assertEqual(manifest["executable"]["sha256"], attempt.sha256(binary))
                self.assertEqual(manifest["executable"]["afterRunSha256"], attempt.sha256(binary))
                self.assertTrue((destination / manifest["executable"]["path"]).is_file())

    def test_capture_surface_runs_bounded_visible_mode_with_its_own_lock(self) -> None:
        for backend in ("dx12", "native-gl-wgl"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                example = root / "crates/renderer/examples/07-offscreen-material"
                example.mkdir(parents=True)
                manifest_path = example / "Cargo.toml"
                manifest_path.write_text("[package]\nname = \"renderer-07-offscreen-material\"\n", encoding="utf-8")
                lock = example / "Cargo.lock"
                lock.write_text("version = 4\npackage = []\n", encoding="utf-8")
                (root / "Cargo.lock").write_text("context only", encoding="utf-8")
                asset = root / "dragon.gltf"
                asset.write_bytes(b"input")
                source = root / "source.rs"
                source.write_bytes(b"fixed source")
                destination = root / "attempt"
                binary = root / "target" / "renderer-07-offscreen-material.exe"

                def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                    if command[:2] == ["cargo", "build"]:
                        binary.parent.mkdir(parents=True, exist_ok=True)
                        binary.write_bytes(b"compiled")
                        record = {
                            "reason": "compiler-artifact",
                            "manifest_path": str(manifest_path),
                            "executable": str(binary),
                            "target": {"name": "renderer-07-offscreen-material", "kind": ["bin"]},
                        }
                        return subprocess.CompletedProcess(command, 0, json.dumps(record) + "\n", "")
                    workload_dir = destination / "workload"
                    workload_dir.mkdir(exist_ok=True)
                    if backend == "dx12":
                        payload = {"schemaVersion": 1, "requested": True,
                                   "deviceOpenCompleted": True,
                                   "activationContract": "fluxel-rhi 8400f8f35933a5d0464014a8671799890f67ff49"}
                        name = "dx12-debug-request.json"
                    else:
                        payload = {"schemaVersion": 1, "installed": True,
                                   "debugOutputAvailable": True, "activationMarkerObserved": True,
                                   "logSaturated": False,
                                   "activationMessages": ['DebugMessageLogEntry { source: 33354, msg_type: 33384, id: 1180190028, severity: 33387, message: "fluxel-rendering WGL diagnostic observer active" }'],
                                   "nativeDiagnostics": []}
                        name = "wgl-debug-observer.json"
                    (workload_dir / name).write_text(json.dumps(payload), encoding="utf-8")
                    return subprocess.CompletedProcess(command, 0, "ran retained binary", "")

                with mock.patch.object(attempt, "ROOT", root), mock.patch.object(
                    attempt, "working_tree", return_value={"head": "head"}
                ), mock.patch.object(attempt, "source_files", return_value=[Path("source.rs")]), mock.patch.object(
                    attempt.subprocess, "run", side_effect=fake_run
                ) as run, mock.patch.object(
                    sys, "argv", ["attempt", "--backend", backend, "--workload",
                                  "e0-capture-material-surface", "--validate", "--dragon", str(asset), "--plane", str(asset),
                                  "--attempt-dir", str(destination)]
                ):
                    self.assertEqual(attempt.main(), 0)
                build_command = run.call_args_list[0].args[0]
                command = run.call_args_list[1].args[0]
                self.assertEqual(Path(build_command[build_command.index("--manifest-path") + 1]).as_posix(),
                                 "crates/renderer/examples/07-offscreen-material/Cargo.toml")
                self.assertEqual(build_command[build_command.index("--features") + 1], backend)
                self.assertIn("--visible", command)
                self.assertIn("--exit-after-capture", command)
                runtime_env = run.call_args_list[1].kwargs["env"]
                self.assertEqual(runtime_env.get("FLUXEL_DX12_DEBUG"), "1" if backend == "dx12" else None)
                manifest = json.loads((destination / "attempt.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["validation"]["status"], "observed")
                self.assertIn("sha256", manifest["validation"])
                self.assertEqual(command[command.index("--dragon") + 1], str(asset.resolve()))
                manifest = json.loads((destination / "attempt.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["locks"][0]["sha256"], attempt.sha256(lock))
                self.assertEqual(manifest["locks"][0]["path"],
                                 "crates/renderer/examples/07-offscreen-material/Cargo.lock")
                self.assertEqual(manifest["executable"]["sha256"], attempt.sha256(binary))
                self.assertEqual(manifest["executable"]["afterRunSha256"], attempt.sha256(binary))
                self.assertTrue((destination / manifest["executable"]["path"]).is_file())

if __name__ == "__main__":
    unittest.main()
