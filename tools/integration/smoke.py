#!/usr/bin/env python3
"""Retain bounded S0/S1 integration attempts without asserting support."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib
import zipfile
from typing import Any


SCHEMA_VERSION = 2
BACKENDS = ("dx12", "native-gl-wgl")
RHI_REVISION = "0f7f785d4f0ccc62e125dd6cf560d91333ec85cf"
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "examples" / "rendering-integration"


def classify_dx12_diagnostics(stderr: Path) -> dict[str, Any]:
    """Retain every native diagnostic; the rebuilt copy workload has no waivers."""
    lines = stderr.read_text(encoding="utf-8", errors="replace").splitlines()
    native = [line for line in lines if line.startswith("D3D12 [")]
    return {
        "native_callback_observed": bool(native),
        "unreviewed_native_diagnostics": native,
        "policy": "tools/integration/README.md",
        "policy_sha256": sha256(ROOT / "tools/integration/README.md"),
    }


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(directory: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(directory), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_consistency(inputs: list[dict[str, Any]], root: Path = ROOT) -> dict[str, Any]:
    """Detect edits during an attempt instead of attributing a mixed build."""
    before = {str(Path(item["path"])): item["sha256"] for item in inputs}
    candidates = set(before)
    source_directories = {
        (Path(name).parent, Path(name).suffix)
        for name in before if Path(name).suffix in {".rs", ".wgsl", ".py"}
    }
    for directory, suffix in source_directories:
        candidates.update(str(path.relative_to(root)) for path in (root / directory).rglob("*" + suffix))
    changed = []
    for name in sorted(candidates):
        after = sha256(root / name)
        if after != before.get(name):
            changed.append({"path": name, "before_sha256": before.get(name), "after_sha256": after})
    return {"status": "changed" if changed else "stable", "changed_inputs": changed}


def checkout_inventory(directory: Path) -> dict[str, Any]:
    status = git_value(directory, "status", "--porcelain")
    return {
        "path": str(directory),
        "revision": git_value(directory, "rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "role": "local_checkout_inventory",
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_attempts(path: Path) -> list[dict[str, Any]]:
    try:
        attempts = json.loads(path.read_text(encoding="utf-8")).get("attempts", [])
        return attempts if isinstance(attempts, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def attempt_name(backend: str) -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{backend}-{timestamp}-{os.getpid()}"


def log_result(log_directory: Path, exit_status: int | None, timed_out: bool,
               lifecycle: str, error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exit_status": exit_status,
        "timed_out": timed_out,
        "child_process_lifecycle": lifecycle,
        "raw_logs": [
            {"path": str(log_directory / "stdout.log"), "sha256": sha256(log_directory / "stdout.log")},
            {"path": str(log_directory / "stderr.log"), "sha256": sha256(log_directory / "stderr.log")},
        ],
    }
    if error is not None:
        result["runner_error"] = error
    return result


def run_command(command: list[str], cwd: Path, timeout: float,
                log_directory: Path, environment: dict[str, str] | None = None) -> dict[str, Any]:
    """Write straight to files; a timeout leaves descendants explicitly unknown."""
    log_directory.mkdir(parents=True, exist_ok=True)
    stdout_path = log_directory / "stdout.log"
    stderr_path = log_directory / "stderr.log"
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                text=True,
                stdout=stdout,
                stderr=stderr,
                creationflags=creation_flags,
                env=environment,
            )
        except OSError as error:
            stderr.write(f"runner error: {error}\n")
            stderr.flush()
            return log_result(log_directory, None, False, "not_started", str(error))
        try:
            exit_status = process.wait(timeout=timeout)
            return log_result(log_directory, exit_status, False, "parent_exited")
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stderr.write(
                "runner timeout: parent termination was requested; descendant process "
                "lifecycle is unverified.\n"
            )
            stderr.flush()
            return log_result(log_directory, None, True, "unverified_after_parent_termination")


def retained_files(directory: Path) -> list[dict[str, str]]:
    """List immutable attempt artifacts without interpreting their contents."""
    return [
        {"path": str(path), "sha256": sha256(path) or ""}
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def freeze_source_inputs(inputs: list[dict[str, Any]], destination: Path,
                         root: Path = ROOT) -> dict[str, Any]:
    """Retain the exact dirty-worktree inputs, refusing a capture-time edit."""
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in inputs:
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"source snapshot path is outside root: {relative}")
            data = (root / relative).read_bytes()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"source changed while freezing: {relative}")
            archive.writestr(relative.as_posix(), data)
        archive.writestr("snapshot-inputs.json", json.dumps(inputs, sort_keys=True, indent=2))
    return {"path": str(destination), "sha256": sha256(destination),
            "files": len(inputs), "kind": "exact_worktree_inputs"}


def validation_evidence(backend: str, run_directory: Path, requested: str) -> dict[str, Any]:
    """Classify only concrete diagnostics artifacts from this attempt."""
    if requested != "enabled":
        return {"requested": requested, "status": "disabled", "reason": "runner disabled validation"}
    if backend == "dx12":
        marker = run_directory / "dx12-debug-request.json"
        if not marker.is_file():
            return {"requested": requested, "status": "unverified", "reason": "DX12 device-open diagnostic marker is absent"}
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"requested": requested, "status": "unverified", "reason": "DX12 diagnostic marker is malformed"}
        contract = record.get("activationContract")
        if (record.get("requested") is True and record.get("deviceOpenCompleted") is True
                and isinstance(contract, str) and RHI_REVISION in contract
                and "EnableDebugLayer" in contract):
            return {"requested": requested, "status": "observed", "reason": "the selected DX12 device opened after the pinned RHI debug-layer activation branch", "artifact": {"path": str(marker), "sha256": sha256(marker)}, "activation_contract": record.get("activationContract")}
        return {"requested": requested, "status": "unverified", "reason": "DX12 device-open marker did not record a debug request"}
    markers = sorted(run_directory.rglob("wgl-debug-observer.json"))
    primary = run_directory / "wgl-debug-observer.json"
    if primary not in markers:
        return {"requested": requested, "status": "unverified", "reason": "WGL KHR_debug observer artifact is absent"}
    diagnostic_count = 0
    for marker in markers:
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"requested": requested, "status": "unverified", "reason": f"WGL observer artifact is malformed: {marker}"}
        messages = record.get("activationMessages", [])
        marker_returned = isinstance(messages, list) and any(
            isinstance(message, str) and "fluxel-rendering WGL diagnostic observer active" in message
            for message in messages
        )
        if not (record.get("installed") is True and record.get("activationMarkerObserved") is True
                and marker_returned and record.get("logSaturated") is False
                and isinstance(record.get("nativeDiagnostics"), list)):
            return {"requested": requested, "status": "unverified", "reason": f"WGL observer did not activate: {marker}"}
        diagnostic_count += len(record["nativeDiagnostics"])
    return {"requested": requested, "status": "observed", "reason": "the exact WGL contexts returned their observer application markers",
            "artifact": {"path": str(primary), "sha256": sha256(primary)},
            "additional_artifacts": [{"path": str(marker), "sha256": sha256(marker)} for marker in markers if marker != primary],
            "native_diagnostic_count": diagnostic_count}


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolved_git_packages(lock: Path) -> list[dict[str, str]]:
    """Cargo.lock, rather than a neighbouring checkout, identifies Git sources."""
    if not lock.is_file():
        return []
    result = []
    for package in tomllib.loads(lock.read_text(encoding="utf-8")).get("package", []):
        source = package.get("source")
        if isinstance(source, str) and source.startswith("git+"):
            result.append({"name": package["name"], "version": package["version"], "source": source})
    return result


def find_built_executable(build_stdout: Path, manifest: Path) -> str | None:
    """Extract Cargo's actual binary path, including CARGO_TARGET_DIR overrides."""
    package_name = tomllib.loads(manifest.read_text(encoding="utf-8"))["package"]["name"]
    executable = None
    for line in build_stdout.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") != "compiler-artifact":
            continue
        reported_manifest = message.get("manifest_path")
        if (
            message.get("executable")
            and reported_manifest
            and Path(reported_manifest).resolve() == manifest.resolve()
            and message.get("target", {}).get("name") == package_name
            and "bin" in message.get("target", {}).get("kind", [])
        ):
            executable = message["executable"]
    return executable


def retain_executable(source: Path, attempt_directory: Path) -> dict[str, Any]:
    """Keep the executed binary after another backend overwrites Cargo's output."""
    digest = sha256(source)
    if digest is None:
        raise ValueError(f"Cargo executable is missing: {source}")
    destination = attempt_directory / ("consumer" + source.suffix)
    shutil.copy2(source, destination)
    if sha256(destination) != digest or sha256(source) != digest:
        raise ValueError("Cargo executable changed while retaining the attempt binary")
    return {"path": str(destination), "sha256": digest, "cargo_reported_path": str(source)}


def built_fluxel_packages(build_stdout: Path) -> list[dict[str, Any]]:
    """Retain Cargo's actual compiled package identities and active features."""
    result = []
    for line in build_stdout.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        package_id = message.get("package_id", "")
        target = message.get("target", {})
        target_name = target.get("name", "")
        if message.get("reason") != "compiler-artifact" or not (
            target_name.startswith("fluxel_") or "github.com/fluxel-project/" in package_id
        ):
            continue
        fragment = package_id.partition("#")[2]
        name = fragment.rpartition("@")[0] if "@" in fragment else target_name.replace("_", "-")
        result.append({
            "name": name, "package_id": package_id,
            "manifest_path": message.get("manifest_path"),
            "target": target_name, "target_kind": target.get("kind", []),
            "features": sorted(message.get("features", [])),
            "fresh": message.get("fresh"), "filenames": message.get("filenames", []),
        })
    return result


def verify_built_sources(inventory: list[dict[str, Any]], locked: list[dict[str, str]],
                         required: set[str] | None = None) -> dict[str, Any]:
    """Reject path overrides of the selected Git dependencies in actual builds."""
    expected: dict[str, set[str]] = {}
    for package in locked:
        expected.setdefault(package["name"], set()).add(package["source"].partition("#")[0])
    mismatches = []
    present = set()
    for package in inventory:
        name = package["name"]
        present.add(name)
        if name in expected and package["package_id"].partition("#")[0] not in expected[name]:
            mismatches.append({"name": name, "actual": package["package_id"], "expected": sorted(expected[name])})
        if name == "fluxel-renderer" and (
            not package.get("manifest_path") or
            Path(package["manifest_path"]).resolve() != (ROOT / "crates/renderer/Cargo.toml").resolve()
        ):
            mismatches.append({"name": name, "actual_manifest": package.get("manifest_path"), "reason": "renderer is outside this source snapshot"})
    if required is None:
        required = {"fluxel-assets", "fluxel-rhi", "fluxel-renderer", "fluxel-host"}
    missing = sorted(required - present)
    return {"status": "verified" if not mismatches and not missing else "mismatch",
            "mismatches": mismatches, "missing_required_packages": missing}


def verify_git_checkouts(inventory: list[dict[str, Any]],
                         locked: list[dict[str, str]]) -> dict[str, Any]:
    """Verify the actual Cargo Git trees, not just their package-ID strings."""
    revisions: dict[str, set[str]] = {}
    for package in locked:
        revisions.setdefault(package["name"], set()).add(package["source"].partition("#")[2])
    trees: dict[str, dict[str, Any]] = {}
    mismatches = []
    for package in inventory:
        if package["name"] not in revisions:
            continue
        manifest = package.get("manifest_path")
        root = git_value(Path(manifest).parent, "rev-parse", "--show-toplevel") if manifest else None
        if root is None:
            mismatches.append({"name": package["name"], "reason": "built Git checkout is unavailable"})
            continue
        if root not in trees:
            status = git_value(Path(root), "status", "--porcelain", "--untracked-files=all")
            # Cargo writes this exact marker after checking out a Git source.
            changes = None if status is None else [line for line in status.splitlines()
                                                    if line != "?? .cargo-ok"]
            trees[root] = {"path": root, "revision": git_value(Path(root), "rev-parse", "HEAD"),
                           "changes": changes}
        tree = trees[root]
        if tree["revision"] not in revisions[package["name"]] or tree["changes"] != []:
            mismatches.append({"name": package["name"], "checkout": root,
                               "expected_revisions": sorted(revisions[package["name"]])})
    if not trees:
        mismatches.append({"reason": "no built Git checkout was inspected"})
    return {"status": "verified" if not mismatches else "mismatch",
            "checkouts": list(trees.values()), "mismatches": mismatches}


def required_built_packages(backend: str) -> set[str]:
    """Every selected S0/S1 route creates a Host-owned native window Surface."""
    if backend not in BACKENDS:
        raise ValueError(f"unsupported integration backend: {backend}")
    return {"fluxel-assets", "fluxel-rhi", "fluxel-renderer", "fluxel-host"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--mode", choices=("build", "run"), default="run")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--validation", choices=("enabled", "disabled"), default="enabled")
    args = parser.parse_args()
    artifact_manifest = ARTIFACT / "Cargo.toml"
    artifact_package = (
        tomllib.loads(artifact_manifest.read_text(encoding="utf-8"))
        if artifact_manifest.is_file() else {}
    )
    if artifact_package.get("package", {}).get("metadata", {}).get("fluxel", {}).get("status") == "retired-after-renderer-reset":
        parser.error("the previous integration consumer targets removed Renderer APIs; rebuild it before running smoke evidence")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    rendering = checkout_inventory(ROOT)
    revision = rendering["revision"] or "unknown-rendering-revision"
    closure = ROOT / "target" / "integration" / "s0-s1" / revision
    index = closure / "manifest.json"
    attempt_dir = closure / "attempts" / attempt_name(args.backend)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    print(f"Retaining attempt {attempt_dir.name}: {attempt_dir}", flush=True)
    artifact_manifest = ARTIFACT / "Cargo.toml"
    artifact_lock = ARTIFACT / "Cargo.lock"
    attempt: dict[str, Any] = {
        "id": attempt_dir.name,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "backend": args.backend,
        "mode": args.mode,
        "feature": args.backend,
        "workload": {
            "selection": "copy+raster+compute+read-set+surface+device-replacement",
            "submissions_before_wait": 3,
            "copy_source_byte_lengths": [23, 37, 22],
            "copy_source_ranges": [[4, 20], [8, 32], [4, 16]],
            "copy_output_byte_lengths": [16, 24, 12],
            "copy_nonzero_source_offsets": "required",
            "copy_dynamic_range_plan_reuse": "required",
            "copy_binding_refreshes": 3,
        },
        "timeout_seconds": args.timeout_seconds,
        "directory": str(attempt_dir.relative_to(closure)),
        "rendering_checkout": rendering,
        "source_inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in (
                ROOT / "Cargo.toml",
                ROOT / "Cargo.lock",
                ROOT / "crates/renderer/Cargo.toml",
                *sorted((ROOT / "crates/renderer/src").rglob("*.rs")),
                artifact_manifest,
                artifact_lock,
                *sorted((ARTIFACT / "src").rglob("*.rs")),
                *sorted((ARTIFACT / "assets").rglob("*.wgsl")),
                *sorted((ROOT / "tools/integration").glob("*.py")),
                ROOT / "tools/integration/README.md",
                ROOT / "documents/s0-s1-support-decision.md",
                ROOT / "documents/integration-known-diagnostics.md",
                ROOT / "documents/integration-manifest.schema.json",
            )
        ],
        "toolchain": {"rustc_verbose": command_output(["rustc", "-Vv"]), "cargo": command_output([args.cargo, "-V"])},
        "environment": {
            "os": platform.platform(), "rustc_host": command_output(["rustc", "-vV"]),
            "windows_video_controllers": command_output([
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress",
            ]) if os.name == "nt" else None,
        },
        "validation": {"requested": args.validation,
                       "route": "FLUXEL_DX12_DEBUG" if args.backend == "dx12" else "KHR_debug message-log query via scoped WGL native interop",
                       "status": "unverified", "reason": "Inspect native diagnostics; a request alone does not prove activation."},
        "locks": [{"path": str(artifact_lock), "sha256": sha256(artifact_lock)}],
        "resolved_git_packages": resolved_git_packages(artifact_lock),
    }

    # The archive preserves uncommitted and new source files as reviewable
    # inputs. Later source-consistency checks still reject edits during a run.
    snapshot_error = None
    if artifact_manifest.is_file():
        try:
            attempt["source_snapshot"] = freeze_source_inputs(
                attempt["source_inputs"], attempt_dir / "source-inputs.zip", ROOT
            )
        except (OSError, ValueError) as error:
            snapshot_error = str(error)
            attempt["source_snapshot"] = {"status": "failed", "reason": snapshot_error}

    if not artifact_manifest.is_file() or snapshot_error:
        (attempt_dir / "stdout.log").write_text("", encoding="utf-8")
        message = (f"source snapshot failed: {snapshot_error}\n" if snapshot_error else
                   f"independent artifact manifest is missing: {artifact_manifest}\n")
        (attempt_dir / "stderr.log").write_text(message, encoding="utf-8")
        attempt["commands"] = [{"kind": "build", "command": [], "result": log_result(attempt_dir, None, False, "not_started", message.strip())}]
        exit_code = 2
    else:
        build = [
            args.cargo, "build", "--manifest-path", str(artifact_manifest), "--locked",
            "--no-default-features", "--features", args.backend,
            "--message-format=json-render-diagnostics",
        ]
        build_result = run_command(build, ARTIFACT, args.timeout_seconds, attempt_dir)
        attempt["commands"] = [{"kind": "build", "command": build, "result": build_result}]
        exit_code = 0 if build_result["exit_status"] == 0 else 1
        attempt["built_fluxel_packages"] = built_fluxel_packages(attempt_dir / "stdout.log")
        attempt["built_source_verification"] = verify_built_sources(
            attempt["built_fluxel_packages"], attempt["resolved_git_packages"],
            required_built_packages(args.backend)
        )
        if attempt["built_source_verification"]["status"] != "verified":
            exit_code = 1
        attempt["built_git_checkout_verification"] = verify_git_checkouts(
            attempt["built_fluxel_packages"], attempt["resolved_git_packages"]
        )
        if attempt["built_git_checkout_verification"]["status"] != "verified":
            exit_code = 1
        attempt["source_consistency_after_build"] = source_consistency(attempt["source_inputs"])
        if attempt["source_consistency_after_build"]["status"] != "stable":
            exit_code = 1
        if args.mode == "run" and exit_code == 0:
            executable = find_built_executable(attempt_dir / "stdout.log", artifact_manifest)
            executable_evidence = None
            executable_error = "Cargo did not report an executable for the integration package."
            if executable is not None:
                try:
                    executable_evidence = retain_executable(Path(executable), attempt_dir)
                    executable = executable_evidence["path"]
                except (OSError, ValueError) as error:
                    executable_error = str(error)
                    executable = None
            if executable is None:
                error = executable_error
                (attempt_dir / "run-error.log").write_text(error + "\n", encoding="utf-8")
                attempt["commands"].append({"kind": "direct_executable", "command": [], "result": {"runner_error": error, "raw_logs": [{"path": str(attempt_dir / "run-error.log"), "sha256": sha256(attempt_dir / "run-error.log")} ]}})
                exit_code = 1
            else:
                attempt["executable"] = executable_evidence
                run_directory = attempt_dir / "run"
                child_environment = os.environ.copy()
                child_environment["FLUXEL_EVIDENCE_DIR"] = str(run_directory.resolve())
                child_environment["FLUXEL_INTEGRATION_VALIDATION"] = args.validation
                if args.backend == "dx12":
                    if args.validation == "enabled":
                        child_environment["FLUXEL_DX12_DEBUG"] = "1"
                    else:
                        child_environment.pop("FLUXEL_DX12_DEBUG", None)
                result = run_command(
                    [executable, "--output", str(run_directory.resolve())], ARTIFACT,
                    args.timeout_seconds, run_directory, child_environment
                )
                result["retained_files"] = retained_files(run_directory)
                attempt["validation"] = validation_evidence(args.backend, run_directory, args.validation)
                if args.backend == "dx12":
                    attempt["validation"]["diagnostics"] = classify_dx12_diagnostics(run_directory / "stderr.log")
                attempt["commands"].append({"kind": "direct_executable", "command": [executable, "--output", str(run_directory.resolve())], "result": result})
                exit_code = 0 if result["exit_status"] == 0 else 1
                if args.validation == "enabled" and attempt["validation"].get("status") != "observed":
                    exit_code = 1
                after_hash = sha256(Path(executable))
                attempt["executable"]["after_run_sha256"] = after_hash
                if after_hash != attempt["executable"]["sha256"]:
                    exit_code = 1
                if args.backend == "dx12" and attempt["validation"]["diagnostics"]["unreviewed_native_diagnostics"]:
                    exit_code = 1
                if (
                    args.backend == "native-gl-wgl"
                    and attempt["validation"].get("native_diagnostic_count", 0) > 0
                ):
                    exit_code = 1

    attempt["source_consistency_after_attempt"] = source_consistency(attempt["source_inputs"])
    if attempt["source_consistency_after_attempt"]["status"] != "stable":
        exit_code = 1
    attempt["runner_exit_status"] = exit_code
    parent = ROOT.parent
    decision = ROOT / "documents" / "s0-s1-support-decision.md"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "S0/S1",
        "artifact": {"owner": "fluxel-rendering", "path": str(ARTIFACT), "independent": True},
        "rendering_checkout": rendering,
        "sibling_checkout_inventory": {name: checkout_inventory(parent / name) for name in ("fluxel-rhi", "fluxel-bases", "fluxel-host", "fluxel-jsbridge")},
        "root_lock": {"path": str(ROOT / "Cargo.lock"), "sha256": sha256(ROOT / "Cargo.lock")},
        "support_decision": {"path": str(decision), "sha256": sha256(decision), "review": "unverified"},
        "attempts": load_attempts(index) + [attempt],
        "capability_snapshot": {"status": "unverified", "reason": "No reviewed complete capability snapshot is linked."},
        "outputs": {"status": "unverified", "reason": "Attempt artifacts contain workload-specific facts; closure-level output review is incomplete."},
        "tests": {"status": "unverified", "reason": "No focused logical-case or backend-execution results are retained."},
        "measurements": {"status": "unverified", "reason": "Raw workload measurements require a reviewed closure-level baseline aggregation."},
        "unsupported_or_untested": ["Vulkan", "WebGPU", "WebGL2", "full DX12/WGL closure matrix remains unverified"],
        "known_limitations": ["Sibling checkout inventory is not evidence of Cargo's built sources.", "A clean exit does not establish visual output, lifecycle correctness, or support."],
    }
    write_json(index, manifest)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
