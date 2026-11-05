#!/usr/bin/env python3
"""Create one immutable, formal S7 Deferred native evidence attempt."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import sys
import tomllib
from typing import Any

if __package__:
    from .smoke_attempt import actual_test_count, attempt_name, load_attempts, log_result, retained_files, run_command, write_json
    from .smoke_evidence import (built_fluxel_packages, command_output, find_built_executable, freeze_source_inputs,
        required_built_packages, resolved_git_packages, retain_executable, validation_evidence, verify_built_sources,
        verify_git_checkouts)
    from .smoke_inputs import (build_environment, checkout_inventory, classify_dx12_diagnostics,
        controlled_child_environment, sha256, source_consistency, source_input_paths)
    from .smoke_s7 import S7_SCHEMA_VERSION, S7_STAGE, s7_nested_workload_evidence, s7_workload_contract
else:
    from smoke_attempt import actual_test_count, attempt_name, load_attempts, log_result, retained_files, run_command, write_json
    from smoke_evidence import (built_fluxel_packages, command_output, find_built_executable, freeze_source_inputs,
        required_built_packages, resolved_git_packages, retain_executable, validation_evidence, verify_built_sources,
        verify_git_checkouts)
    from smoke_inputs import (build_environment, checkout_inventory, classify_dx12_diagnostics,
        controlled_child_environment, sha256, source_consistency, source_input_paths)
    from smoke_s7 import S7_SCHEMA_VERSION, S7_STAGE, s7_nested_workload_evidence, s7_workload_contract


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "examples" / "rendering-integration"
TOOLCHAIN = "1.87.0"
BACKENDS = ("dx12", "native-gl-wgl")
WORKLOADS = ("s7-deferred", "s7-deferred-surface", "s7-deferred-device-isolation")
WINDOW_CAPTURE_POLICY = "schema2-nondisturbing"


def _commands(args: argparse.Namespace) -> list[list[str]]:
    commands = []
    for value in args.test_command_json:
        try:
            command = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"--test-command-json is not JSON: {error}") from error
        if not isinstance(command, list) or not command or any(not isinstance(part, str) or not part for part in command):
            raise ValueError("--test-command-json must be a non-empty JSON string array")
        commands.append(command)
    if commands:
        return commands
    if not args.focused_test_group:
        return []
    return [
        [args.cargo, "test", "--manifest-path", str(ARTIFACT / "Cargo.toml"), "--locked", "--no-default-features", "--features", args.backend, "--bin", "rendering-integration"],
        [args.cargo, "test", "--manifest-path", str(ROOT / "Cargo.toml"), "-p", "fluxel-renderer", "--lib", "deferred"],
        [args.cargo, "test", "--manifest-path", str(ROOT / "examples" / "external-renderer" / "Cargo.toml"), "--locked"],
        [sys.executable, "-m", "unittest", "discover", "--start-directory", "tools/integration", "--pattern", "test_s7*.py"],
    ]


def _test_cwd(command: list[str]) -> Path:
    """Evidence-tool tests import from the repository; Cargo uses its manifest path."""
    return ROOT if Path(command[0]).name.lower().startswith("python") else ARTIFACT


def _same_source_inputs(previous: object, inputs: list[dict[str, Any]]) -> bool:
    """Only an exact frozen input set may suppress another focused test group."""
    if not isinstance(previous, dict) or previous.get("source_inputs") != inputs:
        return False
    snapshot = previous.get("source_snapshot")
    return isinstance(snapshot, dict) and snapshot.get("kind") == "exact_worktree_inputs"


def _successful_test(result: object) -> bool:
    return (isinstance(result, dict) and result.get("exit_status") == 0
            and result.get("test_command_recognized") is True
            and isinstance(result.get("executed"), int) and result["executed"] > 0)


def _completed_focused_group(previous: object, inputs: list[dict[str, Any]]) -> bool:
    """Recognize all four required groups, never an arbitrary historical test."""
    if not _same_source_inputs(previous, inputs):
        return False
    commands = previous.get("commands", [])
    if not isinstance(commands, list):
        return False
    kinds = set()
    for item in commands:
        if not isinstance(item, dict) or item.get("kind") != "test" or not _successful_test(item.get("result")):
            continue
        command = item.get("command", [])
        if not isinstance(command, list):
            continue
        text = " ".join(command)
        if str(ROOT / "crates") in text or ("-p fluxel-renderer" in text and "--lib deferred" in text):
            kinds.add("renderer")
        elif str(ROOT / "examples" / "external-renderer" / "Cargo.toml") in text:
            kinds.add("external")
        elif str(ARTIFACT / "Cargo.toml") in text and "--bin rendering-integration" in text:
            kinds.add("integration")
        elif "unittest discover" in text and "test_s7*.py" in text:
            kinds.add("tools")
    return kinds == {"renderer", "integration", "external", "tools"}


def _attempt(args: argparse.Namespace, closure: Path, child_environment: dict[str, str], child_evidence: dict[str, Any]) -> tuple[dict[str, Any], int]:
    artifact_manifest, artifact_lock = ARTIFACT / "Cargo.toml", ARTIFACT / "Cargo.lock"
    attempt_dir = closure / "attempts" / attempt_name(args.backend)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    print(f"Retaining formal S7 attempt {attempt_dir.name}: {attempt_dir}", flush=True)
    rendering = checkout_inventory(ROOT)
    inputs = [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in source_input_paths(ROOT)]
    contract = s7_workload_contract(args.workload)
    attempt: dict[str, Any] = {
        "id": attempt_dir.name, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "backend": args.backend, "feature": args.backend, "mode": "run", "workload": contract,
        "timeout_seconds": args.timeout_seconds, "directory": str(attempt_dir.relative_to(closure)),
        "rendering_checkout": rendering, "source_inputs": inputs,
        # New attempts require non-disturbance facts. Existing immutable
        # archives omit this marker and remain correctness-reviewable only.
        "window_capture_policy": WINDOW_CAPTURE_POLICY,
        "toolchain": {"rustc_verbose": command_output(["rustc", f"+{TOOLCHAIN}", "-Vv"]), "cargo": command_output([args.cargo, f"+{TOOLCHAIN}", "-V"])},
        "environment": {"os": platform.platform()}, "child_environment": {"build_test": child_evidence},
        "build_environment": build_environment(ROOT),
        "locks": [{"path": str(artifact_lock), "sha256": sha256(artifact_lock)}, {"path": str(ROOT / "Cargo.lock"), "sha256": sha256(ROOT / "Cargo.lock")}],
        "resolved_git_packages": resolved_git_packages(artifact_lock),
        "validation": {"requested": "enabled", "status": "unverified", "reason": "direct native execution has not started"},
    }
    snapshot_error = None
    if artifact_manifest.is_file():
        try:
            attempt["source_snapshot"] = freeze_source_inputs(inputs, attempt_dir / "source-inputs.zip", ROOT)
        except (OSError, ValueError) as error:
            snapshot_error = str(error)
            attempt["source_snapshot"] = {"status": "failed", "reason": snapshot_error}
    if not artifact_manifest.is_file() or snapshot_error or attempt["build_environment"]["status"] != "auditable":
        message = "independent artifact manifest is missing" if not artifact_manifest.is_file() else (snapshot_error or "build environment is not auditable")
        (attempt_dir / "stdout.log").write_text("", encoding="utf-8")
        (attempt_dir / "stderr.log").write_text(message + "\n", encoding="utf-8")
        attempt["commands"] = [{"kind": "build", "command": [], "result": log_result(attempt_dir, None, False, "not_started", message)}]
        return attempt, 2
    build = [args.cargo, "build", "--manifest-path", str(artifact_manifest), "--locked", "--no-default-features", "--features", args.backend, "--message-format=json-render-diagnostics"]
    build_result = run_command(build, ARTIFACT, args.timeout_seconds, attempt_dir, child_environment)
    attempt["commands"] = [{"kind": "build", "command": build, "result": build_result}]
    exit_code = 0 if build_result["exit_status"] == 0 else 1
    attempt["built_fluxel_packages"] = built_fluxel_packages(attempt_dir / "stdout.log")
    attempt["built_source_verification"] = verify_built_sources(attempt["built_fluxel_packages"], attempt["resolved_git_packages"], required_built_packages(args.backend))
    attempt["built_git_checkout_verification"] = verify_git_checkouts(attempt["built_fluxel_packages"], attempt["resolved_git_packages"])
    if attempt["built_source_verification"]["status"] != "verified" or attempt["built_git_checkout_verification"]["status"] != "verified":
        exit_code = 1
    attempt["source_consistency_after_build"] = source_consistency(inputs)
    if attempt["source_consistency_after_build"]["status"] != "stable":
        exit_code = 1
    if exit_code == 0:
        executable = find_built_executable(attempt_dir / "stdout.log", artifact_manifest)
        if executable is None:
            error = "Cargo did not report an executable for the integration package."
            (attempt_dir / "run-error.log").write_text(error + "\n", encoding="utf-8")
            attempt["commands"].append({"kind": "direct_executable", "command": [], "result": {"runner_error": error}})
            exit_code = 1
        else:
            try:
                attempt["executable"] = retain_executable(Path(executable), attempt_dir)
                executable = attempt["executable"]["path"]
            except (OSError, ValueError) as error:
                attempt["commands"].append({"kind": "direct_executable", "command": [], "result": {"runner_error": str(error)}})
                exit_code = 1
    if exit_code == 0:
        run_dir = attempt_dir / "run"
        extras = {"RUSTUP_TOOLCHAIN": TOOLCHAIN, "FLUXEL_EVIDENCE_DIR": str(run_dir.resolve()), "FLUXEL_INTEGRATION_VALIDATION": "enabled"}
        if args.backend == "dx12":
            extras["FLUXEL_DX12_DEBUG"] = "1"
        run_environment, run_evidence = controlled_child_environment(extras)
        attempt["child_environment"]["run"] = run_evidence
        command = [executable, "--output", str(run_dir.resolve()), contract["entry_argument"]]
        result = run_command(command, ARTIFACT, args.timeout_seconds, run_dir, run_environment)
        result["retained_files"] = retained_files(run_dir)
        result["nested_workloads"] = s7_nested_workload_evidence(run_dir, args.workload)
        attempt["nested_workload_records"] = result["nested_workloads"]["records"]
        attempt["validation"] = validation_evidence(args.backend, run_dir, "enabled")
        if args.backend == "dx12":
            attempt["validation"]["diagnostics"] = classify_dx12_diagnostics(run_dir / "stderr.log")
        attempt["commands"].append({"kind": "direct_executable", "command": command, "result": result})
        if result["exit_status"] != 0 or result["nested_workloads"]["status"] != "verified" or attempt["validation"]["status"] != "observed":
            exit_code = 1
        if args.backend == "dx12" and attempt["validation"]["diagnostics"]["unreviewed_native_diagnostics"]:
            exit_code = 1
        if args.backend == "native-gl-wgl" and attempt["validation"].get("native_diagnostic_count", 0) > 0:
            exit_code = 1
        attempt["executable"]["after_run_sha256"] = sha256(Path(executable))
        if attempt["executable"]["after_run_sha256"] != attempt["executable"]["sha256"]:
            exit_code = 1
    if exit_code == 0:
        for index, command in enumerate(_commands(args)):
            result = run_command(command, _test_cwd(command), args.timeout_seconds, attempt_dir / "focused-tests" / str(index), child_environment)
            result["executed"] = actual_test_count(command, result)
            result["test_command_recognized"] = result["executed"] is not None
            attempt["commands"].append({"kind": "test", "command": command, "result": result})
            if result["exit_status"] != 0 or not result["executed"]:
                exit_code = 1
    attempt["source_consistency_after_attempt"] = source_consistency(inputs)
    if attempt["source_consistency_after_attempt"]["status"] != "stable":
        exit_code = 1
    attempt["runner_exit_status"] = exit_code
    attempt["retained_attempt_files"] = retained_files(attempt_dir)
    return attempt, exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--focused-test-group", action="store_true", help="run the closure's one renderer, consumer, and evidence-tool test group")
    parser.add_argument("--test-command-json", action="append", default=[])
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    try:
        commands = _commands(args)
    except ValueError as error:
        parser.error(str(error))
    revision = checkout_inventory(ROOT)["revision"] or "unknown-rendering-revision"
    closure = ROOT / "target" / "integration" / "s7" / revision
    index = closure / "manifest.json"
    current_inputs = [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        for path in source_input_paths(ROOT)
    ]
    if args.focused_test_group and any(
        _completed_focused_group(previous, current_inputs) for previous in load_attempts(index)
    ):
        parser.error("this unchanged S7 source closure already retains a focused test group")
    child_environment, child_evidence = controlled_child_environment({"RUSTUP_TOOLCHAIN": TOOLCHAIN})
    attempt, exit_code = _attempt(args, closure, child_environment, child_evidence)
    parent = ROOT.parent
    attempts = load_attempts(index) + [attempt]
    manifest = {
        "schema_version": S7_SCHEMA_VERSION, "stage": S7_STAGE,
        "artifact": {"owner": "fluxel-rendering", "path": str(ARTIFACT), "independent": True},
        "rendering_checkout": checkout_inventory(ROOT),
        "sibling_checkout_inventory": {name: checkout_inventory(parent / name) for name in ("fluxel-rhi", "fluxel-bases", "fluxel-host", "fluxel-jsbridge")},
        "root_lock": {"path": str(ROOT / "Cargo.lock"), "sha256": sha256(ROOT / "Cargo.lock")},
        "historical_foundation_support_decision": {
            "path": str(ROOT / "documents" / "s0-s1-support-decision.md"),
            "sha256": sha256(ROOT / "documents" / "s0-s1-support-decision.md"),
            "status": "historical_reference_only", "scope": "S0/S1", "cannot_approve_current_support": True,
        },
        "attempts": attempts,
        "capability_snapshot": {"status": "unverified", "reason": "reviewer derives the formal matrix closure"},
        "outputs": {"status": "unverified", "reason": "reviewer checks the retained raw inventory"},
        "tests": {
            "status": "unverified" if not commands else "retained_by_selected_attempt",
            "reason": "A test group runs once for an unchanged source closure; reviewer checks its nonzero execution.",
            "selected_attempt": attempt["id"] if commands else None,
        },
        "measurements": {"status": "unverified", "reason": "S7 records correctness evidence, not a performance baseline"},
        "unsupported_or_untested": ["Vulkan", "WebGPU", "WebGL2"],
        "known_limitations": ["One attempt proves one exact backend/workload cell; reviewer selects the complete matrix."],
    }
    write_json(index, manifest)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
