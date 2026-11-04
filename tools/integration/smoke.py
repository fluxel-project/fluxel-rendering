#!/usr/bin/env python3
"""Retain bounded native integration attempts without asserting stage support."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from typing import Any


# Schema v4 adds S5-specific nested workload evidence. S0-S3 manifests keep
# their v3 contract so a schema validator does not accept them as S5 evidence.
S0_S3_SCHEMA_VERSION = 3
S5_SCHEMA_VERSION = 4
S6_SCHEMA_VERSION = 5
BACKENDS = ("dx12", "native-gl-wgl")
WORKLOADS = ("s0-s3", "s5-forward", "s5-surface", "s5-device-isolation", "s6-composed", "s6-surface")
RHI_REVISION = "b5d569feace548a6ed498901deaf31a5efcaca61"
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "examples" / "rendering-integration"
EXCLUDED_SOURCE_PARTS = {".git", "target", "draft"}
SENSITIVE_SOURCE_NAMES = {".env", ".env.local", ".envrc"}
SENSITIVE_SOURCE_SUFFIXES = {".key", ".pem", ".p12", ".pfx", ".token"}
SENSITIVE_CONFIG_ASSIGNMENT = re.compile(r"(?im)^\s*(?:token|password|secret|api[-_]?key)\s*=")
RUST_INCLUDE = re.compile(r"include_(?:str|bytes)!\s*\(\s*([^)]*?)\s*\)")


def schema_version_for_workload(workload: str) -> int:
    """Keep historical and S5 manifest schemas disjoint."""
    if workload.startswith("s6-"):
        return S6_SCHEMA_VERSION
    return S5_SCHEMA_VERSION if workload.startswith("s5-") else S0_S3_SCHEMA_VERSION


def stage_for_workload(workload: str) -> str:
    """Label every S5 contract as S5 evidence in the outer manifest."""
    if workload.startswith("s6-"):
        return "S6 typed Forward integration evidence"
    return "S5 Forward integration evidence" if workload.startswith("s5-") else "S0-S3 integration evidence"

if __package__:
    from .smoke_attempt import (
        actual_test_count, attempt_name, load_attempts, log_result, retained_files, run_command,
        write_json,
    )
    from .smoke_evidence import (
        built_fluxel_packages, command_output, find_built_executable, freeze_source_inputs,
        nested_workload_evidence, required_built_packages, resolved_git_packages,
        retain_executable, s5_device_isolation_required_raw_artifacts,
        s5_required_raw_artifacts, s5_static_asset_raw_artifacts,
        s5_surface_required_raw_artifacts, s5_workload_contract, validation_evidence,
        verify_built_sources, verify_git_checkouts as _verify_git_checkouts,
    )
    from .smoke_inputs import (
        build_environment, checkout_inventory, classify_dx12_diagnostics,
        closure_escape_count, controlled_child_environment, cargo_config_has_secret,
        external_cargo_config_count, git_value, sensitive_build_inputs, sha256,
        source_consistency, source_input_paths, source_path_is_safe,
        uncontrolled_build_script_count,
    )
    from .smoke_s6 import s6_nested_workload_evidence, s6_workload_contract
else:
    from smoke_attempt import (
        actual_test_count, attempt_name, load_attempts, log_result, retained_files, run_command,
        write_json,
    )
    from smoke_evidence import (
        built_fluxel_packages, command_output, find_built_executable, freeze_source_inputs,
        nested_workload_evidence, required_built_packages, resolved_git_packages,
        retain_executable, s5_device_isolation_required_raw_artifacts,
        s5_required_raw_artifacts, s5_static_asset_raw_artifacts,
        s5_surface_required_raw_artifacts, s5_workload_contract, validation_evidence,
        verify_built_sources, verify_git_checkouts as _verify_git_checkouts,
    )
    from smoke_inputs import (
        build_environment, checkout_inventory, classify_dx12_diagnostics,
        closure_escape_count, controlled_child_environment, cargo_config_has_secret,
        external_cargo_config_count, git_value, sensitive_build_inputs, sha256,
        source_consistency, source_input_paths, source_path_is_safe,
        uncontrolled_build_script_count,
    )
    from smoke_s6 import s6_nested_workload_evidence, s6_workload_contract


def verify_git_checkouts(inventory: list[dict[str, Any]], locked: list[dict[str, str]]) -> dict[str, Any]:
    """Keep the orchestration seam patchable while checkout parsing stays isolated."""
    return _verify_git_checkouts(inventory, locked, git_lookup=git_value)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--workload", choices=WORKLOADS, default="s0-s3",
                        help="bounded executable workload; each choice owns a separate evidence closure")
    parser.add_argument("--mode", choices=("build", "run"), default="run")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--validation", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--frame-hold-ms", type=int,
                        help="explicit attended-capture hold (1..120000); never inherited from FLUXEL_*")
    parser.add_argument("--test-command-json", action="append", default=[],
                        help="JSON array command recorded as runner-owned focused test evidence")
    args = parser.parse_args()
    test_commands: list[list[str]] = []
    for encoded in args.test_command_json:
        try:
            command = json.loads(encoded)
        except json.JSONDecodeError as error:
            parser.error(f"--test-command-json is not JSON: {error}")
        if not isinstance(command, list) or not command or any(not isinstance(item, str) or not item for item in command):
            parser.error("--test-command-json must be a non-empty JSON string array")
        test_commands.append(command)
    artifact_manifest = ARTIFACT / "Cargo.toml"
    artifact_package = (
        tomllib.loads(artifact_manifest.read_text(encoding="utf-8"))
        if artifact_manifest.is_file() else {}
    )
    if artifact_package.get("package", {}).get("metadata", {}).get("fluxel", {}).get("status") == "retired-after-renderer-reset":
        parser.error("the previous integration consumer targets removed Renderer APIs; rebuild it before running smoke evidence")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.frame_hold_ms is not None and not 1 <= args.frame_hold_ms <= 120000:
        parser.error("--frame-hold-ms must be 1 through 120000")

    rendering = checkout_inventory(ROOT)
    build_environment_evidence = build_environment(ROOT)
    build_child_environment, build_child_evidence = controlled_child_environment()
    revision = rendering["revision"] or "unknown-rendering-revision"
    closure = ROOT / "target" / "integration" / args.workload / revision
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
        "workload": (s6_workload_contract(args.workload) if args.workload.startswith("s6-") else
                     s5_workload_contract(args.workload) if args.workload.startswith("s5-") else {
            "selection": "copy+raster+compute+read-set+s2-shader+s2-texture+s3-material+s3-material-replacement+surface+device-replacement",
            "submissions_before_wait": 3,
            "copy_source_byte_lengths": [23, 37, 22],
            "copy_source_ranges": [[4, 20], [8, 32], [4, 16]],
            "copy_output_byte_lengths": [16, 24, 12],
            "copy_nonzero_source_offsets": "required",
            "copy_dynamic_range_plan_reuse": "required",
            "copy_binding_refreshes": 3,
        }),
        "timeout_seconds": args.timeout_seconds,
        "directory": str(attempt_dir.relative_to(closure)),
        "rendering_checkout": rendering,
        "source_inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in source_input_paths(ROOT)
        ],
        "toolchain": {"rustc_verbose": command_output(["rustc", "-Vv"]), "cargo": command_output([args.cargo, "-V"])},
        "environment": {
            "os": platform.platform(), "rustc_host": command_output(["rustc", "-vV"]),
            "windows_video_controllers": command_output([
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress",
            ]) if os.name == "nt" else None,
        },
        "build_environment": build_environment_evidence,
        "child_environment": {"build_and_test": build_child_evidence},
        "validation": {"requested": args.validation,
                       "route": "FLUXEL_DX12_DEBUG" if args.backend == "dx12" else "KHR_debug message-log query via scoped WGL native interop",
                       "status": "unverified", "reason": "Inspect native diagnostics; a request alone does not prove activation."},
        # The consumer lock controls Cargo resolution; the workspace lock is
        # also archived as a source input and recorded per attempt so a review
        # can distinguish the exact source closure from dependency resolution.
        "locks": [
            {"path": str(artifact_lock), "sha256": sha256(artifact_lock)},
            {"path": str(ROOT / "Cargo.lock"), "sha256": sha256(ROOT / "Cargo.lock")},
        ],
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

    if not artifact_manifest.is_file() or snapshot_error or build_environment_evidence["status"] != "auditable":
        (attempt_dir / "stdout.log").write_text("", encoding="utf-8")
        message = (f"source snapshot failed: {snapshot_error}\n" if snapshot_error else
                   "build environment is not auditable; no build was started\n"
                   if build_environment_evidence["status"] != "auditable" else
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
        build_result = run_command(build, ARTIFACT, args.timeout_seconds, attempt_dir, build_child_environment)
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
                run_extras = {"FLUXEL_EVIDENCE_DIR": str(run_directory.resolve()),
                              "FLUXEL_INTEGRATION_VALIDATION": args.validation}
                if args.backend == "dx12":
                    if args.validation == "enabled":
                        run_extras["FLUXEL_DX12_DEBUG"] = "1"
                if args.frame_hold_ms is not None:
                    run_extras["FLUXEL_INTEGRATION_FRAME_HOLD_MS"] = str(args.frame_hold_ms)
                child_environment, run_child_evidence = controlled_child_environment(run_extras)
                attempt["child_environment"]["run"] = run_child_evidence
                executable_command = [executable, "--output", str(run_directory.resolve())]
                if args.workload.startswith("s6-"):
                    executable_command.append(s6_workload_contract(args.workload)["entry_argument"])
                elif args.workload.startswith("s5-"):
                    executable_command.append(s5_workload_contract(args.workload)["entry_argument"])
                result = run_command(
                    executable_command, ARTIFACT,
                    args.timeout_seconds, run_directory, child_environment
                )
                result["retained_files"] = retained_files(run_directory)
                result["nested_workloads"] = (
                    s6_nested_workload_evidence(run_directory, args.workload)
                    if args.workload.startswith("s6-")
                    else nested_workload_evidence(run_directory, args.workload)
                )
                attempt["validation"] = validation_evidence(args.backend, run_directory, args.validation)
                if args.backend == "dx12":
                    attempt["validation"]["diagnostics"] = classify_dx12_diagnostics(run_directory / "stderr.log")
                attempt["commands"].append({"kind": "direct_executable", "command": executable_command, "result": result})
                exit_code = 0 if result["exit_status"] == 0 else 1
                if result["nested_workloads"]["status"] != "verified":
                    exit_code = 1
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

    if exit_code == 0:
        for test_index, command in enumerate(test_commands):
            test_result = run_command(command, ARTIFACT, args.timeout_seconds, attempt_dir / "focused-tests" / str(test_index), build_child_environment)
            count = actual_test_count(command, test_result)
            test_result["test_command_recognized"] = count is not None
            test_result["executed"] = count
            attempt["commands"].append({"kind": "test", "command": command, "result": test_result})
            if test_result["exit_status"] != 0 or count is None or count <= 0:
                exit_code = 1

    attempt["source_consistency_after_attempt"] = source_consistency(attempt["source_inputs"])
    if attempt["source_consistency_after_attempt"]["status"] != "stable":
        exit_code = 1
    attempt["runner_exit_status"] = exit_code
    # Failed attempts are evidence too. Record every file produced by the
    # runner, including the frozen source closure and failure logs, instead of
    # relying only on the successful executable-run subdirectory.
    attempt["retained_attempt_files"] = retained_files(attempt_dir)
    parent = ROOT.parent
    decision = ROOT / "documents" / "s0-s1-support-decision.md"
    manifest = {
        "schema_version": schema_version_for_workload(args.workload),
        "stage": stage_for_workload(args.workload),
        "artifact": {"owner": "fluxel-rendering", "path": str(ARTIFACT), "independent": True},
        "rendering_checkout": rendering,
        "sibling_checkout_inventory": {name: checkout_inventory(parent / name) for name in ("fluxel-rhi", "fluxel-bases", "fluxel-host", "fluxel-jsbridge")},
        "root_lock": {"path": str(ROOT / "Cargo.lock"), "sha256": sha256(ROOT / "Cargo.lock")},
        "historical_foundation_support_decision": {
            "path": str(decision),
            "sha256": sha256(decision),
            "status": "historical_reference_only",
            "scope": "S0/S1",
            "cannot_approve_current_support": True,
        },
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
