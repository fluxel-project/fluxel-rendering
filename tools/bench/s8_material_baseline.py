#!/usr/bin/env python3
"""Capture an auditable S8 material-heavy baseline.

This has no candidate or comparison switch. It freezes inputs, runs setup-only
smokes, then samples each representative scenario independently.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONSUMER = ROOT / "examples" / "rendering-integration"
MANIFEST = CONSUMER / "Cargo.toml"
LOCKS = (ROOT / "Cargo.lock", CONSUMER / "Cargo.lock")
MSRV = "1.87.0"
SCENARIOS = ("cold", "warm", "replacement-10pct")
ACTUAL_DRAWS, ACTUAL_TEXTURES = 10_000, 1_000
REQUIRED_FACTS = (
    "submitElapsedNs", "preparationElapsedNs", "graphCompileElapsedNs", "acceptedBufferAllocations",
    "acceptedBufferAllocationBytes", "acceptedTextureAllocations", "pendingPeak",
    "residentStaticUploadBytes", "frameUploadBytes", "geometryCacheHits",
    "textureCacheHits", "frameAllocationReuses", "planCompilations",
    "visibleCandidates", "batches", "draws",
)

INTEGRATION_TOOLS = ROOT / "tools" / "integration"
if str(INTEGRATION_TOOLS) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_TOOLS))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from s8_material_oracle import review_run
from smoke_evidence import (built_fluxel_packages, freeze_source_inputs, required_built_packages,
                            resolved_git_packages, retain_executable, verify_built_sources,
                            verify_git_checkouts)
from smoke_inputs import build_environment, controlled_child_environment, source_consistency, source_input_paths


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def archive_inputs() -> list[dict[str, str]]:
    """Archive bounded native inputs and all measurement/oracle Python inputs."""
    paths = [*source_input_paths(ROOT), *(ROOT / "tools" / "bench").rglob("*.py")]
    result = []
    for path in sorted({item.resolve() for item in paths if item.is_file()}, key=relative):
        value = digest(path)
        if value is None:
            raise ValueError(f"source input disappeared: {path}")
        result.append({"path": relative(path), "sha256": value})
    return result


def source_is_stable(inputs: list[dict[str, str]]) -> None:
    check = source_consistency(inputs, ROOT)
    if check["status"] != "stable":
        changes = check.get("changed_inputs", [])
        names = ", ".join(str(item.get("path", "unknown")) for item in changes)
        raise RuntimeError(f"source changed during capture: {names}")


def git_snapshot() -> dict[str, str | None]:
    return {"head": command_output(["git", "rev-parse", "HEAD"]),
            "status": command_output(["git", "status", "--porcelain=v1"])}


def host_environment() -> dict[str, Any]:
    def powershell(expression: str) -> Any:
        output = command_output(["powershell", "-NoProfile", "-Command", expression]) if os.name == "nt" else None
        try:
            return json.loads(output) if output else None
        except json.JSONDecodeError:
            return output
    return {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
            "cpu": powershell("Get-CimInstance Win32_Processor | Select Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"),
            "gpu": powershell("Get-CimInstance Win32_VideoController | Select Name,DriverVersion | ConvertTo-Json -Compress"),
            "power_scheme": command_output(["powercfg", "/GETACTIVESCHEME"]) if os.name == "nt" else None}


def persist(output: Path, progress: dict[str, Any]) -> None:
    progress["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    (output / "baseline.partial.json").write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_error(output: Path, progress: dict[str, Any], stage: str, error: BaseException) -> None:
    progress["status"] = "failed"
    progress.setdefault("errors", []).append({"stage": stage, "type": type(error).__name__, "message": str(error),
                                                  "at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    persist(output, progress)


def run_process(command: list[str], cwd: Path, log: Path, timeout: float, environment: dict[str, str]) -> dict[str, Any]:
    """Record log/hash/wall time for successful, failed, timed-out and spawn-failed children."""
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    with log.open("w", encoding="utf-8") as stream:
        try:
            completed = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, env=environment,
                                       timeout=timeout, check=False)
            state: dict[str, Any] = {"exit_status": completed.returncode}
        except subprocess.TimeoutExpired:
            state = {"exit_status": None, "timed_out": True}
        except OSError as error:
            state = {"exit_status": None, "spawn_error": str(error)}
    return {"command": command, **state, "process_wall_ns": time.perf_counter_ns() - started,
            "log": relative(log), "log_sha256": digest(log)}


def executable(target: Path) -> Path:
    return target / "release" / ("rendering-integration.exe" if os.name == "nt" else "rendering-integration")


def integer_fact(data: dict[str, Any], name: str) -> int:
    value = data.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"S8 facts require non-boolean integer {name}")
    return value


def observed_raw_input_readback_hashes(directory: Path) -> dict[str, list[dict[str, str | None]]]:
    """Hash every already-written input/readback artifact, including failed runs."""
    inputs, readbacks, artifacts = [], [], []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        artifacts.append(path)
        name = path.name.lower()
        if "input" in name:
            inputs.append(path)
        if "raw" in name or "readback" in name:
            readbacks.append(path)
    def hashes(files: list[Path]) -> list[dict[str, str | None]]:
        return [{"path": path.relative_to(directory).as_posix(), "sha256": digest(path)} for path in sorted(files)]
    return {"raw_input_artifacts": hashes(inputs), "raw_readback_artifacts": hashes(readbacks),
            "all_artifacts": hashes(artifacts)}


def raw_input_readback_hashes(directory: Path) -> dict[str, list[dict[str, str | None]]]:
    """Require the complete artifact pair before a successful run can count."""
    observed = observed_raw_input_readback_hashes(directory)
    if not observed["raw_input_artifacts"] or not observed["raw_readback_artifacts"]:
        raise ValueError(f"missing raw input or raw/readback artifact: {directory}")
    return observed


def facts(directory: Path, scenario: str, smoke: bool) -> dict[str, Any]:
    path = directory / "s8-material-baseline" / "s8-material-baseline.json"
    if not path.is_file():
        raise ValueError(f"missing S8 workload facts: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed S8 workload facts: {path}: {error}") from error
    for name in REQUIRED_FACTS:
        integer_fact(data, name)
    if data.get("workload") != "s8-material-baseline":
        raise ValueError("S8 facts report a different workload")
    if data.get("role") != "representative" or data.get("primaryMetric") != "cpu-preparation-submit":
        raise ValueError("S8 facts violate the representative CPU metric contract")
    if data.get("gpuTiming") != "unobserved" or data.get("scenario") != scenario:
        raise ValueError(f"S8 facts scenario/GPU contract mismatch for {scenario}")
    if data.get("smoke") is not smoke:
        raise ValueError("S8 facts smoke mode does not match invoked mode")
    if not smoke and (data.get("draws") != ACTUAL_DRAWS or data.get("textures") != ACTUAL_TEXTURES
                      or data.get("allTexturesReferenced") is not True):
        raise ValueError("actual S8 sample must prove 10000 draws, 1000 textures, and all references")
    return data


def workload_command(binary: Path, output: Path, scenario: str, smoke: bool) -> list[str]:
    command = [str(binary), "--output", str(output), "--s8-material-only", "--s8-scenario", scenario]
    if smoke:
        command.append("--s8-smoke")
    return command


def sample_record(run: dict[str, Any], directory: Path, scenario: str, smoke: bool) -> dict[str, Any]:
    if run.get("exit_status") != 0:
        raise RuntimeError(f"{scenario} process failed; see {run['log']}")
    return {"scenario": scenario, "run": run, "facts": facts(directory, scenario, smoke),
            "raw_input_readback_artifacts": raw_input_readback_hashes(directory),
            "pixel_oracle": review_run(directory)}


def run_group(kind: str, binary: Path, output: Path, repeats: int, smoke: bool, timeout: float,
              environment: dict[str, str], progress: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for scenario in SCENARIOS:
        for index in range(repeats):
            directory = output / kind / scenario / f"{index:02d}"
            if directory.exists():
                raise ValueError(f"refusing to reuse per-campaign sample directory: {directory}")
            run = run_process(workload_command(binary, directory, scenario, smoke), CONSUMER,
                              directory / "process.log", timeout, environment)
            attempted = {"kind": kind, "scenario": scenario, "index": index, "run": run}
            progress["attempts"].append(attempted)
            attempted["observed_raw_input_readback_artifacts"] = observed_raw_input_readback_hashes(directory)
            persist(output, progress)
            record = sample_record(run, directory, scenario, smoke)
            attempted.update({"facts": record["facts"],
                              "raw_input_readback_artifacts": record["raw_input_readback_artifacts"],
                              "pixel_oracle": record["pixel_oracle"]})
            persist(output, progress)
            results.append(record)
    return results


def distribution(values: list[int]) -> dict[str, int]:
    if not values:
        raise ValueError("cannot summarize an empty measurement series")
    return {"minimum": min(values), "median": int(statistics.median(values)), "maximum": max(values)}


def metric_summary(samples: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    result = {}
    for scenario in SCENARIOS:
        selected = [sample for sample in samples if sample["scenario"] == scenario]
        result[scenario] = {"primary_submit_elapsed_ns": distribution([x["facts"]["submitElapsedNs"] for x in selected]),
                            "process_wall_ns": distribution([x["run"]["process_wall_ns"] for x in selected]),
                            **{name: distribution([x["facts"][name] for x in selected]) for name in REQUIRED_FACTS}}
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("dx12", "native-gl-wgl"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="new per-campaign directory; never reused")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--run-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--build-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--cargo", default="cargo")
    result = parser.parse_args()
    if result.samples < 7:
        parser.error("--samples must be at least 7")
    if result.warmup < 1 or result.run_timeout_seconds <= 0 or result.build_timeout_seconds <= 0:
        parser.error("--warmup and both timeouts must be positive")
    return result


def main() -> int:
    args = arguments()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("--output must be a new per-campaign directory")
    output.mkdir(parents=True)
    progress: dict[str, Any] = {"schema_version": 1, "kind": "s8-material-baseline-progress", "status": "partial",
                                "state": "created", "attempts": [], "errors": []}
    persist(output, progress)
    stage = "initialization"
    try:
        stage = "source-closure"
        inputs = archive_inputs()
        archive = freeze_source_inputs(inputs, output / "source-inputs.zip", ROOT)
        source_is_stable(inputs)
        build_evidence = build_environment(ROOT)
        progress["initial"] = {"source": {"inputs": inputs, "exact_worktree_inputs": archive,
                                             "locks": [{"path": relative(lock), "sha256": digest(lock)} for lock in LOCKS]},
                               "git": git_snapshot(), "environment": host_environment(),
                               "toolchain": {"msrv": MSRV, "rustc": command_output(["rustc", f"+{MSRV}", "-Vv"]),
                                             "cargo": command_output([args.cargo, f"+{MSRV}", "-V"])},
                               "parameters": {"backend": args.backend, "samples": args.samples, "warmup": args.warmup,
                                              "build_timeout_seconds": args.build_timeout_seconds,
                                              "run_timeout_seconds": args.run_timeout_seconds,
                                              "actual_workload": "10000 visible draws; 1000 actual texture assets"},
                               "build_environment": build_evidence}
        persist(output, progress)
        if build_evidence["status"] != "auditable":
            raise RuntimeError("uncontrolled build environment: " + "; ".join(build_evidence["reasons"]))
        child, child_evidence = controlled_child_environment({"RUSTUP_TOOLCHAIN": MSRV})
        progress["initial"]["controlled_child_environment"] = child_evidence
        persist(output, progress)
        stage = "build"
        # This is intentionally outside a unique campaign directory: Cargo's
        # fixed release cache is reusable across baseline/candidate campaigns.
        # The executable actually run is copied below into this campaign.
        target = ROOT / "target" / "s8-material-baseline" / args.backend
        build = run_process([args.cargo, f"+{MSRV}", "build", "--manifest-path", str(MANIFEST), "--locked", "--release",
                             "--no-default-features", "--features", args.backend, "--message-format=json-render-diagnostics",
                             "--target-dir", str(target)], CONSUMER,
                            output / "logs" / "build.log", args.build_timeout_seconds, child)
        locked_packages = resolved_git_packages(CONSUMER / "Cargo.lock")
        build_inventory = built_fluxel_packages(output / "logs" / "build.log")
        source_verification = verify_built_sources(build_inventory, locked_packages, required_built_packages(args.backend))
        checkout_verification = verify_git_checkouts(build_inventory, locked_packages)
        progress["attempts"].append({"kind": "build", "run": build, "target_cache": str(target),
                                     "built_fluxel_packages": build_inventory,
                                     "built_source_verification": source_verification,
                                     "built_git_checkout_verification": checkout_verification})
        persist(output, progress)
        if build.get("exit_status") != 0:
            raise RuntimeError(f"build failed; see {build['log']}")
        if source_verification["status"] != "verified" or checkout_verification["status"] != "verified":
            raise RuntimeError("built package source/checkout verification did not match the locked consumer")
        cargo_binary = executable(target)
        if not cargo_binary.is_file():
            raise RuntimeError(f"build produced no release consumer: {cargo_binary}")
        retained_directory = output / "executed-binary"
        retained_directory.mkdir(exist_ok=False)
        retained = retain_executable(cargo_binary, retained_directory)
        binary = Path(retained["path"])
        before = {"path": relative(binary), "bytes": binary.stat().st_size, "sha256": digest(binary)}
        progress["executable_retention"] = retained
        progress["executable_before"] = before
        persist(output, progress)
        stage = "smoke"
        smoke = run_group("smoke", binary, output, 1, True, args.run_timeout_seconds, child, progress)
        stage = "warmup"
        run_group("warmup", binary, output, args.warmup, False, args.run_timeout_seconds, child, progress)
        stage = "samples"
        samples = run_group("samples", binary, output, args.samples, False, args.run_timeout_seconds, child, progress)
        stage = "final-source-check"
        source_is_stable(inputs)
        after = {"path": relative(binary), "bytes": binary.stat().st_size, "sha256": digest(binary)}
        if after != before:
            raise RuntimeError("release executable changed after smoke/sample capture")
        record = {"schema_version": 1, "kind": "s8-material-baseline", "status": "complete",
                  "scope": "Selected native material-heavy baseline; no optimization or closure decision",
                  "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "decision_boundary": "baseline only; thresholds/candidates require later observed-noise evidence",
                  "roles": {"cold": "representative cold material admission", "warm": "representative steady frame",
                            "replacement-10pct": "representative texture-generation refresh", "smoke": "setup/correctness only"},
                  "source": {**progress["initial"]["source"], "git_after": git_snapshot(), "manifest_sha256": digest(MANIFEST),
                             "resolved_git_packages": locked_packages, "built_fluxel_packages": build_inventory,
                             "built_source_verification": source_verification,
                             "built_git_checkout_verification": checkout_verification},
                  "environment": progress["initial"]["environment"], "toolchain": progress["initial"]["toolchain"],
                  "parameters": progress["initial"]["parameters"], "build": build, "controlled_child_environment": child_evidence,
                  "executable_before": before, "executable_after": after, "smoke": smoke, "samples": samples,
                  "summary": metric_summary(samples), "unobserved": {"gpu_duration_ns": "No timestamp-query evidence is configured.",
                  "heap_allocator": "No heap/allocator profiler is configured."}}
        (output / "baseline.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        progress.update({"status": "complete", "state": "complete"})
        persist(output, progress)
        print(output / "baseline.json")
        return 0
    except Exception as error:
        append_error(output, progress, stage, error)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"S8 baseline failed: {error}", file=sys.stderr)
        raise SystemExit(1)
