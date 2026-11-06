#!/usr/bin/env python3
"""Capture reproducible, non-comparative S5 Forward baseline samples.

This is deliberately separate from ``tools/integration/smoke.py``.  The smoke
runner retains correctness and diagnostic evidence; this tool measures a
frozen, already-smoked source snapshot.  It does not make a performance pass
or fail decision and it does not compare revisions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tomllib
import statistics
from typing import Any


INTEGRATION_TOOLS = Path(__file__).resolve().parents[1] / "integration"
if str(INTEGRATION_TOOLS) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_TOOLS))
from smoke_evidence import freeze_source_inputs
from smoke_inputs import source_consistency, source_input_paths


ROOT = Path(__file__).resolve().parents[2]
CONSUMER = ROOT / "crates/renderer/examples/rendering-integration"
MANIFEST = CONSUMER / "Cargo.toml"
LOCKS = (ROOT / "Cargo.lock", CONSUMER / "Cargo.lock")
REQUIRED_CACHE_FACTS = (
    "residentCopyDelta",
    "geometryHitDelta",
    "textureHitDelta",
    "frameReuseDelta",
    "opaqueCandidateDelta",
    "opaqueBatchDelta",
    "opaqueDrawDelta",
)
REQUIRED_MEASUREMENT_FACTS = (
    "preparationElapsedNs",
    "graphCompileElapsedNs",
    "acceptedBufferAllocations",
    "acceptedBufferAllocationBytes",
    "acceptedTextureAllocations",
    "pendingPeak",
    "residentStaticUploadBytes",
    "frameUploadBytes",
    "geometryCacheHits",
    "textureCacheHits",
    "frameAllocationReuses",
    "planCompilations",
)


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_snapshot() -> dict[str, Any]:
    return {
        "revision": command_output(["git", "rev-parse", "HEAD"]),
        "status": command_output(["git", "status", "--porcelain=v1"]),
    }


def baseline_input_paths(root: Path = ROOT) -> list[Path]:
    """Extend the smoke closure with every Python file that drives this tool."""
    bench_tools = root / "tools" / "bench"
    paths = [*source_input_paths(root), *bench_tools.glob("*.py")]
    resolved_root = root.resolve()
    return sorted({path.resolve() for path in paths if path.is_file()},
                  key=lambda path: path.relative_to(resolved_root).as_posix())


def source_inputs(root: Path = ROOT) -> list[dict[str, str]]:
    """Hash exact source bytes, including dirty and untracked inputs."""
    inputs = []
    resolved_root = root.resolve()
    for path in baseline_input_paths(root):
        value = digest(path)
        if value is None:
            raise ValueError(f"baseline source input disappeared: {path}")
        inputs.append({"path": path.relative_to(resolved_root).as_posix(), "sha256": value})
    return inputs


def require_stable_source(inputs: list[dict[str, str]], root: Path = ROOT) -> None:
    """Reject edits or new relevant Python inputs before they can be attributed."""
    consistency = source_consistency(inputs, root)
    if consistency["status"] != "stable":
        changed = ", ".join(item["path"] for item in consistency["changed_inputs"])
        raise RuntimeError(f"Rendering source changed during capture: {changed}")


def capture_source_snapshot(output: Path, root: Path = ROOT) -> dict[str, Any]:
    """Archive the precise input closure, then prove it remained stable."""
    inputs = source_inputs(root)
    frozen = freeze_source_inputs(inputs, output / "source-inputs.zip", root)
    require_stable_source(inputs, root)
    return {"inputs": inputs, "archive": frozen}


def locked_packages(lock: Path, names: set[str]) -> list[dict[str, str]]:
    """Record selected Git sources instead of inferring sibling checkout HEADs."""
    try:
        packages = tomllib.loads(lock.read_text(encoding="utf-8")).get("package", [])
    except (OSError, tomllib.TOMLDecodeError):
        return []
    return [
        {
            key: value
            for key, value in (
                ("name", package.get("name")),
                ("version", package.get("version")),
                ("source", package.get("source")),
            )
            if isinstance(value, str)
        }
        for package in packages
        if isinstance(package, dict) and package.get("name") in names
    ]


def powershell_json(expression: str) -> Any:
    if os.name != "nt":
        return None
    text = command_output(["powershell", "-NoProfile", "-Command", expression])
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "cpu": powershell_json("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"),
        "gpu": powershell_json("Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress"),
        "power_scheme": command_output(["powercfg", "/GETACTIVESCHEME"]) if os.name == "nt" else None,
    }


def run(command: list[str], cwd: Path, log: Path, timeout: float) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    with log.open("w", encoding="utf-8") as stream:
        try:
            completed = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            return {
                "command": command,
                "exit_status": completed.returncode,
                "wall_ns": time.perf_counter_ns() - started,
                "log": str(log),
                "log_sha256": digest(log),
            }
        except subprocess.TimeoutExpired:
            return {
                "command": command,
                "exit_status": None,
                "timed_out": True,
                "wall_ns": time.perf_counter_ns() - started,
                "log": str(log),
                "log_sha256": digest(log),
            }


def build_command(cargo: str, backend: str, target: Path) -> list[str]:
    return [
        cargo, "build", "--manifest-path", str(MANIFEST), "--locked", "--release",
        "--no-default-features", "--features", backend, "--target-dir", str(target),
    ]


def executable(target: Path) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return target / "release" / f"rendering-integration{suffix}"


def facts(run_directory: Path) -> dict[str, Any]:
    cache_path = run_directory / "s5-forward" / "s5-forward-cache-facts.json"
    measurement_path = run_directory / "s5-forward" / "s5-forward-measurements.json"
    if not cache_path.is_file():
        raise ValueError(f"missing S5 cache facts: {cache_path}")
    if not measurement_path.is_file():
        raise ValueError(f"missing S5 measurement facts: {measurement_path}")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_CACHE_FACTS if not isinstance(cache.get(key), int)]
    if missing:
        raise ValueError(f"S5 cache facts are missing integer fields: {', '.join(missing)}")
    missing = [key for key in REQUIRED_MEASUREMENT_FACTS if not isinstance(measurement.get(key), int)]
    if missing:
        raise ValueError(f"S5 measurement facts are missing integer fields: {', '.join(missing)}")
    return {
        "cache_raw": cache,
        "measurement_raw": measurement,
        "resident_static_upload_bytes": measurement["residentStaticUploadBytes"],
        "frame_upload_bytes": measurement["frameUploadBytes"],
        "preparation_elapsed_ns": measurement["preparationElapsedNs"],
        "graph_compile_elapsed_ns": measurement["graphCompileElapsedNs"],
        "accepted_buffer_allocations": measurement["acceptedBufferAllocations"],
        "accepted_buffer_allocation_bytes": measurement["acceptedBufferAllocationBytes"],
        "accepted_texture_allocations": measurement["acceptedTextureAllocations"],
        "pending_peak": measurement["pendingPeak"],
        "candidate_count": cache["opaqueCandidateDelta"],
        "batch_count": cache["opaqueBatchDelta"],
        "draw_count": cache["opaqueDrawDelta"],
    }


def distribution(values: list[int]) -> dict[str, int]:
    if not values:
        raise ValueError("cannot summarize an empty measurement series")
    return {
        "minimum": min(values),
        "median": int(statistics.median(values)),
        "maximum": max(values),
    }


def sample_runs(binary: Path, output: Path, samples: int, timeout: float) -> list[dict[str, Any]]:
    results = []
    for index in range(samples):
        directory = output / "runs" / f"sample-{index:02d}"
        if directory.exists():
            raise ValueError(f"refusing to reuse sample directory: {directory}")
        result = run([str(binary), "--output", str(directory), "--s5-only"], CONSUMER, directory / "process.log", timeout)
        if result.get("exit_status") != 0:
            raise RuntimeError(f"S5 workload sample {index} failed; see {result['log']}")
        result["cache_facts"] = facts(directory)
        results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("dx12", "native-gl-wgl"), required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--output", type=Path, required=True,
                        help="new empty result directory; never reuse a prior baseline")
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("--samples must be at least 3 for a noise-aware baseline")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    output = args.output.resolve()
    if output.exists():
        parser.error("--output must not already exist")
    output.mkdir(parents=True)
    snapshot_before = git_snapshot()
    source_snapshot = capture_source_snapshot(output)

    # A clean target directory measures a build from no local artifacts.  A
    # separate warm directory receives one untimed build before timed samples.
    cold_builds = []
    for index in range(args.samples):
        target = output / "build" / f"cold-{index:02d}"
        result = run(build_command(args.cargo, args.backend, target), CONSUMER,
                     output / "logs" / f"cold-build-{index:02d}.log", args.timeout_seconds)
        if result.get("exit_status") != 0:
            raise RuntimeError(f"cold build {index} failed; see {result['log']}")
        binary = executable(target)
        if not binary.is_file():
            raise RuntimeError(f"cold build {index} did not produce {binary}")
        result["executable_bytes"] = binary.stat().st_size
        result["executable_sha256"] = digest(binary)
        cold_builds.append(result)

    warm_target = output / "build" / "warm"
    warmup = run(build_command(args.cargo, args.backend, warm_target), CONSUMER,
                 output / "logs" / "warmup-build.log", args.timeout_seconds)
    if warmup.get("exit_status") != 0:
        raise RuntimeError(f"warm-up build failed; see {warmup['log']}")
    warm_builds = []
    for index in range(args.samples):
        result = run(build_command(args.cargo, args.backend, warm_target), CONSUMER,
                     output / "logs" / f"warm-build-{index:02d}.log", args.timeout_seconds)
        if result.get("exit_status") != 0:
            raise RuntimeError(f"warm build {index} failed; see {result['log']}")
        warm_builds.append(result)
    binary = executable(warm_target)
    if not binary.is_file():
        raise RuntimeError(f"warm build did not produce {binary}")

    process_samples = sample_runs(binary, output, args.samples, args.timeout_seconds)
    record = {
        "schema_version": 1,
        "kind": "s5-forward-baseline",
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision_boundary": "baseline only; no cross-revision improvement or regression claim",
        "workload_role": "representative: offscreen S5 Forward external-consumer workload",
        "guard_boundary": "surface presentation is a separate correctness/guard workload and is not timed here",
        "source": {
            # HEAD is useful orientation only. The frozen archive and its
            # per-file hashes, including dirty/untracked bytes, identify this
            # baseline's actual source state.
            "git_head_baseline": snapshot_before["revision"],
            "git_status_baseline": snapshot_before["status"],
            "exact_worktree_inputs": source_snapshot["archive"],
            "inputs": source_snapshot["inputs"],
            "locks": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in LOCKS],
            "resolved_rhi": locked_packages(CONSUMER / "Cargo.lock", {"fluxel-rhi"}),
            "consumer_manifest_sha256": digest(MANIFEST),
        },
        "environment": environment(),
        "toolchain": {"rustc": command_output(["rustc", "-Vv"]), "cargo": command_output([args.cargo, "-V"])},
        "parameters": {"backend": args.backend, "samples": args.samples, "timeout_seconds": args.timeout_seconds, "profile": "release"},
        "measurements": {
            "cold_builds": cold_builds,
            "warmup_build": warmup,
            "warm_builds": warm_builds,
            "release_executable_bytes": binary.stat().st_size,
            "release_executable_sha256": digest(binary),
            "s5_process_samples": process_samples,
            "summary": {
                "cold_build_wall_ns": distribution([sample["wall_ns"] for sample in cold_builds]),
                "warm_build_wall_ns": distribution([sample["wall_ns"] for sample in warm_builds]),
                "cold_release_executable_bytes": distribution([
                    sample["executable_bytes"] for sample in cold_builds
                ]),
                "process_wall_ns": distribution([sample["wall_ns"] for sample in process_samples]),
                "preparation_elapsed_ns": distribution([
                    sample["cache_facts"]["preparation_elapsed_ns"] for sample in process_samples
                ]),
                "graph_compile_elapsed_ns": distribution([
                    sample["cache_facts"]["graph_compile_elapsed_ns"] for sample in process_samples
                ]),
            },
        },
        "unobserved": {
            "gpu_duration_ns": "No timestamp-query measurement is configured.",
        },
    }
    require_stable_source(source_snapshot["inputs"])
    snapshot_after = git_snapshot()
    record["source"]["git_head_after_capture"] = snapshot_after["revision"]
    record["source"]["git_status_after_capture"] = snapshot_after["status"]
    path = output / "baseline.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"s5 baseline failed: {error}", file=sys.stderr)
        raise SystemExit(1)
