#!/usr/bin/env python3
"""Capture a reproducible, non-comparative S6 composed-workload baseline.

The primary metric is the wall latency of the complete eleven-submission
offscreen process. Surface work remains a correctness guard and is never timed
here. This tool records a baseline only: it makes no cross-revision claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

BENCH_TOOLS = Path(__file__).resolve().parent
if str(BENCH_TOOLS) not in sys.path:
    sys.path.insert(0, str(BENCH_TOOLS))
import s5_baseline as shared


INTEGRATION_TOOLS = Path(__file__).resolve().parents[1] / "integration"
if str(INTEGRATION_TOOLS) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_TOOLS))
from s6_review_parts import offscreen


ROOT = shared.ROOT
CONSUMER = shared.CONSUMER
MANIFEST = shared.MANIFEST
LOCKS = shared.LOCKS
REQUIRED_MEASUREMENT_FACTS = (
    "preparationElapsedNs", "graphCompileElapsedNs", "acceptedBufferAllocations",
    "acceptedBufferAllocationBytes", "acceptedTextureAllocations", "residentCopyBytes",
    "frameUploadBytes", "geometryHits", "textureHits", "frameReuses", "opaqueCandidates",
    "opaqueBatches", "opaqueDraws", "pendingPeak", "acceptedSubmissions", "compilations",
)


def facts(run_directory: Path) -> dict[str, Any]:
    """Validate S6's frozen eleven-frame facts and independent offscreen oracle."""
    run = run_directory / "s6-composed"
    failures = offscreen.review(run)
    if failures:
        raise ValueError("S6 offscreen review failed: " + "; ".join(failures))
    path = run / "s6-composed-measurements.json"
    if not path.is_file():
        raise ValueError(f"missing S6 measurement facts: {path}")
    measurement = json.loads(path.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_MEASUREMENT_FACTS if not isinstance(measurement.get(field), int)]
    if missing:
        raise ValueError("S6 measurement facts are missing integer fields: " + ", ".join(missing))
    if measurement["acceptedSubmissions"] != 11:
        raise ValueError("S6 baseline requires exactly eleven accepted offscreen submissions")
    return {
        "measurement_raw": measurement,
        "preparation_elapsed_ns": measurement["preparationElapsedNs"],
        "graph_compile_elapsed_ns": measurement["graphCompileElapsedNs"],
        "accepted_buffer_allocations": measurement["acceptedBufferAllocations"],
        "accepted_buffer_allocation_bytes": measurement["acceptedBufferAllocationBytes"],
        "accepted_texture_allocations": measurement["acceptedTextureAllocations"],
        "resident_copy_bytes": measurement["residentCopyBytes"],
        "frame_upload_bytes": measurement["frameUploadBytes"],
        "geometry_hits": measurement["geometryHits"],
        "texture_hits": measurement["textureHits"],
        "frame_reuses": measurement["frameReuses"],
        "pending_peak": measurement["pendingPeak"],
        "plan_compilations": measurement["compilations"],
    }


def persist_progress(output: Path, record: dict[str, Any]) -> None:
    """Durably retain initial metadata and each attempted outcome, including failures."""
    (output / "capture-progress.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sample_runs(binary: Path, output: Path, samples: int, timeout: float,
                progress: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for index in range(samples):
        directory = output / "runs" / f"sample-{index:02d}"
        if directory.exists():
            raise ValueError(f"refusing to reuse sample directory: {directory}")
        result = shared.run(
            [str(binary), "--output", str(directory), "--s6-only"],
            CONSUMER,
            directory / "process.log",
            timeout,
        )
        progress["outcomes"].append({"kind": "process", "index": index, "result": result})
        persist_progress(output, progress)
        if result.get("exit_status") != 0:
            raise RuntimeError(f"S6 workload sample {index} failed; see {result['log']}")
        result["facts"] = facts(directory)
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
    if args.samples < 5:
        parser.error("--samples must be at least 5 for the fixed S6 baseline")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    output = args.output.resolve()
    if output.exists():
        parser.error("--output must not already exist")
    output.mkdir(parents=True)
    git_before = shared.git_snapshot()
    source = shared.capture_source_snapshot(output)
    progress = {
        "schema_version": 1,
        "kind": "s6-composed-baseline-progress",
        "initial": {
            "source": source, "git": git_before, "environment": shared.environment(),
            "toolchain": {"rustc": shared.command_output(["rustc", "-Vv"]),
                          "cargo": shared.command_output([args.cargo, "-V"])},
            "parameters": {"backend": args.backend, "samples": args.samples,
                           "timeout_seconds": args.timeout_seconds, "entry": "--s6-only"},
        },
        "outcomes": [],
    }
    persist_progress(output, progress)

    cold_builds = []
    for index in range(args.samples):
        target = output / "build" / f"cold-{index:02d}"
        result = shared.run(shared.build_command(args.cargo, args.backend, target), CONSUMER,
                            output / "logs" / f"cold-build-{index:02d}.log", args.timeout_seconds)
        progress["outcomes"].append({"kind": "cold-build", "index": index, "result": result})
        persist_progress(output, progress)
        if result.get("exit_status") != 0:
            raise RuntimeError(f"cold build {index} failed; see {result['log']}")
        binary = shared.executable(target)
        if not binary.is_file():
            raise RuntimeError(f"cold build {index} did not produce {binary}")
        result["executable_bytes"] = binary.stat().st_size
        result["executable_sha256"] = shared.digest(binary)
        cold_builds.append(result)

    warm_target = output / "build" / "warm"
    warmup = shared.run(shared.build_command(args.cargo, args.backend, warm_target), CONSUMER,
                         output / "logs" / "warmup-build.log", args.timeout_seconds)
    progress["outcomes"].append({"kind": "warmup-build", "result": warmup})
    persist_progress(output, progress)
    if warmup.get("exit_status") != 0:
        raise RuntimeError(f"warm-up build failed; see {warmup['log']}")
    warm_builds = []
    for index in range(args.samples):
        result = shared.run(shared.build_command(args.cargo, args.backend, warm_target), CONSUMER,
                            output / "logs" / f"warm-build-{index:02d}.log", args.timeout_seconds)
        progress["outcomes"].append({"kind": "warm-build", "index": index, "result": result})
        persist_progress(output, progress)
        if result.get("exit_status") != 0:
            raise RuntimeError(f"warm build {index} failed; see {result['log']}")
        warm_builds.append(result)
    binary = shared.executable(warm_target)
    if not binary.is_file():
        raise RuntimeError(f"warm build did not produce {binary}")

    shared.require_stable_source(source["inputs"])
    samples = sample_runs(binary, output, args.samples, args.timeout_seconds, progress)
    summary = {
        "cold_build_wall_ns": shared.distribution([sample["wall_ns"] for sample in cold_builds]),
        "warm_build_wall_ns": shared.distribution([sample["wall_ns"] for sample in warm_builds]),
        "release_executable_bytes": shared.distribution([sample["executable_bytes"] for sample in cold_builds]),
        "process_wall_ns": shared.distribution([sample["wall_ns"] for sample in samples]),
    }
    for key in ("preparation_elapsed_ns", "graph_compile_elapsed_ns", "accepted_buffer_allocations",
                "accepted_buffer_allocation_bytes", "accepted_texture_allocations", "resident_copy_bytes",
                "frame_upload_bytes", "geometry_hits", "texture_hits", "frame_reuses", "pending_peak",
                "plan_compilations"):
        summary[key] = shared.distribution([sample["facts"][key] for sample in samples])
    record = {
        "schema_version": 1,
        "kind": "s6-composed-baseline",
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision_boundary": "baseline only; no cross-revision improvement or regression claim",
        "workload_role": "representative: eleven-submission S6 offscreen typed-feature process",
        "guard_boundary": "surface presentation is a correctness guard and is not timed",
        "source": {
            "git_head_baseline": git_before["revision"],
            "git_status_baseline": git_before["status"],
            "exact_worktree_inputs": source["archive"], "inputs": source["inputs"],
            "locks": [{"path": str(path.relative_to(ROOT)), "sha256": shared.digest(path)} for path in LOCKS],
            "resolved_rhi": shared.locked_packages(CONSUMER / "Cargo.lock", {"fluxel-rhi"}),
            "consumer_manifest_sha256": shared.digest(MANIFEST),
        },
        "environment": shared.environment(),
        "toolchain": {"rustc": shared.command_output(["rustc", "-Vv"]),
                      "cargo": shared.command_output([args.cargo, "-V"])},
        "parameters": {"backend": args.backend, "samples": args.samples,
                       "timeout_seconds": args.timeout_seconds, "profile": "release", "entry": "--s6-only"},
        "measurements": {"cold_builds": cold_builds, "warmup_build": warmup,
                         "warm_builds": warm_builds, "release_executable_bytes": binary.stat().st_size,
                         "release_executable_sha256": shared.digest(binary), "s6_process_samples": samples,
                         "summary": summary},
        "unobserved": {
            "gpu_duration_ns": "No timestamp-query measurement is configured.",
            "heap_allocator": "No heap or allocator profiler is configured.",
        },
    }
    shared.require_stable_source(source["inputs"])
    git_after = shared.git_snapshot()
    record["source"]["git_head_after_capture"] = git_after["revision"]
    record["source"]["git_status_after_capture"] = git_after["status"]
    path = output / "baseline.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"s6 baseline failed: {error}", file=sys.stderr)
        raise SystemExit(1)
