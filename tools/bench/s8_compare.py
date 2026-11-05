#!/usr/bin/env python3
"""Interleave a frozen S8 native baseline with one freshly built candidate."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONSUMER = ROOT / "examples" / "rendering-integration"
MANIFEST = CONSUMER / "Cargo.toml"
MSRV = "1.87.0"
SCENARIOS = ("cold", "warm", "replacement-10pct")
COUNTERS = ("acceptedBufferAllocations", "acceptedBufferAllocationBytes", "acceptedTextureAllocations",
            "residentStaticUploadBytes", "frameUploadBytes", "pendingPeak", "draws", "geometryCacheHits",
            "textureCacheHits", "frameAllocationReuses", "planCompilations", "visibleCandidates", "batches")
INPUTS = ("s8-material-input.json", "s8-material-geometry.bin", "s8-material-textures-generation-1.rgba",
          "s8-material-textures-generation-2-replaced.rgba", "s8-material-variant-0.txt",
          "s8-material-variant-1.txt", "s8-material-variant-2.txt", "s8-material-variant-3.txt",
          "s8-material-replacement-input.json")

if str(ROOT / "tools" / "integration") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "integration"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_evidence import (built_fluxel_packages, freeze_source_inputs, required_built_packages, resolved_git_packages,
                            retain_executable, verify_built_sources, verify_git_checkouts)
from smoke_inputs import build_environment, controlled_child_environment, source_consistency, source_input_paths
import s8_material_baseline as base


def persist(output: Path, progress: dict[str, Any]) -> None:
    progress["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    (output / "comparison.partial.json").write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def error(output: Path, progress: dict[str, Any], stage: str, exc: BaseException) -> None:
    progress["status"] = "failed"; progress.setdefault("errors", []).append(
        {"stage": stage, "type": type(exc).__name__, "message": str(exc), "at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    persist(output, progress)


def source_archive() -> list[dict[str, str]]:
    return base.archive_inputs()


def relative(path: Path) -> str:
    return base.relative(path)


def digest(path: Path) -> str | None:
    return base.digest(path)


def retained_baseline(record: dict[str, Any], baseline: Path) -> tuple[Path, str]:
    if record.get("kind") != "s8-material-baseline" or record.get("status") != "complete":
        raise ValueError("baseline is not a completed S8 material baseline")
    backend = record.get("parameters", {}).get("backend")
    exe = record.get("executable_after", {})
    source = record.get("source", {})
    executable = ROOT / str(exe.get("path", ""))
    archive = ROOT / str(source.get("exact_worktree_inputs", {}).get("path", ""))
    if backend not in ("dx12", "native-gl-wgl") or not executable.is_file() or digest(executable) != exe.get("sha256"):
        raise ValueError("baseline backend or retained executable hash is invalid")
    if not archive.is_file() or digest(archive) != source.get("exact_worktree_inputs", {}).get("sha256"):
        raise ValueError("baseline exact source archive is missing or changed")
    locks = source.get("locks")
    if not isinstance(locks, list) or not locks:
        raise ValueError("baseline does not retain lock evidence")
    with zipfile.ZipFile(archive) as frozen:
        names = set(frozen.namelist())
        for item in locks:
            name = item.get("path") if isinstance(item, dict) else None
            frozen_hash = hashlib.sha256(frozen.read(name)).hexdigest() if isinstance(name, str) and name in names else None
            if not isinstance(name, str) or name not in names or frozen_hash != item.get("sha256"):
                raise ValueError("baseline archive does not retain every lock")
    return executable, backend


def input_hashes(directory: Path) -> dict[str, str]:
    if not directory.is_absolute():
        directory = ROOT / directory
    workload = directory / "s8-material-baseline"
    found = {name: digest(workload / name) for name in INPUTS if (workload / name).is_file()}
    required = set(INPUTS[:-1])
    if not required <= found.keys() or any(value is None for value in found.values()):
        raise ValueError(f"missing material workload inputs: {directory}")
    return found


def compare_pair(first: dict[str, Any], second: dict[str, Any]) -> None:
    if "facts" not in first or "facts" not in second:
        raise RuntimeError("A/B attempt failed before facts/oracle validation")
    if input_hashes(Path(first["directory"])) != input_hashes(Path(second["directory"])):
        raise ValueError("A/B workload input bytes differ")
    for name in COUNTERS:
        if first["facts"].get(name) != second["facts"].get(name):
            raise ValueError(f"GPU/plan counter changed: {name}")


def median(values: list[int]) -> int:
    if not values: raise ValueError("empty metric series")
    return int(statistics.median(values))


def complete_pairs(samples: list[dict[str, Any]], scenarios: tuple[str, ...], pairs: int) -> bool:
    for scenario in scenarios:
        actual: dict[int, list[str]] = {}
        for row in samples:
            if row.get("scenario") == scenario and isinstance(row.get("pair"), int) and row.get("label") in ("baseline", "candidate"):
                actual.setdefault(row["pair"], []).append(row["label"])
        if set(actual) != set(range(pairs)) or any(sorted(labels) != ["baseline", "candidate"] for labels in actual.values()):
            return False
    return True


def decision(samples: list[dict[str, Any]], scenarios: tuple[str, ...], pairs: int, screen: bool) -> dict[str, Any]:
    summary: dict[str, Any] = {}; complete = complete_pairs(samples, scenarios, pairs)
    formal = not screen and pairs >= 7 and set(scenarios) == set(SCENARIOS) and complete
    primary_ok = formal
    for scenario in scenarios:
        by_pair = {pair: {} for pair in range(pairs)}
        for row in samples:
            if row.get("scenario") == scenario and row.get("pair") in by_pair and row.get("label") in ("baseline", "candidate"):
                by_pair[row["pair"]][row["label"]] = row["facts"]["submitElapsedNs"]
        original = [item["baseline"] for item in by_pair.values() if "baseline" in item]
        chosen = [item["candidate"] for item in by_pair.values() if "candidate" in item]
        ratios = [item["candidate"] / item["baseline"] for item in by_pair.values() if set(item) == {"baseline", "candidate"}]
        b, c = median(original), median(chosen); improvement = 1 - c / b
        summary[scenario] = {"baseline_median_submit_elapsed_ns": b, "candidate_median_submit_elapsed_ns": c,
                             "candidate_over_baseline": c / b, "paired_candidate_over_baseline": ratios,
                             "paired_ratio_median": statistics.median(ratios) if ratios else None, "improvement": improvement}
        primary_ok = primary_ok and c * 10 <= b * 9
    return {"classification": "cheap-screen" if screen else "formal" if formal else "incomplete",
            "complete_unique_pairs": complete, "primary_acceptance": primary_ok, "scenarios": summary,
            "scope": "S8 candidate comparison only; not a whole-S8 closure decision"}


def guard_decision(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}; accepted = True
    for scenario in SCENARIOS:
        a = median([x["facts"]["submitElapsedNs"] for x in samples if x["label"] == "baseline" and x["scenario"] == scenario])
        b = median([x["facts"]["submitElapsedNs"] for x in samples if x["label"] == "candidate" and x["scenario"] == scenario])
        allowed = max(500_000, a * .10); regression = b - a
        result[scenario] = {"baseline_median_submit_elapsed_ns": a, "candidate_median_submit_elapsed_ns": b,
                            "allowed_regression_ns": allowed, "regression_ns": regression, "accepted": regression <= allowed}
        accepted = accepted and regression <= allowed
    return {"accepted": accepted, "scenarios": result, "role": "small 100-draw/10-texture guard"}


def record(binary: Path, directory: Path, scenario: str, smoke: bool, environment: dict[str, str]) -> dict[str, Any]:
    run = base.run_process(base.workload_command(binary, directory, scenario, smoke), CONSUMER,
                           directory / "process.log", 180.0, environment)
    observed = base.observed_raw_input_readback_hashes(directory)
    item: dict[str, Any] = {"directory": relative(directory), "scenario": scenario, "run": run,
                            "observed_raw_input_readback_artifacts": observed}
    if run.get("exit_status") == 0:
        try:
            checked = base.sample_record(run, directory, scenario, smoke)
            item.update({"facts": checked["facts"], "raw_input_readback_artifacts": checked["raw_input_readback_artifacts"],
                         "pixel_oracle": checked["pixel_oracle"]})
        except Exception as exc:
            item["validation_error"] = {"type": type(exc).__name__, "message": str(exc)}
    return item


def build_candidate(args: argparse.Namespace, output: Path, progress: dict[str, Any], environment: dict[str, str], backend: str) -> Path:
    target = ROOT / "target" / "s8-material-baseline" / backend
    run = base.run_process([args.cargo, f"+{MSRV}", "build", "--manifest-path", str(MANIFEST), "--locked", "--release",
                            "--no-default-features", "--features", backend, "--message-format=json-render-diagnostics",
                            "--target-dir", str(target)], CONSUMER, output / "logs" / "candidate-build.log", 900.0, environment)
    locked = resolved_git_packages(CONSUMER / "Cargo.lock"); inventory = built_fluxel_packages(output / "logs" / "candidate-build.log")
    source_check = verify_built_sources(inventory, locked, required_built_packages(backend)); checkout = verify_git_checkouts(inventory, locked)
    progress["attempts"].append({"kind": "candidate-build", "run": run, "target_cache": str(target), "built_fluxel_packages": inventory,
                                 "built_source_verification": source_check, "built_git_checkout_verification": checkout}); persist(output, progress)
    if run.get("exit_status") != 0 or source_check["status"] != "verified" or checkout["status"] != "verified":
        raise RuntimeError("candidate build or locked-source verification failed")
    cargo_binary = base.executable(target)
    (output / "candidate-executable").mkdir(exist_ok=False)
    retained = retain_executable(cargo_binary, output / "candidate-executable")
    path = Path(retained["path"]); progress["candidate_executable"] = {**retained, "bytes": path.stat().st_size}; persist(output, progress)
    return path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=7); parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    parser.add_argument("--screen", action="store_true"); parser.add_argument("--cargo", default="cargo")
    result = parser.parse_args()
    if result.pairs < 1 or (not result.screen and result.pairs < 7): parser.error("formal comparison requires --pairs >= 7")
    if result.screen and result.pairs > 2: parser.error("--screen permits at most 2 pairs")
    return result


def main() -> int:
    args = arguments(); output = args.output.resolve()
    if output.exists(): raise ValueError("--output must be new")
    output.mkdir(parents=True); progress: dict[str, Any] = {"schema_version": 1, "kind": "s8-material-comparison", "status": "partial", "attempts": [], "errors": []}; persist(output, progress)
    stage = "initialization"
    try:
        baseline_path = args.baseline.resolve(); baseline = json.loads(baseline_path.read_text(encoding="utf-8")); old_binary, backend = retained_baseline(baseline, baseline_path)
        scenarios = tuple(args.scenario or SCENARIOS); inputs = source_archive(); archive = freeze_source_inputs(inputs, output / "candidate-source-inputs.zip", ROOT)
        environment_evidence = build_environment(ROOT)
        progress["initial"] = {"baseline": {"path": str(baseline_path), "sha256": digest(baseline_path), "backend": backend,
                                             "executable": {"path": relative(old_binary), "sha256": digest(old_binary)}},
                               "candidate_source": {"inputs": inputs, "exact_worktree_inputs": archive}, "parameters": {"pairs": args.pairs, "scenarios": scenarios, "screen": args.screen}, "build_environment": environment_evidence,
                               "host_environment": base.host_environment(), "toolchain": {"msrv": MSRV, "rustc": base.command_output(["rustc", f"+{MSRV}", "-Vv"]), "cargo": base.command_output([args.cargo, f"+{MSRV}", "-V"])} }; persist(output, progress)
        if environment_evidence["status"] != "auditable": raise RuntimeError("uncontrolled build environment")
        environment, child_evidence = controlled_child_environment({"RUSTUP_TOOLCHAIN": MSRV}); progress["initial"]["controlled_child_environment"] = child_evidence; persist(output, progress)
        stage = "candidate-build"; candidate = build_candidate(args, output, progress, environment, backend)
        for smoke in (False, True):
            kind = "primary" if not smoke else "small-guard"
            for scenario in (scenarios if not smoke else SCENARIOS):
                for index in range(args.pairs):
                    labels = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
                    pair: list[dict[str, Any]] = []
                    for label in labels:
                        directory = output / kind / scenario / f"pair-{index:02d}" / label
                        item = record(old_binary if label == "baseline" else candidate, directory, scenario, smoke, environment)
                        item.update({"label": label, "pair": index, "smoke": smoke}); progress["attempts"].append(item); persist(output, progress); pair.append(item)
                    compare_pair(pair[0], pair[1])
        base.source_is_stable(inputs)
        if digest(old_binary) != baseline["executable_after"]["sha256"] or digest(candidate) != progress["candidate_executable"]["sha256"]:
            raise RuntimeError("a frozen executable changed during comparison")
        primary = [x for x in progress["attempts"] if x.get("kind") != "candidate-build" and not x.get("smoke")]
        guards = [x for x in progress["attempts"] if x.get("smoke")]
        final = {"schema_version": 1, "kind": "s8-material-comparison", "status": "complete", "baseline": progress["initial"]["baseline"],
                 "candidate_source": progress["initial"]["candidate_source"], "candidate_executable": progress["candidate_executable"],
                 "parameters": progress["initial"]["parameters"], "host_environment": progress["initial"]["host_environment"],
                 "toolchain": progress["initial"]["toolchain"], "attempts": progress["attempts"],
                 "comparison": decision(primary, scenarios, args.pairs, args.screen), "small_guard": guard_decision(guards),
                 "decision": "screen is not a final decision" if args.screen else "formal threshold/correctness/guard evidence only; not whole-S8 closure"}
        (output / "comparison.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        progress.update({"status": "complete"}); persist(output, progress); print(output / "comparison.json"); return 0
    except Exception as exc:
        error(output, progress, stage, exc); raise


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"S8 comparison failed: {exc}", file=sys.stderr); raise SystemExit(1)
