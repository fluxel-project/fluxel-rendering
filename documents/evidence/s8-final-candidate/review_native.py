#!/usr/bin/env python3
"""Fail-closed semantic audit of the archived native final matrix only.

The audit never reads the live workspace's runtime/evidence directories.  It
uses the materialized archive and frozen helper/source snapshot supplied on the
command line; all absolute producer paths are rebased in memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def rebase(value: str, logical: Path, frozen: Path) -> Path | None:
    value = value.replace("\\", "/")
    marker = "/target/"
    if marker in value:
        return logical / "target" / value.split(marker, 1)[1]
    origin = "e:/moyy/program/rust/fluxel/fluxel-rendering/"
    if value.lower().startswith(origin):
        return frozen / value[len(origin):]
    return None


def hash_records(value: object, logical: Path, frozen: Path, failures: list[str]) -> int:
    """Check all archived file hash records; unrelated source paths are ignored."""
    count = 0
    if isinstance(value, dict):
        path, expected = value.get("path"), value.get("sha256")
        if isinstance(path, str) and isinstance(expected, str) and len(expected) == 64:
            candidate = rebase(path, logical, frozen)
            if candidate is not None:
                count += 1
                if not candidate.is_file():
                    failures.append(f"missing archived hash target: {candidate.as_posix()}")
                elif digest(candidate) != expected:
                    failures.append(f"archived hash mismatch: {candidate.as_posix()}")
        for key, child in value.items():
            # Demo metadata retains a prior performance executable solely for
            # diagnosis.  It is explicitly not a selected final artifact.
            if key == "prior_performance_executable":
                continue
            count += hash_records(child, logical, frozen, failures)
    elif isinstance(value, list):
        for child in value:
            count += hash_records(child, logical, frozen, failures)
    return count


def zip_runtime_map(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        result = {}
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            if name.endswith((".rs", "/Cargo.toml", "/Cargo.lock")) or name in {"Cargo.toml", "Cargo.lock"}:
                result[name] = hashlib.sha256(archive.read(info)).hexdigest()
        return result


def source_zip_checks(logical: Path, frozen: Path, failures: list[str]) -> dict[str, object]:
    zips = sorted(logical.glob("target/**/source-inputs.zip"))
    final_zips = [path for path in zips if any(token in path.as_posix() for token in (
        "final-regression-", "final-lifecycle-", "finalCPU", "final-cpu"))]
    # The native 18 attempts retain their source snapshots under integration.
    final_zips += [path for path in zips if "/integration/" in path.as_posix()]
    final_zips = list(dict.fromkeys(final_zips))
    if not final_zips:
        failures.append("no final CPU/native/lifecycle source archives found")
        return {"archives": 0, "runtimeFiles": 0}
    maps = []
    for path in final_zips:
        try:
            maps.append((path, zip_runtime_map(path)))
        except (OSError, zipfile.BadZipFile) as error:
            failures.append(f"unreadable source archive {path.name}: {error}")
    baseline = maps[0][1] if maps else {}
    if not baseline:
        failures.append("final source archive has no Rust/manifest/lock files")
    for path, source_map in maps[1:]:
        if source_map != baseline:
            failures.append(f"production Rust/manifest/lock source differs: {path.as_posix()}")
    frozen_map = {
        path.relative_to(frozen).as_posix(): digest(path)
        for path in frozen.rglob("*") if path.is_file()
        and (path.suffix == ".rs" or path.name in {"Cargo.toml", "Cargo.lock"})
    }
    missing = sorted(name for name, value in baseline.items() if frozen_map.get(name) != value)
    if missing:
        failures.append("final runtime source differs from frozen source: " + ", ".join(missing[:8]))
    return {"archives": len(final_zips), "runtimeFiles": len(baseline), "frozenRuntimeFiles": len(frozen_map)}


def lifecycle_checks(logical: Path, frozen: Path, failures: list[str]) -> dict[str, object]:
    import sys
    helper_root = frozen / "tools" / "bench"
    sys.path.insert(0, str(helper_root))
    import s8_lifecycle_review  # archived helper only
    roots = sorted(logical.glob("target/bench/s8/final-lifecycle-*/run/s8-material-lifecycle"))
    if len(roots) != 2:
        failures.append(f"expected exactly two final lifecycle roots, found {len(roots)}")
        return {"roots": len(roots)}
    results = []
    for root in roots:
        try:
            results.append(s8_lifecycle_review.review(root))
        except Exception as error:  # reviewer must never convert an incomplete proof to green
            failures.append(f"lifecycle semantic review failed for {root.parent.parent.name}: {error}")
    return {"roots": len(roots), "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logical, frozen = args.logical.resolve(), args.frozen_source.resolve()
    failures: list[str] = []
    records = 0
    final = logical / "target" / "final-regression-20261105T0825"
    # Only final artifacts are candidates.  The materialized archive also
    # intentionally retains older failed/historical attempts whose raw files
    # are not part of the final matrix, so scanning every JSON would turn that
    # retention into a false closure failure.
    selected_json = [
        final / "source-inputs.json",
        final / "native-matrix" / "native-matrix.json",
        final / "demos" / "attempts.json",
    ]
    for manifest in logical.glob("target/integration/**/manifest.json"):
        ledger = load(manifest)
        if isinstance(ledger, dict):
            current = [attempt for attempt in ledger.get("attempts", [])
                       if isinstance(attempt, dict)
                       and str(attempt.get("timestamp_utc", "")).startswith("2026-11-05T08")]
            if not current:
                failures.append(f"no final native attempt selected in {manifest.as_posix()}")
            selected_json.extend([])
            for attempt in current:
                records += hash_records(attempt, logical, frozen, failures)
    for path in selected_json:
        try:
            records += hash_records(load(path), logical, frozen, failures)
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"unreadable final JSON {path.as_posix()}: {error}")
    source = source_zip_checks(logical, frozen, failures)
    lifecycle = lifecycle_checks(logical, frozen, failures)
    payload = {
        "schemaVersion": 1,
        "scope": "archived final native hash/source/lifecycle semantic review",
        "hashRecordsChecked": records,
        "source": source,
        "lifecycle": lifecycle,
        "status": "verified" if not failures else "open",
        "failures": failures,
        "humanVisualInspection": "hash-bound record only; this tool does not make a human approval decision",
        "stageClosureDecision": "not made by this tool",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
