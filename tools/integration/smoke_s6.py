"""Strict manifest-first evidence checks for the S6 typed Forward workloads."""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Any

if __package__:
    from .smoke_inputs import sha256
else:
    from smoke_inputs import sha256


S6_SCHEMA_VERSION = 5
S6_STAGE = "S6 typed Forward integration evidence"

_CONTRACTS = {
    "s6-composed": {
        "selection": "s6-composed-offscreen-typed-quad",
        "entry_argument": "--s6-only",
        "directory": "s6-composed",
        "workload": "s6-composed-offscreen-typed-quad",
        "cases": (
            "original-three-frame-cache",
            "enabled-disabled-enabled",
            "independent-providers",
            "three-pending-replacements",
        ),
    },
    "s6-surface": {
        "selection": "s6-composed-surface-resize",
        "entry_argument": "--s6-surface-only",
        "directory": "s6-surface",
        "workload": "s6-composed-surface-resize",
        "cases": ("mesh-after-quad-initial", "quad-after-mesh-resized"),
    },
}


def s6_workload_contract(workload: str) -> dict[str, Any]:
    """Return the immutable runner/reviewer contract for one S6 workload."""
    try:
        contract = _CONTRACTS[workload]
    except KeyError as error:
        raise ValueError(f"not an S6 workload: {workload}") from error
    directory = contract["directory"]
    return {
        "selection": contract["selection"],
        "entry_argument": contract["entry_argument"],
        "required_nested_workload": f"{directory}/s6-{workload.removeprefix('s6-')}-workload.json",
        "expected_cases": list(contract["cases"]),
        "manifest_contract": "schemaVersion=1; exact named cases; every raw artifact enumerated by one case",
    }


def s6_nested_workload_evidence(run_directory: Path, workload: str) -> dict[str, Any]:
    """Verify one S6 manifest enumerates every emitted raw artifact by case."""
    contract = _CONTRACTS[workload]
    directory = run_directory / contract["directory"]
    manifest = directory / f"s6-{workload.removeprefix('s6-')}-workload.json"
    failures: list[str] = []
    records: list[dict[str, Any]] = []
    if not manifest.is_file():
        failures.append(f"missing required nested workload: {manifest.relative_to(run_directory).as_posix()}")
        return {"status": "incomplete", "records": records, "failures": failures}
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "incomplete",
            "records": records,
            "failures": [f"malformed S6 workload manifest: {error}"],
        }
    workload_root = directory.resolve()
    relative = manifest.relative_to(run_directory).as_posix()
    records.append({"path": relative, "sha256": sha256(manifest), "status": "retained"})
    if not isinstance(value, dict):
        failures.append("S6 workload manifest must be an object")
    elif value.get("schemaVersion") != 1 or value.get("workload") != contract["workload"]:
        failures.append("S6 workload manifest has an unexpected schema or workload identity")
    else:
        cases = value.get("cases")
        if not isinstance(cases, list):
            failures.append("S6 workload manifest does not enumerate cases")
        else:
            names: list[str] = []
            artifacts: set[Path] = set()
            for case in cases:
                if not isinstance(case, dict) or not isinstance(case.get("name"), str):
                    failures.append("S6 workload manifest has a case without a string name")
                    continue
                names.append(case["name"])
                listed = case.get("artifacts")
                if not isinstance(listed, list) or not listed or any(
                    not isinstance(name, str) or not name for name in listed
                ):
                    failures.append(f"S6 case {case['name']!r} has no complete artifact enumeration")
                    continue
                for name in listed:
                    path = Path(name)
                    windows_path = PureWindowsPath(name)
                    if (
                        path.is_absolute()
                        or path.drive
                        or ".." in path.parts
                        or windows_path.is_absolute()
                        or bool(windows_path.drive)
                        or ".." in windows_path.parts
                    ):
                        failures.append(f"S6 case artifact has an unsafe path: {name}")
                        continue
                    artifact = directory / path
                    resolved = artifact.resolve()
                    if resolved == manifest.resolve():
                        failures.append(f"S6 case artifact must not enumerate its workload manifest: {name}")
                    elif not resolved.is_relative_to(workload_root):
                        failures.append(f"S6 case artifact escapes workload root: {name}")
                    elif resolved in artifacts:
                        failures.append(f"S6 case artifact is duplicated or aliased: {name}")
                    else:
                        artifacts.add(resolved)
                        if not artifact.is_file():
                            failures.append(f"missing S6 raw artifact: {relative.rsplit('/', 1)[0]}/{name}")
            if sorted(names) != sorted(contract["cases"]) or len(names) != len(set(names)):
                failures.append("S6 workload manifest does not enumerate the expected named cases exactly")
            if not artifacts:
                failures.append("S6 workload manifest enumerated no raw artifacts")
            actual_files: set[Path] = set()
            for actual in directory.rglob("*"):
                if not actual.is_file() or actual == manifest:
                    continue
                resolved = actual.resolve()
                if not resolved.is_relative_to(workload_root):
                    failures.append(
                        f"S6 emitted artifact escapes workload root: {actual.relative_to(directory).as_posix()}"
                    )
                else:
                    actual_files.add(resolved)
            for unlisted in actual_files - artifacts:
                failures.append(
                    f"S6 emitted artifact is not manifest-enumerated: {unlisted.relative_to(workload_root).as_posix()}"
                )
            for missing in artifacts - actual_files:
                failures.append(
                    f"S6 manifest-enumerated artifact is absent from output closure: {missing.relative_to(workload_root).as_posix()}"
                )
    return {"status": "verified" if not failures else "incomplete", "records": records, "failures": failures}
