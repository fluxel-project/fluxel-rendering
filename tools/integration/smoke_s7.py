"""Strict inventory checks for the formal S7 Deferred workloads."""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Any

if __package__:
    from .smoke_inputs import sha256
else:
    from smoke_inputs import sha256


S7_SCHEMA_VERSION = 6
S7_STAGE = "S7 Deferred integration evidence"

_CONTRACTS = {
    "s7-deferred": {
        "selection": "s7-deferred-native-offscreen-probe",
        "entry_argument": "--s7-deferred-only",
        "directory": "s7-deferred",
        "manifest": "s7-deferred-workload.json",
        "cases": (
            "flat", "tilt", "bend-first", "bend-second",
            "code-node-first", "code-node-second", "pbr-dielectric", "pbr-metal",
            "pbr-rough", "pbr-base-color", "pbr-normal-transform",
        ),
        "subworkloads": (
            ("s7-deferred-cache-workload.json", "s7-deferred-cache", ("cold", "warm-state", "warm-settled", "tint", "texture-replacement", "source-replacement")),
        ),
    },
    "s7-deferred-surface": {
        "selection": "s7-deferred-color-surface-resize",
        "entry_argument": "--s7-deferred-surface-only",
        "directory": "s7-deferred-surface",
        "manifest": "s7-deferred-surface-workload.json",
        "cases": ("initial", "resized"),
        "subworkloads": (
            ("s7-deferred-surface-pbr-workload.json", "s7-deferred-pbr-surface", ("metal", "rough")),
            ("s7-deferred-surface-deformation-workload.json", "s7-deferred-surface-deformation", ("first", "second")),
        ),
    },
    "s7-deferred-device-isolation": {
        "selection": "s7-deferred-device-isolation",
        "entry_argument": "--s7-device-isolation-only",
        "directory": "s7-deferred-device-isolation",
        "manifest": "s7-deferred-device-isolation-workload.json",
        "cases": (
            "old-cold", "old-warm", "new-cold", "new-warm", "old-return",
            "foreign-submission-rejected", "new-replacement", "old-replacement",
        ),
        "subworkloads": (),
    },
}


def s7_workload_contract(workload: str) -> dict[str, Any]:
    """Return the immutable runner/reviewer contract for one S7 workload."""
    try:
        value = _CONTRACTS[workload]
    except KeyError as error:
        raise ValueError(f"not an S7 workload: {workload}") from error
    directory = value["directory"]
    return {
        "selection": value["selection"], "entry_argument": value["entry_argument"],
        "directory": directory,
        "required_nested_workload": f"{directory}/{value['manifest']}",
        "expected_cases": list(value["cases"]),
        "subworkloads": [name for name, _identity, _cases in value["subworkloads"]],
        "manifest_contract": (
            "schemaVersion=1; exact named cases; every raw artifact is listed exactly once "
            "by sharedArtifacts, a case, or a declared subworkload manifest"
        ),
    }


def _safe_name(name: object) -> Path | None:
    if not isinstance(name, str) or not name:
        return None
    path, windows = Path(name), PureWindowsPath(name)
    if (path.is_absolute() or path.drive or ".." in path.parts or windows.is_absolute()
            or windows.drive or ".." in windows.parts):
        return None
    return path


def _manifest_inventory(
    root: Path, manifest: Path, identity: str, expected_cases: tuple[str, ...],
    expected_subworkloads: tuple[tuple[str, str, tuple[str, ...]], ...],
    records: list[dict[str, Any]], failures: list[str],
) -> set[Path]:
    """Validate one manifest and return its declared data artifacts."""
    if not manifest.is_file():
        failures.append(f"missing required nested workload: {manifest.resolve().relative_to(root).as_posix()}")
        return set()
    records.append({"path": manifest.resolve().relative_to(root).as_posix(), "sha256": sha256(manifest), "status": "retained"})
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"malformed S7 workload manifest: {error}")
        return set()
    if not isinstance(value, dict) or value.get("schemaVersion") != 1 or value.get("workload") != identity:
        failures.append("S7 workload manifest has an unexpected schema or workload identity")
        return set()
    declared: set[Path] = set()
    def collect(names: object, owner: str) -> None:
        if not isinstance(names, list) or not names:
            failures.append(f"S7 {owner} has no complete artifact enumeration")
            return
        for name in names:
            relative = _safe_name(name)
            if relative is None:
                failures.append(f"S7 {owner} has an unsafe artifact path: {name!r}")
                continue
            artifact = (manifest.parent / relative).resolve()
            if not artifact.is_relative_to(root.resolve()) or artifact == manifest.resolve():
                failures.append(f"S7 {owner} artifact escapes its workload root: {name}")
            elif artifact in declared:
                failures.append(f"S7 artifact is duplicated or aliased: {name}")
            else:
                declared.add(artifact)
    collect(value.get("sharedArtifacts"), "sharedArtifacts")
    cases = value.get("cases")
    names: list[str] = []
    if not isinstance(cases, list):
        failures.append("S7 workload manifest does not enumerate cases")
    else:
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("name"), str):
                failures.append("S7 workload manifest has a case without a string name")
                continue
            names.append(case["name"])
            collect(case.get("artifacts"), f"case {case['name']!r}")
    if sorted(names) != sorted(expected_cases) or len(names) != len(set(names)):
        failures.append("S7 workload manifest does not enumerate the expected named cases exactly")
    expected_names = [name for name, _identity, _cases in expected_subworkloads]
    if value.get("subworkloads", []) != expected_names:
        failures.append("S7 workload manifest does not declare the expected subworkloads exactly")
    for name, nested_identity, nested_cases in expected_subworkloads:
        nested = manifest.parent / name
        _manifest_inventory(root, nested, nested_identity, nested_cases, (), records, failures)
    return declared


def s7_nested_workload_evidence(run_directory: Path, workload: str) -> dict[str, Any]:
    """Reject incomplete, aliased, or unaccounted S7 workload output."""
    contract = _CONTRACTS[workload]
    directory = run_directory / contract["directory"]
    root = directory.resolve()
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    _manifest_inventory(root, directory / contract["manifest"], contract["selection"], contract["cases"], contract["subworkloads"], records, failures)
    if directory.is_dir():
        declared_paths: set[Path] = set()
        for manifest in directory.rglob("*workload.json"):
            try:
                value = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for names in [value.get("sharedArtifacts")] + [case.get("artifacts") for case in value.get("cases", []) if isinstance(case, dict)]:
                if isinstance(names, list):
                    for name in names:
                        relative = _safe_name(name)
                        if relative is not None:
                            declared_paths.add((manifest.parent / relative).resolve())
        manifest_paths = {path.resolve() for path in directory.rglob("*workload.json")}
        expected_manifests = {directory.joinpath(contract["manifest"]).resolve()} | {
            directory.joinpath(name).resolve() for name, _identity, _cases in contract["subworkloads"]
        }
        for unexpected in sorted(manifest_paths - expected_manifests):
            failures.append(f"S7 emitted undeclared workload manifest: {unexpected.relative_to(root).as_posix()}")
        actual = {path.resolve() for path in directory.rglob("*") if path.is_file()}
        recorded = {record["path"] for record in records}
        for path in sorted(actual):
            relative = path.relative_to(root).as_posix()
            if relative not in recorded:
                records.append({"path": relative, "sha256": sha256(path), "status": "retained"})
        for path in sorted(actual - declared_paths - manifest_paths):
            failures.append(f"S7 emitted artifact is not manifest-enumerated: {path.relative_to(root).as_posix()}")
        for path in sorted(declared_paths - actual):
            failures.append(f"S7 manifest-enumerated artifact is absent from output closure: {path.relative_to(root).as_posix()}")
    else:
        failures.append(f"missing S7 workload output directory: {contract['directory']}")
    return {"status": "verified" if not failures else "incomplete", "records": records, "failures": failures}
