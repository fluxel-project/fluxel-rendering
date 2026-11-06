"""Evidence-ledger checks shared by the formal S7 reviewer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile


BACKENDS = ("dx12", "native-gl-wgl")
WORKLOADS = ("s7-deferred", "s7-deferred-surface", "s7-deferred-device-isolation")


def sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def object_at(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {}, [f"missing or malformed JSON: {path}: {error}"]
    return (value, []) if isinstance(value, dict) else ({}, [f"JSON object required: {path}"])


def contained(root: Path, text: object) -> Path | None:
    if not isinstance(text, str) or not text:
        return None
    path = (root / text).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path


def retained_path(root: Path, value: object) -> Path | None:
    """Resolve an original runner path inside this retained attempt only."""
    path = contained(root, value)
    if path is not None:
        return path
    if isinstance(value, str):
        original = Path(value)
        if root.name in original.parts:
            relative = Path(*original.parts[original.parts.index(root.name) + 1:])
            return contained(root, str(relative))
    return None


def hash_records(root: Path, records: object) -> list[str]:
    if not isinstance(records, list) or not records:
        return ["hash-bound artifact records are missing"]
    failures = []
    for item in records:
        value = item.get("path") if isinstance(item, dict) else None
        path = retained_path(root, value)
        expected = item.get("sha256") if isinstance(item, dict) else None
        if path is None or not isinstance(expected, str) or sha256(path) != expected:
            failures.append("artifact hash record is missing, escapes closure, or mismatches")
    return failures


def source_snapshot_failures(attempt: dict[str, Any], directory: Path) -> list[str]:
    """Verify frozen source/locks inside the retained archive, never worktree paths."""
    snapshot = attempt.get("source_snapshot", {})
    path = contained(directory, snapshot.get("path") if isinstance(snapshot, dict) else None)
    if path is None:  # runner stores an absolute attempt path in this field.
        candidate = Path(snapshot.get("path", "")) if isinstance(snapshot, dict) else Path()
        if directory.name in candidate.parts:
            candidate = directory.joinpath(*candidate.parts[candidate.parts.index(directory.name) + 1:])
        else:
            candidate = Path()
        path = candidate if candidate.is_file() else None
    if not isinstance(snapshot, dict) or snapshot.get("kind") != "exact_worktree_inputs" or path is None or sha256(path) != snapshot.get("sha256"):
        return ["exact retained source snapshot is missing or hash-mismatched"]
    failures = []
    try:
        with zipfile.ZipFile(path) as archive:
            listed = attempt.get("source_inputs", [])
            frozen = json.loads(archive.read("snapshot-inputs.json"))
            if frozen != listed: failures.append("snapshot input index differs from attempt source_inputs")
            for item in listed:
                name = item.get("path") if isinstance(item, dict) else None
                member = name.replace("\\", "/") if isinstance(name, str) else None
                if not isinstance(member, str) or hashlib.sha256(archive.read(member)).hexdigest() != item.get("sha256"):
                    failures.append("snapshot member is absent or hash-mismatched")
            for lock in attempt.get("locks", []):
                # Absolute original paths are informational; find their root-relative
                # suffix in the immutable input archive.
                original = str(lock.get("path", "")).replace("\\", "/") if isinstance(lock, dict) else ""
                consumer_locks = (
                    "crates/renderer/examples/rendering-integration/Cargo.lock",
                    "examples/rendering-integration/Cargo.lock",
                )
                name = next((candidate for candidate in consumer_locks if original.endswith(candidate)), "Cargo.lock")
                if hashlib.sha256(archive.read(name)).hexdigest() != lock.get("sha256"):
                    failures.append(f"frozen lock bytes mismatch: {name}")
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        failures.append(f"retained source snapshot is unreadable: {error}")
    return failures


def attempt_failures(attempt: dict[str, Any], closure: Path, backend: str, workload: str) -> list[str]:
    failures: list[str] = []
    if attempt.get("backend") != backend or attempt.get("runner_exit_status") != 0:
        failures.append("attempt backend or successful runner result is not retained")
    spec = attempt.get("workload")
    if not isinstance(spec, dict) or not isinstance(spec.get("selection"), str):
        failures.append("attempt does not retain the selected workload identity")
    directory = contained(closure, attempt.get("directory"))
    if directory is None or not directory.is_dir():
        return failures + ["attempt directory is absent or escapes closure"]
    failures += source_snapshot_failures(attempt, directory)
    failures += hash_records(directory, attempt.get("retained_attempt_files"))
    if attempt.get("source_consistency_after_attempt", {}).get("status") != "stable":
        failures.append("source was not stable after attempt")
    built = attempt.get("built_source_verification", {})
    checkout = attempt.get("built_git_checkout_verification", {})
    if built.get("status") != "verified" or built.get("mismatches") or built.get("missing_required_packages"):
        failures.append("built source verification is not retained")
    if checkout.get("status") != "verified" or checkout.get("mismatches") or not checkout.get("checkouts"):
        failures.append("built Git checkout verification is not retained")
    if attempt.get("validation", {}).get("status") != "observed":
        failures.append("native validation was not observed")
    validation = attempt.get("validation", {})
    if backend == "native-gl-wgl" and validation.get("native_diagnostic_count") != 0:
        failures.append("WGL native diagnostic count is nonzero or missing")
    if backend == "dx12":
        diagnostics = validation.get("diagnostics", {})
        if diagnostics.get("unreviewed_native_diagnostics"):
            failures.append("DX12 unreviewed native diagnostics remain")
    commands = attempt.get("commands")
    if not isinstance(commands, list) or not commands:
        return failures + ["command records are absent"]
    tests = 0
    direct = 0
    for command in commands:
        result = command.get("result", {}) if isinstance(command, dict) else {}
        if result.get("exit_status") != 0 or result.get("timed_out") is not False:
            failures.append("a retained command did not complete successfully")
        failures += hash_records(directory, result.get("raw_logs"))
        if command.get("kind") == "test":
            if result.get("test_command_recognized") is True and type(result.get("executed")) is int and result["executed"] > 0:
                tests += 1
        if command.get("kind") == "direct_executable":
            direct += 1
            # Exact entry-argument matching is performed against the canonical
            # smoke_s7 contract by the caller, after ledger provenance passes.
            if result.get("nested_workloads", {}).get("status") != "verified":
                failures.append("nested workload verification is not retained")
    # The formal runner intentionally executes the focused test group exactly
    # once per same-source six-attempt closure; matrix_failures checks its
    # union, so other native attempts must not be rejected for omitting it.
    if direct != 1:
        failures.append("attempt must retain exactly one direct executable command")
    return failures


def workload_manifest(run: Path, workload: str) -> tuple[dict[str, Any], list[str]]:
    manifest, failures = object_at(run / f"{workload}-workload.json")
    if manifest.get("schemaVersion") != 1 or manifest.get("workload") != workload:
        failures.append("nested workload schema or identity is invalid")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        failures.append("nested workload has no cases")
        return manifest, failures
    names: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str) or case["name"] in names:
            failures.append("nested workload case name is absent or duplicated")
            continue
        names.add(case["name"])
        artifacts = case.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            failures.append(f"{case['name']}: no artifact inventory")
            continue
        for item in artifacts:
            # Native manifests deliberately contain relative string paths. Their
            # hashes live in the runner's immutable retained-file ledger, not
            # in a second Rust-side schema.
            path = contained(run, item.get("path") if isinstance(item, dict) else item)
            if path is None or not path.is_file():
                failures.append(f"{case['name']}: listed artifact is absent or escapes run")
            elif isinstance(item, dict) and sha256(path) != item.get("sha256"):
                failures.append(f"{case['name']}: listed artifact hash mismatch")
    return manifest, failures
