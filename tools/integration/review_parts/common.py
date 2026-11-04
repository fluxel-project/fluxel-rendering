#!/usr/bin/env python3
"""Review two explicit, current-source S0-S3 native integration attempts.

This is intentionally a bounded integration-evidence review, not a stage
closure or support decision. It does not amend the smoke manifest, consult
historical decisions as approval, or treat a screenshot, an enabled environment
variable, or an exit code as a visual or diagnostic approval.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[3]
BACKENDS = ("dx12", "native-gl-wgl")
_SMOKE_SPEC = importlib.util.spec_from_file_location(
    "integration_smoke", Path(__file__).resolve().parents[1] / "smoke.py"
)
assert _SMOKE_SPEC and _SMOKE_SPEC.loader
_SMOKE = importlib.util.module_from_spec(_SMOKE_SPEC)
_SMOKE_SPEC.loader.exec_module(_SMOKE)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result(name: str, passed: bool, reasons: list[str], **evidence: Any) -> dict[str, Any]:
    return {"requirement": name, "status": "passed" if passed else "open",
            "reasons": reasons, "evidence": evidence}


def read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as error:
        return None, str(error)


def under(directory: Path, item: str) -> Path | None:
    path = Path(item)
    candidate = path if path.is_absolute() else directory / path
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError:
        return None
    return candidate


def hash_records(records: list[dict[str, Any]], directory: Path) -> list[str]:
    failures: list[str] = []
    for record in records:
        path = under(directory, str(record.get("path", "")))
        if path is None:
            failures.append(f"artifact path escapes attempt: {record.get('path')!r}")
            continue
        actual = sha256(path)
        if actual is None:
            failures.append(f"missing retained artifact: {path}")
        elif actual != record.get("sha256"):
            failures.append(f"artifact hash mismatch: {path}")
    return failures


def current_source_failures(attempt: dict[str, Any], root: Path) -> list[str]:
    failures: list[str] = []
    resolved_root = root.resolve()
    recorded: set[Path] = set()
    scan_roots: set[tuple[Path, str]] = set()
    for source in attempt.get("source_inputs", []):
        item = source.get("path")
        if not isinstance(item, str):
            failures.append("source input has no path")
            continue
        path = Path(item)
        if path.is_absolute():
            try:
                path = path.resolve().relative_to(resolved_root)
            except ValueError:
                failures.append(f"source input is outside current root: {item}")
                continue
        recorded.add(path)
        if path.suffix in {".rs", ".wgsl", ".py"}:
            scan_roots.add((path.parent, path.suffix))
        actual = sha256(root / path)
        if actual != source.get("sha256"):
            failures.append(f"current source differs: {path}")
    candidates = {path.relative_to(resolved_root) for path in _SMOKE.source_input_paths(root)}
    for directory, suffix in scan_roots:
        for path in (root / directory).rglob("*" + suffix):
            if _SMOKE.source_path_is_safe(path, root):
                candidates.add(path.resolve().relative_to(resolved_root))
    for relative in candidates:
        if relative not in recorded:
            failures.append(f"current source was added after attempt: {relative}")
    return failures


def frozen_source_failures(attempt: dict[str, Any], closure: Path) -> list[str]:
    snapshot = attempt.get("source_snapshot")
    if not isinstance(snapshot, dict):
        return ["exact source-input archive is missing"]
    failures = hash_records([snapshot], closure)
    path = under(closure, str(snapshot.get("path", "")))
    if failures or path is None:
        return failures or ["source archive escapes closure"]
    if snapshot.get("kind") != "exact_worktree_inputs":
        failures.append("source archive does not identify an exact worktree capture")
    inputs = attempt.get("source_inputs")
    if not isinstance(inputs, list) or not inputs:
        failures.append("source archive has no declared source inputs")
        return failures
    if snapshot.get("files") != len(inputs):
        failures.append("source archive file count differs from declared inputs")
    try:
        with zipfile.ZipFile(path) as archive:
            expected: dict[str, str] = {}
            for item in inputs:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
                    failures.append("source input record is malformed")
                    continue
                member = Path(item["path"])
                if member.is_absolute() or ".." in member.parts:
                    failures.append(f"source input is unsafe for archive: {member}")
                    continue
                expected[member.as_posix()] = item["sha256"]
            if len(archive.namelist()) != len(expected) + 1 or set(archive.namelist()) != set(expected) | {"snapshot-inputs.json"}:
                failures.append("source archive membership does not match retained inputs")
            if json.loads(archive.read("snapshot-inputs.json")) != inputs:
                failures.append("source archive input manifest differs from attempt")
            for name, digest in expected.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    failures.append(f"frozen source hash differs: {name}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError, UnicodeDecodeError, zipfile.BadZipFile) as error:
        failures.append(f"cannot verify source archive: {error}")
    return failures


def locate_attempt(manifest: dict[str, Any], identifier: str, backend: str) -> tuple[dict[str, Any] | None, list[str]]:
    attempts = [item for item in manifest.get("attempts", []) if item.get("id") == identifier]
    if len(attempts) != 1:
        return None, [f"expected exactly one attempt named {identifier}, found {len(attempts)}"]
    attempt = attempts[0]
    if attempt.get("backend") != backend:
        return None, [f"attempt {identifier} is {attempt.get('backend')!r}, expected {backend}"]
    return attempt, []


def strictly_increasing(values: Any) -> bool:
    return isinstance(values, list) and len(values) == 3 and all(
        isinstance(value, int) and not isinstance(value, bool) for value in values
    ) and values[0] < values[1] < values[2]
