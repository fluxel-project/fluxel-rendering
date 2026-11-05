#!/usr/bin/env python3
"""Fail-closed independent review of the six-attempt S7 native matrix.

This tool only verifies retained evidence.  It cannot make an S7 closure
decision and deliberately treats schema drift or unproved facts as open.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

from s7_review_parts import common
from s7_review_parts import facts
from smoke_s7 import S7_SCHEMA_VERSION, S7_STAGE, s7_nested_workload_evidence, s7_workload_contract
try:
    # Implemented separately from ledger review so the pixel oracle has no
    # access to acceptance plumbing or saved expected-pixel shortcuts.
    from s7_oracles import review_offscreen, review_surface, review_device
except ImportError:  # Fail closed until the tracked independent oracle exists.
    review_offscreen = review_surface = review_device = None


Selection = tuple[dict[str, Any], Path]


def relative_path(run: Path, value: object) -> Path | None:
    return common.contained(run, value)


def nested_failures(attempt: dict[str, Any], closure: Path, workload: str) -> list[str]:
    directory = common.contained(closure, attempt.get("directory"))
    if directory is None:
        return ["attempt directory invalid"]
    run = directory / "run"
    contract = s7_workload_contract(workload)
    if attempt.get("workload") != contract:
        return ["attempt does not retain the exact S7 runner workload contract"]
    direct = [item for item in attempt.get("commands", []) if isinstance(item, dict) and item.get("kind") == "direct_executable"]
    if len(direct) != 1 or contract["entry_argument"] not in direct[0].get("command", []):
        return ["direct executable command does not select the canonical S7 workload"]
    evidence = s7_nested_workload_evidence(directory / "run", workload)
    failures = list(evidence["failures"])
    # The runner saves this inventory, including each raw file hash, in the
    # attempt ledger.  A nested manifest is an enumeration, not a hash store.
    records = attempt.get("nested_workload_records")
    if not isinstance(records, list) or records != evidence["records"]:
        failures.append("attempt does not bind the exact validated nested raw inventory")
    retained = list(attempt.get("retained_attempt_files", []))
    for command in attempt.get("commands", []):
        if isinstance(command, dict) and isinstance(command.get("result"), dict):
            retained.extend(command["result"].get("retained_files", []))
    bound = set()
    for item in retained:
        if isinstance(item, dict):
            path = common.retained_path(directory, item.get("path"))
            if path is not None:
                bound.add((path.relative_to(directory).as_posix(), item.get("sha256")))
    for artifact in (run / contract["directory"]).rglob("*"):
        if artifact.is_file():
            relative = str(artifact.relative_to(directory)).replace("\\", "/")
            digest = common.sha256(artifact)
            if (relative, digest) not in bound:
                failures.append(f"nested raw artifact is not hash-bound in attempt ledger: {relative}")
    if workload == "s7-deferred":
        if review_offscreen is None:
            failures.append("tracked independent offscreen oracle is unavailable")
        else:
            failures += review_offscreen(run / contract["directory"])
        failures += facts.review_offscreen(run / contract["directory"])
    elif workload == "s7-deferred-surface":
        if review_surface is None:
            failures.append("tracked independent Surface oracle is unavailable")
        else:
            failures += review_surface(
                run / contract["directory"],
                require_nondisturbing_capture=(
                    attempt.get("window_capture_policy") == "schema2-nondisturbing"
                ),
            )
        failures += facts.review_surface(run / contract["directory"])
    else:
        if review_device is None:
            failures.append("tracked independent two-device pixel oracle is unavailable")
        else:
            failures += review_device(run / contract["directory"])
        failures += facts.review_device(run / contract["directory"])
    return failures


def matrix_failures(selected: dict[tuple[str, str], Selection]) -> list[str]:
    expected = {(backend, workload) for backend in common.BACKENDS for workload in common.WORKLOADS}
    if set(selected) != expected:
        return ["S7 requires DX12 and WGL for all three selected workloads"]
    identities = [(attempt.get("id"), str(closure)) for attempt, closure in selected.values()]
    failures = []
    if len(identities) != len(set(identities)):
        failures.append("selected matrix reuses an attempt identity")
    closures = []
    for attempt, attempt_closure in selected.values():
        records = attempt.get("source_inputs", [])
        closures.append(sorted((item.get("path"), item.get("sha256")) for item in records if isinstance(item, dict)))
    if not closures or not closures[0] or any(value != closures[0] for value in closures):
        failures.append("matrix attempts do not retain one exact nonempty source closure")
    groups: set[str] = set()
    for attempt, attempt_closure in selected.values():
        for command in attempt.get("commands", []):
            if not isinstance(command, dict) or command.get("kind") != "test":
                continue
            result = command.get("result", {})
            directory = common.contained(attempt_closure, attempt.get("directory"))
            count = logged_test_count(command.get("command", []), result, directory)
            if result.get("exit_status") == 0 and result.get("test_command_recognized") is True and count is not None and count > 0 and count == result.get("executed"):
                text = " ".join(str(part).replace("\\", "/") for part in command.get("command", []))
                if "fluxel-renderer" in text and "--lib deferred" in text: groups.add("renderer-deferred")
                elif "rendering-integration/Cargo.toml" in text and "--bin rendering-integration" in text: groups.add("native-consumer")
                elif "external-renderer/Cargo.toml" in text: groups.add("external-consumer")
                elif "unittest discover" in text and "--start-directory tools/integration" in text and "--pattern test_s7*.py" in text: groups.add("evidence-tools")
                else: failures.append("focused test record is not one of the bounded S7 test groups")
    if groups != {"renderer-deferred", "native-consumer", "external-consumer", "evidence-tools"}:
        failures.append("same-source matrix lacks the exact four successful focused S7 test groups")
    return failures


def logged_test_count(command: object, result: object, directory: Path | None) -> int | None:
    """Re-parse retained hash-bound logs instead of trusting `executed`."""
    if not isinstance(command, list) or not command or not isinstance(result, dict): return None
    executable = Path(str(command[0])).name.lower()
    cargo = executable in {"cargo", "cargo.exe"} and "test" in command[1:]
    unit = executable.startswith("python") and command[1:3] == ["-m", "unittest"]
    if not cargo and not unit: return None
    values = []
    for record in result.get("raw_logs", []):
        path = common.retained_path(directory, record.get("path")) if directory is not None and isinstance(record, dict) else None
        if path is None or not path.is_file() or common.sha256(path) != record.get("sha256"): return None
        text = path.read_text(encoding="utf-8", errors="replace")
        values += re.findall(r"(?m)^test result: .*?\b(\d+) passed;" if cargo else r"(?m)^Ran (\d+) tests?", text)
    return sum(map(int, values)) if values else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt", nargs=3, action="append", required=True,
                        metavar=("ID", "BACKEND", "WORKLOAD"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    closure = args.closure.resolve()
    ledger, ledger_failures = common.object_at(closure / "manifest.json")
    if ledger.get("schema_version") != S7_SCHEMA_VERSION or ledger.get("stage") != S7_STAGE:
        ledger_failures.append("not the current S7 evidence ledger schema")
    attempts = ledger.get("attempts", [])
    checks = [{"requirement": "closure ledger", "status": "open" if ledger_failures else "passed",
               "reasons": ledger_failures}]
    selected: dict[tuple[str, str], Selection] = {}
    for identifier, backend, workload in args.attempt:
        reasons: list[str] = []
        matches = [item for item in attempts if isinstance(item, dict) and item.get("id") == identifier]
        if len(matches) != 1:
            reasons.append("attempt identity is absent or non-unique in closure ledger")
            attempt = {}
        else:
            attempt = matches[0]
        key = (backend, workload)
        if backend not in common.BACKENDS or workload not in common.WORKLOADS or key in selected:
            reasons.append("unknown or duplicate matrix selector")
        else:
            reasons += common.attempt_failures(attempt, closure, backend, workload)
            selected[key] = (attempt, closure)
        checks.append({"requirement": f"{backend} {workload} provenance/commands/diagnostics",
                       "status": "open" if reasons else "passed", "reasons": reasons})
    reasons = matrix_failures(selected)
    checks.append({"requirement": "complete same-source native matrix", "status": "open" if reasons else "passed", "reasons": reasons})
    for backend in common.BACKENDS:
        for workload in common.WORKLOADS:
            item = selected.get((backend, workload))
            reasons = nested_failures(*item, workload) if item else ["selected attempt unavailable"]
            checks.append({"requirement": f"{backend} {workload} named inputs/readback/lifecycle",
                           "status": "open" if reasons else "passed", "reasons": reasons})
    status = "verified" if all(item["status"] == "passed" for item in checks) else "open"
    payload = {"schemaVersion": 1, "scope": "S7 deferred formal evidence only", "checks": checks,
               "evidence_status": status, "stage_closure_decision": "not_made_by_this_tool"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if status == "verified" else 1


if __name__ == "__main__":
    sys.exit(main())
