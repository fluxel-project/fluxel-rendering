#!/usr/bin/env python3
"""Review the selected S6 native matrix without declaring stage completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import s5_review as base
from smoke_s6 import S6_SCHEMA_VERSION, S6_STAGE, s6_workload_contract
from s6_review_parts import common, offscreen, surface
from s6_review_parts.oracle import FRAME_CASES


BACKENDS = ("dx12", "native-gl-wgl")
WORKLOADS = ("s6-composed", "s6-surface")
Selection = tuple[dict[str, Any], Path, str]


def targeted_test_groups(command: list[str]) -> set[str]:
    """Identify the bounded consumers tested; log counts are checked separately."""
    parts = [part.replace("\\", "/") for part in command]
    joined = " ".join(parts)
    groups = set()
    if "test" in parts and any("cargo" in Path(part).stem for part in parts[:1]):
        if ("fluxel-renderer" in parts or "../../Cargo.toml" in parts) and any(
            part in ("forward", "forward_composed_tests", "forward_composed")
            for part in parts
        ):
            groups.add("renderer-forward")
        if "external-renderer/Cargo.toml" in joined:
            groups.add("external-consumer")
        if "cli" in joined and ("rendering-integration" in joined or "Cargo.toml" in parts):
            groups.add("consumer-cli")
    if "unittest" in parts and ("test_s6" in joined or "test_*.py" in joined):
        groups.add("evidence-tools")
    return groups


def matrix_failures(selected: dict[tuple[str, str], Selection]) -> list[str]:
    failures = []
    if set(selected) != {(backend, workload) for backend in BACKENDS for workload in WORKLOADS}:
        failures.append("S6 requires both offscreen and Surface workloads on DX12 and WGL")
    source_sets = []
    identifiers = []
    test_groups = set()
    for attempt, _closure, identifier in selected.values():
        identifiers.append(identifier)
        inputs = attempt.get("source_inputs", [])
        source_sets.append(sorted((item.get("path"), item.get("sha256")) for item in inputs))
        for command in attempt.get("commands", []):
            result = command.get("result", {})
            if (command.get("kind") == "test" and result.get("exit_status") == 0
                    and result.get("test_command_recognized") is True
                    and type(result.get("executed")) is int and result["executed"] > 0):
                test_groups.update(targeted_test_groups(command.get("command", [])))
    if len(identifiers) != len(set(identifiers)):
        failures.append("selected native workloads must have unique attempt identities")
    if not source_sets or not source_sets[0] or any(item != source_sets[0] for item in source_sets):
        failures.append("selected native attempts do not share one exact nonempty source closure")
    required = {"renderer-forward", "external-consumer", "consumer-cli", "evidence-tools"}
    missing = required - test_groups
    if missing:
        failures.append("same-source targeted test records missing: " + ", ".join(sorted(missing)))
    return failures


def workload_failures(item: Selection, workload: str, visual: dict[str, Any],
                      visual_base: Path) -> list[str]:
    attempt, closure, identifier = item
    directory = closure / attempt["directory"]
    run = directory / "run" / workload
    files = [path for path in run.rglob("*") if path.is_file()]
    failures = base.retained_failures(attempt, closure, files)
    if workload == "s6-composed":
        failures += offscreen.review(run)
        failures += common.visual_offscreen_failures(
            visual, visual_base, identifier, attempt, closure, run, list(FRAME_CASES))
    else:
        failures += surface.review(run)
        failures += common.visual_surface_failures(
            visual, visual_base, identifier, attempt, closure, run)
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt", nargs=4, action="append", required=True,
                        metavar=("CLOSURE", "ID", "BACKEND", "WORKLOAD"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = []
    selected: dict[tuple[str, str], Selection] = {}

    def check(requirement: str, reasons: list[str]) -> None:
        checks.append({"requirement": requirement, "status": "open" if reasons else "passed",
                       "reasons": reasons})

    visual, error = common.read_object(args.visual_review)
    if not isinstance(visual.get("reviewer"), str) or not visual.get("reviewer", "").strip():
        error.append("visual review must identify the person or agent that inspected the images")
    check("visual review identity", error)
    for closure_text, identifier, backend, workload in args.attempt:
        closure = Path(closure_text).resolve()
        manifest, reasons = common.read_object(closure / "manifest.json")
        if manifest.get("schema_version") != S6_SCHEMA_VERSION or manifest.get("stage") != S6_STAGE:
            reasons.append("manifest is not the S6 schema-v5 evidence ledger")
        attempt, lookup_failures = base._REVIEW.locate_attempt(manifest, identifier, backend)
        reasons += lookup_failures
        valid = backend in BACKENDS and workload in WORKLOADS
        if not valid:
            reasons.append("unknown S6 backend/workload selector")
        if (backend, workload) in selected:
            reasons.append("duplicate backend/workload selection")
            valid = False
        if attempt and valid:
            directory = closure / attempt.get("directory", "")
            if not directory.resolve().is_relative_to(closure):
                reasons.append("attempt directory escapes its closure")
            else:
                reasons += base.common_failures(
                    attempt, closure, backend, workload,
                    contract=s6_workload_contract(workload), require_tests=False)
                selected[(backend, workload)] = (attempt, closure, identifier)
        check(f"{backend} {workload} source/commands/diagnostics", reasons)
    check("complete same-source native matrix and targeted tests", matrix_failures(selected))
    for backend in BACKENDS:
        for workload in WORKLOADS:
            item = selected.get((backend, workload))
            reasons = (workload_failures(item, workload, visual, args.visual_review.parent)
                       if item else ["selected attempt unavailable"])
            check(f"{backend} {workload} inputs/output/lifecycle/visual", reasons)
    payload = {
        "schema_version": 1,
        "scope": "S6 typed Forward native evidence only",
        "checks": checks,
        "evidence_status": "verified" if all(item["status"] == "passed" for item in checks) else "open",
        "stage_closure_decision": "not_made_by_this_tool",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if payload["evidence_status"] == "verified" else 1


if __name__ == "__main__":
    sys.exit(main())
