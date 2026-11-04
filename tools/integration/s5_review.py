#!/usr/bin/env python3
"""Fail-closed reviewer for one explicit S5 Forward evidence closure.

This reviewer only reports whether the six selected runner attempts contain
current, hash-bound evidence.  It deliberately makes no support or closure
decision and treats every absent fact as open evidence.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import struct
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("integration_review", Path(__file__).with_name("review.py"))
assert _SPEC and _SPEC.loader
_REVIEW = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REVIEW)
_SMOKE = _REVIEW._SMOKE
BACKENDS = ("dx12", "native-gl-wgl")


def read_json(path: Path) -> tuple[Any | None, str | None]:
    return _REVIEW.read_json(path)


def pixel_failures(actual: bytes, expected: bytes, mask: bytes, scenario: str = "base") -> list[str]:
    if len(expected) != 96 * 96 * 4 or len(actual) != len(expected) or len(mask) != 96 * 96:
        return [f"invalid RGBA/mask shape: actual={len(actual)}, expected={len(expected)}, mask={len(mask)}"]
    oracle, oracle_mask = fixture_oracle(scenario)
    if expected != oracle or mask != oracle_mask:
        return ["provided Forward expected pixels or stable mask differ from the independent fixture oracle"]
    covered = {expected[index * 4:(index + 1) * 4] for index, item in enumerate(mask) if item}
    if bytes((0, 0, 0, 0)) not in covered or len(covered) < 3:
        return ["stable mask does not cover background and the scenario's visible triangles"]
    return [f"stable pixel differs at {index % 96},{index // 96}"
            for index, item in enumerate(mask) if item
            and actual[index * 4:(index + 1) * 4] != expected[index * 4:(index + 1) * 4]][:1]


@lru_cache(maxsize=7)
def fixture_oracle(scenario: str = "base") -> tuple[bytes, bytes]:
    """Recompute s5_forward.rs's expected_image/stable_mask without runner output."""
    triangles = (((8.64,38.4),(29.76,38.4),(19.2,11.52)), ((34.56,38.4),(55.68,38.4),(45.12,11.52)), ((60.48,38.4),(81.6,38.4),(71.04,11.52)), ((36.48,84.48),(59.52,84.48),(48.,57.6)))
    clear = (0, 0, 0, 0)
    palettes = {
        "base": ((255,0,0,255),(255,0,255,255),(0,0,255,255),(255,255,0,255)),
        "frame-1": (clear,(255,0,255,255),(0,0,255,255),(255,255,0,255)),
        "frame-2": (clear,clear,(0,0,255,255),(255,255,0,255)),
        "new": (clear,(0,255,255,255),(255,255,0,255),(255,255,0,255)),
    }
    colors = palettes["base" if scenario in ("frame-0", "old") else scenario]
    def edge(a, b, p): return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
    def distance(a, b, p):
        dx, dy = b[0]-a[0], b[1]-a[1]; length = dx*dx+dy*dy
        t = max(0., min(1., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length)); x, y = a[0]+t*dx, a[1]+t*dy
        return ((p[0]-x)**2+(p[1]-y)**2)**.5
    pixels = bytearray(96*96*4); mask = bytearray(96*96)
    for y in range(96):
        for x in range(96):
            point = (x+.5, y+.5); offset = (y*96+x)*4
            for triangle, color in zip(triangles, colors):
                signs = [edge(triangle[i], triangle[(i+1)%3], point) for i in range(3)]
                if all(sign >= 0 for sign in signs) or all(sign <= 0 for sign in signs): pixels[offset:offset+4] = bytes(color)
            mask[y*96+x] = int(all(distance(triangle[i], triangle[(i+1)%3], point) > 2.0 for triangle in triangles for i in range(3)))
    return bytes(pixels), bytes(mask)


def retained_failures(attempt: dict[str, Any], directory: Path, files: list[Path]) -> list[str]:
    records: list[dict[str, Any]] = list(attempt.get("retained_attempt_files", []))
    for command in attempt.get("commands", []):
        if isinstance(command, dict) and isinstance(command.get("result"), dict):
            records.extend(command["result"].get("retained_files", []))
    indexed = {(Path(item.get("path", "")).resolve(), item.get("sha256")) for item in records if isinstance(item, dict)}
    failures = []
    for path in files:
        digest = _REVIEW.sha256(path)
        if digest is None:
            failures.append(f"missing raw artifact: {path.relative_to(directory)}")
        elif (path.resolve(), digest) not in indexed:
            failures.append(f"raw artifact is not retained with its hash: {path.relative_to(directory)}")
    return failures


def common_failures(attempt: dict[str, Any], closure: Path, backend: str, workload: str,
                    *, contract: dict[str, Any] | None = None,
                    require_tests: bool = True) -> list[str]:
    failures = _REVIEW.frozen_source_failures(attempt, closure)
    failures += _REVIEW.current_source_failures(attempt, ROOT)
    if contract is None:
        contract = _SMOKE.s5_workload_contract(workload)
    if attempt.get("mode") != "run" or attempt.get("backend") != backend or attempt.get("runner_exit_status") != 0:
        failures.append("selected attempt is not a successful run for its requested backend")
    # Schema version belongs to the smoke manifest; each attempt retains its
    # contract directly in `workload` (not a workload-name string).
    if attempt.get("workload") != contract:
        failures.append("attempt does not retain the exact workload contract")
    if attempt.get("built_source_verification", {}).get("status") != "verified" or attempt.get("built_git_checkout_verification", {}).get("status") != "verified":
        failures.append("built package/source provenance is not verified")
    if attempt.get("source_consistency_after_attempt", {}).get("status") != "stable":
        failures.append("attempt source consistency is not stable")
    if attempt.get("source_consistency_after_build", {}).get("status") != "stable" or attempt.get("build_environment", {}).get("status") != "auditable":
        failures.append("build source/environment closure is not stable and auditable")
    directory = closure / attempt.get("directory", "")
    executable = attempt.get("executable")
    if not isinstance(executable, dict):
        failures.append("retained executable is missing")
    else:
        failures += _REVIEW.hash_records([executable], directory)
        if executable.get("after_run_sha256") != executable.get("sha256"):
            failures.append("executable changed during the attempt")
    locks = attempt.get("locks", [])
    if not locks or any(_REVIEW.sha256(Path(item.get("path", ""))) != item.get("sha256") for item in locks):
        failures.append("lock evidence is missing or changed")
    commands = attempt.get("commands", [])
    if sum(item.get("kind") == "build" for item in commands) != 1:
        failures.append("attempt must retain exactly one build command")
    for command in commands:
        result = command.get("result", {})
        if not command.get("command") or result.get("exit_status") != 0 or result.get("timed_out") is not False or result.get("child_process_lifecycle") != "parent_exited":
            failures.append("command did not finish successfully with a known process lifecycle")
        logs = result.get("raw_logs", [])
        if not logs:
            failures.append("command has no retained raw logs")
        failures += _REVIEW.hash_records(logs + result.get("retained_files", []), directory)
    direct = [command for command in attempt.get("commands", []) if isinstance(command, dict) and command.get("kind") == "direct_executable"]
    if len(direct) != 1 or not all(item.get("result", {}).get("exit_status") == 0 and item.get("result", {}).get("nested_workloads", {}).get("status") == "verified" for item in direct):
        failures.append("attempt lacks a successful direct runner and verified nested workload evidence")
    if direct and contract["entry_argument"] not in direct[0].get("command", []):
        failures.append("direct command does not select the declared workload")
    nested = (
        _SMOKE.s6_nested_workload_evidence(directory / "run", workload)
        if workload.startswith("s6-")
        else _SMOKE.nested_workload_evidence(directory / "run", workload)
    )
    failures += nested.get("failures", [])
    if attempt.get("validation", {}).get("status") != "observed":
        failures.append("native validation evidence is not observed")
    diagnostic = _REVIEW.diagnostics_review(attempt, closure)
    if diagnostic.get("status") != "passed": failures.append("native diagnostics review remains open")
    tests = [command for command in attempt.get("commands", []) if isinstance(command, dict) and command.get("kind") == "test"]
    if require_tests and not any(command.get("result", {}).get("exit_status") == 0 and command.get("result", {}).get("test_command_recognized") and int(command.get("result", {}).get("executed", 0)) > 0 for command in tests):
        failures.append("attempt lacks a successful non-zero targeted test record")
    for command in tests:
        count = _SMOKE.actual_test_count(command.get("command", []), command.get("result", {}))
        if not count or count != command.get("result", {}).get("executed"):
            failures.append("recorded test count does not match the retained command logs")
    return failures


def forward_input_failures(run: Path) -> list[str]:
    """Check the fixed scenario's changes independently of expected/readback files."""
    failures = []
    versions = []
    for number in range(3):
        value, error = read_json(run / f"s5-forward-frame-{number}-input.json")
        if error or not isinstance(value, dict):
            failures.append(f"missing frame {number} inputs")
            continue
        versions.append(value.get("sceneVersion"))
        if value.get("firstMeshClipTranslationZ") != (0 if number == 0 else 2) or value.get("secondMeshTint") != ([0] * 4 if number == 2 else [1] * 4):
            failures.append(f"frame {number} inputs do not match the independently expected change")
    if not _REVIEW.strictly_increasing(versions):
        failures.append("frame snapshots do not have increasing scene versions")
    old, old_error = read_json(run / "s5-forward-replacement-old-input.json")
    new, new_error = read_json(run / "s5-forward-replacement-new-input.json")
    if old_error or new_error or not isinstance(old, dict) or not isinstance(new, dict):
        failures.append("replacement inputs are missing")
    else:
        for kind in ("geometry", "material", "texture"):
            before, after = old.get(kind, {}), new.get(kind, {})
            if (not isinstance(before, dict) or not isinstance(after, dict)
                    or type(before.get("slot")) is not int or before.get("slot") != after.get("slot")
                    or type(before.get("generation")) is not int or type(after.get("generation")) is not int
                    or after["generation"] <= before["generation"]):
                failures.append(f"replacement {kind} does not preserve identity and advance generation")
    return failures


def forward_failures(attempt: dict[str, Any], closure: Path) -> tuple[list[str], Path | None]:
    run = closure / attempt.get("directory", "") / "run" / "s5-forward"
    names = list(_SMOKE.s5_required_raw_artifacts()) + ["s5-forward-workload.json", "s5-forward-external-clear-actual.rgba"]
    files = [run / name for name in names]
    failures = retained_failures(attempt, closure, files)
    failures += forward_input_failures(run)
    workload, error = read_json(run / "s5-forward-workload.json")
    oracle, oracle_error = read_json(run / "s5-forward-oracle.json")
    if error or not isinstance(workload, dict) or workload.get("workload") != "s5-scene-forward-four-layouts":
        failures.append("Forward nested workload identity is missing or invalid")
    if oracle_error or not isinstance(oracle, dict) or oracle.get("mask") != "s5-forward-stable-mask.bin" or oracle.get("expected") != "s5-forward-expected.rgba" or oracle.get("actual") != "s5-forward-readback.rgba":
        failures.append("Forward independent oracle declaration is missing or invalid")
    try:
        expected = (run / "s5-forward-expected.rgba").read_bytes(); mask = (run / "s5-forward-stable-mask.bin").read_bytes()
        failures += pixel_failures((run / "s5-forward-readback.rgba").read_bytes(), expected, mask)
        for number in range(3):
            failures += pixel_failures((run / f"s5-forward-frame-{number}-actual.rgba").read_bytes(), (run / f"s5-forward-frame-{number}-expected.rgba").read_bytes(), mask, f"frame-{number}")
        for generation in ("old", "new"):
            failures += pixel_failures((run / f"s5-forward-replacement-{generation}-actual.rgba").read_bytes(), (run / f"s5-forward-replacement-{generation}-expected.rgba").read_bytes(), mask, generation)
        if any((run / name).read_bytes() != bytes(96 * 96 * 4) for name in ("s5-forward-empty-actual.rgba", "s5-forward-empty-expected.rgba", "s5-forward-external-clear-actual.rgba")):
            failures.append("Forward empty or external clear readback is not transparent black")
        required_colors = {bytes(value) for value in ([255,0,0,255], [255,0,255,255], [0,0,255,255], [255,255,0,255])}
        if not required_colors <= {expected[index * 4:(index + 1) * 4] for index, item in enumerate(mask) if item}:
            failures.append("Forward expected pixels do not prove the four shader/material outputs")
    except OSError as error:
        failures.append(f"cannot read Forward pixel evidence: {error}")
    facts, error = read_json(run / "s5-forward-cache-facts.json")
    required_facts = {"residentCopyDelta": 0, "geometryHitDelta": 10, "textureHitDelta": 10,
                      "frameReuseDelta": 1, "opaqueCandidateDelta": 10, "opaqueBatchDelta": 10,
                      "opaqueDrawDelta": 10, "pending": 3}
    if error or not isinstance(facts, dict) or any(facts.get(key) != value for key, value in required_facts.items()) or facts.get("freshStateCompilations") != facts.get("warmPlanCompilations", -1) + 1 or not isinstance(facts.get("finalCompilations"), int):
        failures.append("Forward cache facts are missing or malformed")
    metrics, error = read_json(run / "s5-forward-measurements.json")
    fields = ("preparationElapsedNs", "graphCompileElapsedNs", "acceptedBufferAllocations",
              "acceptedBufferAllocationBytes", "acceptedTextureAllocations", "residentStaticUploadBytes",
              "frameUploadBytes", "geometryCacheHits", "textureCacheHits", "frameAllocationReuses", "planCompilations")
    if (error or not isinstance(metrics, dict) or metrics.get("schemaVersion") != 1
            or metrics.get("pendingPeak") != 3 or not metrics.get("preparationBoundary")
            or any(type(metrics.get(key)) is not int or metrics[key] <= 0 for key in fields)):
        failures.append("Forward measurement facts do not cover the completed workload")
    return failures, run


def device_failures(attempt: dict[str, Any], closure: Path, forward_run: Path | None) -> list[str]:
    run = closure / attempt.get("directory", "") / "run" / "s5-forward-device-isolation"
    files = [run / name for name in list(_SMOKE.s5_device_isolation_required_raw_artifacts()) + ["s5-device-isolation-workload.json"]]
    failures = retained_failures(attempt, closure, files)
    facts, error = read_json(run / "s5-device-isolation.json")
    if error or not isinstance(facts, dict) or facts.get("oldDeviceId") == facts.get("newDeviceId") or facts.get("sharedLogicalAssetGenerations") is not True or facts.get("oldCompletedProductsRetained") is not True or facts.get("newDeviceResidentCopyBytes", 0) <= 0 or facts.get("oldReuseResidentCopyBytes") != 0:
        failures.append("two-device cache/replacement facts are missing or inconsistent")
    if not isinstance(facts, dict) or any(facts.get(key) != expected for key, expected in {
        "newDeviceGeometryHits": 0, "newDeviceTextureHits": 0,
        "oldReuseGeometryHits": 4, "oldReuseTextureHits": 4,
    }.items()):
        failures.append("two-device facts do not prove fresh realization and retained old cache hits")
    if forward_run is None:
        failures.append("cannot independently assess device pixels without the paired Forward stable mask")
    else:
        try:
            mask = (forward_run / "s5-forward-stable-mask.bin").read_bytes(); expected = (run / "s5-device-isolation-expected.rgba").read_bytes()
            for name in ("s5-device-isolation-old-actual.rgba", "s5-device-isolation-new-actual.rgba", "s5-device-isolation-old-reuse-actual.rgba"):
                failures += pixel_failures((run / name).read_bytes(), expected, mask)
        except OSError as error:
            failures.append(f"cannot read device-isolation pixels: {error}")
    return failures


def surface_failures(attempt: dict[str, Any], closure: Path, human: dict[str, Any] | None, human_base: Path, identifier: str) -> list[str]:
    run = closure / attempt.get("directory", "") / "run" / "s5-forward-surface"
    files = [run / name for name in list(_SMOKE.s5_surface_required_raw_artifacts()) + ["s5-forward-surface-workload.json"]]
    failures = retained_failures(attempt, closure, files)
    workload, error = read_json(run / "s5-forward-surface-workload.json")
    if error or not isinstance(workload, dict) or workload.get("hostResizeReconfiguredBeforeAcquire") is not True:
        failures.append("surface workload lacks pre-acquire resize/reconfigure evidence")
    failures += human_link_failures(human, human_base, identifier, attempt)
    for frame, extent in enumerate(((320, 180), (256, 144))):
        capture, error = read_json(run / f"surface-client-frame-{frame}.capture.json")
        if error or not isinstance(capture, dict) or capture.get("clientExtent") != list(extent) or capture.get("automaticVisualPass") is not False:
            failures.append(f"surface capture metadata for frame {frame} is incomplete")
        try:
            data = (run / f"surface-client-frame-{frame}.bmp").read_bytes()
            if len(data) != 54 + extent[0] * extent[1] * 4 or data[:2] != b"BM" or struct.unpack_from("<ii", data, 18) != (extent[0], -extent[1]):
                failures.append(f"surface BMP for frame {frame} has invalid native pixels/header")
            else:
                at = lambda x, y: data[54 + (y * extent[0] + x) * 4:58 + (y * extent[0] + x) * 4]
                corners = ((0, 0), (extent[0]-1, 0), (0, extent[1]-1), (extent[0]-1, extent[1]-1))
                if any(at(x, y)[:3] != b"\0\0\0" for x, y in corners):
                    failures.append(f"surface BMP for frame {frame} lacks black corner pixels")
                x, y = extent[0] * 8 // 20, extent[1] * 57 // 100
                if at(x, y)[:3] != bytes((0, 0, 255)):
                    failures.append(f"surface BMP for frame {frame} lacks the independent red triangle interior")
        except (OSError, struct.error) as error:
            failures.append(f"cannot read surface BMP {frame}: {error}")
    return failures


def human_link_failures(human: dict[str, Any] | None, base: Path, identifier: str, attempt: dict[str, Any]) -> list[str]:
    """Require a passed, hash-bound human verdict for each native frame."""
    if not human or human.get("status") != "reviewed" or identifier not in human.get("attempt_ids", []):
        return ["surface attempt has no reviewed human-review index entry"]
    records: set[tuple[Path, str]] = set()
    for command in attempt.get("commands", []):
        if isinstance(command, dict):
            records.update((Path(item.get("path", "")).resolve(), item.get("sha256", "")) for item in command.get("result", {}).get("retained_files", []) if isinstance(item, dict))
    entries = [item for item in human.get("evidence", []) if isinstance(item, dict) and item.get("attempt_id") == identifier]
    if len(entries) != 1 or not isinstance(entries[0].get("frames"), list):
        return ["human review has no single typed frame set for surface attempt"]
    required = {f"surface-client-frame-{frame}.bmp" for frame in range(2)}; seen: set[str] = set(); failures = []
    for frame in entries[0]["frames"]:
        if not isinstance(frame, dict) or frame.get("human_verdict") != "passed":
            failures.append("human review frame has no explicit passed verdict"); continue
        linked: list[Path] = []
        for key in ("screenshot", "capture", "verdict"):
            value = frame.get(key)
            if not isinstance(value, dict) or not isinstance(value.get("path"), str) or not isinstance(value.get("sha256"), str):
                failures.append("human review frame lacks a typed hash record"); break
            path = Path(value["path"]); path = path if path.is_absolute() else base / path
            if (path.resolve(), value["sha256"]) not in records or _REVIEW.sha256(path) != value["sha256"]:
                failures.append("human review record is not the retained attempt artifact"); break
            linked.append(path)
        if len(linked) == 3:
            shot, capture, verdict = linked
            capture_value, capture_error = read_json(capture); verdict_value, verdict_error = read_json(verdict)
            if capture_error or verdict_error or not isinstance(capture_value, dict) or not isinstance(verdict_value, dict) or capture_value.get("image") != shot.name or verdict_value.get("image") != shot.name or verdict_value.get("status") != "passed":
                failures.append("human review links inconsistent capture/verdict evidence")
            seen.add(shot.name)
    if seen != required: failures.append("human review does not bind both expected surface screenshots")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-review", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt", nargs=4, action="append", metavar=("CLOSURE", "ID", "BACKEND", "WORKLOAD"), required=True,
                        help="repeat for each selected manifest closure, id, backend, and S5 workload")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks: list[dict[str, Any]] = []
    human, human_error = read_json(args.human_review); human = human if isinstance(human, dict) else None
    selected: dict[tuple[str, str], tuple[dict[str, Any], Path, str]] = {}
    for closure_text, identifier, backend, workload in args.attempt:
        closure = Path(closure_text); manifest, error = read_json(closure / "manifest.json")
        attempt, reasons = (None, [error or "manifest is not an object"]) if not isinstance(manifest, dict) else _REVIEW.locate_attempt(manifest, identifier, backend)
        if isinstance(manifest, dict) and (manifest.get("schema_version") != 4 or manifest.get("stage") != "S5 Forward integration evidence"):
            reasons.append("manifest is not the S5 schema-v4 evidence ledger")
        valid_selection = backend in BACKENDS and workload in ("s5-forward", "s5-device-isolation", "s5-surface")
        if not valid_selection: reasons.append("unknown S5 backend/workload selector")
        if (backend, workload) in selected: reasons.append("duplicate backend/workload selection")
        if attempt and valid_selection:
            reasons += common_failures(attempt, closure, backend, workload)
            selected[(backend, workload)] = (attempt, closure, identifier)
        checks.append({"requirement":f"{backend} {workload} attempt", "status":"passed" if not reasons else "open", "reasons":reasons})
    for backend in BACKENDS:
        forward_item = selected.get((backend, "s5-forward")); device_item = selected.get((backend, "s5-device-isolation")); surface_item = selected.get((backend, "s5-surface"))
        forward, run = forward_failures(forward_item[0], forward_item[1]) if forward_item else (["Forward attempt unavailable"], None)
        checks.append({"requirement":f"{backend} Forward raw/oracle/cache", "status":"passed" if not forward else "open", "reasons":forward})
        device = device_failures(device_item[0], device_item[1], run) if device_item else ["device attempt unavailable"]
        checks.append({"requirement":f"{backend} device isolation", "status":"passed" if not device else "open", "reasons":device})
        surface = surface_failures(surface_item[0], surface_item[1], human, args.human_review.parent, surface_item[2]) if surface_item else ["surface attempt unavailable"]
        checks.append({"requirement":f"{backend} surface capture/human review", "status":"passed" if not surface else "open", "reasons":surface})
    payload = {"schema_version": 1, "scope": "S5 Forward evidence only", "human_review_error": human_error, "checks": checks, "evidence_status": "verified" if checks and all(item["status"] == "passed" for item in checks) else "open", "stage_closure_decision": "not_made_by_this_tool"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if payload["evidence_status"] == "verified" else 1


if __name__ == "__main__":
    sys.exit(main())
