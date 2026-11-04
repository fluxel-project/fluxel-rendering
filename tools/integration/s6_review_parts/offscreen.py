"""Independent S6 offscreen typed-Forward evidence review."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import common, oracle


CASES = tuple(oracle.FRAME_CASES)
_INITIAL = CASES[:-2]
_REPLACED = ("pending-new", "pending-new-warm")
_FEATURE_ENABLED = {case: case != "toggle-disabled" for case in CASES}
_FEATURE_VISIBLE = {case: case not in ("toggle-disabled", "provider-mesh-only") for case in CASES}
_MESH_VISIBLE = {case: case != "provider-marker-only" for case in CASES}


def _object(path: Path, label: str, failures: list[str]) -> dict[str, Any]:
    value, errors = common.read_object(path)
    if errors:
        failures.append(f"{label}: {errors[0]}")
    return value


def _input_failures(run: Path) -> list[str]:
    failures: list[str] = []
    mesh_assets: dict[str, dict[str, Any]] = {}
    quad_assets: dict[str, dict[str, Any]] = {}
    for case in CASES:
        value = _object(run / f"s6-{case}-input.json", f"{case} input", failures)
        if not value:
            continue
        if (value.get("case") != case or type(value.get("sceneVersion")) is not int
                or value.get("typedVersion") != value.get("sceneVersion")):
            failures.append(f"{case}: input does not retain matching case, snapshot, and typed versions")
        for field, actual in (
            ("meshVisible", _MESH_VISIBLE[case]),
            ("featureEnabled", _FEATURE_ENABLED[case]),
            ("featureVisible", _FEATURE_VISIBLE[case]),
        ):
            if value.get(field) is not actual:
                failures.append(f"{case}: input {field} does not match the independent scenario")
        if value.get("featureOrder") != (-10 if case == "frame-0" else 10):
            failures.append(f"{case}: input feature order is not the declared ordering scenario")
        marker = value.get("marker")
        expected_center = [2.0, 0.0, 0.0] if case == "provider-mesh-only" else [0.0, 0.0, 0.0]
        if not isinstance(marker, dict) or marker.get("id") != 7 or marker.get("center") != expected_center or marker.get("halfExtent") != [0.55, 0.55] or marker.get("tint") != [1.0, 0.0, 0.0, 1.0]:
            failures.append(f"{case}: marker center, half extent, or tint differs from the independent input")
        mesh = value.get("mesh")
        expected_bounds = ([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]) if case == "provider-marker-only" else ([-0.5, -0.5, 0.0], [0.5, 0.5, 0.0])
        expected_transform = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [2.0 if case == "provider-marker-only" else 0.0, 0.0, 0.0, 1.0]]
        if not isinstance(mesh, dict) or mesh.get("bounds") != {"minimum": expected_bounds[0], "maximum": expected_bounds[1]} or mesh.get("transformColumns") != expected_transform:
            failures.append(f"{case}: mesh bounds or transform differs from the independent input")
        assets = value.get("assets")
        if not isinstance(assets, dict):
            failures.append(f"{case}: input lacks feature and mesh asset identities")
            continue
        mesh = assets.get("mesh")
        quad = assets.get("quad")
        if not isinstance(mesh, dict) or not isinstance(quad, dict):
            failures.append(f"{case}: input lacks mesh or quad asset identities")
            continue
        for owner, asset_set in (("mesh", mesh), ("quad", quad)):
            for kind in ("geometry", "material", "texture"):
                identity = asset_set.get(kind)
                if (not isinstance(identity, dict) or type(identity.get("slot")) is not int
                        or type(identity.get("slotGeneration")) is not int
                        or type(identity.get("contentGeneration")) is not int):
                    failures.append(f"{case}: {owner} {kind} three-part identity is malformed")
        mesh_assets[case] = mesh
        quad_assets[case] = quad
    if all(case in quad_assets for case in CASES):
        old = quad_assets["pending-old"]
        baseline_mesh = mesh_assets.get("pending-old")
        for case in _INITIAL:
            if quad_assets[case] != old:
                failures.append(f"{case}: initial quad asset identity/generation drifted before replacement")
            if mesh_assets.get(case) != baseline_mesh:
                failures.append(f"{case}: built-in mesh asset identity/generation drifted")
        for case in _REPLACED:
            for kind in ("geometry", "material", "texture"):
                before, after = old.get(kind), quad_assets[case].get(kind)
                if (not isinstance(before, dict) or not isinstance(after, dict)
                        or before.get("slot") != after.get("slot")
                        or before.get("slotGeneration") != after.get("slotGeneration")
                        or type(before.get("contentGeneration")) is not int
                        or type(after.get("contentGeneration")) is not int
                        or after["contentGeneration"] <= before["contentGeneration"]):
                    failures.append(f"{case}: replacement {kind} did not preserve identity and advance generation")
    return failures


def _pixel_failures(run: Path) -> list[str]:
    failures: list[str] = []
    try:
        mask = (run / "s6-stable-mask.bin").read_bytes()
    except OSError as error:
        return [f"cannot read independent stable mask: {error}"]
    for case in CASES:
        try:
            expected = (run / f"s6-{case}-expected.rgba").read_bytes()
            actual = (run / f"s6-{case}-actual.rgba").read_bytes()
        except OSError as error:
            failures.append(f"{case}: cannot read expected/readback pixels: {error}")
            continue
        failures += oracle.pixel_failures(case, actual, expected, mask)
        failures += common.preview_failures(run / f"s6-{case}-expected.ppm", expected)
        failures += common.preview_failures(run / f"s6-{case}-actual.ppm", actual)
    return failures


def _pending_failures(run: Path) -> list[str]:
    failures: list[str] = []
    assets = _object(run / "s6-pending-assets.json", "pending assets", failures)
    lifecycle = _object(run / "s6-pending-lifecycle.json", "pending lifecycle", failures)
    old_assets, new_assets = assets.get("old"), assets.get("new")
    for kind in ("geometry", "material", "texture"):
        old = old_assets.get(kind) if isinstance(old_assets, dict) else None
        new = new_assets.get(kind) if isinstance(new_assets, dict) else None
        if not isinstance(old, dict) or not isinstance(new, dict) or old.get("id") != new.get("id") or type(old.get("generation")) is not int or type(new.get("generation")) is not int or new["generation"] <= old["generation"]:
            failures.append(f"pending {kind} does not preserve identity with a newer generation")
    expected_retirements = [
        {"case": "pending-old", "pendingAfter": 2},
        {"case": "pending-new", "pendingAfter": 1},
        {"case": "pending-new-warm", "pendingAfter": 0},
    ]
    if (lifecycle.get("schemaVersion") != 1 or lifecycle.get("pendingAfterThree") != 3
            or lifecycle.get("fourthSubmission") != "Backpressure"
            or lifecycle.get("callbacksUnchangedOnBackpressure") is not True
            or lifecycle.get("retirements") != expected_retirements):
        failures.append("pending lifecycle does not prove three owners, fourth backpressure, and ordered retirement")
    try:
        if (run / "s6-pending-new-quad-geometry-stream-0.bin").read_bytes() != oracle.geometry_bytes("quad", 0.5):
            failures.append("replacement quad geometry does not retain the independently scaled footprint")
        if (run / "s6-pending-new-quad-texture.rgba").read_bytes() != bytes((255, 255, 0, 255)):
            failures.append("replacement quad texture does not retain the independent yellow texel")
        material = (run / "s6-pending-new-quad-material.txt").read_text(encoding="utf-8")
        if not all(token in material for token in ("s6_replacement_quad", "tint.rgb * 0.5", "vec3<f32>(0.25)")):
            failures.append("replacement quad material does not retain the independent replacement shader")
    except OSError as error:
        failures.append(f"cannot read replacement quad raw assets: {error}")
    return failures


def _warm_failures(run: Path) -> list[str]:
    failures: list[str] = []
    facts = _object(run / "s6-composed-cache.json", "warm cache", failures)
    pairs = (
        ("warmGeometryHitsBefore", "warmGeometryHitsAfter"),
        ("warmTextureHitsBefore", "warmTextureHitsAfter"),
        ("warmPrepareCallbacksBefore", "warmPrepareCallbacksAfter"),
        ("warmContributeCallbacksBefore", "warmContributeCallbacksAfter"),
    )
    for before, after in pairs:
        if type(facts.get(before)) is not int or type(facts.get(after)) is not int or facts[after] <= facts[before]:
            failures.append(f"warm cache lacks increasing {after}")
    before_plan = facts.get("warmPlanCompilationsBefore")
    after_plan = facts.get("warmPlanCompilationsAfter")
    if type(before_plan) is not int or type(after_plan) is not int or before_plan != after_plan:
        failures.append("warm cache compiled another Graph plan")
    return failures


def _measurement_failures(run: Path) -> list[str]:
    failures: list[str] = []
    facts = _object(run / "s6-composed-measurements.json", "S6 measurements", failures)
    nonnegative = (
        "preparationElapsedNs", "graphCompileElapsedNs", "geometryHits", "textureHits",
        "frameReuses",
    )
    positive = (
        "acceptedBufferAllocations", "acceptedBufferAllocationBytes",
        "acceptedTextureAllocations", "residentCopyBytes", "frameUploadBytes",
        "opaqueCandidates", "opaqueBatches", "opaqueDraws", "compilations",
    )
    if (facts.get("schemaVersion") != 1 or not isinstance(facts.get("preparationBoundary"), str)
            or not facts.get("preparationBoundary") or facts.get("pendingPeak") != 3
            or facts.get("acceptedSubmissions") != len(CASES)
            or any(type(facts.get(field)) is not int or facts[field] < 0 for field in nonnegative)
            or any(type(facts.get(field)) is not int or facts[field] <= 0 for field in positive)
            or facts.get("geometryHits", 0) <= 0 or facts.get("textureHits", 0) <= 0):
        failures.append("S6 measurements do not cover the eleven completed offscreen submissions")
    return failures


def review(run: Path) -> list[str]:
    """Return every S6 offscreen evidence failure without making support claims."""
    failures = common.initial_asset_failures(run)
    workload = _object(run / "s6-composed-workload.json", "S6 workload", failures)
    if workload.get("schemaVersion") != 1 or workload.get("workload") != "s6-composed-offscreen-typed-quad":
        failures.append("S6 offscreen workload identity is missing or invalid")
    cases = workload.get("cases")
    if not isinstance(cases, list) or sorted(
        item.get("name") for item in cases if isinstance(item, dict)
    ) != sorted(("original-three-frame-cache", "enabled-disabled-enabled", "independent-providers", "three-pending-replacements")):
        failures.append("S6 workload does not enumerate the four required offscreen case groups")
    failures += _input_failures(run)
    failures += _pixel_failures(run)
    failures += _pending_failures(run)
    failures += _warm_failures(run)
    failures += _measurement_failures(run)
    return failures
