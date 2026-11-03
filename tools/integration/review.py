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


ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ("dx12", "native-gl-wgl")
_SMOKE_SPEC = importlib.util.spec_from_file_location("integration_smoke", Path(__file__).with_name("smoke.py"))
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


def fixed_raster_oracle(source: bytes) -> tuple[list[int] | None, list[str]]:
    """Derive the fixture's stable-pixel oracle from its recorded vertex bytes."""
    reasons: list[str] = []
    if len(source) != 72:
        return None, ["raster vertex input is not exactly three 24-byte vertices"]
    values = struct.unpack("<18f", source)
    if not all(math.isfinite(value) for value in values):
        return None, ["raster vertex input contains non-finite values"]
    positions = [(values[index * 6], values[index * 6 + 1]) for index in range(3)]
    fixture_positions = [(-0.75, -0.65), (0.55, -0.50), (-0.20, 0.75)]
    if any(abs(actual - expected) > 0.000_001
           for position, fixture in zip(positions, fixture_positions)
           for actual, expected in zip(position, fixture)):
        reasons.append("raster vertex positions differ from the fixed 128x128 fixture")
    colors = [values[index * 6 + 2:index * 6 + 6] for index in range(3)]
    if any(component < 0.0 or component > 1.0 for color in colors for component in color):
        reasons.append("raster vertex color is outside unorm range")
    if any(any(abs(component - colors[0][channel]) > 0.000_001
               for channel, component in enumerate(color)) for color in colors[1:]):
        reasons.append("raster fixture does not use one solid per-generation color")
    color = [round(component * 255.0) for component in colors[0]]
    return color if not reasons else None, reasons


def dense_raster_failures(pixels: bytes, color: list[int], label: str) -> list[str]:
    reasons: list[str] = []
    if len(pixels) != 128 * 128 * 4:
        reasons.append(f"{label} readback is not a 128x128 RGBA image")
        return reasons
    checked = 0
    vertices = [(16.0, 105.6), (99.2, 96.0), (51.2, 16.0)]
    for y in range(128):
        for x in range(128):
            point = (x + 0.5, y + 0.5)
            signs: list[float] = []
            edge_band = False
            for edge in range(3):
                a, b = vertices[edge], vertices[(edge + 1) % 3]
                dx, dy = b[0] - a[0], b[1] - a[1]
                sign = dx * (point[1] - a[1]) - dy * (point[0] - a[0])
                signs.append(sign)
                if abs(sign) / math.sqrt(dx * dx + dy * dy) <= 2.0:
                    edge_band = True
            if edge_band:
                continue
            inside = all(sign > 0.0 for sign in signs) or all(sign < 0.0 for sign in signs)
            offset = (y * 128 + x) * 4
            expected = bytes(color if inside else [0, 0, 0, 0])
            if pixels[offset:offset + 4] != expected:
                reasons.append(f"{label} differs at stable pixel ({x}, {y})")
                break
            checked += 1
        else:
            continue
        break
    if checked < 1000:
        reasons.append(f"{label} did not produce enough independent stable samples")
    return reasons


def raster_failures(run: Path, value: Any) -> list[str]:
    reasons: list[str] = []
    if not isinstance(value, dict):
        return ["raster workload is missing"]
    if (value.get("extent") != [128, 128] or value.get("vertexStride") != 24
            or value.get("vertexCount") != 3 or value.get("clearRgba") != [0, 0, 0, 0]
            or value.get("snapshotRetentionVerified") is not True
            or value.get("perFrameRetirementVerified") is not True
            or value.get("planCompilations") != 1):
        reasons.append("raster workload metadata does not match the fixed fixture contract")
    generations = value.get("generations")
    if not isinstance(generations, list) or len(generations) != 3:
        return reasons + ["raster workload must name exactly three generations"]
    generation_ids = [item.get("generation") if isinstance(item, dict) else None for item in generations]
    if not strictly_increasing(generation_ids):
        reasons.append("raster generation identifiers are not three increasing integers")
    for index, generation in enumerate(generations):
        source = run / "raster" / f"raster-input-{index}.bin"
        raw = run / "raster" / f"raster-readback-{index}.rgba"
        if not source.is_file() or not raw.is_file():
            reasons.append(f"missing raster raw input or readback for generation {index}")
            continue
        color, input_reasons = fixed_raster_oracle(source.read_bytes())
        reasons.extend(f"raster generation {index}: {reason}" for reason in input_reasons)
        if color is None:
            continue
        if not isinstance(generation, dict) or generation.get("rgba") != color:
            reasons.append(f"raster generation {index} metadata color does not match vertex input")
        reasons.extend(dense_raster_failures(raw.read_bytes(), color, f"raster generation {index}"))
    return reasons


def read_set_failures(run: Path, directory: str = "read-set",
                      workload: str = "s1-private-read-set-raster-three-generations",
                      explicit_expected: list[list[int]] | None = None,
                      texture_permutation: tuple[int, int, int, int] = (0, 1, 2, 3)) -> list[str]:
    root = run / directory
    path = root / "read-set-workload.json"
    value, error = read_json(path)
    if error or not isinstance(value, dict):
        return ["missing or invalid read-set workload"]
    reasons: list[str] = []
    if (value.get("workload") != workload
            or value.get("extent") != [8, 8] or not strictly_increasing(value.get("assetGenerations"))
            or value.get("uniformDestinationRanges") != [[0, 256], [256, 512], [512, 768]]
            or value.get("sourceByteLengths") != [20, 20, 20]
            or value.get("planReuseVerified") is not True
            or value.get("snapshotRetentionVerified") is not True
            or value.get("perFrameRetirementVerified") is not True):
        reasons.append("read-set workload metadata does not match the three-generation contract")
    for index in range(3):
        source = root / f"read-set-input-{index}.bin"
        expected = root / f"read-set-expected-{index}.rgba"
        readback = root / f"read-set-readback-{index}.rgba"
        if not source.is_file() or not expected.is_file() or not readback.is_file():
            reasons.append(f"missing read-set raw artifact for generation {index}")
            continue
        data = source.read_bytes()
        if len(data) != 20:
            reasons.append(f"read-set input {index} is not exactly 20 bytes")
            continue
        tint = struct.unpack("<4f", data[:16])
        if not all(math.isfinite(component) and 0.0 <= component <= 1.0 for component in tint):
            reasons.append(f"read-set input {index} has invalid tint floats")
            continue
        texel = bytes(data[16 + component] for component in texture_permutation)
        color = bytes(round(component * channel) for component, channel in zip(tint, texel))
        if explicit_expected is not None:
            if index >= len(explicit_expected) or explicit_expected[index] != list(color):
                reasons.append(f"read-set explicit expected color {index} disagrees with its raw material input")
                continue
            image = bytes(explicit_expected[index]) * 64
        else:
            image = color * 64
        if expected.read_bytes() != image:
            reasons.append(f"read-set expected image {index} is not derived from its raw input")
        if readback.read_bytes() != image:
            reasons.append(f"read-set readback {index} is not derived from its raw input")
    return reasons


def stable_raster_mask_failures(mask: bytes, pixels: bytes, color: list[int], label: str) -> list[str]:
    """Validate the retained edge-independent mask before trusting its selected pixels."""
    reasons: list[str] = []
    if len(mask) != 128 * 128:
        return [f"{label} oracle mask is not 128x128"]
    if len(pixels) != 128 * 128 * 4:
        return [f"{label} RGBA image is not 128x128"]
    vertices = [(16.0, 105.6), (99.2, 96.0), (51.2, 16.0)]
    checked = 0
    for y in range(128):
        for x in range(128):
            point = (x + 0.5, y + 0.5)
            signs: list[float] = []
            stable = True
            for edge in range(3):
                a, b = vertices[edge], vertices[(edge + 1) % 3]
                dx, dy = b[0] - a[0], b[1] - a[1]
                sign = dx * (point[1] - a[1]) - dy * (point[0] - a[0])
                signs.append(sign)
                if abs(sign) / math.sqrt(dx * dx + dy * dy) <= 2.0:
                    stable = False
            included = mask[y * 128 + x]
            if included not in (0, 1) or bool(included) != stable:
                reasons.append(f"{label} oracle mask disagrees with the fixed edge-distance rule at ({x}, {y})")
                return reasons
            if not stable:
                continue
            expected = bytes(color if (all(sign > 0.0 for sign in signs) or all(sign < 0.0 for sign in signs)) else [0, 0, 0, 0])
            offset = (y * 128 + x) * 4
            if pixels[offset:offset + 4] != expected:
                reasons.append(f"{label} differs at masked stable pixel ({x}, {y})")
                return reasons
            checked += 1
    if checked < 1000:
        reasons.append(f"{label} oracle mask has too few stable pixels")
    return reasons


def shader_route_failures(root: Path, prefix: str) -> list[str]:
    value, error = read_json(root / f"{prefix}-route.json")
    if error or not isinstance(value, dict):
        return [f"{prefix} route evidence is missing or malformed"]
    if value.get("route") not in ("direct", "naga"):
        return [f"{prefix} did not retain a supported RHI shader route"]
    if not all(isinstance(value.get(field), str) and value[field] for field in ("format", "environment", "abiKey")):
        return [f"{prefix} route lacks format, environment, or ABI evidence"]
    return []


ADMISSION_CAPABILITY_FIELDS = {
    "multiview": "multiview",
    "pushConstants": "push_constants",
    "bindingArrays": "binding_arrays",
    "runtimeSizedBindingArrays": "runtime_sized_binding_arrays",
    "storageBuffers": "storage_buffers",
    "storageTextures": "storage_textures",
    "nonuniformUniformBufferIndexing": "nonuniform_uniform_buffer_indexing",
    "nonuniformStorageBufferIndexing": "nonuniform_storage_buffer_indexing",
    "nonuniformSampledTextureIndexing": "nonuniform_sampled_texture_indexing",
    "nonuniformStorageTextureIndexing": "nonuniform_storage_texture_indexing",
    "nonuniformSamplerIndexing": "nonuniform_sampler_indexing",
}
ADMISSION_LIMIT_FIELDS = {
    "maxMultiviewViews": "max_multiview_views",
    "maxPushConstantBytes": "max_push_constant_bytes",
}
ADMISSION_NONUNIFORM_FIELDS = {
    "uniformBuffers", "storageBuffers", "sampledTextures", "storageTextures",
    "samplers", "bindingArrays", "runtimeSizedBindingArrays",
}
# These logical Naga requirements deliberately bypass generic device-capability
# admission.  The producer puts their corresponding concrete RHI feature name
# in ShaderDesc.required_features, where Device::shader_route owns admission.
# Keep this mapping in lockstep with shader::requirements_from_validation.
ROUTE_OWNED_DEMAND_FEATURES = {
    "primitive-index": "primitive-index",
    "early-depth-test": "shader-early-depth-test",
    "ray-query": "ray-query",
}
ROUTE_OWNED_DEMAND_CAPABILITIES = {
    "primitive-index": "shader_primitive_index",
    "early-depth-test": "shader_early_depth_test",
    "ray-query": "ray_query",
}
ROUTE_OWNED_FRAGMENT_DEMANDS = {"primitive-index", "early-depth-test"}
TARGET_ENTRY_STAGES = {
    "vertex", "fragment", "compute", "task", "mesh", "ray-generation",
    "any-hit", "closest-hit", "miss", "intersection", "callable",
}


def admission_result(status: str, demand: str | None = None, kind: str | None = None) -> dict[str, Any]:
    if status == "accepted":
        return {"status": "accepted"}
    return {"status": "refused", "error": {"kind": kind, "demand": demand}}


def admission_demand_result(demand: str, capabilities: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    """Recompute the public consumer admission contract without route inference."""
    nonuniform = facts["nonuniform"]
    if demand in ROUTE_OWNED_DEMAND_FEATURES:
        return admission_result("accepted")
    if demand == "multiview":
        views = facts["multiviewViews"]
        if views is None:
            return admission_result("refused", demand, "invalid-pipeline-facts")
        if not capabilities["multiview"] or views == 0 or views > capabilities["limits"]["maxMultiviewViews"]:
            return admission_result("refused", demand, "unsupported")
        return admission_result("accepted")
    if demand == "push-constant":
        size = facts["pushConstantBytes"]
        if size == 0:
            return admission_result("refused", demand, "invalid-pipeline-facts")
        if not capabilities["pushConstants"] or size > capabilities["limits"]["maxPushConstantBytes"]:
            return admission_result("refused", demand, "unsupported")
        return admission_result("accepted")
    if demand == "multisampled-shading":
        return (admission_result("refused", demand, "invalid-pipeline-facts")
                if facts["multisampleCount"] < 2
                else admission_result("refused", demand, "unrepresentable"))

    def nonuniform_result(supported: bool, resource_supported: bool, concrete: bool) -> dict[str, Any]:
        if not concrete or not nonuniform["bindingArrays"]:
            return admission_result("refused", demand, "invalid-pipeline-facts")
        if (not capabilities["bindingArrays"] or not resource_supported or not supported
                or (nonuniform["runtimeSizedBindingArrays"]
                    and not capabilities["runtimeSizedBindingArrays"])):
            return admission_result("refused", demand, "unsupported")
        return admission_result("accepted")

    if demand == "uniform-buffer-array-non-uniform-indexing":
        return nonuniform_result(capabilities["nonuniformUniformBufferIndexing"], True,
                                 nonuniform["uniformBuffers"])
    if demand == "storage-texture-array-non-uniform-indexing":
        return nonuniform_result(capabilities["nonuniformStorageTextureIndexing"],
                                 capabilities["storageTextures"], nonuniform["storageTextures"])
    if demand == "sampler-non-uniform-indexing":
        return nonuniform_result(capabilities["nonuniformSamplerIndexing"], True,
                                 nonuniform["samplers"])
    if demand == "sampled-texture-and-storage-buffer-array-non-uniform-indexing":
        if not nonuniform["sampledTextures"] and not nonuniform["storageBuffers"]:
            return admission_result("refused", demand, "invalid-pipeline-facts")
        if nonuniform["sampledTextures"]:
            result = nonuniform_result(capabilities["nonuniformSampledTextureIndexing"], True, True)
            if result["status"] == "refused":
                return result
        if nonuniform["storageBuffers"]:
            return nonuniform_result(capabilities["nonuniformStorageBufferIndexing"],
                                     capabilities["storageBuffers"], True)
        return admission_result("accepted")
    return admission_result("refused", demand, "unrepresentable")


def route_owned_stage_failures(target_entries: list[dict[str, Any]], demands: list[str], label: str) -> list[str]:
    """Check the public fragment-entry precondition enforced by Device::shader_route.

    `targetEntries` is recorded directly from the ShaderDesc passed to
    shader_route.  Composed WGSL alone would not prove the selected target.
    """
    if not any(demand in ROUTE_OWNED_FRAGMENT_DEMANDS for demand in demands):
        return []
    if not any(entry["stage"] == "fragment" for entry in target_entries):
        return [f"{label} route-owned fragment demand lacks a Fragment target entry"]
    return []


def shader_admission_failures(run: Path, root: Path, label: str, must_be_accepted: bool,
                              capability_root: Path | None = None) -> list[str]:
    value, error = read_json(root / "shader-admission.json")
    if error or not isinstance(value, dict):
        return [f"{label} shader admission evidence is missing or malformed"]
    reasons: list[str] = []
    demands, features = value.get("nagaCapabilityDemands"), value.get("rhiFeatureNames")
    if (value.get("schemaVersion") != 1 or not isinstance(demands, list)
            or not isinstance(features, list)
            or any(not isinstance(item, str) or not item for item in demands + features)
            or len(set(demands)) != len(demands) or len(set(features)) != len(features)):
        return [f"{label} shader admission demand lists are invalid"]
    feature_names = set(features)
    for demand, required_feature in ROUTE_OWNED_DEMAND_FEATURES.items():
        if demand in demands and required_feature not in feature_names:
            reasons.append(
                f"{label} shader admission demand {demand} is not passed to the RHI route as {required_feature}"
            )
    capabilities, facts = value.get("deviceCapabilities"), value.get("pipelineFacts")
    if not isinstance(capabilities, dict) or set(capabilities) != set(ADMISSION_CAPABILITY_FIELDS) | {"limits"}:
        return [f"{label} shader admission capability facts are incomplete"]
    if (any(type(capabilities[name]) is not bool for name in ADMISSION_CAPABILITY_FIELDS)
            or not isinstance(capabilities["limits"], dict)
            or set(capabilities["limits"]) != set(ADMISSION_LIMIT_FIELDS)
            or any(type(capabilities["limits"][name]) is not int or capabilities["limits"][name] < 0
                   for name in ADMISSION_LIMIT_FIELDS)):
        return [f"{label} shader admission capability facts have invalid types"]
    if not isinstance(facts, dict) or set(facts) != {"multiviewViews", "pushConstantBytes", "multisampleCount", "nonuniform"}:
        return [f"{label} shader admission pipeline facts are incomplete"]
    if (facts["multiviewViews"] is not None
            and (type(facts["multiviewViews"]) is not int or facts["multiviewViews"] < 0)
            or type(facts["pushConstantBytes"]) is not int or facts["pushConstantBytes"] < 0
            or type(facts["multisampleCount"]) is not int or facts["multisampleCount"] < 1
            or not isinstance(facts["nonuniform"], dict)
            or set(facts["nonuniform"]) != ADMISSION_NONUNIFORM_FIELDS
            or any(type(facts["nonuniform"][name]) is not bool for name in ADMISSION_NONUNIFORM_FIELDS)):
        return [f"{label} shader admission pipeline facts have invalid types"]
    target_entries = value.get("targetEntries")
    if not isinstance(target_entries, list) or not target_entries:
        return [f"{label} shader admission target entries are missing or malformed"]
    for entry in target_entries:
        if (not isinstance(entry, dict) or set(entry) != {"name", "stage", "workgroupSize"}
                or not isinstance(entry["name"], str) or not entry["name"]
                or entry["stage"] not in TARGET_ENTRY_STAGES):
            return [f"{label} shader admission target entries are missing or malformed"]
        workgroup_size = entry["workgroupSize"]
        if (workgroup_size is not None
                and (not isinstance(workgroup_size, list) or len(workgroup_size) != 3
                     or any(type(size) is not int or size < 0 for size in workgroup_size))):
            return [f"{label} shader admission target entries are missing or malformed"]

    snapshot, snapshot_error = read_json((capability_root or run) / "capabilities.json")
    actual = snapshot.get("deviceCapabilities") if isinstance(snapshot, dict) else None
    if snapshot_error or not isinstance(actual, dict):
        return [f"{label} shader admission cannot be tied to the run capability snapshot"]
    for demand, capability_name in ROUTE_OWNED_DEMAND_CAPABILITIES.items():
        if demand in demands and actual.get(capability_name) is not True:
            reasons.append(
                f"{label} shader admission route-owned demand {demand} lacks true {capability_name} in the run snapshot"
            )
    reasons.extend(route_owned_stage_failures(target_entries, demands, label))
    for recorded, actual_name in ADMISSION_CAPABILITY_FIELDS.items():
        if actual.get(actual_name) is not capabilities[recorded]:
            reasons.append(f"{label} shader admission capability {recorded} disagrees with the run snapshot")
    actual_limits = actual.get("limits")
    if not isinstance(actual_limits, dict):
        reasons.append(f"{label} run capability snapshot lacks limits for shader admission")
    else:
        for recorded, actual_name in ADMISSION_LIMIT_FIELDS.items():
            if actual_limits.get(actual_name) != capabilities["limits"][recorded]:
                reasons.append(f"{label} shader admission limit {recorded} disagrees with the run snapshot")

    per_demand = value.get("perDemand")
    if not isinstance(per_demand, list) or len(per_demand) != len(demands):
        return reasons + [f"{label} shader admission per-demand evidence is incomplete"]
    seen: list[str] = []
    expected_results: list[dict[str, Any]] = []
    for item in per_demand:
        if not isinstance(item, dict) or set(item) != {"demand", "outcome"} or not isinstance(item["demand"], str):
            reasons.append(f"{label} shader admission has malformed per-demand evidence")
            continue
        demand = item["demand"]
        seen.append(demand)
        expected = admission_demand_result(demand, capabilities, facts)
        expected_results.append(expected)
        if item["outcome"] != expected:
            reasons.append(f"{label} shader admission outcome for {demand} is not derived from its facts")
    if seen != demands or len(set(seen)) != len(seen):
        reasons.append(f"{label} shader admission per-demand evidence does not exactly match its demands")
    expected_admission = next((item for item in expected_results if item["status"] == "refused"), admission_result("accepted"))
    if value.get("admission") != expected_admission:
        reasons.append(f"{label} shader admission aggregate outcome does not match its per-demand facts")
    if must_be_accepted and value.get("admission") != admission_result("accepted"):
        reasons.append(f"{label} executed workload has a refused shader admission")
    return reasons


def pipeline_cache_hit_failures(value: Any, label: str) -> list[str]:
    if not isinstance(value, dict) or value.get("pipelineCacheHitVerified") is not True:
        return [f"{label} did not retain verified pipeline-cache hit evidence"]
    return []


def s2_shader_invocation_failures(run: Path, root: Path, label: str,
                                  capability_root: Path) -> tuple[list[str], dict[str, Any] | None]:
    value, error = read_json(root / "s2-shader-workload.json")
    if error or not isinstance(value, dict) or value.get("workload") != "s2-direct-shader-raster-three-generations":
        return [f"missing or invalid {label} workload"], None
    reasons = shader_route_failures(root, "s2-shader")
    reasons.extend(shader_admission_failures(run, root, label, value.get("status") == "executed", capability_root))
    if value.get("status") != "executed" or not strictly_increasing(value.get("assetGenerations")) or value.get("planCompilations") != 1:
        reasons.append(f"{label} did not execute three retained generations")
    layout = value.get("vertexLayout")
    if layout != {"stride": 24, "attributes": [{"semantic": "POSITION", "location": 0}, {"semantic": "COLOR_0", "location": 1}]}:
        reasons.append(f"{label} lacks the reflected position/color vertex layout")
    source = root / "s2-shader-artifact.wgsl"
    interface = root / "s2-shader-interface.txt"
    if (not source.is_file() or not interface.is_file()
            or not all(token in source.read_text(encoding="utf-8", errors="replace")
                       for token in ("vs_main", "fs_main", "@location(0)", "@location(1)"))):
        reasons.append(f"{label} artifact or vertex interface evidence is incomplete")
    mask = root / "s2-shader-oracle-mask.bin"
    if not mask.is_file():
        reasons.append(f"{label} oracle mask is missing")
        mask_bytes = None
    else:
        mask_bytes = mask.read_bytes()
    for index in range(3):
        raw, expected, readback = (root / f"s2-shader-{kind}-{index}.{suffix}" for kind, suffix in (("input", "bin"), ("expected", "rgba"), ("readback", "rgba")))
        if not raw.is_file() or not expected.is_file() or not readback.is_file():
            reasons.append(f"missing {label} raw input, expected, or readback for generation {index}")
            continue
        color, failures = fixed_raster_oracle(raw.read_bytes())
        reasons.extend(f"S2 generation {index}: {failure}" for failure in failures)
        if color is not None and mask_bytes is not None:
            reasons.extend(stable_raster_mask_failures(mask_bytes, expected.read_bytes(), color, f"S2 expected {index}"))
            reasons.extend(stable_raster_mask_failures(mask_bytes, readback.read_bytes(), color, f"S2 readback {index}"))
    return reasons, value


def s2_shader_failures(run: Path) -> list[str]:
    first_root = run / "s2-shader"
    same_root = run / "s2-shader-same-device-cache-hit"
    replacement_root = run / "s2-shader-replacement-device"
    first_reasons, first = s2_shader_invocation_failures(run, first_root, "S2 initial Shader invocation", run)
    same_reasons, same = s2_shader_invocation_failures(run, same_root, "S2 same-device cache invocation", run)
    replacement_reasons, replacement = s2_shader_invocation_failures(
        run, replacement_root, "S2 replacement-device invocation", replacement_root)
    reasons = first_reasons + same_reasons + replacement_reasons
    invocations = ((first, False, 0, 1, 1), (same, True, 1, 0, 1),
                   (replacement, False, 1, 1, 2))
    for value, expected_hit, before, built, after in invocations:
        if not isinstance(value, dict):
            continue
        if (value.get("consumerInvocationCacheHit") is not expected_hit
                or value.get("consumerInvocationExpectedCacheHit") is not expected_hit
                or value.get("consumerPipelineBuildSuccessesBefore") != before
                or value.get("consumerPipelineBuildSuccessesThisInvocation") != built
                or value.get("consumerPipelineBuildSuccessesAfter") != after):
            reasons.append("S2 consumer cache hit/miss or pipeline-build-success counters are inconsistent")
    if all(isinstance(value, dict) for value in (first, same, replacement)):
        if not isinstance(first.get("deviceId"), str) or not isinstance(same.get("deviceId"), str) or not isinstance(replacement.get("deviceId"), str):
            reasons.append("S2 invocation evidence lacks a concrete DeviceId")
        elif first["deviceId"] != same["deviceId"]:
            reasons.append("S2 same-device cache invocation changed DeviceId")
        elif first["deviceId"] == replacement["deviceId"]:
            reasons.append("S2 replacement-device invocation did not use a new DeviceId")
        transition, error = read_json(run / "s2-cache-device-transition.json")
        if (error or not isinstance(transition, dict)
                or transition.get("oldDeviceId") != first.get("deviceId")
                or transition.get("replacementDeviceId") != replacement.get("deviceId")
                or any(transition.get(field) is not True for field in (
                    "bothProductsPresentBeforeRetirement", "oldProductRetiredAfterCompletion",
                    "replacementProductRetained",
                ))):
            reasons.append("S2 cache device-transition evidence does not match the three invocation devices")
    return reasons


def s2_texture_failures(run: Path) -> list[str]:
    root = run / "s2-texture"
    value, error = read_json(root / "read-set-workload.json")
    if error or not isinstance(value, dict) or value.get("workload") != "s2-direct-shader-texture-read-set-three-generations":
        return ["missing or invalid S2 texture Shader workload"]
    reasons = shader_route_failures(root, "s2-texture")
    reasons.extend(shader_admission_failures(run, root, "S2 texture Shader workload", True))
    source, artifact, interface = root / "s2-texture-input.wgsl", root / "s2-texture-artifact.wgsl", root / "s2-texture-interface.txt"
    if not all(path.is_file() for path in (source, artifact, interface)):
        reasons.append("S2 texture Shader source, artifact, or reflection evidence is missing")
    else:
        input_source = source.read_text(encoding="utf-8", errors="replace")
        composed = artifact.read_text(encoding="utf-8", errors="replace")
        reflected = interface.read_text(encoding="utf-8", errors="replace")
        if (input_source not in composed
                or not all(token in input_source for token in ("s2_tint", "s2_texture", "textureLoad", "vs_main", "fs_main"))):
            reasons.append("S2 texture Shader artifact does not contain its direct texture program")
        if not all(token in reflected for token in ("group: 0", "binding: 0", "binding: 1", "Uniform", "Texture")):
            reasons.append("S2 texture Shader reflection does not prove both read-set bindings")
    reasons.extend(read_set_failures(
        run, "s2-texture", "s2-direct-shader-texture-read-set-three-generations",
        [[17, 0, 241, 255], [0, 33, 125, 255], [67, 199, 0, 255]],
    ))
    return reasons


def s3_material_failures(run: Path) -> list[str]:
    root = run / "s3-material"
    reasons = material_shader_failures(root, "s3-material", False)
    reasons.extend(read_set_failures(run, "s3-material", "s3-material-read-set-raster-three-generations",
                                     [[17, 0, 0, 255], [0, 33, 0, 255], [0, 0, 211, 255]]))
    replacement = root / "replacement"
    reasons.extend(material_shader_failures(replacement, "s3-material", True))
    reasons.extend(read_set_failures(run, "s3-material/replacement",
                                     "s3-material-replacement-read-set-raster-three-generations",
                                     [[241, 0, 0, 255], [0, 231, 0, 255], [0, 0, 199, 255]],
                                     (2, 0, 1, 3)))
    preflight, preflight_error = read_json(root / "s3-material-binding-preflight.json")
    if (preflight_error or not isinstance(preflight, dict) or preflight.get("schemaVersion") != 1
            or any(preflight.get(field) is not True for field in (
                "actualReadSetLayoutChecked", "fragmentReadOnlyAccessChecked",
                "incompatibleReflectionRejectedBeforeBackend", "foreignTokenRejectedBeforeBackend",
            ))):
        reasons.append("S3 Material binding preflight evidence is missing or incomplete")
    generations, error = read_json(root / "s3-material-generations.json")
    if (error or not isinstance(generations, dict) or generations.get("sameAsset") is not True
            or generations.get("variantChanged") is not True
            or type(generations.get("original")) is not int
            or type(generations.get("replacement")) is not int
            or generations["original"] >= generations["replacement"]):
        reasons.append("S3 Material replacement generation evidence is missing or inconsistent")
    return reasons


def material_shader_failures(root: Path, prefix: str, replacement: bool) -> list[str]:
    label = "S3 replacement Material" if replacement else "S3 Material"
    reasons = shader_route_failures(root, prefix)
    reasons.extend(shader_admission_failures(root.parents[1] if replacement else root.parent, root, label, True))
    contribution, artifact = root / f"{prefix}-contribution.wgsl", root / f"{prefix}-artifact.wgsl"
    variant, definition = root / f"{prefix}-variant.txt", root / f"{prefix}-definition.txt"
    interface = root / f"{prefix}-interface.txt"
    if not all(path.is_file() for path in (contribution, artifact, variant, definition, interface)):
        return reasons + [f"{label} contribution, artifact, variant, definition, or reflection evidence is missing"]
    source, composed = contribution.read_text(encoding="utf-8", errors="replace"), artifact.read_text(encoding="utf-8", errors="replace")
    if (not variant.read_text(encoding="utf-8", errors="replace").strip()
            or source not in composed
            or not all(token in source for token in ("var<uniform> tint", "material_texture", "sample_color", "custom_tint", "evaluate_material"))):
        reasons.append(f"{label} contribution is incomplete or was not the composed artifact input")
    if replacement and not all(token in source for token in ("c.b", "c.r", "c.g")):
        reasons.append("S3 replacement Material did not retain its channel-reordering static function")
    reasons.extend(material_interface_failures(interface, label))
    return reasons


def material_interface_failures(path: Path, label: str) -> list[str]:
    """Validate both read-set resources from retained ShaderInterface Debug evidence."""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = text.split("ReflectedBinding {")[1:]
    requirements = (
        (0, ("group: 0", "binding: 0", "kind: Uniform", "shape: Some(", "Uniform(", "Fragment", "BindingAccess", "read: true", "write: false", "query: false", "atomic: false")),
        (1, ("group: 0", "binding: 1", "kind: Texture", "shape: Some(", "Texture2dF32", "Fragment", "BindingAccess", "read: true", "write: false", "query: false", "atomic: false")),
    )
    reasons: list[str] = []
    for binding, required in requirements:
        block = next((item for item in blocks if f"binding: {binding}" in item), None)
        if block is None or not all(token in block for token in required):
            reasons.append(f"{label} reflection does not prove the group 0 binding {binding} kind, shape, Fragment read access")
    return reasons


def compute_requirement_checks(capability: Any) -> dict[str, bool] | None:
    if not isinstance(capability, dict) or not isinstance(capability.get("deviceCapabilities"), dict):
        return None
    caps = capability["deviceCapabilities"]
    limits = caps.get("limits")
    if not isinstance(limits, dict):
        return None
    def at_least(name: str, minimum: int) -> bool:
        number = limits.get(name)
        return type(number) is int and number >= minimum
    def axis(name: str, index: int, minimum: int) -> bool:
        values = limits.get(name)
        return (isinstance(values, list) and len(values) == 3
                and type(values[index]) is int and values[index] >= minimum)
    return {"compute": caps.get("compute") is True,
            "storage_buffers": caps.get("storage_buffers") is True,
            "buffer_bytes": at_least("max_buffer_bytes", 96 * 4),
            "bind_groups": at_least("max_bind_groups", 1),
            "bindings_per_group": at_least("max_bindings_per_group", 1),
            "workgroups_x": axis("max_compute_workgroups", 0, 2),
            "workgroups_y": axis("max_compute_workgroups", 1, 1),
            "workgroups_z": axis("max_compute_workgroups", 2, 1),
            "workgroup_size_x": axis("max_compute_workgroup_size", 0, 64),
            "workgroup_size_y": axis("max_compute_workgroup_size", 1, 1),
            "workgroup_size_z": axis("max_compute_workgroup_size", 2, 1)}


def compute_failures(run: Path, backend: str) -> list[str]:
    value, error = read_json(run / "compute" / "compute-workload.json")
    if error or not isinstance(value, dict) or value.get("workload") != "s1-fixed-compute-three-generations":
        return ["missing or invalid compute workload"]
    reasons: list[str] = []
    reasons.extend(shader_admission_failures(
        run, run / "compute", "compute Shader workload",
        value.get("status") != "unsupported-before-submission",
    ))
    composed, interface = run / "compute" / "compute-composed.wgsl", run / "compute" / "compute-interface.txt"
    route, route_error = read_json(run / "compute" / "compute-route.json")
    if (not composed.is_file() or not interface.is_file()
            or not all(token in composed.read_text(encoding="utf-8", errors="replace")
                       for token in ("cs_main", "@compute", "@group(0)", "@binding(0)"))):
        reasons.append("compute Shader artifact or reflection evidence is incomplete")
    if (route_error or not isinstance(route, dict) or route.get("route") not in ("direct", "naga", "unsupported")
            or (route.get("route") != "unsupported" and not all(isinstance(route.get(field), str) and route[field]
                                                                    for field in ("format", "environment", "abiKey")))):
        reasons.append("compute Shader route evidence is missing or malformed")
    capability, capability_error = read_json(run / "capabilities.json")
    expected_checks = compute_requirement_checks(capability)
    admission, admission_error = read_json(run / "compute" / "compute-requirements.json")
    if (capability_error or expected_checks is None or admission_error or not isinstance(admission, dict)
            or admission.get("checks") != expected_checks
            or any(not isinstance(part, bool) for part in admission.get("checks", {}).values())
            or admission.get("supported") is not all(expected_checks.values())):
        return reasons + ["compute admission checks are missing or disagree with actual capability requirements"]
    supported = all(expected_checks.values())
    if value.get("status") == "unsupported-before-submission":
        if supported:
            return reasons + ["unsupported compute conflicts with satisfied runtime fixture requirements"]
        if (run / "compute-refusal.json").exists():
            return reasons + ["compute evidence must use compute-workload status, not compute-refusal.json"]
        root = run / "compute"
        allowed = {f"compute-{kind}-{index}.bin" for kind in ("input", "expected") for index in range(3)}
        for index in range(3):
            source, expected_file = root / f"compute-input-{index}.bin", root / f"compute-expected-{index}.bin"
            if not source.is_file() or not expected_file.is_file():
                reasons.append(f"unsupported compute lacks retained input or expected artifact {index}")
                continue
            before = source.read_bytes()
            expected = b"".join(((int.from_bytes(before[offset:offset + 4], "little") * 3 + 7) & 0xffffffff).to_bytes(4, "little") for offset in range(0, len(before), 4)) if len(before) == 384 else b""
            if expected_file.read_bytes() != expected:
                reasons.append(f"unsupported compute expected artifact {index} does not match its raw input")
        stale = [path.name for path in root.glob("compute-*.bin") if path.name not in allowed]
        return reasons + (["unsupported compute retained native submission artifacts"] if stale else [])
    if not supported:
        reasons.append("compute workload ran without its runtime fixture requirements")
    if (value.get("elements") != 96 or value.get("workgroupSize") != 64
            or value.get("operation") != "u32.wrapping_mul(3).wrapping_add(7)"
            or not strictly_increasing([item.get("generation") if isinstance(item, dict) else None for item in value.get("originalsByGeneration", [])])
            or value.get("snapshotRetentionVerified") is not True
            or value.get("perFrameRetirementVerified") is not True
            or value.get("planCompilations") != 1):
        reasons.append("compute workload metadata does not match the fixed fixture contract")
    for index in range(3):
        source = run / "compute" / f"compute-input-{index}.bin"
        expected_file = run / "compute" / f"compute-expected-{index}.bin"
        output = run / "compute" / f"compute-readback-{index}.bin"
        if not source.is_file() or not expected_file.is_file() or not output.is_file():
            reasons.append(f"missing compute raw artifact for generation {index}")
            continue
        before = source.read_bytes()
        expected = b"".join(((int.from_bytes(before[offset:offset + 4], "little") * 3 + 7) & 0xffffffff).to_bytes(4, "little") for offset in range(0, len(before), 4)) if len(before) == 384 else b""
        if expected_file.read_bytes() != expected or output.read_bytes() != expected:
            reasons.append(f"compute oracle mismatch for generation {index}")
    return reasons


def replacement_failures(run: Path) -> list[str]:
    value, error = read_json(run / "device-replacement.json")
    if error or not isinstance(value, dict):
        return ["missing or invalid device replacement evidence"]
    reasons: list[str] = []
    if value.get("schemaVersion") != 1:
        reasons.append("device replacement evidence has an unsupported schema")
    old, replacement = value.get("oldDeviceId"), value.get("replacementDeviceId")
    if not isinstance(old, str) or not old or not isinstance(replacement, str) or not replacement or old == replacement:
        reasons.append("device replacement evidence does not prove distinct devices")
    for field in ("foreignPipelineSubmissionRejectedBeforeBackendWork",
                  "foreignSubmissionRejectedBeforeBackendWait",
                  "logicalSnapshotRetainedThroughRefusal",
                  "replacementPipelineSubmissionReadbackVerified",
                  "rejectionBackendStatsUnchanged"):
        if value.get(field) is not True:
            reasons.append(f"device replacement evidence did not establish {field}")
    artifacts: dict[str, Path] = {}
    for field in ("input", "expected", "readback"):
        item = value.get(field)
        path = under(run, item) if isinstance(item, str) else None
        if path is None or not path.is_file():
            reasons.append(f"device replacement {field} path is missing or escapes the run")
        else:
            artifacts[field] = path
    if len(artifacts) == 3:
        color, input_reasons = fixed_raster_oracle(artifacts["input"].read_bytes())
        reasons.extend(input_reasons)
        expected, error = read_json(artifacts["expected"])
        if error or not isinstance(expected, dict) or expected.get("extent") != [128,128] or expected.get("rgba") != color:
            reasons.append("device replacement expected contract does not match its raw input")
        if color is not None:
            raw = artifacts["readback"].read_bytes()
            reasons.extend(dense_raster_failures(raw, color, "device replacement"))
            if value.get("readbackBytes") != len(raw):
                reasons.append("device replacement readback byte count differs from raw output")
    return reasons


def workload_check(run: Path, backend: str) -> dict[str, Any]:
    required = {
        "copy": (run / "workload.json", "s0-copy-three-generations"),
        "raster": (run / "raster" / "raster-workload.json", "s1-fixed-raster-three-generations"),
        "surface": (run / "surface" / "surface-workload.json", "s1-renderer-surface-raster-three-generations"),
    }
    reasons: list[str] = []
    loaded: dict[str, Any] = {}
    for name, (path, workload) in required.items():
        value, error = read_json(path)
        if error or not isinstance(value, dict):
            reasons.append(f"missing or invalid {name} workload: {path}")
        elif value.get("workload") != workload:
            reasons.append(f"unexpected {name} workload identifier")
        else:
            loaded[name] = value
    for index in range(3):
        for suffix in ("input", "expected", "readback"):
            if not (run / f"{suffix}-{index}.bin").is_file():
                reasons.append(f"missing copy {suffix} artifact {index}")
        expected = (run / f"expected-{index}.bin").read_bytes() if (run / f"expected-{index}.bin").is_file() else None
        actual = (run / f"readback-{index}.bin").read_bytes() if (run / f"readback-{index}.bin").is_file() else None
        if expected is not None and actual is not None and (not expected or actual != expected):
            reasons.append(f"copy oracle mismatch for generation {index}")
    surface = loaded.get("surface", {})
    for field in ("hostResizeEventVerified", "clientExtentVerified", "completionBeforeReconfigure"):
        if surface.get(field) is not True:
            reasons.append(f"surface workload did not establish {field}")
    if not (run / "surface" / "surface-events.jsonl").is_file():
        reasons.append("missing raw surface event log")
    else:
        lines = (run / "surface" / "surface-events.jsonl").read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines or any(read_json_line(line) is None for line in lines):
            reasons.append("surface event log is not valid raw JSONL")
    reasons.extend(raster_failures(run, loaded.get("raster")))
    reasons.extend(read_set_failures(run))
    reasons.extend(s2_shader_failures(run))
    reasons.extend(s2_texture_failures(run))
    reasons.extend(s3_material_failures(run))
    reasons.extend(compute_failures(run, backend))
    reasons.extend(replacement_failures(run))
    reasons.extend(surface_configuration_failures(run))
    return result("raw workload artifacts and oracles", not reasons, reasons, run=str(run))


def read_json_line(line: str) -> Any | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def surface_configuration_failures(run: Path) -> list[str]:
    path = run / "surface" / "surface-configuration.jsonl"
    if not path.is_file():
        return ["missing raw surface capability/configuration observations"]
    observations = [read_json_line(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(observations) != 3:
        return ["surface requires exactly three observed configuration checks"]
    reasons: list[str] = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict) or observation.get("configurationSupport") is not True:
            reasons.append(f"surface configuration {index} did not pass exact descriptor support")
            continue
        caps, config = observation.get("capabilities"), observation.get("configuration")
        if not isinstance(caps, dict) or not isinstance(config, dict):
            reasons.append(f"surface configuration {index} omits capability/descriptor facts")
            continue
        for field, choices in {"format": "formats", "presentMode": "presentModes", "alphaMode": "alphaModes", "colorSpace": "colorSpaces"}.items():
            if not isinstance(caps.get(choices), list) or config.get(field) not in caps[choices]:
                reasons.append(f"surface configuration {index} uses unadvertised {field}")
        extent, minimum, maximum = config.get("extent"), caps.get("minExtent"), caps.get("maxExtent")
        if (any(not isinstance(item, list) or len(item) != 2 or any(type(axis) is not int for axis in item)
                for item in (extent, minimum, maximum))
                or extent != observation.get("extent")
                or any(not minimum[axis] <= extent[axis] <= maximum[axis] or extent[axis] <= 0 for axis in (0, 1))):
            reasons.append(f"surface configuration {index} has invalid extent capability facts")
        usages, usage = caps.get("usages"), config.get("usage")
        if (type(usages) is not int or type(usage) is not int or usage <= 0
                or usage & usages != usage or usage & 1 == 0):
            reasons.append(f"surface configuration {index} has unsupported render-attachment usage")
    return reasons


def attempt_review(attempt: dict[str, Any], closure: Path, root: Path) -> list[dict[str, Any]]:
    identifier = str(attempt.get("id"))
    directory = closure / str(attempt.get("directory", ""))
    records: list[dict[str, Any]] = []
    reasons: list[str] = []
    if attempt.get("mode") != "run" or attempt.get("runner_exit_status") != 0:
        reasons.append("attempt did not complete a successful run")
    if attempt.get("source_consistency_after_build", {}).get("status") != "stable":
        reasons.append("source changed during build")
    if attempt.get("source_consistency_after_attempt", {}).get("status") != "stable":
        reasons.append("source changed during run")
    if attempt.get("build_environment", {}).get("status") != "auditable":
        reasons.append("build environment was not retained as an auditable, non-secret closure")
    if attempt.get("built_source_verification", {}).get("status") != "verified":
        reasons.append("Cargo built-source verification is not verified")
    checkout_verification = attempt.get("built_git_checkout_verification")
    if not isinstance(checkout_verification, dict) or checkout_verification.get("status") != "verified":
        reasons.append("Cargo built Git checkout verification is not verified")
    else:
        checkouts = checkout_verification.get("checkouts")
        expected_revisions = {
            package.get("source", "").partition("#")[2]
            for package in attempt.get("resolved_git_packages", []) if isinstance(package, dict)
            and isinstance(package.get("source"), str) and package["source"].startswith("git+")
        }
        if not isinstance(checkouts, list) or not checkouts:
            reasons.append("Cargo built Git checkout verification retained no inspected checkout")
        for checkout in checkouts if isinstance(checkouts, list) else []:
            if (not isinstance(checkout, dict) or not isinstance(checkout.get("path"), str)
                    or not checkout.get("path") or not isinstance(checkout.get("revision"), str)
                    or not checkout.get("revision") or checkout.get("changes") != []):
                reasons.append("Cargo built Git checkout is incomplete, dirty, or lacks its resolved revision")
                break
            if checkout["revision"] not in expected_revisions:
                reasons.append("Cargo built Git checkout revision is not pinned by the retained lock evidence")
                break
    if not attempt.get("source_inputs"):
        reasons.append("attempt did not retain source input hashes")
    reasons.extend(current_source_failures(attempt, root))
    reasons.extend(frozen_source_failures(attempt, closure))
    executable = attempt.get("executable")
    if not isinstance(executable, dict):
        reasons.append("attempt has no retained executable")
    else:
        reasons.extend(hash_records([executable], directory))
        if executable.get("after_run_sha256") != executable.get("sha256"):
            reasons.append("retained executable changed during its run")
    records.append(result("current, source-stable attempt", not reasons, reasons, attempt=identifier))

    lock_reasons: list[str] = []
    for lock in attempt.get("locks", []):
        lock_path = Path(str(lock.get("path", "")))
        if sha256(lock_path) != lock.get("sha256"):
            lock_reasons.append(f"lock hash differs or is missing: {lock_path}")
    if not attempt.get("locks"):
        lock_reasons.append("attempt records no lock")
    records.append(result("pinned lock evidence", not lock_reasons, lock_reasons, attempt=identifier))

    artifacts: list[dict[str, Any]] = []
    for command in attempt.get("commands", []):
        artifacts.extend(command.get("result", {}).get("raw_logs", []))
        artifacts.extend(command.get("result", {}).get("retained_files", []))
    if not attempt.get("commands"):
        hash_reasons = ["attempt records no build/run commands"]
    elif not artifacts:
        hash_reasons = ["attempt records no retained logs or artifacts"]
    else:
        hash_reasons = hash_records(artifacts, closure)
    records.append(result("recursive retained artifact hashes", not hash_reasons, hash_reasons,
                          attempt=identifier, artifacts=len(artifacts)))
    run = directory / "run"
    records.append(workload_check(run, str(attempt.get("backend"))))
    return records


def diagnostics_review(attempt: dict[str, Any], closure: Path) -> dict[str, Any]:
    validation = attempt.get("validation", {})
    diagnostic = validation.get("diagnostics", {})
    reasons: list[str] = []
    if validation.get("status") != "observed":
        reasons.append("validation activation is not explicitly verified")
    if diagnostic.get("unreviewed_native_diagnostics"):
        reasons.append("unreviewed native diagnostics are retained")
    if validation.get("native_diagnostic_count", 0) != 0:
        reasons.append("one or more native contexts retained diagnostics")
    artifact = validation.get("artifact")
    if not isinstance(artifact, dict):
        reasons.append("validation has no hashed current-attempt artifact")
    else:
        reasons.extend(hash_records([artifact], closure))
    backend = attempt.get("backend")
    additional = validation.get("additional_artifacts", [])
    if not isinstance(additional, list):
        reasons.append("additional diagnostic artifacts are malformed")
        additional = []
    for extra in additional:
        extra_validation = dict(validation, artifact=extra, additional_artifacts=[])
        report = diagnostics_review(dict(attempt, validation=extra_validation), closure)
        reasons.extend(report["reasons"])
    if isinstance(artifact, dict):
        marker = under(closure, str(artifact.get("path", "")))
        if (backend == "native-gl-wgl" and marker is not None
                and (marker.parent / "device-replacement.json").is_file() and not additional):
            reasons.append("WGL replacement context has no diagnostic activation artifact")
        value, error = read_json(marker) if marker is not None else (None, "artifact path escapes closure")
        if error or not isinstance(value, dict):
            reasons.append("validation artifact is not valid JSON")
        elif backend == "dx12":
            if value.get("requested") is not True or value.get("deviceOpenCompleted") is not True:
                reasons.append("DX12 artifact does not prove debug-enabled device open")
            contract = value.get("activationContract")
            if (not isinstance(contract, str) or "0f7f785d4f0ccc62e125dd6cf560d91333ec85cf" not in contract
                    or "EnableDebugLayer" not in contract):
                reasons.append("DX12 artifact lacks the pinned activation contract")
        elif backend == "native-gl-wgl":
            if value.get("logSaturated") is not False:
                reasons.append("WGL observer log was saturated or omitted its capacity result")
            version = value.get("contextVersion")
            parsed = re.match(r"^(\d+)\.(\d+)", version) if isinstance(version, str) else None
            core_debug = parsed is not None and tuple(map(int, parsed.groups())) >= (4, 3)
            if (value.get("coreDebugOutput") is not core_debug
                    or not isinstance(value.get("khrDebugExtension"), bool)
                    or value.get("debugOutputAvailable") is not (core_debug or value.get("khrDebugExtension") is True)
                    or value.get("debugOutputAvailable") is not True):
                reasons.append("WGL debug output lacks consistent core-version or extension admission evidence")
            messages = value.get("activationMessages")
            marker = (isinstance(messages, list) and any(
                isinstance(message, str) and "fluxel-rendering WGL diagnostic observer active" in message
                for message in messages))
            if value.get("installed") is not True or value.get("activationMarkerObserved") is not True or not marker:
                reasons.append("WGL observer lacks the returned application marker")
            if not isinstance(value.get("nativeDiagnostics"), list) or value.get("nativeDiagnostics"):
                reasons.append("WGL observer retained or omitted native diagnostics")
    return result(f"{backend} validation activation and diagnostics", not reasons, reasons,
                  attempt=attempt.get("id"), artifact=artifact)


def public_struct_fields(source: str, name: str) -> dict[str, str]:
    """Read the selected revision's simple public capability record, fail closed.

    This is a schema check, not native feature discovery. The producer still
    reads values from its actual Device; no support values are inferred here.
    """
    body = re.search(r"^pub struct " + re.escape(name) + r"\s*\{(.*?)^\}", source, re.M | re.S)
    if body is None:
        return {}
    return dict(re.findall(r"^\s*pub (\w+):\s*([^\n]+),\s*$", body.group(1), re.M))


def capability_snapshot_failures(value: dict[str, Any], source: str) -> list[str]:
    reasons: list[str] = []
    facts = value.get("deviceCapabilities")
    fields = public_struct_fields(source, "DeviceCapabilities")
    limits = public_struct_fields(source, "DeviceLimits")
    if not fields or not limits:
        return ["selected RHI capability declaration could not be decoded"]
    if not isinstance(facts, dict) or set(facts) != set(fields):
        return ["capability snapshot does not enumerate exactly the selected RHI capability fields"]
    for name, kind in fields.items():
        item = facts[name]
        if kind == "bool":
            if not isinstance(item, bool):
                reasons.append(f"capability {name} is not boolean")
        elif kind == "DeviceLimits":
            if not isinstance(item, dict) or set(item) != set(limits):
                reasons.append("capability snapshot does not enumerate exactly the selected RHI limits")
                continue
            for limit, limit_kind in limits.items():
                number = item[limit]
                numbers = number if limit_kind == "[u32; 3]" and isinstance(number, list) and len(number) == 3 else [number]
                bits = 64 if limit_kind == "u64" else 32
                if (limit_kind not in ("u32", "u64", "[u32; 3]")
                        or (limit_kind == "[u32; 3]" and len(numbers) != 3)
                        or any(type(part) is not int or part < 0 or part >= 2**bits for part in numbers)):
                    reasons.append(f"capability limit {limit} has invalid type or range")
        elif kind in ("Arc<[LaneInfo]>", "Vec<CooperativeMatrixConfiguration>"):
            if not isinstance(item, list):
                reasons.append(f"capability {name} is not a structured list")
        else:
            reasons.append(f"capability {name} has an unreviewed RHI schema type {kind}")
    aliases = {"compute": "compute", "storageBuffers": "storage_buffers", "storageTextures": "storage_textures",
               "nativeInterop": "native_interop", "debugMarkers": "debug_markers", "multiQueue": "multi_queue"}
    for alias, field in aliases.items():
        if value.get(alias) is not facts.get(field):
            reasons.append(f"capability alias {alias} disagrees with the complete device snapshot")
    raw_limits = facts.get("limits")
    if isinstance(raw_limits, dict):
        for alias, field in {"maxBufferBytes": "max_buffer_bytes", "maxVertexBuffers": "max_vertex_buffers",
                             "maxVertexAttributes": "max_vertex_attributes", "maxColorAttachments": "max_color_attachments"}.items():
            if not isinstance(value.get("limits"), dict) or value["limits"].get(alias) != raw_limits.get(field):
                reasons.append(f"capability limit alias {alias} disagrees with the complete device snapshot")
    lanes = facts.get("lanes")
    if isinstance(lanes, list):
        if not lanes or type(value.get("lanes")) is not int or value["lanes"] != len(lanes):
            reasons.append("capability lane count is missing, empty or inconsistent")
        ordinals: set[int] = set()
        for lane in lanes:
            if not isinstance(lane, dict) or set(lane) != {"id", "class", "routes"}:
                reasons.append("capability lane is not a complete structured LaneInfo")
                continue
            identifier = lane.get("id")
            if (not isinstance(identifier, dict) or set(identifier) != {"device", "ordinal"}
                    or identifier.get("device") != value.get("deviceId")
                    or type(identifier.get("ordinal")) is not int or not 0 <= identifier["ordinal"] < 2**32
                    or identifier["ordinal"] in ordinals):
                reasons.append("capability lane identity is invalid, foreign or duplicated")
            else:
                ordinals.add(identifier["ordinal"])
            if lane.get("class") not in ("graphics", "compute", "transfer") or not isinstance(lane.get("routes"), list):
                reasons.append("capability lane class or routes are malformed")
        for lane in lanes:
            if not isinstance(lane, dict) or not isinstance(lane.get("routes"), list):
                continue
            for route in lane["routes"]:
                if not isinstance(route, dict) or set(route) != {"target", "route"}:
                    reasons.append("capability lane route is malformed")
                    continue
                target = route["target"]
                if (not isinstance(target, dict) or set(target) != {"device", "ordinal"}
                        or target.get("device") != value.get("deviceId")
                        or type(target.get("ordinal")) is not int or target["ordinal"] not in ordinals
                        or route["route"] not in ("same_queue", "native_gpu_wait", "ordered_by_native_queue", "unsupported")):
                    reasons.append("capability lane route target or kind is invalid")
    configurations = facts.get("cooperative_matrix_configurations")
    if isinstance(configurations, list):
        matrix_fields = public_struct_fields(source, "CooperativeMatrixConfiguration")
        for configuration in configurations:
            if not isinstance(configuration, dict) or not matrix_fields or set(configuration) != set(matrix_fields):
                reasons.append("cooperative matrix configuration omits selected RHI fields")
                continue
            for field, kind in matrix_fields.items():
                item = configuration[field]
                if ((kind == "u32" and (type(item) is not int or not 0 < item < 2**32))
                        or (kind == "bool" and not isinstance(item, bool))):
                    reasons.append(f"cooperative matrix configuration {field} is invalid")
                if kind in ("CooperativeMatrixComponentType", "CooperativeMatrixScope"):
                    declaration = re.search(r"^pub enum " + kind + r"\s*\{(.*?)^\}", source, re.M | re.S)
                    variants = [variant.lower().replace("queuefamily", "queue_family") for variant in
                                re.findall(r"^\s*(\w+),", declaration.group(1), re.M)] if declaration else []
                    if item not in variants:
                        reasons.append(f"cooperative matrix configuration {field} has an unknown enum value")
    return reasons


def capability_review(attempt: dict[str, Any], closure: Path) -> dict[str, Any]:
    directory = closure / str(attempt.get("directory", "")) / "run"
    paths = sorted(directory.rglob("*capabilit*.json"))
    expected = (
        ("primary", directory / "capabilities.json", directory / "s2-shader" / "s2-shader-workload.json",
         "oldDeviceId"),
        ("replacement", directory / "s2-shader-replacement-device" / "capabilities.json",
         directory / "s2-shader-replacement-device" / "s2-shader-workload.json", "replacementDeviceId"),
    )
    reasons: list[str] = []
    if set(paths) != {snapshot for _, snapshot, _, _ in expected}:
        reasons.append("selected attempt has missing or unexpected raw capability snapshots")
    replacement, replacement_error = read_json(directory / "device-replacement.json")
    workload, workload_error = read_json(directory / "workload.json")
    if replacement_error or not isinstance(replacement, dict):
        reasons.append("capability snapshots lack device replacement identity evidence")
    if workload_error or not isinstance(workload, dict) or not isinstance(workload.get("adapter"), str) or not workload["adapter"]:
        reasons.append("capability snapshots lack the selected adapter identity evidence")
    roots = [Path(item["path"]) for item in attempt.get("built_git_checkout_verification", {}).get("checkouts", [])
             if isinstance(item, dict) and isinstance(item.get("path"), str)
             and (Path(item["path"]) / "crates/core/src/api/foundation.rs").is_file()]
    declaration = None
    if len(roots) != 1:
        reasons.append("complete capability review lacks the unique built RHI declaration")
    else:
        declaration = (roots[0] / "crates/core/src/api/foundation.rs").read_text(encoding="utf-8")
    device_ids: list[Any] = []
    for label, snapshot_path, shader_workload_path, replacement_field in expected:
        value, error = read_json(snapshot_path)
        shader_workload, shader_workload_error = read_json(shader_workload_path)
        if (error or not isinstance(value, dict) or not isinstance(value.get("adapter"), str)
                or not value["adapter"]):
            reasons.append(f"{label} capability snapshot lacks an adapter identity")
            continue
        if value.get("backend") != attempt.get("backend"):
            reasons.append(f"{label} capability snapshot backend does not match the selected attempt")
        if not workload_error and isinstance(workload, dict) and value.get("adapter") != workload.get("adapter"):
            reasons.append(f"{label} capability snapshot adapter does not match the selected adapter")
        expected_device = replacement.get(replacement_field) if isinstance(replacement, dict) else None
        if not isinstance(expected_device, str) or value.get("deviceId") != expected_device:
            reasons.append(f"{label} capability snapshot DeviceId does not match device replacement evidence")
        else:
            device_ids.append(expected_device)
        if (shader_workload_error or not isinstance(shader_workload, dict)
                or value.get("deviceId") != shader_workload.get("deviceId")):
            reasons.append(f"{label} capability snapshot DeviceId does not match its shader workload")
        if not isinstance(value.get("compute"), bool):
            reasons.append(f"{label} capability snapshot must explicitly report a boolean compute capability")
        elif declaration is not None:
            reasons.extend(f"{label}: {reason}" for reason in capability_snapshot_failures(value, declaration))
    if len(device_ids) == 2 and device_ids[0] == device_ids[1]:
        reasons.append("primary and replacement capability snapshots must identify distinct devices")
    return result(f"{attempt.get('backend')} raw capability snapshot", not reasons, reasons,
                  attempt=attempt.get("id"), files=[str(path) for path in paths])


def measurement_review(attempt: dict[str, Any], closure: Path) -> dict[str, Any]:
    run = closure / str(attempt.get("directory", "")) / "run"
    copy, error = read_json(run / "workload.json")
    reasons: list[str] = []
    if error or not isinstance(copy, dict) or not isinstance(copy.get("timingsNanoseconds"), dict) or not isinstance(copy.get("rhiCounters"), dict):
        reasons.append("copy workload lacks raw timing/counter measurements")
    return result(f"{attempt.get('backend')} raw workload measurements", not reasons, reasons,
                  attempt=attempt.get("id"))


def retained_attempt_artifacts(selected: dict[str, dict[str, Any]], closure: Path) -> dict[str, set[tuple[Path, str]]]:
    """Only raw artifacts retained by the selected attempt may support review."""
    retained: dict[str, set[tuple[Path, str]]] = {}
    for attempt in selected.values():
        identifier = attempt.get("id")
        if not isinstance(identifier, str) or not identifier:
            continue
        directory = closure / str(attempt.get("directory", ""))
        records = []
        for command in attempt.get("commands", []):
            if isinstance(command, dict) and isinstance(command.get("result"), dict):
                records.extend(command["result"].get("raw_logs", []))
                records.extend(command["result"].get("retained_files", []))
        entries: set[tuple[Path, str]] = set()
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                continue
            candidate = under(directory, record["path"])
            if candidate is not None and isinstance(record.get("sha256"), str):
                entries.add((candidate.resolve(), record["sha256"]))
        retained[identifier] = entries
    return retained


def linked_raw_record(value: Any, identifier: str, retained: dict[str, set[tuple[Path, str]]],
                      base: Path) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str) or not isinstance(value.get("sha256"), str):
        return False
    path = Path(value["path"])
    path = path if path.is_absolute() else base / path
    return (path.resolve(), value["sha256"]) in retained.get(identifier, set()) and sha256(path) == value["sha256"]


def attempt_raw_record(path: Path, identifier: str, retained: dict[str, set[tuple[Path, str]]]) -> dict[str, str] | None:
    digest = sha256(path)
    if digest is None or (path.resolve(), digest) not in retained.get(identifier, set()):
        return None
    return {"path": str(path), "sha256": digest}


def required_review_records(identifier: str, retained: dict[str, set[tuple[Path, str]]],
                            root: Path, relative_paths: list[str]) -> list[dict[str, str]] | None:
    records = [attempt_raw_record(root / relative, identifier, retained) for relative in relative_paths]
    return records if all(record is not None for record in records) else None


def visual_entry_failures(entry: dict[str, Any], identifier: str,
                          retained: dict[str, set[tuple[Path, str]]], root: Path, base: Path) -> list[str]:
    surface = root / "run" / "surface"
    screenshots = [surface / f"surface-client-frame-{frame}.bmp" for frame in range(3)]
    frames = entry.get("frames")
    unexpected = [path.name for path in surface.glob("surface-client-frame-*.bmp")
                  if path.name not in {screenshot.name for screenshot in screenshots}]
    if unexpected:
        return [f"visual evidence has unsupported extra surface screenshot(s) for {identifier}: {unexpected}"]
    if (not all(screenshot.is_file() for screenshot in screenshots)
            or not isinstance(frames, list) or len(frames) != len(screenshots)):
        return [f"visual evidence does not cover every retained surface screenshot for {identifier}"]
    reasons: list[str] = []
    expected = {item.name for item in screenshots}
    seen: set[str] = set()
    for frame in frames:
        if not isinstance(frame, dict) or frame.get("human_verdict") != "passed":
            reasons.append(f"visual evidence has no passing human verdict for {identifier}")
            continue
        screenshot, capture, verdict = frame.get("screenshot"), frame.get("capture"), frame.get("verdict")
        if not all(linked_raw_record(item, identifier, retained, base) for item in (screenshot, capture, verdict)):
            reasons.append(f"visual evidence is not retained raw evidence for {identifier}")
            continue
        screenshot_path = Path(str(screenshot["path"])); screenshot_path = screenshot_path if screenshot_path.is_absolute() else base / screenshot_path
        if screenshot_path.name not in expected or screenshot_path.name in seen:
            reasons.append(f"visual evidence has an unknown or duplicate screenshot for {identifier}")
            continue
        capture_value, capture_error = read_json(Path(str(capture["path"])) if Path(str(capture["path"])).is_absolute() else base / str(capture["path"]))
        verdict_value, verdict_error = read_json(Path(str(verdict["path"])) if Path(str(verdict["path"])).is_absolute() else base / str(verdict["path"]))
        if (capture_error or verdict_error or not isinstance(capture_value, dict) or not isinstance(verdict_value, dict)
                or capture_value.get("image") != screenshot_path.name or capture_value.get("automaticVisualPass") is not False
                or verdict_value.get("image") != screenshot_path.name or verdict_value.get("status") != "passed"):
            reasons.append(f"visual evidence does not match the retained capture scenario for {identifier}")
        seen.add(screenshot_path.name)
    if seen != expected:
        reasons.append(f"visual evidence omitted a retained surface screenshot for {identifier}")
    return reasons


def scenario_entry_failures(entry: dict[str, Any], identifier: str,
                            retained: dict[str, set[tuple[Path, str]]], root: Path,
                            base: Path, scenario: str, paths: list[str]) -> list[str]:
    if entry.get("scenario") != scenario:
        return [f"{scenario} review has the wrong scenario label for {identifier}"]
    if entry.get("verdict") != "passed":
        return [f"{scenario} review has no explicit passing verdict for {identifier}"]
    required = required_review_records(identifier, retained, root, paths)
    raw = entry.get("raw_evidence")
    if required is None or not isinstance(raw, list):
        return [f"{scenario} review lacks retained mandatory raw evidence for {identifier}"]
    required_pairs = {(Path(item["path"]).resolve(), item["sha256"]) for item in required}
    supplied_pairs = set()
    for item in raw:
        if not isinstance(item, dict) or not linked_raw_record(item, identifier, retained, base):
            continue
        candidate = Path(item["path"])
        supplied_pairs.add(((candidate if candidate.is_absolute() else base / candidate).resolve(), item["sha256"]))
    return [] if required_pairs <= supplied_pairs else [f"{scenario} review substitutes or omits mandatory raw evidence for {identifier}"]


def external_review(path: Path | None, attempts: set[str], kind: str,
                    selected: dict[str, dict[str, Any]] | None = None,
                    closure: Path | None = None) -> dict[str, Any]:
    if path is None:
        return result(kind, False, ["explicit review file was not supplied"])
    value, error = read_json(path)
    if error or not isinstance(value, dict):
        return result(kind, False, [f"invalid review file: {error}"])
    reasons: list[str] = []
    if value.get("status") != "reviewed":
        reasons.append("review status is not 'reviewed'")
    if set(value.get("attempt_ids", [])) != attempts:
        reasons.append("review does not name exactly the selected attempts")
    if selected is None or closure is None:
        reasons.append("review was not linked to selected attempt records")
        return result(kind, False, reasons, path=str(path))
    selected_by_id = {attempt.get("id"): attempt for attempt in selected.values()
                      if isinstance(attempt, dict) and isinstance(attempt.get("id"), str)}
    retained = retained_attempt_artifacts(selected_by_id, closure)
    entries = value.get("evidence")
    if not isinstance(entries, list) or {item.get("attempt_id") for item in entries if isinstance(item, dict)} != attempts:
        reasons.append("review lacks one typed evidence entry for each selected attempt")
        entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            reasons.append("review evidence entry is malformed")
            continue
        identifier = entry.get("attempt_id")
        if not isinstance(identifier, str):
            reasons.append("review evidence has no attempt identifier")
            continue
        attempt = selected_by_id.get(identifier)
        if not isinstance(attempt, dict):
            reasons.append("review evidence refers to an unselected attempt")
            continue
        attempt_root = closure / str(attempt.get("directory", ""))
        if kind == "reviewed visual output":
            reasons.extend(visual_entry_failures(entry, identifier, retained, attempt_root, path.parent))
        elif kind == "linked focused test evidence":
            log = entry.get("log")
            command_index = entry.get("command_index")
            commands = attempt.get("commands")
            command = (commands[command_index] if isinstance(commands, list) and type(command_index) is int
                       and 0 <= command_index < len(commands) else None)
            command_result = command.get("result") if isinstance(command, dict) else None
            test_logs = command_result.get("raw_logs") if isinstance(command_result, dict) else None
            if (not isinstance(command, dict) or command.get("kind") != "test"
                    or not isinstance(command.get("command"), list) or not command["command"]
                    or not isinstance(command_result, dict) or command_result.get("exit_status") != 0
                    or command_result.get("test_command_recognized") is not True
                    or type(command_result.get("executed")) is not int or command_result["executed"] <= 0
                    or not isinstance(test_logs, list) or log not in test_logs
                    or not linked_raw_record(log, identifier, retained, path.parent)):
                reasons.append(f"test evidence lacks a retained command, execution count, or log for {identifier}")
            else:
                log_path = Path(log["path"]); log_path = log_path if log_path.is_absolute() else path.parent / log_path
                if not log_path.read_text(encoding="utf-8", errors="replace").strip():
                    reasons.append(f"test evidence has an empty retained log for {identifier}")
        elif kind == "reviewed native failure, loss, and device-replacement evidence":
            reasons.extend(scenario_entry_failures(
                entry, identifier, retained, attempt_root, path.parent, "device-replacement",
                ["run/device-replacement.json", "run/device-replacement-input.bin",
                 "run/device-replacement-expected.json", "run/device-replacement-readback.rgba"]))
        else:
            paths = []
            for directory in ("read-set", "s2-texture", "s3-material", "s3-material/replacement"):
                paths.append(f"run/{directory}/read-set-workload.json")
                for index in range(3):
                    paths.extend(f"run/{directory}/read-set-{kind}-{index}.{suffix}"
                                 for kind, suffix in (("input", "bin"), ("expected", "rgba"), ("readback", "rgba")))
            reasons.extend(scenario_entry_failures(entry, identifier, retained, attempt_root, path.parent,
                                                   "read-set", paths))
    return result(kind, not reasons, reasons, path=str(path), entries=len(entries))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--dx12-attempt", required=True)
    parser.add_argument("--wgl-attempt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path)
    parser.add_argument("--test-evidence", type=Path, action="append", default=[])
    parser.add_argument("--fault-review", type=Path,
                        help="reviewed native failure/loss/device-replacement evidence")
    parser.add_argument("--readset-review", type=Path,
                        help="reviewed material-like read-set and alias/import evidence")
    args = parser.parse_args()
    manifest_path = args.closure / "manifest.json"
    manifest, error = read_json(manifest_path)
    if error or not isinstance(manifest, dict):
        print(f"cannot read closure manifest: {error}", file=sys.stderr)
        return 2
    checks: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for backend, identifier in (("dx12", args.dx12_attempt), ("native-gl-wgl", args.wgl_attempt)):
        attempt, reasons = locate_attempt(manifest, identifier, backend)
        checks.append(result(f"explicit {backend} attempt selection", attempt is not None, reasons, attempt=identifier))
        if attempt is not None:
            selected[backend] = attempt
            checks.extend(attempt_review(attempt, args.closure, ROOT))
    for attempt in selected.values():
        checks.append(diagnostics_review(attempt, args.closure))
        checks.append(capability_review(attempt, args.closure))
        checks.append(measurement_review(attempt, args.closure))
    checks.append(external_review(args.visual_review, set(selected_identifier for selected_identifier in (args.dx12_attempt, args.wgl_attempt)), "reviewed visual output", selected, args.closure))
    test_reviews = [external_review(path, set((args.dx12_attempt, args.wgl_attempt)), "linked focused test evidence", selected, args.closure) for path in args.test_evidence]
    if not test_reviews:
        checks.append(result("linked focused test evidence", False, ["no test evidence links supplied"], count=0))
    else:
        checks.extend(test_reviews)
    identifiers = {args.dx12_attempt, args.wgl_attempt}
    checks.append(external_review(args.fault_review, identifiers,
                                  "reviewed native failure, loss, and device-replacement evidence", selected, args.closure))
    checks.append(external_review(args.readset_review, identifiers,
                                  "reviewed read-set, alias, import, and plan-reuse evidence", selected, args.closure))
    evidence_complete = bool(checks) and all(item["status"] == "passed" for item in checks)
    report = {"schema_version": 1, "scope": "S0-S3 bounded integration evidence review",
              "selected_attempts": {key: value.get("id") for key, value in selected.items()},
              "checks": checks,
              "evidence_status": "all_required_evidence_present" if evidence_complete else "open",
              "historical_foundation_support_decision": "not_consulted_for_current_evidence",
              "stage_support_decision": "not_made_by_this_tool",
              "stage_closure_decision": "not_made_by_this_tool"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if evidence_complete else 1


if __name__ == "__main__":
    sys.exit(main())
