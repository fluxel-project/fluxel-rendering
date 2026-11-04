from __future__ import annotations

from .common import *
from .raster import *

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
