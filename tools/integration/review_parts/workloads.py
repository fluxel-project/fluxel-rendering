from __future__ import annotations

from .common import *
from .raster import *
from .shader import *

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
