from __future__ import annotations

from .common import *
from .raster import *
from .shader import *
from .workloads import *
from .common import _SMOKE

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
            if (not isinstance(contract, str) or _SMOKE.RHI_REVISION not in contract
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
