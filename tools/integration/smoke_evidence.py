#!/usr/bin/env python3
"""Workload, validation, Cargo-build, and package evidence helpers."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tomllib
from typing import Any
import zipfile
if __package__:
    from .smoke_inputs import git_value, sha256
    from . import capture_contract
else:
    from smoke_inputs import git_value, sha256
    import capture_contract
ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ("dx12", "native-gl-wgl")
RHI_REVISION = "b5d569feace548a6ed498901deaf31a5efcaca61"
def s5_static_asset_raw_artifacts() -> tuple[str, ...]:
    """Return the immutable mesh, material, and texture inputs for S5."""
    artifacts = [
        "s5-forward-geometry-0.txt",
        "s5-forward-geometry-1.txt",
        "s5-forward-geometry-2.txt",
        "s5-forward-geometry-3.txt",
    ]
    artifacts.extend(f"s5-forward-geometry-0-stream-{stream}.bin" for stream in range(1))
    artifacts.extend(f"s5-forward-geometry-1-stream-{stream}.bin" for stream in range(3))
    artifacts.extend(f"s5-forward-geometry-2-stream-{stream}.bin" for stream in range(2))
    artifacts.extend(f"s5-forward-geometry-3-stream-{stream}.bin" for stream in range(1))
    artifacts.extend(f"s5-forward-material-{index}.txt" for index in range(4))
    artifacts.extend(f"s5-forward-texture-{index}.rgba" for index in range(4))
    return tuple(artifacts)


def s5_required_raw_artifacts() -> tuple[str, ...]:
    """Return every raw input, independent expectation, and readback S5 needs."""
    artifacts = [
        "s5-forward-expected.rgba",
        "s5-forward-readback.rgba",
        "s5-forward-stable-mask.bin",
        "s5-forward-oracle.json",
        "s5-forward-cache-facts.json",
        "s5-forward-measurements.json",
        *s5_static_asset_raw_artifacts(),
    ]
    for frame in range(3):
        artifacts.extend((
            f"s5-forward-frame-{frame}-input.json",
            f"s5-forward-frame-{frame}-expected.rgba",
            f"s5-forward-frame-{frame}-actual.rgba",
        ))
    artifacts.extend((
        "s5-forward-empty-expected.rgba",
        "s5-forward-empty-actual.rgba",
    ))
    for generation in ("old", "new"):
        artifacts.extend((
            f"s5-forward-replacement-{generation}-input.json",
            f"s5-forward-replacement-{generation}-expected.rgba",
            f"s5-forward-replacement-{generation}-actual.rgba",
        ))
    return tuple(artifacts)


def s5_device_isolation_required_raw_artifacts() -> tuple[str, ...]:
    """Return the shared inputs and both-device observations for isolation."""
    artifacts = list(s5_static_asset_raw_artifacts())
    artifacts.extend((
        "s5-device-isolation-expected.rgba",
        "s5-device-isolation-old-actual.rgba",
        "s5-device-isolation-old-reuse-actual.rgba",
        "s5-device-isolation-new-actual.rgba",
        "s5-device-isolation.json",
        "s5-device-isolation-shared-assets.json",
    ))
    return tuple(artifacts)


def s5_surface_required_raw_artifacts() -> tuple[str, ...]:
    """Return the native inputs, captures, and Host lifecycle observations."""
    artifacts = [
        "forward-surface-geometry.bin",
        "forward-surface-texture.rgba",
        "forward-surface-expected.json",
        "surface-events.jsonl",
        "surface-configuration.jsonl",
    ]
    for frame in range(2):
        artifacts.extend((
            f"surface-client-frame-{frame}.bmp",
            f"surface-client-frame-{frame}.capture.json",
            f"surface-client-frame-{frame}.verdict.json",
        ))
    return tuple(artifacts)


def s5_workload_contract(workload: str) -> dict[str, Any]:
    """Describe the selected S5 executable contract before a run begins."""
    if workload == "s5-forward":
        nested = "s5-forward/s5-forward-workload.json"
        raw = s5_required_raw_artifacts()
        return {
            "selection": "s5-scene-forward-four-layouts",
            "entry_argument": "--s5-only",
            "required_nested_workload": nested,
            "required_raw_artifacts": [f"s5-forward/{artifact}" for artifact in raw],
        }
    if workload == "s5-device-isolation":
        nested = "s5-forward-device-isolation/s5-device-isolation-workload.json"
        raw = s5_device_isolation_required_raw_artifacts()
        return {
            "selection": "s5-forward-two-device-cache-isolation",
            "entry_argument": "--s5-device-only",
            "required_nested_workload": nested,
            "required_raw_artifacts": [f"s5-forward-device-isolation/{artifact}" for artifact in raw],
        }
    if workload == "s5-surface":
        nested = "s5-forward-surface/s5-forward-surface-workload.json"
        raw = s5_surface_required_raw_artifacts()
        return {
            "selection": "s5-forward-surface-resize",
            "entry_argument": "--s5-surface-only",
            "required_nested_workload": nested,
            "required_raw_artifacts": [f"s5-forward-surface/{artifact}" for artifact in raw],
        }
    raise ValueError(f"not an S5 workload: {workload}")


def nested_workload_evidence(run_directory: Path, workload: str) -> dict[str, Any]:
    """Retain and validate every workload JSON below one executable run root."""
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in sorted(run_directory.rglob("*workload.json")):
        relative = path.relative_to(run_directory).as_posix()
        record: dict[str, Any] = {"path": relative, "sha256": sha256(path)}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            record["status"] = "malformed"
            record["reason"] = str(error)
            failures.append(f"malformed nested workload: {relative}")
        else:
            if not isinstance(value, dict) or not isinstance(value.get("workload"), str):
                record["status"] = "malformed"
                record["reason"] = "object has no string workload name"
                failures.append(f"malformed nested workload: {relative}")
            else:
                record["status"] = "retained"
                record["workload"] = value["workload"]
                if "sharedAssets" in value:
                    record["sharedAssets"] = value["sharedAssets"]
                if "oldReuseReadback" in value:
                    record["oldReuseReadback"] = value["oldReuseReadback"]
        records.append(record)

    if workload == "s5-forward":
        expected = "s5-forward/s5-forward-workload.json"
        s5 = next((record for record in records if record["path"] == expected), None)
        if s5 is None:
            failures.append(f"missing required nested workload: {expected}")
        elif s5.get("workload") != "s5-scene-forward-four-layouts":
            failures.append("S5 Forward nested workload has an unexpected identity")
        for artifact in s5_required_raw_artifacts():
            name = f"s5-forward/{artifact}"
            if not (run_directory / name).is_file():
                failures.append(f"missing required S5 raw artifact: {name}")
    elif workload == "s5-device-isolation":
        expected = "s5-forward-device-isolation/s5-device-isolation-workload.json"
        isolated = next((record for record in records if record["path"] == expected), None)
        if isolated is None:
            failures.append(f"missing required nested workload: {expected}")
        elif isolated.get("workload") != "s5-forward-two-device-cache-isolation":
            failures.append("S5 device isolation nested workload has an unexpected identity")
        elif (
            isolated.get("sharedAssets") != "s5-device-isolation-shared-assets.json"
            or isolated.get("oldReuseReadback") != "s5-device-isolation-old-reuse-actual.rgba"
        ):
            failures.append("S5 device isolation nested workload lacks shared-asset or old-reuse facts")
        for artifact in s5_device_isolation_required_raw_artifacts():
            name = f"s5-forward-device-isolation/{artifact}"
            if not (run_directory / name).is_file():
                failures.append(f"missing required S5 device-isolation raw artifact: {name}")
    elif workload == "s5-surface":
        surface_directory = run_directory / "s5-forward-surface"
        expected = "s5-forward-surface/s5-forward-surface-workload.json"
        surface = next((record for record in records if record["path"] == expected), None)
        if surface is None:
            failures.append(f"missing required nested workload: {expected}")
        elif surface.get("workload") != "s5-forward-surface-resize":
            failures.append("S5 surface nested workload has an unexpected identity")
        else:
            try:
                workload_value = json.loads((run_directory / expected).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                failures.append(f"malformed S5 surface workload facts: {error}")
            else:
                if (
                    workload_value.get("initialExtent") != [320, 180]
                    or workload_value.get("resizedExtent") != [256, 144]
                    or workload_value.get("presentedFrames") != 2
                    or workload_value.get("completionBeforeReconfigure") is not True
                    or workload_value.get("visualEvidence") != "native-client-capture"
                ):
                    failures.append("S5 surface workload lacks resize, completion, or visual-capture facts")
        for artifact in s5_surface_required_raw_artifacts():
            name = f"s5-forward-surface/{artifact}"
            if not (run_directory / name).is_file():
                failures.append(f"missing required S5 surface raw artifact: {name}")
        if not failures:
            try:
                expected_pixels = json.loads((surface_directory / "forward-surface-expected.json").read_text(encoding="utf-8"))
                if expected_pixels != {
                    "triangleRgba": [255, 0, 0, 255],
                    "backgroundRgb": [0, 0, 0],
                    "nativeCapture": True,
                }:
                    failures.append("S5 surface independent expected pixels are malformed")
                events = [json.loads(line) for line in (surface_directory / "surface-events.jsonl").read_text(encoding="utf-8").splitlines() if line]
                if not any(
                    event.get("phase") == "forward-resize"
                    and event.get("requestedExtent") == [256, 144]
                    and event.get("clientExtent") == [256, 144]
                    and event.get("selectedEvent") == "resized"
                    for event in events
                ):
                    failures.append("S5 surface lacks a confirmed Host resize observation")
                configurations = [json.loads(line) for line in (surface_directory / "surface-configuration.jsonl").read_text(encoding="utf-8").splitlines() if line]
                if [configuration.get("extent") for configuration in configurations] != [[320, 180], [256, 144]] or not all(
                    configuration.get("configurationSupport") is True
                    and configuration.get("requiredUsage") == "render_attachment"
                    for configuration in configurations
                ):
                    failures.append("S5 surface configuration evidence does not cover both extents")
                for frame, extent in enumerate(([320, 180], [256, 144])):
                    capture = json.loads((surface_directory / f"surface-client-frame-{frame}.capture.json").read_text(encoding="utf-8"))
                    verdict = json.loads((surface_directory / f"surface-client-frame-{frame}.verdict.json").read_text(encoding="utf-8"))
                    image = f"surface-client-frame-{frame}.bmp"
                    if (capture_contract.capture_failures(capture, frame, extent, image)
                            or capture_contract.verdict_failures(
                                verdict, frame, image, capture.get("schemaVersion"))):
                        failures.append(f"S5 surface capture verdict for frame {frame} is incomplete or failed")
            except (OSError, json.JSONDecodeError) as error:
                failures.append(f"malformed S5 surface raw evidence: {error}")
    return {"status": "verified" if not failures else "incomplete", "records": records,
            "failures": failures}


def freeze_source_inputs(inputs: list[dict[str, Any]], destination: Path,
                         root: Path = ROOT) -> dict[str, Any]:
    """Retain the exact dirty-worktree inputs, refusing a capture-time edit."""
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in inputs:
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"source snapshot path is outside root: {relative}")
            data = (root / relative).read_bytes()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"source changed while freezing: {relative}")
            archive.writestr(relative.as_posix(), data)
        archive.writestr("snapshot-inputs.json", json.dumps(inputs, sort_keys=True, indent=2))
    return {"path": str(destination), "sha256": sha256(destination),
            "files": len(inputs), "kind": "exact_worktree_inputs"}


def validation_evidence(backend: str, run_directory: Path, requested: str) -> dict[str, Any]:
    """Classify only concrete diagnostics artifacts from this attempt."""
    if requested != "enabled":
        return {"requested": requested, "status": "disabled", "reason": "runner disabled validation"}
    if backend == "dx12":
        marker = run_directory / "dx12-debug-request.json"
        if not marker.is_file():
            return {"requested": requested, "status": "unverified", "reason": "DX12 device-open diagnostic marker is absent"}
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"requested": requested, "status": "unverified", "reason": "DX12 diagnostic marker is malformed"}
        contract = record.get("activationContract")
        if (record.get("requested") is True and record.get("deviceOpenCompleted") is True
                and isinstance(contract, str) and RHI_REVISION in contract
                and "EnableDebugLayer" in contract):
            return {"requested": requested, "status": "observed", "reason": "the selected DX12 device opened after the pinned RHI debug-layer activation branch", "artifact": {"path": str(marker), "sha256": sha256(marker)}, "activation_contract": record.get("activationContract")}
        return {"requested": requested, "status": "unverified", "reason": "DX12 device-open marker did not record a debug request"}
    markers = sorted(run_directory.rglob("wgl-debug-observer.json"))
    primary = run_directory / "wgl-debug-observer.json"
    if primary not in markers:
        return {"requested": requested, "status": "unverified", "reason": "WGL KHR_debug observer artifact is absent"}
    diagnostic_count = 0
    for marker in markers:
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"requested": requested, "status": "unverified", "reason": f"WGL observer artifact is malformed: {marker}"}
        messages = record.get("activationMessages", [])
        marker_returned = isinstance(messages, list) and any(
            isinstance(message, str) and "fluxel-rendering WGL diagnostic observer active" in message
            for message in messages
        )
        if not (record.get("installed") is True and record.get("activationMarkerObserved") is True
                and marker_returned and record.get("logSaturated") is False
                and isinstance(record.get("nativeDiagnostics"), list)):
            return {"requested": requested, "status": "unverified", "reason": f"WGL observer did not activate: {marker}"}
        diagnostic_count += len(record["nativeDiagnostics"])
    return {"requested": requested, "status": "observed", "reason": "the exact WGL contexts returned their observer application markers",
            "artifact": {"path": str(primary), "sha256": sha256(primary)},
            "additional_artifacts": [{"path": str(marker), "sha256": sha256(marker)} for marker in markers if marker != primary],
            "native_diagnostic_count": diagnostic_count}


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolved_git_packages(lock: Path) -> list[dict[str, str]]:
    """Cargo.lock, rather than a neighbouring checkout, identifies Git sources."""
    if not lock.is_file():
        return []
    result = []
    for package in tomllib.loads(lock.read_text(encoding="utf-8")).get("package", []):
        source = package.get("source")
        if isinstance(source, str) and source.startswith("git+"):
            result.append({"name": package["name"], "version": package["version"], "source": source})
    return result


def find_built_executable(build_stdout: Path, manifest: Path) -> str | None:
    """Extract Cargo's actual binary path, including CARGO_TARGET_DIR overrides."""
    package_name = tomllib.loads(manifest.read_text(encoding="utf-8"))["package"]["name"]
    executable = None
    for line in build_stdout.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") != "compiler-artifact":
            continue
        reported_manifest = message.get("manifest_path")
        if (
            message.get("executable")
            and reported_manifest
            and Path(reported_manifest).resolve() == manifest.resolve()
            and message.get("target", {}).get("name") == package_name
            and "bin" in message.get("target", {}).get("kind", [])
        ):
            executable = message["executable"]
    return executable


def retain_executable(source: Path, attempt_directory: Path) -> dict[str, Any]:
    """Keep the executed binary after another backend overwrites Cargo's output."""
    digest = sha256(source)
    if digest is None:
        raise ValueError(f"Cargo executable is missing: {source}")
    destination = attempt_directory / ("consumer" + source.suffix)
    shutil.copy2(source, destination)
    if sha256(destination) != digest or sha256(source) != digest:
        raise ValueError("Cargo executable changed while retaining the attempt binary")
    return {"path": str(destination), "sha256": digest, "cargo_reported_path": str(source)}


def built_fluxel_packages(build_stdout: Path) -> list[dict[str, Any]]:
    """Retain Cargo's actual compiled package identities and active features."""
    result = []
    external_manifest = (ROOT / "crates/renderer/examples/external-renderer/Cargo.toml").resolve()
    for line in build_stdout.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        package_id = message.get("package_id", "")
        target = message.get("target", {})
        target_name = target.get("name", "")
        manifest = message.get("manifest_path")
        is_external_consumer = isinstance(manifest, str) and Path(manifest).resolve() == external_manifest
        if message.get("reason") != "compiler-artifact" or not (
            target_name.startswith("fluxel_") or "github.com/fluxel-project/" in package_id
            or is_external_consumer
        ):
            continue
        fragment = package_id.partition("#")[2]
        name = fragment.rpartition("@")[0] if "@" in fragment else target_name.replace("_", "-")
        result.append({
            "name": name, "package_id": package_id,
            "manifest_path": message.get("manifest_path"),
            "target": target_name, "target_kind": target.get("kind", []),
            "features": sorted(message.get("features", [])),
            "fresh": message.get("fresh"), "filenames": message.get("filenames", []),
        })
    return result


def verify_built_sources(inventory: list[dict[str, Any]], locked: list[dict[str, str]],
                         required: set[str] | None = None) -> dict[str, Any]:
    """Reject path overrides of the selected Git dependencies in actual builds."""
    expected: dict[str, set[str]] = {}
    for package in locked:
        expected.setdefault(package["name"], set()).add(package["source"].partition("#")[0])
    mismatches = []
    present = set()
    for package in inventory:
        name = package["name"]
        present.add(name)
        if name in expected and package["package_id"].partition("#")[0] not in expected[name]:
            mismatches.append({"name": name, "actual": package["package_id"], "expected": sorted(expected[name])})
        local_manifests = {
            "external-renderer": ROOT / "crates/renderer/examples/external-renderer/Cargo.toml",
            "fluxel-renderer": ROOT / "crates/renderer/Cargo.toml",
            "fluxel-shader": ROOT / "crates/shader/Cargo.toml",
            "fluxel-material": ROOT / "crates/material/Cargo.toml",
        }
        if name in local_manifests and (
            not package.get("manifest_path")
            or Path(package["manifest_path"]).resolve() != local_manifests[name].resolve()
        ):
            mismatches.append({"name": name, "actual_manifest": package.get("manifest_path"),
                               "reason": f"{name} is outside this source snapshot"})
    if required is None:
        required = {"external-renderer", "fluxel-assets", "fluxel-rhi", "fluxel-renderer", "fluxel-shader", "fluxel-material", "fluxel-host"}
    missing = sorted(required - present)
    return {"status": "verified" if not mismatches and not missing else "mismatch",
            "mismatches": mismatches, "missing_required_packages": missing}


def verify_git_checkouts(
    inventory: list[dict[str, Any]],
    locked: list[dict[str, str]],
    git_lookup=git_value,
) -> dict[str, Any]:
    """Verify the actual Cargo Git trees, not just their package-ID strings."""
    revisions: dict[str, set[str]] = {}
    for package in locked:
        revisions.setdefault(package["name"], set()).add(package["source"].partition("#")[2])
    trees: dict[str, dict[str, Any]] = {}
    mismatches = []
    for package in inventory:
        if package["name"] not in revisions:
            continue
        manifest = package.get("manifest_path")
        root = git_lookup(Path(manifest).parent, "rev-parse", "--show-toplevel") if manifest else None
        if root is None:
            mismatches.append({"name": package["name"], "reason": "built Git checkout is unavailable"})
            continue
        if root not in trees:
            status = git_lookup(Path(root), "status", "--porcelain", "--untracked-files=all")
            # Cargo writes this exact marker after checking out a Git source.
            changes = None if status is None else [line for line in status.splitlines()
                                                    if line != "?? .cargo-ok"]
            trees[root] = {"path": root, "revision": git_lookup(Path(root), "rev-parse", "HEAD"),
                           "changes": changes}
        tree = trees[root]
        if tree["revision"] not in revisions[package["name"]] or tree["changes"] != []:
            mismatches.append({"name": package["name"], "checkout": root,
                               "expected_revisions": sorted(revisions[package["name"]])})
    if not trees:
        mismatches.append({"reason": "no built Git checkout was inspected"})
    return {"status": "verified" if not mismatches else "mismatch",
            "checkouts": list(trees.values()), "mismatches": mismatches}


def required_built_packages(backend: str) -> set[str]:
    """Every selected S0-S3 route creates a Host-owned native window Surface."""
    if backend not in BACKENDS:
        raise ValueError(f"unsupported integration backend: {backend}")
    return {"external-renderer", "fluxel-assets", "fluxel-rhi", "fluxel-renderer", "fluxel-shader", "fluxel-material", "fluxel-host"}
