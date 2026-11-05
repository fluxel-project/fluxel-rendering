"""A complete, hash-bound S5 ledger used by the reviewer CLI tests."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import zipfile


def _write(path: Path, value: bytes | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    else:
        path.write_bytes(value)


def _record(review, path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": review._REVIEW.sha256(path) or ""}


def _source_tree(root: Path, review) -> list[dict[str, str]]:
    for relative in (
        "Cargo.toml", "Cargo.lock", "build.rs", "tools/integration/README.md",
        "documents/s0-s1-support-decision.md", "documents/integration-known-diagnostics.md",
        "documents/integration-manifest.schema.json",
    ):
        _write(root / relative, b"fixture\n")
    inputs = []
    for path in review._SMOKE.source_input_paths(root):
        inputs.append({"path": path.relative_to(root.resolve()).as_posix(), "sha256": review._REVIEW.sha256(path)})
    return inputs


def _snapshot(closure: Path, inputs: list[dict[str, str]], root: Path, review) -> dict[str, object]:
    archive = closure / "source-inputs.zip"
    with zipfile.ZipFile(archive, "x") as output:
        for item in inputs:
            output.writestr(item["path"], (root / item["path"]).read_bytes())
        output.writestr("snapshot-inputs.json", json.dumps(inputs))
    return {"path": archive.name, "sha256": review._REVIEW.sha256(archive),
            "files": len(inputs), "kind": "exact_worktree_inputs"}


def _bmp(width: int, height: int) -> bytes:
    pixels = bytearray(width * height * 4)
    x, y = width * 8 // 20, height * 57 // 100
    pixels[(y * width + x) * 4:(y * width + x + 1) * 4] = b"\0\0\xff\xff"
    header = b"BM" + bytes(16) + struct.pack("<ii", width, -height) + bytes(28)
    return header + pixels


def _diagnostic(backend: str) -> dict:
    if backend == "dx12":
        return {"requested": True, "deviceOpenCompleted": True,
                "activationContract": f"EnableDebugLayer {__import__('smoke_evidence').RHI_REVISION}"}
    return {"logSaturated": False, "contextVersion": "4.6", "coreDebugOutput": True,
            "khrDebugExtension": True, "debugOutputAvailable": True, "installed": True,
            "activationMarkerObserved": True,
            "activationMessages": ["fluxel-rendering WGL diagnostic observer active"],
            "nativeDiagnostics": []}


def _forward(run: Path, review) -> None:
    base, mask = review.fixture_oracle(scenario="base")
    for name in review._SMOKE.s5_static_asset_raw_artifacts():
        _write(run / name, b"input")
    _write(run / "s5-forward-workload.json", {"workload": "s5-scene-forward-four-layouts"})
    _write(run / "s5-forward-oracle.json", {"mask": "s5-forward-stable-mask.bin", "expected": "s5-forward-expected.rgba", "actual": "s5-forward-readback.rgba"})
    _write(run / "s5-forward-expected.rgba", base); _write(run / "s5-forward-readback.rgba", base)
    _write(run / "s5-forward-stable-mask.bin", mask)
    _write(run / "s5-forward-measurements.json", dict(
        schemaVersion=1, preparationBoundary="accepted preparation elapsed time",
        pendingPeak=3, **{key: 1 for key in (
            "preparationElapsedNs", "graphCompileElapsedNs", "acceptedBufferAllocations",
            "acceptedBufferAllocationBytes", "acceptedTextureAllocations", "residentStaticUploadBytes",
            "frameUploadBytes", "geometryCacheHits", "textureCacheHits", "frameAllocationReuses", "planCompilations")}
    ))
    _write(run / "s5-forward-cache-facts.json", {"residentCopyDelta": 0, "geometryHitDelta": 10,
           "textureHitDelta": 10, "frameReuseDelta": 1, "opaqueCandidateDelta": 10,
           "opaqueBatchDelta": 10, "opaqueDrawDelta": 10, "pending": 3,
           "warmPlanCompilations": 2, "freshStateCompilations": 3, "finalCompilations": 3})
    for frame in range(3):
        expected, _ = review.fixture_oracle(scenario=f"frame-{frame}")
        _write(run / f"s5-forward-frame-{frame}-input.json", {
            "sceneVersion": frame + 1,
            "firstMeshClipTranslationZ": 0 if frame == 0 else 2,
            "secondMeshTint": [0] * 4 if frame == 2 else [1] * 4,
        })
        _write(run / f"s5-forward-frame-{frame}-expected.rgba", expected)
        _write(run / f"s5-forward-frame-{frame}-actual.rgba", expected)
    empty = bytes(96 * 96 * 4)
    for name in ("s5-forward-empty-expected.rgba", "s5-forward-empty-actual.rgba", "s5-forward-external-clear-actual.rgba"):
        _write(run / name, empty)
    for generation in ("old", "new"):
        expected, _ = review.fixture_oracle(scenario=generation)
        generation_number = 1 if generation == "old" else 2
        _write(run / f"s5-forward-replacement-{generation}-input.json", {
            kind: {"slot": index, "generation": generation_number}
            for index, kind in enumerate(("geometry", "material", "texture"))
        })
        _write(run / f"s5-forward-replacement-{generation}-expected.rgba", expected)
        _write(run / f"s5-forward-replacement-{generation}-actual.rgba", expected)


def _device(run: Path, review) -> None:
    expected, _ = review.fixture_oracle(scenario="base")
    for name in review._SMOKE.s5_static_asset_raw_artifacts(): _write(run / name, b"input")
    _write(run / "s5-device-isolation-workload.json", {"workload": "s5-forward-two-device-cache-isolation", "sharedAssets": "s5-device-isolation-shared-assets.json", "oldReuseReadback": "s5-device-isolation-old-reuse-actual.rgba"})
    for name in ("s5-device-isolation-expected.rgba", "s5-device-isolation-old-actual.rgba", "s5-device-isolation-old-reuse-actual.rgba", "s5-device-isolation-new-actual.rgba"): _write(run / name, expected)
    _write(run / "s5-device-isolation.json", {"oldDeviceId": "old", "newDeviceId": "new", "sharedLogicalAssetGenerations": True, "oldCompletedProductsRetained": True, "newDeviceResidentCopyBytes": 1, "oldReuseResidentCopyBytes": 0, "newDeviceGeometryHits": 0, "newDeviceTextureHits": 0, "oldReuseGeometryHits": 4, "oldReuseTextureHits": 4})
    _write(run / "s5-device-isolation-shared-assets.json", {"shared": True})


def _surface(run: Path, review) -> None:
    _write(run / "s5-forward-surface-workload.json", {"workload": "s5-forward-surface-resize", "initialExtent": [320, 180], "resizedExtent": [256, 144], "presentedFrames": 2, "completionBeforeReconfigure": True, "visualEvidence": "native-client-capture", "hostResizeReconfiguredBeforeAcquire": True})
    _write(run / "forward-surface-geometry.bin", b"geometry"); _write(run / "forward-surface-texture.rgba", b"texture")
    _write(run / "forward-surface-expected.json", {"triangleRgba": [255, 0, 0, 255], "backgroundRgb": [0, 0, 0], "nativeCapture": True})
    _write(run / "surface-events.jsonl", b'{"phase":"forward-resize","requestedExtent":[256,144],"clientExtent":[256,144],"selectedEvent":"resized"}\n')
    _write(run / "surface-configuration.jsonl", b'{"extent":[320,180],"configurationSupport":true,"requiredUsage":"render_attachment"}\n{"extent":[256,144],"configurationSupport":true,"requiredUsage":"render_attachment"}\n')
    for frame, extent in enumerate(((320, 180), (256, 144))):
        image = f"surface-client-frame-{frame}.bmp"
        _write(run / image, _bmp(*extent))
        _write(run / f"surface-client-frame-{frame}.capture.json", {"schemaVersion": 1, "frame": frame, "clientExtent": list(extent), "pixelFormat": "BGRA8-top-down", "image": image, "automaticVisualPass": False})
        _write(run / f"surface-client-frame-{frame}.verdict.json", {"schemaVersion": 1, "frame": frame, "image": image, "oracle": "desktop-capture-corners-and-stable-triangle-interior", "status": "passed"})


def build(root: Path, review) -> dict[str, object]:
    """Create all six manifests, raw files, source archive, and human review."""
    source = root / "source"; inputs = _source_tree(source, review); attempts = []; args = []; human = []
    for backend in ("dx12", "native-gl-wgl"):
        for workload, populate in (("s5-forward", _forward), ("s5-device-isolation", _device), ("s5-surface", _surface)):
            closure = root / f"{backend}-{workload}"; run = closure / "attempt" / "run" / ({"s5-forward": "s5-forward", "s5-device-isolation": "s5-forward-device-isolation", "s5-surface": "s5-forward-surface"}[workload])
            populate(run, review)
            marker = run.parent / ("dx12-debug-request.json" if backend == "dx12" else "wgl-debug-observer.json"); _write(marker, _diagnostic(backend))
            executable = closure / "attempt" / "consumer.exe"; _write(executable, b"fixture executable")
            lock = closure / "attempt" / "Cargo.lock"; _write(lock, b"fixture lock\n")
            logs = []
            for kind, text in (("build", "build complete\n"), ("direct", "runner complete\n"), ("test", "Ran 1 test in 0.001s\nOK\n")):
                log = closure / "attempt" / "logs" / f"{kind}.log"; _write(log, text.encode("utf-8")); logs.append(_record(review, log))
            retained = [_record(review, path) for path in run.rglob("*") if path.is_file()]
            identifier = f"{backend}-{workload}"; validation = {"status": "observed", "native_diagnostic_count": 0, "diagnostics": {"unreviewed_native_diagnostics": []}, "artifact": _record(review, marker)}
            success = {"exit_status": 0, "timed_out": False, "child_process_lifecycle": "parent_exited"}
            attempt = {"id": identifier, "directory": "attempt", "mode": "run", "backend": backend, "runner_exit_status": 0, "workload": review._SMOKE.s5_workload_contract(workload), "source_inputs": inputs, "source_snapshot": _snapshot(closure, inputs, source, review), "built_source_verification": {"status": "verified"}, "built_git_checkout_verification": {"status": "verified"}, "source_consistency_after_attempt": {"status": "stable"}, "source_consistency_after_build": {"status": "stable"}, "build_environment": {"status": "auditable"}, "executable": {**_record(review, executable), "after_run_sha256": review._REVIEW.sha256(executable)}, "locks": [_record(review, lock)], "validation": validation, "retained_attempt_files": retained, "commands": [{"kind": "build", "command": ["cargo", "build"], "result": {**success, "raw_logs": [logs[0]]}}, {"kind": "direct_executable", "command": [str(executable), review._SMOKE.s5_workload_contract(workload)["entry_argument"]], "result": {**success, "raw_logs": [logs[1]], "nested_workloads": {"status": "verified"}, "retained_files": retained}}, {"kind": "test", "command": ["python", "-m", "unittest", "fixture"], "result": {**success, "raw_logs": [logs[2]], "test_command_recognized": True, "executed": 1}}]}
            _write(closure / "manifest.json", {"schema_version": 4, "stage": "S5 Forward integration evidence", "attempts": [attempt]})
            args.extend(["--attempt", str(closure), identifier, backend, workload]); attempts.append((attempt, closure, run))
            if workload == "s5-surface":
                frames = []
                for frame in range(2):
                    frames.append({key: _record(review, run / f"surface-client-frame-{frame}.{suffix}") for key, suffix in (("screenshot", "bmp"), ("capture", "capture.json"), ("verdict", "verdict.json"))} | {"human_verdict": "passed"})
                human.append({"attempt_id": identifier, "frames": frames})
    human_path = root / "human-review.json"; _write(human_path, {"status": "reviewed", "attempt_ids": [entry["attempt_id"] for entry in human], "evidence": human})
    return {"args": args, "human": human_path, "output": root / "report.json", "attempts": attempts, "source": source}
