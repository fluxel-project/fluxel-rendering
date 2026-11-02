"""Regression tests for the bounded S0/S1 closure reviewer."""
import importlib.util
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile


SPEC = importlib.util.spec_from_file_location("review", Path(__file__).with_name("review.py"))
assert SPEC and SPEC.loader
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": review.sha256(path) or ""}


CAPABILITY_DECLARATION = """pub struct DeviceCapabilities {
    pub compute: bool,
    pub storage_buffers: bool,
    pub storage_textures: bool,
    pub native_interop: bool,
    pub debug_markers: bool,
    pub multi_queue: bool,
    pub lanes: Arc<[LaneInfo]>,
    pub cooperative_matrix_configurations: Vec<CooperativeMatrixConfiguration>,
    pub limits: DeviceLimits,
}
pub struct DeviceLimits {
    pub max_buffer_bytes: u64,
    pub max_vertex_buffers: u32,
    pub max_vertex_attributes: u32,
    pub max_color_attachments: u32,
    pub max_compute_workgroups: [u32; 3],
}
pub struct CooperativeMatrixConfiguration {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_type: CooperativeMatrixComponentType,
    pub b_type: CooperativeMatrixComponentType,
    pub c_type: CooperativeMatrixComponentType,
    pub result_type: CooperativeMatrixComponentType,
    pub saturating_accumulation: bool,
    pub scope: CooperativeMatrixScope,
}
pub enum CooperativeMatrixComponentType {
    Float16,
    BFloat16,
}
pub enum CooperativeMatrixScope {
    Subgroup,
    QueueFamily,
}
"""


def capability_fixture(compute: object = True) -> dict:
    return {"adapter": "actual adapter", "deviceId": "DeviceId(1)", "compute": compute,
            "storageBuffers": True, "storageTextures": False, "nativeInterop": True,
            "debugMarkers": True, "multiQueue": False, "lanes": 1,
            "limits": {"maxBufferBytes": 4096, "maxVertexBuffers": 4, "maxVertexAttributes": 8, "maxColorAttachments": 4},
            "deviceCapabilities": {"compute": compute, "storage_buffers": True, "storage_textures": False,
                "native_interop": True, "debug_markers": True, "multi_queue": False,
                "lanes": [{"id": {"device": "DeviceId(1)", "ordinal": 0}, "class": "graphics", "routes": []}],
                "cooperative_matrix_configurations": [{"m": 16, "n": 16, "k": 16, "a_type": "float16", "b_type": "bfloat16",
                    "c_type": "float16", "result_type": "float16", "saturating_accumulation": False, "scope": "queue_family"}],
                "limits": {"max_buffer_bytes": 4096, "max_vertex_buffers": 4, "max_vertex_attributes": 8,
                           "max_color_attachments": 4, "max_compute_workgroups": [64, 64, 64]}}}


class ReviewTests(unittest.TestCase):
    def test_compute_capability_accepts_actual_support_and_requires_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            closure = Path(temporary)
            run = closure / "run"
            run.mkdir()
            declaration = closure / "rhi/crates/core/src/api/foundation.rs"
            declaration.parent.mkdir(parents=True)
            declaration.write_text(CAPABILITY_DECLARATION, encoding="utf-8")
            attempt = {"id": "wgl", "backend": "native-gl-wgl", "directory": ".",
                       "built_git_checkout_verification": {"checkouts": [{"path": str(closure / "rhi")}]}}
            for capability in (True, False, None, "false", 0):
                with self.subTest(capability=capability):
                    (run / "capabilities.json").write_text(json.dumps(capability_fixture(capability)), encoding="utf-8")
                    report = review.capability_review(attempt, closure)
                    self.assertEqual(report["status"], "passed" if isinstance(capability, bool) else "open")

    def test_complete_capabilities_reject_omissions_schema_changes_and_foreign_lanes(self) -> None:
        original = capability_fixture()
        self.assertEqual(review.capability_snapshot_failures(original, CAPABILITY_DECLARATION), [])
        for mutation in ("missing_feature", "missing_limit", "numeric_bool", "foreign_lane", "invalid_route", "wrong_alias", "oversized_limit", "matrix_enum", "matrix_omission"):
            value = copy.deepcopy(original)
            facts = value["deviceCapabilities"]
            if mutation == "missing_feature": del facts["storage_textures"]
            if mutation == "missing_limit": del facts["limits"]["max_buffer_bytes"]
            if mutation == "numeric_bool": facts["storage_buffers"] = 1
            if mutation == "foreign_lane": facts["lanes"][0]["id"]["device"] = "DeviceId(2)"
            if mutation == "invalid_route": facts["lanes"][0]["routes"] = [{"target": {"device": "DeviceId(1)", "ordinal": 9}, "route": "same_queue"}]
            if mutation == "wrong_alias": value["storageBuffers"] = False
            if mutation == "oversized_limit": facts["limits"]["max_compute_workgroups"] = [2**32, 1, 1]
            if mutation == "matrix_enum": facts["cooperative_matrix_configurations"][0]["scope"] = "unknown_scope"
            if mutation == "matrix_omission": del facts["cooperative_matrix_configurations"][0]["saturating_accumulation"]
            with self.subTest(mutation=mutation):
                self.assertTrue(review.capability_snapshot_failures(value, CAPABILITY_DECLARATION))
        changed_source = CAPABILITY_DECLARATION.replace("    pub compute: bool,", "    pub compute: bool,\n    pub new_feature: bool,")
        self.assertTrue(review.capability_snapshot_failures(original, changed_source))

    def test_replacement_expected_contract_is_json_not_rgba_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            source = b"".join(struct.pack("<6f", x, y, 1.0, 1.0, 0.0, 1.0)
                              for x, y in [(-0.75,-0.65),(0.55,-0.50),(-0.20,0.75)])
            (run / "input.bin").write_bytes(source)
            (run / "expected.json").write_text('{"rgba":[255,255,0,255],"extent":[128,128]}', encoding="utf-8")
            # The raw output deliberately omits the triangle; a metadata-only
            # check must not certify it, and JSON is not a raw RGBA oracle.
            (run / "output.rgba").write_bytes(bytes(128*128*4))
            value = {"schemaVersion":1,"oldDeviceId":"old","replacementDeviceId":"new",
                     "input":"input.bin","expected":"expected.json","readback":"output.rgba",
                     "readbackBytes":128*128*4,
                     "foreignPipelineSubmissionRejectedBeforeBackendWork":True,
                     "foreignSubmissionRejectedBeforeBackendWait":True,
                     "logicalSnapshotRetainedThroughRefusal":True,
                     "replacementPipelineSubmissionReadbackVerified":True,
                     "rejectionBackendStatsUnchanged":True}
            (run / "device-replacement.json").write_text(json.dumps(value), encoding="utf-8")
            reasons = review.replacement_failures(run)
            self.assertTrue(any("stable pixel" in reason for reason in reasons))
            self.assertFalse(any("expected contract" in reason for reason in reasons))

    def make_attempt(self, closure: Path, identifier: str, backend: str) -> dict:
        directory = closure / "attempts" / identifier
        run = directory / "run"
        (run / "raster").mkdir(parents=True)
        (run / "surface").mkdir()
        (run / "compute").mkdir()
        workloads = {
            run / "workload.json": "s0-copy-three-generations",
            run / "raster" / "raster-workload.json": "s1-fixed-raster-three-generations",
            run / "surface" / "surface-workload.json": "s1-renderer-surface-raster-three-generations",
            run / "compute" / "compute-workload.json": "s1-fixed-compute-three-generations",
        }
        for path, name in workloads.items():
            value = {"workload": name}
            if name.startswith("s1-renderer"):
                value.update(hostResizeEventVerified=True, clientExtentVerified=True, completionBeforeReconfigure=True)
            path.write_text(json.dumps(value), encoding="utf-8")
        for index in range(3):
            for path in (run / f"input-{index}.bin", run / f"expected-{index}.bin", run / f"readback-{index}.bin", run / "raster" / f"raster-input-{index}.bin", run / "raster" / f"raster-readback-{index}.rgba", run / "compute" / f"compute-input-{index}.bin", run / "compute" / f"compute-readback-{index}.bin"):
                path.write_bytes(b"x")
        (run / "surface" / "surface-events.jsonl").write_text("{}\n", encoding="utf-8")
        log = directory / "stdout.log"; log.write_text("ok\n", encoding="utf-8")
        return {"id": identifier, "backend": backend, "mode": "run", "runner_exit_status": 0,
                "directory": str(directory.relative_to(closure)), "locks": [], "source_inputs": [],
                "source_consistency_after_build": {"status": "stable"}, "source_consistency_after_attempt": {"status": "stable"},
                "built_source_verification": {"status": "verified"}, "commands": [{"result": {"raw_logs": [record(log)], "retained_files": [record(path) for path in run.rglob("*") if path.is_file()]}}],
                "validation": {"status": "verified", "diagnostics": {"activation_evidence": "activation.txt", "unreviewed_native_diagnostics": []}}}

    def test_recursive_missing_artifact_keeps_workload_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            closure = Path(temporary)
            attempt = self.make_attempt(closure, "dx", "dx12")
            (closure / attempt["directory"] / "run" / "raster" / "raster-readback-1.rgba").unlink()
            report = review.workload_check(closure / attempt["directory"] / "run", "dx12")
            self.assertEqual(report["status"], "open")
            self.assertTrue(any("raster" in item for item in report["reasons"]))

    def test_hash_mismatch_is_not_covered_by_manifest_membership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); artifact = root / "artifact"; artifact.write_bytes(b"one")
            records = [record(artifact)]; artifact.write_bytes(b"two")
            self.assertTrue(review.hash_records(records, root))

    def test_selected_attempt_backend_cannot_be_substituted(self) -> None:
        attempt, reasons = review.locate_attempt({"attempts": [{"id": "wgl", "backend": "native-gl-wgl"}]}, "wgl", "dx12")
        self.assertIsNone(attempt)
        self.assertIn("expected dx12", reasons[0])

    def test_visual_review_needs_explicit_review_and_hashed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); review_file = directory / "visual.json"
            review_file.write_text(json.dumps({"status": "reviewed", "attempt_ids": ["dx", "wgl"]}), encoding="utf-8")
            self.assertEqual(review.external_review(review_file, {"dx", "wgl"}, "visual")["status"], "open")

    def test_empty_provenance_cannot_pass_attempt_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            closure = Path(temporary)
            attempt = {"id": "dx", "backend": "dx12", "mode": "run", "runner_exit_status": 0,
                       "directory": "attempts/dx", "locks": [], "source_inputs": [], "commands": [],
                       "source_consistency_after_build": {"status": "stable"},
                       "source_consistency_after_attempt": {"status": "stable"},
                       "built_source_verification": {"status": "verified"}}
            records = review.attempt_review(attempt, closure, closure)
            self.assertTrue(all(item["status"] == "open" for item in records[:3]))

    def test_wgl_requires_observed_current_observer_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            closure = Path(temporary); marker = closure / "observer.json"
            marker.write_text(json.dumps({"installed": True, "logSaturated": False, "activationMarkerObserved": True,
                                          "contextVersion": "4.6 vendor", "coreDebugOutput": True,
                                          "khrDebugExtension": False, "debugOutputAvailable": True,
                                          "activationMessages": ["fluxel-rendering WGL diagnostic observer active"],
                                          "nativeDiagnostics": []}), encoding="utf-8")
            attempt = {"id": "wgl", "backend": "native-gl-wgl",
                       "validation": {"status": "observed", "artifact": record(marker)}}
            self.assertEqual(review.diagnostics_review(attempt, closure)["status"], "passed")
            for saturated in (True, None):
                observer = json.loads(marker.read_text(encoding="utf-8"))
                observer["logSaturated"] = saturated
                marker.write_text(json.dumps(observer), encoding="utf-8")
                attempt["validation"]["artifact"] = record(marker)
                self.assertEqual(review.diagnostics_review(attempt, closure)["status"], "open")
            marker.write_text(json.dumps({"installed": True, "logSaturated": False, "activationMarkerObserved": True,
                                          "activationMessages": [], "nativeDiagnostics": []}), encoding="utf-8")
            attempt["validation"]["artifact"] = record(marker)
            self.assertEqual(review.diagnostics_review(attempt, closure)["status"], "open")
            attempt["validation"]["status"] = "unverified"
            self.assertEqual(review.diagnostics_review(attempt, closure)["status"], "open")

    def test_surface_configuration_requires_actual_capability_membership(self) -> None:
        observation = {"extent": [320, 180], "configurationSupport": True,
            "capabilities": {"formats": ["Rgba8Unorm"], "presentModes": ["Fifo"], "alphaModes": ["Opaque"],
                             "colorSpaces": ["Srgb"], "usages": 1, "minExtent": [1, 1], "maxExtent": [640, 480]},
            "configuration": {"extent": [320, 180], "format": "Rgba8Unorm", "presentMode": "Fifo",
                              "alphaMode": "Opaque", "colorSpace": "Srgb", "usage": 1, "maximumFrameLatency": 1}}
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "surface").mkdir()
            path = run / "surface/surface-configuration.jsonl"
            path.write_text("\n".join(json.dumps(observation) for _ in range(3)), encoding="utf-8")
            self.assertEqual(review.surface_configuration_failures(run), [])
            for mutation in ("mode", "extent", "usage", "refusal", "omission"):
                changed = copy.deepcopy(observation)
                if mutation == "mode": changed["capabilities"]["presentModes"] = ["Immediate"]
                if mutation == "extent": changed["capabilities"]["maxExtent"] = [100, 100]
                if mutation == "usage": changed["capabilities"]["usages"] = 2
                if mutation == "refusal": changed["configurationSupport"] = False
                if mutation == "omission": del changed["configuration"]
                path.write_text("\n".join(json.dumps(changed) for _ in range(3)), encoding="utf-8")
                with self.subTest(mutation=mutation):
                    self.assertTrue(review.surface_configuration_failures(run))
    def test_dx12_rejects_fabricated_activation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            closure = Path(temporary); marker = closure / "dx12.json"
            marker.write_text(json.dumps({"requested": True, "deviceOpenCompleted": True,
                                          "activationContract": "EnableDebugLayer but no pinned revision"}), encoding="utf-8")
            attempt = {"id": "dx", "backend": "dx12",
                       "validation": {"status": "observed", "artifact": record(marker)}}
            self.assertEqual(review.diagnostics_review(attempt, closure)["status"], "open")

    def test_raster_requires_three_real_generations_and_dense_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary); (run / "raster").mkdir()
            workload = {"extent": [128, 128], "vertexStride": 24, "vertexCount": 3,
                        "clearRgba": [0, 0, 0, 0], "generations": [],
                        "snapshotRetentionVerified": True, "perFrameRetirementVerified": True,
                        "planCompilations": 1}
            self.assertTrue(any("exactly three" in reason for reason in review.raster_failures(run, workload)))

    def test_read_set_rejects_tampered_expected_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary); directory = run / "read-set"; directory.mkdir()
            (directory / "read-set-workload.json").write_text(json.dumps({
                "workload": "s1-private-read-set-raster-three-generations", "extent": [8, 8],
                "assetGenerations": [1, 2, 3], "uniformDestinationRanges": [[0, 256], [256, 512], [512, 768]],
                "sourceByteLengths": [20, 20, 20], "planReuseVerified": True,
                "snapshotRetentionVerified": True, "perFrameRetirementVerified": True}), encoding="utf-8")
            for index in range(3):
                source = struct.pack("<4f", 1.0, 0.0, 1.0, 1.0) + bytes([7, 9, 11, 255])
                (directory / f"read-set-input-{index}.bin").write_bytes(source)
                image = bytes([7, 0, 11, 255]) * 64
                (directory / f"read-set-expected-{index}.rgba").write_bytes(image)
                (directory / f"read-set-readback-{index}.rgba").write_bytes(image)
            (directory / "read-set-expected-1.rgba").write_bytes(b"tampered")
            self.assertTrue(any("expected image 1" in reason for reason in review.read_set_failures(run)))

    def test_compute_refusal_uses_actual_requirements_on_each_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary); (run / "compute").mkdir()
            (run / "compute" / "compute-workload.json").write_text(json.dumps({
                "workload": "s1-fixed-compute-three-generations", "status": "unsupported-before-submission"}), encoding="utf-8")
            for backend in ("native-gl-wgl", "dx12"):
                value = capability_fixture(True)
                limits = value["deviceCapabilities"]["limits"]
                limits.update(max_bind_groups=1, max_bindings_per_group=1, max_compute_workgroup_size=[64, 1, 1])
                for missing in ("compute", "storage_buffers", "workgroup_size"):
                    candidate = copy.deepcopy(value)
                    if missing == "workgroup_size": candidate["deviceCapabilities"]["limits"]["max_compute_workgroup_size"][0] = 63
                    else: candidate["deviceCapabilities"][missing] = False
                    (run / "capabilities.json").write_text(json.dumps(candidate), encoding="utf-8")
                    checks = review.compute_requirement_checks(candidate)
                    (run / "compute" / "compute-requirements.json").write_text(json.dumps({"checks": checks, "supported": False}), encoding="utf-8")
                    self.assertEqual(review.compute_failures(run, backend), [], (backend, missing))
                (run / "capabilities.json").write_text(json.dumps(value), encoding="utf-8")
                checks = review.compute_requirement_checks(value)
                (run / "compute" / "compute-requirements.json").write_text(json.dumps({"checks": checks, "supported": True}), encoding="utf-8")
                self.assertTrue(review.compute_failures(run, backend))
            (run / "compute-refusal.json").write_text("{}", encoding="utf-8")
            self.assertTrue(review.compute_failures(run, "native-gl-wgl"))

    def test_source_archive_rejects_added_or_tampered_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            closure = Path(temporary); archive = closure / "inputs.zip"
            inputs = [{"path": "src/lib.rs", "sha256": review.sha256(Path(__file__))}]
            with zipfile.ZipFile(archive, "w") as frozen:
                frozen.writestr("src/lib.rs", b"not the declared source")
                frozen.writestr("extra.rs", b"extra")
                frozen.writestr("snapshot-inputs.json", json.dumps(inputs))
            attempt = {"source_inputs": inputs, "source_snapshot": {**record(archive), "files": 1,
                       "kind": "exact_worktree_inputs"}}
            failures = review.frozen_source_failures(attempt, closure)
            self.assertTrue(any("membership" in reason for reason in failures))
            self.assertTrue(any("hash differs" in reason for reason in failures))


if __name__ == "__main__":
    unittest.main()
