"""Regression tests for the bounded S0-S3 integration evidence reviewer."""
from __future__ import annotations

import sys
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from review_tests.support import *


class ReviewTests(unittest.TestCase):
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
        log = directory / "stdout.log"; log.write_text("Ran 1 test\nOK\n", encoding="utf-8")
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

    def test_typed_external_review_rejects_unretained_claims_and_accepts_raw_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            closure = Path(temporary)
            attempt = self.make_attempt(closure, "dx", "dx12")
            log = closure / attempt["directory"] / "stdout.log"
            attempt["commands"].append({"kind": "test", "command": ["python", "-m", "unittest"],
                "result": {"exit_status": 0, "executed": 1, "test_command_recognized": True, "raw_logs": [record(log)]}})
            review_file = closure / "tests.json"
            forged = closure / "forged.log"; forged.write_text("forged\n", encoding="utf-8")
            base = {"status": "reviewed", "attempt_ids": ["dx"], "evidence": [{
                "attempt_id": "dx", "command_index": 1,
                "log": record(forged),
            }]}
            review_file.write_text(json.dumps(base), encoding="utf-8")
            selected = {"dx12": attempt}
            self.assertEqual(review.external_review(review_file, {"dx"}, "linked focused test evidence", selected, closure)["status"], "open")
            base["evidence"][0]["log"] = record(log)
            review_file.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(review.external_review(review_file, {"dx"}, "linked focused test evidence", selected, closure)["status"], "passed")
            base["evidence"][0]["attempt_id"] = "wrong-id"
            base["attempt_ids"] = ["wrong-id"]
            review_file.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(review.external_review(review_file, {"wrong-id"}, "linked focused test evidence", selected, closure)["status"], "open")
            base["evidence"][0]["attempt_id"] = "dx"
            base["attempt_ids"] = ["dx"]
            attempt["commands"][1]["result"]["exit_status"] = 1
            review_file.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(review.external_review(review_file, {"dx"}, "linked focused test evidence", selected, closure)["status"], "open")

    def test_visual_review_rejects_failed_human_verdict_even_with_real_capture_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            closure = Path(temporary); attempt = self.make_attempt(closure, "dx", "dx12")
            surface = closure / attempt["directory"] / "run" / "surface"
            bmp = surface / "surface-client-frame-0.bmp"; bmp.write_bytes(b"BM")
            capture = surface / "surface-client-frame-0.capture.json"
            capture.write_text(json.dumps({"image": bmp.name, "automaticVisualPass": False}), encoding="utf-8")
            verdict = surface / "surface-client-frame-0.verdict.json"
            verdict.write_text(json.dumps({"image": bmp.name, "status": "passed"}), encoding="utf-8")
            attempt["commands"][0]["result"]["retained_files"].extend([record(bmp), record(capture), record(verdict)])
            review_file = closure / "visual.json"
            review_file.write_text(json.dumps({"status": "reviewed", "attempt_ids": ["dx"], "evidence": [{
                "attempt_id": "dx", "frames": [{"screenshot": record(bmp), "capture": record(capture),
                "verdict": record(verdict), "human_verdict": "failed"}],
            }]}), encoding="utf-8")
            self.assertEqual(review.external_review(review_file, {"dx"}, "reviewed visual output", {"dx": attempt}, closure)["status"], "open")

    def test_visual_review_rejects_retained_extra_surface_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            closure = Path(temporary); attempt = self.make_attempt(closure, "dx", "dx12")
            surface = closure / attempt["directory"] / "run" / "surface"
            frames = []
            retained = attempt["commands"][0]["result"]["retained_files"]
            for index in range(4):
                bmp = surface / f"surface-client-frame-{index}.bmp"; bmp.write_bytes(b"BM")
                capture = surface / f"surface-client-frame-{index}.capture.json"
                capture.write_text(json.dumps({"image": bmp.name, "automaticVisualPass": False}), encoding="utf-8")
                verdict = surface / f"surface-client-frame-{index}.verdict.json"
                verdict.write_text(json.dumps({"image": bmp.name, "status": "passed"}), encoding="utf-8")
                retained.extend([record(bmp), record(capture), record(verdict)])
                if index < 3:
                    frames.append({"screenshot": record(bmp), "capture": record(capture),
                                   "verdict": record(verdict), "human_verdict": "passed"})
            review_file = closure / "visual.json"
            review_file.write_text(json.dumps({"status": "reviewed", "attempt_ids": ["dx"], "evidence": [{
                "attempt_id": "dx", "frames": frames,
            }]}), encoding="utf-8")
            self.assertEqual(review.external_review(
                review_file, {"dx"}, "reviewed visual output", {"dx": attempt}, closure
            )["status"], "open")

    def test_fault_review_rejects_an_unrelated_retained_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            closure = Path(temporary); attempt = self.make_attempt(closure, "dx", "dx12")
            log = closure / attempt["directory"] / "stdout.log"
            review_file = closure / "fault.json"
            review_file.write_text(json.dumps({"status": "reviewed", "attempt_ids": ["dx"], "evidence": [{
                "attempt_id": "dx", "scenario": "device-replacement", "raw_evidence": [record(log)],
            }]}), encoding="utf-8")
            self.assertEqual(review.external_review(
                review_file, {"dx"}, "reviewed native failure, loss, and device-replacement evidence",
                {"dx": attempt}, closure)["status"], "open")

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

    def test_s3_read_set_requires_its_explicit_rgb_expectations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary); directory = run / "s3-material"; directory.mkdir()
            (directory / "read-set-workload.json").write_text(json.dumps({
                "workload": "s3-material-read-set-raster-three-generations", "extent": [8, 8],
                "assetGenerations": [1, 2, 3], "uniformDestinationRanges": [[0, 256], [256, 512], [512, 768]],
                "sourceByteLengths": [20, 20, 20], "planReuseVerified": True,
                "snapshotRetentionVerified": True, "perFrameRetirementVerified": True}), encoding="utf-8")
            tints = ([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0])
            texels = ([17, 129, 241, 255], [231, 33, 125, 255], [67, 199, 211, 255])
            colors = ([17, 0, 0, 255], [0, 33, 0, 255], [0, 0, 211, 255])
            for index, (tint, texel, color) in enumerate(zip(tints, texels, colors)):
                (directory / f"read-set-input-{index}.bin").write_bytes(struct.pack("<4f", *tint) + bytes(texel))
                image = bytes(color) * 64
                (directory / f"read-set-expected-{index}.rgba").write_bytes(image)
                (directory / f"read-set-readback-{index}.rgba").write_bytes(image)
            self.assertEqual(review.read_set_failures(
                run, "s3-material", "s3-material-read-set-raster-three-generations", [list(color) for color in colors]), [])
            self.assertTrue(any("explicit expected color 1" in reason for reason in review.read_set_failures(
                run, "s3-material", "s3-material-read-set-raster-three-generations",
                [[17, 0, 0, 255], [0, 0, 33, 255], [0, 0, 211, 255]])))

    def test_s3_replacement_swizzle_oracle_accepts_valid_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary); directory = run / "s3-material" / "replacement"; directory.mkdir(parents=True)
            (directory / "read-set-workload.json").write_text(json.dumps({
                "workload": "s3-material-replacement-read-set-raster-three-generations", "extent": [8, 8],
                "assetGenerations": [4, 5, 6], "uniformDestinationRanges": [[0, 256], [256, 512], [512, 768]],
                "sourceByteLengths": [20, 20, 20], "planReuseVerified": True,
                "snapshotRetentionVerified": True, "perFrameRetirementVerified": True}), encoding="utf-8")
            tints = ([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0])
            texels = ([17, 129, 241, 255], [231, 33, 125, 255], [67, 199, 211, 255])
            colors = ([241, 0, 0, 255], [0, 231, 0, 255], [0, 0, 199, 255])
            for index, (tint, texel, color) in enumerate(zip(tints, texels, colors)):
                (directory / f"read-set-input-{index}.bin").write_bytes(struct.pack("<4f", *tint) + bytes(texel))
                image = bytes(color) * 64
                (directory / f"read-set-expected-{index}.rgba").write_bytes(image)
                (directory / f"read-set-readback-{index}.rgba").write_bytes(image)
            arguments = (run, "s3-material/replacement", "s3-material-replacement-read-set-raster-three-generations",
                         [list(color) for color in colors], (2, 0, 1, 3))
            self.assertEqual(review.read_set_failures(*arguments), [])
            damaged_input = bytearray((directory / "read-set-input-0.bin").read_bytes())
            damaged_input[18] = 1
            (directory / "read-set-input-0.bin").write_bytes(damaged_input)
            self.assertTrue(any("explicit expected color 0" in reason for reason in review.read_set_failures(*arguments)))
            (directory / "read-set-input-0.bin").write_bytes(struct.pack("<4f", *tints[0]) + bytes(texels[0]))
            (directory / "read-set-readback-2.rgba").write_bytes(b"damaged")
            self.assertTrue(any("readback 2" in reason for reason in review.read_set_failures(*arguments)))

    def test_s2_mask_cannot_hide_an_arbitrary_edge_region(self) -> None:
        pixels = bytes(128 * 128 * 4)
        self.assertTrue(any("mask disagrees" in reason for reason in review.stable_raster_mask_failures(
            bytes(128 * 128), pixels, [0, 0, 0, 0], "S2 test")))

    def test_shader_admission_route_owned_demands_require_their_rhi_feature_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            root = run / "s2-shader"
            root.mkdir()
            capability_fields = {
                "multiview": False, "push_constants": False, "binding_arrays": False,
                "runtime_sized_binding_arrays": False, "storage_buffers": False,
                "storage_textures": False, "nonuniform_uniform_buffer_indexing": False,
                "nonuniform_storage_buffer_indexing": False,
                "nonuniform_sampled_texture_indexing": False,
                "nonuniform_storage_texture_indexing": False,
                "nonuniform_sampler_indexing": False,
                "shader_primitive_index": True, "shader_early_depth_test": True,
                "ray_query": True,
                "limits": {"max_multiview_views": 0, "max_push_constant_bytes": 0},
            }
            (run / "capabilities.json").write_text(
                json.dumps({"deviceCapabilities": capability_fields}), encoding="utf-8")
            evidence = {
                "schemaVersion": 1,
                "nagaCapabilityDemands": ["primitive-index", "early-depth-test", "ray-query"],
                "rhiFeatureNames": ["primitive-index", "shader-early-depth-test", "ray-query"],
                "deviceCapabilities": {
                    "multiview": False, "pushConstants": False, "bindingArrays": False,
                    "runtimeSizedBindingArrays": False, "storageBuffers": False,
                    "storageTextures": False, "nonuniformUniformBufferIndexing": False,
                    "nonuniformStorageBufferIndexing": False,
                    "nonuniformSampledTextureIndexing": False,
                    "nonuniformStorageTextureIndexing": False,
                    "nonuniformSamplerIndexing": False,
                    "limits": {"maxMultiviewViews": 0, "maxPushConstantBytes": 0},
                },
                "pipelineFacts": {
                    "multiviewViews": None, "pushConstantBytes": 0, "multisampleCount": 1,
                    "nonuniform": {"uniformBuffers": False, "storageBuffers": False,
                                   "sampledTextures": False, "storageTextures": False,
                                   "samplers": False, "bindingArrays": False,
                                   "runtimeSizedBindingArrays": False},
                },
                "targetEntries": [
                    {"name": "vs_main", "stage": "vertex", "workgroupSize": None},
                    {"name": "fs_main", "stage": "fragment", "workgroupSize": None},
                ],
                "perDemand": [
                    {"demand": "primitive-index", "outcome": {"status": "accepted"}},
                    {"demand": "early-depth-test", "outcome": {"status": "accepted"}},
                    {"demand": "ray-query", "outcome": {"status": "accepted"}},
                ],
                "admission": {"status": "accepted"},
            }
            path = root / "shader-admission.json"
            path.write_text(json.dumps(evidence), encoding="utf-8")
            self.assertEqual(review.shader_admission_failures(run, root, "test", True), [])

            evidence["targetEntries"][1]["stage"] = "vertex"
            path.write_text(json.dumps(evidence), encoding="utf-8")
            self.assertTrue(any("Fragment target entry" in reason
                                for reason in review.shader_admission_failures(run, root, "test", True)))
            evidence["targetEntries"][1]["stage"] = "fragment"

            evidence["rhiFeatureNames"].remove("shader-early-depth-test")
            path.write_text(json.dumps(evidence), encoding="utf-8")
            self.assertTrue(any("early-depth-test" in reason
                                for reason in review.shader_admission_failures(run, root, "test", True)))

            evidence["rhiFeatureNames"].append("shader-early-depth-test")
            evidence["perDemand"][0]["outcome"] = {
                "status": "refused",
                "error": {"kind": "unrepresentable", "demand": "primitive-index"},
            }
            path.write_text(json.dumps(evidence), encoding="utf-8")
            self.assertTrue(any("outcome for primitive-index" in reason
                                for reason in review.shader_admission_failures(run, root, "test", True)))

            evidence["perDemand"][0]["outcome"] = {"status": "accepted"}
            capability_fields["shader_primitive_index"] = False
            (run / "capabilities.json").write_text(
                json.dumps({"deviceCapabilities": capability_fields}), encoding="utf-8")
            path.write_text(json.dumps(evidence), encoding="utf-8")
            self.assertTrue(any("shader_primitive_index" in reason
                                for reason in review.shader_admission_failures(run, root, "test", True)))

    def test_s3_replacement_is_a_required_recursive_workload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary); (run / "s3-material").mkdir()
            reasons = review.s3_material_failures(run)
            self.assertTrue(any("replacement" in reason.lower() for reason in reasons))

    def test_s3_reflection_requires_both_bindings_shapes_and_fragment_read_access(self) -> None:
        valid = """
ShaderInterface {
    bindings: [
        ReflectedBinding {
            group: 0,
            binding: 1,
            kind: Texture,
            shape: Some(
                Texture2dF32,
            ),
            visibility: {
                Fragment,
            },
            entry_access: {
                EntryPoint { name: "fs_main", stage: Fragment }: BindingAccess {
                    read: true,
                    write: false,
                    query: false,
                    atomic: false,
                },
            },
        },
        ReflectedBinding {
            group: 0,
            binding: 0,
            kind: Uniform,
            shape: Some(
                Uniform(
                    Vec4(
                        F32,
                    ),
                ),
            ),
            visibility: {
                Fragment,
            },
            entry_access: {
                EntryPoint { name: "fs_main", stage: Fragment }: BindingAccess {
                    read: true,
                    write: false,
                    query: false,
                    atomic: false,
                },
            },
        },
    ],
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "s3-material-interface.txt"
            path.write_text(valid, encoding="utf-8")
            self.assertEqual(review.material_interface_failures(path, "S3 Material"), [])
            path.write_text(valid.replace("shape: Some(\n                Texture2dF32,\n            )", "shape: None"), encoding="utf-8")
            self.assertTrue(any("binding 1" in reason for reason in review.material_interface_failures(path, "S3 Material")))
            path.write_text(valid.replace("write: false", "write: true", 1), encoding="utf-8")
            self.assertTrue(any("binding 1" in reason for reason in review.material_interface_failures(path, "S3 Material")))

    def test_s2_texture_is_a_required_independent_workload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary); run.mkdir(exist_ok=True)
            self.assertTrue(any("S2 texture" in reason for reason in review.s2_texture_failures(run)))

    def test_s2_pipeline_cache_hit_is_required_and_must_be_true(self) -> None:
        label = "S2 direct Shader workload"
        self.assertTrue(review.pipeline_cache_hit_failures({}, label))
        self.assertTrue(review.pipeline_cache_hit_failures({"pipelineCacheHitVerified": False}, label))
        self.assertEqual(review.pipeline_cache_hit_failures({"pipelineCacheHitVerified": True}, label), [])

    def test_compute_refusal_uses_actual_requirements_on_each_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary); (run / "compute").mkdir()
            def write_shader_admission(snapshot: dict) -> None:
                actual = snapshot["deviceCapabilities"]
                for actual_name in review.ADMISSION_CAPABILITY_FIELDS.values():
                    actual.setdefault(actual_name, False)
                actual["limits"].setdefault("max_multiview_views", 0)
                actual["limits"].setdefault("max_push_constant_bytes", 0)
                capabilities = {
                    recorded: actual[actual_name]
                    for recorded, actual_name in review.ADMISSION_CAPABILITY_FIELDS.items()
                }
                capabilities["limits"] = {
                    recorded: actual["limits"][actual_name]
                    for recorded, actual_name in review.ADMISSION_LIMIT_FIELDS.items()
                }
                (run / "compute" / "shader-admission.json").write_text(json.dumps({
                    "schemaVersion": 1, "nagaCapabilityDemands": [], "rhiFeatureNames": [],
                    "deviceCapabilities": capabilities,
                    "pipelineFacts": {
                        "multiviewViews": None, "pushConstantBytes": 0, "multisampleCount": 1,
                        "nonuniform": {"uniformBuffers": False, "storageBuffers": False,
                                       "sampledTextures": False, "storageTextures": False,
                                       "samplers": False, "bindingArrays": False,
                                       "runtimeSizedBindingArrays": False},
                    },
                    "targetEntries": [
                        {"name": "cs_main", "stage": "compute", "workgroupSize": [64, 1, 1]},
                    ],
                    "perDemand": [], "admission": {"status": "accepted"},
                }), encoding="utf-8")
            (run / "compute" / "compute-composed.wgsl").write_text(
                "@group(0) @binding(0) var<storage, read_write> values: array<u32>;\n@compute @workgroup_size(64) fn cs_main() {}\n",
                encoding="utf-8")
            (run / "compute" / "compute-interface.txt").write_text("compute interface\n", encoding="utf-8")
            (run / "compute" / "compute-route.json").write_text(
                json.dumps({"route": "naga", "format": "Wgsl", "environment": "test", "abiKey": "test-abi"}),
                encoding="utf-8")
            (run / "compute" / "compute-workload.json").write_text(json.dumps({
                "workload": "s1-fixed-compute-three-generations", "status": "unsupported-before-submission"}), encoding="utf-8")
            for index in range(3):
                source = bytes(384)
                (run / "compute" / f"compute-input-{index}.bin").write_bytes(source)
                (run / "compute" / f"compute-expected-{index}.bin").write_bytes((7).to_bytes(4, "little") * 96)
            for backend in ("native-gl-wgl", "dx12"):
                value = capability_fixture(True)
                limits = value["deviceCapabilities"]["limits"]
                limits.update(max_bind_groups=1, max_bindings_per_group=1, max_compute_workgroup_size=[64, 1, 1])
                for missing in ("compute", "storage_buffers", "workgroup_size"):
                    candidate = copy.deepcopy(value)
                    if missing == "workgroup_size": candidate["deviceCapabilities"]["limits"]["max_compute_workgroup_size"][0] = 63
                    else: candidate["deviceCapabilities"][missing] = False
                    write_shader_admission(candidate)
                    (run / "capabilities.json").write_text(json.dumps(candidate), encoding="utf-8")
                    checks = review.compute_requirement_checks(candidate)
                    (run / "compute" / "compute-requirements.json").write_text(json.dumps({"checks": checks, "supported": False}), encoding="utf-8")
                    self.assertEqual(review.compute_failures(run, backend), [], (backend, missing))
                write_shader_admission(value)
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


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    from review_tests.capability import CapabilityReviewTests

    suite = loader.loadTestsFromTestCase(ReviewTests)
    suite.addTests(loader.loadTestsFromTestCase(CapabilityReviewTests))
    return suite


if __name__ == "__main__":
    unittest.main()
