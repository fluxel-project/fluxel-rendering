from __future__ import annotations

from .support import *


class CapabilityReviewTests(unittest.TestCase):
    def write_capability_review_evidence(self, run: Path, primary: dict, replacement: dict) -> None:
        (run / "s2-shader").mkdir(exist_ok=True)
        (run / "s2-shader-replacement-device").mkdir(exist_ok=True)
        (run / "capabilities.json").write_text(json.dumps(primary), encoding="utf-8")
        (run / "s2-shader-replacement-device/capabilities.json").write_text(json.dumps(replacement), encoding="utf-8")
        (run / "workload.json").write_text(json.dumps({"adapter": "actual adapter"}), encoding="utf-8")
        (run / "device-replacement.json").write_text(json.dumps({
            "oldDeviceId": "DeviceId(1)", "replacementDeviceId": "DeviceId(2)"}), encoding="utf-8")
        (run / "s2-shader/s2-shader-workload.json").write_text(
            json.dumps({"deviceId": "DeviceId(1)"}), encoding="utf-8")
        (run / "s2-shader-replacement-device/s2-shader-workload.json").write_text(
            json.dumps({"deviceId": "DeviceId(2)"}), encoding="utf-8")

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
                    self.write_capability_review_evidence(
                        run, capability_fixture(capability), capability_fixture(True, "DeviceId(2)"))
                    report = review.capability_review(attempt, closure)
                    self.assertEqual(report["status"], "passed" if isinstance(capability, bool) else "open")

    def test_capability_review_rejects_missing_extra_and_misattributed_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            closure = Path(temporary)
            run = closure / "run"
            run.mkdir()
            declaration = closure / "rhi/crates/core/src/api/foundation.rs"
            declaration.parent.mkdir(parents=True)
            declaration.write_text(CAPABILITY_DECLARATION, encoding="utf-8")
            attempt = {"id": "wgl", "backend": "native-gl-wgl", "directory": ".",
                       "built_git_checkout_verification": {"checkouts": [{"path": str(closure / "rhi")} ]}}
            self.write_capability_review_evidence(run, capability_fixture(), capability_fixture(True, "DeviceId(2)"))
            self.assertEqual(review.capability_review(attempt, closure)["status"], "passed")
            for mutation in ("missing_primary", "missing_replacement", "extra", "primary_wrong_device",
                             "replacement_wrong_device", "primary_wrong_adapter", "replacement_wrong_adapter"):
                with self.subTest(mutation=mutation):
                    self.write_capability_review_evidence(run, capability_fixture(), capability_fixture(True, "DeviceId(2)"))
                    if mutation == "missing_primary":
                        (run / "capabilities.json").unlink()
                    elif mutation == "missing_replacement":
                        (run / "s2-shader-replacement-device/capabilities.json").unlink()
                    elif mutation == "extra":
                        (run / "spoofed-capabilities.json").write_text(json.dumps(capability_fixture()), encoding="utf-8")
                    elif mutation == "primary_wrong_device":
                        (run / "capabilities.json").write_text(
                            json.dumps(capability_fixture(True, "DeviceId(2)")), encoding="utf-8")
                    elif mutation == "replacement_wrong_device":
                        (run / "s2-shader-replacement-device/capabilities.json").write_text(
                            json.dumps(capability_fixture(True, "DeviceId(1)")), encoding="utf-8")
                    else:
                        path = run / ("capabilities.json" if mutation == "primary_wrong_adapter"
                                      else "s2-shader-replacement-device/capabilities.json")
                        value = capability_fixture(True, "DeviceId(1)" if mutation == "primary_wrong_adapter" else "DeviceId(2)")
                        value["adapter"] = "spoofed adapter"
                        path.write_text(json.dumps(value), encoding="utf-8")
                    self.assertEqual(review.capability_review(attempt, closure)["status"], "open")
                    for path in (run / "capabilities.json", run / "s2-shader-replacement-device/capabilities.json",
                                 run / "spoofed-capabilities.json"):
                        if path.exists():
                            path.unlink()

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
