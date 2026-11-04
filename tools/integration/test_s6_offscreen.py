"""Tamper-oriented unit tests for the independent S6 offscreen reviewer."""

import json
from pathlib import Path
import tempfile
import unittest

from s6_review_parts import offscreen, oracle


class S6OffscreenTests(unittest.TestCase):
    def fixture(self, run: Path) -> None:
        run.mkdir()
        (run / "s6-stable-mask.bin").write_bytes(oracle.image("frame-0")[1])
        for kind, texture in (("mesh", bytes(oracle.MESH)), ("quad", bytes([255] * 4))):
            (run / f"s6-initial-{kind}-geometry-stream-0.bin").write_bytes(oracle.geometry_bytes(kind))
            (run / f"s6-initial-{kind}-texture.rgba").write_bytes(texture)
            (run / f"s6-initial-{kind}-geometry.txt").write_text("fixture", encoding="utf-8")
            (run / f"s6-initial-{kind}-material.txt").write_text(
                "textureLoad(material_texture) * tint" if kind == "mesh" else "textureLoad(quad_texture) tint.bgr * 0.75 vec3<f32>(0.125) * remapped",
                encoding="utf-8",
            )
        initial = {"slot": 2, "slotGeneration": 1, "contentGeneration": 1}
        replaced = {"slot": 2, "slotGeneration": 1, "contentGeneration": 2}
        mesh_assets = {"slot": 1, "slotGeneration": 1, "contentGeneration": 1}
        names = []
        for case, facts in oracle.FRAME_CASES.items():
            expected, _ = oracle.image(case)
            for suffix, data in (("expected.rgba", expected), ("actual.rgba", expected)):
                (run / f"s6-{case}-{suffix}").write_bytes(data)
            for suffix in ("expected.ppm", "actual.ppm"):
                (run / f"s6-{case}-{suffix}").write_bytes(
                    b"P6\n96 96\n255\n" + b"".join(expected[index:index + 3] for index in range(0, len(expected), 4))
                )
            quad = replaced if case in ("pending-new", "pending-new-warm") else initial
            (run / f"s6-{case}-input.json").write_text(json.dumps({
                "case": case, "sceneVersion": 1, "meshVisible": facts[0],
                "typedVersion": 1,
                "featureEnabled": case != "toggle-disabled",
                "featureVisible": case not in ("toggle-disabled", "provider-mesh-only"),
                "featureOrder": -10 if case == "frame-0" else 10,
                "marker": {
                    "id": 7,
                    "center": [2.0, 0.0, 0.0] if case == "provider-mesh-only" else [0.0, 0.0, 0.0],
                    "halfExtent": [0.55, 0.55], "tint": [1.0, 0.0, 0.0, 1.0],
                },
                "mesh": {
                    "bounds": {"minimum": [2.0, 2.0, 2.0], "maximum": [3.0, 3.0, 3.0]},
                    "transformColumns": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [2.0, 0.0, 0.0, 1.0]],
                } if case == "provider-marker-only" else {
                    "bounds": {"minimum": [-0.5, -0.5, 0.0], "maximum": [0.5, 0.5, 0.0]},
                    "transformColumns": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                },
                "assets": {
                    "mesh": {kind: dict(mesh_assets) for kind in ("geometry", "material", "texture")},
                    "quad": {kind: dict(quad) for kind in ("geometry", "material", "texture")},
                },
            }), encoding="utf-8")
            names.append(case)
        (run / "s6-pending-assets.json").write_text(json.dumps({
            "old": {kind: {"id": 2, "generation": 1} for kind in ("geometry", "material", "texture")},
            "new": {kind: {"id": 2, "generation": 2} for kind in ("geometry", "material", "texture")},
        }), encoding="utf-8")
        (run / "s6-pending-lifecycle.json").write_text(json.dumps({
            "schemaVersion": 1, "pendingAfterThree": 3, "fourthSubmission": "Backpressure",
            "callbacksUnchangedOnBackpressure": True,
            "retirements": [{"case": "pending-old", "pendingAfter": 2}, {"case": "pending-new", "pendingAfter": 1}, {"case": "pending-new-warm", "pendingAfter": 0}],
        }), encoding="utf-8")
        (run / "s6-pending-new-quad-geometry-stream-0.bin").write_bytes(oracle.geometry_bytes("quad", 0.5))
        (run / "s6-pending-new-quad-texture.rgba").write_bytes(bytes((255, 255, 0, 255)))
        (run / "s6-pending-new-quad-material.txt").write_text(
            "s6_replacement_quad tint.rgb * 0.5 vec3<f32>(0.25)", encoding="utf-8"
        )
        (run / "s6-composed-cache.json").write_text(json.dumps({
            "warmGeometryHitsBefore": 1, "warmGeometryHitsAfter": 2,
            "warmTextureHitsBefore": 1, "warmTextureHitsAfter": 2,
            "warmPrepareCallbacksBefore": 1, "warmPrepareCallbacksAfter": 2,
            "warmContributeCallbacksBefore": 1, "warmContributeCallbacksAfter": 2,
            "warmPlanCompilationsBefore": 4, "warmPlanCompilationsAfter": 4,
        }), encoding="utf-8")
        (run / "s6-composed-measurements.json").write_text(json.dumps({
            "schemaVersion": 1, "preparationBoundary": "accepted preparation elapsed time",
            "pendingPeak": 3, "acceptedSubmissions": len(oracle.FRAME_CASES),
            "preparationElapsedNs": 0, "graphCompileElapsedNs": 0,
            "acceptedBufferAllocations": 1, "acceptedBufferAllocationBytes": 1,
            "acceptedTextureAllocations": 1, "residentCopyBytes": 1, "frameUploadBytes": 1,
            "geometryHits": 1, "textureHits": 1, "frameReuses": 0,
            "opaqueCandidates": 1, "opaqueBatches": 1, "opaqueDraws": 1, "compilations": 1,
        }), encoding="utf-8")
        groups = (("original-three-frame-cache", names[:3]), ("enabled-disabled-enabled", names[3:6]), ("independent-providers", names[6:8]), ("three-pending-replacements", names[8:]))
        (run / "s6-composed-workload.json").write_text(json.dumps({
            "schemaVersion": 1, "workload": "s6-composed-offscreen-typed-quad",
            "cases": [{"name": name, "artifacts": []} for name, _ in groups],
        }), encoding="utf-8")

    def test_valid_fixture_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"; self.fixture(run)
            self.assertEqual(offscreen.review(run), [])

    def test_material_whitespace_is_semantic_but_debug_generation_is_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            self.fixture(run)
            material = run / "s6-initial-mesh-material.txt"
            material.write_text("textureLoad(material_texture, vec2<i32>(0,0),0)*tint;", encoding="utf-8")
            self.assertEqual(offscreen.review(run), [])
            assets = run / "s6-pending-assets.json"
            assets.write_text('{"old":{"geometry":{"generation":ContentGeneration(1)}}}', encoding="utf-8")
            self.assertTrue(any("malformed" in reason for reason in offscreen.review(run)))

    def test_tampered_input_expected_or_readback_is_rejected(self) -> None:
        for name, path in (("input", "s6-frame-0-input.json"), ("expected", "s6-frame-0-expected.rgba"), ("readback", "s6-frame-0-actual.rgba")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary) / "run"; self.fixture(run)
                if name == "input":
                    value = json.loads((run / path).read_text(encoding="utf-8")); value["featureOrder"] = 99
                    (run / path).write_text(json.dumps(value), encoding="utf-8")
                else:
                    data = bytearray((run / path).read_bytes()); data[0] ^= 1; (run / path).write_bytes(data)
                self.assertTrue(offscreen.review(run))
