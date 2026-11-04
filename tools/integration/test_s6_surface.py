"""Focused regression tests for the independent S6 Surface reviewer."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest

try:
    from .s6_review_parts import oracle, surface
except ImportError:
    from s6_review_parts import oracle, surface


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _ppm(raw: bytes) -> bytes:
    return b"P6\n96 96\n255\n" + b"".join(raw[index:index + 3] for index in range(0, len(raw), 4))


def _assets(slot: int) -> dict[str, dict[str, int]]:
    return {kind: {"slot": slot + index, "slotGeneration": 0, "contentGeneration": 1}
            for index, kind in enumerate(("geometry", "material", "texture"))}


def _bmp(path: Path, frame: int, extent: tuple[int, int]) -> None:
    rgba, _ = oracle.image(f"frame-{frame}", *extent)
    bgra = b"".join(rgba[index:index + 3][::-1] + b"\xff" for index in range(0, len(rgba), 4))
    width, height = extent
    path.write_bytes(b"BM" + struct.pack("<IHHI", 54 + len(bgra), 0, 0, 54)
                     + struct.pack("<IiiHHIIiiII", 40, width, -height, 1, 32, 0,
                                   len(bgra), 0, 0, 0, 0) + bgra)


def fixture(run: Path) -> None:
    """Create complete valid raw Surface evidence for a reviewer/ledger test."""
    run.mkdir(parents=True, exist_ok=True)
    artifact_names: list[list[str]] = [[], []]
    for kind, texture in (("mesh", bytes(oracle.MESH)), ("quad", bytes([255] * 4))):
        (run / f"s6-initial-{kind}-geometry-stream-0.bin").write_bytes(oracle.geometry_bytes(kind))
        (run / f"s6-initial-{kind}-texture.rgba").write_bytes(texture)
    (run / "s6-initial-mesh-material.txt").write_text(
        "textureLoad(material_texture, vec2<i32>(0,0),0)* tint\n", encoding="utf-8")
    (run / "s6-initial-quad-material.txt").write_text(
        "textureLoad(quad_texture) tint.bgr * 0.75 vec3<f32>(0.125) * remapped\n",
        encoding="utf-8",
    )
    artifact_names[0] += [
        "s6-initial-mesh-geometry-stream-0.bin", "s6-initial-quad-geometry-stream-0.bin",
        "s6-initial-mesh-texture.rgba", "s6-initial-quad-texture.rgba",
        "s6-initial-mesh-material.txt", "s6-initial-quad-material.txt",
    ]
    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    for frame, (extent, order, color) in enumerate(zip(((320, 180), (256, 144)), (-10, 10), (oracle.MESH, oracle.QUAD))):
        input_name = f"s6-surface-frame-{frame}-input.json"
        _json(run / input_name, {
            "schemaVersion": 1, "sceneVersion": frame + 1,
            "featureSnapshotSceneVersion": frame + 1, "featureEnabled": True,
            "surfaceExtent": list(extent), "companionExtent": [96, 96],
            "surfaceInteriorRgba": list(color),
            "mesh": {"bounds": {"minimum": [-0.5, -0.5, 0.0], "maximum": [0.5, 0.5, 0.0]},
                     "transformColumns": identity, "assets": _assets(1)},
            "marker": {"id": 7, "center": [0.0, 0.0, 0.0], "halfExtent": [0.55, 0.55],
                       "tint": [1.0, 0.0, 0.0, 1.0], "order": order, "assets": _assets(4)},
        })
        expected, mask = oracle.image(f"frame-{frame}")
        for name, data in ((f"s6-surface-frame-{frame}-expected.rgba", expected),
                           (f"s6-surface-frame-{frame}-readback.rgba", expected),
                           (f"s6-surface-frame-{frame}-expected.ppm", _ppm(expected)),
                           (f"s6-surface-frame-{frame}-readback.ppm", _ppm(expected))):
            (run / name).write_bytes(data)
            artifact_names[frame].append(name)
        (run / "s6-surface-stable-mask.bin").write_bytes(mask)
        _bmp(run / f"surface-client-frame-{frame}.bmp", frame, extent)
        _json(run / f"surface-client-frame-{frame}.capture.json", {"schemaVersion": 1, "clientExtent": list(extent), "pixelFormat": "BGRA8-top-down", "automaticVisualPass": False, "image": f"surface-client-frame-{frame}.bmp"})
        _json(run / f"surface-client-frame-{frame}.verdict.json", {"schemaVersion": 1, "frame": frame, "image": f"surface-client-frame-{frame}.bmp", "status": "passed"})
        artifact_names[frame] += [input_name, f"surface-client-frame-{frame}.bmp", f"surface-client-frame-{frame}.capture.json", f"surface-client-frame-{frame}.verdict.json"]
        if frame == 0:
            artifact_names[frame].append("s6-surface-stable-mask.bin")
    (run / "s6-surface-lifecycle.jsonl").write_text("\n".join(json.dumps({"sequence": number, "event": event, "frame": frame, "extent": list(extent)}) for number, event, frame, extent in surface._LIFECYCLE) + "\n", encoding="utf-8")
    (run / "surface-events.jsonl").write_text(json.dumps({"requestedExtent": [256, 144], "clientExtent": [256, 144], "selectedEvent": "resized"}) + "\n", encoding="utf-8")
    (run / "surface-configuration.jsonl").write_text("\n".join(json.dumps(row) for row in ({"extent": [320, 180], "configurationSupport": True}, {"extent": [256, 144], "configurationSupport": True})) + "\n", encoding="utf-8")
    _json(run / "s6-surface-reconfigure-input.json", {"previousExtent": [320, 180], "observedClientExtent": [256, 144], "nextOperation": "configure"})
    artifact_names[0] += ["s6-surface-lifecycle.jsonl", "surface-events.jsonl", "surface-configuration.jsonl", "s6-surface-reconfigure-input.json"]
    _json(run / "s6-surface-workload.json", {"schemaVersion": 1, "workload": "s6-composed-surface-resize", "initialExtent": [320, 180], "resizedExtent": [256, 144], "surfaceProof": "native-client-capture", "readbackProof": "same-snapshot-offscreen-companion", "featureCallbacks": {"prepare": 4, "contribute": 4}, "cases": [{"name": name, "artifacts": artifact_names[index]} for index, name in enumerate(("mesh-after-quad-initial", "quad-after-mesh-resized"))]})


class S6SurfaceReviewTests(unittest.TestCase):
    fixture = staticmethod(fixture)

    def test_fixture_passes_independent_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            fixture(run)
            self.assertEqual(surface.review(run), [])

    def test_rejects_bmp_geometry_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            fixture(run)
            bmp = bytearray((run / "surface-client-frame-1.bmp").read_bytes())
            bmp[54:57] = b"\x00\xff\x00"
            (run / "surface-client-frame-1.bmp").write_bytes(bmp)
            self.assertTrue(any("geometry oracle" in item for item in surface.review(run)))

    def test_rejects_resize_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            fixture(run)
            rows = (run / "s6-surface-lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
            rows[2], rows[3] = rows[3], rows[2]
            (run / "s6-surface-lifecycle.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
            self.assertTrue(any("lifecycle" in item for item in surface.review(run)))
