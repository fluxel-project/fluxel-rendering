"""Independently review the bounded animated demo's real-window evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

from capture_contract import WINDOW_STABLE_MASK_POLICY, capture_failures, window_stable_mask
from s7_oracles.core import bmp, check, color, q, raster


def review(root: Path) -> dict:
    geometry = (root / "demo-positions.bin").read_bytes()
    texture = (root / "demo-texture.rgba").read_bytes()
    material = (root / "demo-material.txt").read_text(encoding="utf-8")
    if len(geometry) != 36 or len(texture) != 4:
        raise ValueError("demo does not retain one packed POSITION triangle and texel")
    # This oracle supports the actual built-in Color material, not arbitrary WGSL.
    if "textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint" not in material:
        raise ValueError("unsupported demo material")
    vertices = list(struct.iter_unpack("<fff", geometry))
    expected_inputs = {f"demo-evidence-frame-{i}-input.json" for i in range(11)}
    expected_raw = {f"demo-evidence-frame-{i}-readback.rgba" for i in range(11)}
    if {p.name for p in root.glob("demo-evidence-frame-*-input.json")} != expected_inputs:
        raise ValueError("demo inputs must cover exactly eleven consecutive frames")
    if {p.name for p in root.glob("demo-evidence-frame-*-readback.rgba")} != expected_raw:
        raise ValueError("demo readbacks must cover exactly eleven consecutive frames")
    captures = {f"surface-client-frame-{i}.capture.json" for i in (0, 5, 10)}
    if {p.name for p in root.glob("*.capture.json")} != captures:
        raise ValueError("demo needs exactly three separated window captures")
    results, hashes, versions, route = [], set(), [], None
    for frame in range(11):
        data = json.loads((root / f"demo-evidence-frame-{frame}-input.json").read_text())
        if data["frame"] != frame or data["extent"] != [640, 480]:
            raise ValueError("demo frame or extent differs from retained input contract")
        current_route = data["route"]
        if current_route not in ("forward", "deferred") or (route and current_route != route):
            raise ValueError("demo route changed within an attempt")
        route = current_route
        matrix, tint = data["transformColumns"], data["tint"]
        points = [[sum(matrix[c][r] * (*vertex, 1)[c] for c in range(4))
                   for r in range(3)] for vertex in vertices]
        rgba = (bytes(q(texture[i] / 255 * tint[i]) for i in range(4))
                if route == "forward" else color(points, texture, tint))
        expected, mask = raster(points, rgba, data["extent"])
        (root / f"demo-review-frame-{frame}-expected.rgba").write_bytes(expected)
        (root / f"demo-review-frame-{frame}-stable-mask.bin").write_bytes(mask)
        raw = (root / f"demo-evidence-frame-{frame}-readback.rgba").read_bytes()
        result = {"frame": frame, "readback": check(raw, expected, mask)}
        hashes.add(hashlib.sha256(raw).hexdigest())
        versions.append(data["sceneVersion"])
        if frame in (0, 5, 10):
            image = f"surface-client-frame-{frame}.bmp"
            meta = json.loads((root / f"surface-client-frame-{frame}.capture.json").read_text())
            failures = capture_failures(meta, frame, data["extent"], image)
            if meta.get("schemaVersion") != 2 or failures:
                raise ValueError(f"current demo capture is invalid: {failures}")
            pixels, _, _ = bmp(root / image, expected, mask)
            result["window"] = check(pixels, expected, window_stable_mask(mask, data["extent"]), bmp_mode=True)
        results.append(result)
    if len(hashes) != 11 or len(set(versions)) != 11:
        raise ValueError("demo animation did not produce eleven distinct frame snapshots")
    return {"status": "passed", "route": route, "denseReadbacks": 11,
            "windowCaptures": 3, "frames": results,
            "windowDecorationExclusion": WINDOW_STABLE_MASK_POLICY,
            "scope": "fixed Color triangle; stable pixels; manual screenshot inspection required"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}))
