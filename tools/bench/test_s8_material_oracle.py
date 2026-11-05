import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import s8_material_oracle as oracle


def _columns(index: int):
    # Four 8x8 tiles in a 16x16 framebuffer, represented exactly as the Rust
    # column-major RenderTransform serialization.
    col, row = index % 2, index // 2
    return [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0], [-0.5 + col, 0.5 - row, 0, 1]]


def _definition(variant: int) -> str:
    root = "s8_color_variant_%d" % variant
    body = (
        "let c = textureLoad(material_texture, vec2<i32>(0, 0), 0); "
        "return vec4<f32>(vec3<f32>(1.0) - c.rgb, c.a) * tint;"
        if variant == 3
        else "return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint;"
    )
    return 'MaterialDefinition {\n name: "%s",\n wgsl: "fn %s() -> vec4<f32> { %s }",\n material_tint material_texture\n}\n' % (root, root, body)


def _make_run(root: Path, scenario: str = "cold") -> None:
    textures = bytes([10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255])
    records = [
        {"index": n, "geometry": "g", "materialVariant": n, "textureIndex": n, "tint": [1, 1 if n != 1 else 0, 1, 1], "transformColumns": _columns(n)}
        for n in range(4)
    ]
    source = {
        "schemaVersion": 1, "scenario": scenario, "draws": 4, "textures": 4,
        "extent": [16, 16], "sharedGeometry": {"bytes": "s8-material-geometry.bin"},
        "textureGeneration1": {"bytes": "s8-material-textures-generation-1.rgba", "textures": [{"index": n, "id": "texture-%d" % n, "generation": "generation-1"} for n in range(4)]},
        "drawRecords": records, "allTexturesReferenced": True,
    }
    (root / "s8-material-input.json").write_text(json.dumps(source), encoding="utf-8")
    geometry = b"".join(struct.pack("<9f", *position, 0, 0, 1, 1, 1, 1) for position in ((-.92, -.92, 0), (.92, -.92, 0), (-.92, .92, 0)))
    (root / "s8-material-geometry.bin").write_bytes(geometry)
    (root / "s8-material-textures-generation-1.rgba").write_bytes(textures)
    for n in range(4):
        (root / ("s8-material-variant-%d.txt" % n)).write_text(_definition(n), encoding="utf-8")
    replacement = {0: bytes([200, 30, 50, 255])} if scenario == "replacement-10pct" else {}
    if scenario == "replacement-10pct":
        (root / "s8-material-textures-generation-2-replaced.rgba").write_bytes(replacement[0])
        (root / "s8-material-replacement-input.json").write_text(json.dumps({"schemaVersion": 1, "replacedTextures": [{"index": 0, "id": "texture-0", "generation": "generation-2", "rgba": list(replacement[0])}]}), encoding="utf-8")
    for frame in oracle._RAW_BY_SCENARIO[scenario]:
        raw = bytearray(16 * 16 * 4)
        # Independent hand-derived pixel centres and colors for these four
        # 8x8 tiles; never call the oracle to manufacture its passing input.
        pixels = [(2, 5), (10, 5), (2, 13), (10, 13)]
        colors = [bytes([10, 20, 30, 255]), bytes([40, 0, 60, 255]),
                  bytes([70, 80, 90, 255]), bytes([155, 145, 135, 255])]
        if frame == "replacement-10pct":
            colors[0] = bytes([200, 30, 50, 255])
        for (x, y), color in zip(pixels, colors):
            raw[(y * 16 + x) * 4:(y * 16 + x + 1) * 4] = color
        (root / ("s8-material-%s-raw.rgba" % frame)).write_bytes(raw)


class S8MaterialOracleTests(unittest.TestCase):
    def test_accepts_cold_and_reports_all_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _make_run(root)
            self.assertEqual(oracle.review_run(root), {"scenario": "cold", "rawFrames": 1, "drawChecks": 4, "backgroundChecks": 4, "textureReferences": 4})

    def test_accepts_warm_and_checks_each_pending_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _make_run(root, "warm")
            self.assertEqual(oracle.review_run(root)["drawChecks"], 12)

    def test_replacement_input_changes_selected_texel(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _make_run(root, "replacement-10pct")
            self.assertEqual(oracle.review_run(root)["rawFrames"], 3)

    def test_replacement_does_not_authorize_new_pixels_in_primer_or_stale_final(self):
        for frame, color in (("replacement-prime", [200, 30, 50, 255]),
                             ("replacement-10pct", [10, 20, 30, 255])):
            with self.subTest(frame=frame), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); _make_run(root, "replacement-10pct")
                path = root / f"s8-material-{frame}-raw.rgba"
                raw = bytearray(path.read_bytes())
                raw[(5 * 16 + 2) * 4:(5 * 16 + 3) * 4] = bytes(color)
                path.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, "foreground"):
                    oracle.review_run(root)

    def test_rejects_foreground_raw_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _make_run(root)
            path = root / "s8-material-cold-raw.rgba"; raw = bytearray(path.read_bytes())
            raw[(5 * 16 + 2) * 4] ^= 0xFF; path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "foreground"):
                oracle.review_run(root)

    def test_rejects_transform_and_material_evidence_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _make_run(root)
            value = json.loads((root / "s8-material-input.json").read_text()); value["drawRecords"][0]["transformColumns"][3][0] = 5; (root / "s8-material-input.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "controlled tiled affine"):
                oracle.review_run(root)
            _make_run(root)
            path = root / "s8-material-variant-3.txt"; path.write_text(path.read_text().replace("1.0) - c.rgb", "c.rgb"))
            with self.assertRaisesRegex(ValueError, "fragment body"):
                oracle.review_run(root)


if __name__ == "__main__":
    unittest.main()
