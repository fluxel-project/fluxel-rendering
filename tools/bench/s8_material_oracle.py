"""Input-derived pixel oracle for the native S8 material workload.

This intentionally consumes the staged bytes and JSON rather than a renderer
snapshot.  It samples one well-inside-triangle and one well-outside-triangle
pixel per draw; that makes the 100x100 native workload cheap to check without
depending on edge rasterisation rules.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any


_RAW_BY_SCENARIO = {
    "cold": ("cold",),
    "warm": ("warm-prime", "warm-settle", "warm"),
    "replacement-10pct": (
        "replacement-prime",
        "replacement-settle",
        "replacement-10pct",
    ),
}
_PREFIX = "s8-material-"


def _fail(message: str) -> None:
    raise ValueError("S8 material oracle: " + message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail("cannot read %s: %s" % (path, error))
    if not isinstance(value, dict):
        _fail("%s must contain an object" % path.name)
    return value


def _required(value: dict[str, Any], key: str) -> Any:
    if key not in value:
        _fail("input is missing %r" % key)
    return value[key]


def _one_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        _fail("expected exactly one %s below %s, found %d" % (name, root, len(matches)))
    return matches[0]


def _byte_file(root: Path, relative: str, expected: int | None = None) -> bytes:
    path = _one_file(root, relative)
    try:
        data = path.read_bytes()
    except OSError as error:
        _fail("cannot read %s: %s" % (path, error))
    if expected is not None and len(data) != expected:
        _fail("%s has %d bytes, expected %d" % (relative, len(data), expected))
    return data


def _material_contract(root: Path) -> None:
    """Check the actual serialized MaterialDefinitions, not a copied oracle rule."""
    normal_body = "return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint;"
    inverse_body = (
        "let c = textureLoad(material_texture, vec2<i32>(0, 0), 0); "
        "return vec4<f32>(vec3<f32>(1.0) - c.rgb, c.a) * tint;"
    )
    for variant in range(4):
        path = _one_file(root, "s8-material-variant-%d.txt" % variant)
        try:
            definition = path.read_text(encoding="utf-8")
        except OSError as error:
            _fail("cannot read %s: %s" % (path, error))
        root_name = "s8_color_variant_%d" % variant
        if ('name: "%s"' % root_name) not in definition or ("fn %s() -> vec4<f32>" % root_name) not in definition:
            _fail("variant %d does not retain its controlled root function" % variant)
        if "material_texture" not in definition or "material_tint" not in definition:
            _fail("variant %d does not retain material resources" % variant)
        body = inverse_body if variant == 3 else normal_body
        if body not in definition:
            _fail("variant %d does not retain its controlled WGSL fragment body" % variant)


def _geometry_contract(data: bytes) -> None:
    """Parse the shared stride-36 vertex stream and reject altered fixture bytes."""
    expected_positions = ((-0.92, -0.92, 0.0), (0.92, -0.92, 0.0), (-0.92, 0.92, 0.0))
    for vertex, position in enumerate(expected_positions):
        values = struct.unpack_from("<9f", data, vertex * 36)
        if any(abs(values[index] - position[index]) > 1e-6 for index in range(3)):
            _fail("geometry POSITION for vertex %d is not the controlled S8 triangle" % vertex)
        if values[3:5] != (0.0, 0.0) or values[5:9] != (1.0, 1.0, 1.0, 1.0):
            _fail("geometry UV/COLOR for vertex %d is not the controlled S8 layout" % vertex)


def _transform_contract(columns: Any, draw: int, draws: int) -> None:
    if not isinstance(columns, list) or len(columns) != 4 or any(not isinstance(c, list) or len(c) != 4 for c in columns):
        _fail("transformColumns must be a 4x4 matrix")
    cols = math.ceil(math.sqrt(draws))
    rows = math.ceil(draws / cols)
    x, y = draw % cols, draw // cols
    wanted = ((1.0 / cols, 0.0, 0.0, 0.0), (0.0, 1.0 / rows, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (-1.0 + (2.0 * x + 1.0) / cols, 1.0 - (2.0 * y + 1.0) / rows, 0.0, 1.0))
    try:
        actual = tuple(tuple(float(item) for item in column) for column in columns)
    except (TypeError, ValueError) as error:
        _fail("invalid transformColumns: %s" % error)
    if any(not math.isfinite(item) for column in actual for item in column) or any(abs(actual[c][r] - wanted[c][r]) > 1e-6 for c in range(4) for r in range(4)):
        _fail("draw %d transformColumns is not the controlled tiled affine transform" % draw)


def _pixel_point(columns: Any, local_x: float, local_y: float, extent: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(columns, list) or len(columns) != 4:
        _fail("transformColumns must have four columns")
    try:
        # RenderTransform::from_columns: p' = x*c0 + y*c1 + z*c2 + c3.
        ndc_x = local_x * float(columns[0][0]) + local_y * float(columns[1][0]) + float(columns[3][0])
        ndc_y = local_x * float(columns[0][1]) + local_y * float(columns[1][1]) + float(columns[3][1])
    except (IndexError, TypeError, ValueError) as error:
        _fail("invalid transformColumns: %s" % error)
    width, height = extent
    # Pixel centres are x + .5, y + .5.  The conservative sample points are
    # deliberately farther than 0.3 local units from every triangle edge.
    x = math.floor((ndc_x * 0.5 + 0.5) * width)
    y = math.floor((0.5 - ndc_y * 0.5) * height)
    if not (0 <= x < width and 0 <= y < height):
        _fail("transform places stable sample outside framebuffer")
    return x, y


def _expected(texel: bytes, tint: Any, variant: int) -> bytes:
    if not isinstance(tint, list) or len(tint) != 4:
        _fail("draw tint must have four components")
    try:
        scale = [float(x) for x in tint]
    except (TypeError, ValueError) as error:
        _fail("invalid tint: %s" % error)
    if any(x not in (0.0, 1.0) for x in scale):
        _fail("S8 tint must be the recorded binary fixture tint")
    rgb = list(texel[:3])
    if variant == 3:
        rgb = [255 - x for x in rgb]
    elif variant not in (0, 1, 2):
        _fail("unknown materialVariant %r" % variant)
    return bytes([round(rgb[i] * scale[i]) for i in range(3)] + [round(texel[3] * scale[3])])


def _read_replacements(root: Path, input_value: dict[str, Any], texture_count: int) -> dict[int, bytes]:
    scenario = input_value["scenario"]
    if scenario != "replacement-10pct":
        return {}
    value = _load_json(_one_file(root, "s8-material-replacement-input.json"))
    entries = _required(value, "replacedTextures")
    if not isinstance(entries, list):
        _fail("replacedTextures must be an array")
    packed = _byte_file(root, "s8-material-textures-generation-2-replaced.rgba", ((texture_count + 9) // 10) * 4)
    replacements: dict[int, bytes] = {}
    for offset, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("index"), int):
            _fail("replacement entry %d has no integer index" % offset)
        index = entry["index"]
        if index < 0 or index >= texture_count or index % 10 != 0 or index in replacements:
            _fail("replacement index %r is not the unique tenth-texture set" % index)
        initial_descriptor = input_value["textureGeneration1"]["textures"][index]
        if not isinstance(initial_descriptor, dict) or entry.get("id") != initial_descriptor.get("id") or entry.get("generation") == initial_descriptor.get("generation"):
            _fail("replacement index %d must retain asset id and change generation" % index)
        rgba = entry.get("rgba")
        if not isinstance(rgba, list) or len(rgba) != 4 or any(not isinstance(x, int) or x < 0 or x > 255 for x in rgba):
            _fail("replacement index %d has invalid rgba" % index)
        packed_texel = packed[offset * 4 : offset * 4 + 4]
        if len(packed_texel) != 4 or packed_texel != bytes(rgba):
            _fail("replacement bytes do not agree with replacement input at index %d" % index)
        replacements[index] = packed_texel
    expected = set(range(0, texture_count, 10))
    if set(replacements) != expected:
        _fail("replacement input does not contain exactly every tenth texture")
    return replacements


def _raw_names(root: Path) -> set[str]:
    return {path.name for path in root.rglob("s8-material-*-raw.rgba")}


def review_run(run_output_root: Path) -> dict[str, int | str]:
    """Validate all required S8 raw readbacks and return compact coverage facts.

    Raises ValueError for malformed evidence, missing coverage, or a pixel mismatch.
    """
    root = Path(run_output_root)
    input_path = _one_file(root, "s8-material-input.json")
    source = _load_json(input_path)
    if source.get("schemaVersion") != 1:
        _fail("unsupported input schema")
    scenario = _required(source, "scenario")
    if scenario not in _RAW_BY_SCENARIO:
        _fail("unknown scenario %r" % scenario)
    extent_value = _required(source, "extent")
    if not isinstance(extent_value, list) or len(extent_value) != 2 or any(not isinstance(x, int) or x <= 0 for x in extent_value):
        _fail("extent must contain two positive integer dimensions")
    extent = (extent_value[0], extent_value[1])
    textures = _required(source, "textures")
    records = _required(source, "drawRecords")
    if not isinstance(textures, int) or textures <= 0 or not isinstance(records, list) or not records:
        _fail("input needs positive textures and nonempty drawRecords")
    if source.get("draws") != len(records) or source.get("allTexturesReferenced") is not True:
        _fail("draw count/allTexturesReferenced evidence is inconsistent")
    _material_contract(root)
    geometry = _required(source, "sharedGeometry")
    if not isinstance(geometry, dict) or geometry.get("bytes") != "s8-material-geometry.bin":
        _fail("shared geometry descriptor is not the S8 byte artifact")
    _geometry_contract(_byte_file(root, "s8-material-geometry.bin", 108))
    generation = _required(source, "textureGeneration1")
    if not isinstance(generation, dict) or generation.get("bytes") != "s8-material-textures-generation-1.rgba":
        _fail("generation 1 descriptor is not the S8 byte artifact")
    descriptors = _required(generation, "textures")
    if not isinstance(descriptors, list) or len(descriptors) != textures:
        _fail("generation 1 descriptor count differs from textures")
    initial = _byte_file(root, "s8-material-textures-generation-1.rgba", textures * 4)
    replacement = _read_replacements(root, source, textures)
    expected_names = {_PREFIX + name + "-raw.rgba" for name in _RAW_BY_SCENARIO[scenario]}
    present_names = _raw_names(root)
    if present_names != expected_names:
        _fail("raw readbacks are %r, expected exactly %r" % (sorted(present_names), sorted(expected_names)))
    checks = 0
    background_checks = 0
    used_textures: set[int] = set()
    seen_draws: set[int] = set()
    for raw_name in sorted(expected_names):
        raw = _byte_file(root, raw_name, extent[0] * extent[1] * 4)
        frame_replacements = replacement if raw_name == "s8-material-replacement-10pct-raw.rgba" else {}
        for record in records:
            if not isinstance(record, dict):
                _fail("draw record is not an object")
            draw = record.get("index")
            texture = record.get("textureIndex")
            variant = record.get("materialVariant")
            if not isinstance(draw, int) or draw < 0 or draw >= len(records) or draw in seen_draws:
                _fail("draw indices must be unique and contiguous")
            seen_draws.add(draw)
            _transform_contract(record.get("transformColumns"), draw, len(records))
            if not isinstance(texture, int) or not 0 <= texture < textures:
                _fail("draw has invalid textureIndex")
            if not isinstance(variant, int):
                _fail("draw has non-integer materialVariant")
            used_textures.add(texture)
            texel = frame_replacements.get(texture, initial[texture * 4 : texture * 4 + 4])
            foreground = _pixel_point(record.get("transformColumns"), -0.45, -0.45, extent)
            background = _pixel_point(record.get("transformColumns"), 0.55, 0.55, extent)
            wanted = _expected(texel, record.get("tint"), variant)
            for label, point, expected in (("foreground", foreground, wanted), ("background", background, b"\0\0\0\0")):
                x, y = point
                actual = raw[(y * extent[0] + x) * 4 : (y * extent[0] + x + 1) * 4]
                if label == "foreground":
                    if actual[3] != expected[3] or any(abs(actual[i] - expected[i]) > 1 for i in range(3)):
                        _fail("%s draw %d %s pixel %r is %r, expected %r" % (raw_name, draw, label, point, list(actual), list(expected)))
                    checks += 1
                elif actual != expected:
                    _fail("%s draw %d background pixel %r is %r, expected transparent clear" % (raw_name, draw, point, list(actual)))
                else:
                    background_checks += 1
        seen_draws.clear()
    if used_textures != set(range(textures)):
        _fail("not every texture index contributes a checked foreground pixel")
    return {"scenario": scenario, "rawFrames": len(expected_names), "drawChecks": checks, "backgroundChecks": background_checks, "textureReferences": len(used_textures)}
