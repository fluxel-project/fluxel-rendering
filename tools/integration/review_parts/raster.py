from __future__ import annotations

from .common import *

def fixed_raster_oracle(source: bytes) -> tuple[list[int] | None, list[str]]:
    """Derive the fixture's stable-pixel oracle from its recorded vertex bytes."""
    reasons: list[str] = []
    if len(source) != 72:
        return None, ["raster vertex input is not exactly three 24-byte vertices"]
    values = struct.unpack("<18f", source)
    if not all(math.isfinite(value) for value in values):
        return None, ["raster vertex input contains non-finite values"]
    positions = [(values[index * 6], values[index * 6 + 1]) for index in range(3)]
    fixture_positions = [(-0.75, -0.65), (0.55, -0.50), (-0.20, 0.75)]
    if any(abs(actual - expected) > 0.000_001
           for position, fixture in zip(positions, fixture_positions)
           for actual, expected in zip(position, fixture)):
        reasons.append("raster vertex positions differ from the fixed 128x128 fixture")
    colors = [values[index * 6 + 2:index * 6 + 6] for index in range(3)]
    if any(component < 0.0 or component > 1.0 for color in colors for component in color):
        reasons.append("raster vertex color is outside unorm range")
    if any(any(abs(component - colors[0][channel]) > 0.000_001
               for channel, component in enumerate(color)) for color in colors[1:]):
        reasons.append("raster fixture does not use one solid per-generation color")
    color = [round(component * 255.0) for component in colors[0]]
    return color if not reasons else None, reasons


def dense_raster_failures(pixels: bytes, color: list[int], label: str) -> list[str]:
    reasons: list[str] = []
    if len(pixels) != 128 * 128 * 4:
        reasons.append(f"{label} readback is not a 128x128 RGBA image")
        return reasons
    checked = 0
    vertices = [(16.0, 105.6), (99.2, 96.0), (51.2, 16.0)]
    for y in range(128):
        for x in range(128):
            point = (x + 0.5, y + 0.5)
            signs: list[float] = []
            edge_band = False
            for edge in range(3):
                a, b = vertices[edge], vertices[(edge + 1) % 3]
                dx, dy = b[0] - a[0], b[1] - a[1]
                sign = dx * (point[1] - a[1]) - dy * (point[0] - a[0])
                signs.append(sign)
                if abs(sign) / math.sqrt(dx * dx + dy * dy) <= 2.0:
                    edge_band = True
            if edge_band:
                continue
            inside = all(sign > 0.0 for sign in signs) or all(sign < 0.0 for sign in signs)
            offset = (y * 128 + x) * 4
            expected = bytes(color if inside else [0, 0, 0, 0])
            if pixels[offset:offset + 4] != expected:
                reasons.append(f"{label} differs at stable pixel ({x}, {y})")
                break
            checked += 1
        else:
            continue
        break
    if checked < 1000:
        reasons.append(f"{label} did not produce enough independent stable samples")
    return reasons


def raster_failures(run: Path, value: Any) -> list[str]:
    reasons: list[str] = []
    if not isinstance(value, dict):
        return ["raster workload is missing"]
    if (value.get("extent") != [128, 128] or value.get("vertexStride") != 24
            or value.get("vertexCount") != 3 or value.get("clearRgba") != [0, 0, 0, 0]
            or value.get("snapshotRetentionVerified") is not True
            or value.get("perFrameRetirementVerified") is not True
            or value.get("planCompilations") != 1):
        reasons.append("raster workload metadata does not match the fixed fixture contract")
    generations = value.get("generations")
    if not isinstance(generations, list) or len(generations) != 3:
        return reasons + ["raster workload must name exactly three generations"]
    generation_ids = [item.get("generation") if isinstance(item, dict) else None for item in generations]
    if not strictly_increasing(generation_ids):
        reasons.append("raster generation identifiers are not three increasing integers")
    for index, generation in enumerate(generations):
        source = run / "raster" / f"raster-input-{index}.bin"
        raw = run / "raster" / f"raster-readback-{index}.rgba"
        if not source.is_file() or not raw.is_file():
            reasons.append(f"missing raster raw input or readback for generation {index}")
            continue
        color, input_reasons = fixed_raster_oracle(source.read_bytes())
        reasons.extend(f"raster generation {index}: {reason}" for reason in input_reasons)
        if color is None:
            continue
        if not isinstance(generation, dict) or generation.get("rgba") != color:
            reasons.append(f"raster generation {index} metadata color does not match vertex input")
        reasons.extend(dense_raster_failures(raw.read_bytes(), color, f"raster generation {index}"))
    return reasons


def read_set_failures(run: Path, directory: str = "read-set",
                      workload: str = "s1-private-read-set-raster-three-generations",
                      explicit_expected: list[list[int]] | None = None,
                      texture_permutation: tuple[int, int, int, int] = (0, 1, 2, 3)) -> list[str]:
    root = run / directory
    path = root / "read-set-workload.json"
    value, error = read_json(path)
    if error or not isinstance(value, dict):
        return ["missing or invalid read-set workload"]
    reasons: list[str] = []
    if (value.get("workload") != workload
            or value.get("extent") != [8, 8] or not strictly_increasing(value.get("assetGenerations"))
            or value.get("uniformDestinationRanges") != [[0, 256], [256, 512], [512, 768]]
            or value.get("sourceByteLengths") != [20, 20, 20]
            or value.get("planReuseVerified") is not True
            or value.get("snapshotRetentionVerified") is not True
            or value.get("perFrameRetirementVerified") is not True):
        reasons.append("read-set workload metadata does not match the three-generation contract")
    for index in range(3):
        source = root / f"read-set-input-{index}.bin"
        expected = root / f"read-set-expected-{index}.rgba"
        readback = root / f"read-set-readback-{index}.rgba"
        if not source.is_file() or not expected.is_file() or not readback.is_file():
            reasons.append(f"missing read-set raw artifact for generation {index}")
            continue
        data = source.read_bytes()
        if len(data) != 20:
            reasons.append(f"read-set input {index} is not exactly 20 bytes")
            continue
        tint = struct.unpack("<4f", data[:16])
        if not all(math.isfinite(component) and 0.0 <= component <= 1.0 for component in tint):
            reasons.append(f"read-set input {index} has invalid tint floats")
            continue
        texel = bytes(data[16 + component] for component in texture_permutation)
        color = bytes(round(component * channel) for component, channel in zip(tint, texel))
        if explicit_expected is not None:
            if index >= len(explicit_expected) or explicit_expected[index] != list(color):
                reasons.append(f"read-set explicit expected color {index} disagrees with its raw material input")
                continue
            image = bytes(explicit_expected[index]) * 64
        else:
            image = color * 64
        if expected.read_bytes() != image:
            reasons.append(f"read-set expected image {index} is not derived from its raw input")
        if readback.read_bytes() != image:
            reasons.append(f"read-set readback {index} is not derived from its raw input")
    return reasons


def stable_raster_mask_failures(mask: bytes, pixels: bytes, color: list[int], label: str) -> list[str]:
    """Validate the retained edge-independent mask before trusting its selected pixels."""
    reasons: list[str] = []
    if len(mask) != 128 * 128:
        return [f"{label} oracle mask is not 128x128"]
    if len(pixels) != 128 * 128 * 4:
        return [f"{label} RGBA image is not 128x128"]
    vertices = [(16.0, 105.6), (99.2, 96.0), (51.2, 16.0)]
    checked = 0
    for y in range(128):
        for x in range(128):
            point = (x + 0.5, y + 0.5)
            signs: list[float] = []
            stable = True
            for edge in range(3):
                a, b = vertices[edge], vertices[(edge + 1) % 3]
                dx, dy = b[0] - a[0], b[1] - a[1]
                sign = dx * (point[1] - a[1]) - dy * (point[0] - a[0])
                signs.append(sign)
                if abs(sign) / math.sqrt(dx * dx + dy * dy) <= 2.0:
                    stable = False
            included = mask[y * 128 + x]
            if included not in (0, 1) or bool(included) != stable:
                reasons.append(f"{label} oracle mask disagrees with the fixed edge-distance rule at ({x}, {y})")
                return reasons
            if not stable:
                continue
            expected = bytes(color if (all(sign > 0.0 for sign in signs) or all(sign < 0.0 for sign in signs)) else [0, 0, 0, 0])
            offset = (y * 128 + x) * 4
            if pixels[offset:offset + 4] != expected:
                reasons.append(f"{label} differs at masked stable pixel ({x}, {y})")
                return reasons
            checked += 1
    if checked < 1000:
        reasons.append(f"{label} oracle mask has too few stable pixels")
    return reasons
