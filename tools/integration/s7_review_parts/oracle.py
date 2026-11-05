"""Non-verdict binary parsing helpers.

Independent verdicts live in :mod:`s7_oracles`: accepting an image by merely
comparing it with an archived expected buffer is explicitly forbidden.
"""
from __future__ import annotations

import struct
from pathlib import Path


def shape_failures(actual: Path, expected: Path, mask: Path) -> list[str]:
    try:
        raw, saved, stable = actual.read_bytes(), expected.read_bytes(), mask.read_bytes()
    except OSError as error:
        return [f"cannot read raw pixel evidence: {error}"]
    if len(raw) != len(saved) or len(raw) != len(stable) * 4 or not stable:
        return ["raw RGBA and mask shapes are inconsistent"]
    return []


def bmp_failures(path: Path, expected: bytes, mask: bytes) -> list[str]:
    try:
        data = path.read_bytes()
        if data[:2] != b"BM" or len(data) < 54:
            return ["capture is not a BMP"]
        offset = struct.unpack_from("<I", data, 10)[0]
        width, height = struct.unpack_from("<ii", data, 18)
        if width <= 0 or height == 0 or offset > len(data):
            return ["BMP dimensions or offset invalid"]
        height = abs(height)
        if len(expected) != width * height * 4 or len(mask) != width * height:
            return ["BMP has no same-shape independent companion oracle"]
        stride = ((width * 4 + 3) // 4) * 4
        if offset + stride * height > len(data):
            return ["BMP pixel payload is truncated"]
        for y in range(height):
            source_y = y if struct.unpack_from("<i", data, 22)[0] < 0 else height - 1 - y
            for x in range(width):
                index = y * width + x
                if mask[index]:
                    actual = data[offset + source_y * stride + x * 4:offset + source_y * stride + x * 4 + 4]
                    rgba = expected[index * 4:index * 4 + 4]
                    if actual != bytes((rgba[2], rgba[1], rgba[0], rgba[3])):
                        return [f"BMP differs at stable pixel {x},{y}"]
    except (OSError, struct.error) as error:
        return [f"cannot parse BMP: {error}"]
    return []
