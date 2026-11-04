"""Analytic S6 geometry and material expectations, independent of readback."""

from __future__ import annotations

from functools import lru_cache
import struct


CLEAR = (0, 0, 0, 0)
MESH = (0, 255, 0, 255)
QUAD = (32, 32, 223, 255)
REPLACEMENT_QUAD = (191, 64, 0, 255)
TRIANGLE = ((-0.45, -0.40), (0.45, -0.40), (0.0, 0.45))
FRAME_CASES = {
    "frame-0": (True, True, False, 0.55, QUAD),
    "frame-1": (True, True, True, 0.55, QUAD),
    "frame-2": (True, True, True, 0.55, QUAD),
    "toggle-enabled": (True, True, True, 0.55, QUAD),
    "toggle-disabled": (True, False, True, 0.55, QUAD),
    "toggle-reenabled": (True, True, True, 0.55, QUAD),
    "provider-marker-only": (False, True, True, 0.55, QUAD),
    "provider-mesh-only": (True, False, True, 0.55, QUAD),
    "pending-old": (True, True, True, 0.55, QUAD),
    "pending-new": (True, True, True, 0.275, REPLACEMENT_QUAD),
    "pending-new-warm": (True, True, True, 0.275, REPLACEMENT_QUAD),
}


def edge(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def stable(point: tuple[float, float]) -> bool:
    # The shared workload mask excludes both old and replacement quad edges.
    quad_edges = all(abs(abs(axis) - half) > 0.04 for axis in point for half in (0.55, 0.275))
    triangle_edges = all(abs(edge(TRIANGLE[i], TRIANGLE[(i + 1) % 3], point)) > 0.036
                         for i in range(3))
    return quad_edges and triangle_edges


@lru_cache(maxsize=32)
def image(case: str, width: int = 96, height: int = 96) -> tuple[bytes, bytes]:
    """Generate from the named scenario, never from supplied expected pixels."""
    mesh, quad, quad_after, half, color = FRAME_CASES[case]
    rgba = bytearray(width * height * 4)
    mask = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            p = (2 * (x + 0.5) / width - 1, 1 - 2 * (y + 0.5) / height)
            in_mesh = mesh and all(edge(TRIANGLE[i], TRIANGLE[(i + 1) % 3], p) >= 0
                                   for i in range(3))
            in_quad = quad and all(abs(axis) <= half for axis in p)
            pixel = CLEAR
            if in_mesh:
                pixel = MESH
            if in_quad and (quad_after or not in_mesh):
                pixel = color
            offset = (y * width + x) * 4
            rgba[offset:offset + 4] = bytes(pixel)
            mask[y * width + x] = int(stable(p))
    return bytes(rgba), bytes(mask)


def pixel_failures(case: str, actual: bytes, expected: bytes, supplied_mask: bytes) -> list[str]:
    oracle, mask = image(case)
    if expected != oracle:
        return [f"{case}: expected image differs from independent geometry/material oracle"]
    if supplied_mask != mask:
        return [f"{case}: supplied mask differs from the fixed independent mask"]
    if len(actual) != len(oracle):
        return [f"{case}: readback byte length is invalid"]
    for index, covered in enumerate(mask):
        if covered and actual[index * 4:index * 4 + 4] != oracle[index * 4:index * 4 + 4]:
            return [f"{case}: stable pixel differs at {index % 96},{index // 96}"]
    return []


def geometry_bytes(kind: str, scale: float = 1.0) -> bytes:
    """Independent interleaved position/UV/white-color input specification."""
    if kind == "mesh":
        vertices = [(x, y, 0.0, 0.0, 0.0) for x, y in TRIANGLE]
    else:
        vertices = [(-1., -1., 0., 0., 1.), (1., -1., 0., 1., 1.),
                    (1., 1., 0., 1., 0.), (-1., -1., 0., 0., 1.),
                    (1., 1., 0., 1., 0.), (-1., 1., 0., 0., 0.)]
    return b"".join(struct.pack("<9f", x * scale, y * scale, z, u, v, 1., 1., 1., 1.)
                    for x, y, z, u, v in vertices)
