"""Fail-closed metadata contracts for native client-window captures.

Schema 1 is retained for historical desktop-crop evidence.  Schema 2 is the
current Windows Graphics Capture evidence and additionally proves that capture
did not change the target window's focus-facing state.
"""
from __future__ import annotations

from typing import Any


_METHOD = "windows-graphics-capture-window"
_ORACLE = "window-graphics-capture-corners-exact-and-stable-triangle-interior"
WINDOW_DECORATION_CORNER_SIZE = 16
WINDOW_STABLE_MASK_POLICY = "exclude bottom two 16x16 corner squares from schema-2 window capture only"


def capture_failures(value: Any, frame: int, extent: tuple[int, int] | list[int], image: str) -> list[str]:
    """Validate one persisted capture descriptor without accepting schema drift."""
    if not isinstance(value, dict):
        return ["capture metadata is not an object"]
    if not isinstance(extent, (tuple, list)) or not _two_positive_ints(list(extent)):
        return ["capture reviewer received an invalid expected client extent"]
    expected_extent = list(extent)
    if (value.get("frame") != frame or value.get("clientExtent") != expected_extent
            or value.get("pixelFormat") != "BGRA8-top-down"
            or value.get("automaticVisualPass") is not False or value.get("image") != image):
        return ["capture metadata does not bind the actual BMP"]
    version = value.get("schemaVersion")
    if version == 1:
        # Existing immutable S5/S6/S7 evidence predates the non-disturbance
        # fields.  It remains reviewable but cannot claim those new facts.
        return []
    if version != 2 or value.get("method") != _METHOD:
        return ["capture metadata uses an unsupported schema or method"]
    crop = value.get("clientCrop")
    content = value.get("captureContentExtent")
    if (not _four_nonnegative_ints(crop) or not _two_positive_ints(content)
            or crop[2:] != expected_extent
            or crop[0] + crop[2] > content[0] or crop[1] + crop[3] > content[1]):
        return ["capture metadata has an invalid content/client crop"]
    if value.get("captureTimeoutMs") != 2000:
        return ["capture metadata does not retain the bounded capture timeout"]
    before, after = value.get("windowStateBefore"), value.get("windowStateAfter")
    if before != after or value.get("windowStateUnchanged") is not True:
        return ["capture changed the observed window state"]
    if not _non_disturbing_window_state(before):
        return ["capture did not retain a visible, non-minimized, non-topmost window state"]
    return []


def verdict_failures(value: Any, frame: int, image: str, capture_schema: Any) -> list[str]:
    """Bind an oracle result to its capture schema and retained BMP."""
    if not isinstance(value, dict) or value.get("frame") != frame or value.get("image") != image or value.get("status") != "passed":
        return ["capture verdict is absent or failed"]
    if capture_schema == 1:
        return []
    if capture_schema != 2 or value.get("schemaVersion") != 2 or value.get("oracle") != _ORACLE:
        return ["capture verdict does not identify the Windows Graphics Capture oracle"]
    return []


def window_stable_mask(mask: bytes, extent: tuple[int, int] | list[int]) -> bytes:
    """Return a window-only mask excluding frozen DWM bottom-corner antialiasing.

    This function never mutates `mask`.  It is for schema-2 captured HWND BMPs
    only; callers must retain the original mask for RHI raw/readback checks.
    """
    if not isinstance(extent, (tuple, list)) or not _two_positive_ints(list(extent)):
        raise ValueError("window stable mask requires a positive two-dimensional extent")
    width, height = extent
    if len(mask) != width * height:
        raise ValueError("window stable mask shape does not match the client extent")
    result = bytearray(mask)
    corner = min(WINDOW_DECORATION_CORNER_SIZE, width, height)
    for y in range(height - corner, height):
        for x in range(corner):
            result[y * width + x] = 0
        for x in range(width - corner, width):
            result[y * width + x] = 0
    return bytes(result)


def _two_positive_ints(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(type(item) is int and item > 0 for item in value)


def _four_nonnegative_ints(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(type(item) is int and item >= 0 for item in value)


def _non_disturbing_window_state(value: Any) -> bool:
    return (isinstance(value, dict) and type(value.get("foregroundHwnd")) is int
            and value.get("visible") is True and value.get("minimized") is False
            and value.get("topmost") is False)
