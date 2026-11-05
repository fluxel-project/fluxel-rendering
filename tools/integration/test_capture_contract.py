"""Regression cases for capture metadata, independent of the native capture code."""
from __future__ import annotations

import unittest

try:
    from . import capture_contract
except ImportError:
    import capture_contract


def current() -> dict[str, object]:
    state = {"foregroundHwnd": 101, "visible": True, "minimized": False, "topmost": False}
    return {
        "schemaVersion": 2, "frame": 3, "clientExtent": [320, 180],
        "method": "windows-graphics-capture-window", "pixelFormat": "BGRA8-top-down",
        "image": "surface-client-frame-3.bmp", "captureContentExtent": [336, 219],
        "clientCrop": [8, 31, 320, 180], "captureTimeoutMs": 2000,
        "windowStateBefore": state, "windowStateAfter": dict(state),
        "windowStateUnchanged": True, "automaticVisualPass": False,
    }


class CaptureContractTests(unittest.TestCase):
    def test_current_window_capture_passes(self) -> None:
        value = current()
        self.assertEqual(capture_contract.capture_failures(value, 3, (320, 180), value["image"]), [])
        self.assertEqual(capture_contract.verdict_failures({
            "schemaVersion": 2, "frame": 3, "image": value["image"],
            "oracle": "window-graphics-capture-corners-exact-and-stable-triangle-interior", "status": "passed",
        }, 3, value["image"], 2), [])

    def test_legacy_schema_remains_reviewable(self) -> None:
        value = current()
        value.update({"schemaVersion": 1})
        for key in ("method", "captureContentExtent", "clientCrop", "captureTimeoutMs", "windowStateBefore", "windowStateAfter", "windowStateUnchanged"):
            value.pop(key)
        self.assertEqual(capture_contract.capture_failures(value, 3, (320, 180), value["image"]), [])

    def test_changed_foreground_or_topmost_is_rejected(self) -> None:
        value = current()
        value["windowStateAfter"] = {**value["windowStateAfter"], "foregroundHwnd": 202}
        value["windowStateUnchanged"] = False
        self.assertTrue(capture_contract.capture_failures(value, 3, (320, 180), value["image"]))
        value = current()
        value["windowStateBefore"] = {**value["windowStateBefore"], "topmost": True}
        value["windowStateAfter"] = dict(value["windowStateBefore"])
        self.assertTrue(capture_contract.capture_failures(value, 3, (320, 180), value["image"]))

    def test_bad_crop_and_schema2_verdict_are_rejected(self) -> None:
        value = current()
        value["clientCrop"] = [8, 31, 321, 180]
        self.assertTrue(capture_contract.capture_failures(value, 3, (320, 180), value["image"]))
        self.assertTrue(capture_contract.verdict_failures({"schemaVersion": 1, "frame": 3,
                         "image": value["image"], "status": "passed"}, 3, value["image"], 2))

    def test_window_mask_excludes_exact_corners_and_never_mutates_raw_mask(self) -> None:
        width, height = 40, 40
        raw_mask = bytes([1]) * (width * height)
        window_mask = capture_contract.window_stable_mask(raw_mask, [width, height])
        self.assertEqual(raw_mask, bytes([1]) * (width * height))
        self.assertEqual(window_mask[(height - 1) * width], 0)
        self.assertEqual(window_mask[(height - 1) * width + width - 1], 0)
        self.assertEqual(window_mask[(height - 17) * width], 1)
        self.assertEqual(window_mask[(height - 1) * width + 16], 1)

    def test_window_mask_keeps_nearby_background_mutations_rejectable(self) -> None:
        from s7_oracles.core import check

        width, height = 40, 40
        expected = bytearray(width * height * 4)
        center = (height // 2) * width + width // 2
        expected[center * 4:center * 4 + 4] = b"\xff\0\0\xff"
        captured = bytearray(expected)
        captured[center * 4:center * 4 + 4] = b"\0\0\xff\xff"
        captured[(height - 1) * width * 4 + 16 * 4:(height - 1) * width * 4 + 16 * 4 + 4] = b"\xff\xff\xff\xff"
        with self.assertRaisesRegex(ValueError, "background differs"):
            check(captured, expected, capture_contract.window_stable_mask(bytes([1]) * (width * height), [width, height]), bmp_mode=True)
