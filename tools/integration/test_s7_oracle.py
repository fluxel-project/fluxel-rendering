from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

from s7_review_parts import oracle


class S7OracleTests(unittest.TestCase):
    def test_shape_rejects_inconsistent_raw_and_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); actual, expected, mask = root / "a", root / "e", root / "m"
            expected.write_bytes(bytes((0, 0, 0, 0, 10, 20, 30, 255)))
            actual.write_bytes(bytes((0, 0, 0, 0, 11, 20, 30))); mask.write_bytes(bytes((1, 1)))
            self.assertTrue(oracle.shape_failures(actual, expected, mask))

    def test_bmp_rejects_changed_stable_pixel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.bmp"
            header = bytearray(54); header[:2] = b"BM"; struct.pack_into("<I", header, 10, 54)
            struct.pack_into("<I", header, 14, 40); struct.pack_into("<ii", header, 18, 1, -1)
            struct.pack_into("<HH", header, 26, 1, 32); struct.pack_into("<I", header, 34, 4)
            path.write_bytes(header + bytes((2, 2, 2, 255)))
            self.assertTrue(oracle.bmp_failures(path, bytes((1, 2, 3, 255)), bytes((1,))))


if __name__ == "__main__":
    unittest.main()
