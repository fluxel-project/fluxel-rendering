from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("s5_review", Path(__file__).with_name("s5_review.py"))
assert SPEC and SPEC.loader
s5_review = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(s5_review)


class S5ReviewTests(unittest.TestCase):
    def test_pixel_oracle_rejects_shape_and_changed_stable_pixel(self) -> None:
        self.assertTrue(s5_review.pixel_failures(b"", b"", b""))
        expected, mask = s5_review.fixture_oracle(); actual = expected
        self.assertEqual(s5_review.pixel_failures(actual, expected, mask), [])
        changed = bytearray(actual); changed[4] = 1
        self.assertIn("stable pixel differs", s5_review.pixel_failures(bytes(changed), expected, mask)[0])

    def test_oracle_rejects_runner_supplied_expected_or_mask_tamper(self) -> None:
        expected, mask = s5_review.fixture_oracle(); tampered = bytearray(expected); tampered[0] = 1
        self.assertIn("independent fixture oracle", s5_review.pixel_failures(expected, bytes(tampered), mask)[0])
        self.assertIn("invalid RGBA/mask shape", s5_review.pixel_failures(expected, expected, bytes(bytearray(mask))[:-1])[0])

    def test_retained_artifact_must_have_matching_hash_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); raw = root / "raw"; raw.write_bytes(b"evidence")
            attempt = {"retained_attempt_files": [], "commands": []}
            self.assertTrue(s5_review.retained_failures(attempt, root, [raw]))
            attempt["retained_attempt_files"] = [{"path": str(raw), "sha256": s5_review._REVIEW.sha256(raw)}]
            self.assertEqual(s5_review.retained_failures(attempt, root, [raw]), [])

    def test_surface_rejects_missing_human_review_and_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); attempt = {"directory": "attempt", "commands": [], "retained_attempt_files": []}
            failures = s5_review.surface_failures(attempt, root, None, root, "surface")
            self.assertTrue(any("human-review" in item for item in failures))
            self.assertTrue(any("missing raw artifact" in item for item in failures))

    def test_human_review_must_bind_retained_capture_screenshot_and_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); run = root / "attempt" / "run" / "s5-forward-surface"; run.mkdir(parents=True)
            records = []
            frames = []
            for index in range(2):
                bmp = run / f"surface-client-frame-{index}.bmp"; bmp.write_bytes(b"BM")
                capture = run / f"surface-client-frame-{index}.capture.json"; capture.write_text(json.dumps({"image": bmp.name, "automaticVisualPass": False}))
                verdict = run / f"surface-client-frame-{index}.verdict.json"; verdict.write_text(json.dumps({"image": bmp.name, "status": "passed"}))
                triple = [bmp, capture, verdict]; records += [{"path": str(path), "sha256": s5_review._REVIEW.sha256(path)} for path in triple]
                frames.append({"screenshot": records[-3], "capture": records[-2], "verdict": records[-1], "human_verdict": "passed"})
            attempt = {"commands": [{"result": {"retained_files": records}}]}
            review = {"status": "reviewed", "attempt_ids": ["surface"], "evidence": [{"attempt_id": "surface", "frames": frames}]}
            self.assertEqual(s5_review.human_link_failures(review, root, "surface", attempt), [])
            frames[0]["screenshot"]["sha256"] = "forged"
            self.assertTrue(s5_review.human_link_failures(review, root, "surface", attempt))

    def test_bmp_oracle_rejects_triangle_interior_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); run = root / "attempt" / "run" / "s5-forward-surface"; run.mkdir(parents=True)
            width, height = 320, 180; pixels = bytearray(width * height * 4)
            x, y = width * 8 // 20, height * 57 // 100; pixels[(y * width + x) * 4:(y * width + x + 1) * 4] = b"\0\0\xff\xff"
            bmp = b"BM" + bytes(16) + width.to_bytes(4, "little", signed=True) + (-height).to_bytes(4, "little", signed=True) + bytes(28) + pixels
            (run / "surface-client-frame-0.bmp").write_bytes(bmp)
            attempt = {"directory": "attempt", "commands": [], "retained_attempt_files": []}
            failures = s5_review.surface_failures(attempt, root, None, root, "surface")
            self.assertFalse(any("red triangle interior" in item for item in failures))
            pixels[(y * width + x) * 4:(y * width + x + 1) * 4] = b"\0\0\0\xff"
            (run / "surface-client-frame-0.bmp").write_bytes(bmp[:54] + pixels)
            failures = s5_review.surface_failures(attempt, root, None, root, "surface")
            self.assertTrue(any("red triangle interior" in item for item in failures))


from s5_review_tests.main import S5ReviewMainTests


if __name__ == "__main__":
    unittest.main()
