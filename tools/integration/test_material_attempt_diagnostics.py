"""Focused contract tests for material-attempt native diagnostics."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import material_attempt_diagnostics as diagnostics


class MaterialAttemptDiagnosticsTests(unittest.TestCase):
    def attempt(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "attempt"
        (root / "workload").mkdir(parents=True)
        (root / "stderr.log").write_text("", encoding="utf-8")
        return root

    def write(self, attempt: Path, name: str, payload: dict[str, object]) -> None:
        (attempt / "workload" / name).write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def wgl_record(**changes: object) -> dict[str, object]:
        record: dict[str, object] = {
            "schemaVersion": 1,
            "installed": True,
            "debugOutputAvailable": True,
            "activationMarkerObserved": True,
            "logSaturated": False,
            "activationMessages": ['DebugMessageLogEntry { source: 33354, msg_type: 33384, id: 1180190028, severity: 33387, message: "fluxel-rendering WGL diagnostic observer active" }'],
            "nativeDiagnostics": [],
        }
        record.update(changes)
        return record

    def test_unrequested_is_explicitly_unverified(self) -> None:
        self.assertEqual(
            diagnostics.inspect(self.attempt(), "dx12", False),
            {"requested": False, "status": "unverified"},
        )

    def test_missing_record_fails(self) -> None:
        result = diagnostics.inspect(self.attempt(), "dx12", True)
        self.assertNotEqual(result["status"], "observed")
        self.assertTrue(result["errors"])

    def test_dx12_requires_activation_and_rejects_native_message(self) -> None:
        attempt = self.attempt()
        self.write(
            attempt,
            "dx12-debug-request.json",
            {
                "schemaVersion": 1,
                "requested": True,
                "deviceOpenCompleted": True,
                "activationContract": f"fluxel-rhi {diagnostics.RHI_REVISION}",
            },
        )
        self.assertEqual(diagnostics.inspect(attempt, "dx12", True)["status"], "observed")
        (attempt / "stderr.log").write_text("D3D12 [error] invalid descriptor\n", encoding="utf-8")
        self.assertEqual(diagnostics.inspect(attempt, "dx12", True)["status"], "failed")

    def test_wgl_valid_record_is_observed(self) -> None:
        attempt = self.attempt()
        self.write(attempt, "wgl-debug-observer.json", self.wgl_record())
        self.assertEqual(diagnostics.inspect(attempt, "native-gl-wgl", True)["status"], "observed")

    def test_wgl_missing_marker_fails(self) -> None:
        attempt = self.attempt()
        self.write(
            attempt,
            "wgl-debug-observer.json",
            self.wgl_record(activationMarkerObserved=False, activationMessages=[]),
        )
        self.assertEqual(diagnostics.inspect(attempt, "native-gl-wgl", True)["status"], "failed")

    def test_native_or_plain_text_cannot_forge_returned_marker(self) -> None:
        for text in [diagnostics.ACTIVATION_MARKER,
                     diagnostics.ACTIVATION_RECORD.replace("source: 33354", "source: 33350")]:
            attempt = self.attempt()
            self.write(attempt, "wgl-debug-observer.json", self.wgl_record(activationMessages=[text]))
            self.assertEqual(diagnostics.inspect(attempt, "native-gl-wgl", True)["status"], "failed")

    def test_wgl_unknown_saturation_fails(self) -> None:
        attempt = self.attempt()
        self.write(attempt, "wgl-debug-observer.json", self.wgl_record(logSaturated=None))
        self.assertEqual(diagnostics.inspect(attempt, "native-gl-wgl", True)["status"], "failed")

    def test_wgl_native_message_fails(self) -> None:
        attempt = self.attempt()
        self.write(
            attempt,
            "wgl-debug-observer.json",
            self.wgl_record(nativeDiagnostics=["GL_INVALID_OPERATION"]),
        )
        self.assertEqual(diagnostics.inspect(attempt, "native-gl-wgl", True)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
