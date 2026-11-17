"""Validate actual native diagnostic activation for a material attempt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ACTIVATION_MARKER = "fluxel-rendering WGL diagnostic observer active"
ACTIVATION_RECORD = (
    f'DebugMessageLogEntry {{ source: {0x824A}, msg_type: {0x8268}, '
    f'id: {0x4658454C}, severity: {0x826B}, message: "{ACTIVATION_MARKER}" }}'
)
RHI_REVISION = "8400f8f35933a5d0464014a8671799890f67ff49"


def inspect(attempt: Path, backend: str, requested: bool) -> dict[str, Any]:
    if not requested:
        return {"requested": False, "status": "unverified"}
    name = "dx12-debug-request.json" if backend == "dx12" else "wgl-debug-observer.json"
    path = attempt / "workload" / name
    result: dict[str, Any] = {
        "requested": True,
        "status": "unverified",
        "path": path.relative_to(attempt).as_posix(),
        "errors": [],
    }
    try:
        raw = path.read_bytes()
        result["sha256"] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("schemaVersion") != 1:
            raise ValueError("diagnostic activation record has an unsupported shape")
        if backend == "dx12":
            if data.get("requested") is not True or data.get("deviceOpenCompleted") is not True:
                result["errors"].append("DX12 debug activation was not requested before device open")
            contract = data.get("activationContract", "")
            if not isinstance(contract, str) or RHI_REVISION not in contract:
                result["errors"].append("DX12 activation contract does not name the selected RHI revision")
            stderr = (attempt / "stderr.log").read_text(encoding="utf-8", errors="replace")
            messages = [line for line in stderr.splitlines() if line.startswith("D3D12 [")]
            result["nativeDiagnostics"] = messages
            if messages:
                result["errors"].append("DX12 emitted native diagnostics")
        else:
            for flag in ("installed", "debugOutputAvailable", "activationMarkerObserved"):
                if data.get(flag) is not True:
                    result["errors"].append(f"WGL observer did not prove {flag}")
            if data.get("logSaturated") is not False:
                result["errors"].append("WGL diagnostic log saturation is unknown or true")
            activation = data.get("activationMessages")
            if not isinstance(activation, list) or not all(isinstance(x, str) for x in activation):
                result["errors"].append("WGL activation messages have an invalid shape")
            elif not any(entry == ACTIVATION_RECORD for entry in activation):
                result["errors"].append("WGL context did not return the activation marker")
            messages = data.get("nativeDiagnostics")
            result["nativeDiagnostics"] = messages
            if not isinstance(messages, list) or messages:
                result["errors"].append("WGL native diagnostic list is missing, malformed or nonempty")
        result["status"] = "observed" if not result["errors"] else "failed"
    except (OSError, ValueError, TypeError) as error:
        result["errors"].append(str(error))
    return result
