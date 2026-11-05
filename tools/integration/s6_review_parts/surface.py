"""Independent raw-artifact review for the S6 native Surface workload."""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any

if __package__:
    from . import common, oracle
else:
    import common  # type: ignore[no-redef]
    import oracle  # type: ignore[no-redef]
try:
    from .. import capture_contract
except ImportError:
    import capture_contract  # type: ignore[no-redef]


_CASES = ("mesh-after-quad-initial", "quad-after-mesh-resized")
_EXTENTS = ((320, 180), (256, 144))
_ORDERS = (-10, 10)
_COLORS = (oracle.MESH, oracle.QUAD)
_LIFECYCLE = (
    (0, "configure", None, (320, 180)),
    (1, "acquire", 0, (320, 180)),
    (2, "complete", 0, (320, 180)),
    (3, "resize", None, (256, 144)),
    (4, "configure", None, (256, 144)),
    (5, "acquire", 1, (256, 144)),
    (6, "complete", 1, (256, 144)),
)


def _object(path: Path, failures: list[str]) -> dict[str, Any]:
    value, errors = common.read_object(path)
    failures.extend(errors)
    return value


def _bytes(path: Path, failures: list[str]) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        failures.append(f"cannot read {path.name}: {error}")
        return b""


def _assets(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"geometry", "material", "texture"}:
        return False
    return all(isinstance(asset, dict) and all(
        type(asset.get(key)) is int and asset[key] >= 0
        for key in ("slot", "slotGeneration", "contentGeneration")) for asset in value.values())


def _input_failures(run: Path) -> list[str]:
    failures: list[str] = []
    versions: list[int] = []
    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    for frame, (extent, order, color) in enumerate(zip(_EXTENTS, _ORDERS, _COLORS)):
        value = _object(run / f"s6-surface-frame-{frame}-input.json", failures)
        if not value:
            continue
        scene = value.get("sceneVersion")
        versions.append(scene if type(scene) is int else -1)
        if (value.get("schemaVersion") != 1
                or value.get("featureSnapshotSceneVersion") != scene
                or value.get("featureEnabled") is not True
                or value.get("surfaceExtent") != list(extent)
                or value.get("companionExtent") != [96, 96]
                or value.get("surfaceInteriorRgba") != list(color)):
            failures.append(f"frame {frame} input does not retain its actual frame facts")
        mesh, marker = value.get("mesh"), value.get("marker")
        if (not isinstance(mesh, dict) or mesh.get("bounds") != {
                "minimum": [-0.5, -0.5, 0.0], "maximum": [0.5, 0.5, 0.0]}
                or mesh.get("transformColumns") != identity or not _assets(mesh.get("assets"))):
            failures.append(f"frame {frame} input lacks actual mesh bounds, transform, or assets")
        if (not isinstance(marker, dict) or marker.get("id") != 7
                or marker.get("center") != [0.0, 0.0, 0.0]
                or marker.get("halfExtent") != [0.55, 0.55]
                or marker.get("tint") != [1.0, 0.0, 0.0, 1.0]
                or marker.get("order") != order or not _assets(marker.get("assets"))):
            failures.append(f"frame {frame} input lacks actual marker state or assets")
    if len(versions) == 2 and not versions[1] > versions[0]:
        failures.append("Surface resize did not produce a newer scene snapshot")
    return failures


def _bmp(path: Path, extent: tuple[int, int]) -> tuple[bytes | None, str | None]:
    try:
        value = path.read_bytes()
    except OSError as error:
        return None, str(error)
    if len(value) < 54 or value[:2] != b"BM":
        return None, "missing BMP signature/header"
    size, _, _, offset = struct.unpack_from("<IHHI", value, 2)
    header_size, width, height, planes, bits, compression, image_size = struct.unpack_from(
        "<IiiHHII", value, 14)
    if (size != len(value) or offset != 54 or header_size != 40
            or (width, height) != (extent[0], -extent[1]) or planes != 1 or bits != 32
            or compression != 0 or image_size != extent[0] * extent[1] * 4
            or len(value) != offset + image_size):
        return None, "BMP dimensions, top-down format, or byte count is invalid"
    return value[offset:], None


def _capture_failures(run: Path, frame: int, extent: tuple[int, int]) -> list[str]:
    failures: list[str] = []
    pixels, error = _bmp(run / f"surface-client-frame-{frame}.bmp", extent)
    if error:
        return [f"frame {frame} native capture: {error}"]
    capture = _object(run / f"surface-client-frame-{frame}.capture.json", failures)
    verdict = _object(run / f"surface-client-frame-{frame}.verdict.json", failures)
    failures += [f"frame {frame} {reason}" for reason in capture_contract.capture_failures(
        capture, frame, extent, f"surface-client-frame-{frame}.bmp")]
    failures += [f"frame {frame} {reason}" for reason in capture_contract.verdict_failures(
        verdict, frame, f"surface-client-frame-{frame}.bmp", capture.get("schemaVersion"))]
    expected, mask = oracle.image(f"frame-{frame}", *extent)
    # Only schema-2 Windows Graphics Capture BMPs include the two DWM rounded
    # bottom corners. Raw RHI evidence and schema-1 historical BMPs use their
    # original complete stable masks.
    visual_mask = (capture_contract.window_stable_mask(mask, extent)
                   if capture.get("schemaVersion") == 2 else mask)
    assert pixels is not None
    for index, stable in enumerate(visual_mask):
        if stable and pixels[index * 4:index * 4 + 3] != expected[index * 4:index * 4 + 3][::-1]:
            failures.append(f"frame {frame} native BMP differs from geometry oracle at {index % extent[0]},{index // extent[0]}")
            break
    return failures


def _lifecycle_failures(run: Path) -> list[str]:
    try:
        rows = [json.loads(line) for line in (run / "s6-surface-lifecycle.jsonl").read_text(
            encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        return [f"missing or malformed Surface lifecycle evidence: {error}"]
    actual = [(row.get("sequence"), row.get("event"), row.get("frame"),
               tuple(row.get("extent", []))) for row in rows if isinstance(row, dict)]
    return [] if actual == list(_LIFECYCLE) else [
        "Surface lifecycle does not prove configure/acquire/complete/resize ordering"]


def _host_failures(run: Path) -> list[str]:
    failures: list[str] = []
    try:
        events = [json.loads(line) for line in (run / "surface-events.jsonl").read_text(
            encoding="utf-8").splitlines() if line]
        configs = [json.loads(line) for line in (run / "surface-configuration.jsonl").read_text(
            encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        return [f"missing or malformed Host lifecycle evidence: {error}"]
    if len(events) != 1 or events[0].get("requestedExtent") != [256, 144] or events[0].get(
            "clientExtent") != [256, 144] or events[0].get("selectedEvent") not in {"resized", "restored"}:
        failures.append("Host resize event does not confirm the resized client extent")
    if [row.get("extent") for row in configs] != [[320, 180], [256, 144]] or any(
            row.get("configurationSupport") is not True for row in configs):
        failures.append("Surface configuration evidence is missing supported initial/resized configurations")
    value = _object(run / "s6-surface-reconfigure-input.json", failures)
    if value.get("previousExtent") != [320, 180] or value.get("observedClientExtent") != [256, 144] or value.get("nextOperation") != "configure":
        failures.append("Surface reconfigure input is not bound to the observed resize")
    return failures


def review(run: Path) -> list[str]:
    """Review one `run/s6-surface` directory without trusting its oracle files."""
    failures: list[str] = []
    manifest = _object(run / "s6-surface-workload.json", failures)
    if (manifest.get("schemaVersion") != 1 or manifest.get("workload") != "s6-composed-surface-resize"
            or manifest.get("initialExtent") != [320, 180] or manifest.get("resizedExtent") != [256, 144]
            or manifest.get("surfaceProof") != "native-client-capture"
            or manifest.get("readbackProof") != "same-snapshot-offscreen-companion"
            or manifest.get("featureCallbacks") != {"prepare": 4, "contribute": 4}):
        failures.append("S6 Surface manifest lacks the bounded workload contract")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or [case.get("name") for case in cases if isinstance(case, dict)] != list(_CASES):
        failures.append("S6 Surface manifest does not retain both ordered cases")
    failures += common.initial_asset_failures(run)
    failures += _input_failures(run)
    for frame in range(2):
        expected = _bytes(run / f"s6-surface-frame-{frame}-expected.rgba", failures)
        actual = _bytes(run / f"s6-surface-frame-{frame}-readback.rgba", failures)
        mask = _bytes(run / "s6-surface-stable-mask.bin", failures)
        failures += oracle.pixel_failures(f"frame-{frame}", actual, expected, mask)
        failures += common.preview_failures(run / f"s6-surface-frame-{frame}-expected.ppm", expected)
        failures += common.preview_failures(run / f"s6-surface-frame-{frame}-readback.ppm", actual)
        failures += _capture_failures(run, frame, _EXTENTS[frame])
    failures += _lifecycle_failures(run)
    failures += _host_failures(run)
    return failures
