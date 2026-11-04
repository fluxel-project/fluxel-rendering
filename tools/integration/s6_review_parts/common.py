"""Shared raw-input and visual provenance checks for S6."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import s5_review as base
from .oracle import MESH, geometry_bytes


def read_object(path: Path) -> tuple[dict[str, Any], list[str]]:
    value, error = base.read_json(path)
    if error or not isinstance(value, dict):
        return {}, [f"missing or malformed object: {path.name}: {error}"]
    return value, []


def initial_asset_failures(run: Path) -> list[str]:
    failures = []
    for kind, texture in (("mesh", bytes(MESH)), ("quad", bytes([255] * 4))):
        expected = {
            f"s6-initial-{kind}-geometry-stream-0.bin": geometry_bytes(kind),
            f"s6-initial-{kind}-texture.rgba": texture,
        }
        for name, data in expected.items():
            try:
                if (run / name).read_bytes() != data:
                    failures.append(f"initial {kind} input differs from independent specification: {name}")
            except OSError as error:
                failures.append(str(error))
        try:
            material = (run / f"s6-initial-{kind}-material.txt").read_text(encoding="utf-8")
            required = ("textureLoad(material_texture", "* tint") if kind == "mesh" else (
                "textureLoad(quad_texture", "tint.bgr * 0.75", "vec3<f32>(0.125)", "* remapped")
            compact = "".join(material.split())
            if not all("".join(token.split()) in compact for token in required):
                failures.append(f"initial {kind} material does not retain the selected texture/tint function")
        except OSError as error:
            failures.append(str(error))
    return failures


def preview_failures(path: Path, raw: bytes, width: int = 96, height: int = 96) -> list[str]:
    try:
        expected = f"P6\n{width} {height}\n255\n".encode() + b"".join(
            raw[index:index + 3] for index in range(0, len(raw), 4))
        if path.read_bytes() != expected:
            return [f"preview is not the retained readback RGB: {path.name}"]
    except OSError as error:
        return [str(error)]
    return []


def visual_offscreen_failures(human: dict[str, Any], human_base: Path, identifier: str,
                             attempt: dict[str, Any], closure: Path,
                             run: Path, names: list[str]) -> list[str]:
    if human.get("status") != "reviewed" or identifier not in human.get("attempt_ids", []):
        return ["offscreen attempt has no reviewed visual index entry"]
    entries = [item for item in human.get("evidence", []) if isinstance(item, dict)
               and item.get("attempt_id") == identifier]
    if len(entries) != 1 or not isinstance(entries[0].get("frames"), list):
        return ["offscreen visual review has no single typed frame set"]
    retained = base._REVIEW.retained_attempt_artifacts({identifier: attempt}, closure)
    seen = []
    failures = []
    for frame in entries[0]["frames"]:
        if not isinstance(frame, dict) or frame.get("visual_verdict") != "passed":
            failures.append("offscreen frame lacks an explicit passed visual verdict")
            continue
        name = frame.get("name")
        if name not in names:
            failures.append("offscreen visual review names an unknown scenario")
            continue
        seen.append(name)
        for key, suffix in (("raw", "actual.rgba"), ("preview", "actual.ppm")):
            record = frame.get(key)
            if not base._REVIEW.linked_raw_record(record, identifier, retained, human_base):
                failures.append(f"{name}: visual {key} is not a retained hash-bound artifact")
            elif isinstance(record, dict):
                path = Path(record["path"])
                path = path if path.is_absolute() else human_base / path
                if path.resolve() != (run / f"s6-{name}-{suffix}").resolve():
                    failures.append(f"{name}: visual {key} points at another scenario")
    if sorted(seen) != sorted(names):
        failures.append("visual review does not cover every offscreen scenario exactly once")
    return failures


def visual_surface_failures(review: dict[str, Any], review_base: Path, identifier: str,
                            attempt: dict[str, Any], closure: Path, run: Path) -> list[str]:
    if review.get("status") != "reviewed" or identifier not in review.get("attempt_ids", []):
        return ["Surface attempt has no reviewed visual index entry"]
    entries = [item for item in review.get("evidence", []) if isinstance(item, dict)
               and item.get("attempt_id") == identifier]
    if len(entries) != 1 or not isinstance(entries[0].get("frames"), list):
        return ["Surface visual review has no single typed frame set"]
    retained = base._REVIEW.retained_attempt_artifacts({identifier: attempt}, closure)
    seen = []
    failures = []
    for frame in entries[0]["frames"]:
        if not isinstance(frame, dict) or frame.get("visual_verdict") != "passed":
            failures.append("Surface frame lacks an explicit passed visual verdict")
            continue
        index = frame.get("index")
        if type(index) is not int or index not in (0, 1):
            failures.append("Surface visual review names an unknown frame")
            continue
        seen.append(index)
        for key, suffix in (("screenshot", "bmp"), ("capture", "capture.json"),
                            ("verdict", "verdict.json")):
            record = frame.get(key)
            if not base._REVIEW.linked_raw_record(record, identifier, retained, review_base):
                failures.append(f"Surface {index}: {key} is not a retained hash-bound artifact")
                continue
            path = Path(record["path"])
            path = path if path.is_absolute() else review_base / path
            if path.resolve() != (run / f"surface-client-frame-{index}.{suffix}").resolve():
                failures.append(f"Surface {index}: {key} points at another frame")
        for key, suffix in (("companion_raw", "rgba"), ("companion_preview", "ppm")):
            record = frame.get(key)
            if not base._REVIEW.linked_raw_record(record, identifier, retained, review_base):
                failures.append(f"Surface {index}: {key} is not a retained hash-bound artifact")
                continue
            path = Path(record["path"])
            path = path if path.is_absolute() else review_base / path
            if path.resolve() != (run / f"s6-surface-frame-{index}-readback.{suffix}").resolve():
                failures.append(f"Surface {index}: {key} points at another companion frame")
    if sorted(seen) != [0, 1]:
        failures.append("Surface visual review must cover both native captures exactly once")
    return failures
