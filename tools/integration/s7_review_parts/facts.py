"""Actual lifecycle facts, kept separate from the independent pixel oracle."""
from __future__ import annotations

import json
from pathlib import Path


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"{path.name} is not an object")
    return value


def review_offscreen(output: Path) -> list[str]:
    failures = []
    try:
        for path in output.glob("s7-*-facts.json"):
            value = _object(path)
            if "pendingBefore" in value:
                pending = [value.get(key) for key in ("pendingBefore", "pendingAfterDeferred", "pendingAfterDeferredReadback", "pendingAfterForward", "pendingAfterForwardReadback")]
                if pending != [0, 1, 0, 1, 0]: failures.append(f"{path.name}: completion ownership is not 0/1/0/1/0")
            if "passes" in value and value["passes"] not in (4, 5): failures.append(f"{path.name}: graph pass count is invalid")
            if "gbufferFormats" in value and value["gbufferFormats"] != ["Rgba8Unorm", "Rgba8Unorm", "Depth32Float"]:
                failures.append(f"{path.name}: actual G-buffer formats differ")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        failures.append(f"offscreen lifecycle facts unreadable: {error}")
    return failures


def review_surface(output: Path) -> list[str]:
    failures = []
    try:
        for path in output.glob("s7-deferred-surface-*-facts.json"):
            value = _object(path)
            if value.get("graphPasses") == 3:
                if value.get("pendingAfterComplete") != 0:
                    failures.append(f"{path.name}: Surface completion did not retire pending owner")
            elif value.get("graphPasses") == 4:
                if value.get("computeBeforeGeometry") is not True or value.get("pending") != [0, 1, 0, 1, 0]:
                    failures.append(f"{path.name}: deformation completion/compute ordering invalid")
            else:
                failures.append(f"{path.name}: Surface graph pass count invalid")
        lifecycle = output / "s7-deferred-surface-lifecycle.jsonl"
        events = [json.loads(line) for line in lifecycle.read_text(encoding="utf-8").splitlines() if line]
        if [item.get("event") for item in events] != ["configure", "acquire", "complete", "resize", "configure", "acquire", "complete"]:
            failures.append("Surface lifecycle does not retain configure/acquire/complete ordering")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        failures.append(f"Surface lifecycle facts unreadable: {error}")
    return failures


def review_device(output: Path) -> list[str]:
    """Check real two-device diagnostics, cache counters and the rejection."""
    failures = []
    try:
        summary = _object(output / "device-isolation-facts.json")
        if summary.get("oldDeviceId") == summary.get("newDeviceId") or not summary.get("sameCpuSceneAndAssets") or not summary.get("oldPendingRetired") or not summary.get("foreignSubmissionRejected"):
            failures.append("two-device summary does not prove isolation/rejection/retirement")
        rejected = _object(output / "foreign-submission-rejected-facts.json")
        if rejected.get("submission") != "rejected" or rejected.get("completion") != "not-consumed" or rejected.get("oldOwnerPending") is not True:
            failures.append("foreign submission was not rejected without consuming owner completion")
        warm = ("old-warm", "new-warm")
        for name in warm:
            facts = _object(output / f"{name}-facts.json"); before, after = facts.get("deviceStatsBefore", {}), facts.get("deviceStatsAfter", {})
            if facts.get("passes") != 4 or facts.get("pendingAfterReadback") != 0 or any(after.get(key, 0) <= before.get(key, 0) for key in ("geometryHits", "textureHits", "gbufferProgramHits", "lightingProgramHits")) or after.get("residentUploadBytes") != before.get("residentUploadBytes"):
                failures.append(f"{name}: actual cache/retirement facts do not prove warm device-local reuse")
        returned = _object(output / "old-return-facts.json"); before, after = returned.get("deviceStatsBefore", {}), returned.get("deviceStatsAfter", {})
        if returned.get("passes") != 4 or returned.get("pendingAfterReadback") != 0 or any(after.get(key, 0) <= before.get(key, 0) for key in ("geometryHits", "textureHits", "gbufferProgramHits", "lightingProgramHits", "graphHits")) or after.get("residentUploadBytes") != before.get("residentUploadBytes"):
            failures.append("old-return: settled owner did not hit its Graph cache")
        for name in ("old-cold", "new-cold"):
            facts = _object(output / f"{name}-facts.json"); before, after = facts.get("deviceStatsBefore", {}), facts.get("deviceStatsAfter", {})
            if facts.get("passes") != 4 or facts.get("pendingAfterReadback") != 0 or any(after.get(key, 0) != before.get(key, 0) for key in ("geometryHits", "textureHits", "gbufferProgramHits", "lightingProgramHits", "graphHits")) or after.get("residentUploadBytes", 0) <= before.get("residentUploadBytes", 0):
                failures.append(f"{name}: actual graph/completion facts invalid")
        for name in ("new-replacement", "old-replacement"):
            facts = _object(output / f"{name}-facts.json"); before, after = facts.get("deviceStatsBefore", {}), facts.get("deviceStatsAfter", {})
            if facts.get("passes") != 4 or facts.get("pendingAfterReadback") != 0 or after.get("textureHits") != before.get("textureHits") or after.get("residentUploadBytes", 0) <= before.get("residentUploadBytes", 0):
                failures.append(f"{name}: replacement did not allocate its new device-local texture")
        inputs = {name: _object(output / f"{name}-input.json") for name in ("old-cold", "old-warm", "new-cold", "new-warm", "old-return", "new-replacement", "old-replacement")}
        baseline = inputs["old-cold"].get("assetRefs")
        if not isinstance(baseline, dict):
            failures.append("old-cold has no serialized asset references")
            return failures
        if any(inputs[name].get("assetRefs") != baseline for name in ("old-warm", "new-cold", "new-warm", "old-return")):
            failures.append("same CPU scene/assets are not proven by actual serialized asset references")
        for name in ("new-replacement", "old-replacement"):
            replaced = inputs[name].get("assetRefs", {})
            old_texture, new_texture = baseline.get("texture", {}) if isinstance(baseline, dict) else {}, replaced.get("texture", {}) if isinstance(replaced, dict) else {}
            if replaced.get("geometry") != baseline.get("geometry") or replaced.get("material") != baseline.get("material") or old_texture.get("slot") != new_texture.get("slot") or old_texture.get("slotGeneration") != new_texture.get("slotGeneration") or new_texture.get("contentGeneration", 0) <= old_texture.get("contentGeneration", 0):
                failures.append(f"{name}: replacement identity/generation facts invalid")
    except (OSError, AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
        failures.append(f"device-isolation facts unreadable: {error}")
    return failures
