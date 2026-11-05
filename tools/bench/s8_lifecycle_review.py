"""Independent raw-pixel review for non-timed material-heavy lifecycle cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import s8_material_oracle as oracle


CASES = ("device-old-cold", "pending-old", "pending-replacement",
         "device-new-cold", "device-old-return")


def review_pixels(root: Path, name: str) -> dict:
    if name not in CASES:
        raise ValueError("unknown lifecycle case")
    case_root = root / name
    case = json.loads((case_root / "lifecycle-input.json").read_text())
    if case.get("fixtureInputRoot") != "../inputs":
        raise ValueError("unexpected lifecycle source location")
    raw = (case_root / "raw.rgba").read_bytes()
    root = root / "inputs"
    source = json.loads((root / "s8-material-input.json").read_text())
    generation = "first-10pct-replaced" if name == "pending-replacement" else "original"
    if case.get("case") != name or case.get("generationSelection") != generation:
        raise ValueError("lifecycle input has the wrong case or generation selection")
    if source.get("extent") != [512, 512] or len(source.get("drawRecords", [])) != 10000:
        raise ValueError("lifecycle case is not the material-heavy 10,000 draw fixture")
    if (source.get("schemaVersion") != 1 or source.get("draws") != 10000
            or source.get("textures") != 1000 or source.get("allTexturesReferenced") is not True
            or source.get("scenario") != "replacement-10pct"):
        raise ValueError("lifecycle source descriptors do not identify the complete fixture")
    oracle._material_contract(root)
    records = source["drawRecords"]
    descriptors = source["textureGeneration1"]["textures"]
    if len(descriptors) != 1000:
        raise ValueError("lifecycle case does not reference 1,000 actual textures")
    if len({entry["id"] for entry in descriptors}) != 1000 or any(
            entry.get("index") != i or entry.get("rgbaOffset") != i * 4
            for i, entry in enumerate(descriptors)):
        raise ValueError("texture descriptors do not bind distinct retained texels")
    if (source["sharedGeometry"].get("bytes") != "s8-material-geometry.bin"
            or source["textureGeneration1"].get("bytes") != "s8-material-textures-generation-1.rgba"):
        raise ValueError("lifecycle source does not bind retained byte artifacts")
    oracle._geometry_contract((root / "s8-material-geometry.bin").read_bytes())
    initial = (root / "s8-material-textures-generation-1.rgba").read_bytes()
    if len(initial) != 4000:
        raise ValueError("wrong original texture byte extent")
    replacements = oracle._read_replacements(root, source, 1000)
    if len(replacements) != 100:
        raise ValueError("lifecycle case must retain exactly 10% replacement inputs")
    if len(raw) != 512 * 512 * 4:
        raise ValueError("wrong raw pixel extent")
    seen, textures = set(), set()
    for record in records:
        draw, texture = record["index"], record["textureIndex"]
        if type(draw) is not int or not 0 <= draw < 10000 or draw in seen:
            raise ValueError("duplicate or invalid draw identity")
        if type(texture) is not int or not 0 <= texture < 1000:
            raise ValueError("invalid texture identity")
        seen.add(draw); textures.add(texture)
        oracle._transform_contract(record["transformColumns"], draw, 10000)
        texel = (replacements.get(texture, initial[texture * 4:texture * 4 + 4])
                 if generation != "original" else initial[texture * 4:texture * 4 + 4])
        wanted = oracle._expected(texel, record["tint"], record["materialVariant"])
        for label, position, expected in (
            ("foreground", (-.45, -.45), wanted),
            ("background", (.55, .55), b"\0\0\0\0"),
        ):
            x, y = oracle._pixel_point(record["transformColumns"], *position, [512, 512])
            actual = raw[(y * 512 + x) * 4:(y * 512 + x + 1) * 4]
            valid = (actual == expected if label == "background" else
                     actual[3] == expected[3] and all(abs(actual[i] - expected[i]) <= 1 for i in range(3)))
            if not valid:
                raise ValueError(f"{name} {label} draw {draw}: {list(actual)} != {list(expected)}")
    if textures != set(range(1000)):
        raise ValueError("not every texture contributes checked pixels")
    return {"case": name, "drawChecks": len(seen), "textures": len(textures),
            "generationSelection": generation, "status": "passed"}


def review(root: Path) -> dict:
    def load(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    manifest_name = "s8-material-lifecycle-workload.json"
    manifest = load(manifest_name)
    if (manifest.get("schemaVersion") != 1 or manifest.get("workload") != "s8-material-lifecycle"
            or manifest.get("timingMetrics") != "excluded"):
        raise ValueError("wrong lifecycle manifest contract")
    cases = manifest.get("cases", [])
    if [case.get("name") for case in cases] != list(CASES):
        raise ValueError("lifecycle manifest omits or reorders required cases")
    listed = manifest.get("sharedArtifacts", []) + manifest.get("rootArtifacts", [])
    for case in cases:
        expected = [f"{case['name']}/{name}" for name in ("lifecycle-input.json", "raw.rgba", "facts.json")]
        if case.get("artifacts") != expected:
            raise ValueError("case artifact list differs from actual required inputs/raw/facts")
        listed += case["artifacts"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if len(set(listed)) != len(listed) or set(listed) | {manifest_name} != actual:
        raise ValueError("lifecycle manifest has duplicate, omitted or unlisted artifacts")
    pending = load("pending-lifecycle.json")
    if (pending.get("pending") != [1, 2, 2, 1, 0]
            or pending.get("foreignSubmissionRejected") is not True
            or pending.get("oldOwnerRetainedAcrossForeignRejection") is not True):
        raise ValueError("in-flight owners or foreign refusal are not proved")
    devices = load("device-cache.json")
    old_id, new_id = devices["oldDeviceId"], devices["newDeviceId"]
    if not old_id or not new_id or old_id == new_id:
        raise ValueError("lifecycle devices are not distinct")
    facts, pixels = {}, []
    for sequence, name in enumerate(CASES):
        case = load(f"{name}/lifecycle-input.json")
        if case.get("submitSequence") != sequence or case.get("deviceId") != (
                new_id if name == "device-new-cold" else old_id):
            raise ValueError("case has the wrong submit sequence or device")
        facts[name] = load(f"{name}/facts.json")
        expected_pending = {"pending-old": (0, 1, 1), "pending-replacement": (1, 2, 0)}.get(name, (0, 1, 0))
        if tuple(facts[name].get(key) for key in (
                "pendingBefore", "pendingAfterSubmit", "pendingAfterReadback")) != expected_pending:
            raise ValueError("case completion does not retire exactly its accepted owner")
        pixels.append(review_pixels(root, name))
    for name in ("device-old-cold", "device-new-cold"):
        before, after = facts[name]["cacheBefore"], facts[name]["cacheAfter"]
        if (after["geometryHits"] != before["geometryHits"]
                or after["textureHits"] != before["textureHits"]
                or after["acceptedTextureAllocations"] - before["acceptedTextureAllocations"] != 1001
                or after["opaqueDraws"] - before["opaqueDraws"] != 10000):
            raise ValueError("cold device adopted existing products or omitted real draws")
    returned = facts["device-old-return"]
    before, after = returned["cacheBefore"], returned["cacheAfter"]
    if (after["textureHits"] - before["textureHits"] != 1000
            or after["geometryHits"] - before["geometryHits"] != 1
            or after["acceptedTextureAllocations"] != before["acceptedTextureAllocations"]
            or after["residentCopyBytes"] != before["residentCopyBytes"]):
        raise ValueError("return to old device did not reuse its completed exact products")
    for name, case_name in (("newCold", "device-new-cold"), ("oldReturn", "device-old-return")):
        if devices[name] != {"before": facts[case_name]["cacheBefore"], "after": facts[case_name]["cacheAfter"]}:
            raise ValueError("root cache observations differ from per-case facts")
    return {"status": "passed", "cases": pixels, "pending": pending["pending"],
            "scope": "native high-cardinality replacement/device paths; fault injection is witness-only"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
