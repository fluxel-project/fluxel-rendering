"""Read-only semantic replay over archived final evidence only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(r"E:\fv-c24fb5c\logical")
FROZEN = Path(r"E:\fv-c24fb5c\frozen-demo-source")
sys.path.insert(0, str(FROZEN / "tools" / "integration"))
sys.path.insert(0, str(FROZEN / "tools" / "bench"))
import demo_review  # type: ignore # archived module, deliberately not workspace tooling
import s8_lifecycle_review  # type: ignore
import s8_material_oracle  # type: ignore


def demos() -> list[dict]:
    original = Path.write_bytes
    def readonly(self: Path, data: bytes) -> int:
        if not self.name.startswith("demo-review-"):
            raise RuntimeError(f"review attempted unexpected write: {self}")
        if not self.is_file() or self.read_bytes() != data:
            raise RuntimeError(f"derived demo byte mismatch: {self}")
        return len(data)
    Path.write_bytes = readonly
    try:
        demo_root = ROOT / "target" / "final-regression-20261105T0825" / "demos"
        roots = sorted({p.parent for p in demo_root.rglob("demo-positions.bin")})
        return [demo_review.review(p) for p in roots]
    finally:
        Path.write_bytes = original


def performance() -> list[dict]:
    results = []
    for campaign in sorted(ROOT.glob("target/bench/s8/final-*-candidateAB-20261105T0905")):
        # Input files occur once per actual native run; archived oracle rejects
        # missing/extra raws and rederives foreground/background pixels.
        runs = sorted({p.parent for p in campaign.rglob("s8-material-input.json")})
        checks = [s8_material_oracle.review_run(p) for p in runs]
        results.append({"campaign": campaign.name, "runs": len(checks),
                        "drawChecks": sum(x["drawChecks"] for x in checks),
                        "backgroundChecks": sum(x["backgroundChecks"] for x in checks),
                        "textures": sorted({x["textureReferences"] for x in checks})})
    return results


def lifecycle() -> list[dict]:
    roots = sorted(p.parent / "run" / "s8-material-lifecycle" for p in ROOT.glob("target/bench/s8/final-lifecycle-*/attempt.json"))
    return [s8_lifecycle_review.review(p) for p in roots]


def main() -> None:
    report = {"schemaVersion": 1, "scope": "archived helper replay, no build/test/GPU execution",
              "demos": demos(), "performance": performance(), "lifecycle": lifecycle()}
    (HERE / "semantic-review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
