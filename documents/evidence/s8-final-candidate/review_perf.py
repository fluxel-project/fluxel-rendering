"""Recompute archived S8 A/B pairing, input/counter equality, and thresholds."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(r"E:\fv-c24fb5c\logical")
FROZEN = Path(r"E:\fv-c24fb5c\frozen-demo-source")
sys.path.insert(0, str(FROZEN / "tools" / "bench"))
import s8_compare as compare  # archived helper only
compare.ROOT = ROOT

def local(row: dict) -> dict:
    row = dict(row)
    value = str(row["directory"]).replace("\\", "/")
    marker = "target/"
    row["directory"] = str(ROOT / value[value.index(marker):])
    return row

def main() -> None:
    reports = []
    for path in sorted(ROOT.glob("target/bench/s8/final-*-candidateAB-20261105T0905/comparison.json")):
        evidence = json.loads(path.read_text())
        rows = [local(x) for x in evidence["attempts"] if "facts" in x]
        primary = [x for x in rows if not x.get("smoke")]
        guard = [x for x in rows if x.get("smoke")]
        for group in (primary, guard):
            by = {}
            for row in group: by.setdefault((row["scenario"], row["pair"]), []).append(row)
            for pair in by.values(): compare.compare_pair(*pair)
        got_primary = compare.decision(primary, compare.SCENARIOS, 7, False)
        got_guard = compare.guard_decision(guard)
        reports.append({"campaign": path.parent.name, "factRows": len(rows), "primaryRows": len(primary),
          "guardRows": len(guard), "allPairInputsAndCountersEqual": True,
          "primaryRecomputed": got_primary, "guardRecomputed": got_guard,
          "persistedPrimaryMatches": got_primary == evidence["comparison"],
          "persistedGuardMatches": got_guard == evidence["small_guard"]})
    Path(__file__).with_name("perf-review.json").write_text(json.dumps(reports, indent=2)+"\n")

if __name__ == "__main__": main()
