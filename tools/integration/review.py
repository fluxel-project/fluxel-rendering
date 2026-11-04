#!/usr/bin/env python3
"""Review two explicit, current-source S0-S3 native integration attempts.

This compatibility entry point re-exports the established reviewer helpers while
keeping the evidence checks organized by responsibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REVIEW_DIR = Path(__file__).resolve().parent
if str(_REVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(_REVIEW_DIR))

from review_parts.common import *
from review_parts.raster import *
from review_parts.shader import *
from review_parts.workloads import *
from review_parts.reports import *
from review_parts.evidence import *
from review_parts.common import _SMOKE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--dx12-attempt", required=True)
    parser.add_argument("--wgl-attempt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path)
    parser.add_argument("--test-evidence", type=Path, action="append", default=[])
    parser.add_argument("--fault-review", type=Path,
                        help="reviewed native failure/loss/device-replacement evidence")
    parser.add_argument("--readset-review", type=Path,
                        help="reviewed material-like read-set and alias/import evidence")
    args = parser.parse_args()
    manifest_path = args.closure / "manifest.json"
    manifest, error = read_json(manifest_path)
    if error or not isinstance(manifest, dict):
        print(f"cannot read closure manifest: {error}", file=sys.stderr)
        return 2
    checks: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for backend, identifier in (("dx12", args.dx12_attempt), ("native-gl-wgl", args.wgl_attempt)):
        attempt, reasons = locate_attempt(manifest, identifier, backend)
        checks.append(result(f"explicit {backend} attempt selection", attempt is not None, reasons, attempt=identifier))
        if attempt is not None:
            selected[backend] = attempt
            checks.extend(attempt_review(attempt, args.closure, ROOT))
    for attempt in selected.values():
        checks.append(diagnostics_review(attempt, args.closure))
        checks.append(capability_review(attempt, args.closure))
        checks.append(measurement_review(attempt, args.closure))
    checks.append(external_review(args.visual_review, set(selected_identifier for selected_identifier in (args.dx12_attempt, args.wgl_attempt)), "reviewed visual output", selected, args.closure))
    test_reviews = [external_review(path, set((args.dx12_attempt, args.wgl_attempt)), "linked focused test evidence", selected, args.closure) for path in args.test_evidence]
    if not test_reviews:
        checks.append(result("linked focused test evidence", False, ["no test evidence links supplied"], count=0))
    else:
        checks.extend(test_reviews)
    identifiers = {args.dx12_attempt, args.wgl_attempt}
    checks.append(external_review(args.fault_review, identifiers,
                                  "reviewed native failure, loss, and device-replacement evidence", selected, args.closure))
    checks.append(external_review(args.readset_review, identifiers,
                                  "reviewed read-set, alias, import, and plan-reuse evidence", selected, args.closure))
    evidence_complete = bool(checks) and all(item["status"] == "passed" for item in checks)
    report = {"schema_version": 1, "scope": "S0-S3 bounded integration evidence review",
              "selected_attempts": {key: value.get("id") for key, value in selected.items()},
              "checks": checks,
              "evidence_status": "all_required_evidence_present" if evidence_complete else "open",
              "historical_foundation_support_decision": "not_consulted_for_current_evidence",
              "stage_support_decision": "not_made_by_this_tool",
              "stage_closure_decision": "not_made_by_this_tool"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if evidence_complete else 1


if __name__ == "__main__":
    sys.exit(main())
