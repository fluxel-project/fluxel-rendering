"""Independent, archive-only structural audit for the S8 final evidence bundle.

This deliberately imports no workspace tooling.  It validates the content-addressed
archive, materializes it below an isolated review root, and checks the principal
ledger invariants without modifying producer bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "evidence.zip"
OUT = HERE / "archive-review.json"
REVIEW_ROOT = Path(r"E:\fv-c24fb5c")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe(name: str) -> bool:
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and ".." not in p.parts and "\\" not in name


def native(path: Path) -> str:
    """Use Windows extended paths for archive members beyond legacy MAX_PATH."""
    value = str(path.resolve())
    return "\\\\?\\" + value if os.name == "nt" else value


def materialize(z: zipfile.ZipFile, manifest: dict) -> Path:
    if REVIEW_ROOT.exists():
        logical = REVIEW_ROOT / "logical"
        if logical.is_dir():
            return logical
        raise RuntimeError(f"refusing incomplete review root: {REVIEW_ROOT}")
    blobs = REVIEW_ROOT / "blobs"
    logical = REVIEW_ROOT / "logical"
    blobs.mkdir(parents=True)
    for item in manifest["files"]:
        blob = item["blob"]
        destination = REVIEW_ROOT / blob
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with z.open(blob) as inp, open(native(destination), "wb") as out:
                shutil.copyfileobj(inp, out)
        target = logical / PurePosixPath(item["path"])
        os.makedirs(native(target.parent), exist_ok=True)
        try:
            os.link(native(destination), native(target))
        except OSError:
            # Some Windows configurations reject long hard-link destinations.
            # A byte-for-byte copy keeps the review isolated and correct.
            shutil.copyfile(native(destination), native(target))
    return logical


def source_zip_check(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        unsafe = [n for n in names if not safe(n)]
        duplicate = len(names) != len(set(names))
        # The extraction is a relocation proof and is deliberately separate from
        # producer paths.  It is not used to import current-workspace modules.
        extracted = REVIEW_ROOT / "frozen-demo-source"
        extracted.mkdir(parents=True, exist_ok=True)
        for info in z.infolist():
            if safe(info.filename):
                target = extracted / PurePosixPath(info.filename)
                os.makedirs(native(target.parent), exist_ok=True)
                with z.open(info) as inp, target.open("wb") as out:
                    shutil.copyfileobj(inp, out)
    return {"sha256": digest(path), "entries": len(names), "safe": not unsafe,
            "duplicates": duplicate, "extractedUnderReviewRoot": str(extracted.relative_to(REVIEW_ROOT))}


def read_json(root: Path, relative: str) -> dict:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def main() -> int:
    result = {"schemaVersion": 1, "reviewer": "archive-only Python stdlib audit",
              "archive": {"path": "evidence.zip", "bytes": ARCHIVE.stat().st_size,
                          "sha256": digest(ARCHIVE)}, "checks": {}, "open": []}
    with zipfile.ZipFile(ARCHIVE) as z:
        entries = z.namelist()
        manifest = json.loads(z.read("manifest.json"))
        files = manifest["files"]
        paths = [x["path"] for x in files]
        blob_names = {x["blob"] for x in files}
        bad = []
        for item in files:
            if not (safe(item["path"]) and safe(item["blob"])):
                bad.append(item["path"])
                continue
            info = z.getinfo(item["blob"])
            with z.open(info) as f:
                h = hashlib.sha256(f.read()).hexdigest()
            if h != item["sha256"] or info.file_size != item["bytes"]:
                bad.append(item["path"])
        result["checks"]["contentAddressedBundle"] = {
            "zipEntries": len(entries), "logicalFiles": len(files), "uniqueBlobs": len(blob_names),
            "allPathsSafe": all(safe(n) for n in entries) and not bad,
            "duplicateZipEntries": len(entries) != len(set(entries)),
            "duplicateLogicalPaths": len(paths) != len(set(paths)), "badOrMismatched": bad,
        }
        root = materialize(z, manifest)
    logical_bad = [x["path"] for x in files if not (root / PurePosixPath(x["path"])).is_file()
                   or digest(root / PurePosixPath(x["path"])) != x["sha256"]]
    result["checks"]["materialization"] = {"root": str(REVIEW_ROOT), "logicalFiles": len(files),
                                               "allLogicalBytesVerified": not logical_bad,
                                               "badLogicalFiles": logical_bad, "usesOriginalTarget": False}
    demo_zip = root / "target/final-regression-20261105T0825/demos/source-inputs.zip"
    result["checks"]["frozenDemoSource"] = source_zip_check(demo_zip)
    native = read_json(root, "target/final-regression-20261105T0825/native-matrix/native-matrix.json")
    demos = read_json(root, "target/final-regression-20261105T0825/demos/attempts.json")
    perf = read_json(root, "target/final-regression-20261105T0825/performance/comparisons.json")
    result["checks"]["ledgers"] = {
        "native": {"status": native.get("status"), "planned": native.get("planned"), "attempts": len(native.get("attempts", [])),
                   "allExitZero": all(x.get("exitCode") == 0 and x.get("status") == "complete" for x in native.get("attempts", []))},
        "demos": {"status": demos.get("status"), "backends": len(demos.get("backends", [])),
                  "routes": sum(len(x.get("demos", [])) for x in demos.get("backends", []))},
        "performance": {"status": perf.get("status"), "comparisons": len(perf.get("comparisons", []))},
    }
    comparisons = perf.get("comparisons", perf.get("attempts", []))
    detailed = [read_json(root, x["comparison"].replace("\\", "/")) for x in comparisons]
    result["checks"]["performanceShape"] = [{"backend": x.get("baseline", {}).get("backend"),
        "status": x.get("status"), "completeUniquePairs": x.get("comparison", {}).get("complete_unique_pairs"),
        "primaryAcceptance": x.get("comparison", {}).get("primary_acceptance"),
        "smallGuard": (x.get("small_guard") or {}).get("accepted"),
        "pairs": (x.get("parameters") or {}).get("pairs"), "attempts": len(x.get("attempts", []))} for x in detailed]
    lifecycle = [json.loads(p.read_text(encoding="utf-8")) for p in root.glob("target/bench/s8/final-lifecycle-*/attempt.json")]
    result["checks"]["lifecycle"] = [{"backend": x.get("backend"), "status": x.get("status"),
        "exitZero": (x.get("run") or {}).get("exit_status") == 0,
        "independentReview": (x.get("independent_review") or {}).get("status")} for x in lifecycle]
    # Logs are retained as evidence; parse them as text rather than rerunning tests.
    log = (root / "target/final-regression-20261105T0825/workspace-test-final.log").read_text(errors="replace")
    result["checks"]["globalCpuEvidence"] = {"workspaceLogHasOk": "test result: ok" in log,
        "note": "GPU attempt ledgers intentionally have no repeated logical test arrays; global logs are the CPU behavior evidence."}
    result["checks"]["visualEvidence"] = {"nativeInspection": (root / "target/final-regression-20261105T0825/native-matrix/visual-inspection.json").is_file(),
        "demoInspection": (root / "target/final-regression-20261105T0825/demos/visual-inspection.json").is_file(),
        "note": "Records bind inspected images; this automated reviewer does not claim human inspection."}
    result["checks"]["semanticReplay"] = {
        "demoLifecycleAndRawPixels": "semantic-review.json",
        "performancePairsCountersAndThresholds": "perf-review.json",
        "status": "separate archived-helper reports; native/source/CPU subgate is separate",
    }
    semantic = HERE / "semantic-review.json"
    perf_review = HERE / "perf-review.json"
    native_review = HERE / "native-source-review.json"
    native_result = json.loads(native_review.read_text(encoding="utf-8")) if native_review.is_file() else {}
    native_oracle = HERE / "native-oracle-review.json"
    native_oracle_result = json.loads(native_oracle.read_text(encoding="utf-8")) if native_oracle.is_file() else {}
    semantic_result = json.loads(semantic.read_text(encoding="utf-8")) if semantic.is_file() else {}
    perf_result = json.loads(perf_review.read_text(encoding="utf-8")) if perf_review.is_file() else []
    demos_ok = len(semantic_result.get("demos", [])) == 4 and all(x.get("status") == "passed" for x in semantic_result.get("demos", []))
    lifecycle_ok = len(semantic_result.get("lifecycle", [])) == 2 and all(x.get("status") == "passed" for x in semantic_result.get("lifecycle", []))
    raw_perf = semantic_result.get("performance", [])
    perf_ok = len(perf_result) == 2 and len(raw_perf) == 2 and all(x.get("runs") == 84
        and x.get("drawChecks") == 989800 and x.get("backgroundChecks") == 989800 for x in raw_perf) and all(x.get("factRows") == 84 and x.get("primaryRows") == 42
        and x.get("guardRows") == 42 and x.get("allPairInputsAndCountersEqual")
        and x.get("persistedPrimaryMatches") and x.get("persistedGuardMatches")
        and x.get("primaryRecomputed", {}).get("primary_acceptance") is True
        and x.get("guardRecomputed", {}).get("accepted") is True for x in perf_result)
    native_ok = len(native_oracle_result.get("native", [])) == 18 and all(x.get("status") == "passed" for x in native_oracle_result.get("native", []))
    cpu_ok = len(native_oracle_result.get("cpu", [])) == 8 and all(x.get("accepted") for x in native_oracle_result.get("cpu", []))
    result["checks"]["subgates"] = {
        "integrity": (not result["checks"]["contentAddressedBundle"]["badOrMismatched"] and not logical_bad
            and result["checks"]["contentAddressedBundle"]["allPathsSafe"]
            and not result["checks"]["contentAddressedBundle"]["duplicateZipEntries"]
            and not result["checks"]["contentAddressedBundle"]["duplicateLogicalPaths"]),
        "demoRawAndWindow": demos_ok,
        "performanceRawPairsCountersThresholds": perf_ok,
        "lifecycleRawFacts": lifecycle_ok,
        "nativeSource": native_result.get("status") == "verified" and not native_result.get("failures")
            and native_result.get("hashRecordsChecked") == 3486 and native_result.get("source", {}).get("frozenRuntimeFiles") == 284,
        "nativeOraclesAndCpu": native_oracle_result.get("status") == "verified" and native_ok and cpu_ok,
        "nativeHashRecords": native_result.get("hashRecordsChecked"),
    }
    result["open"] = [name for name, passed in result["checks"]["subgates"].items() if not passed]
    result["conclusion"] = "verified" if all(result["checks"]["subgates"].values()) else "integrity-pass; semantic-open"
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["conclusion"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
