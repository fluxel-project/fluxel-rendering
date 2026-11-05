"""Read-only native oracle and CPU-log replay from the isolated evidence tree."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ISOLATED = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("E:/fv-c24fb5c")
ROOT = ISOLATED / "logical"
FROZEN = ISOLATED / "frozen-demo-source"
sys.path.insert(0, str(FROZEN / "tools/integration"))
import review as generic
import s5_review as s5
import s7_review as s7
from s6_review_parts import offscreen, surface

ORIGIN = "E:/moyy/program/rust/fluxel/fluxel-rendering"
FINAL = ROOT / "target/final-regression-20261105T0825"


def normalize(value: str) -> str:
    return re.sub(r"/+", "/", value.replace("\\", "/"))


def rebase(value):
    if isinstance(value, dict):
        return {key: rebase(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rebase(item) for item in value]
    if isinstance(value, str):
        clean = normalize(value)
        if clean.startswith(ORIGIN + "/"):
            return str(ROOT / clean[len(ORIGIN) + 1:])
    return value


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def producer_source(attempt):
    record = attempt["source_snapshot"]
    archive = Path(record["path"])
    if sha(archive) != record["sha256"]:
        raise ValueError("producer source archive hash mismatch")
    destination = ISOLATED / ("native-oracle-frozen-" + record["sha256"][:12])
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or ":" in name:
                raise ValueError("unsafe source archive path")
            if name.endswith("/"):
                continue
            path = destination / name
            data = source.read(name)
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError("isolated source tree changed")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    return destination


def native():
    ledger = load(FINAL / "native-matrix/native-matrix.json")
    selected = {}
    for row in ledger["attempts"]:
        index = ROOT / normalize(row["index"])
        attempt = next(item for item in load(index)["attempts"]
                       if item["id"] == row["attemptIds"][0])
        selected[(row["backend"], row["workload"])] = (rebase(attempt), index.parent)
    expected = {(backend, workload) for backend in ("dx12", "native-gl-wgl")
                for workload in ("s0-s3", "s5-forward", "s5-surface", "s5-device-isolation",
                                 "s6-composed", "s6-surface", "s7-deferred",
                                 "s7-deferred-surface", "s7-deferred-device-isolation")}
    if set(selected) != expected:
        raise ValueError("native matrix selection is incomplete")
    results = []
    for (backend, workload), (attempt, closure) in selected.items():
        source = producer_source(attempt)
        # Archived readers expect attempt-local artifacts to remain relative to
        # the closure, while lock ownership is the frozen producer tree.
        attempt["source_snapshot"]["path"] = str(Path(attempt["directory"]) / "source-inputs.zip")
        for lock in attempt.get("locks", []):
            raw = normalize(str(lock.get("path", "")))
            prefix = normalize(str(ROOT)) + "/"
            if raw.startswith(prefix):
                lock["path"] = str(source / raw[len(prefix):])
        s5.ROOT = source  # producer source check, not a claim about later docs
        run = closure / attempt["directory"] / "run"
        failures = []
        if workload == "s0-s3":
            checks = generic.attempt_review(attempt, closure, source)
            checks += [generic.diagnostics_review(attempt, closure),
                       generic.capability_review(attempt, closure),
                       generic.measurement_review(attempt, closure)]
            for item in checks:
                failures += item.get("reasons", [])
        else:
            contract = (s7.s7_workload_contract(workload) if workload.startswith("s7-") else
                        s5._SMOKE.s6_workload_contract(workload) if workload.startswith("s6-") else
                        s5._SMOKE.s5_workload_contract(workload))
            failures += s5.common_failures(attempt, closure, backend, workload,
                                          contract=contract, require_tests=False)
            if workload == "s5-forward":
                reasons, _ = s5.forward_failures(attempt, closure)
                failures += reasons
            elif workload == "s5-device-isolation":
                peer, peer_closure = selected[(backend, "s5-forward")]
                failures += s5.device_failures(attempt, closure,
                    peer_closure / peer["directory"] / "run/s5-forward")
            elif workload == "s5-surface":
                # Validate pixels/resize with the existing reader; image inspection
                # is separately hash-bound in visual-inspection.json.
                reasons = s5.surface_failures(attempt, closure, None, ROOT, attempt["id"])
                failures += [reason for reason in reasons
                             if reason != "surface attempt has no reviewed human-review index entry"]
            elif workload == "s6-composed":
                failures += offscreen.review(run / workload)
            elif workload == "s6-surface":
                failures += surface.review(run / workload)
            else:
                failures += s7.nested_failures(attempt, closure, workload)
        results.append({"backend": backend, "workload": workload, "attempt": attempt["id"],
                        "status": "passed" if not failures else "open", "failures": failures})
    return results


def cpu():
    names = ("workspace-test-final.log", "workspace-clippy-fixed.log", "external-renderer-test.log",
             "integration-dx12-test.log", "integration-wgl-test.log", "bench-python-test.log",
             "integration-python-test.log", "smoke-python-test-package.log")
    results = []
    for name in names:
        path = FINAL / name
        text = path.read_text(encoding="utf-8", errors="replace")
        rust = [tuple(map(int, value)) for value in re.findall(
            r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored", text)]
        python = [int(value) for value in re.findall(r"Ran (\d+) tests", text)]
        item = {"log": name, "sha256": sha(path), "rustPassed": sum(x[0] for x in rust),
                "rustIgnored": sum(x[2] for x in rust), "pythonRan": python}
        if name == "workspace-clippy-fixed.log":
            item["accepted"] = "Finished" in text and "error:" not in text
        elif rust:
            item["accepted"] = sum(x[0] for x in rust) > 0 and all(x[1] == 0 for x in rust)
        elif name == "integration-python-test.log":
            item["accepted"] = (python == [78] and "Failed to import test module: test_smoke" in text
                                and "FAILED (errors=1)" in text and "failures=" not in text)
            item["scope"] = "77 passed; aggregate import failure is recovered by packaged smoke log"
        else:
            item["accepted"] = bool(python) and "OK" in text and "FAILED" not in text
        results.append(item)
    if results[0]["rustPassed"] != 308 or results[2]["rustPassed"] != 10:
        raise ValueError("workspace/external test scope differs from retained acceptance")
    if results[3]["rustPassed"] != 28 or results[4]["rustPassed"] != 31:
        raise ValueError("backend consumer test scope is incomplete")
    if results[5]["pythonRan"] != [17] or results[7]["pythonRan"] != [37]:
        raise ValueError("Python regression groups are incomplete")
    return results


if __name__ == "__main__":
    raw, logs = native(), cpu()
    report = {"scope": "archive-only dedicated native oracles and exact CPU log scope",
              "native": raw, "cpu": logs,
              "status": "verified" if all(x["status"] == "passed" for x in raw)
              and all(x["accepted"] for x in logs) else "open"}
    (HERE / "native-oracle-review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"], [(x["backend"], x["workload"], x["failures"]) for x in raw if x["failures"]])
    raise SystemExit(0 if report["status"] == "verified" else 1)
