"""Synthetic process evidence for reviewer tests; never native acceptance evidence."""

from __future__ import annotations

import json
from pathlib import Path

from s5_review_tests.fixture import _diagnostic, _record, _snapshot, _source_tree, _write
from smoke_s6 import S6_SCHEMA_VERSION, S6_STAGE, s6_workload_contract
from .oracle import FRAME_CASES


def build(root: Path, base, populate_offscreen, populate_surface) -> dict:
    source = root / "source"
    inputs = _source_tree(source, base)
    selections = []
    attempts = []
    visual_entries = []
    for backend in ("dx12", "native-gl-wgl"):
        for workload, populate in (("s6-composed", populate_offscreen), ("s6-surface", populate_surface)):
            identifier = f"{backend}-{workload}"
            closure = root / identifier
            run = closure / "attempt" / "run" / workload
            run.parent.mkdir(parents=True)
            populate(run)
            contract = s6_workload_contract(workload)
            manifest_path = run / f"{workload}-workload.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            # The per-workload fixtures test semantic content. This ledger also
            # closes every file under a named case for the real manifest checker.
            names = sorted(path.name for path in run.iterdir() if path.is_file() and path != manifest_path)
            cases = contract["expected_cases"]
            manifest["cases"] = [{"name": name, "artifacts": names[index::len(cases)]}
                                 for index, name in enumerate(cases)]
            _write(manifest_path, manifest)
            marker = run.parent / ("dx12-debug-request.json" if backend == "dx12" else "wgl-debug-observer.json")
            _write(marker, _diagnostic(backend))
            executable = closure / "attempt" / "consumer.exe"
            lock = closure / "attempt" / "Cargo.lock"
            _write(executable, b"synthetic executable, not native evidence")
            _write(lock, b"synthetic lock")
            retained = [_record(base, path) for path in run.rglob("*") if path.is_file()]
            success = {"exit_status": 0, "timed_out": False, "child_process_lifecycle": "parent_exited"}
            commands = []
            for kind, command in (
                ("build", ["cargo", "build"]),
                ("direct_executable", [str(executable), contract["entry_argument"]]),
            ):
                log = closure / "attempt" / "logs" / f"{kind}.log"
                _write(log, b"synthetic process succeeded\n")
                result = {**success, "raw_logs": [_record(base, log)]}
                if kind == "direct_executable":
                    result.update(nested_workloads={"status": "verified"}, retained_files=retained)
                commands.append({"kind": kind, "command": command, "result": result})
            if not attempts:
                test_commands = [
                    ["cargo", "test", "-p", "fluxel-renderer", "--lib", "forward"],
                    ["cargo", "test", "--manifest-path", "../../crates/renderer/examples/external-renderer/Cargo.toml", "--lib"],
                    ["cargo", "test", "--manifest-path", "Cargo.toml", "cli"],
                    ["python", "-m", "unittest", "discover", "-p", "test_s6*.py"],
                ]
                for index, command in enumerate(test_commands):
                    log = closure / "attempt" / "logs" / f"test-{index}.log"
                    output = ("Ran 1 test in 0.001s\nOK\n" if command[0] == "python"
                              else "test fixture ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n")
                    _write(log, output.encode())
                    commands.append({"kind": "test", "command": command, "result": {
                        **success, "raw_logs": [_record(base, log)],
                        "test_command_recognized": True, "executed": 1,
                    }})
            attempt = {
                "id": identifier, "directory": "attempt", "mode": "run", "backend": backend,
                "runner_exit_status": 0, "workload": contract, "source_inputs": inputs,
                "source_snapshot": _snapshot(closure, inputs, source, base),
                "built_source_verification": {"status": "verified"},
                "built_git_checkout_verification": {"status": "verified"},
                "source_consistency_after_attempt": {"status": "stable"},
                "source_consistency_after_build": {"status": "stable"},
                "build_environment": {"status": "auditable"},
                "executable": {**_record(base, executable), "after_run_sha256": base._REVIEW.sha256(executable)},
                "locks": [_record(base, lock)], "commands": commands,
                "retained_attempt_files": retained,
                "validation": {"status": "observed", "native_diagnostic_count": 0,
                               "diagnostics": {"unreviewed_native_diagnostics": []},
                               "artifact": _record(base, marker)},
            }
            write_manifest(closure, attempt)
            frames = []
            if workload == "s6-composed":
                for name in FRAME_CASES:
                    frames.append({"name": name, "visual_verdict": "passed",
                                   "raw": _record(base, run / f"s6-{name}-actual.rgba"),
                                   "preview": _record(base, run / f"s6-{name}-actual.ppm")})
            else:
                for index in (0, 1):
                    frames.append({"index": index, "visual_verdict": "passed", **{
                        key: _record(base, run / f"surface-client-frame-{index}.{suffix}")
                        for key, suffix in (("screenshot", "bmp"), ("capture", "capture.json"),
                                            ("verdict", "verdict.json"))},
                        "companion_raw": _record(base, run / f"s6-surface-frame-{index}-readback.rgba"),
                        "companion_preview": _record(base, run / f"s6-surface-frame-{index}-readback.ppm"),
                    })
            visual_entries.append({"attempt_id": identifier, "frames": frames})
            selections.extend(["--attempt", str(closure), identifier, backend, workload])
            attempts.append((attempt, closure, run))
    visual = root / "visual.json"
    _write(visual, {"status": "reviewed", "reviewer": "synthetic test fixture",
                    "attempt_ids": [item["attempt_id"] for item in visual_entries],
                    "evidence": visual_entries})
    return {"source": source, "args": selections, "attempts": attempts,
            "visual": visual, "output": root / "report.json"}


def write_manifest(closure: Path, attempt: dict) -> None:
    _write(closure / "manifest.json", {"schema_version": S6_SCHEMA_VERSION,
                                      "stage": S6_STAGE, "attempts": [attempt]})
