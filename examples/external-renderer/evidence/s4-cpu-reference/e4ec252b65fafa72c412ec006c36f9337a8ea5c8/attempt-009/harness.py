#!/usr/bin/env python3
"""Capture immutable, locked S4 CPU-reference evidence for one exact commit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


COMMIT = "e4ec252b65fafa72c412ec006c36f9337a8ea5c8"
ACTIVE_DIRECTORY: Path | None = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_hashes(directory: Path) -> None:
    hashes = [
        {"path": str(path.relative_to(directory)), "sha256": sha256(path)}
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "hashes.json"
    ]
    write_json(directory / "hashes.json", hashes)


def git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def capture(name: str, command: list[str], directory: Path, cwd: Path) -> dict[str, object]:
    result = subprocess.run(command, text=True, capture_output=True, check=False, cwd=cwd)
    stdout = f"{name}.stdout.log"
    stderr = f"{name}.stderr.log"
    (directory / stdout).write_text(result.stdout, encoding="utf-8")
    (directory / stderr).write_text(result.stderr, encoding="utf-8")
    return {
        "name": name,
        "command": command,
        "cwd": str(cwd),
        "exit_code": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def main() -> int:
    global ACTIVE_DIRECTORY
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="attempt-001")
    parser.add_argument("--commit", default=COMMIT)
    parser.add_argument("--validate-stale", action="store_true")
    parser.add_argument("--simulate-test-failure", action="store_true")
    parser.add_argument("--simulate-exception", action="store_true")
    args = parser.parse_args()

    example = Path(__file__).resolve().parents[1]
    repository = example.parents[1]
    actual_commit = git(repository, "rev-parse", "HEAD")
    if actual_commit != args.commit:
        raise RuntimeError(f"refusing to label evidence for {args.commit} while HEAD is {actual_commit}")

    directory = example / "evidence" / "s4-cpu-reference" / args.commit / args.attempt
    if directory.exists():
        raise RuntimeError(f"evidence attempt already exists: {directory}")
    directory.mkdir(parents=True)
    ACTIVE_DIRECTORY = directory
    harness_copy = directory / "harness.py"
    shutil.copy2(Path(__file__).resolve(), harness_copy)

    product_paths = git(
        repository,
        "ls-tree",
        "-r",
        "--name-only",
        args.commit,
        "Cargo.toml",
        "Cargo.lock",
        "crates/renderer",
        "crates/material",
        "crates/shader",
        "examples/external-renderer",
    ).splitlines()
    product_paths = [
        path
        for path in product_paths
        if "/evidence/" not in path
        and "/scripts/" not in path
        and (path.endswith(".rs") or path.endswith("Cargo.toml") or path.endswith("Cargo.lock"))
    ]
    records = []
    for relative in product_paths:
        path = repository / relative
        if not path.is_file():
            raise RuntimeError(f"tracked target input is absent: {relative}")
        expected_blob = git(repository, "rev-parse", f"{args.commit}:{relative}")
        current_blob = git(repository, "hash-object", "--", relative)
        if current_blob != expected_blob:
            raise RuntimeError(f"tracked target input differs from {args.commit}: {relative}")
        records.append({"path": relative, "git_blob": expected_blob, "sha256": sha256(path)})
    write_json(directory / "input.json", {
        "target_commit": args.commit,
        "tracked_inputs": records,
        "harness": {"path": "harness.py", "sha256": sha256(harness_copy)},
    })

    commands = [
        ["cargo", "+1.87.0", "test", "--locked"],
        ["cargo", "+1.87.0", "run", "--locked"],
    ]
    write_json(directory / "expected.json", {
        "commit": args.commit,
        "commands": commands,
        "harness": "The runner is separately hashed in input.json and is not represented as a target-commit blob.",
        "expected": "Both commands exit zero; test includes the public-facade consumer and independent private-Graph compile-fail fixture.",
    })
    if args.simulate_exception:
        raise RuntimeError("intentional runner exception validation")

    if args.simulate_test_failure:
        commands[0] = ["cargo", "+1.87.0", "test", "--locked", "--package", "external-renderer-does-not-exist"]
    raw_evidence = example / "target" / "external-renderer-evidence"
    stale_marker = raw_evidence / ".capture-stale-marker"
    if args.validate_stale:
        raw_evidence.mkdir(parents=True, exist_ok=True)
        stale_marker.write_text("stale\n", encoding="utf-8")
    if raw_evidence.exists():
        shutil.move(str(raw_evidence), str(directory / "previous-raw-evidence"))

    results = [capture("test", commands[0], directory, example), capture("run", commands[1], directory, example)]
    test_stdout = (directory / "test.stdout.log").read_text(encoding="utf-8")
    match = re.search(r"running (\d+) tests", test_stdout)
    test_count = int(match.group(1)) if match else 0
    results[0]["discovered_tests"] = test_count
    if results[1]["exit_code"] == 0:
        if not (raw_evidence / "object-adapter").is_dir() or not (raw_evidence / "ecs-adapter").is_dir():
            raise RuntimeError("run succeeded without object/ECS raw evidence")
        shutil.copytree(raw_evidence, directory / "raw-evidence")
    if args.validate_stale:
        if not (directory / "previous-raw-evidence" / ".capture-stale-marker").is_file():
            raise RuntimeError("stale raw evidence was not isolated before the run")
        if (directory / "raw-evidence" / ".capture-stale-marker").exists():
            raise RuntimeError("stale raw evidence leaked into the new run")
    write_json(directory / "actual.json", results)
    failures = [result for result in results if result["exit_code"] != 0]
    if test_count == 0:
        failures.append({"name": "test-discovery", "detail": "runner observed zero tests"})
    if failures:
        write_json(directory / "error.json", failures)
    else:
        (directory / "error.txt").write_text("no error\n", encoding="utf-8")

    write_hashes(directory)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        if ACTIVE_DIRECTORY is not None:
            write_json(ACTIVE_DIRECTORY / "error.json", {"unhandled_exception": repr(error)})
            write_hashes(ACTIVE_DIRECTORY)
        raise
