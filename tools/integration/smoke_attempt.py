#!/usr/bin/env python3
"""Process, log, and immutable-attempt helpers for smoke evidence."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any
if __package__:
    from .smoke_inputs import sha256
else:
    from smoke_inputs import sha256
def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_attempts(path: Path) -> list[dict[str, Any]]:
    try:
        attempts = json.loads(path.read_text(encoding="utf-8")).get("attempts", [])
        return attempts if isinstance(attempts, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def attempt_name(backend: str) -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{backend}-{timestamp}-{os.getpid()}"


def log_result(log_directory: Path, exit_status: int | None, timed_out: bool,
               lifecycle: str, error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exit_status": exit_status,
        "timed_out": timed_out,
        "child_process_lifecycle": lifecycle,
        "raw_logs": [
            {"path": str(log_directory / "stdout.log"), "sha256": sha256(log_directory / "stdout.log")},
            {"path": str(log_directory / "stderr.log"), "sha256": sha256(log_directory / "stderr.log")},
        ],
    }
    if error is not None:
        result["runner_error"] = error
    return result


def run_command(command: list[str], cwd: Path, timeout: float,
                log_directory: Path, environment: dict[str, str] | None = None) -> dict[str, Any]:
    """Write straight to files; a timeout leaves descendants explicitly unknown."""
    log_directory.mkdir(parents=True, exist_ok=True)
    stdout_path = log_directory / "stdout.log"
    stderr_path = log_directory / "stderr.log"
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                text=True,
                stdout=stdout,
                stderr=stderr,
                creationflags=creation_flags,
                env=environment,
            )
        except OSError as error:
            stderr.write(f"runner error: {error}\n")
            stderr.flush()
            return log_result(log_directory, None, False, "not_started", str(error))
        try:
            exit_status = process.wait(timeout=timeout)
            return log_result(log_directory, exit_status, False, "parent_exited")
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stderr.write(
                "runner timeout: parent termination was requested; descendant process "
                "lifecycle is unverified.\n"
            )
            stderr.flush()
            return log_result(log_directory, None, True, "unverified_after_parent_termination")


def actual_test_count(command: list[str], result: dict[str, Any]) -> int | None:
    """Accept only runner-executed test commands with an explicit nonzero count."""
    executable = Path(command[0]).name.lower() if command else ""
    cargo = executable in {"cargo", "cargo.exe"} and "test" in command[1:]
    unittest = executable.startswith("python") and len(command) >= 3 and command[1:3] == ["-m", "unittest"]
    if not cargo and not unittest:
        return None
    logs = result.get("raw_logs")
    if not isinstance(logs, list):
        return None
    text = "\n".join((Path(item.get("path", "")).read_text(encoding="utf-8", errors="replace")
                      for item in logs if isinstance(item, dict) and isinstance(item.get("path"), str)))
    counts = (re.findall(r"(?m)^test result: .*?\b(\d+) passed;", text) if cargo
              else re.findall(r"(?m)^Ran (\d+) tests?", text))
    return sum(int(count) for count in counts) if counts else None


def retained_files(directory: Path) -> list[dict[str, str]]:
    """List immutable attempt artifacts without interpreting their contents."""
    return [
        {"path": str(path), "sha256": sha256(path) or ""}
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
