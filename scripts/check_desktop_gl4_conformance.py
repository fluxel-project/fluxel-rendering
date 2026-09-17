#!/usr/bin/env python3
"""Gate a real desktop GL-family context against what the crate promises of it.

Why this exists
---------------
Six of the nine command-domain capabilities are settled by a probe that needs a
real driver, and nothing that is not a real driver can exercise them.  465 crate
tests passed while three of those probes asked for a draw and a dispatch with no
program current: every fixture they had answered the capability from a table
rather than from a context, so the only thing that could ever disagree was
hardware.  The defect was found by running ``examples/windows-gl4`` by hand and
reading the JSON it printed.  This checker is that reading, automated.

What it deliberately does not do
--------------------------------
It creates no window, owns no message pump, and knows nothing about WGL -- the
fixture does that, and it takes its drawable from whatever host provides one
over the standard handle traits.  It also re-derives nothing the report states:
the requirements below are what a desktop core profile is *entitled* to be asked
for, so a disagreement between this file and the crate is a review finding
rather than a second implementation of discovery competing with the first.

Fail-closed
-----------
A required capability that is false, a required fact that is missing or empty,
or a report that does not open all fail the run.  ``RECORDED_OPEN`` is the one
subtlety: those rows are open items in the series plan, so they are expected to
be *exactly* as recorded, and one that starts answering differently fails the
run too -- not because the new answer is worse, but because a ledger entry
changed and nobody adjudicated it.  Silence is never a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


DEFAULT_MANIFEST = Path("examples/windows-gl4/Cargo.toml")
DEFAULT_OUT = Path("target/gl4-conformance/report.json")

# Core-mandated at OpenGL 4.3, which is the floor of the desktop profile this
# gate opens: compute shaders, shader storage buffers and indirect dispatch are
# core in 4.3, indirect draw in 4.0, image load/store in 4.2, and timer queries
# in 3.3.  A conformant desktop core context that reports 4.3 therefore owes all
# six, which is why they can be required rather than merely recorded.
REQUIRED_CAPABILITIES = (
    "compute",
    "storage-buffer",
    "storage-image",
    "indirect-draw",
    "indirect-dispatch",
    "timer-query",
)

# Rows this gate must not silently bless.  Each is an open item in
# ``0.15-plan.md`` with its own stable number, and the recorded value is what the
# ledger currently claims, so the gate fails when the claim stops matching the
# context rather than when the claim is inconvenient.
RECORDED_OPEN = {
    "multi-draw-indirect": (False, "P2-15"),
    "multi-draw": (False, "P2-15"),
    "multiview": (False, "P2-14"),
}
RECORDED_SURFACE_FACTS = ("unavailable", "P1-15")

REQUIRED_SCALARS = (
    "profile",
    "version",
    "shading_language_version",
    "vendor",
    "renderer",
    "driver_or_browser",
    "owner_thread",
)
REQUIRED_LISTS = (
    "reported_extensions",
    "typed_extensions",
    "capabilities",
    "limits",
    "other_flags",
)
REQUIRED_FLAGS = ("opened", "debug", "forward_compatible", "robust_access", "no_error")

# The two alignment rows are the trap P0-8 closed: they are moduli, not
# capacities, so a context is *better* the smaller they are and a floor on them
# rejects the drivers it should prefer.  They are required to be present and
# positive so that the row cannot disappear unnoticed.
REQUIRED_LIMITS = (
    "max_texture_size",
    "uniform_buffer_offset_alignment",
    "storage_buffer_offset_alignment",
    "max_multiview_view_count",
)
DESKTOP_TEXTURE_FLOOR = 16_384
MIN_REPORTED_EXTENSIONS = 32
MIN_TYPED_EXTENSIONS = 4
PROFILE = re.compile(r"major:\s*(\d+)\s*,\s*minor:\s*(\d+)")
DESKTOP_CORE_FLOOR = (4, 3)


def extract_json(stdout: str) -> dict:
    """Returns the one JSON object the fixture printed.

    The fixture prints a single flat object and nothing else, but a build line
    can still reach stdout if the toolchain decides to be helpful, so the object
    is taken by its own braces rather than by assuming a clean stream.
    """
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError("the fixture printed no JSON object")
    parsed = json.loads(stdout[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"the fixture printed a {type(parsed).__name__}, not an object")
    return parsed


def check(report: dict, expected_extent: tuple[int, int] | None = None) -> list[str]:
    """Returns every way the report contradicts the crate's own contract."""
    problems: list[str] = []

    for flag in REQUIRED_FLAGS:
        if not isinstance(report.get(flag), bool):
            problems.append(f"{flag} is missing or not a boolean")
    for name in REQUIRED_SCALARS:
        value = report.get(name)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{name} is missing or empty")
    for name in REQUIRED_LISTS:
        value = report.get(name)
        if not isinstance(value, list) or not value:
            problems.append(f"{name} is missing, empty, or not a list")

    if report.get("opened") is not True:
        problems.append("the context did not open")
        return problems

    profile = report.get("profile", "")
    if not isinstance(profile, str) or not profile.startswith("Desktop"):
        problems.append(f"the profile is {profile!r}, not a desktop profile")
    else:
        matched = PROFILE.search(profile)
        if not matched:
            problems.append(f"the profile {profile!r} names no version")
        elif (int(matched.group(1)), int(matched.group(2))) < DESKTOP_CORE_FLOOR:
            problems.append(
                f"the profile {profile!r} is below the desktop core floor "
                f"{DESKTOP_CORE_FLOOR[0]}.{DESKTOP_CORE_FLOOR[1]}"
            )

    extent = report.get("drawable_extent")
    if (
        not isinstance(extent, list)
        or len(extent) != 2
        or not all(isinstance(part, int) and part > 0 for part in extent)
    ):
        problems.append(f"drawable_extent is {extent!r}, not two positive integers")
    elif expected_extent is not None and tuple(extent) != expected_extent:
        problems.append(f"drawable_extent is {tuple(extent)}, not the requested {expected_extent}")

    problems.extend(_check_extensions(report))
    problems.extend(_check_capabilities(report))
    problems.extend(_check_limits(report))

    recorded, row = RECORDED_SURFACE_FACTS
    surface_facts = report.get("surface_facts")
    if surface_facts != recorded:
        problems.append(
            f"surface_facts is {surface_facts!r}, not the recorded {recorded!r} of {row} -- "
            f"adjudicate it and update the ledger and this gate together"
        )
    return problems


def _check_extensions(report: dict) -> list[str]:
    problems: list[str] = []
    reported = report.get("reported_extensions")
    if isinstance(reported, list):
        if report.get("reported_extension_count") != len(reported):
            problems.append(
                f"reported_extension_count is {report.get('reported_extension_count')} "
                f"but the list holds {len(reported)}"
            )
        if len(reported) < MIN_REPORTED_EXTENSIONS:
            problems.append(
                f"only {len(reported)} extensions were reported, below the "
                f"{MIN_REPORTED_EXTENSIONS} a desktop driver lists"
            )
        blank = [index for index, name in enumerate(reported) if not isinstance(name, str) or not name]
        if blank:
            problems.append(f"reported_extensions has unnamed entries at {blank[:4]}")
    typed = report.get("typed_extensions")
    if isinstance(typed, list) and len(typed) < MIN_TYPED_EXTENSIONS:
        problems.append(
            f"only {len(typed)} extensions are typed, below the {MIN_TYPED_EXTENSIONS} "
            f"this crate models for every profile"
        )
    return problems


def _check_capabilities(report: dict) -> list[str]:
    problems: list[str] = []
    rows = report.get("capabilities")
    if not isinstance(rows, list):
        return problems
    resolved: dict[str, object] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != 2:
            problems.append(f"capabilities holds a row that is not a pair: {row!r}")
            continue
        name, value = row
        if not isinstance(name, str) or not isinstance(value, bool):
            problems.append(f"capabilities holds a row that is not (name, bool): {row!r}")
            continue
        if name in resolved:
            problems.append(f"capability {name} is reported twice")
        resolved[name] = value

    for name in REQUIRED_CAPABILITIES:
        if name not in resolved:
            problems.append(f"capability {name} is absent from the report")
        elif resolved[name] is not True:
            problems.append(
                f"capability {name} is false, but a desktop core "
                f"{DESKTOP_CORE_FLOOR[0]}.{DESKTOP_CORE_FLOOR[1]} context owes it"
            )
    for name, (recorded, row) in RECORDED_OPEN.items():
        if name not in resolved:
            problems.append(f"capability {name} is absent from the report")
        elif resolved[name] != recorded:
            problems.append(
                f"capability {name} is now {resolved[name]} where the ledger records "
                f"{recorded} ({row}) -- adjudicate it and update the plan and this gate together"
            )
    return problems


def _check_limits(report: dict) -> list[str]:
    problems: list[str] = []
    rows = report.get("limits")
    if not isinstance(rows, list):
        return problems
    facts: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != 2 or not all(isinstance(part, str) for part in row):
            problems.append(f"limits holds a row that is not a pair of strings: {row!r}")
            continue
        facts[row[0]] = row[1]

    for name in REQUIRED_LIMITS:
        if name not in facts:
            problems.append(f"limit {name} is absent from the report")
        elif not facts[name]:
            problems.append(f"limit {name} is empty")
    if "uniform_buffer_offset_alignment" in facts:
        try:
            alignment = int(facts["uniform_buffer_offset_alignment"])
        except ValueError:
            problems.append(
                f"uniform_buffer_offset_alignment is {facts['uniform_buffer_offset_alignment']!r}, "
                f"not a number"
            )
        else:
            if alignment <= 0:
                problems.append(f"uniform_buffer_offset_alignment is {alignment}, not a modulus")
    if "max_texture_size" in facts:
        try:
            texture = int(facts["max_texture_size"])
        except ValueError:
            problems.append(f"max_texture_size is {facts['max_texture_size']!r}, not a number")
        else:
            if texture < DESKTOP_TEXTURE_FLOOR:
                problems.append(
                    f"max_texture_size is {texture}, below the desktop floor "
                    f"{DESKTOP_TEXTURE_FLOOR}"
                )
    return problems


def run_fixture(manifest: Path, frames: int, extent: tuple[int, int], timeout: float) -> tuple[dict, bytes]:
    """Runs the fixture and returns the report it printed, with its raw bytes."""
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--offline",
            "--quiet",
            "--locked",
            "--manifest-path",
            str(manifest),
            "--",
            "--frames",
            str(frames),
            "--extent",
            f"{extent[0]}x{extent[1]}",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"the fixture exited {completed.returncode}: {completed.stderr.strip()[-400:]}"
        )
    return extract_json(completed.stdout), completed.stdout.encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=None,
                        help=f"fixture manifest, default {DEFAULT_MANIFEST}")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--extent", default="640x480", help="WIDTHxHEIGHT")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=None,
                        help=f"where to keep the report, default {DEFAULT_OUT}")
    parser.add_argument("--report", type=Path, default=None,
                        help="check a retained report instead of opening a context")
    arguments = parser.parse_args()

    try:
        width, height = (int(part) for part in arguments.extent.lower().split("x"))
    except ValueError:
        parser.error("--extent has to look like 640x480")

    root = arguments.root.resolve()
    if arguments.report is not None:
        raw = arguments.report.read_bytes()
        report = extract_json(raw.decode("utf-8"))
        source = arguments.report
    else:
        manifest = arguments.manifest or (root / DEFAULT_MANIFEST)
        report, raw = run_fixture(manifest, arguments.frames, (width, height), arguments.timeout)
        source = arguments.out or (root / DEFAULT_OUT)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(raw)

    digest = hashlib.sha256(raw).hexdigest()
    print(f"report {source} ({len(raw)} bytes, sha256 {digest})")
    print(
        f"  profile {report.get('profile')} on {report.get('renderer')}, "
        f"{report.get('reported_extension_count')} extensions"
    )
    for row in report.get("capabilities") or []:
        if isinstance(row, list) and len(row) == 2:
            print(f"  {'ok ' if row[1] else 'no '} {row[0]}")

    # A retained report is checked for internal validity only: its drawable was
    # chosen by whoever captured it, so comparing it against this invocation's
    # default extent would fail every report that used a different one.
    requested = (width, height) if arguments.report is None else None
    problems = check(report, requested)
    if problems:
        print("\nthe context contradicts the contract:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("\ndesktop GL-family conformance passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
