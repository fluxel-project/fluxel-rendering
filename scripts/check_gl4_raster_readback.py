#!/usr/bin/env python3
"""Gate the picture a real desktop GL-family context rasterized.

Why this exists
---------------
``0.15-plan.md`` records the native GL4 path as having *no image at all*: every
reading of it was a counter, a capability row or a driver string, and the series
review had to say so.  ``examples/windows-gl4 --draws N --readback PATH`` closes
that by driving one measured frame and reading its render target back, and this
checker is what reads the result: the raw bytes are compared against the picture
the workload's own kernel and geometry require, and a magnified top-down PNG is
written beside them so that a reviewer can *open* the frame rather than infer it
from hex (``CLAUDE.md`` §5).

What is left here, now that the picture moved out
-------------------------------------------------
The picture is one fact about the workload and both GL-family surfaces assert
it, so the expectations and the checks live in ``gl_raster_picture`` and are
re-exported here whole.  What is this file's own is the *runner*: a window, a
message pump and ``cargo run`` over the caller's drawable.  That half is WGL's
and cannot be shared with a device that has no window.

What it deliberately does not do
--------------------------------
It creates no window and knows nothing about WGL -- the fixture does that.  It
re-derives no part of the driver's report: the expected picture is what the
crate's raster kernel and the workload's geometry state, so a disagreement
between this file and the crate is a review finding rather than a second
implementation competing with the first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

# Every script in this directory is run both as a program and as a module by its
# test, which loads it by path; neither puts the directory itself on `sys.path`,
# so the one sibling import this file makes is set up here rather than left to
# whichever caller happened to be first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gl_raster_picture import (  # noqa: E402  (after the path it needs)
    EXPECTED_CLEAR,
    EXPECTED_COLOUR,
    EXPECTED_COVERED,
    EXPECTED_EXTENT,
    EXPECTED_ROW_ORDER,
    SCALE,
    check,
    extract_json,
    magnify,
    pixels_from_hex,
    write_png,
)

# The names above are this module's answer as much as the imported module's: the
# gate is one contract stated once, and a caller reading `check_gl4_raster_readback`
# should not have to know where the statement lives.  Listed rather than aliased
# one by one so that a name dropped from the import is a NameError here.
__all__ = [
    "EXPECTED_CLEAR",
    "EXPECTED_COLOUR",
    "EXPECTED_COVERED",
    "EXPECTED_EXTENT",
    "EXPECTED_ROW_ORDER",
    "SCALE",
    "check",
    "extract_json",
    "magnify",
    "pixels_from_hex",
    "write_png",
]

DEFAULT_MANIFEST = Path("examples/windows-gl4/Cargo.toml")
DEFAULT_OUT = Path("target/gl4-raster-readback")


def run_fixture(manifest: Path, draws: int, extent: tuple[int, int], colour: Path,
                timeout: float) -> dict:
    """Runs the fixture over a real drawable and returns the report it printed."""
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
            "--extent",
            f"{extent[0]}x{extent[1]}",
            "--draws",
            str(draws),
            "--readback",
            str(colour),
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
    return extract_json(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=None,
                        help=f"fixture manifest, default {DEFAULT_MANIFEST}")
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--extent", default="640x480", help="WIDTHxHEIGHT of the window")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=None,
                        help=f"where to keep the artifacts, default {DEFAULT_OUT}")
    arguments = parser.parse_args()

    try:
        width, height = (int(part) for part in arguments.extent.lower().split("x"))
    except ValueError:
        parser.error("--extent has to look like 640x480")

    root = arguments.root.resolve()
    manifest = arguments.manifest or (root / DEFAULT_MANIFEST)
    out = arguments.out or (root / DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=True)
    colour_path = out / "colour.rgba"

    try:
        report = run_fixture(manifest, arguments.draws, (width, height), colour_path,
                             arguments.timeout)
    except (RuntimeError, ValueError) as error:
        print(f"the run produced no report: {error}", file=sys.stderr)
        return 1
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    if not colour_path.exists():
        print(f"the run did not write {colour_path}", file=sys.stderr)
        return 1
    raw = colour_path.read_bytes()

    workload = report.get("workload") or {}
    context = workload.get("colour") or {}
    print(f"artifacts in {out}")
    print(
        f"  {report.get('renderer')} ({report.get('driver_or_browser')}), "
        f"profile {report.get('profile')}"
    )
    print(
        f"  {workload.get('draws_requested')} draw(s), mode {workload.get('mode')}, "
        f"target {context.get('extent')} {context.get('row_order')}"
    )

    problems = check(report, raw)
    if problems:
        print("\nthe frame contradicts what the workload requires:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    image_width, image_height, rows = magnify(raw, EXPECTED_EXTENT, SCALE)
    png_path = out / "colour.png"
    write_png(png_path, image_width, image_height, rows)
    print(
        f"\nthe frame is the picture the kernel and the geometry require: "
        f"{len(EXPECTED_COVERED)} covered pixels of {EXPECTED_COLOUR} on {EXPECTED_CLEAR}"
    )
    print(f"open {png_path} ({image_width}x{image_height}, top-down, x{SCALE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
