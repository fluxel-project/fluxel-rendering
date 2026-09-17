#!/usr/bin/env python3
"""The picture the compatibility adapter's workload requires, on any surface.

Why this is a module and not a part of one gate
-----------------------------------------------
``0.15-plan.md`` owes the same frame twice: once from a real desktop GL context
(``examples/windows-gl4``) and once from a real GLES 3.1 context
(``examples/android-gles``).  Both fixtures drive *one* implementation of the
workload -- the graph, the kernel, the geometry and the four-by-four target are
shared by construction -- so the expected picture is one fact about the workload
rather than one fact per surface.  Stating it in each gate would be two
statements that a change to the workload has to be applied to twice, and the
failure mode of drift is the worst kind: a gate that keeps passing after the
frame it gates has changed.

What follows from that is what is *not* here.  Getting a context is a surface
fact -- a window on WGL, an EGL pbuffer on GLES -- so the runners live in their
own scripts and this module knows nothing about either.  It reads a report and
the bytes that report names, and it returns problems.

The two statements of the picture are checked against each other
----------------------------------------------------------------
Each fixture writes the pixels twice: as bytes in a file, and as a hex grid in
the report it prints.  Neither is trusted on its own -- the file could be
truncated and the report could be stale -- so ``check`` requires them to agree,
and a run that disagrees with itself fails before any expected value is
consulted.

Fail-closed
-----------
A missing file, a length that is not the extent's, a picture that is not the
extent the report states, a covered pixel outside the two the geometry puts
there, a colour that is not the kernel's, a row order that is not the family's --
each fails the run.  A blank or all-black frame is the failure this exists for:
it passes every counter in the report and is what a rendering path that quietly
stopped drawing looks like.

Orientation is checked rather than assumed
------------------------------------------
The bytes are the family's, which is bottom-up, and the workload's two covered
pixels sit in the row *above the bottom* only.  So a readback that had been
flipped would put them one row higher in the byte stream, and this gate fails on
it.  The magnified PNG is top-down, because that is what a viewer sees.

Standard library only, like every script in this directory: the artifacts are a
picture a reviewer opens and a report a checker reads, and neither is worth a
third-party runtime in the release path (``CLAUDE.md`` §4).
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
import zlib


# What the workload's raster kernel and geometry require of the frame, restated
# here because this file's answer must not be the crate's own.  Each was
# adjudicated by a numbered row in ``0.15-plan.md``; a change to any of them is a
# change to the *workload*, so it fails this gate and the plan and this file are
# updated together rather than one quietly following the other.
#
# - the extent of the render target the workload compiles,
# - the fragment colour `FIXED_COLOR_FRAGMENT` writes, in RGBA8,
# - the clear the pass stores over the rest of the target,
# - the row order the family's copy verbs produce,
# - and the two pixels the triangle covers, as (row, column) from the region's
#   own origin: NDC y = -0.25 is the second row and NDC x = +/-0.25 the middle
#   two columns of a four-by-four target whose pixel centres are at +/-0.25.
EXPECTED_EXTENT = (4, 4)
EXPECTED_COLOUR = (48, 176, 112, 255)
EXPECTED_CLEAR = (0, 0, 0, 255)
EXPECTED_ROW_ORDER = "gl-bottom-left"
EXPECTED_COVERED = ((1, 1), (1, 2))

# The magnification the PNG is written at.  Forty-eight puts one target pixel on a
# 48-pixel block, so the two covered pixels are unmistakable in a 192x192 image
# opened at any size, and the picture is small enough to read as a whole.
SCALE = 48


def extract_json(stdout: str) -> dict:
    """Returns the one JSON object a fixture printed.

    A fixture prints a single object and nothing else, but a build line can still
    reach stdout if the toolchain decides to be helpful -- and on Android one
    does, because the runner's push and its shell are what the stream is shared
    with -- so the object is taken by its own braces rather than by assuming a
    clean stream.
    """
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError("the fixture printed no JSON object")
    parsed = json.loads(stdout[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"the fixture printed a {type(parsed).__name__}, not an object")
    return parsed


def pixels_from_hex(rows: object) -> bytes:
    """Returns the report's hex grid as the bytes it stands for."""
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"the report's pixels are {rows!r}, not a non-empty list")
    out = bytearray()
    for row in rows:
        if not isinstance(row, str) or len(row) != 8:
            raise ValueError(f"the report carries {row!r} where an RGBA8 word belongs")
        try:
            out.extend(bytes.fromhex(row))
        except ValueError as error:
            raise ValueError(f"the report carries {row!r}, which is not hex") from error
    return bytes(out)


def check(report: dict, raw: bytes) -> list[str]:
    """Returns every way the picture contradicts what the workload requires."""
    problems: list[str] = []
    if report.get("opened") is not True:
        return ["the context did not open"]

    workload = report.get("workload")
    if not isinstance(workload, dict):
        return [
            "the report carries no workload block -- the fixture ran without --draws, so no frame "
            "was driven and there is no picture to gate"
        ]
    if workload.get("passes") != 1:
        problems.append(f"the workload ran {workload.get('passes')} passes, not one")
    if not workload.get("draws_requested"):
        problems.append("the workload drew nothing")

    colour = workload.get("colour")
    if not isinstance(colour, dict):
        return problems + [
            "the run reported no colour target -- the fixture ran without --readback, so the frame "
            "was driven and thrown away"
        ]

    extent = colour.get("extent")
    if not isinstance(extent, list) or tuple(extent) != EXPECTED_EXTENT:
        return problems + [
            f"the colour target is {extent!r}, not the {EXPECTED_EXTENT[0]}x{EXPECTED_EXTENT[1]} "
            f"the workload compiles"
        ]
    width, height = EXPECTED_EXTENT
    expected_bytes = width * height * 4

    if colour.get("bytes") != expected_bytes:
        problems.append(
            f"the report says the colour target is {colour.get('bytes')} bytes where an "
            f"{width}x{height} RGBA8 level is {expected_bytes}"
        )
    if len(raw) != expected_bytes:
        return problems + [
            f"the file holds {len(raw)} bytes where an {width}x{height} RGBA8 level is "
            f"{expected_bytes}"
        ]
    if colour.get("row_order") != EXPECTED_ROW_ORDER:
        problems.append(
            f"the report states the row order {colour.get('row_order')!r}, which is not the "
            f"{EXPECTED_ROW_ORDER!r} this family's copy verbs produce -- the consumer that flips "
            f"the image is told which order it has, so a wrong statement is a wrong image"
        )

    # The two statements of one picture, checked against each other before the
    # expected values are consulted: a run that disagrees with itself has failed
    # for a reason no expected picture can diagnose.
    try:
        stated = pixels_from_hex(colour.get("pixels_rgba8"))
    except ValueError as error:
        return problems + [str(error)]
    if stated != raw:
        problems.append(
            "the report's hex grid and the file it names are different pictures -- one of them is "
            "not what this run produced"
        )
        return problems

    covered = {(row, column) for row, column in EXPECTED_COVERED}
    for row in range(height):
        for column in range(width):
            offset = (row * width + column) * 4
            pixel = tuple(raw[offset : offset + 4])
            if (row, column) in covered:
                if pixel != EXPECTED_COLOUR:
                    problems.append(
                        f"the covered pixel at row {row}, column {column} is {pixel}, not the "
                        f"kernel's {EXPECTED_COLOUR}"
                    )
            elif pixel != EXPECTED_CLEAR:
                problems.append(
                    f"the pixel at row {row}, column {column} is {pixel}, not the clear "
                    f"{EXPECTED_CLEAR} -- only the two the triangle covers may be anything else"
                )
    if not any(
        tuple(raw[(row * width + column) * 4 : (row * width + column) * 4 + 4]) == EXPECTED_COLOUR
        for row in range(height)
        for column in range(width)
    ):
        problems.append(
            "no pixel carries the kernel's colour: the frame was read but nothing was drawn into it"
        )
    return problems


def write_png(path: Path, width: int, height: int, rows: list[bytes]) -> None:
    """Writes an 8-bit RGB PNG of ``rows``, top-down, standard library only."""

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    flat = b"".join(b"\x00" + row for row in rows)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(flat, 9))
        + chunk(b"IEND", b"")
    )


def magnify(raw: bytes, extent: tuple[int, int], scale: int) -> tuple[int, int, list[bytes]]:
    """Returns a top-down, magnified RGB rendering of the raw bottom-up picture.

    The flip is the one this gate exists to make visible: the bytes arrive in the
    family's order and a viewer sees the frame the way the rasterizer wrote it, so
    the rows are reversed here rather than being left for whoever opens the file
    to work out.
    """
    width, height = extent
    top_down: list[bytes] = []
    for row in reversed(range(height)):
        pixels = [raw[(row * width + column) * 4 : (row * width + column) * 4 + 3]
                  for column in range(width)]
        line = b"".join(pixel * scale for pixel in pixels)
        for _ in range(scale):
            top_down.append(line)
    return width * scale, height * scale, top_down
