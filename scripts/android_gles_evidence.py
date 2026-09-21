#!/usr/bin/env python3
"""Run the GLES fixture on an attached Android device and gate its picture.

Why this exists
---------------
``0.15-plan.md`` owes one item under the GL-family evidence: the GLES 3.1
triangle.  The desktop half is ``check_gl4_raster_readback.py``, which needs a
window and a desktop driver; this is the half that runs where GLES actually
lives, on a device reached over ``adb``.  The frame it drives is the *same*
frame -- one implementation of the workload, one expected picture, asserted by
one shared checker -- so what differs between the two scripts is only how a
context is obtained and how a binary gets to the machine that opens it.

What it does, in order
----------------------
Cross-compiles ``examples/android-gles`` for the device's ABI with the NDK's
linker, pushes the binary into the device's temporary directory, runs it there,
pulls the colour target back, and gates the picture against the workload's own
kernel and geometry before writing a magnified top-down PNG a reviewer opens.

Python rather than a shell script, though the runner is one platform's
(``CLAUDE.md`` §4): the platform-specific part is a narrow adapter over ``adb``
and the NDK, and stating it here keeps a second device -- a phone, a different
emulator image, an ARM ABI -- a change to this adapter rather than a second
script.  It is also what keeps the runner out of the path-conversion trap that
``adb`` falls into when a POSIX shell rewrites ``/data/local/tmp`` into a
Windows path before the tool ever sees it.

What this evidence does and does not prove
------------------------------------------
It proves a GLES 3.1 implementation accepted a 3.1 pbuffer context and executed
the frame, on a device whose guest reports ``OpenGL ES 3.1`` and a GLSL ES 3.10
shading language version.  It does **not** prove a particular physical device
did: an emulator's guest GL strings are a presented profile, and LDPlayer
executes the command stream through its own host-side renderer.  That is
``CLAUDE.md`` §4.5 -- a driver string is not a correctness proof -- and it is why
the report's ``renderer`` field is recorded rather than believed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
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
    SCALE,
    check,
    extract_json,
    magnify,
    write_png,
)

DEFAULT_MANIFEST = Path("examples/android-gles/Cargo.toml")
DEFAULT_OUT = Path("target/gles-raster-readback")

# The ABI the release gate's device is: an x86_64 emulator image.  Stated as a
# default rather than derived, because deriving an ABI from a running device is a
# second thing that can be wrong and the wrong answer is a linker error.
DEFAULT_TARGET = "x86_64-linux-android"

# The API level the NDK's linker wrapper is chosen for.  It decides which
# ``libEGL.so`` and ``libGLESv2.so`` stubs get linked, so it is the floor this
# evidence claims: a device older than this would not have loaded the binary at
# all, which is a louder failure than a wrong picture.
DEFAULT_API = 29

# Where the NDK is looked for when the environment does not say.  Under `target/`
# because it is a fetched toolchain rather than a source artifact (`CLAUDE.md`
# §7); the override exists because a machine can perfectly well have its own.
DEFAULT_NDK = Path("target/tools/android-ndk-r28")

# The device-side directory.  Deliberately *not* under `/data/local/tmp` itself,
# so that cleaning up afterwards cannot remove a file somebody else put there.
REMOTE_DIR = "/data/local/tmp/fluxel-gles-evidence"
BINARY_NAME = "fluxel-android-gles-harness"
REMOTE_COLOUR = "colour.rgba"


class Failure(Exception):
    """Something the run cannot continue past, already described for a reader."""


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Runs ``command`` and returns the completed process, never raising on status.

    Every caller here decides for itself what a nonzero exit means, because they
    do not all mean the same thing: a missing device is a usage problem and a
    refused context is a finding about the driver.
    """
    return subprocess.run(command, capture_output=True, text=True, check=False, **kwargs)


def resolve_ndk(explicit: Path | None, root: Path) -> Path:
    """Returns the NDK root, refusing one that cannot link for the target."""
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    from_environment = os.environ.get("FLUXEL_ANDROID_NDK")
    if from_environment:
        candidates.append(Path(from_environment))
    candidates.append(root / DEFAULT_NDK)

    for candidate in candidates:
        wrapper = linker_wrapper(candidate, DEFAULT_API)
        if wrapper is not None:
            return candidate
    looked = ", ".join(str(candidate) for candidate in candidates)
    raise Failure(
        f"no NDK with an API {DEFAULT_API} linker was found; looked in {looked}. "
        f"Set FLUXEL_ANDROID_NDK to an NDK r26 or newer."
    )


def linker_wrapper(ndk: Path, api: int) -> Path | None:
    """The NDK's clang wrapper for this target and API, if this NDK has one.

    The platform-specific directories under ``bin`` are what carry the API level:
    an NDK ships one wrapper per (ABI, API) pair, and the unversioned
    ``clang`` beside them links against no Android platform at all.
    """
    name = f"{DEFAULT_TARGET}{api}-clang.cmd"
    for candidate in (
        ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin" / name,
        ndk / "toolchains/llvm/prebuilt/linux-x86_64/bin" / name.removesuffix(".cmd"),
        ndk / "toolchains/llvm/prebuilt/darwin-x86_64/bin" / name.removesuffix(".cmd"),
    ):
        if candidate.exists():
            return candidate
    return None


def resolve_adb(explicit: Path | None) -> str:
    """Returns the ``adb`` to drive, refusing a machine that has none."""
    if explicit is not None:
        if not explicit.exists():
            raise Failure(f"the adb at {explicit} does not exist")
        return str(explicit)
    found = shutil.which("adb")
    if found is None:
        raise Failure(
            "no adb on PATH; set FLUXEL_ADB to one, or put the platform-tools "
            "directory on PATH"
        )
    return found


def attached_devices(devices_stdout: str) -> list[str]:
    """Returns the serials an ``adb devices`` listing reports as ready.

    Rows that are not ready -- ``offline``, ``unauthorized``, ``no permissions``
    -- are left out rather than reported as devices, because the failure a caller
    needs from them is "the device is not usable", not a serial that fails later
    at a push.
    """
    attached: list[str] = []
    for line in devices_stdout.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "device":
            attached.append(fields[0])
    return attached


def choose_serial(attached: list[str], explicit: str | None) -> str:
    """Picks the one device to run on, refusing an ambiguous or empty list.

    Ambiguity is refused rather than resolved: a gate that picked whichever
    device answered first would report evidence from a machine the caller did not
    name, which is worse than not running.
    """
    if explicit is not None:
        if explicit not in attached:
            raise Failure(f"{explicit} is not attached and ready; attached: {attached}")
        return explicit
    if not attached:
        raise Failure("no device is attached and ready")
    if len(attached) > 1:
        raise Failure(f"{len(attached)} devices are attached ({attached}); pass --serial")
    return attached[0]


def resolve_serial(adb: str, explicit: str | None) -> str:
    """Returns the serial to run on, asking ``adb`` which devices are attached."""
    completed = run([adb, "devices"])
    if completed.returncode != 0:
        raise Failure(f"`adb devices` failed: {completed.stderr.strip()[-400:]}")
    return choose_serial(attached_devices(completed.stdout), explicit)


def build(root: Path, manifest: Path, ndk: Path, offline: bool) -> Path:
    """Cross-compiles the fixture and returns the binary's path."""
    wrapper = linker_wrapper(ndk, DEFAULT_API)
    if wrapper is None:
        raise Failure(f"the NDK at {ndk} no longer has an API {DEFAULT_API} linker")
    environment = dict(os.environ)
    # The variable name is Cargo's own spelling of the target triple, so it is
    # `X86_64_LINUX_ANDROID` and not the triple with its dashes kept.
    environment[f"CARGO_TARGET_{DEFAULT_TARGET.upper().replace('-', '_')}_LINKER"] = str(wrapper)
    command = [
        "cargo",
        "build",
        "--manifest-path",
        str(manifest),
        "--target",
        DEFAULT_TARGET,
        "--release",
    ]
    if offline:
        command.append("--offline")
    completed = run(command, cwd=root, env=environment)
    if completed.returncode != 0:
        raise Failure(
            f"the fixture did not build: {completed.stderr.strip()[-800:]}\n"
            f"if the target is missing, `rustup target add {DEFAULT_TARGET}`"
        )
    binary = manifest.parent / "target" / DEFAULT_TARGET / "release" / BINARY_NAME
    if not binary.exists():
        raise Failure(f"the build succeeded but left no binary at {binary}")
    return binary


def shell(adb: str, serial: str, command: str) -> subprocess.CompletedProcess:
    """Runs one command on the device."""
    return run([adb, "-s", serial, "shell", command])


def push(adb: str, serial: str, local: Path, remote: str) -> None:
    """Pushes ``local`` to ``remote``, creating the device-side directory."""
    prepared = shell(adb, serial, f"mkdir -p {REMOTE_DIR}")
    if prepared.returncode != 0:
        raise Failure(
            f"the device refused {REMOTE_DIR}: {prepared.stderr.strip()[-400:]}"
        )
    completed = run([adb, "-s", serial, "push", str(local), remote])
    if completed.returncode != 0:
        raise Failure(f"the binary did not reach the device: {completed.stderr.strip()[-400:]}")
    # `adb push` preserves neither the mode nor, on some images, the ability to
    # run out of `/data/local/tmp` without it.
    made_executable = shell(adb, serial, f"chmod 755 {remote}")
    if made_executable.returncode != 0:
        raise Failure(f"the binary could not be made executable: {made_executable.stderr.strip()[-400:]}")


def pull(adb: str, serial: str, remote: str, local: Path) -> None:
    """Pulls ``remote`` back, refusing a file the run did not write."""
    completed = run([adb, "-s", serial, "pull", remote, str(local)])
    if completed.returncode != 0:
        raise Failure(
            f"the colour target was not read back off the device: "
            f"{completed.stderr.strip()[-400:]}"
        )


def clean_remote(adb: str, serial: str) -> None:
    """Removes this run's device-side files, and says so if it cannot.

    Best effort by design: the evidence is already in hand by the time this runs,
    so a device that refuses the removal is a note rather than a failed run --
    but it is printed, because a silently accumulating temporary directory is
    what the removal is here to prevent.
    """
    removed = shell(adb, serial, f"rm -rf {REMOTE_DIR}")
    if removed.returncode != 0:
        print(f"note: {REMOTE_DIR} was left on the device: {removed.stderr.strip()[-200:]}",
              file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=None,
                        help=f"fixture manifest, default {DEFAULT_MANIFEST}")
    parser.add_argument("--ndk", type=Path, default=None,
                        help=f"NDK root, default FLUXEL_ANDROID_NDK or {DEFAULT_NDK}")
    parser.add_argument("--adb", type=Path, default=None,
                        help="adb to drive, default FLUXEL_ADB or PATH")
    parser.add_argument("--serial", default=None, help="device serial, if several are attached")
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--extent", default="64x64", help="WIDTHxHEIGHT of the pbuffer")
    parser.add_argument("--gles-version", default=None, choices=("3.0", "3.1", "3.2"),
                        help="strict GLES profile; omit to discover highest available")
    parser.add_argument("--mode", default="optimized", choices=("optimized", "oracle"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", type=Path, default=None,
                        help=f"where to keep the artifacts, default {DEFAULT_OUT}")
    parser.add_argument("--keep-remote", action="store_true",
                        help="leave the binary and its output on the device")
    parser.add_argument("--online", action="store_true",
                        help="let cargo reach the registry; the default is --offline")
    arguments = parser.parse_args()

    try:
        width, height = (int(part) for part in arguments.extent.lower().split("x"))
        if width <= 0 or height <= 0:
            raise ValueError
    except ValueError:
        parser.error("--extent has to look like 64x64")

    root = arguments.root.resolve()
    manifest = arguments.manifest or (root / DEFAULT_MANIFEST)
    out = arguments.out or (root / DEFAULT_OUT)

    serial = None
    adb = None
    try:
        ndk = resolve_ndk(arguments.ndk, root)
        adb = resolve_adb(arguments.adb)
        serial = resolve_serial(adb, arguments.serial)
        binary = build(root, manifest, ndk, offline=not arguments.online)
        remote_binary = f"{REMOTE_DIR}/{BINARY_NAME}"
        remote_colour = f"{REMOTE_DIR}/{REMOTE_COLOUR}"
        push(adb, serial, binary, remote_binary)

        invocation = (
            f"{remote_binary} --extent {width}x{height} --draws {arguments.draws} "
            f"--mode {arguments.mode} --readback {remote_colour}"
        )
        if arguments.gles_version is not None:
            invocation += f" --gles-version {arguments.gles_version}"
        completed = run(
            [adb, "-s", serial, "shell", invocation], timeout=arguments.timeout
        )
        if completed.returncode != 0:
            raise Failure(
                f"the fixture exited {completed.returncode} on the device: "
                f"{completed.stderr.strip()[-600:]}"
            )
        report = extract_json(completed.stdout)

        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        colour_path = out / "colour.rgba"
        pull(adb, serial, remote_colour, colour_path)
    except Failure as error:
        print(error, file=sys.stderr)
        return 1
    except (subprocess.TimeoutExpired, ValueError) as error:
        print(f"the run produced no report: {error}", file=sys.stderr)
        return 1
    finally:
        if adb is not None and serial is not None and not arguments.keep_remote:
            clean_remote(adb, serial)

    raw = colour_path.read_bytes()
    workload = report.get("workload") or {}
    context = workload.get("colour") or {}
    print(f"device {serial}, artifacts in {out}")
    print(
        f"  {report.get('renderer')} ({report.get('driver_or_browser')}), "
        f"profile {report.get('profile')}, {report.get('version')}"
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
