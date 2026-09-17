#!/usr/bin/env python3
"""Build the browser integration page, open it headed on the real GPU, capture it.

Why this is one command instead of three
----------------------------------------
The recipe is: build the wasm module, run `wasm-bindgen` over it, put the page
next to the output, and drive `cdp_shot.py` at it.  It lived in the series plan
as two shell commands plus a capture line, with the page itself only ever
existing inside `target/` -- which is regenerable state (CLAUDE.md section 7), so
a `cargo clean` would have taken the page with it and left the recipe unable to
produce anything.  The page is versioned at `crates/rendering-wasm/web/index.html`
and this script copies it into the served directory, so what was captured and what
is committed cannot drift.

What it asserts, and what it only reports
-----------------------------------------
It fails unless the page reaches `FLUXEL-READY` and unless the page states which
signal settled it (`composited: raf` when the compositor handed the frame over,
`composited: timer` when Chrome had stopped sending animation frames because the
window was occluded).  Both are honest captures of a frame the page really drew,
and `Page.captureScreenshot(fromSurface=True)` composites the surface afresh
either way -- but they are not the same statement, so the page says which one it
is rather than the vehicle assuming.  Anything else -- `FLUXEL-DEGRADED`,
`FLUXEL-ERROR`, no title, no signal -- is a failed run.

The browser is headed and GPU-accelerated; `--headless` is never passed
(CLAUDE.md section 5, and the operator's rule that rendering-integration evidence
comes from a real browser).  `cdp_shot.py` closes the browser on every exit path,
including the failure ones.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CAPTURE = Path(__file__).resolve().parent / "cdp_shot.py"

# The page is versioned with the binding crate it drives; the module beside it in
# the served directory is build output and is not.
PAGE = REPO / "crates" / "rendering-wasm" / "web" / "index.html"
CRATE = "fluxel-rendering-wasm"
MODULE = "fluxel_rendering_wasm.wasm"

PAGE_STATE = re.compile(r"^page state: (\S+)\s*$", re.MULTILINE)
DIGEST = re.compile(r"^sha256 ([0-9a-f]{64})\s*$", re.MULTILINE)
COMPOSITED = re.compile(r"composited: (raf|timer)")


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
    """Runs one step, failing the whole run with its own output if it fails."""
    completed = subprocess.run(command, check=False, **kwargs)  # type: ignore[arg-type]
    if completed.returncode != 0:
        raise RuntimeError(f"{command[0]} exited {completed.returncode}")
    return completed


def build(served: Path, timeout: float) -> None:
    """Builds the module and puts the versioned page beside it."""
    run(
        ["cargo", "build", "-p", CRATE, "--target", "wasm32-unknown-unknown",
         "--release", "--locked"],
        cwd=str(REPO),
        timeout=timeout,
    )
    module = REPO / "target" / "wasm32-unknown-unknown" / "release" / MODULE
    if not module.exists():
        raise RuntimeError(f"the build produced no {module}")
    served.mkdir(parents=True, exist_ok=True)
    run(
        ["wasm-bindgen", "--target", "web", "--out-dir", str(served), str(module)],
        cwd=str(REPO),
        timeout=timeout,
    )
    # Copied after the binding so a stale page cannot outlive a fresh module: the
    # two are one artifact, and the digest below is of what was served.
    shutil.copyfile(PAGE, served / "index.html")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None,
                        help="where the PNG goes, default target/evidence/browser-webgl2-integration.png")
    parser.add_argument("--served", type=Path, default=None,
                        help="the directory served to the browser, default target/browser-evidence/webgl2")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--debug-port", type=int, default=9226)
    # Wide enough that the panel stays beside the canvas rather than wrapping
    # below it, and tall enough to show the diagnostics line: the adapter and the
    # diagnostics count are two of the four things this frame is evidence of, and
    # a verdict below the fold is not in the artifact.
    parser.add_argument("--window-size", default="1200,720")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--build-timeout", type=float, default=1200.0)
    parser.add_argument("--chrome-arg", action="append", default=[])
    arguments = parser.parse_args()

    out = (arguments.out or (REPO / "target" / "evidence" / "browser-webgl2-integration.png")).resolve()
    served = (arguments.served or (REPO / "target" / "browser-evidence" / "webgl2")).resolve()

    try:
        build(served, arguments.build_timeout)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"the page was not built: {error}", file=sys.stderr)
        return 1

    page_bytes = (served / "index.html").read_bytes()
    page_digest = hashlib.sha256(page_bytes).hexdigest()

    captured = subprocess.run(
        [sys.executable, str(CAPTURE),
         "--url", f"http://127.0.0.1:{arguments.port}/index.html",
         "--serve-dir", str(served),
         "--serve-port", str(arguments.port),
         "--debug-port", str(arguments.debug_port),
         "--window-size", arguments.window_size,
         "--timeout", str(arguments.timeout),
         "--out", str(out),
         *[f"--chrome-arg={flag}" for flag in arguments.chrome_arg]],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=arguments.timeout + 300,
        check=False,
    )
    print(captured.stdout, end="")
    if captured.stderr.strip():
        print(captured.stderr.strip(), file=sys.stderr)

    state = PAGE_STATE.search(captured.stdout)
    if captured.returncode != 0 or state is None:
        print(f"the capture failed (exit {captured.returncode})", file=sys.stderr)
        return 1
    if state.group(1) != "FLUXEL-READY":
        print(f"the page reached {state.group(1)} rather than FLUXEL-READY", file=sys.stderr)
        return 1

    signal = COMPOSITED.search(captured.stdout)
    if signal is None:
        # The served page has to be the versioned one: a page that does not state
        # its composited signal is not the page this script copied in, and a
        # capture of an unknown page is not evidence of anything.
        print("the page did not state which signal settled it", file=sys.stderr)
        return 1

    digest = DIGEST.search(captured.stdout)
    if digest is None:
        print("the capture reported no digest", file=sys.stderr)
        return 1
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True, check=False
    ).stdout.strip()

    print(f"\npage sha256 {page_digest} ({PAGE.relative_to(REPO)})")
    print(f"frame sha256 {digest.group(1)} ({out})")
    print(f"candidate {commit}, composited by {signal.group(1)}")
    if signal.group(1) == "timer":
        print("note: Chrome had stopped sending animation frames, so the page settled on its "
              "bounded timer rather than the compositor's own signal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
