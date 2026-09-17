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

What it asserts
---------------
It fails unless the page reaches `FLUXEL-READY`, unless the page states which
signal settled it, and unless that signal is `composited: raf` -- the compositor
handing the frame over.  The page settles on its bounded timer when Chrome has
stopped sending animation frames, which is what a window that is occluded,
minimized or backgrounded looks like, and `CLAUDE.md` section 5 names occlusion
as one of the conditions the automatic gate has to fail closed on.  A
timer-settled capture is therefore a failed run rather than a note, even though
the page really drew the frame: `Page.captureScreenshot(fromSurface=True)` may
composite the surface afresh for the screenshot, but that yields a picture *of* a
frame and not evidence that the frame reached the screen -- and the screen is
what the page's readiness signal, and this run, are about.

Because that failure would otherwise depend on where the operator happened to
leave their windows, the vehicle launches Chrome with
`--disable-backgrounding-occluded-windows` in its default arguments.  The flag
does not flatter the run: it stops Chrome from throttling a window it has decided
is not worth animating, so the page really does render its frames and the
compositor really does hand them over.  Without it a headed run behind a terminal
settles on the timer and is indistinguishable from one whose window was genuinely
minimized -- measured on this host, where a no-flag run reported `composited:
timer` and the same run with the flag reported `composited: raf`.  `raf` is
therefore the ordinary outcome rather than a flag the operator has to remember,
and a timer-settled run still means something is wrong.

Anything else -- `FLUXEL-DEGRADED`, `FLUXEL-ERROR`, no title, no signal -- is a
failed run too, and so is a capture `cdp_shot.py` refuses to call a frame: the
vehicle decodes the PNG it wrote and refuses a blank, all-black or all-white one,
which is the failure that otherwise passes every counter, every title check and
the digest below.

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

# Passed on every run, ahead of whatever `--chrome-arg` adds.  Chrome throttles
# animation frames for a window it judges not worth animating, so a headed run
# behind a terminal settles on the page's bounded timer and looks exactly like a
# run whose window was minimized -- turning a gate that fails closed on occlusion
# into a gate that fails on where the operator left their windows.  This flag
# stops the throttling rather than hiding it: the page renders its frames and the
# compositor hands them over, which is what the `composited: raf` assertion is
# about.  Measured on this host: without it the same run reported `timer`.
DEFAULT_CHROME_ARGS = ["disable-backgrounding-occluded-windows"]


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
    """Runs one step, failing the whole run with its own output if it fails."""
    completed = subprocess.run(command, check=False, **kwargs)  # type: ignore[arg-type]
    if completed.returncode != 0:
        raise RuntimeError(f"{command[0]} exited {completed.returncode}")
    return completed


def uncomposited_problem(signal: str) -> str | None:
    """Returns why a capture settled by ``signal`` is not evidence, or ``None``.

    The page races a double `requestAnimationFrame` against a bounded timer and
    reports which one won.  `raf` is the compositor confirming the frame, and is
    the only signal this run accepts.  `timer` is what the page falls back to
    when Chrome has stopped sending animation frames, which is what an occluded,
    minimized or backgrounded window looks like -- and a frame the compositor
    never confirmed has no evidence that it reached the screen, whatever the
    screenshot API then composites for the capture.  Accepting it, even with a
    printed note, is how a run that photographed a window nobody was compositing
    gets recorded as a passing integration run, so the decision lives here rather
    than in a branch nobody can test.

    The vehicle already passes `--disable-backgrounding-occluded-windows`, so this
    is not the ordinary outcome of a window sitting behind another one -- reaching
    it now means the window was genuinely minimized, or that the browser would not
    animate at all.  The message says so rather than sending the operator after a
    flag that is already there.
    """
    if signal == "raf":
        return None
    return (
        "the page settled on its bounded timer rather than the compositor's own signal: "
        "Chrome stopped sending animation frames, so nothing in this run confirms that the "
        "frame reached the screen. This vehicle already passes "
        "--disable-backgrounding-occluded-windows, so an occluded window is not the "
        "explanation left: check that the browser window was not minimized, and that the "
        "GPU is not wedged"
    )


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
    parser.add_argument("--chrome-arg", action="append", default=[],
                        help="an extra Chrome switch, after the ones this vehicle always passes")
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
         *[f"--chrome-arg={flag}" for flag in DEFAULT_CHROME_ARGS + arguments.chrome_arg]],
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
        # One of the ways the capture fails is exit 4: the vehicle decoded the PNG
        # it wrote and refused to call it a frame, because it is a single flat
        # colour or cannot be read at all.  Its own reason is on stderr above.
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
    # The digest is the vehicle's own hash of the PNG it just wrote, so what it
    # establishes is that the numbers printed here name the file on disk -- not
    # that the file is a picture of a page.  Whether the pixels are a frame at all
    # is the vehicle's pixel inspection, and it has already refused the run above
    # if they are not; the page's own statement is what makes the digest worth
    # recording beside the candidate.
    uncomposited = uncomposited_problem(signal.group(1))
    if uncomposited is not None:
        print(uncomposited, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
