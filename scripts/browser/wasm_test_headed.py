#!/usr/bin/env python3
"""Run a browser RHI test suite **headed on the real GPU** and report it.

Why this is two processes and not one
-------------------------------------
`wasm-bindgen-test-runner` has two modes and neither is "headed, automated":

- default: drives a headless Chrome through WebDriver.  This is the CI path and
  it rasterises with whatever the driver config says (SwiftShader here), which
  CLAUDE.md section 4.5 forbids from standing in for real-GPU correctness.
- `NO_HEADLESS=1`: **interactive** mode.  It serves its own harness and waits for
  a browser to open it; it starts no WebDriver and no browser, which is why the
  process looks hung if you expect automation (EXP-031).

So this script pairs the interactive harness with `cdp_shot.py`: the harness
provides the page and the real results, the capture vehicle provides a headed
Chrome or Edge instance on the real adapter and reads the verdict out of the
DOM. Release evidence invokes it twice with explicit `--browser` paths; one
Chromium-family result is not accepted as evidence for the other browser.

The verdict is read from the page text, not the title: the interactive harness
never changes its title.  Both the harness and the browser are torn down on
every exit path, including the failure ones, because a runner left behind holds
the port and makes the *next* run fail with a misleading "failed to spawn
server".

Where the pieces live
---------------------
The three files of this loop are versioned together in `scripts/browser/`, so the
recipe survives a `cargo clean` -- it used to sit under `target/`, which is
regenerable state (CLAUDE.md section 7) and was one cleanup away from being lost.
The two things that are *not* versioned stay under `target/` on purpose: the
`chromedriver` binary, which is a download rather than source and is found by PATH
lookup, and the captures themselves.

Nothing here rebuilds the wasm module: `cargo test` does that, and a stale module
beside a current tree is what the run would then be measuring.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CAPTURE = Path(__file__).resolve().parent / "cdp_shot.py"
READY = re.compile(r"available at (http://\S+)")

# The driver binary is a download, so it is not versioned; the environment
# variable is for a machine that keeps it somewhere else.
DRIVER_DIR = Path(
    os.environ.get("FLUXEL_CHROMEDRIVER_DIR", str(REPO / "target" / "tools" / "chromedriver-win64"))
)

# Software GPU paths are useful for CI diagnosis, but cannot be evidence for a
# headed real-GPU browser run. Keep this gate outside the test binary too.
SOFTWARE_GPU_ARGUMENTS = ("swiftshader", "llvmpipe", "warp", "software")


def cargo_command(features: str) -> list[str]:
    """Return the wasm cargo invocation for one browser backend suite."""
    return [
        "cargo", "test", "-p", "fluxel-rhi", "--target", "wasm32-unknown-unknown",
        "--no-default-features", "--features", features, "--lib", "--locked",
    ]


def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--browser",
        default=None,
        help="explicit Chrome or Edge executable; required by release evidence runs",
    )
    parser.add_argument("--out", default=str(REPO / "target" / "evidence" / "wasm-suite-real-gpu.png"))
    parser.add_argument("--wait-text", default="test result:")
    parser.add_argument("--wait-timeout", type=float, default=300.0)
    parser.add_argument("--debug-port", type=int, default=9225)
    parser.add_argument("--ready-timeout", type=float, default=600.0)
    parser.add_argument("--features", choices=("webgl2", "webgpu"), default="webgl2",
                        help="browser backend feature to build and run")
    parser.add_argument("--chrome-arg", action="append", default=[],
                        help="extra headed Chrome flag; software GPU routes are refused")
    args = parser.parse_args()

    forbidden = [flag for flag in args.chrome_arg
                 if any(token in flag.lower() for token in SOFTWARE_GPU_ARGUMENTS)]
    if forbidden:
        parser.error("headed real-GPU evidence refuses software GPU arguments: " + ", ".join(forbidden))
    if args.features == "webgpu":
        # The first flag asks Chrome to expose the diagnostic fields that the
        # WebGPU test uses to reject software adapters.  The second lets a
        # developer test a real adapter which Chrome's conservative blocklist
        # would otherwise hide; the in-browser assertion still rejects fallback
        # and known software implementations.
        args.chrome_arg.extend(["enable-unsafe-webgpu", "ignore-gpu-blocklist"])

    env = {
        **os.environ,
        # Cargo has no built-in way to run a wasm32 test binary, so the runner
        # has to be named.  Without this cargo tries to execute the `.wasm` as a
        # native program and fails with os error 193.
        "CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER": "wasm-bindgen-test-runner",
        # Interactive, not headless: see the module docstring.
        "NO_HEADLESS": "1",
        "WASM_BINDGEN_TEST_TIMEOUT": "120",
    }
    # The WebDriver JSON pins SwiftShader; interactive mode must not inherit it,
    # or the browser would rasterise in software and the run would prove nothing
    # about the real adapter.
    env.pop("WASM_BINDGEN_TEST_WEBDRIVER_JSON", None)
    env["PATH"] = str(DRIVER_DIR) + os.pathsep + env.get("PATH", "")

    runner = subprocess.Popen(
        cargo_command(args.features),
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    browser: subprocess.Popen | None = None
    try:
        url = None
        assert runner.stdout is not None
        for line in runner.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            match = READY.search(line)
            if match:
                url = match.group(1)
                break
        if url is None:
            print("the harness never announced a page to open", file=sys.stderr)
            return 1

        capture = [
                sys.executable,
                str(CAPTURE),
                "--url",
                url,
                "--wait-text",
                args.wait_text,
                # The interactive harness uses the same `test result:` prefix
                # for failures. Do not turn a completed-but-failing page into
                # a green real-GPU evidence result.
                "--require-text",
                "test result: ok.",
                "--wait-timeout",
                str(args.wait_timeout),
                "--out",
                args.out,
                "--debug-port",
                str(args.debug_port),
            ]
        if args.browser:
            capture.extend(["--chrome", args.browser])
        for flag in args.chrome_arg:
            capture.extend(["--chrome-arg", flag])
        browser = subprocess.Popen(
            capture,
            cwd=str(REPO),
        )
        browser.wait(timeout=args.wait_timeout + 300)
        return browser.returncode or 0
    finally:
        if browser is not None:
            browser.wait()
        kill_tree(runner)
        print("harness stopped")


if __name__ == "__main__":
    raise SystemExit(main())
