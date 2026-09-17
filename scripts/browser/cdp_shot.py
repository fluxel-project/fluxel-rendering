#!/usr/bin/env python3
"""Serve a page, open it in a *headed* Chrome, screenshot it, close the browser.

Why this exists instead of a PrintWindow capture
------------------------------------------------
`PrintWindow(hwnd, hdc, PW_RENDERFULLCONTENT)` works on a native D3D swapchain
window -- that is how the DX12 and Vulkan frames were taken -- and does **not**
work on a browser window: measured 2026-09-17, a plain green page with a red box
came back pure white in the content area while the browser chrome around it
rendered correctly.  Chrome composites the content area on the GPU and does not
paint it into the window DC.  Bringing the window forward and using
`CopyFromScreen` is the other obvious route, and it fails too: a background
process cannot call `SetForegroundWindow` successfully, which is the same
restriction that made CopyFromScreen grab the terminal the first time.

So the screenshot is taken by the browser itself, over the DevTools Protocol.
That is not a weaker capture -- it is the *stronger* one, and it is what EXP-008
asks for: `Page.captureScreenshot` returns the composited surface, not a
serialization of a canvas element, so it shows what a viewer would see.

The browser stays **headed** and GPU-accelerated throughout.  `--headless` is
never passed: the operator's rule is that rendering-integration evidence needs a
real headed browser, and headless plus SwiftShader would be the software
rasterization this is meant to avoid.

No third-party dependency: the WebSocket client below is the minimum needed to
carry one CDP command, written against the standard library because CLAUDE.md
section 4 asks new orchestration to stay on Python 3 stdlib.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

# The page sets its own terminal title; that is the handshake, because a fixed
# sleep is a race against wasm instantiation and first-frame compositing.
TERMINAL = ("FLUXEL-READY", "FLUXEL-DEGRADED", "FLUXEL-ERROR")


class WsError(RuntimeError):
    pass


class WebSocket:
    """Just enough RFC 6455 to send one text frame and read replies."""

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        without_scheme = url.split("://", 1)[1]
        hostport, _, path = without_scheme.partition("/")
        path = "/" + path
        host, _, port = hostport.partition(":")
        self.sock = socket.create_connection((host, int(port or 80)), timeout=timeout)
        self.sock.settimeout(timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        handshake = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {hostport}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(handshake.encode())
        header = self._read_until(b"\r\n\r\n")
        if b" 101 " not in header.split(b"\r\n", 1)[0]:
            raise WsError(f"websocket upgrade refused: {header[:120]!r}")
        expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        )
        if expected.lower() not in header.lower():
            raise WsError("websocket accept key did not match")

    def _read_until(self, marker: bytes) -> bytes:
        data = b""
        while marker not in data:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise WsError("connection closed during handshake")
            data += chunk
        return data

    def _recv_exact(self, count: int) -> bytes:
        data = b""
        while len(data) < count:
            chunk = self.sock.recv(count - len(data))
            if not chunk:
                raise WsError("connection closed mid-frame")
            data += chunk
        return data

    def send_text(self, payload: str) -> None:
        body = payload.encode()
        header = bytearray([0x81])  # FIN + text
        mask = os.urandom(4)
        length = len(body)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header += struct.pack("!H", length)
        else:
            header.append(0x80 | 127)
            header += struct.pack("!Q", length)
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(body))
        self.sock.sendall(bytes(header) + masked)

    def recv_text(self) -> str:
        while True:
            first, second = self._recv_exact(2)
            opcode = first & 0x0F
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._recv_exact(8))[0]
            payload = self._recv_exact(length) if length else b""
            if opcode == 0x8:
                raise WsError("server closed the websocket")
            if opcode in (0x1, 0x2):
                return payload.decode("utf-8", "replace")
            # ping/pong/continuation: ignore, keep reading

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self) -> "WebSocket":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def cdp(ws: WebSocket, message_id: int, method: str, params: dict | None = None) -> dict:
    """Send one CDP command and return its result, skipping unrelated events.

    Events and other replies share the socket, so a command waits for its own
    id rather than for the next frame.
    """
    payload: dict = {"id": message_id, "method": method}
    if params is not None:
        payload["params"] = params
    ws.send_text(json.dumps(payload))
    while True:
        message = json.loads(ws.recv_text())
        if message.get("id") != message_id:
            continue
        if "error" in message:
            raise WsError(f"{method} failed: {message['error']}")
        return message.get("result", {})


def page_text(ws: WebSocket, message_id: int) -> str:
    """The page's rendered text, which is how a harness reports its verdict."""
    result = cdp(
        ws,
        message_id,
        "Runtime.evaluate",
        {
            "expression": "document.body ? document.body.innerText : ''",
            "returnByValue": True,
        },
    )
    return result.get("result", {}).get("value") or ""


def serve(directory: Path, port: int) -> ThreadingHTTPServer:
    handler = lambda *a, **k: SimpleHTTPRequestHandler(*a, directory=str(directory), **k)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def targets(port: int) -> list[dict]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=3) as response:
        return json.loads(response.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    # Omitted when something else already serves the page -- the wasm test
    # runner in interactive mode serves its own harness.
    parser.add_argument("--serve-dir", default=None)
    parser.add_argument("--serve-port", type=int, default=8765)
    parser.add_argument("--debug-port", type=int, default=9222)
    parser.add_argument("--chrome", default=None)
    # The window is the artifact's frame: a page taller or wider than this is
    # captured with its own content clipped, and a verdict line below the fold is
    # a verdict the reviewer of the image never sees.
    parser.add_argument("--window-size", default="1100,640")
    parser.add_argument("--timeout", type=float, default=90.0)
    # Repeatable.  Used to run the *same* page under a different GPU path --
    # `--chrome-arg=use-angle=swiftshader` reproduces what CI sees, and passing
    # nothing runs on the real adapter.  Comparing the two is how a defect gets
    # classified as a driver artifact or as ours.
    parser.add_argument("--chrome-arg", action="append", default=[])
    # Wait for a phrase to appear in the page's own text before capturing, and
    # print that text.  `wasm-bindgen-test-runner` in interactive mode
    # (`NO_HEADLESS`) serves its harness on a port and reports "test result:"
    # only in the DOM, so this is how the real test suite runs headed on the
    # real GPU and still reports machine-readable results.
    parser.add_argument("--wait-text", default=None)
    parser.add_argument("--wait-timeout", type=float, default=300.0)
    args = parser.parse_args()

    chrome = args.chrome or next((c for c in CHROME_CANDIDATES if Path(c).exists()), None)
    if not chrome:
        print("no Chrome or Edge binary found", file=sys.stderr)
        return 2

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="fluxel-headed-"))

    server = serve(Path(args.serve_dir).resolve(), args.serve_port) if args.serve_dir else None
    proc = None
    try:
        launch = [
            chrome,
            f"--user-data-dir={profile}",
            f"--remote-debugging-port={args.debug_port}",
            "--no-first-run",
            "--no-default-browser-check",
            f"--window-size={args.window_size}",
            "--window-position=40,40",
        ]
        launch += [f"--{flag}" if not flag.startswith("-") else flag for flag in args.chrome_arg]
        launch += ["--new-window", args.url]
        proc = subprocess.Popen(
            launch,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # The page this run is about, rather than any tab Chrome happened to
        # open: a fresh profile can bring up a blank page beside the target.
        parsed = urllib.parse.urlparse(args.url)
        wanted = f"{parsed.netloc}{parsed.path}"
        wants_text = args.wait_text is not None

        deadline = time.time() + args.timeout
        page = None
        last_title = "<no target>"
        while time.time() < deadline:
            try:
                for target in targets(args.debug_port):
                    if target.get("type") != "page":
                        continue
                    title = target.get("title", "")
                    target_url = target.get("url", "")
                    if title:
                        last_title = title
                    if wants_text:
                        # The harness reports in the DOM, so the target is the
                        # one serving this run and its title is irrelevant.
                        if target_url.startswith("http") and wanted in target_url:
                            page = target
                            break
                    elif any(title.startswith(marker) for marker in TERMINAL):
                        page = target
                        break
            except Exception:
                pass
            if page:
                break
            time.sleep(0.4)

        if not page:
            print(f"no terminal state within {args.timeout}s; last title: {last_title}")
            return 1

        title = page["title"]
        print(f"page title: {title}")
        # Let the compositor settle after the page's own readiness signal.
        time.sleep(1.0)

        with WebSocket(page["webSocketDebuggerUrl"]) as ws:
            cdp(ws, 1, "Page.enable")
            if wants_text:
                text = ""
                found = False
                text_deadline = time.time() + args.wait_timeout
                while time.time() < text_deadline:
                    text = page_text(ws, 2)
                    if args.wait_text in text:
                        found = True
                        break
                    time.sleep(1.0)
                print("--- page text ---")
                print(text.strip()[-6000:])
                print("--- end page text ---")
                if not found:
                    print(f"waited {args.wait_timeout}s for {args.wait_text!r} in the page text")
                    return 1
            else:
                # A page that reports in its title still says more in its body:
                # which frames were submitted, which diagnostics were recorded,
                # which signal settled it.  Printed rather than left to whoever
                # opens the PNG, so the capture is readable on its own.
                print("--- page text ---")
                print(page_text(ws, 2).strip()[-6000:])
                print("--- end page text ---")
            shot = cdp(
                ws,
                3,
                "Page.captureScreenshot",
                {"format": "png", "fromSurface": True},
            )["data"]
            out.write_bytes(base64.b64decode(shot))

        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f"captured -> {out}")
        print(f"sha256 {digest}")
        print(f"page state: {title}")
        if title.startswith("FLUXEL-ERROR"):
            return 3
        return 0
    finally:
        if proc:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if server:
            server.shutdown()
        shutil.rmtree(profile, ignore_errors=True)
        print("browser closed")


if __name__ == "__main__":
    raise SystemExit(main())
