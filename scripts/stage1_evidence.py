#!/usr/bin/env python3
"""Capture and validate the real Stage 1 Windows presentation evidence.

The platform-neutral orchestration deliberately has no image or UI dependency.
The Windows implementation below is a small ctypes/GDI backend; future native
backends can provide the same WindowBackend operations without changing the
evidence contract.  Each backend receives three real desktop captures of the
same static scene: stable, resized, and minimized/restored.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

# Every script in this directory is run both as a program and as a module by a
# test, which loads it by path; neither puts the directory itself on `sys.path`,
# so the one sibling import this file makes is set up here rather than left to
# whichever caller happened to be first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_provenance import ProvenanceError, require_clean  # noqa: E402  (after the path it needs)


TARGET = "x86_64-pc-windows-msvc"
TITLE = "Fluxel Stage 1 — Windows visible renderer"
WM_CLOSE = 0x0010
SW_MINIMIZE, SW_RESTORE = 6, 9
SWP_NOZORDER, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE = 0x0004, 0x0010, 0x0002, 0x0001
HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
SRCCOPY = 0x00CC0020
BI_RGB = 0
SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79


class EvidenceError(RuntimeError):
    """A hardware evidence precondition or oracle failed."""


class RECT(ctypes.Structure):
    _fields_ = (("left", ctypes.c_long), ("top", ctypes.c_long),
               ("right", ctypes.c_long), ("bottom", ctypes.c_long))


class POINT(ctypes.Structure):
    _fields_ = (("x", ctypes.c_long), ("y", ctypes.c_long))


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = (
        ("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long), ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    )


class RGBQUAD(ctypes.Structure):
    _fields_ = (("rgbBlue", ctypes.c_byte), ("rgbGreen", ctypes.c_byte),
               ("rgbRed", ctypes.c_byte), ("rgbReserved", ctypes.c_byte))


class BITMAPINFO(ctypes.Structure):
    _fields_ = (("bmiHeader", BITMAPINFOHEADER), ("bmiColors", RGBQUAD * 1))


@dataclass(frozen=True)
class Geometry:
    left: int
    top: int
    right: int
    bottom: int
    client_left: int
    client_top: int
    client_width: int
    client_height: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class PixelStats:
    samples: int
    black: int
    red: int
    green: int
    blue: int
    red_expected: int
    green_expected: int
    blue_expected: int
    corners_black: bool


class Win32Backend:
    """The Windows-only native window and desktop capture implementation."""

    def __init__(self) -> None:
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        self._configure_signatures()
        # Coordinates from this evidence process must be physical desktop pixels.
        try:
            self.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except AttributeError:
            self.user32.SetProcessDPIAware()

    def _configure_signatures(self) -> None:
        u, g = self.user32, self.gdi32
        u.FindWindowW.argtypes, u.FindWindowW.restype = (wintypes.LPCWSTR, wintypes.LPCWSTR), wintypes.HWND
        u.IsWindow.argtypes, u.IsWindow.restype = (wintypes.HWND,), wintypes.BOOL
        u.GetWindowRect.argtypes, u.GetWindowRect.restype = (wintypes.HWND, ctypes.POINTER(RECT)), wintypes.BOOL
        u.GetClientRect.argtypes, u.GetClientRect.restype = (wintypes.HWND, ctypes.POINTER(RECT)), wintypes.BOOL
        u.ClientToScreen.argtypes, u.ClientToScreen.restype = (wintypes.HWND, ctypes.POINTER(POINT)), wintypes.BOOL
        u.SetWindowPos.argtypes, u.SetWindowPos.restype = (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT), wintypes.BOOL
        u.ShowWindow.argtypes, u.ShowWindow.restype = (wintypes.HWND, ctypes.c_int), wintypes.BOOL
        u.SetForegroundWindow.argtypes, u.SetForegroundWindow.restype = (wintypes.HWND,), wintypes.BOOL
        u.PostMessageW.argtypes, u.PostMessageW.restype = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM), wintypes.BOOL
        u.SetProcessDpiAwarenessContext.argtypes, u.SetProcessDpiAwarenessContext.restype = (ctypes.c_void_p,), wintypes.BOOL
        u.SetProcessDPIAware.argtypes, u.SetProcessDPIAware.restype = (), wintypes.BOOL
        u.GetSystemMetrics.argtypes, u.GetSystemMetrics.restype = (ctypes.c_int,), ctypes.c_int
        u.GetDC.argtypes, u.GetDC.restype = (wintypes.HWND,), wintypes.HDC
        u.ReleaseDC.argtypes, u.ReleaseDC.restype = (wintypes.HWND, wintypes.HDC), ctypes.c_int
        g.CreateCompatibleDC.argtypes, g.CreateCompatibleDC.restype = (wintypes.HDC,), wintypes.HDC
        g.DeleteDC.argtypes, g.DeleteDC.restype = (wintypes.HDC,), wintypes.BOOL
        g.CreateCompatibleBitmap.argtypes, g.CreateCompatibleBitmap.restype = (wintypes.HDC, ctypes.c_int, ctypes.c_int), wintypes.HBITMAP
        g.SelectObject.argtypes, g.SelectObject.restype = (wintypes.HDC, wintypes.HGDIOBJ), wintypes.HGDIOBJ
        g.DeleteObject.argtypes, g.DeleteObject.restype = (wintypes.HGDIOBJ,), wintypes.BOOL
        g.BitBlt.argtypes, g.BitBlt.restype = (wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD), wintypes.BOOL
        g.GetDIBits.argtypes, g.GetDIBits.restype = (wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.POINTER(BITMAPINFO), wintypes.UINT), ctypes.c_int

    @staticmethod
    def _ok(value: object, operation: str) -> None:
        if not value:
            raise EvidenceError(f"{operation} failed (Win32 error {ctypes.get_last_error()}).")

    def find(self) -> int:
        return int(self.user32.FindWindowW(None, TITLE) or 0)

    def geometry(self, hwnd: int) -> Geometry:
        outer, client, point = RECT(), RECT(), POINT()
        self._ok(self.user32.GetWindowRect(hwnd, ctypes.byref(outer)), "GetWindowRect")
        self._ok(self.user32.GetClientRect(hwnd, ctypes.byref(client)), "GetClientRect")
        self._ok(self.user32.ClientToScreen(hwnd, ctypes.byref(point)), "ClientToScreen")
        return Geometry(outer.left, outer.top, outer.right, outer.bottom, point.x, point.y,
                        client.right - client.left, client.bottom - client.top)

    def assert_complete(self, hwnd: int) -> Geometry:
        geometry = self.geometry(hwnd)
        left, top = self.user32.GetSystemMetrics(SM_XVIRTUALSCREEN), self.user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        right = left + self.user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        bottom = top + self.user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
        if (geometry.client_width < 320 or geometry.client_height < 240 or
                geometry.left < left or geometry.top < top or
                geometry.right > right or geometry.bottom > bottom):
            raise EvidenceError(f"window is not wholly visible: outer={geometry.left},{geometry.top}-"
                                f"{geometry.right},{geometry.bottom}; client={geometry.client_width}x{geometry.client_height}")
        return geometry

    def position(self, hwnd: int, left: int, top: int, width: int, height: int) -> None:
        self._ok(self.user32.SetWindowPos(hwnd, 0, left, top, width, height, SWP_NOZORDER | SWP_NOACTIVATE), "SetWindowPos")

    def minimize(self, hwnd: int) -> None:
        self.user32.ShowWindow(hwnd, SW_MINIMIZE)

    def restore(self, hwnd: int) -> None:
        self.user32.ShowWindow(hwnd, SW_RESTORE)

    def close(self, hwnd: int) -> None:
        self._ok(self.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0), "PostMessage(WM_CLOSE)")

    def capture_bmp(self, hwnd: int, output: Path) -> Geometry:
        # The blit below reads the *screen* at the window's rectangle, so the
        # window has to be what DWM composited there, and on a desktop it is not
        # enough for the window to exist at that spot.  Measured 2026-09-17 with
        # a maximized browser covering the virtual screen: `SetForegroundWindow`
        # is refused for a background process, and re-positioning with
        # `HWND_TOP` left the window at z-order 2 -- still behind the browser,
        # so the gate captured the browser and only the picture oracle noticed.
        # `HWND_TOPMOST` moved it to 0, which is why the capture is bracketed by
        # a topmost raise and an immediate drop: the raise is what makes this
        # window the captured one, and the drop keeps the harness from sitting
        # above the user's desktop for the rest of the run.
        self.user32.SetForegroundWindow(hwnd)
        self._ok(self.user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                          SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE), "SetWindowPos(HWND_TOPMOST)")
        # The drop belongs to *this* block and not to the blit's, which is why
        # there are two: `assert_complete` below is the call that decides whether
        # the window is in a state worth capturing at all, it is the call most
        # likely to raise, and a raise used to leave this harness topmost over the
        # user's desktop for the rest of the run -- the failure that hides itself,
        # because the next capture would then succeed by covering everything.
        try:
            time.sleep(0.15)
            geometry = self.assert_complete(hwnd)
            screen = self.user32.GetDC(0)
            memory = self.gdi32.CreateCompatibleDC(screen)
            bitmap = self.gdi32.CreateCompatibleBitmap(screen, geometry.width, geometry.height)
            if not screen or not memory or not bitmap:
                raise EvidenceError("unable to allocate a GDI screenshot surface.")
            previous = self.gdi32.SelectObject(memory, bitmap)
            try:
                self._ok(self.gdi32.BitBlt(memory, 0, 0, geometry.width, geometry.height, screen,
                                           geometry.left, geometry.top, SRCCOPY), "BitBlt")
                info = BITMAPINFO()
                info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
                info.bmiHeader.biWidth, info.bmiHeader.biHeight = geometry.width, geometry.height
                info.bmiHeader.biPlanes, info.bmiHeader.biBitCount = 1, 32
                info.bmiHeader.biCompression = BI_RGB
                raw = ctypes.create_string_buffer(geometry.width * geometry.height * 4)
                rows = self.gdi32.GetDIBits(memory, bitmap, 0, geometry.height, raw, ctypes.byref(info), 0)
                if rows != geometry.height:
                    raise EvidenceError(f"GetDIBits returned {rows}/{geometry.height} rows.")
                file_header = struct.pack("<2sIHHI", b"BM", 14 + 40 + len(raw), 0, 0, 54)
                dib_header = struct.pack("<IiiHHIIiiII", 40, geometry.width, geometry.height, 1, 32,
                                         BI_RGB, len(raw), 0, 0, 0, 0)
                output.write_bytes(file_header + dib_header + raw.raw)
            finally:
                self.gdi32.SelectObject(memory, previous)
                self.gdi32.DeleteObject(bitmap)
                self.gdi32.DeleteDC(memory)
                self.user32.ReleaseDC(0, screen)
        finally:
            self.user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0,
                                     SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE)
        return geometry


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_bmp_pixel(data: bytes, width: int, height: int, x: int, y: int) -> tuple[int, int, int]:
    # 32-bit BI_RGB is B,G,R,X and stored bottom-up because the DIB height is positive.
    offset = 54 + ((height - 1 - y) * width + x) * 4
    blue, green, red = data[offset:offset + 3]
    return red, green, blue


def inspect_capture(path: Path, geometry: Geometry) -> PixelStats:
    data = path.read_bytes()
    if len(data) < 54 or data[:2] != b"BM":
        raise EvidenceError(f"{path.name} is not a BMP capture.")
    width, height, bits = struct.unpack_from("<iiH", data, 18)[0], struct.unpack_from("<iiH", data, 22)[0], struct.unpack_from("<H", data, 28)[0]
    if bits != 32 or width != geometry.width or height != geometry.height:
        raise EvidenceError(f"{path.name} has unexpected DIB dimensions/format.")
    x0, y0 = geometry.client_left - geometry.left, geometry.client_top - geometry.top
    w, h = geometry.client_width, geometry.client_height
    if x0 < 0 or y0 < 0 or x0 + w > width or y0 + h > height:
        raise EvidenceError("client rectangle is outside the captured desktop bitmap.")
    samples = black = red = green = blue = red_expected = green_expected = blue_expected = 0
    def kind(pixel: tuple[int, int, int]) -> int:
        r, g, b = pixel
        if r >= 150 and r > g * 1.35 and r > b * 1.35: return 1
        if g >= 125 and g > r * 1.20 and g > b * 1.20: return 2
        if b >= 150 and b > r * 1.20 and b > g * 1.10: return 3
        return 0
    for y in range(y0, y0 + h, 2):
        for x in range(x0, x0 + w, 2):
            r, g, b = read_bmp_pixel(data, width, height, x, y)
            samples += 1
            if r <= 16 and g <= 16 and b <= 16: black += 1
            colour = kind((r, g, b))
            if colour == 1:
                red += 1
                red_expected += int(x < x0 + w * .48 and y > y0 + h * .42)
            elif colour == 2:
                green += 1
                green_expected += int(x > x0 + w * .52 and y > y0 + h * .42)
            elif colour == 3:
                blue += 1
                blue_expected += int(x0 + w * .20 < x < x0 + w * .80 and y < y0 + h * .58)
    # DWM may include a few resize-border pixels inside the client rectangle
    # reported during an active resize. Sample well inside each clear corner.
    inset = max(16, min(w, h) // 50)
    corners = [(x0 + inset, y0 + inset), (x0 + w - inset - 1, y0 + inset),
               (x0 + inset, y0 + h - inset - 1),
               (x0 + w - inset - 1, y0 + h - inset - 1)]
    corners_black = all(max(read_bmp_pixel(data, width, height, x, y)) <= 16 for x, y in corners)
    stats = PixelStats(samples, black, red, green, blue, red_expected, green_expected, blue_expected, corners_black)
    if (not corners_black or black < samples * .20 or min(red, green, blue, red_expected, green_expected, blue_expected) < 40):
        raise EvidenceError(f"{path.name} image oracle failed: {asdict(stats)}")
    return stats


def compare_static(first: Path, second: Path, geometry: Geometry) -> dict[str, int]:
    # Scoped to the client rectangle, because that is the only part of these
    # captures the renderer drew.  The capture is the whole window rect and
    # Windows draws the rest, so an unscoped comparison lets the window
    # manager decide the verdict: on 2026-09-17 the title bar's activation
    # colour differed between the stable capture and the just-restored one and
    # this check read it as flicker at 1758/57300, while the rendered pixels
    # were stable to within 18 samples of 206388.  Scoping it the way
    # `inspect_capture` already scopes its colour oracle keeps the check
    # fail-closed -- a real repaint difference inside the client area still
    # crosses the same threshold -- and stops the chrome from deciding it.
    left, right = first.read_bytes(), second.read_bytes()
    if left[18:30] != right[18:30]:
        raise EvidenceError("stable/restore captures have different BMP dimensions.")
    width, height = struct.unpack_from("<ii", left, 18)
    x0, y0 = geometry.client_left - geometry.left, geometry.client_top - geometry.top
    client_width, client_height = geometry.client_width, geometry.client_height
    if x0 < 0 or y0 < 0 or x0 + client_width > width or y0 + client_height > height:
        raise EvidenceError("client rectangle is outside the captured desktop bitmap.")
    differing = samples = 0
    for y in range(y0, y0 + client_height, 4):
        for x in range(x0, x0 + client_width, 4):
            a, b = read_bmp_pixel(left, width, height, x, y), read_bmp_pixel(right, width, height, x, y)
            samples += 1
            differing += int(sum(abs(one - two) for one, two in zip(a, b)) > 24)
    if differing > max(12, int(samples * .003)):
        raise EvidenceError(f"stable/restore differ at {differing}/{samples} sampled client pixels; possible flicker.")
    return {"sampled_pixels": samples, "differing_pixels": differing}


def wait_for_window(win: Win32Backend, process: subprocess.Popen[bytes], seconds: int) -> int:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        hwnd = win.find()
        if hwnd:
            return hwnd
        if process.poll() is not None:
            raise EvidenceError(f"harness exited before creating its window (exit {process.returncode}).")
        time.sleep(.1)
    raise EvidenceError(f"timed out waiting {seconds}s for '{TITLE}'.")


def log_offset(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def wait_log(path: Path, pattern: str, process: subprocess.Popen[bytes], seconds: int, start: int = 0) -> int:
    regex, deadline = re.compile(pattern), time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            with path.open("rb") as log:
                log.seek(start)
                if regex.search(log.read().decode("utf-8", "replace")):
                    return path.stat().st_size
        if process.poll() is not None:
            raise EvidenceError(f"harness exited before log evidence /{pattern}/ (exit {process.returncode}).")
        time.sleep(.08)
    raise EvidenceError(f"timed out waiting {seconds}s for log evidence /{pattern}/.")


def wait_complete(win: Win32Backend, hwnd: int, process: subprocess.Popen[bytes], seconds: int) -> Geometry:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            return win.assert_complete(hwnd)
        except EvidenceError:
            if process.poll() is not None:
                raise
            time.sleep(.08)
    raise EvidenceError("window did not return to a complete visible state.")


def run_checked(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode:
        raise EvidenceError(f"{' '.join(command)} failed: {result.stderr.strip()}")
    return result.stdout


def verify_msvc(repo: Path) -> None:
    if os.environ.get("CARGO_BUILD_TARGET") not in (None, "", TARGET):
        raise EvidenceError(f"CARGO_BUILD_TARGET must be empty or {TARGET}.")
    host = next((line for line in run_checked(["rustc", "-vV"], repo).splitlines() if line.startswith("host:")), "")
    if host != f"host: {TARGET}":
        raise EvidenceError(f"rustc host must be {TARGET}; found '{host}'.")


def artifact(path: Path) -> dict[str, str]:
    return {"file": path.name, "sha256": sha256(path)}


def require_clean_validation(backend: str, stdout_path: Path, stderr_path: Path) -> str:
    """Fail closed on validation output from either redirected harness stream."""
    stdout = stdout_path.read_text("utf-8", "replace")
    stderr = stderr_path.read_text("utf-8", "replace")
    diagnostics = f"[stdout]\n{stdout}\n[stderr]\n{stderr}"
    clean_summary = re.compile(r"^\s*validation diagnostics:\s*clean\s*$", re.IGNORECASE | re.MULTILINE)
    non_clean_summary = re.compile(
        r"^\s*validation diagnostics(?!:\s*clean\s*$).*?$",
        re.IGNORECASE | re.MULTILINE,
    )
    forbidden = (
        ("Validation Error", re.compile(r"\bvalidation\s+error\b", re.IGNORECASE)),
        ("VUID", re.compile(r"\bVUID(?:[-_][A-Za-z0-9_-]+)?\b", re.IGNORECASE)),
        ("non-clean validation diagnostics", non_clean_summary),
    )
    failures = [label for label, pattern in forbidden if pattern.search(diagnostics)]
    if not clean_summary.search(diagnostics) or failures:
        detail = ", ".join(failures) if failures else "missing clean validation diagnostics summary"
        raise EvidenceError(f"{backend} validation gate failed: {detail}.")
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("dx12", "vulkan"), action="append", dest="backends")
    parser.add_argument("--frames", type=int, default=10_000, help="large finite harness frame limit (default: 10000)")
    parser.add_argument("--window-timeout", type=int, default=60)
    parser.add_argument("--exit-timeout", type=int, default=30)
    args = parser.parse_args()
    if sys.platform != "win32":
        raise EvidenceError("Stage 1 capture currently requires the Windows ctypes/GDI backend.")
    if args.frames < 30:
        raise EvidenceError("--frames must leave enough time for all three evidence states.")
    repo = Path(__file__).resolve().parent.parent
    verify_msvc(repo)
    # The captures below cannot be re-derived from a SHA: they are one machine's
    # answer to one drawing, and nothing recomputes them.  The checkout state at
    # capture time is therefore the only thing that ties them to a revision, so a
    # dirty one is refused rather than recorded beside a SHA that does not
    # describe what actually ran.  Untracked files count -- a fixture reading a
    # file git has never seen is exactly the case this is for.
    try:
        commit = require_clean(repo)
    except ProvenanceError as error:
        raise EvidenceError(f"Stage 1 capture requires a clean checkout: {error}") from error
    root = repo / "target" / "evidence" / commit / "stage1"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest: dict[str, object] = {"schema": 1, "commit": commit, "started_at_utc": utc_now(),
                                   "platform": {"os": "Windows", "target": TARGET}, "runs": []}
    win = Win32Backend()
    active_process: subprocess.Popen[bytes] | None = None
    active_hwnd = 0
    try:
        for backend in args.backends or ["dx12", "vulkan"]:
            prefix, stdout_path, stderr_path = root / backend, root / f"{backend}.stdout.log", root / f"{backend}.stderr.log"
            command = ["cargo", "run", "--manifest-path", "examples/windows-dx12/Cargo.toml", "--locked", "--target", TARGET,
                       "--", "--backend", backend, "--frames", str(args.frames)]
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(command, cwd=repo, stdout=stdout, stderr=stderr)
                hwnd = wait_for_window(win, process, args.window_timeout)
                active_process, active_hwnd = process, hwnd
                # Put the entire harness inside the primary desktop before the first capture.
                initial_mark = log_offset(stderr_path)
                win.position(hwnd, 80, 80, 1200, 760)
                marker = wait_log(stderr_path, r"surface event=(Resized|Restored)", process, args.window_timeout, initial_mark)
                marker = wait_log(stderr_path, r"frame accepted", process, args.window_timeout, marker)
                time.sleep(.25)
                stable_path = Path(f"{prefix}-stable.bmp")
                stable_geometry = win.capture_bmp(hwnd, stable_path)
                stable_stats = inspect_capture(stable_path, stable_geometry)

                marker = log_offset(stderr_path)
                win.position(hwnd, 120, 100, 1000, 680)
                marker = wait_log(stderr_path, r"surface event=(Resized|Restored)", process, args.window_timeout, marker)
                marker = wait_log(stderr_path, r"frame accepted", process, args.window_timeout, marker)
                time.sleep(.2)
                resized_path = Path(f"{prefix}-resized.bmp")
                resized_geometry = win.capture_bmp(hwnd, resized_path)
                resized_stats = inspect_capture(resized_path, resized_geometry)

                # Return to the initial dimensions first: stable and restored are now directly comparable.
                marker = log_offset(stderr_path)
                win.position(hwnd, stable_geometry.left, stable_geometry.top, stable_geometry.width, stable_geometry.height)
                marker = wait_log(stderr_path, r"surface event=(Resized|Restored)", process, args.window_timeout, marker)
                marker = log_offset(stderr_path)
                win.minimize(hwnd)
                marker = wait_log(stderr_path, r"surface event=Minimized", process, args.window_timeout, marker)
                time.sleep(.15)
                marker = log_offset(stderr_path)
                win.restore(hwnd)
                marker = wait_log(stderr_path, r"surface event=(Restored|Resized)", process, args.window_timeout, marker)
                wait_complete(win, hwnd, process, args.window_timeout)
                marker = wait_log(stderr_path, r"frame accepted", process, args.window_timeout, marker)
                time.sleep(.25)
                restored_path = Path(f"{prefix}-restored.bmp")
                restored_geometry = win.capture_bmp(hwnd, restored_path)
                restored_stats = inspect_capture(restored_path, restored_geometry)
                flicker = compare_static(stable_path, restored_path, restored_geometry)

                win.close(hwnd)
                try:
                    process.wait(args.exit_timeout)
                except subprocess.TimeoutExpired as error:
                    # Deliberately do not terminate: a forced process exit is not clean lifecycle evidence.
                    raise EvidenceError(f"harness did not exit normally within {args.exit_timeout}s after WM_CLOSE; it remains running.") from error
                if process.returncode:
                    raise EvidenceError(f"harness exited {process.returncode} after WM_CLOSE.")
                active_process, active_hwnd = None, 0
            diagnostics = require_clean_validation(backend, stdout_path, stderr_path)
            adapter = re.search(r"adapter: backend=(?P<backend>[^,]+), name=(?P<name>.*?), vendor=(?P<vendor>[^,]+), device=(?P<device>[^,]+), driver=(?P<driver>.*)", diagnostics)
            if not adapter:
                raise EvidenceError(f"{backend} adapter/driver diagnostic is missing.")
            manifest["runs"].append({"backend": backend, "command": command,
                                     "adapter": adapter.groupdict(), "logs": [artifact(stdout_path), artifact(stderr_path)],
                                     "screenshots": [
                                         {"label": "stable", "captured_at_utc": utc_now(), **artifact(stable_path), "geometry": asdict(stable_geometry), "oracle": asdict(stable_stats)},
                                         {"label": "resized", "captured_at_utc": utc_now(), **artifact(resized_path), "geometry": asdict(resized_geometry), "oracle": asdict(resized_stats)},
                                         {"label": "minimize_restore", "captured_at_utc": utc_now(), **artifact(restored_path), "geometry": asdict(restored_geometry), "oracle": asdict(restored_stats)},
                                     ], "flicker_check": flicker, "result": "pass"})
        manifest["result"] = "pass"
    except Exception as error:
        # Failure is not evidence, but still request the same graceful close so
        # a bad image oracle does not leave a GPU/window process behind.
        if active_process is not None and active_process.poll() is None and active_hwnd:
            try:
                win.close(active_hwnd)
                active_process.wait(args.exit_timeout)
            except (EvidenceError, subprocess.TimeoutExpired):
                pass
        manifest["result"], manifest["failure"] = "fail", str(error)
        raise
    finally:
        manifest["finished_at_utc"] = utc_now()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Stage 1 evidence passed: {root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceError as error:
        print(f"stage1_evidence: {error}", file=sys.stderr)
        raise SystemExit(1)
