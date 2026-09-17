"""The browser integration gate's two fail-closed decisions, without a browser.

Why these are the cases worth keeping
-------------------------------------
The headed WebGL2 integration run is the one artifact of the browser surface that
CLAUDE.md section 5 asks to fail closed on occlusion, blank frames and missing
content, and the two ways it used to pass without any of that were both in the
scripts rather than in the picture: a capture the page settled on its bounded
timer -- what an occluded window looks like -- was printed as a note and accepted,
and the PNG the vehicle wrote was hashed but never looked at, so an all-black or
all-white capture passed every check the run had.

So the two decisions are tested where they are made, and the pixels are tested
against what Chrome really emits: the retained headed frame under ``target/`` when
this machine has one, the repository's own PNG writer for the round trip
everywhere else, and the five PNG filters against frames this module filters
itself.  Nothing here starts a browser -- a unit test that opens a window is a
test that fails for reasons the code cannot see, and the headed run is the main
thread's evidence to take.

Where the fixture colours come from
-----------------------------------
The page-shaped frame below is not a copy of the page; it is the page's *shape*
at a size a test can hold, drawn with the four colours the page's own report and
its retained scene put on the screen: the body background, the canvas clear, the
three fixed fills the retained scene submits, and the light text the panel draws
them beside.  Its purpose is narrow and stated: to show that the flat-capture
floor does not fire on a frame carrying the page's kind of content, so that the
floor's only refusals are the ones it is for.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
import zlib


SCRIPTS = Path(__file__).parents[1]

# One real headed capture, kept as a fixture for the same reason the driver JSON
# beside it is: a GPU is needed to *take* one and none to read it, so this is the
# only way the decoder's filter handling is exercised against Chrome's own PNG
# encoding on a runner with no browser.  It is not a claim about current behaviour
# that can drift -- it is a sample of an encoder whose output the decoder has to
# read, and if the decoder stops reading it the gate fails loudly rather than
# skipping.  sha256 5510228fec601b92a906081a3d3b634d275d7a6009c378e3ed54d56fd52bd39e.
RETAINED_CAPTURE = Path(__file__).parent / "data" / "browser_webgl2_capture_radeon_780m.png"


def load(name: str, path: Path):
    """Loads one script by path, as the sibling modules in this directory do."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CAPTURE = load("cdp_shot", SCRIPTS / "browser" / "cdp_shot.py")
INTEGRATION = load("webgl2_integration", SCRIPTS / "browser" / "webgl2_integration.py")
PICTURE = load("gl_raster_picture", SCRIPTS / "gl_raster_picture.py")

# The picture the compatibility workload owes, read from the report a real context
# produced (an AMD Radeon 780M through the desktop driver).  It is the same
# retained hardware data `test_check_gl4_raster_readback.py` gates on, used here
# for a different question: whether the reader agrees with the writer the gates
# already trust.
HARDWARE_REPORT = Path(__file__).parent / "data" / "gl4_raster_readback_radeon_780m.json"

PAGE_BACKGROUND = (16, 16, 20)
CANVAS_CLEAR = (0, 0, 0)
CANVAS_BORDER = (51, 51, 51)
PANEL_TEXT = (232, 232, 238)
SCENE_FILLS = ((255, 0, 0), (0, 255, 0), (0, 89, 255))


def png_bytes(width: int, height: int, stream: bytes, *, colour_type: int = 2,
              bit_depth: int = 8, interlace: int = 0) -> bytes:
    """Wraps a filtered image stream in a PNG whose header this module chooses.

    The header is a parameter rather than a constant because the reader's refusals
    are part of what is being tested: a layout it cannot read has to fail the run
    rather than be guessed at.
    """

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, bit_depth, colour_type, 0, 0, interlace)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(stream, 6))
        + chunk(b"IEND", b"")
    )


def filtered_png(rows: list[bytes], filter_kind: int) -> bytes:
    """Returns RGB rows as a PNG whose every row uses ``filter_kind``.

    Chrome's own captures use filters 1, 2 and 4 (measured on the retained frame:
    798 rows of Up, 127 of Paeth and 14 of Sub), so the reader's predictors are the
    part of it that a fixture written by the repository's own writer would never
    reach: `scripts/gl_raster_picture.py` writes filter 0 only.  Filtering the rows
    here is what puts every branch of the reader under a test that a GPU-less
    runner can still execute.
    """
    channels = 3
    stride = len(rows[0])
    stream = bytearray()
    previous = bytes(stride)
    for row in rows:
        encoded = bytearray(stride)
        for index in range(stride):
            left = row[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_kind == 0:
                predictor = 0
            elif filter_kind == 1:
                predictor = left
            elif filter_kind == 2:
                predictor = above
            elif filter_kind == 3:
                predictor = (left + above) >> 1
            elif filter_kind == 4:
                predictor = CAPTURE.paeth(left, above, upper_left)
            else:
                raise ValueError(f"filter {filter_kind} is not a PNG filter")
            encoded[index] = (row[index] - predictor) & 0xFF
        stream += bytes([filter_kind]) + encoded
        previous = row
    return png_bytes(len(rows[0]) // channels, len(rows), bytes(stream))


def blend(colour: tuple[int, int, int], background: tuple[int, int, int],
          step: int, steps: int) -> bytes:
    """Returns ``colour`` blended onto ``background``, which is what edges are."""
    return bytes(
        (colour[channel] * step + background[channel] * (steps - step)) // steps
        for channel in range(3)
    )


def page_shaped_frame() -> tuple[int, int, list[bytes]]:
    """Returns a small frame carrying the page's kind of content.

    Deliberately not flat and deliberately not one fill: a canvas with its clear
    and its border, the retained scene's three fills with a graded edge each, and
    runs of light text with the graded ends a rasteriser puts on glyphs.
    """
    width, height = 160, 120
    rows = [bytearray(bytes(PAGE_BACKGROUND) * width) for _ in range(height)]

    def put(x: int, y: int, colour: bytes) -> None:
        rows[y][x * 3 : x * 3 + 3] = colour

    canvas_x, canvas_y, canvas_right, canvas_bottom = 8, 16, 104, 104
    for y in range(canvas_y, canvas_bottom):
        for x in range(canvas_x, canvas_right):
            put(x, y, bytes(CANVAS_CLEAR))
        put(canvas_x, y, bytes(CANVAS_BORDER))
        put(canvas_right - 1, y, bytes(CANVAS_BORDER))

    for index, fill in enumerate(SCENE_FILLS):
        left = canvas_x + 10 + index * 26
        for step in range(24):
            for x in range(left, left + step + 1):
                put(x, canvas_y + 40 + step, bytes(fill))
            put(left + step + 1, canvas_y + 40 + step, blend(fill, CANVAS_CLEAR, 1, 3))

    for line in range(6):
        for column in range(24):
            x = canvas_right + 6 + column * 2
            put(x, 24 + line * 9, bytes(PANEL_TEXT))
            put(x + 1, 24 + line * 9, blend(PANEL_TEXT, PAGE_BACKGROUND, 1, 3))

    return width, height, [bytes(row) for row in rows]


def write_png(directory: Path, name: str, rows: list[bytes]) -> bytes:
    """Writes a picture with the repository's own writer and returns its bytes."""
    path = directory / name
    PICTURE.write_png(path, len(rows[0]) // 3, len(rows), rows)
    return path.read_bytes()


class DecodePngTests(unittest.TestCase):
    def test_every_filter_decodes_to_the_rows_that_were_filtered(self) -> None:
        _, _, rows = page_shaped_frame()
        for filter_kind in range(5):
            with self.subTest(filter=filter_kind):
                image = filtered_png(rows, filter_kind)
                width, height, channels, pixels = CAPTURE.decode_png(image)
                self.assertEqual((width, height, channels), (len(rows[0]) // 3, len(rows), 3))
                self.assertEqual(pixels, b"".join(rows))

    def test_the_reader_agrees_with_the_repositorys_own_writer(self) -> None:
        report = json.loads(HARDWARE_REPORT.read_text(encoding="utf-8"))
        raw = PICTURE.pixels_from_hex(report["workload"]["colour"]["pixels_rgba8"])
        width, height, magnified = PICTURE.magnify(raw, PICTURE.EXPECTED_EXTENT, 3)
        with tempfile.TemporaryDirectory() as directory:
            image = write_png(Path(directory), "colour.png", magnified)
        decoded_width, decoded_height, channels, pixels = CAPTURE.decode_png(image)
        self.assertEqual((decoded_width, decoded_height, channels), (width, height, 3))
        # Row order is where the two pictures differ by construction, so the rows
        # the writer was handed are the rows that have to come back out.
        self.assertEqual(pixels, b"".join(magnified))

    def test_a_capture_that_is_not_a_png_is_refused(self) -> None:
        problems, measured = CAPTURE.inspect_frame(b"this is not a picture at all")
        self.assertEqual(measured, "unreadable")
        self.assertEqual(len(problems), 1)
        self.assertIn("PNG", problems[0])

    def test_a_truncated_capture_is_refused(self) -> None:
        _, _, rows = page_shaped_frame()
        problems, _ = CAPTURE.inspect_frame(filtered_png(rows, 0)[:-40])
        self.assertTrue(problems, "a capture whose data is cut short cannot be read")

    def test_a_layout_the_gate_cannot_read_is_refused(self) -> None:
        _, _, rows = page_shaped_frame()
        stream = b"".join(b"\x00" + row for row in rows)
        for layout, header in (
            ("interlaced", {"interlace": 1}),
            ("sixteen bits", {"bit_depth": 16}),
            ("no header", None),
        ):
            with self.subTest(layout=layout):
                image = png_bytes(len(rows[0]) // 3, len(rows), stream, **(header or {}))
                if header is None:
                    image = image[:8] + image[8:].replace(b"IHDR", b"nope", 1)
                problems, _ = CAPTURE.inspect_frame(image)
                self.assertTrue(problems, f"a {layout} PNG is not a frame this gate can vouch for")


class FlatFrameTests(unittest.TestCase):
    def test_a_single_colour_capture_is_refused(self) -> None:
        for name, colour in (
            ("all-black", (0, 0, 0)),
            ("all-white", (255, 255, 255)),
            ("blank", PAGE_BACKGROUND),
        ):
            with self.subTest(capture=name):
                rows = [bytes(colour) * 12 for _ in range(9)]
                with tempfile.TemporaryDirectory() as directory:
                    image = write_png(Path(directory), "flat.png", rows)
                problems, measured = CAPTURE.inspect_frame(image)
                self.assertIn("one flat colour", measured)
                self.assertEqual(len(problems), 1)
                self.assertIn("single flat colour", problems[0])
                self.assertIn(name if name != "blank" else "blank", problems[0])

    def test_a_capture_that_is_a_fill_and_a_band_is_refused(self) -> None:
        """The near neighbour a one-colour test would let through.

        A fill with a strip of something else in it is still a surface nothing was
        drawn into, and it is one pixel of extra content away from passing the flat
        test above.
        """
        rows = [bytes(CANVAS_CLEAR) * 12 for _ in range(9)]
        rows[4] = bytes(PANEL_TEXT) * 12
        with tempfile.TemporaryDirectory() as directory:
            image = write_png(Path(directory), "band.png", rows)
        problems, _ = CAPTURE.inspect_frame(image)
        self.assertEqual(len(problems), 1)
        self.assertIn("distinct colours", problems[0])

    def test_a_frame_with_the_pages_content_is_not_refused(self) -> None:
        width, height, rows = page_shaped_frame()
        with tempfile.TemporaryDirectory() as directory:
            image = write_png(Path(directory), "page.png", rows)
        problems, measured = CAPTURE.inspect_frame(image)
        self.assertEqual(problems, [], f"the floor refuses only fills: {measured}")
        self.assertIn(f"{width}x{height}", measured)

    def test_the_retained_headed_capture_is_not_refused(self) -> None:
        image = RETAINED_CAPTURE.read_bytes()
        problems, measured = CAPTURE.inspect_frame(image)
        self.assertEqual(
            problems, [], f"a real headed frame is the case that must pass: {measured}"
        )
        width, height, channels, pixels = CAPTURE.decode_png(image)
        self.assertGreater(width, 200)
        self.assertGreater(height, 200)
        present = {pixels[offset : offset + 3] for offset in range(0, len(pixels), channels)}
        # The retained scene's three fills, exactly as it submits them.  If the
        # reader's predictors were wrong these values would be smeared into
        # neighbours, so finding them is what makes the filters above evidence
        # about Chrome's PNGs rather than about this test's own fixtures.
        for fill in SCENE_FILLS:
            self.assertIn(bytes(fill), present, f"the scene's fill {fill} is in the frame")


class UncompositedSignalTests(unittest.TestCase):
    def test_the_compositors_own_signal_is_accepted(self) -> None:
        self.assertIsNone(INTEGRATION.uncomposited_problem("raf"))

    def test_the_bounded_timer_is_a_failed_run_and_says_why(self) -> None:
        problem = INTEGRATION.uncomposited_problem("timer")
        assert problem is not None
        # The message is the whole remedy an operator gets, so it has to name the
        # condition and the way out rather than only the signal.
        self.assertIn("occluded", problem)
        self.assertIn("minimized", problem)

    def test_the_message_does_not_offer_a_flag_the_vehicle_already_passes(self) -> None:
        # The vehicle passes `disable-backgrounding-occluded-windows` on every run,
        # so an error that told the operator to pass it would be telling them to do
        # what has already been done -- and the reason a timer-settled run is now
        # worth failing on is that occlusion is no longer the explanation.
        problem = INTEGRATION.uncomposited_problem("timer")
        assert problem is not None
        self.assertNotIn("--chrome-arg", problem)
        self.assertIn("already passes", problem)


class FakeCapture:
    """Stands in for the two subprocesses ``main`` starts: the vehicle and git.

    The browser must not be started by a unit test, and the decision being tested
    is what ``main`` does with the capture's *answer*, so the answer is what this
    returns.  It is installed in place of the module's ``subprocess`` rather than
    of one call, because both calls are made on the same name.
    """

    def __init__(self, returncode: int = 0, title: str = "FLUXEL-READY",
                 signal: str = "raf", stderr: str = "") -> None:
        self.TimeoutExpired = subprocess.TimeoutExpired
        self.commands: list[list[str]] = []
        self._capture = (returncode, page_output(title, signal), stderr)

    def run(self, command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        self.commands.append(command)
        if any(str(part).endswith("cdp_shot.py") for part in command):
            return subprocess.CompletedProcess(command, *self._capture)
        return subprocess.CompletedProcess(command, 0, "cafebabe\n", "")


def page_output(title: str, signal: str) -> str:
    """Returns the stdout of a capture, in the shape the vehicle prints it."""
    return (
        f"page title: {title}\n"
        "--- page text ---\n"
        f"30 frames: submitted=3 backpressure=27 · composited: {signal}\n"
        "--- end page text ---\n"
        "captured -> frame.png\n"
        "frame: 1779x939, 1997 distinct colours\n"
        f"sha256 {'a' * 64}\n"
        f"page state: {title}\n"
        "browser closed\n"
    )


class IntegrationRunTests(unittest.TestCase):
    """What the whole run returns, with the capture vehicle stood in for."""

    def run_main(self, capture: FakeCapture) -> tuple[int, str, str]:
        with tempfile.TemporaryDirectory() as directory:
            served = Path(directory) / "served"
            served.mkdir()
            (served / "index.html").write_text("<!doctype html><title>page</title>",
                                               encoding="utf-8")
            stdout, stderr = io.StringIO(), io.StringIO()
            with unittest.mock.patch.object(
                sys,
                "argv",
                ["webgl2_integration.py", "--served", str(served),
                 "--out", str(Path(directory) / "frame.png")],
            ), unittest.mock.patch.object(INTEGRATION, "build", lambda *_: None), \
                    unittest.mock.patch.object(INTEGRATION, "subprocess", capture):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = INTEGRATION.main()
        return code, stdout.getvalue(), stderr.getvalue()

    def test_a_page_the_compositor_confirmed_passes(self) -> None:
        code, out, _ = self.run_main(FakeCapture(signal="raf"))
        self.assertEqual(code, 0)
        # The frame's identity is in the log of a passing run, because that is the
        # record the release evidence is read from.
        self.assertIn(f"frame sha256 {'a' * 64}", out)
        self.assertIn("composited by raf", out)

    def test_the_occlusion_flag_is_passed_without_the_operator_asking(self) -> None:
        # The `raf` assertion is only reachable if Chrome keeps animating a window
        # it has decided is not worth animating, and a gate that depends on where
        # the operator left their windows is a gate that gets ignored.  So the flag
        # is a default rather than folklore: on this host the same run reported
        # `timer` without it and `raf` with it.
        capture = FakeCapture()
        self.run_main(capture)
        launch = next(
            command for command in capture.commands
            if any(str(part).endswith("cdp_shot.py") for part in command)
        )
        self.assertIn(
            "--chrome-arg=disable-backgrounding-occluded-windows", launch,
            "the vehicle must not depend on the operator passing the flag",
        )

    def test_a_page_that_settled_on_the_timer_fails(self) -> None:
        code, out, err = self.run_main(FakeCapture(signal="timer"))
        self.assertEqual(code, 1, "a frame the compositor never confirmed is not a passing run")
        self.assertIn("occluded", err)
        # The record is still printed before the run is failed: the capture exists,
        # and what it cannot support is the pass.
        self.assertIn(f"frame sha256 {'a' * 64}", out)

    def test_a_capture_the_vehicle_refused_to_call_a_frame_fails(self) -> None:
        code, _, err = self.run_main(
            FakeCapture(
                returncode=CAPTURE.FRAME_REFUSED,
                stderr="the capture is a single flat colour",
            )
        )
        self.assertEqual(code, 1)
        self.assertIn(f"exit {CAPTURE.FRAME_REFUSED}", err)


if __name__ == "__main__":
    unittest.main()
