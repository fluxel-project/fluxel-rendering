from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPTS))

# The picture contract is its own module and both GL-family gates import it, so
# this test does the same rather than reaching for it through the vehicle: the
# vehicle re-exports what it uses and `pixels_from_hex` is not one of them.
import gl_raster_picture  # noqa: E402  (after the path it needs)

SCRIPT = SCRIPTS / "android_gles_evidence.py"
SPEC = importlib.util.spec_from_file_location("android_gles_evidence", SCRIPT)
assert SPEC and SPEC.loader
VEHICLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VEHICLE
SPEC.loader.exec_module(VEHICLE)

# The report a real device produced while driving the workload -- LDPlayer 9/14's
# GLES stack on an API 29 x86_64 image, whose host-side renderer executes the
# command stream on an AMD Radeon 780M, kept so that a machine with no device can
# still run this gate's logic and so that the passing case is a device's answer
# rather than one written to pass.
#
# The guest strings in it are a *profile*, not hardware: the image reports a
# `nubia NX809J` with an `Adreno (TM) 750`, which no such physical device
# produced.  The file is named for what it says rather than for what it is, so
# that nobody reads the renderer field as a device fingerprint -- `CLAUDE.md`
# §4.5, and the module doc says the same thing beside the gate itself.
HARDWARE_REPORT = Path(__file__).parent / "data/gles_raster_readback_ldplayer_adreno750.json"

# A listing with one usable device, one unauthorized one and one offline one.
# The tail rows are the point: a parser that took every non-header line would
# offer a serial that fails at the push instead of here.
DEVICES = """List of devices attached
emulator-5554\tdevice
ZY223KJHFG\tunauthorized
emulator-5556\toffline
"""


def hardware() -> tuple[dict, bytes]:
    """Returns the hardware report and the bytes its own hex grid stands for.

    The bytes are the report's, and that is the point rather than a shortcut: the
    gate requires the two statements of the picture to agree, so a fixture whose
    file disagreed with its grid would be testing a run that failed for a reason
    no expected picture can diagnose.
    """
    report = json.loads(HARDWARE_REPORT.read_text(encoding="utf-8"))
    return report, gl_raster_picture.pixels_from_hex(report["workload"]["colour"]["pixels_rgba8"])


class AndroidGlesEvidenceTests(unittest.TestCase):
    def test_the_device_frame_is_the_picture_the_workload_requires(self) -> None:
        report, raw = hardware()
        self.assertEqual(VEHICLE.check(report, raw), [])

    def test_the_device_answered_with_the_embedded_profile_the_evidence_is_for(self) -> None:
        """The one reading the owed item is about, asserted rather than eyeballed.

        ``0.15-plan.md`` owes *the GLES 3.1 triangle*, so a run that reached a
        GLES 3.0 context, or a desktop one, would produce a frame that passes the
        picture gate and does not answer the question.  The profile and the
        version string are separate fields and both are checked, because a
        provider that reported 3.0 in one and 3.1 in the other is exactly the
        disagreement this is here to catch.
        """
        report, _ = hardware()
        self.assertEqual(report["profile"], "Embedded { major: 3, minor: 1 }")
        self.assertEqual(report["version"], "OpenGL ES 3.1 v1")

    def test_a_frame_with_nothing_drawn_in_it_fails(self) -> None:
        """The failure the gate exists for: every counter passes, nothing is green."""
        report, raw = hardware()
        # Opaque black everywhere, which is the clear the pass stores: the frame
        # is exactly what the run would leave if the draw had been skipped.  Both
        # statements of it are blanked together, because a run whose report and
        # whose file disagree fails on *that* first -- which is the gate working,
        # and would leave the per-pixel check below untested if only one changed.
        blank = bytes(0 if index % 4 != 3 else 255 for index in range(len(raw)))
        self.assertNotEqual(blank, raw)
        report["workload"]["colour"]["pixels_rgba8"] = ["000000ff"] * (len(raw) // 4)
        problems = VEHICLE.check(report, blank)
        self.assertTrue(problems)
        self.assertTrue(any("covered pixel" in problem for problem in problems))
        self.assertTrue(any("nothing was drawn" in problem for problem in problems))

    def test_a_report_that_disagrees_with_its_own_file_fails(self) -> None:
        """The two statements of one picture, checked against each other first."""
        report, raw = hardware()
        report["workload"]["colour"]["pixels_rgba8"][5] = "ffb070ff"
        problems = VEHICLE.check(report, raw)
        self.assertEqual(len(problems), 1)
        self.assertIn("different pictures", problems[0])

    def test_only_ready_devices_are_offered(self) -> None:
        self.assertEqual(VEHICLE.attached_devices(DEVICES), ["emulator-5554"])

    def test_an_empty_listing_is_no_devices(self) -> None:
        self.assertEqual(VEHICLE.attached_devices("List of devices attached\n\n"), [])

    def test_the_one_device_is_chosen_without_being_named(self) -> None:
        self.assertEqual(VEHICLE.choose_serial(["emulator-5554"], None), "emulator-5554")

    def test_no_device_is_refused_rather_than_guessed(self) -> None:
        with self.assertRaises(VEHICLE.Failure):
            VEHICLE.choose_serial([], None)

    def test_two_devices_are_refused_rather_than_picked_between(self) -> None:
        """Which machine the evidence came from is not a detail to resolve by order."""
        with self.assertRaises(VEHICLE.Failure) as caught:
            VEHICLE.choose_serial(["emulator-5554", "emulator-5556"], None)
        self.assertIn("--serial", str(caught.exception))

    def test_a_named_device_that_is_not_attached_is_refused(self) -> None:
        with self.assertRaises(VEHICLE.Failure):
            VEHICLE.choose_serial(["emulator-5554"], "emulator-5556")

    def test_an_ndk_without_a_linker_for_this_api_is_not_an_ndk(self) -> None:
        """A toolchain is only usable if it can link for the ABI the device runs.

        Pointing the gate at a directory that exists is the likely mistake, so
        the search is on the wrapper rather than on the root: an NDK that has
        none for this target and API produces a linker error several steps later
        that says nothing about the NDK.
        """
        missing = Path(__file__).parents[1]
        self.assertIsNone(VEHICLE.linker_wrapper(missing, VEHICLE.DEFAULT_API))

    def test_the_checked_in_ndk_is_found_when_it_is_present(self) -> None:
        """The path the release gate actually uses, when this checkout has it.

        Skipped rather than failed when the toolchain has not been fetched: it
        lives under `target/`, which is regenerable by design (`CLAUDE.md` §7),
        and a test that failed on a fresh checkout would be asserting that a
        downloaded artifact is committed.
        """
        root = Path(SCRIPT).parents[1]
        ndk = root / VEHICLE.DEFAULT_NDK
        if not ndk.exists():
            self.skipTest(f"{ndk} has not been fetched on this checkout")
        wrapper = VEHICLE.linker_wrapper(ndk, VEHICLE.DEFAULT_API)
        self.assertIsNotNone(wrapper)
        self.assertTrue(wrapper.exists())


if __name__ == "__main__":
    unittest.main()
