#!/usr/bin/env python3
"""Run Fluxel's no-WSI Vulkan RHI conformance slices on an Android device.

The public RHI deliberately does not expose a constructor for a native Vulkan
provider.  That constructor is backend-private so native ownership cannot leak
into a portable application.  Consequently this runner builds the RHI's own
library-test executable, whose tests can exercise the private provider, rather
than introducing an example-only public escape hatch.

The resulting report distinguishes an unavailable setup from a failed RHI
contract.  It does not test Android presentation: no Android Surface, swapchain
or acquire/present operation is created by this no-WSI evidence gate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


DEFAULT_TARGET = "x86_64-linux-android"
DEFAULT_API = 29
DEFAULT_NDK = Path("target/tools/android-ndk-r28")
DEFAULT_OUT = Path("target/android-vulkan-evidence")
REMOTE_DIR = "/data/local/tmp/fluxel-vulkan-evidence"
REMOTE_BINARY = f"{REMOTE_DIR}/fluxel-rhi-vulkan-tests"

# These names intentionally point at RHI library tests, rather than a public
# sample wrapper.  Each test is a real queue/readback witness for the named
# vertical slice, and exact filtering makes the report stable across unrelated
# test additions.
SLICES = (
    (
        "enumerate_request",
        "backend::vulkan::platform::tests::loader_enumeration_request_identity_and_idle",
    ),
    (
        "buffer_transfer_readback",
        "backend::vulkan::platform::tests::upload_copy_and_readback_move_bytes_on_a_real_vulkan_queue",
    ),
    (
        "texture_transfer_readback",
        "backend::vulkan::platform::tests::texture_upload_copy_and_readback_move_texels_on_a_real_vulkan_queue",
    ),
    (
        "compute",
        "backend::vulkan::compute_tests::buffer_compute_dispatch_keeps_dropped_caller_handles_alive_until_readback",
    ),
    (
        "offscreen_raster",
        "backend::vulkan::raster_tests::offscreen_raster_draw_transitions_to_readback_and_retains_native_objects",
    ),
    (
        "sampled_storage_image",
        "backend::vulkan::image_binding_tests::compute_sampled_filtering_and_storage_images_round_trip_through_vulkan",
    ),
)


class Failure(RuntimeError):
    """A setup failure that prevents obtaining Android Vulkan evidence."""


def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run a command without letting a nonzero status lose its diagnostics."""
    return subprocess.run(command, capture_output=True, text=True, check=False, **kwargs)


def target_linker(ndk: Path, target: str, api: int) -> Path | None:
    """Return the NDK's target-and-API-specific linker wrapper, if present."""
    name = f"{target}{api}-clang.cmd"
    for host in ("windows-x86_64", "linux-x86_64", "darwin-x86_64"):
        candidate = ndk / "toolchains" / "llvm" / "prebuilt" / host / "bin" / name
        if candidate.exists():
            return candidate
    return None


def resolve_ndk(explicit: Path | None, root: Path, target: str, api: int) -> Path:
    """Find an NDK that can link the requested ABI at the requested API floor."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    for variable in ("FLUXEL_ANDROID_NDK", "ANDROID_NDK_HOME", "NDK_HOME"):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value))
    candidates.append(root / DEFAULT_NDK)
    for candidate in candidates:
        if target_linker(candidate, target, api) is not None:
            return candidate
    looked = ", ".join(str(candidate) for candidate in candidates)
    raise Failure(
        f"no NDK has a {target} API {api} linker; looked in {looked}. "
        "Set FLUXEL_ANDROID_NDK to NDK r26 or newer."
    )


def resolve_adb(explicit: Path | None) -> str:
    if explicit is not None:
        if not explicit.exists():
            raise Failure(f"the adb at {explicit} does not exist")
        return str(explicit)
    adb = shutil.which("adb")
    if adb is None:
        raise Failure("adb is not on PATH; set FLUXEL_ADB or install Android platform-tools")
    return adb


def ready_serials(adb: str) -> list[str]:
    completed = run([adb, "devices"])
    if completed.returncode != 0:
        raise Failure(f"adb devices failed: {completed.stderr.strip()[-400:]}")
    return [
        fields[0]
        for line in completed.stdout.splitlines()[1:]
        if (fields := line.split()) and len(fields) >= 2 and fields[1] == "device"
    ]


def resolve_serial(adb: str, requested: str | None) -> str:
    attached = ready_serials(adb)
    if requested is not None:
        if requested not in attached:
            raise Failure(f"{requested} is not attached and ready; attached: {attached}")
        return requested
    if len(attached) != 1:
        raise Failure(
            "exactly one ready Android device is required; "
            f"attached: {attached}. Pass --serial to choose one."
        )
    return attached[0]


def adb_shell(adb: str, serial: str, command: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return run([adb, "-s", serial, "shell", command], **kwargs)


def device_property(adb: str, serial: str, name: str) -> str:
    result = adb_shell(adb, serial, f"getprop {name}")
    if result.returncode != 0:
        raise Failure(f"could not query {name}: {result.stderr.strip()[-300:]}")
    return result.stdout.strip()


def assert_target_abi(adb: str, serial: str, target: str) -> None:
    # This runner currently has one explicit ABI rather than guessing a linker
    # from a device.  A mismatch is a setup failure, never a Vulkan failure.
    expected = target.removesuffix("-linux-android")
    abi = device_property(adb, serial, "ro.product.cpu.abi")
    abilist = device_property(adb, serial, "ro.product.cpu.abilist")
    if expected != abi and expected not in abilist.split(","):
        raise Failure(
            f"device ABI {abi!r} ({abilist!r}) cannot run target {target!r}; "
            "add a matching target/linker path rather than publishing mismatched evidence"
        )


def build_test_binary(root: Path, ndk: Path, target: str, api: int, online: bool) -> Path:
    """Cross-build the private-provider RHI test executable and find its path."""
    linker = target_linker(ndk, target, api)
    if linker is None:
        raise Failure(f"NDK {ndk} no longer supplies a linker for {target} API {api}")
    environment = dict(os.environ)
    variable = f"CARGO_TARGET_{target.upper().replace('-', '_')}_LINKER"
    environment[variable] = str(linker)
    command = [
        "cargo",
        "test",
        "-p",
        "fluxel-rhi",
        "--lib",
        "--no-default-features",
        "--features",
        "vulkan",
        "--target",
        target,
        "--no-run",
        "--message-format=json-render-diagnostics",
    ]
    if not online:
        command.append("--offline")
    completed = run(command, cwd=root, env=environment)
    if completed.returncode != 0:
        raise Failure(
            "the Android Vulkan test executable did not build:\n"
            f"{completed.stderr.strip()[-1600:]}"
        )

    # Cargo's JSON artifact stream is the only stable source for a hashed test
    # executable name.  Searching target/ risks selecting an obsolete binary.
    executables: list[Path] = []
    for line in completed.stdout.splitlines():
        try:
            artifact = json.loads(line)
        except json.JSONDecodeError:
            continue
        if artifact.get("reason") != "compiler-artifact":
            continue
        target_info = artifact.get("target") or {}
        if target_info.get("name") != "fluxel_rhi":
            continue
        executable = artifact.get("executable")
        if executable:
            executables.append(Path(executable))
    if len(executables) != 1 or not executables[0].exists():
        raise Failure(
            "Cargo did not report exactly one fluxel-rhi library test executable; "
            f"reported: {[str(path) for path in executables]}"
        )
    return executables[0]


def push_binary(adb: str, serial: str, binary: Path) -> None:
    prepared = adb_shell(adb, serial, f"mkdir -p {REMOTE_DIR}")
    if prepared.returncode != 0:
        raise Failure(f"could not create {REMOTE_DIR}: {prepared.stderr.strip()[-400:]}")
    pushed = run([adb, "-s", serial, "push", str(binary), REMOTE_BINARY])
    if pushed.returncode != 0:
        raise Failure(f"could not push test executable: {pushed.stderr.strip()[-400:]}")
    executable = adb_shell(adb, serial, f"chmod 755 {REMOTE_BINARY}")
    if executable.returncode != 0:
        raise Failure(f"could not chmod test executable: {executable.stderr.strip()[-400:]}")


def vkjson_summary(adb: str, serial: str) -> dict[str, Any]:
    """Record useful driver facts without storing the enormous full vkjson dump."""
    result = adb_shell(adb, serial, "cmd gpu vkjson")
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip()[-600:]}
    try:
        document = json.loads(result.stdout)
        device = (document.get("devices") or [{}])[0]
        extensions = {
            item.get("extensionName")
            # Android's `cmd gpu vkjson` calls this top-level list `extensions`
            # rather than `instanceExtensions`; per-device extensions live on
            # the device object under the same name.
            for item in document.get("extensions") or []
            if isinstance(item, dict)
        }
        properties = device.get("properties") or {}
        core12_properties = (device.get("core12") or {}).get("properties") or {}
        return {
            "available": True,
            "api_version": document.get("apiVersion"),
            "device_name": properties.get("deviceName"),
            "vendor_id": properties.get("vendorID"),
            "device_id": properties.get("deviceID"),
            "device_type": properties.get("deviceType"),
            "driver_name": core12_properties.get("driverName"),
            "driver_info": core12_properties.get("driverInfo"),
            "android_surface_extension": "VK_KHR_android_surface" in extensions,
            "surface_extension": "VK_KHR_surface" in extensions,
        }
    except (ValueError, TypeError, IndexError) as error:
        return {
            "available": False,
            "error": f"cmd gpu vkjson did not return parseable JSON: {error}",
            "raw_tail": result.stdout[-600:],
        }


def run_slice(adb: str, serial: str, name: str, test: str, timeout: float) -> dict[str, Any]:
    invocation = f"{REMOTE_BINARY} {test} --exact --nocapture"
    try:
        completed = adb_shell(adb, serial, invocation, timeout=timeout)
        warnings = [
            line for line in completed.stderr.splitlines() if line.startswith("FORTIFY:")
        ]
        return {
            "name": name,
            "test": test,
            "passed": completed.returncode == 0,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            # Keep emulator/native runtime diagnostics visible even where the
            # RHI assertion itself passed.  They are not silently promoted to
            # an RHI contract failure without a reproducible RHI error.
            "native_runtime_warnings": warnings,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "name": name,
            "test": test,
            "passed": False,
            "exit_code": None,
            "stdout": (error.stdout or "") if isinstance(error.stdout, str) else "",
            "stderr": (error.stderr or "") if isinstance(error.stderr, str) else "",
            "timeout": True,
        }


def cleanup(adb: str, serial: str) -> None:
    # The directory is fixed and scoped to this runner.  Failure is nonfatal:
    # evidence already reached the host by this point.
    removed = adb_shell(adb, serial, f"rm -rf {REMOTE_DIR}")
    if removed.returncode != 0:
        print(f"note: could not remove {REMOTE_DIR}: {removed.stderr.strip()[-300:]}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ndk", type=Path)
    parser.add_argument("--adb", type=Path)
    parser.add_argument("--serial")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--api", type=int, default=DEFAULT_API)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="per-slice device execution timeout in seconds")
    parser.add_argument("--keep-remote", action="store_true")
    parser.add_argument("--online", action="store_true",
                        help="allow Cargo registry access; offline is the default")
    arguments = parser.parse_args()

    root = arguments.root.resolve()
    out = (arguments.out or (root / DEFAULT_OUT)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "fluxel.android-vulkan-evidence.v1",
        "scope": "Vulkan RHI core without Android WSI/presentation",
        "target": arguments.target,
        "api_floor": arguments.api,
        "slices": [],
    }
    adb: str | None = None
    serial: str | None = None
    status = 1
    try:
        ndk = resolve_ndk(arguments.ndk, root, arguments.target, arguments.api)
        adb = resolve_adb(arguments.adb)
        serial = resolve_serial(adb, arguments.serial)
        assert_target_abi(adb, serial, arguments.target)
        report["device"] = {
            "serial": serial,
            "manufacturer": device_property(adb, serial, "ro.product.manufacturer"),
            "model": device_property(adb, serial, "ro.product.model"),
            "android_release": device_property(adb, serial, "ro.build.version.release"),
            "api_level": device_property(adb, serial, "ro.build.version.sdk"),
            "primary_abi": device_property(adb, serial, "ro.product.cpu.abi"),
            "abi_list": device_property(adb, serial, "ro.product.cpu.abilist"),
        }
        report["vulkan"] = vkjson_summary(adb, serial)
        binary = build_test_binary(root, ndk, arguments.target, arguments.api, arguments.online)
        report["test_binary"] = str(binary)
        push_binary(adb, serial, binary)
        report["slices"] = [
            run_slice(adb, serial, name, test, arguments.timeout) for name, test in SLICES
        ]
        status = 0 if all(slice_["passed"] for slice_ in report["slices"]) else 1
    except Failure as error:
        report["setup_error"] = str(error)
    finally:
        if adb is not None and serial is not None and not arguments.keep_remote:
            cleanup(adb, serial)
        report["passed"] = status == 0
        path = out / "report.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Android Vulkan evidence: {out / 'report.json'}")
    if status == 0:
        print(f"all {len(SLICES)} no-WSI Vulkan RHI slices passed on {serial}")
    elif "setup_error" in report:
        print(f"Android Vulkan evidence setup failed: {report['setup_error']}", file=sys.stderr)
    else:
        failed = [slice_["name"] for slice_ in report["slices"] if not slice_["passed"]]
        print(f"Android Vulkan RHI slices failed: {', '.join(failed)}", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
