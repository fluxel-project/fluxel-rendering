"""Build, install, run, and preserve Android Vulkan WSI evidence."""

from __future__ import annotations

import json
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "org.fluxel.vulkanwsi"
ACTIVITY = f"{PACKAGE}/android.app.NativeActivity"


def command(*args: str, check: bool = True, **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=check, **kwargs)


def latest_child(parent: Path, predicate) -> Path:
    candidates = sorted((path for path in parent.iterdir() if predicate(path)), reverse=True)
    if not candidates:
        raise RuntimeError(f"no suitable installation found below {parent}")
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="emulator-5554")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=ROOT / "target/android-vulkan-wsi")
    parser.add_argument(
        "--android-home",
        type=Path,
        default=(os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")),
    )
    parser.add_argument("--ndk", type=Path, default=ROOT / "target/tools/android-ndk-r28")
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"required tool or input is missing: {path}")


def main() -> int:
    args = parse_args()
    if args.android_home is None:
        raise RuntimeError("ANDROID_HOME or ANDROID_SDK_ROOT is required")
    build_tools = latest_child(args.android_home / "build-tools", lambda path: path.is_dir())
    platform = latest_child(
        args.android_home / "platforms",
        lambda path: path.is_dir() and (path / "android.jar").is_file(),
    )
    android_jar = platform / "android.jar"
    linker = args.ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin/x86_64-linux-android29-clang.cmd"
    out = args.out.resolve()
    report_path = out / "report.json"
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"serial": args.serial, "package": PACKAGE, "passed": False}

    def adb(*adb_args: str, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return command("adb", "-s", args.serial, *adb_args, **kwargs)

    try:
        for name in ("aapt.exe", "zipalign.exe", "apksigner.bat"):
            require(build_tools / name)
        require(android_jar)
        require(linker)
        env = os.environ.copy()
        env["CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER"] = str(linker)
        command("cargo", "build", "--manifest-path", "examples/android-vulkan-wsi/Cargo.toml",
                "--target", "x86_64-linux-android", "--release", cwd=ROOT, env=env)
        native = ROOT / "examples/android-vulkan-wsi/target/x86_64-linux-android/release/libfluxel_android_vulkan_wsi.so"
        require(native)
        staging = out / "staging"
        if staging.exists():
            shutil.rmtree(staging)
        library_dir = staging / "lib/x86_64"
        library_dir.mkdir(parents=True)
        shutil.copy2(native, library_dir / native.name)
        unsigned, aligned, signed = (out / "unsigned.apk", out / "aligned.apk", out / "signed.apk")
        command(str(build_tools / "aapt.exe"), "package", "-f", "-0", "so", "-M",
                "examples/android-vulkan-wsi/AndroidManifest.xml", "-I", str(android_jar), "-F", str(unsigned), str(staging), cwd=ROOT)
        command(str(build_tools / "zipalign.exe"), "-f", "4", str(unsigned), str(aligned))
        keystore = Path.home() / ".android/debug.keystore"
        require(keystore)
        command(str(build_tools / "apksigner.bat"), "sign", "--ks", str(keystore),
                "--ks-pass", "pass:android", "--key-pass", "pass:android", "--out", str(signed), str(aligned))
        adb("install", "-r", str(signed))
        adb("logcat", "-c")
        adb("shell", "am", "force-stop", PACKAGE)
        adb("shell", "am", "start", "-n", ACTIVITY)
        deadline = time.monotonic() + args.timeout
        evidence: dict[str, object] | None = None
        logcat = ""
        while time.monotonic() < deadline:
            logcat = adb("logcat", "-d", "-v", "brief").stdout
            for line in reversed(logcat.splitlines()):
                if "FluxelVulkanWSI" not in line or "{\"schema\"" not in line:
                    continue
                try:
                    evidence = json.loads(line[line.index("{\""):])
                except json.JSONDecodeError:
                    continue
                break
            if evidence is not None:
                break
            time.sleep(0.25)
        report["logcat"] = logcat[-12000:]
        report["evidence"] = evidence
        report["passed"] = bool(
            evidence
            and all(evidence.get(key) is True for key in ("acquire", "raster_clear", "reacquire", "reconfigure"))
            and evidence.get("present") == "Accepted"
        )
        if not report["passed"]:
            raise RuntimeError("NativeActivity produced no complete Vulkan WSI evidence")
    except Exception as error:
        report["error"] = str(error)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
