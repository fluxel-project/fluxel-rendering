"""Run and retain one auditable sampled-material device attempt.

The runner preserves the actual workload bytes, command logs, a raw source
snapshot, source hashes, lock identity, and the pre/post working-tree state.
It intentionally does not interpret an earlier output directory as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from typing import Any

from material_attempt_diagnostics import inspect as inspect_diagnostics
from smoke_inputs import excluded_source_parts


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "crates" / "renderer" / "examples" / "rendering-integration" / "Cargo.toml"
INTEGRATION_LOCK = MANIFEST.parent / "Cargo.lock"
SOURCE_ROOTS = (
    Path(".gitignore"),
    Path("Cargo.toml"),
    Path("Cargo.lock"),
    Path("crates/renderer/examples/rendering-integration/Cargo.lock"),
    Path("crates/material"),
    Path("crates/shader"),
    Path("crates/renderer"),
    Path("tools/integration/m0_sampled_attempt.py"),
    Path("tools/integration/smoke_inputs.py"),
    Path("tools/integration/material_attempt_diagnostics.py"),
)
WORKLOAD_SELECTORS = {
    "e0-texture-material-forward-deferred": "--offscreen",
    "e0-texture-material-surface": "--visible",
    "m1-srgb-filter-forward-deferred": "--offscreen",
    "m1-srgb-filter-surface": "--visible",
    "m1-material-bindings-forward-deferred": "--offscreen",
    "m1-material-bindings-surface": "--visible",
    "m1-four-texture-forward-deferred": "--offscreen",
    "m1-four-texture-surface": "--visible",
    "m1-mixed-capture-forward": "--offscreen",
    "m1-mixed-capture-surface": "--visible",
    "m1-material-mips-forward-deferred": "--offscreen",
    "m1-material-mips-surface": "--visible",
    "m1-mipmap-tunnel-forward-deferred": "--offscreen",
    "m1-mipmap-tunnel-surface": "--visible",
    "e0-dynamic-uniform-material-forward": "--offscreen",
    "e0-dynamic-uniform-material-surface": "--visible",
    "e0-capture-material-forward": "--offscreen",
    "e0-capture-material-surface": "--visible",
    "m0-sampled-forward-deferred": "--m0-sampled-forward-only",
    "m1-specular-ibl-forward-deferred": "--offscreen",
    "m1-specular-ibl-surface": "--visible",
    "m1-orm-ibl-sphere-forward-deferred": "--offscreen",
    "m1-orm-ibl-sphere-surface": "--visible",
    "m1-orm-ibl-multi-object-forward-deferred": "--offscreen",
    "m1-orm-ibl-multi-object-surface": "--visible",
    "m1-uv0-pbr-forward-deferred": "--m1-uv0-pbr-only",
    "m1-normal-pbr-forward-deferred": "--m1-normal-pbr-only",
    "m1-environment-irradiance-forward-deferred": "--m1-environment-irradiance-forward-only",
    "m1-surface-effects-forward-deferred": "--m1-surface-effects-only",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout


def working_tree() -> dict[str, str]:
    return {
        "head": output(["git", "rev-parse", "HEAD"]).strip(),
        "status": output(["git", "status", "--porcelain=v1", "--untracked-files=all"]),
        "diff_binary": subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        .stdout.decode("utf-8", "surrogateescape"),
    }


def source_files() -> list[Path]:
    files: list[Path] = []
    for relative in SOURCE_ROOTS:
        path = ROOT / relative
        if path.is_file():
            files.append(relative)
        elif path.is_dir():
            for candidate in path.rglob("*"):
                if not candidate.is_file():
                    continue
                candidate_relative = candidate.relative_to(ROOT)
                if excluded_source_parts(ROOT) & set(candidate_relative.parts):
                    continue
                files.append(candidate_relative)
    return sorted(set(files), key=lambda item: item.as_posix())


def source_records(files: list[Path]) -> list[dict[str, str]]:
    return [{"path": path.as_posix(), "sha256": sha256(ROOT / path)} for path in files]


def source_records_with_errors(
    files: list[Path],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Hash a fixed enumeration without losing an attempt to a deletion race."""
    records: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for path in files:
        try:
            records.append({"path": path.as_posix(), "sha256": sha256(ROOT / path)})
        except FileNotFoundError:
            errors.append({"path": path.as_posix(), "error": "missing-during-hash"})
    return records, errors


def source_archive(destination: Path, files: list[Path]) -> str:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for relative in files:
            archive.write(ROOT / relative, relative.as_posix())
    return sha256(destination)


def archived_source_records(snapshot: Path) -> list[dict[str, str]]:
    """Identify the bytes actually retained, independently of the live tree."""
    with zipfile.ZipFile(snapshot) as archive:
        return sorted(
            [
                {"path": name, "sha256": hashlib.sha256(archive.read(name)).hexdigest()}
                for name in archive.namelist()
            ],
            key=lambda item: item["path"],
        )


def resolved_git_packages(lock_path: Path = INTEGRATION_LOCK) -> list[dict[str, str]]:
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = []
    for package in lock.get("package", []):
        source = package.get("source", "")
        if source.startswith("git+"):
            packages.append(
                {
                    "name": package["name"],
                    "version": package["version"],
                    "source": source,
                    "revision": source.rsplit("#", 1)[-1],
                }
            )
    return sorted(packages, key=lambda item: (item["name"], item["revision"]))


def artifact_records(root: Path) -> list[dict[str, str]]:
    return [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "attempt.json"
    ]


def find_built_executable(build_stdout: str, manifest: Path) -> Path | None:
    """Return Cargo's reported executable for exactly the selected consumer."""
    package_name = tomllib.loads(manifest.read_text(encoding="utf-8"))["package"]["name"]
    executable = None
    for line in build_stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = message.get("target", {})
        reported_manifest = message.get("manifest_path")
        if (
            message.get("reason") == "compiler-artifact"
            and message.get("executable")
            and isinstance(reported_manifest, str)
            and Path(reported_manifest).resolve() == manifest.resolve()
            and target.get("name") == package_name
            and "bin" in target.get("kind", [])
        ):
            executable = Path(message["executable"])
    return executable


def retain_executable(source: Path, attempt: Path) -> dict[str, str]:
    """Copy the exact Cargo artifact before a later build can overwrite it."""
    digest = sha256(source)
    attempt.mkdir(parents=True, exist_ok=True)
    destination = attempt / f"consumer{source.suffix}"
    shutil.copy2(source, destination)
    if sha256(source) != digest or sha256(destination) != digest:
        raise ValueError("Cargo executable changed while retaining the attempt binary")
    return {
        "path": destination.relative_to(attempt).as_posix(),
        "sha256": digest,
        "cargoReportedPath": str(source),
    }


def command_record(
    command: list[str], completed: subprocess.CompletedProcess[str], stdout_name: str, stderr_name: str
) -> dict[str, Any]:
    return {
        "command": command,
        "exitStatus": completed.returncode,
        "stdout": stdout_name,
        "stderr": stderr_name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("dx12", "native-gl-wgl"), required=True)
    parser.add_argument(
        "--workload",
        choices=tuple(WORKLOAD_SELECTORS),
        default="m0-sampled-forward-deferred",
        help="named workload whose selector and evidence declaration are retained",
    )
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--validate", action="store_true", help="require actual native diagnostic activation and no native diagnostics")
    parser.add_argument("--asset", type=Path, help="local KTX input for the E0 texture example")
    parser.add_argument("--model", type=Path, help="original RHI 09 tunnel glTF input")
    parser.add_argument("--dragon", type=Path, help="local dragon glTF for the E0 capture example")
    parser.add_argument("--plane", type=Path, help="local plane glTF for the E0 capture example")
    args = parser.parse_args()
    manifest_path = MANIFEST
    lock_path = INTEGRATION_LOCK
    workload_args: list[str] = []
    if args.workload in {"e0-texture-material-forward-deferred", "e0-texture-material-surface"}:
        if args.asset is None:
            parser.error("the E0 texture example requires --asset")
        manifest_path = ROOT / "crates/renderer/examples/02-texture-material/Cargo.toml"
        lock_path = manifest_path.parent / "Cargo.lock"
        workload_args = ["--asset", str(args.asset.resolve())]
        if args.workload == "e0-texture-material-surface":
            workload_args.append("--exit-after-capture")
    elif args.workload in {"m1-mipmap-tunnel-forward-deferred", "m1-mipmap-tunnel-surface"}:
        if args.asset is None or args.model is None:
            parser.error("the mipmap tunnel example requires --asset and --model")
        manifest_path = ROOT / "crates/renderer/examples/09-mipmap-tunnel-material/Cargo.toml"
        lock_path = manifest_path.parent / "Cargo.lock"
        workload_args = ["--asset", str(args.asset.resolve()), "--model", str(args.model.resolve())]
        if args.workload.endswith("surface"):
            workload_args.append("--exit-after-capture")
    elif args.workload in {
        "m1-four-texture-forward-deferred", "m1-four-texture-surface",
        "m1-mixed-capture-forward", "m1-mixed-capture-surface",
        "m1-material-mips-forward-deferred", "m1-material-mips-surface",
    }:
        manifest_path = ROOT / "crates/renderer/examples/pbr-four-texture-material/Cargo.toml"
        lock_path = manifest_path.parent / "Cargo.lock"
        if args.workload.startswith("m1-mixed-capture-"):
            workload_args.append("--mixed-capture")
        if args.workload.startswith("m1-material-mips-"):
            workload_args.append("--mip-generation")
        if args.workload.endswith("surface"):
            workload_args.append("--exit-after-capture")
    elif args.workload in {
        "m1-srgb-filter-forward-deferred", "m1-srgb-filter-surface",
        "m1-material-bindings-forward-deferred", "m1-material-bindings-surface",
    }:
        manifest_path = ROOT / "crates/renderer/examples/02-texture-material/Cargo.toml"
        lock_path = manifest_path.parent / "Cargo.lock"
        workload_args = ["--srgb-filter"]
        if args.workload.startswith("m1-material-bindings-"):
            workload_args.append("--permuted-bindings")
        if args.workload.endswith("surface"):
            workload_args.append("--exit-after-capture")
    elif args.workload in {
        "m1-specular-ibl-forward-deferred", "m1-specular-ibl-surface",
        "m1-orm-ibl-sphere-forward-deferred", "m1-orm-ibl-sphere-surface",
        "m1-orm-ibl-multi-object-forward-deferred", "m1-orm-ibl-multi-object-surface",
    }:
        example = "pbr-sphere-material" if args.workload.startswith("m1-orm-ibl-") else "pbr-environment-material"
        manifest_path = ROOT / "crates/renderer/examples" / example / "Cargo.toml"
        lock_path = manifest_path.parent / "Cargo.lock"
        if args.workload.endswith("surface"):
            workload_args.append("--exit-after-capture")
        if args.workload.startswith("m1-orm-ibl-multi-object-"):
            workload_args.append("--multi-records")
    elif args.workload in {"e0-dynamic-uniform-material-forward", "e0-dynamic-uniform-material-surface"}:
        manifest_path = ROOT / "crates/renderer/examples/04-dynamic-uniform-material/Cargo.toml"
        lock_path = manifest_path.parent / "Cargo.lock"
        if args.workload == "e0-dynamic-uniform-material-surface":
            workload_args.append("--exit-after-capture")
    elif args.workload in {"e0-capture-material-forward", "e0-capture-material-surface"}:
        if args.dragon is None or args.plane is None:
            parser.error("the capture example requires --dragon and --plane")
        manifest_path = ROOT / "crates/renderer/examples/07-offscreen-material/Cargo.toml"
        lock_path = manifest_path.parent / "Cargo.lock"
        workload_args = ["--dragon", str(args.dragon.resolve()), "--plane", str(args.plane.resolve())]
        if args.workload == "e0-capture-material-surface":
            workload_args.append("--exit-after-capture")
    if args.asset is not None and args.workload not in {"e0-texture-material-forward-deferred", "e0-texture-material-surface", "m1-mipmap-tunnel-forward-deferred", "m1-mipmap-tunnel-surface"}:
        parser.error("--asset applies only to the texture and mipmap-tunnel examples")
    if args.model is not None and not args.workload.startswith("m1-mipmap-tunnel-"):
        parser.error("--model applies only to the mipmap-tunnel example")
    if (args.dragon is not None or args.plane is not None) and args.workload not in {"e0-capture-material-forward", "e0-capture-material-surface"}:
        parser.error("--dragon/--plane apply only to the capture example")

    attempt = args.attempt_dir.resolve()
    if attempt.exists() and any(attempt.iterdir()):
        raise SystemExit(f"refusing to mix a new attempt into nonempty directory: {attempt}")
    attempt.mkdir(parents=True, exist_ok=True)

    before_tree = working_tree()
    files = source_files()
    before_sources, before_errors = source_records_with_errors(files)
    snapshot = attempt / "source-snapshot.zip"
    try:
        snapshot_digest = source_archive(snapshot, files)
    except FileNotFoundError as error:
        before_errors.append(
            {"path": Path(error.filename or "unknown").as_posix(), "error": "missing-during-archive"}
        )
        snapshot_digest = None
    verify_sources, verify_errors = source_records_with_errors(source_files())
    archived_sources = archived_source_records(snapshot) if snapshot_digest else []
    (attempt / "git-before.json").write_text(json.dumps(before_tree, indent=2), encoding="utf-8")

    # A source snapshot that changed while being captured cannot identify the
    # attempted binary. Preserve the preflight evidence, but never launch GPU.
    if (
        before_errors
        or verify_errors
        or before_sources != verify_sources
        or before_sources != archived_sources
    ):
        manifest: dict[str, Any] = {
            "schemaVersion": 1,
            "workload": args.workload,
            "backend": args.backend,
            "command": None,
            "runnerExitStatus": None,
            "sourceStable": False,
            "gpuStarted": False,
            "sourceInputs": before_sources,
            "archivedSourceInputs": archived_sources,
            "sourceErrors": {"before": before_errors, "verify": verify_errors},
            "sourceSnapshot": {"path": snapshot.name, "sha256": snapshot_digest, "compression": "stored"},
            "workingTree": {"before": "git-before.json", "head": before_tree["head"]},
            "artifacts": artifact_records(attempt),
        }
        (attempt / "attempt.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("source changed while capturing preflight snapshot; GPU workload was not started", file=sys.stderr)
        return 2

    feature = "dx12" if args.backend == "dx12" else "native-gl-wgl"
    selector = WORKLOAD_SELECTORS[args.workload]
    build_command = [
        "cargo",
        "build",
        "--manifest-path",
        str(manifest_path.relative_to(ROOT)),
        "--locked",
        "--features",
        feature,
        "--message-format=json-render-diagnostics",
    ]
    build = subprocess.run(
        build_command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    (attempt / "build-stdout.log").write_text(build.stdout, encoding="utf-8")
    (attempt / "build-stderr.log").write_text(build.stderr, encoding="utf-8")
    commands: list[dict[str, Any]] = [
        {"kind": "build", **command_record(build_command, build, "build-stdout.log", "build-stderr.log")}
    ]
    executable_evidence: dict[str, str] | None = None
    run: subprocess.CompletedProcess[str] | None = None
    command: list[str] | None = None
    run_error: str | None = None
    after_build_sources, after_build_errors = source_records_with_errors(source_files())
    build_source_stable = not after_build_errors and before_sources == after_build_sources
    if build.returncode == 0 and build_source_stable:
        cargo_executable = find_built_executable(build.stdout, manifest_path)
        if cargo_executable is None:
            run_error = "Cargo did not report an executable for the selected manifest"
        else:
            try:
                executable_evidence = retain_executable(cargo_executable, attempt)
            except (OSError, ValueError) as error:
                run_error = str(error)
        if executable_evidence is not None:
            command = [
                str(attempt / executable_evidence["path"]),
                selector,
                *workload_args,
                "--output",
                str(attempt / "workload"),
            ]
            runtime_environment = {k: v for k, v in os.environ.items() if not k.startswith("FLUXEL_")}
            controlled_environment = {"FLUXEL_DX12_DEBUG": "1"} if args.validate and args.backend == "dx12" else {}
            runtime_environment.update(controlled_environment)
            run = subprocess.run(command, cwd=ROOT, env=runtime_environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (attempt / "stdout.log").write_text(run.stdout, encoding="utf-8")
            (attempt / "stderr.log").write_text(run.stderr, encoding="utf-8")
            commands.append({"kind": "directExecutable", "controlledEnvironment": controlled_environment, **command_record(command, run, "stdout.log", "stderr.log")})
            executable_evidence["afterRunSha256"] = sha256(attempt / executable_evidence["path"])
            if executable_evidence["afterRunSha256"] != executable_evidence["sha256"]:
                run_error = "retained executable changed during its run"
    elif build.returncode == 0:
        run_error = "source changed during build; retained executable was not run"

    if run is None:
        (attempt / "stdout.log").write_text("", encoding="utf-8")
        (attempt / "stderr.log").write_text((run_error or "build failed") + "\n", encoding="utf-8")
        commands.append({
            "kind": "directExecutable",
            "command": [],
            "exitStatus": None,
            "stdout": "stdout.log",
            "stderr": "stderr.log",
            "runnerError": run_error or "build failed",
        })

    after_tree = working_tree()
    after_files = source_files()
    after_sources, after_errors = source_records_with_errors(after_files)
    stable = not after_errors and before_sources == after_sources
    (attempt / "git-after.json").write_text(json.dumps(after_tree, indent=2), encoding="utf-8")
    validation = inspect_diagnostics(attempt, args.backend, args.validate)
    manifest: dict[str, Any] = {
        "schemaVersion": 2,
        "workload": args.workload,
        "backend": args.backend,
        "validation": validation,
        "command": command,
        "runnerExitStatus": run.returncode if run is not None else (build.returncode or 1),
        "sourceStable": stable,
        "gpuStarted": run is not None,
        "commands": commands,
        "executable": executable_evidence,
        "sourceInputs": before_sources,
        "sourceErrors": {
            "before": before_errors,
            "afterBuild": after_build_errors,
            "after": after_errors,
        },
        "sourceSnapshot": {
            "path": snapshot.name,
            "sha256": snapshot_digest,
            "compression": "stored",
        },
        "locks": [
            {
                "path": lock_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(lock_path),
                "role": "actual cargo --locked workspace lock",
            },
            {
                "path": "Cargo.lock",
                "sha256": sha256(ROOT / "Cargo.lock"),
                "role": "repository context only",
            },
        ],
        "resolvedGitPackages": resolved_git_packages(lock_path),
        "workingTree": {
            "before": "git-before.json",
            "after": "git-after.json",
            "head": before_tree["head"],
        },
        "artifacts": artifact_records(attempt),
    }
    (attempt / "attempt.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if build.returncode:
        return build.returncode
    if run is None or run_error is not None:
        return 2
    if run.returncode:
        return run.returncode
    if not stable:
        print("source changed during attempt; manifest retained but attempt is not source-stable", file=sys.stderr)
        return 2
    if args.validate and validation["status"] != "observed":
        print("native diagnostic activation or clean-log validation failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
