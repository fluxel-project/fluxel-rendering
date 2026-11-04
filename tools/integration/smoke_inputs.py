#!/usr/bin/env python3
"""Auditable source-closure and environment helpers for smoke evidence."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib
import zipfile
from typing import Any
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "examples" / "rendering-integration"
EXCLUDED_SOURCE_PARTS = {".git", "target", "draft"}
SENSITIVE_SOURCE_NAMES = {".env", ".env.local", ".envrc"}
SENSITIVE_SOURCE_SUFFIXES = {".key", ".pem", ".p12", ".pfx", ".token"}
SENSITIVE_CONFIG_ASSIGNMENT = re.compile(r"(?im)^\s*(?:token|password|secret|api[-_]?key)\s*=")
RUST_INCLUDE = re.compile(r"include_(?:str|bytes)!\s*\(\s*([^)]*?)\s*\)")
CHILD_ENV_ALLOWLIST = (
    "APPDATA", "COMSPEC", "CommonProgramFiles", "CommonProgramFiles(x86)",
    "INCLUDE", "LIB", "LIBPATH", "LOCALAPPDATA", "NUMBER_OF_PROCESSORS", "OS",
    "PATH", "PATHEXT", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER", "ProgramData",
    "ProgramFiles", "ProgramFiles(x86)", "RUSTUP_HOME", "RUSTUP_TOOLCHAIN", "SystemRoot",
    "TEMP", "TMP", "UCRTVersion", "UniversalCRTSdkDir", "USERPROFILE", "VCINSTALLDIR",
    "VCToolsInstallDir", "VSINSTALLDIR", "WindowsSdkDir", "WindowsSDKVersion", "WINDIR",
    "CARGO_HOME",
)
def cargo_config_has_secret(path: Path) -> bool:
    try:
        return bool(SENSITIVE_CONFIG_ASSIGNMENT.search(path.read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return True


def source_path_is_safe(path: Path, root: Path) -> bool:
    """Keep snapshots to reviewable build inputs, never local credentials."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return (path.is_file() and not any(part in EXCLUDED_SOURCE_PARTS for part in relative.parts)
            and path.name != "AGENTS.md" and path.name not in SENSITIVE_SOURCE_NAMES
            and not path.name.startswith("credentials") and path.suffix.lower() not in SENSITIVE_SOURCE_SUFFIXES)


def sensitive_build_inputs(root: Path) -> int:
    """Fail closed rather than archive an unreviewable probable secret."""
    artifact = root / "examples" / "rendering-integration"
    external = root / "examples" / "external-renderer"
    count = 0
    for directory in (root / "crates/renderer", root / "crates/shader", root / "crates/material", artifact, external,
                      root / ".cargo", root / "examples/.cargo", artifact / ".cargo"):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            try:
                relative = path.resolve().relative_to(root.resolve())
            except ValueError:
                continue
            config_secret = path.name in {"config", "config.toml"} and cargo_config_has_secret(path)
            if (path.is_file() and not any(part in EXCLUDED_SOURCE_PARTS for part in relative.parts)
                    and (path.name in SENSITIVE_SOURCE_NAMES or path.name.startswith("credentials")
                         or path.suffix.lower() in SENSITIVE_SOURCE_SUFFIXES or config_secret)):
                count += 1
    return count


def closure_escape_count(root: Path) -> int:
    """Reject links and include macros whose exact input cannot be archived."""
    artifact = root / "examples" / "rendering-integration"
    external = root / "examples" / "external-renderer"
    directories = (root / "crates/renderer", root / "crates/shader", root / "crates/material", artifact, external)
    failures = 0
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if any(part in EXCLUDED_SOURCE_PARTS for part in relative.parts):
                continue
            is_junction = getattr(path, "is_junction", lambda: False)()
            if (path.is_symlink() or is_junction) and not source_path_is_safe(path, root):
                failures += 1
                continue
            if (".cargo" in relative.parts or path.name == "AGENTS.md"
                    or not source_path_is_safe(path, root)):
                continue
            if path.suffix != ".rs" or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                failures += 1
                continue
            for expression in RUST_INCLUDE.findall(text):
                literal = expression.strip()
                if not ((literal.startswith('"') and literal.endswith('"'))
                        or (literal.startswith('r"') and literal.endswith('"'))):
                    failures += 1
                    continue
                target = path.parent / literal[literal.find('"') + 1:-1]
                if not source_path_is_safe(target, root):
                    failures += 1
    return failures


def uncontrolled_build_script_count(root: Path) -> int:
    """A build script can read arbitrary process state; no full env closure exists here."""
    artifact = root / "examples" / "rendering-integration"
    external = root / "examples" / "external-renderer"
    return sum((directory / "build.rs").is_file() for directory in (
        root, root / "crates/renderer", root / "crates/shader", root / "crates/material", artifact, external))


def source_tree_paths(directory: Path, root: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [path for path in directory.rglob("*") if source_path_is_safe(path, root)
            and ".cargo" not in path.relative_to(root).parts]


def cargo_config_paths(directory: Path, root: Path) -> list[Path]:
    """Cargo reads only config/config.toml here; credentials are never evidence."""
    config = directory / ".cargo"
    return [path for name in ("config", "config.toml")
            if source_path_is_safe(path := config / name, root) and not cargo_config_has_secret(path)]


def external_cargo_config_count(root: Path) -> int:
    """Cargo searches from the consumer cwd to the filesystem root."""
    boundary = root.resolve()
    directory = boundary.parent
    count = 0
    while True:
        count += sum((directory / ".cargo" / name).is_file() for name in ("config", "config.toml"))
        if directory.parent == directory:
            return count
        directory = directory.parent


def source_input_paths(root: Path = ROOT) -> list[Path]:
    """The complete, non-secret build-input closure for this bounded consumer."""
    artifact = root / "examples" / "rendering-integration"
    external = root / "examples" / "external-renderer"
    paths: list[Path] = [
        root / "Cargo.toml", root / "Cargo.lock", root / "build.rs",
        root / "tools/integration/README.md",
        root / "documents/s0-s1-support-decision.md",
        root / "documents/integration-known-diagnostics.md",
        root / "documents/integration-manifest.schema.json",
    ]
    for package in (root / "crates/renderer", root / "crates/shader",
                    root / "crates/material", artifact, external):
        paths.extend(source_tree_paths(package, root))
    tools = root / "tools/integration"
    paths.extend(path for path in tools.rglob("*.py") if source_path_is_safe(path, root))
    # Cargo is invoked from the consumer. Config in each ancestor on that route
    # can alter the build; only Cargo's two configuration filenames are inputs.
    for directory in (root, root / "examples", artifact, external):
        paths.extend(cargo_config_paths(directory, root))
    resolved_root = root.resolve()
    return sorted({path.resolve() for path in paths if source_path_is_safe(path, root)},
                  key=lambda path: path.relative_to(resolved_root).as_posix())


def build_environment(root: Path = ROOT) -> dict[str, Any]:
    """Record override presence safely and reject config that cannot be archived."""
    build_override_names = ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER",
                            "RUSTC_WORKSPACE_WRAPPER", "RUSTC", "CARGO_BUILD_TARGET",
                            "CARGO_BUILD_RUSTFLAGS", "CC", "CFLAGS", "CXX", "CXXFLAGS")
    names = (*build_override_names, "CARGO_HOME",
             *sorted(name for name in os.environ if name.startswith("CARGO_TARGET_") and name.endswith("_RUSTFLAGS")))
    overrides = {}
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            overrides[name] = {"present": True, "bytes": len(value.encode("utf-8")),
                               "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()}
    reasons = [f"unarchived build override is set: {name}"
               for name in names if name != "CARGO_HOME" and name in overrides]
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    for name in ("config", "config.toml"):
        if (cargo_home / name).is_file():
            reasons.append("effective Cargo home configuration is outside the archived source closure")
            break
    sensitive = sensitive_build_inputs(root)
    if sensitive:
        reasons.append(f"{sensitive} probable secret build input(s) cannot be archived safely")
    escapes = closure_escape_count(root)
    if escapes:
        reasons.append(f"{escapes} linked or included build input(s) cannot be proven safe for the archive")
    build_scripts = uncontrolled_build_script_count(root)
    if build_scripts:
        reasons.append(f"{build_scripts} build script(s) can read unclosed process state")
    external_configs = external_cargo_config_count(root)
    if external_configs:
        reasons.append(f"{external_configs} Cargo configuration file(s) outside the archived source closure")
    return {"overrides": overrides, "status": "rejected" if reasons else "auditable",
            "reasons": reasons}


def controlled_child_environment(extra: dict[str, str] | None = None) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Pass only toolchain essentials; never inherit arbitrary process secrets."""
    environment = {name: os.environ[name] for name in CHILD_ENV_ALLOWLIST if name in os.environ}
    if extra:
        environment.update(extra)
    evidence = {name: {"bytes": len(value.encode("utf-8")),
                       "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()}
                for name, value in sorted(environment.items())}
    return environment, evidence


def classify_dx12_diagnostics(stderr: Path) -> dict[str, Any]:
    """Retain every native diagnostic; the rebuilt copy workload has no waivers."""
    lines = stderr.read_text(encoding="utf-8", errors="replace").splitlines()
    native = [line for line in lines if line.startswith("D3D12 [")]
    return {
        "native_callback_observed": bool(native),
        "unreviewed_native_diagnostics": native,
        "policy": "tools/integration/README.md",
        "policy_sha256": sha256(ROOT / "tools/integration/README.md"),
    }


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(directory: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(directory), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_consistency(inputs: list[dict[str, Any]], root: Path = ROOT) -> dict[str, Any]:
    """Detect edits during an attempt instead of attributing a mixed build."""
    before = {str(Path(item["path"])): item["sha256"] for item in inputs}
    resolved_root = root.resolve()
    candidates = {str(path.relative_to(resolved_root)) for path in source_input_paths(root)} | set(before)
    source_directories = {
        (Path(name).parent, Path(name).suffix)
        for name in before if Path(name).suffix in {".rs", ".wgsl", ".py"}
    }
    for directory, suffix in source_directories:
        candidates.update(str(path.resolve().relative_to(resolved_root)) for path in (root / directory).rglob("*" + suffix)
                          if source_path_is_safe(path, root))
    changed = []
    for name in sorted(candidates):
        after = sha256(root / name)
        if after != before.get(name):
            changed.append({"path": name, "before_sha256": before.get(name), "after_sha256": after})
    return {"status": "changed" if changed else "stable", "changed_inputs": changed}


def checkout_inventory(directory: Path) -> dict[str, Any]:
    status = git_value(directory, "status", "--porcelain")
    return {
        "path": str(directory),
        "revision": git_value(directory, "rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "role": "local_checkout_inventory",
    }
