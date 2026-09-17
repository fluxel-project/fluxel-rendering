#!/usr/bin/env python3
"""Check that every lock file in the tree agrees with the manifests in it.

A ``Cargo.lock`` records the resolved version of each path dependency it names,
and nothing in a workspace-wide gate ever looks at the lock files belonging to
the projects *outside* the workspace.  This repository has three of them --
``examples/windows-gl4``, ``examples/windows-dx12`` and ``examples/android-gles``
-- and they are the harnesses the hardware gates run, so they are exactly the
projects a release is least able to do without and least likely to notice.

The failure this catches is a version bump that updates the workspace and the
three crates in it but not those three locks.  It is not a hypothetical: the
0.15 candidate shipped in that state, and it surfaced only when the first
hardware script tried to build its harness with ``--locked`` and Cargo refused
to resolve ``fluxel-rhi 0.15.0`` against a lock that still said ``0.14.0``.
Every row of the workspace gate passed, because a separate Cargo project is a
separate resolution and no row built one.

The rule is deliberately narrow.  Only entries with no ``source`` are checked,
which is what distinguishes a path dependency -- the kind this repository can
be wrong about -- from a registry or git one, whose version is whatever the
remote says and not this repository's business.  A name with no manifest in the
tree is skipped for the same reason: ``fluxel-host`` and ``fluxel-assets``
arrive from their own repositories, and an example lock is not evidence about
them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import tomllib


@dataclass(frozen=True)
class Mismatch:
    lock: Path
    package: str
    locked: str
    declared: str


def manifest_versions(root: Path) -> dict[str, tuple[str, Path]]:
    """Map every in-tree package name to the version its manifest declares.

    A workspace member inherits its version, so the inherited form is resolved
    against the root manifest rather than reported as a literal.  The value is
    the manifest's own path so a violation can name where the truth lives.
    """
    inherited = ""
    root_manifest = root / "Cargo.toml"
    if root_manifest.is_file():
        document = tomllib.loads(root_manifest.read_text(encoding="utf-8"))
        inherited = document.get("workspace", {}).get("package", {}).get("version", "")

    versions: dict[str, tuple[str, Path]] = {}
    for manifest in sorted(root.glob("**/Cargo.toml")):
        if "target" in manifest.parts:
            continue
        document = tomllib.loads(manifest.read_text(encoding="utf-8"))
        package = document.get("package")
        if not package or "name" not in package:
            continue
        declared = package.get("version")
        if isinstance(declared, dict) and declared.get("workspace") is True:
            declared = inherited
        if isinstance(declared, str):
            versions[package["name"]] = (declared, manifest)
    return versions


def check(root: Path) -> list[Mismatch]:
    versions = manifest_versions(root)
    mismatches: list[Mismatch] = []
    for lock in sorted(root.glob("**/Cargo.lock")):
        if "target" in lock.parts:
            continue
        document = tomllib.loads(lock.read_text(encoding="utf-8"))
        for entry in document.get("package", []):
            # No `source` means resolved from a path, which is the only kind of
            # dependency whose version this repository gets to decide.
            if "source" in entry or entry["name"] not in versions:
                continue
            declared = versions[entry["name"]][0]
            if entry.get("version") != declared:
                mismatches.append(
                    Mismatch(lock, entry["name"], str(entry.get("version")), declared)
                )
    return mismatches


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Rendering repository root (default: inferred from this script).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the successful check message.")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"Lock file check failed: --root is not a directory: {root}", file=sys.stderr)
        return 2
    mismatches = check(root)
    if mismatches:
        versions = manifest_versions(root)
        for mismatch in mismatches:
            lock = mismatch.lock.relative_to(root).as_posix()
            manifest = versions[mismatch.package][1].relative_to(root).as_posix()
            print(
                f"{lock}: {mismatch.package} is locked at {mismatch.locked} "
                f"but {manifest} declares {mismatch.declared}"
            )
        print(
            "A path dependency changed version and this project's lock file did not "
            "follow.  Run `cargo update --workspace --offline` in that project.",
            file=sys.stderr,
        )
        return 1
    if not args.quiet:
        locks = [lock for lock in root.glob("**/Cargo.lock") if "target" not in lock.parts]
        print(f"Lock file check passed ({len(locks)} lock files agree with their manifests).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
