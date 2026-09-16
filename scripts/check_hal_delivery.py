#!/usr/bin/env python3
"""Independent-consumer check for the patched wgpu-hal delivery path.

`fluxel-rhi` needs patched wgpu-hal 30.0.1 (upstream gfx-rs/wgpu#10221). The
workspace once carried that source behind a root `[patch.crates-io]`, which
Cargo reads only from the *root* manifest of a build: every consumer that
depended on this crate through the documented Git dependency silently resolved
the unpatched crates.io package and ran different DX12 sampler code than the
repository's own CI. No in-workspace gate can observe that, because inside this
repository the root patch does apply.

This script reproduces the consumer's view instead. It builds a throwaway crate
*outside* the repository, declares exactly the dependency README documents for
an external user, and resolves it through a `file://` Git URL so `cargo` runs
its real Git dependency path (clone, rev checkout, path-dependency resolution
relative to the checked-out manifest) without needing a push. It then asserts:

  * a package named `fluxel-wgpu-hal` is in the resolve graph, sourced from the
    Git checkout under `crates/wgpu-hal` -- not from the registry;
  * no registry package named `wgpu-hal` is in the resolve graph at all, so the
    consumer cannot silently link the unpatched source.

Exits 0 and prints `PASS` only when both hold. Any other outcome is a failure,
including `cargo metadata` itself failing: a delivery path that cannot be
resolved by a consumer is not delivered.

Usage:
    python scripts/check_hal_delivery.py [--rev <sha>] [--keep]

`--rev` defaults to the repository's current HEAD. The rev must be committed;
an uncommitted working tree is not part of a Git dependency, which is exactly
what this check is about.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The one dependency line an external consumer is told to write. Kept as a
# literal so the check fails if the documented path and the checked path drift.
CONSUMER_MANIFEST = textwrap.dedent(
    """
    [package]
    name = "fluxel-hal-delivery-consumer"
    version = "0.0.0"
    edition = "2021"
    publish = false

    [workspace]

    [dependencies.fluxel-rhi]
    git = "{url}"
    rev = "{rev}"
    features = ["dx12"]
    """
).strip()


def git_url(repo: Path) -> str:
    """Return a `file://` URL for `repo`, in the spelling cargo accepts on this OS."""
    return repo.as_uri()


def run_metadata(consumer_dir: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["cargo", "metadata", "--format-version", "1"],
        cwd=consumer_dir,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def classify(packages: list[dict]) -> list[str]:
    """Return the failure reasons for one resolve graph; empty means delivered."""
    problems: list[str] = []
    patched = [p for p in packages if p["name"] == "fluxel-wgpu-hal"]
    registry_hal = [p for p in packages if p["name"] == "wgpu-hal"]

    if not patched:
        problems.append(
            "no package named `fluxel-wgpu-hal` in the consumer's resolve graph: "
            "the patched source is not delivered"
        )
    for package in patched:
        source = package.get("source") or ""
        manifest = Path(package["manifest_path"])
        if not source.startswith("git+"):
            problems.append(
                f"fluxel-wgpu-hal resolved from `{source or 'a path source'}` "
                "instead of the Git checkout"
            )
        if manifest.parent.name != "wgpu-hal" or manifest.parent.parent.name != "crates":
            problems.append(
                f"fluxel-wgpu-hal resolved to {manifest}, which is not "
                "`crates/wgpu-hal` inside the checkout"
            )
    if registry_hal:
        sources = ", ".join(sorted({(p.get("source") or "path") for p in registry_hal}))
        problems.append(
            f"the consumer also resolved an unpatched `wgpu-hal` package ({sources}); "
            "the DX12 sampler fix would not reach it"
        )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rev",
        default=None,
        help="commit to resolve (default: the repository's current HEAD)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the throwaway consumer directory for inspection",
    )
    args = parser.parse_args(argv)

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    rev = args.rev or head
    if args.rev is None:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if dirty:
            print(
                "note: the working tree has uncommitted changes; resolving HEAD "
                f"{head[:12]}, which is the commit a consumer would actually get",
                file=sys.stderr,
            )

    consumer_dir = Path(tempfile.mkdtemp(prefix="fluxel-hal-delivery-"))
    try:
        (consumer_dir / "src").mkdir()
        (consumer_dir / "src" / "lib.rs").write_text("", encoding="utf-8")
        (consumer_dir / "Cargo.toml").write_text(
            CONSUMER_MANIFEST.format(url=git_url(REPO_ROOT), rev=rev) + "\n",
            encoding="utf-8",
        )

        code, stdout, stderr = run_metadata(consumer_dir)
        if code != 0:
            print("FAIL: `cargo metadata` failed for the external consumer\n", file=sys.stderr)
            print(stderr.strip() or stdout.strip(), file=sys.stderr)
            return 1

        metadata = json.loads(stdout)
        problems = classify(metadata["packages"])

        resolved = next(
            (p for p in metadata["packages"] if p["name"] == "fluxel-wgpu-hal"), None
        )
        if resolved is not None:
            print(f"fluxel-wgpu-hal {resolved['version']}")
            print(f"  source:   {resolved.get('source')}")
            print(f"  manifest: {resolved['manifest_path']}")
        print(f"consumer rev: {rev}")

        if problems:
            print("\nFAIL", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1

        print("PASS: the patched wgpu-hal is delivered through the crate's own manifest")
        return 0
    finally:
        if args.keep:
            print(f"kept consumer at {consumer_dir}")
        else:
            shutil.rmtree(consumer_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
