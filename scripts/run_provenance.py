#!/usr/bin/env python3
"""The identity of the run that produced an artifact.

Why this is a module and not a few lines in each report writer
--------------------------------------------------------------
Two gates in this directory write a JSON artifact that a later reader treats as
evidence -- the state-cache screening and the GL4 raster readback -- and a third
(``stage1_evidence.py``) refuses to capture at all unless the checkout it is
about to name is the checkout that ran.  All three have to answer the same three
questions about the artifact: which revision is this, is that revision what
actually ran, and what ran it.  Answering them separately in each script is how
they drift -- one records the commit and not the toolchain, another records both
and not the dirt -- and a reader then cannot tell a report that was clean from a
report that never looked.

The ``dirty`` field is the one worth stating on purpose.  Evidence attributed to
a SHA that does not match the working tree is evidence for a revision that does
not exist, and a report that omitted the difference would be read as though
there were none.  So the field is always written, and only a caller that cannot
produce *anything* useful from a dirty tree asks for it to be refused.

This says nothing about hardware.  A commit and a toolchain do not tell a reader
which GPU produced a frame or whether the frame is right; those are the gates'
own business, and this module exists so that the part which is the same
everywhere is stated once.

Standard library only, and no third-party runtime: every artifact here is in the
release path (``CLAUDE.md`` §4).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import platform
import re
import subprocess


class ProvenanceError(RuntimeError):
    """A run whose identity could not be established."""


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ProvenanceError(
            f"git {' '.join(arguments)} failed in {repo}: "
            f"{completed.stderr.strip()[-200:]}"
        )
    return completed.stdout


def commit_of(repo: Path) -> str:
    """HEAD, as the 40-character SHA every artifact in this repo is keyed by."""
    commit = _git(repo, "rev-parse", "HEAD").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ProvenanceError(f"HEAD resolved to {commit!r}, not a 40-character SHA")
    return commit


def changes_in(repo: Path) -> list[str]:
    """Every change in the checkout, as ``git status --porcelain`` states them.

    Untracked files are included, because one is enough to make the revision a
    report names different from the revision that ran: a check that counted only
    tracked modifications would call a checkout clean while a fixture read a
    file git has never seen.  This is the same rule ``conformance.ps1`` applies
    before it will record a GPU run, stated once here for the reports that
    cannot enforce it but must still not hide it.
    """
    return [
        line
        for line in _git(repo, "status", "--porcelain", "--untracked-files=all").splitlines()
        if line.strip()
    ]


def toolchain() -> str:
    """What ran the artifact, named the way its own version command names it.

    The Python version rather than the Rust one on purpose: these are the
    scripts' own reports, and the Rust toolchain a *fixture* was built with is
    the fixture's answer, recorded by the gate that drove it.  Naming the
    interpreter here is what makes a report reproducible rather than merely
    re-readable.
    """
    return f"python {platform.python_version()} on {platform.system()}"


def record(repo: Path) -> dict:
    """The identity of a run, in the shape an artifact's ``provenance`` block takes.

    Written whether or not the tree is clean: the point is that a reader can see
    which revision the artifact is evidence for, and a dirty tree is a fact about
    the artifact rather than a reason to refuse one -- a screening run against a
    work in progress is still a measurement of that work.
    """
    return {
        "commit": commit_of(repo),
        "dirty": changes_in(repo),
        "toolchain": toolchain(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def require_clean(repo: Path) -> str:
    """The commit, given that the checkout is exactly the revision it names.

    For the captures whose artifact cannot be re-derived from a SHA -- a frame's
    pixels, a screenshot -- and whose whole claim is that they are what that
    revision produced.  Nothing about those artifacts can be recomputed later,
    so the checkout state at capture time is the only thing that ties them to a
    revision, and a dirty one leaves them tied to nothing.
    """
    commit = commit_of(repo)
    changes = changes_in(repo)
    if changes:
        raise ProvenanceError(
            "the checkout is not the revision this artifact would name:\n  "
            + "\n  ".join(changes)
        )
    return commit
