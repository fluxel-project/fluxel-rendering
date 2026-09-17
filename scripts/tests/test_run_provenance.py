from __future__ import annotations

from datetime import datetime
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPTS))

SCRIPT = SCRIPTS / "run_provenance.py"
SPEC = importlib.util.spec_from_file_location("run_provenance", SCRIPT)
assert SPEC and SPEC.loader
PROVENANCE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROVENANCE
SPEC.loader.exec_module(PROVENANCE)


def git(directory: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=str(directory), capture_output=True, text=True, check=True
    )
    return completed.stdout


def scratch_checkout(directory: Path) -> Path:
    """A one-commit repository, so the identity of a run can be arranged rather than found.

    Made here rather than read off this repository because the property under
    test is what the module does with a *dirty* tree, and the one thing this
    repository cannot be relied on for is being in a particular state while the
    suite runs.
    """
    git(directory, "init", "--quiet")
    git(
        directory,
        "-c", "user.name=provenance test",
        "-c", "user.email=test@example.invalid",
        "commit", "--quiet", "--allow-empty", "-m", "the revision a report would name",
    )
    return directory


class CommitTests(unittest.TestCase):
    def test_the_commit_is_the_forty_character_sha_git_reports(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            repo = scratch_checkout(Path(scratch))
            expected = git(repo, "rev-parse", "HEAD").strip()
            self.assertEqual(PROVENANCE.commit_of(repo), expected)
            self.assertRegex(expected, r"^[0-9a-f]{40}$")

    def test_a_directory_that_is_not_a_checkout_is_refused_rather_than_guessed(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            with self.assertRaises(PROVENANCE.ProvenanceError):
                PROVENANCE.commit_of(Path(scratch))


class ChangesTests(unittest.TestCase):
    def test_a_clean_checkout_has_nothing_to_report(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            repo = scratch_checkout(Path(scratch))
            self.assertEqual(PROVENANCE.changes_in(repo), [])

    def test_an_untracked_file_makes_the_checkout_dirty(self) -> None:
        # The reason untracked files are counted at all: a fixture reads what is
        # on disk, so a file git has never seen is enough to make the revision a
        # report names different from the revision that ran.
        with tempfile.TemporaryDirectory() as scratch:
            repo = scratch_checkout(Path(scratch))
            (repo / "a-file-the-fixture-read").write_text("content\n", encoding="utf-8")
            changes = PROVENANCE.changes_in(repo)
            self.assertEqual(len(changes), 1)
            self.assertIn("a-file-the-fixture-read", changes[0])

    def test_a_modified_tracked_file_makes_the_checkout_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            repo = scratch_checkout(Path(scratch))
            tracked = repo / "tracked.txt"
            tracked.write_text("first\n", encoding="utf-8")
            git(repo, "add", "tracked.txt")
            git(
                repo,
                "-c", "user.name=provenance test",
                "-c", "user.email=test@example.invalid",
                "commit", "--quiet", "-m", "add a tracked file",
            )
            self.assertEqual(PROVENANCE.changes_in(repo), [])
            tracked.write_text("second\n", encoding="utf-8")
            self.assertEqual(len(PROVENANCE.changes_in(repo)), 1)


class RequireCleanTests(unittest.TestCase):
    def test_a_clean_checkout_yields_its_commit(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            repo = scratch_checkout(Path(scratch))
            self.assertEqual(PROVENANCE.require_clean(repo), PROVENANCE.commit_of(repo))

    def test_a_dirty_checkout_is_refused_and_names_what_is_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            repo = scratch_checkout(Path(scratch))
            (repo / "leftover.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            with self.assertRaises(PROVENANCE.ProvenanceError) as caught:
                PROVENANCE.require_clean(repo)
            self.assertIn("leftover.png", str(caught.exception))


class RecordTests(unittest.TestCase):
    def test_the_record_carries_the_commit_the_toolchain_and_the_clock(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            repo = scratch_checkout(Path(scratch))
            record = PROVENANCE.record(repo)
            self.assertEqual(record["commit"], PROVENANCE.commit_of(repo))
            self.assertEqual(record["dirty"], [])
            self.assertTrue(record["toolchain"].startswith("python "))
            # Parsed rather than compared to a pattern: a timestamp nothing can
            # read back is not a timestamp, and the offset is what makes it one.
            stamped = datetime.fromisoformat(record["recorded_at_utc"])
            self.assertIsNotNone(stamped.tzinfo)

    def test_a_dirty_checkout_is_still_recorded_and_says_so(self) -> None:
        # The field, not a refusal: a screening against work in progress is still
        # a measurement of that work, and the artifact has to say which revision
        # it is evidence for.
        with tempfile.TemporaryDirectory() as scratch:
            repo = scratch_checkout(Path(scratch))
            (repo / "uncommitted.txt").write_text("work in progress\n", encoding="utf-8")
            record = PROVENANCE.record(repo)
            self.assertEqual(len(record["dirty"]), 1)
            self.assertIn("uncommitted.txt", record["dirty"][0])


if __name__ == "__main__":
    unittest.main()
