from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "check_lockfile_versions.py"
SPEC = importlib.util.spec_from_file_location("check_lockfile_versions", SCRIPT)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


class CheckLockfileVersionsTests(unittest.TestCase):
    def write(self, root: Path, relative: str, text: str) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def workspace(self, root: Path, version: str = "0.15.0") -> None:
        self.write(root, "Cargo.toml", f'[workspace.package]\nversion = "{version}"\n')
        self.write(
            root,
            "crates/rhi/Cargo.toml",
            '[package]\nname = "fluxel-rhi"\nversion.workspace = true\n',
        )

    def lock(self, root: Path, relative: str, name: str, version: str, source: bool = False) -> None:
        origin = 'source = "git+https://example.invalid/x#abc"\n' if source else ""
        self.write(
            root,
            relative,
            f'[[package]]\nname = "{name}"\nversion = "{version}"\n{origin}',
        )

    def test_an_empty_tree_has_nothing_to_disagree_about(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(CHECKER.check(Path(directory)), [])

    def test_a_matching_lock_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.workspace(root)
            self.lock(root, "examples/harness/Cargo.lock", "fluxel-rhi", "0.15.0")
            self.assertEqual(CHECKER.check(root), [])

    def test_a_stale_path_dependency_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.workspace(root)
            self.lock(root, "examples/harness/Cargo.lock", "fluxel-rhi", "0.14.0")
            observed = CHECKER.check(root)
            self.assertEqual(len(observed), 1)
            self.assertEqual(
                (observed[0].package, observed[0].locked, observed[0].declared),
                ("fluxel-rhi", "0.14.0", "0.15.0"),
            )

    def test_an_inherited_version_is_resolved_not_read_as_a_literal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.workspace(root, version = "2.1.0")
            self.lock(root, "examples/harness/Cargo.lock", "fluxel-rhi", "2.1.0")
            self.assertEqual(CHECKER.check(root), [])

    def test_a_registry_or_git_version_is_not_this_trees_business(self) -> None:
        # `fluxel-host` arrives from its own repository at whatever revision the
        # consumer pinned; a lock that names a git source is evidence about that
        # repository and must not be read as a drifted path dependency.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.workspace(root)
            self.write(root, "crates/host/Cargo.toml", '[package]\nname = "fluxel-host"\nversion = "9.9.9"\n')
            self.lock(root, "examples/harness/Cargo.lock", "fluxel-host", "1.0.0", source = True)
            self.assertEqual(CHECKER.check(root), [])

    def test_each_lock_file_is_checked_independently(self) -> None:
        # The defect this exists for was three separate projects going stale at
        # once, so one clean lock must not excuse another.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.workspace(root)
            self.lock(root, "examples/a/Cargo.lock", "fluxel-rhi", "0.15.0")
            self.lock(root, "examples/b/Cargo.lock", "fluxel-rhi", "0.14.0")
            observed = CHECKER.check(root)
            self.assertEqual(
                [item.lock.relative_to(root).as_posix() for item in observed],
                ["examples/b/Cargo.lock"],
            )

    def test_a_name_with_no_manifest_in_the_tree_is_skipped(self) -> None:
        # A path dependency pointing outside the repository has no manifest here
        # to compare against, and guessing one would be worse than saying nothing.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.workspace(root)
            self.lock(root, "examples/harness/Cargo.lock", "fluxel-elsewhere", "0.1.0")
            self.assertEqual(CHECKER.check(root), [])

    def test_a_literal_version_manifest_is_compared_directly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.workspace(root)
            self.write(root, "crates/hal/Cargo.toml", '[package]\nname = "fluxel-wgpu-hal"\nversion = "30.0.1"\n')
            self.lock(root, "examples/harness/Cargo.lock", "fluxel-wgpu-hal", "30.0.1")
            self.assertEqual(CHECKER.check(root), [])
            self.lock(root, "examples/harness/Cargo.lock", "fluxel-wgpu-hal", "30.0.0")
            self.assertEqual(len(CHECKER.check(root)), 1)


if __name__ == "__main__":
    unittest.main()
