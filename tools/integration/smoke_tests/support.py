"""One normal import path shared by all split smoke test modules."""

from __future__ import annotations

if __package__ and "." in __package__:
    # Package execution keeps the integration parent in the module path.
    from .. import smoke
else:
    # ``unittest discover -s tools/integration`` imports ``smoke_tests`` as a
    # top-level package, so there is no parent package to traverse.
    import smoke
