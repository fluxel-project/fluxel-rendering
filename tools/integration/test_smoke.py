"""Explicit aggregate for the split smoke runner tests."""
import unittest

if __package__:
    from .smoke_tests import cli, closure, evidence
else:
    from smoke_tests import cli, closure, evidence

def load_tests(loader: unittest.TestLoader, _tests: unittest.TestSuite, _pattern: str | None) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    for module in (cli, evidence, closure):
        suite.addTests(loader.loadTestsFromModule(module))
    return suite

if __name__ == "__main__":
    unittest.main()
