# Renderer unit tests

Unit test implementations live here, grouped by the component they exercise:

- `graph/`: bindings, compilation, and fixed fixture lowering.
- `renderer/`: retirement facts and submission ownership/lifecycle.

Production modules load their test files with `#[cfg(test)]` and `#[path]`
declarations. The tests remain logical children of their owning module, preserving
private access without widening the library's visibility. Public API integration
tests belong in the crate-root `tests/` directory.
