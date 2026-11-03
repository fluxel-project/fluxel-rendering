# Renderer unit tests

Unit test implementations live here, grouped by the component they exercise:

- `graph/`: bindings, compilation, and fixed fixture lowering.
- `renderer/`: retirement facts and submission ownership/lifecycle.
- `render_scene/`: transactional scene updates, generational IDs, immutable
  snapshots, dense removal/compaction, dirty ranges, and CPU-only concurrent
  snapshot reads.
- `feature/`: renderer-issued feature handles and contexts, issuer-bound typed
  storage, snapshot/version validation, CPU phase collection, and structured
  cross-scene, cross-issuer, and replay refusal.

Production modules load their test files with `#[cfg(test)]` and `#[path]`
declarations. The tests remain logical children of their owning module, preserving
private access without widening the library's visibility. Public API integration
tests belong in the crate-root `tests/` directory. These tests are logic evidence;
they do not exercise Graph execution or GPU behavior, and do not establish the
bounded public CPU facade is logic evidence only: it does not exercise Graph
execution, GPU behavior, or a future FramePipeline contract.
