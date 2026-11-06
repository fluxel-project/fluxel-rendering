# Renderer unit tests

Unit tests live here, grouped by the production component they exercise:

- `graph/`: declarations, bindings, compilation, state, and fixed lowering.
- `renderer/`: submission, retirement, lifecycle, and structured failures.
- `render_scene/`: transactional updates, generational identities, snapshots,
  compaction, dirty ranges, and CPU-only concurrent reads.
- `feature/`: issuer-bound feature handles, typed storage, version checks, CPU
  phase collection, and foreign/replay refusal.
- `pipeline/`: controlled frame preparation and candidate provenance.
- `preparation/`: geometry streams and logical Shader-interface validation.

Production modules load these files with `#[cfg(test)]` and `#[path]`
declarations, preserving private access without widening library visibility.
Public API integration tests belong in the crate-root `tests/` directory. These
tests are logic evidence; CPU preparation tests do not replace runnable native
consumer validation.
