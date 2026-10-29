# ADR-0024: Integrate Graph into a data-oriented extensible Renderer

**Status:** Accepted target; migration is pending.

## Decision

The target workspace has `fluxel-shader`, `fluxel-material`, and
`fluxel-renderer`. Material owns private semantics and emits generic Shader
contributions; Renderer/FramePipeline combines them with pass policy to form a
Shader-owned composition request. Shader owns composition, logical
validation/reflection, and source artifacts. The opened RHI device owns its
route, target ABI, and backend lowering.

RenderGraph and `FramePipeline` are renderer modules, not target crates.
Graph owns resource declarations, versions, dependencies, lifetimes, and
portable planning; RHI executes its portable result. `fluxel-assets` stays in
Bases. Renderer owns private per-device residency and separate frame-data upload
lifetimes, both governed by RHI completion evidence.

RenderScene uses typed dense storage, generational renderer IDs, batched updates,
and snapshots. Features operate on batches. `FramePipeline` selects policy and
phases; Features contribute declared Graph work and do not submit independently.

## MVP consequence

The first product closure is one real frame containing the built-in mesh path, a
custom material, an external Rust `RenderFeature`, and an external Rust
`FramePipeline`. The public trait/context facade is shaped against a small
external prototype during Forward work; Graph compiler, scheduler, allocator,
RHI submission, and backend objects remain private. A second materially
different Feature is required before wider cross-pass or shader-extension
contracts stabilize.

## Migration consequence

Existing standalone Graph, fixed renderer, and WASM capsule are migration inputs.
Their consumers migrate with exact revisions, locks, and an integration artifact.
Foundation verifies direct RHI execution, compatible plan reuse across overlapping
frames, and current partial/indeterminate outcome retention before later stages.
