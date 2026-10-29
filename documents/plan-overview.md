# Fluxel Rendering Product Plan

> Status: active target direction. The organization [roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md)
> owns ecosystem order; [version-plan.md](version-plan.md) supplies implementation
> gates. Target architecture is not a claim of completed implementation.

## Product direction

Fluxel provides a complete, extensible, data-oriented real-time renderer without
requiring an ECS, game scene model, host, or particular rendering strategy.
Renderer owns views/camera data, visibility, draw preparation, sorting, batching,
GPU residency coordination, and frame execution.

The target workspace has three principal crates: `fluxel-shader`,
`fluxel-material`, and `fluxel-renderer`. RenderGraph and the replaceable
`FramePipeline` SPI are renderer modules. Shader provides reusable compilation
infrastructure; Material owns material semantics and consumes Shader. Neither
depends on Renderer. See [workspace architecture](design-overview.md).

`fluxel-assets` remains in `fluxel-bases`, retaining logical identity, immutable
content generations, producer coordination, and CPU cache policy. Callers own
loading, decoding, hot reload, and streaming policy; platform services supply
I/O. Renderer realizes asset snapshots per device and retains GPU resources
through completion. This plan creates no independent Assets project.

## Delivery order

1. **Reuse and integration baseline.** Reuse the existing Bases asset core and
   select exact stable RHI dependencies; identify migration gaps.
2. **Renderer graph foundation.** Internalize Graph and prove direct RHI
   execution, reusable plans, overlapping frames, and safe resource retirement.
3. **Material semantics and Shader core.** Define material/fragment contracts
   and build the independent Shader crate using WGSL/Naga, reflection,
   artifacts, variant identity, and caching.
4. **Material and Renderer MVP.** Render a real textured 3D mesh through
   Material, Shader, a minimal Forward pipeline, internal Graph, and RHI.
   Include batch incremental scene updates and a minimal external Rust Feature.
5. **Material and pipeline expansion.** Grow code nodes, PBR, Forward+, shadows,
   PostFX, and selected Features. Prove Deferred over shared scene/material
   contracts as a second pipeline.
6. **Performance and resource strategy.** Measure representative workloads,
   residency, memory stability, uploads, and frames in flight. Select streaming,
   LOD, GPU culling, indirect work, and multi-queue optimization by actual need.

Each early milestone delivers a runnable vertical slice. Advanced algorithms
are not all prerequisites for the MVP. The separate Graph crate,
fixed renderer recipes, and WASM capsule are migration inputs, not the target
three-crate structure.

## Undecided

Capture/replay, Canvas 2D and basic text, declarative UI, a Blender plugin,
and JavaScript wrappers are undecided. They have no delivery order or retained
design in this plan.

## MVP acceptance

```text
Bases snapshots + batched scene changes
  -> RenderScene / RenderView + built-in Mesh + independent Rust Feature
  -> visibility, phase sorting, batching, material/shader preparation
  -> replaceable FramePipeline -> renderer-internal RenderGraph
  -> stable RHI -> observed textured 3D output
  -> completion-based reuse and retirement
```

The external Feature owns typed data, accepts batch updates, participates in a
pipeline phase, and declares all GPU resource accesses to Graph. Adding it must
not require a core object-type switch or separate GPU submissions. A small
fixture suffices; complete terrain, water, particles, and skinning are later
workloads. ECS-style and ordinary adapters share the batch boundary without an
ECS dependency. Stabilize extension contracts only after the external proof.

Each milestone needs structured refusal cases, exact participating revisions
and lock inputs, focused tests, and relevant real-target integration evidence.
Stable RHI does not certify every optional operation or backend: its
[open work](https://github.com/fluxel-project/fluxel-rhi/blob/main/TODO.md) and
[verification limits](https://github.com/fluxel-project/fluxel-rhi/blob/main/documents/verification.md)
remain explicit. One crate's green tests do not close an ecosystem milestone.
