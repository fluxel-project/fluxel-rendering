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
3. **Shader and Material core.** First prove minimal WGSL/Naga validation,
   ShaderArtifact production, and real RHI shader/pipeline creation. Then define
   material functions, parameter/uniform schemas, and composition IR for the
   first material. Material generates evaluation WGSL and converts its IR into
   Shader-owned `ShaderCompositionRequest`; Shader never consumes MaterialIr.
   Add reflection, variant identity, and caching around this actual consumer.
4. **Material and Renderer MVP.** Render a real textured 3D mesh through
   Material, Shader, a minimal Forward pipeline, internal Graph, and RHI.
   Explicit checkpoints establish scene/frame snapshots, the `FramePipeline`
   SPI and phase policy, Rust plugin contracts, and the integrated Forward
   path with batch updates and a minimal external Rust Feature. Verify that
   external consumer and an external FramePipeline implementation. Stabilize
   only the exercised extension subset.
5. **Material and pipeline expansion.** Grow code nodes, PBR, Forward+, shadows,
   PostFX, and selected Features. Prove Deferred over shared scene/material
   contracts as a second pipeline.
6. **Performance and resource strategy.** Measure representative workloads,
   residency, memory stability, uploads, and frames in flight. Select streaming,
   LOD, GPU culling, indirect work, and multi-queue optimization by actual need.

Items 1 and 2 form one foundation closure. After ownership and exact dependency
selection, independent Bases-reuse and Graph/RHI work can proceed in parallel;
their combined integration proof remains required before the next milestone.
Graph fixtures can use direct RHI resources and small prepared shaders without
waiting for the full Material/Shader system or an asset-loading framework.

The MVP proves pipeline replacement, typed batch-oriented RenderFeatures, and
the controlled material/shader seams needed by the selected feature. Expansion
consumes these verified contracts. Rust crate plugins are in scope; dynamic DLL
loading and a stable plugin binary ABI are unscheduled.

The renderer owns the FramePipeline module but exposes a public Rust extension
trait and controlled context; external implementation does not expose Graph
compiler internals. Cross-pass and shader-extension contracts wait for a second,
materially different Feature consumer. Full authoring, Code Nodes, and PBR do not
block the first textured output.

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
pipeline phase, and declares all frame GPU resource accesses to Graph. Adding it must
not require a core object-type switch or independent Feature submissions. A small
fixture suffices; complete terrain, water, particles, and skinning are later
workloads. ECS-style and ordinary adapters share the batch boundary without an
ECS dependency. Stabilize only the contracts covered by the external proof.

Graph declarations cover resources inside prepared material bindings, with
reflection-derived usage and preparation-resolved concrete resource ranges.
Residency transfers may use the renderer's upload coordinator outside Graph;
their producer dependencies, states, and completion enter Graph/RHI before any
consumer records. GPU skinning, simulation, culling, and other frame work remain
Graph passes. Cached plans separate stable definitions from per-frame bindings;
topology or incompatible usage changes trigger recompilation.

Each milestone needs structured refusal cases, exact participating revisions
and lock inputs, focused tests, and relevant real-target integration evidence.
Stable RHI does not certify every optional operation or backend: its
[open work](https://github.com/fluxel-project/fluxel-rhi/blob/main/TODO.md) and
[verification limits](https://github.com/fluxel-project/fluxel-rhi/blob/main/documents/verification.md)
remain explicit. One crate's green tests do not close an ecosystem milestone.

## Future engine runtime direction

The active plan is a rendering ecosystem plan. A future Rust + TypeScript game
engine needs separate ownership and planning for Game Scene, gameplay/script
runtime, the game loop, Rust/TS bindings, and script hot reload. This is a
long-term direction, not an approved implementation milestone or an MVP gate.

```text
TypeScript Gameplay / Game Scene
  -> Scene Adapter -> Batch ChangeSets -> Rust RenderScene
  -> FramePipeline -> RenderGraph -> RHI
```

Adapters own external identity mappings and submit batches; RenderScene keeps
generational RenderIds and frame snapshots. `fluxel-jsbridge` adapts browser and
mini-game platform lifecycle; it does not own a future TypeScript gameplay VM.
JavaScript wrapper scope remains undecided.
