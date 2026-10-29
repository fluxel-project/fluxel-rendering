# Fluxel Rendering Implementation Plan

> Status: active target plan, without release-number promises. The organization
> [roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md) owns
> order; this document supplies rendering scope and acceptance.
> Existing packages and fixed recipes do not establish target completion.

## Delivery model

```text
Stable RHI + existing Bases asset core
  -> renderer-internal Graph foundation and integration
  -> material semantics + independent Shader core
  -> Material + data-oriented Renderer MVP + external Feature proof
  -> material/pipeline/Feature expansion
  -> measured performance and resource strategies
```

Each milestone names an integration owner, exact producer/consumer revisions,
lock inputs, and normal, boundary, and failure cases. Reuse existing code where
it satisfies the contract. New dependencies, identity domains, schedulers, or
abstraction layers require a demonstrated consumer. Completion requires a real
integrated artifact, not independently green library tests.

## 0. Reuse and dependency baseline

Owners: Rendering integration; Bases and RHI own their respective contracts.

Keep `fluxel-assets` in Bases. Reuse `AssetId<K>`, immutable
`ContentGeneration`, snapshots, producer coordination, and CPU cache policy.
Callers retain lookup, decoding, loading, retry, hot reload, and streaming
policy; platform services supply I/O. Do not rebuild this core.

Select an exact stable RHI release/revision and matching features. In an
implementation change, migrate affected manifests, locks, examples, the WASM
capsule, and real consumers together. The current `0.16` migration dependency
does not prove compatibility with the now-stable independent RHI. Inventory
retained Graph/renderer implementation and evidence before moving it.

**Acceptance:** a reproducible manifest identifies actual Bases, Rendering,
RHI, and applicable Host/JSBridge revisions, builds the real producer with its
consumer, and proves snapshot identity/generation handling and minimal GPU
execution. Record untested routes. RHI maintenance and selected TODOs remain
in RHI; completing every optional capability is not an entry gate.

## 1. Renderer-internal Graph foundation

Owner: `fluxel-renderer`, internal `graph` module.

Move useful graph implementation, tests, and diagnostics into Renderer. Retire
the standalone `fluxel-rendergraph` package as consumers migrate; do not
maintain two engines. Preserve logical resources/versions, typed pass-local
access, definedness, hazards, deterministic DAG scheduling, transient lifetimes,
and diagnostics. Graph remains unaware of materials and rendering algorithms.

Use the current [RHI Core contract](https://github.com/fluxel-project/fluxel-rhi/blob/main/crates/core/DESIGN.md)
directly for device facts, descriptors, recording, submission, and completion.
Support selected deferred and GL immediate routes without another CPU command
framework or duplicate capability model. Dedicated transient allocation is the
baseline; aliasing and multi-queue optimization are follow-up choices.

**Acceptance:**

- Run a minimal graph on DX12, Vulkan, and a named GL-family route, retaining
  observed output and validation evidence for each claim. WebGL2 evidence
  does not certify desktop GL, GLES, or a mini-game host.
- Reuse a compiled plan; submit 2–3 live frames before waiting, retain distinct
  completion points, and prevent early reuse/retirement. Cover caller-handle
  drops and partial submission failure; per-frame device-idle is not this proof.
- Reject mismatched device/context identity or generation, invalid imports,
  incompatible usages, and unavailable capabilities before backend work.
  Loss invalidates device-specific plans; restoration recompiles for the new
  identity while preserving logical asset snapshots.
- Consult available lowering facts where correctness, resource planning, or
  material cost depends on them. Missing facts trigger a bounded consumer-led
  RHI request, not an invented graph capability or cost model.

## 2. Material semantics and Shader core

Owners: `fluxel-material` for semantics; `fluxel-shader` for compilation.

First define the minimum material semantics, normalized fragment metadata,
binding ABI, and shader interface needed by the first renderer. Material IR
records typed semantics and composition; WGSL/Naga remain the executable
language/IR. Do not implement a second shader language or compiler.

Build Shader as an independent crate reusable by Material, compute, particles,
and PostFX. It owns composition, reflection, artifacts/interfaces, target
requirements, canonical variants, and compilation caches. WGSL is the initial
author-facing input; other source formats require a selected consumer and
validation scope. Shader does not depend on Material or Renderer. Material
consumes Shader and does not depend on Renderer or own graph passes/submission.

**Acceptance:** compose multiple fragments, compare metadata with reflected
interfaces, and produce the artifacts/bindings needed by the selected RHI.
Cover static-key changes, dynamic-value sharing, cache invalidation, missing
fragments, type/binding mismatch, and unsupported target features.

## 3. Material and data-oriented Renderer MVP

Owner: Rendering, with an independently compiled Rust Feature consumer.

Implement the minimum MaterialGraph/IR, material assets/instances, parameters,
and compiled variants needed for a real textured 3D mesh. Define selected
surface outputs and explicit refusal for unsupported semantics; do not silently
approximate. Dynamic values share variants; static choices affect identity.

Renderer owns render-only camera/views, stable generational RenderIds, typed
dense storage, and batch incremental updates. Caller adapters map ECS entities
or ordinary scene objects to RenderIds. Establish a frame snapshot/version
boundary before concurrent reads. Validate empty batches, stale IDs,
duplicate/conflicting updates, removal/reuse, and capacity limits. Use dirty
ranges rather than full-world copies/uploads. Do not add an ECS, query language,
universal object layout, or general system scheduler.

Prove basic visibility, phase-specific deterministic ordering/batching, an
internal replaceable `FramePipeline` SPI, and a built-in minimal Forward path.
Object sorting is distinct from Graph scheduling. Dynamic dispatch may occur
per Feature batch, not through per-object virtual `draw()`. Keep room for
GPU-generated indirect arguments without forcing a final CPU DrawList.

**Acceptance:**

```text
Bases snapshots + batched updates
  -> Material -> Shader -> prepared material variant
  -> RenderScene / RenderView + built-in Mesh + independent Feature
  -> phase queues / FramePipeline -> internal Graph -> RHI
  -> observed textured 3D output and Feature contribution
  -> completion-based retirement across overlapping frames
```

The external Feature lives in a separate Rust crate and owns typed data. It
must prove batch updates, pipeline-phase participation, and declared Graph
resource access without modifying Renderer Core for a new object type. Cover
unavailable phases, incompatible shader/binding requirements, failed preparation,
and unsupported capabilities. All GPU work is Graph-visible; Features cannot
submit separately or own another frame scheduler. Retain an ordinary adapter
and an ECS-style batch fixture without adding an ECS dependency. Stabilize
extension contracts only after this proof. Complete terrain, water, particles,
or skinning systems are not MVP gates.

## 4. Material, pipeline, and Feature expansion

Owners: Material/Shader for semantics; Renderer/Feature crates for algorithms.

Expand code nodes, graph lowering, variants, PBR, Forward+, shadows, and PostFX
incrementally. Prove Deferred as an independent second `FramePipeline` using
the same scene/material contracts. Built-in pipelines remain replaceable.
Shadows and PostFX may be reusable passes rather than complete pipelines.

Use materially different Features, such as skinning and terrain, as selected
workloads to test extension boundaries. Terrain rendering/LOD, water passes,
GPU particles, skinning uploads/compute, and area-light shader extensions are
rendering work. Gameplay simulation, animation state machines, source formats,
and disk streaming remain outside Renderer. Controlled shader/material
extensions are necessary alongside Graph contributions for lighting plugins.

**Acceptance:** reuse scene/material inputs across Forward and Deferred and
add a selected external Feature without a core object-type switch. Specify
phase/shader compatibility and structured refusal per pipeline. Support Rust
crate plugins; dynamic DLL loading and a stable plugin binary ABI are unscheduled.

## 5. Performance and resource strategies

Owner: Renderer integration; callers/platform services retain source policy.

Measure CPU/GPU workload, upload volume, memory budgets, cached-plan reuse,
and retirement before selecting optimization. Grow residency, streaming/LOD,
GPU culling, indirect work, batching, and async compute according to measured
need and device capabilities.

**Acceptance:** retain reproducible baseline/candidate workloads, timings,
allocation/upload counts, and declared thresholds. Exercise replacement while
old content is in flight, eviction under pressure, loss/recreation, cancellation,
and upload failure without stale bindings or early release. Keep CPU asset
cache policy, per-device residency, and graph transients distinct.

## Undecided

- Capture/replay
- Canvas 2D and basic text
- Declarative UI
- Blender plugin
- JavaScript wrappers

These items have no delivery order, design, API, or acceptance plan.

## Evidence and completion

Follow the ecosystem [evidence policy](https://github.com/fluxel-project/.github/blob/main/EVIDENCE_POLICY.md).
Retain exact revisions/locks, invocation, inputs, expected/actual output,
backend/device/runtime identity, diagnostics, and unsupported/untested routes.
Visual behavior needs inspected real-run output. Tests and compilation alone
do not certify GPU or presentation behavior.

RHI stability does not certify every optional capability or backend. Its
[TODOs](https://github.com/fluxel-project/fluxel-rhi/blob/main/TODO.md) and
[verification guide](https://github.com/fluxel-project/fluxel-rhi/blob/main/documents/verification.md)
remain authoritative; missing Apple GPU evidence leaves Metal runtime acceptance
open. Re-run affected producer/consumer combinations when contracts change.
Documentation alignment alone closes no runtime gate.
