# Fluxel Rendering workspace architecture

> Architecture status: post-foundation target. The active work order is [the
> implementation plan](version-plan.md). [RHI design](https://github.com/fluxel-project/fluxel-rhi/blob/main/crates/core/DESIGN.md)
> is the authoritative portable GPU contract; crate rustdoc and contract tests
> define descriptor-level behavior.

Fluxel Rendering turns renderer-selected scene data into portable GPU work.
Its architecture favors one owner for each meaning and reuses the portable GPU
contract instead of duplicating it for crate separation.

## Frame execution

```text
RenderScene
        -> FramePipeline
        -> RenderGraph
        -> Shader / Pipeline + Material Runtime
        -> RHI
        -> private DX12 / Vulkan / Metal / WebGPU / GL backends
```

- `RenderScene` is render-domain scene data: no gameplay logic or GPU objects.
- `FramePipeline` decides how this frame renders: Forward, Deferred, tile,
  path-tracing, or a custom strategy.
- `RenderGraph` plans GPU work: versions, dependencies, lifetimes, culling,
  scheduling, and graph diagnostics.
- At execution, a pass selects its prepared shader/pipeline and binds its
  material-runtime data through graph pass-local authority.
- `RHI` owns actual resources, commands, submission, completion, presentation,
  device lifecycle, and portable capabilities.
- Backends implement the RHI privately.

## Material compilation

```text
MaterialGraph
        -> Material IR
        -> Shader System
        -> ShaderArtifact / Pipeline requirements
```

This is a compilation and generation relationship, not the frame-execution
path above. A compiled material variant is runtime input to the pipeline; its
authoring graph is not walked by a pass.

## Ownership DAG and portable contracts

The architecture is an ownership DAG, not a single crate dependency chain.
The target has three principal crates: `fluxel-shader`, `fluxel-material`, and
`fluxel-renderer`. RenderGraph and the `FramePipeline` SPI are internal renderer
modules. The existing standalone graph crate and WASM capsule are migration
inputs. The intended semantic ownership relationships are:

```text
fluxel-material -> fluxel-shader
fluxel-renderer -> fluxel-material
fluxel-renderer -> fluxel-shader
fluxel-renderer contains RenderGraph + FramePipeline
RHI -> private backends
```

`renderer` consumes material/shader services and RenderGraph. This does not
restrict portable RHI-contract reuse, which is intentionally direct:

```text
Renderer -----------------+
RenderGraph --------------+
Shader / Pipeline --------+--> RHI portable API
Material runtime ---------+
```

RHI is not an API that upper layers must reach only by forwarding through their
immediate neighbor. These direct uses are dependency edges to the shared GPU
foundation, but they do not transfer each caller's higher-level semantic
ownership to RHI.
RenderGraph directly uses RHI portable descriptors, formats, usages, capability
facts, resource uses, command scopes, submission primitives, and presentation
facts. It must never touch backend-private native objects.

There is no `GraphTargetProfile`, graph capability projection, or mandatory
Graph/RHI bridge. Renderer may contain ordinary glue code to resolve asset
residency and invoke graph recording, but it does not translate a duplicated
GPU contract. A per-frame graph binding can directly contain portable RHI
buffers, textures, pipelines, bindings, and frame attachments.

| Layer | Owns | Does not own |
| --- | --- | --- |
| `fluxel-renderer` | RenderScene, views, Features, FramePipeline and reference pipelines, visibility, sorting, batching, residency, draw preparation, retirement, internal Graph | Gameplay/ECS, platform I/O, backend objects, host policy |
| Renderer `graph` module | Logical resources/versions, dependencies, definedness, hazards, lifetimes, transitions, deterministic plans | Scene/material semantics, asset policy, backend allocation mechanisms |
| `fluxel-material` | MaterialGraph, code nodes, typed IR, parameters, instances, material variants, Shader requests | Renderer dependency, graph passes, submission, executable shader IR |
| `fluxel-shader` | Fragments, composition, WGSL/Naga, reflection, binding ABI, artifacts, variants, compilation caches | MaterialGraph semantics, scenes, rendering strategy |
| `fluxel-rhi` | Portable GPU resources, capabilities, recording, submission, completion, presentation, device lifecycle | Renderer policy, graph declarations, gameplay/asset policy, native objects in its public API |

`fluxel-assets` stays in `fluxel-bases` and owns durable `AssetId<K>`, immutable
`ContentGeneration`, producer coordination, and CPU cache policy. Callers own
loading, decoding, hot reload, and streaming policy; Host/adapters supply I/O.
It gains no GPU or platform dependency. Renderer
GPU residency is per-device and private. Typed geometry or material references
must ultimately refer to that identity; `MaterialInstance` may have a distinct
renderer identity because an instance is not a material asset.

## Data and extension boundaries

RenderScene is a render-only data store, not another ECS. Adapters push batched
incremental updates into typed dense storage using stable generational RenderIds.
They own mappings from external entities/scene nodes. Frame snapshots separate
updates from concurrent readers; dirty ranges bound uploads. Layout follows
access patterns rather than requiring one universal SoA shape.

Features operate on batches and own typed data. FramePipeline selects phases
and strategy; Features contribute packets, shader/material requirements, and
graph-visible GPU work. Object sorting differs from Graph dependency scheduling.
GPU-generated indirect work need not become a final CPU DrawList. All GPU
accesses and submission stay under Graph authority.

The MVP proves built-in Mesh plus a minimal Feature in an independent Rust
crate before extension interfaces stabilize. It accepts batch updates,
participates in a phase, and declares resources without a core object-type
switch. Full terrain, water, particles, skinning, and area lights are later
workloads. Rust crate plugins are supported; dynamic DLL loading is unscheduled.

## Migration and delivery

The current workspace still has `fluxel-rendergraph`, `fluxel-renderer`, and
`fluxel-rendering-wasm`, and pins an older RHI integration. Internalizing Graph,
creating Material/Shader crates, and upgrading real pinned consumers are planned
implementation work. Preserve code, tests, and evidence instead of maintaining
parallel engines. RHI is stable; select exact releases/revisions and re-prove
their composition rather than assuming old examples remain compatible.

Reuse Bases, establish internal Graph/RHI execution, then prove Shader and the
Material-driven Renderer MVP with its external Feature. Material/pipeline
expansion and measured performance/resource strategies complete the active scope.
The five undecided features are listed in the product plan without design or
sequencing commitments. The
[implementation plan](version-plan.md) supplies the evidence gates.

## Graph and RHI integration

```text
FramePipeline
        -> RenderGraph definition / compile
CompiledGraph + per-frame RHI bindings
        -> RHI RecordedWork
        -> RHI SubmissionPlan
```

RenderGraph compiles against the current RHI device and its authoritative
portable capability facts. If a device is lost or rebuilt, recompile the graph.
Avoid speculative compatibility fingerprints and cross-device reuse until
profiling proves their value.

Graph imports may be logical slots for reusable definitions, but they bind
directly to portable RHI resources. RHI/graph validation checks device identity
and generation, descriptor compatibility, allowed usage, known state,
pipeline/binding compatibility, and presentation lifetime. `FrameAttachment`
remains distinct from `Texture`.

RenderGraph owns logical lifetime, alias analysis, transition selection, and
GPU dependency planning. RHI supplies allocation requirements, realizes the
chosen allocation, and lowers the supplied native alias and synchronization
instructions. Dedicated transient allocation is the correct fallback; aliasing
is optional.

## Validation and evidence

Each layer reports the contract it owns: renderer errors cover scene,
residency, material variants, and pipeline policy; graph errors cover
declarations, versions, dependencies, lifetime, and planning; RHI errors cover
portable resources, recording, submission, completion, presentation, and
device state. Do not invent a cross-layer error taxonomy that hides the owner.

Correctness requires more than a local green test. Evidence distinguishes
CPU-only protocol tests, compile/link checks, and real hardware conformance.
New portable capabilities require validation, lowering, lifetime/loss handling,
and conformance evidence before publication.

See [renderer design](design-renderer.md),
[RenderGraph design](design-rendergraph.md),
[material design](design-material.md), and
[RHI design](https://github.com/fluxel-project/fluxel-rhi/blob/main/crates/core/DESIGN.md) for layer-specific detail.
