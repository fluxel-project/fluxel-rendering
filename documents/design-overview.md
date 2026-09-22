# Fluxel Rendering workspace architecture

> Architecture status: post-foundation target. The active work order is [the
> implementation plan](version-plan.md). [RHI design](../crates/rhi/documents/design-rhi.md)
> is the authoritative portable GPU contract; crate rustdoc and contract tests
> define descriptor-level behavior.

Fluxel Rendering turns renderer-selected scene data into portable GPU work.
Its architecture favors one owner for each meaning and reuses the portable GPU
contract instead of duplicating it for crate separation.

## Frame path

```text
Application / Game Scene
        -> RenderScene
        -> FramePipeline SPI <-> Material / Shader
        -> RenderGraph
        -> RHI
        -> private DX12 / Vulkan / Metal / WebGPU / GL backends
```

- `RenderScene` is render-domain scene data: no gameplay logic or GPU objects.
- `FramePipeline` decides how this frame renders: Forward, Deferred, tile,
  path-tracing, or a custom strategy.
- `Material` and `Shader` resolve shader variants, bindings, and pipeline
  requirements while the frame pipeline builds passes.
- `RenderGraph` plans GPU work: versions, dependencies, lifetimes, culling,
  scheduling, and graph diagnostics.
- `RHI` owns actual resources, commands, submission, completion, presentation,
  device lifecycle, and portable capabilities.
- Backends implement the RHI privately.

## Dependency direction and ownership

The core GPU-planning dependency spine is:

```text
fluxel-renderer
        -> fluxel-rendergraph
        -> fluxel-rhi
        -> private backends
```

This spine is not the complete crate dependency graph. Renderer also consumes
material and shader services during frame construction; those subsystems may
use RHI portable shader, pipeline, and binding vocabulary. Whether material and
shader become separate crates is a later boundary decision.
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
| `fluxel-renderer` | RenderScene, FramePipeline SPI, visibility, culling, sorting, asset-to-GPU residency, material/variant selection, draw preparation | Graph planning, RHI resource lifecycle, native backend objects, host/window policy |
| `fluxel-rendergraph` | Logical resources and versions, declarations, dependencies, culling, logical lifetimes, deterministic graph plan | Scenes, editable material graphs, asset policy, physical allocation strategy, native backend details |
| `fluxel-rhi` | Portable GPU resources, capabilities, recording, submission, completion, presentation, device lifecycle | Renderer policy, graph declarations, gameplay/asset policy, native objects in its public API |

`fluxel-assets` owns durable `AssetId<K>` and `ContentGeneration`. Renderer
GPU residency is per-device and private. Typed geometry or material references
must ultimately refer to that identity; `MaterialInstance` may have a distinct
renderer identity because an instance is not a material asset.

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

RenderGraph owns logical lifetime and alias analysis. RHI owns allocation
requirements, allocation realization, native alias barriers, heap placement,
and backend-specific synchronization. Dedicated transient allocation is the
correct fallback; aliasing is optional.

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
[RHI design](../crates/rhi/documents/design-rhi.md) for layer-specific detail.
