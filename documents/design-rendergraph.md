# Fluxel RenderGraph design

RenderGraph is Fluxel's GPU-work planner. It turns a frame pipeline's declared
passes and logical resources into a deterministic plan, then records that plan
directly with the portable `fluxel-rhi` API. It is not a second GPU API and it
is not a backend abstraction.

The current implementation order is in [the implementation plan](version-plan.md).
The portable GPU contract—devices, descriptors, resource uses, command
recording, submission, completion, presentation, and capabilities—is defined
only by [the RHI design](../crates/rhi/documents/design-rhi.md).

## Position in the frame path

```text
RenderScene
        -> FramePipeline
        -> RenderGraph
        -> Shader / Pipeline + Material Runtime
        -> RHI portable API
        -> private backends
```

This is the frame-execution path, not a universal crate dependency chain.
Material compilation is separately `MaterialGraph -> Material IR -> Shader
System -> ShaderArtifact / Pipeline requirements`. Renderer, RenderGraph,
shader/pipeline code, and material runtime may each reuse RHI portable
contracts; only graph execution retains graph pass-local resource authority.

`FramePipeline` resolves material and shader variants before it declares a
pass. Consequently it can declare the pass's reads, writes, attachments,
pipeline, bindings, and draw or dispatch work accurately. RenderGraph then
plans that work; it does not interpret material graphs, select a rendering
strategy, load assets, or own presentation.

## Ownership

RenderGraph owns graph-specific semantics:

- logical buffers, textures, imports, exports, and resource versions;
- definedness and typed pass-local access;
- declared reads and writes, plus RAW, WAR, WAW, and explicit dependencies;
- DAG construction, roots, dead-pass culling, deterministic scheduling, and
  compile diagnostics;
- logical resource lifetime and transient alias opportunities.

RHI owns portable GPU semantics and physical realization. RenderGraph uses RHI
types directly where those meanings already exist, including descriptors,
formats, usage flags, shader stages, resource uses, pipeline and binding
handles, command recording scopes, submission primitives, capability facts,
transient-allocation requirements, and `FrameAttachment`.

Backends alone own native API details such as `VkImage`, command buffers,
native barriers, queue families, descriptor heaps, fences, semaphores, heap
offsets, and native synchronization state. RenderGraph must not access those
details.

This division is deliberate: graph-specific semantics belong here, portable GPU
semantics are reused from RHI, and backend semantics stay private. A duplicate
Graph descriptor, capability, command, or binding vocabulary is not an
abstraction benefit.

## Definition, compilation, and execution

The ordinary path is short:

```text
graph definition
        -> compile
CompiledGraph + per-frame RHI bindings
        -> record through RHI
RHI RecordedWork
        -> RHI SubmissionPlan
```

`CompiledGraph` is an immutable graph result associated with the current RHI
device and its capability environment. Device loss or replacement means the
graph is compiled again. The design intentionally does not promise cross-device
reuse through capability, allocation, or presentation fingerprints.

A graph may retain logical import slots when that makes a stable definition
reusable. Per-frame bindings bind those slots directly to RHI `Buffer`,
`Texture`, `Pipeline`, binding objects, or `FrameAttachment` values. RHI and
RenderGraph validate device identity and generation, descriptors, permitted
usage, known resource state, presentation lifetime, and pipeline/binding
compatibility without copying that model into a bridge.

`GraphInstantiation` or `GraphExecutionPlan` may exist only when they provide a
real graph-only benefit, such as stable dynamic bindings, deterministic
diagnostics, CPU-only compile testing, capture provenance, visualization, or
plan caching. They are not mandatory layers. `RecordedWork` and
`SubmissionPlan` remain RHI objects.

## Pass recording

Pass callbacks receive graph pass-local authority, including its pass-local
resolver, rather than the RHI recorder itself. The authority holds the
corresponding RHI portable raster, compute, or copy recording scope and
resolves only resources, pipelines, and bindings declared for the pass.
Resource-bearing commands accept only values resolved through that authority,
so a callback cannot introduce undeclared resources.

This graph authority owns permission and resource resolution only. Its
callback-facing operations reuse RHI's command vocabulary and semantics
directly—`draw`, `dispatch`, `copy`, pipeline binding, bind-group binding,
viewport, and scissor—and delegate to the held RHI scope without a second
command model or later translation. This preserves the graph declaration as
the authority for dependencies, lifetimes, culling, and alias analysis.

Recorded commands derive RHI `ResourceUse`. Graph declarations provide the
planning contract; RHI command recording provides the portable execution
contract. Both layers validate their own facts and report structured errors at
the owning boundary.

## Imports, exports, and presentation

Imports describe graph-visible logical resources and their incoming contract.
They may bind portable RHI resources directly. A logical slot is a graph reuse
mechanism, not an isolation mechanism between the graph and RHI.

Exports describe the final graph semantic use for a later graph or external
consumer. Presentation follows RHI's contract: `FrameAttachment != Texture`.
A graph may model a logical presentation version when dependency analysis
requires it, but its final binding is the RHI `FrameAttachment`; it does not
need a second presentation-resource system.

## Transients and scheduling

RenderGraph derives logical lifetimes, non-overlap, and alias opportunities.
RHI supplies physical allocation requirements and realizes allocation,
dedicated fallback, alias barriers, heap placement, and backend strategy.
Aliasing is an optional optimization, never a prerequisite for correctness.

The baseline scheduling plan is deterministic and correct. Multi-queue work,
parallel recording, command caching, and physical aliasing require measured
need and must preserve the same portable semantics.

## Diagnostics and boundaries

Graph diagnostics explain invalid declarations, versions, ranges,
initialization, dependency conflicts, culling, and unsupported portable
requirements. RHI diagnostics explain invalid device/resource/pipeline/binding
objects, recording, submission, completion, and presentation. Renderer
diagnostics explain scene, material, variant, residency, and pipeline policy.
Errors are not wrapped into a new cross-layer hierarchy merely to make the
layers look symmetrical.

RenderGraph has a direct dependency on `fluxel-rhi`, but no dependency on a
backend-private API. Renderer depends on both and supplies scene preparation
and frame-pipeline policy; it is ordinary glue code, not a Graph/RHI protocol
translator.
