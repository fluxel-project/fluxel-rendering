# Fluxel RenderGraph design

> Target ownership: the `graph` module inside `fluxel-renderer`, not an
> independent target crate. The existing `fluxel-rendergraph` package is a
> migration source whose implementation, tests, and diagnostics are retained.
> Module ownership preserves graph/scene semantic separation.

RenderGraph is Fluxel's GPU-work planner. It turns a frame pipeline's declared
passes and logical resources into a deterministic plan, then records that plan
directly with the portable `fluxel-rhi` API. It is not a second GPU API and it
is not a backend abstraction.

The current implementation order is in [the implementation plan](version-plan.md).
The portable GPU contract—devices, descriptors, resource uses, command
recording, submission, completion, presentation, and capabilities—is defined
only by [the RHI design](https://github.com/fluxel-project/fluxel-rhi/blob/main/crates/core/DESIGN.md).

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
Material compilation is separately `MaterialGraph -> Material IR -> generated
evaluation WGSL and generic source/interface contribution -> FramePipeline adds
a pass template and other program inputs -> ShaderCompositionRequest -> Shader
System -> ShaderArtifact / Pipeline requirements`. Renderer, RenderGraph,
shader/pipeline code, and material runtime may each reuse RHI portable
contracts; only graph execution retains graph pass-local resource authority.

`FramePipeline` resolves material and shader variants before it declares a
pass. Consequently it can declare the pass's reads, writes, attachments,
pipeline, bindings, and draw or dispatch work accurately. RenderGraph then
plans that work; it does not interpret material graphs, select a rendering
strategy, load assets, or own presentation.

A pass is an execution node, not a material or a shader program. A raster pass
may record many draws, switch among prepared graphics pipelines, and use a
vertex stage plus an optional fragment stage for each draw. A compute pass may
record one or more dispatches using compute pipelines. A copy or clear pass
needs no shader. Compute and raster work are separate pipeline uses: a single
pass is never a combined vertex/compute/fragment program. A material may also
participate in several passes, such as depth and main shading. These
relationships do not make the complete Material/Shader system a prerequisite
for a minimal Graph/RHI execution proof.

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
transient-allocation requirements, acquired `SurfaceImage` leases, and
`AttachmentView`.

Backends alone own native API details such as `VkImage`, command buffers,
native barriers, queue families, descriptor heaps, fences, semaphores, heap
offsets, and native synchronization state. RenderGraph must not access those
details.

This division is deliberate: graph-specific semantics belong here, portable GPU
semantics are reused from RHI, and backend semantics stay private. A duplicate
Graph descriptor, capability, command, or binding vocabulary is not an
abstraction benefit.

## Definition, compilation, and execution

The foundation gate reuses a compiled plan across 2–3 overlapping frames,
retains distinct completion points, and proves safe reuse/retirement without
waiting for device idle each frame. Cover wrong-device resource binds, partial
and indeterminate execution outcomes, and device loss. Features declare all
frame GPU accesses through Graph and do not submit separately. Renderer residency uploads
may precede graph recording only through the shared upload coordinator and its
imported producer dependencies; they are not a feature submission path. The
implementation plan defines the selected real-backend matrix and exact
dependency evidence.

The ordinary path is short:

```text
graph definition
        -> compile
CompiledGraph + per-frame RHI bindings
        -> RHI SubmissionPlan + explicit transitions
        -> begin RHI execution
        -> deferred native recording or immediate issue
        -> finish outcomes + completion evidence
```

`CompiledGraph` is an immutable graph result associated with the current RHI
device and its capability environment. Device loss or replacement means the
graph is compiled again. The design intentionally does not promise cross-device
reuse through capability, allocation, or presentation fingerprints.

A graph definition contains pass topology, logical resources, declared uses,
and stable binding layouts. Compilation derives a device-specific plan from
that definition. Per-frame instantiation supplies changing imports, attachment
images, dynamic offsets, and prepared binding values. Changing a resource value
inside an already-declared binding can reuse the plan only after its concrete
use manifest is revalidated. This covers material sets whose many texture or
buffer instances change between frames. New concrete ranges and read-only
resource sets can remain compatible with the declared access envelope. A changed
pass set, dependency edge, incompatible access/range contract, alias-induced
hazard, or binding layout requires recompilation or refusal. Resource-instance
churn alone is not a topology change. Instantiation must prove compatibility;
it cannot skip hazard checks to preserve a cache hit.

A graph may retain logical import slots when that makes a stable definition
reusable. Per-frame bindings bind those slots directly to RHI `Buffer`,
`Texture`, graphics/compute pipelines, binding objects, or acquired `SurfaceImage` values. RHI and
RenderGraph validate RHI device provenance, descriptors, permitted
usage, known resource state, presentation lifetime, and pipeline/binding
compatibility without copying that model into a bridge.

Every pipeline/binding declaration has a resource-use manifest. Shader
reflection supplies the abstract binding slots, stages, and access classes;
material and draw preparation resolve those slots to concrete buffer and texture
references, subresource or byte ranges, and the RHI device provenance
of each resource. The pipeline must declare that concrete manifest as graph
reads or writes before recording begins. Graph validates it against the pass
declaration and folds buffer/texture accesses into hazard analysis, transitions,
and dependencies. Samplers are retained and validated as bindings, but do not
create memory hazards of their own. Binding a valid object alone is never proof
that the resources it contains are graph-declared.

`GraphInstantiation` or `GraphExecutionPlan` may exist only when they provide a
real graph-only benefit, such as stable dynamic bindings, deterministic
diagnostics, CPU-only compile testing, visualization, or
plan caching. They are not mandatory layers. `RecordedWork` and
`SubmissionPlan` remain RHI objects.

## Pass recording

Pass callbacks receive graph pass-local authority, including its pass-local
resolver, rather than the RHI recorder itself. The authority holds the
corresponding RHI portable raster, compute, or copy recording scope and
resolves only resources, pipelines, and bindings declared for the pass.
Resource-bearing commands accept only values resolved through that authority,
so a callback cannot introduce undeclared resources.

Before a pass callback records any command, graph instantiation verifies the
prepared binding manifest against the compiled declaration: binding layout,
concrete resources and ranges, access mode/stage, RHI device provenance,
and current imported state must agree. A mismatch is a structured graph or RHI
error before backend recording. This check applies again when a cached plan is
instantiated with refreshed bindings.

This graph authority owns permission and resource resolution only. Its
callback-facing operations reuse RHI's command vocabulary and semantics
directly—`draw`, `dispatch`, `copy`, pipeline binding, bind-group binding,
viewport, and scissor—and delegate to the held RHI scope without a second
command model or later translation. This preserves the graph declaration as
the authority for dependencies, lifetimes, culling, and alias analysis.

Graph declarations and pass-local command checks provide the access-planning
contract. RHI validates portable command shapes and provenance; it does not infer
the graph's hazards or supply missing barriers. Do not assume the historical
command-derived `ResourceUse` interface exists in the selected RHI. Both layers
validate their own facts and report errors at the owning boundary.

## Imports, exports, and presentation

Imports describe graph-visible logical resources and their incoming contract.
They may bind portable RHI resources directly. A logical slot is a graph reuse
mechanism, not an isolation mechanism between the graph and RHI.

Exports describe the final graph semantic use for a later graph or external
consumer. Presentation follows RHI's acquired-surface lease contract:
`SurfaceImage` is an exclusive lease, and `AttachmentView` has separate surface
and texture-view variants. A graph may model a
logical presentation version when dependency analysis requires it, but its final
binding is the acquired RHI `SurfaceImage`; it does not need a second
presentation-resource system.

## Transients and scheduling

RenderGraph derives logical lifetimes, non-overlap, alias opportunities, and
portable transition/dependency requirements. It can choose placement only through
the portable RHI allocation contract. RHI supplies physical allocation
requirements, realizes selected placement or the dedicated fallback, and each
backend lowers its own barriers, alias operations, heaps, and synchronization.
A wait alone does not supply memory visibility. Aliasing is an optional
optimization, never a prerequisite for correctness.

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
