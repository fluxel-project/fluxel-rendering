# Fluxel RenderGraph architecture

This document describes the planned RenderGraph architecture. The [workspace
architecture](design-overview.md) defines only cross-layer invariants
and Graph/capture integration boundaries; it is not a RHI type or error
inventory. The strict RHI-before-Graph delivery gates are in the
[implementation plan](version-plan.md). All RHI-facing types, errors, and validation are
governed by the public API, its rustdoc/tests, and [RHI architecture](../crates/rhi/documents/design-rhi.md)
with the relevant ADRs.

## Purpose and boundary

```text
authoring
  -> logical validation, versions, dependencies, roots, culling, lifetime
  -> GraphTargetProfile capability/lane/storage-class compilation
  -> immutable CompiledGraph
  -> logical frame inputs + GraphInstantiation
  -> GraphExecutionPlan
  -> renderer-private bridge + PreparedRhiBindings
  -> RHI RecordedWork + SubmissionPlan
  -> completion and retirement
```

Renderer decides what a frame means. RenderGraph decides which declared work is
valid and necessary, how logical contents flow, and what ordering/lifetime the
work requires. RHI validates and executes portable recorded work and submission
plans. A renderer-owned, workspace-private bridge lowers graph IR into RHI
objects; neither public crate depends on the other. Backend selects native
synchronization, descriptors, allocation realization, encoders, and queues.

RenderGraph owns no scene, asset manager, material policy, native handle,
barrier, queue family, heap, fence, command list, swapchain, or host lifecycle.
Trace is diagnostics; it is not portable replay.

## Required model

The design depends on four inseparable properties:

1. typed pass-local handles;
2. logical buffer/texture content versions and definedness;
3. immutable, target-aware `CompiledGraph` plus per-frame instantiation; and
4. an explicit `GraphExecutionPlan -> SubmissionPlan` boundary.

A mutable graph definition is never itself an executable native plan. A compiled
graph contains no device object. It is profile-affine: its graph-relevant
capability, route, and storage-class facts are part of its cache identity.
Device affinity begins only when the renderer-private bridge prepares concrete
RHI bindings and lowers the logical execution plan.

The capability flow has one authority:

```text
RHI EnabledCapabilities
        |
        | renderer-private, lossless-for-Graph projection
        v
GraphTargetProfile
        |
        v
RenderGraph::compile -> CompiledGraph
```

`GraphTargetProfile` is not a second hardware-capability system. It may omit
facts irrelevant to graph compilation, but it must never manufacture,
strengthen, or reinterpret an RHI capability. The bridge owns the projection
and tests every projected fact against the source capability snapshot.

## Pass declaration and authority

The stable pass kinds for this train are raster, compute, and copy. `Host` is
not frozen without a real consumer that proves whether the correct abstraction
is a pass, node, continuation, or completion callback.

Each declaration contains name, kind, complete resource uses, attachments,
observable effects, and optional scheduling hints. Kind controls legal command
vocabulary. Effects define roots. Hints may alter policy but never semantics.

Applications do not hand-write both a declaration and an execute closure.
`add_pass` first runs one setup recipe with `PassBuilder`; the builder is the
only source of uses and attachments and returns the typed handle bundle retained
for that pass. Later execution receives that bundle, a
`PassResourceResolver`, and a recorder restricted to the declared `PassKind`.
`PassDeclaration` is the compiler's normalized result. This prevents two
dependency sources of truth.

Setup runs synchronously once, every builder method reports
`GraphAuthoringError` immediately, and the retained handle bundle is owned and
`'static`. Raster/compute pipelines, bind-group layouts, and samplers use typed
graph object slots. Instantiation binds logical per-frame slot values; logical
bind-group recipes refer only to setup-produced resource handles and are
realized by the bridge after physical resource preparation. The raster,
compute, and copy recipes consume only resolver-produced logical references,
never raw RHI objects.

Setup executes before recording and creates typed handles such as `BufferRead`,
`BufferWrite`, `TextureRead`, `TextureWrite`, `ColorAttachment`,
`DepthStencilAttachment`, and `PresentationWrite`. Execute receives a
resolver that accepts only handles created for that pass. It cannot fetch an
arbitrary global RHI resource.

Attachment builder descriptors carry load/store/clear/render-area and resolve
target policy, while the compiler-owned normalized form carries the resulting
`before`/`after` versions. A resolve target is a separate write with a returned
version. Direct rendering or resolve into a presentation target stays in the
logical `PresentationVersion` domain; it never becomes `TextureVersion`.
Direct MSAA resolve to a frame version is admitted only when the active
presentation facts in `GraphTargetProfile` prove the target, format, sample
count, and resolve route. Otherwise Graph must declare an ordinary single-sample
intermediate resolve target followed by a final raster write to
`PresentationVersion`.

This is a correctness boundary: graph dependencies are derived from the full
declaration, and bridge/RHI tooling can mechanically verify every command
reference against it. An explicit `depends_on` supplements non-resource
semantics; it cannot replace a resource read/write declaration.

## Resources, versions, and definedness

Logical resources are:

- graph-created transient buffer or texture;
- persistent `ImportSlot` for a buffer or texture;
- logical presentation slot;
- exported/extracted result; or
- a future typed external-memory resource, which is not an ordinary import.

`BufferVersion` and `TextureVersion` are content lineage, not physical identity.
Every write produces one definite successor; every read names a definite
version. Texture versions track mip/layer/aspect ranges and buffers track byte
ranges.

A full write replaces a range, a partial write inherits untouched ranges from
the predecessor, and discard explicitly removes prior definedness. Reads from
never-initialized, never-imported, discarded, or non-inherited content fail.
The compiler constructs RAW, WAR, and WAW dependencies. Write ordering remains
necessary even when two accesses have equal texture-layout intent.

## Imports and exports

Import and export contracts speak in portable semantic use, not native state:

- imports declare a typed slot, descriptor, initial semantic contract, required
  usage, and definedness;
- exports declare final semantic use and retention boundary.

Graph instantiation validates logical frame inputs, slot coverage, descriptor
contracts, semantic-use compatibility, and definedness. It does not receive or
validate a physical resource, device identity, generation, or completion lease.

The renderer-private bridge resolves each slot into `PreparedRhiBindings` and
checks the concrete RHI resource's device identity/generation, descriptor,
allowed usage, Fluxel-known state, ownership, and completion-safe retention.
RHI remains the authority for whether the object is valid and executable.

The bridge rejects binding the same physical generation to incompatible live
logical slots unless a separately designed alias/external-ownership contract
permits it. External memory/synchronization requires separate ownership-transfer,
producer/consumer synchronization, handle-safety, loss, and failure design.

## Roots and culling

Observable roots are export, present, and readback.
The compiler walks backward from roots and retains only required passes. It
reports retained/culled passes, root paths, and deterministic cull reasons.

There is no stable `NeverCull` escape hatch. A pass that must remain states the
observable reason. External/host-effect callbacks remain deferred with the Host
execution model rather than becoming empty roots. Likewise, UE-style `NeverMerge`, `NeverParallel`, and
fork/join fence indices do not enter the public API.

## Compilation

Compilation occurs in a fixed semantic order:

1. authoring reference and pass-handle validation;
2. range, usage, and definedness validation;
3. import/export/root/readback/present validation;
4. target capability and route validation;
5. dependency/hazard construction and cycle detection;
6. reverse-reachability culling;
7. logical serial schedule, with proved pairwise lane routes enabled only by a
   later scheduling milestone;
8. resource lifetime intervals;
9. profile-backed storage compatibility and conservative no-alias planning,
   with reuse/alias decisions enabled only after their evidence gate;
10. immutable plan and deterministic report.

`GraphTargetProfile` supplies graph-relevant enabled facts, lane routes,
presentation routes, and opaque storage compatibility classes projected by the
bridge. Graph owns logical overlap and can propose compatible lifetimes, but it
does not call an RHI allocation service or claim that a proposal is physically
realizable. Dedicated-only constraints and final allocation realization remain
RHI concerns.

`CompiledGraph` records graph generation, a capability-compatibility
fingerprint, allocation/presentation fingerprints, an immutable logical plan
template, retained recipes, and report. It is not the per-frame
`GraphExecutionPlan`. Changes to graph definition, enabled
capabilities, relevant format/surface routes, lane routes, or allocation profile
invalidate the cache entry. No native object is cached in it.

## Per-frame instantiation and concrete preparation

`GraphInstantiation` combines a `CompiledGraph` with logical frame inputs. It
validates that every import, presentation, object, and parameter slot is present
exactly once and satisfies the compiled logical contract. Missing, duplicate,
and unknown slots are distinct errors. A successful instantiation produces a
device-independent `GraphExecutionPlan`.

The renderer-private bridge then combines that plan with concrete inputs:

```text
GraphExecutionPlan
        +
PreparedRhiBindings
RHI FrameAttachment
RHI resources, pipelines, bindings, and samplers
        |
        v
RecordedWork + SubmissionPlan
```

Before allocation or encoder creation, bridge/RHI preflight validates device
identity and generation, descriptors and allowed usage, Fluxel-known incoming
state, completion-safe retention, frame generation and exactly-once consumption,
object compatibility, and the exact capability snapshot behind the target
profile. Preflight failure is side-effect free. Readback roots are baseline
graph behavior; only a general Host pass is deferred.

## Dependency and RHI boundary

The boundary has one source of truth for each question:

- `GraphDependency` explains why graph nodes are ordered;
- `BufferUse` and `TextureUse` describe range/stages/access/layout intent;
- `ExecutionDependency` describes the happens-before RHI preserves;
- `AliasBoundary` describes a logical storage-occupancy change.

The recorder independently captures command-ordered actual uses. Lowering calls
`graph_bridge::validate_recorded_work` to prove that actual uses are covered by
the graph declaration and that no static contradiction exists. This is not a
second graph dependency derivation: Graph owns declared content/definedness and
the dependency DAG; RHI owns actual execution semantics and submission hazard
preflight. Backend still chooses the native implementation. Public barriers and
native state enums remain absent.

## Scheduling

Async compute is a logical lane assignment, not a separate public API. The base
serial lane executes every correct plan. A target with more lanes supplies a
pairwise dependency route: `Ordered`, `Gpu`, `Collapse`, or `Unsupported`.
Host waiting is not a lane route. More lanes do not claim hardware overlap.

Parallel setup/recording, batching, raster-scope merge, transition coalescing,
command caches, pass fusion/split, and cost-model scheduling are optional
lowering policies. They must be observationally equivalent to serial lowering
and may not change declared uses, roots, or ordering.

## Transient reuse and alias

Two optimizations remain distinct:

- cross-frame compatible reuse of a physical allocation after completion; and
- same-frame aliasing of distinct logical resources whose lifetimes do not
  overlap and whose opaque requirements are compatible.

Graph computes logical intervals and may group resources only by opaque
compatibility facts already present in `GraphTargetProfile`. It emits a logical
transient proposal. The renderer bridge asks RHI to realize that proposal, and
RHI may refuse it or select dedicated allocations. The conservative no-alias,
one-allocation-per-resource realization is mandatory.
`AliasBoundary` names prior occupants, new resource, first use, and required
first contents. Native alias barriers and memory placement remain backend
private.

Imported, exported, external, and presentable resources cannot participate in
ordinary transient aliasing while retained. Physical reuse is segregated by
device and bridge-owned plan generation and waits for terminal GPU completion.
Aliasing is never required for correctness.

## Presentation and readback

Graph declares a logical presentation slot, not a `FrameAttachment` or texture.
Present is an observable root. During lowering, the bridge binds an acquired RHI
`FrameAttachment` and consumes it with
`SubmissionPlanBuilder::present_after(frame, PlanPoint)` and exposes the
resulting present receipt; Graph does not invent a parallel presentation
request. Present mode stays in RHI presentation configuration.

```text
acquire -> prepare RHI presentation binding -> declared logical graph use -> planned present
        -> accepted submission -> present completion/loss
```

Every frame is consumed exactly once by present, explicit `abandon`, target loss,
or device loss. If a presentable image cannot be sampled/copied, the graph uses
an intermediate texture and an explicit supported final route.

Readback is an encoded copy root plus a completion-aware ticket and actual
returned layout; it is never immediate mapping. Upload is a retained-byte
`UploadJob` encoded into recorder order, never an implicit submit.

## Diagnostics and capture boundary

`CompileReport` and graph visualization derive from the same deterministic plan
data. Trace covers passes, resource versions/uses, dependencies, roots/culling,
lanes, semantic transitions, lifetimes, allocation requirements, aliases,
transient statistics, and submission mapping.

Graph may serialize this as `GraphTraceArtifact` for diagnostics. It cannot be
accepted by replay. Graph also exports canonical `FrozenGraphIR`, while replay
uses captured `PortableCommandIR`; Graph IR is provenance and validation.
Recompiling it is a separately labeled comparison mode.

## Validation and errors

The graph error model separates authoring, compile, instantiation, recording,
and lowering. It preserves pass/resource/version IDs and covers stale handles,
wrong-pass authority, undeclared use, undefined read, checked range failures,
format/usage/route mismatch, overlapping subresources, cycle, missing root,
import semantic mismatch, missing logical slot, alias overlap, and invalid
profile facts. Bridge/RHI errors separately cover wrong device/generation,
disallowed usage, unsafe retention, missing concrete binding, frame double
consume, allocation refusal, and backend lowering failure.

Compile and report output must be deterministic for identical graph and target
facts. CPU-only graph and bridge fixtures cover every validation class and plan
protocol; production backends prove output, completion, and refusal behavior.

## Module map

One possible internal layout is:

```text
crates/rendergraph/src/
  authoring/       logical resources, versions, pass declarations and handles
  validation/      range/use/definedness/import/root checks
  dependency/      hazards, explicit edges, cycles and culling
  schedule/        lanes and execution dependencies
  lifetime/        intervals, requirements and alias planning
  compile/         immutable CompiledGraph and CompileReport
  instantiate/     logical per-frame inputs and slot validation
  plan/            device-independent GraphExecutionPlan
  trace/           deterministic report, visualization and FrozenGraphIR
  test_support/    CPU-only graph/profile fixtures
```

The module shape may evolve. The ownership boundary and compilation phases may
not be collapsed merely to mirror one backend.

## Deferred graph features

The following remain evidence-gated after the complete scene-preparation plan:
a general Host pass, typed blackboard, reusable subgraph/templates, conditional passes, history/temporal
rings, multi-device and external-memory/sync nodes, XR compositor nodes,
sparse/residency and video nodes, adaptive memory/cost scheduling, and debugger
step/checkpoint features. They cannot reserve empty public traits or handles.

The graph phase is complete when the scoped graph and scene-preparation gates
pass on the full backend matrix, not when every deferred domain has been guessed
in advance.
