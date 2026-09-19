# Fluxel RenderGraph architecture

This document describes the RenderGraph architecture completed by `0.18` and
`0.19`. The [foundation interface
contract](design-foundation-interfaces.md) defines only cross-layer invariants
and Graph/capture integration boundaries; it is not a RHI type or error
inventory. The strict RHI-before-Graph delivery gates are in the [version
plan](version-plan.md). All RHI-facing types, errors, and validation are
governed exclusively by [RHI public API v1](design-rhi.md) and its `rhi-design`
modules, especially sections 37-38 and 50-51.

## Purpose and boundary

```text
authoring
  -> logical validation, versions, dependencies, roots, culling, lifetime
  -> target capability/lane/allocation-requirements compilation
  -> immutable CompiledGraph
  -> per-frame GraphInstantiation
  -> GraphExecutionPlan
  -> RHI RecordedWork + SubmissionPlan
  -> completion and retirement
```

Renderer decides what a frame means. RenderGraph decides which declared work is
valid and necessary, how logical contents flow, and what ordering/lifetime the
work requires. RHI validates and executes the portable plan. Backend selects
native synchronization, descriptors, allocation realization, encoders, and
queues.

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
graph contains no device object, but its capability and opaque allocation
profiles are part of its cache identity.

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
graph object slots. Instantiation binds compatible live device objects; logical
bind-group recipes refer only to setup-produced resource handles and are
realized after transient allocation. The raster, compute, and copy recorders
consume only resolver-produced references, never raw RHI objects.

Setup executes before recording and creates typed handles such as `BufferRead`,
`BufferWrite`, `TextureRead`, `TextureWrite`, `ColorAttachment`,
`DepthStencilAttachment`, and `FrameAttachmentWrite`. Execute receives a
resolver that accepts only handles created for that pass. It cannot fetch an
arbitrary global RHI resource.

Attachment builder descriptors carry load/store/clear/render-area and resolve
target policy, while the compiler-owned normalized form carries the resulting
`before`/`after` versions. A resolve target is a separate write with a returned
version. Direct rendering or resolve into a presentation target stays in the
`FrameAttachmentVersion` domain; `FrameAttachment` never becomes `Texture`.
Direct MSAA resolve to a frame version is admitted only when the active
presentation facts and RHI route facts prove the target, format, sample count,
and resolve route. Otherwise Graph must declare an ordinary single-sample
intermediate resolve target followed by a final raster write to
`FrameAttachmentVersion`.

This is a correctness boundary: graph dependencies are derived from the full
declaration, and the RHI/capture layer can mechanically verify every command
reference against it. An explicit `depends_on` supplements non-resource
semantics; it cannot replace a resource read/write declaration.

## Resources, versions, and definedness

Logical resources are:

- graph-created transient buffer or texture;
- persistent imported RHI buffer or texture;
- acquired presentation `FrameAttachment`;
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

- imports declare descriptor, physical device resource identity, initial
  semantic use, initial contents, external ownership, required usage, and lease;
- exports declare final semantic use and retention boundary.

Instantiation validates those declarations against Fluxel-known ownership and
history. It cannot promise to query a driver for a universal actual state. The
checks include identity/generation, descriptor, allowed usage, semantic-use
compatibility, definedness, and completion-safe lease.

The same physical generation cannot back two live logical resources without an
explicit alias/external-ownership contract. External memory/synchronization
requires separate ownership-transfer, producer/consumer synchronization,
handle-safety, loss, and failure design.

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
7. logical serial schedule (`0.19` may select proved pairwise lane routes);
8. resource lifetime intervals;
9. opaque allocation-requirements queries and conservative no-alias planning
   (`0.19` activates compatible reuse/alias decisions);
10. immutable plan and deterministic report.

The target supplies enabled facts, lane routes, presentation facts, and an
opaque `TransientAllocationRequirements` query. This is required even though
Graph owns logical overlap: size, alignment, compatibility class, and
dedicated-only constraints decide whether target storage can actually alias.

`CompiledGraph` records graph generation, target device identity, capability
fingerprint, allocation/presentation fingerprints, an immutable logical plan
template, retained recipes, and report. It is not the per-frame
`GraphExecutionPlan`. Changes to graph definition, enabled
capabilities, relevant format/surface routes, lane routes, or allocation profile
invalidate the cache entry. No native object is cached in it.

## Per-frame instantiation

Instantiation binds persistent imports, acquired frames, and declared
layout/sampler/pipeline object slots to a compiled graph. Before calling a
provider, allocating a transient, or opening an encoder it validates:

- identity and generation;
- descriptor and required/allowed usage;
- initial semantic use against Fluxel-known history;
- initial contents and definedness;
- owner lease and retention;
- frame generation and exactly-once consumption; and
- every resource and object binding is present once and has the required device
  generation, compatibility fingerprint, and target signature.

Failure is side-effect free at this stage. Missing, duplicate, and unknown slots
are different errors. A successful instantiation binds the compiled template
into a per-frame `GraphExecutionPlan`; lowering then creates recorded work, a
`SubmissionPlan`, and typed export/readback/present/retirement output slots.
Readback roots are P0 graph behavior; only a general Host pass is deferred.

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

Graph consumes API v1 `TransientAllocationService`: RHI returns opaque
`AllocationRequirements`, Graph computes logical intervals and a proposed
`TransientAllocationPlan`, and RHI returns a realization. The conservative
no-alias plan is mandatory.
`AliasBoundary` names prior occupants, new resource, first use, and required
first contents. Native alias barriers and memory placement remain backend
private.

Imported, exported, external, and presentable resources cannot participate in
ordinary transient aliasing while leased. Reuse is segregated by device and
compiled-graph generation and waits for terminal GPU completion. A no-alias,
one-allocation-per-resource fallback must always work; aliasing is never required
for correctness.

## Presentation and readback

An acquired frame enters as `FrameAttachment`, not texture. Present is an
observable root. Lowering consumes the frame with
`SubmissionPlanBuilder::present_after(frame, PlanPoint)` and exposes the
resulting present receipt; Graph does not invent a parallel presentation
request. Present mode stays in RHI presentation configuration.

```text
acquire -> bind FrameAttachment -> declared graph use -> planned present
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

`0.19` may serialize this as `GraphTraceArtifact` for diagnostics. It cannot be
accepted by replay. Graph also exports canonical `FrozenGraphIR`, but normal
`0.20` replay uses captured `PortableCommandIR`; Graph IR is provenance and
validation. Recompiling it is a separately labeled comparison mode.

## Validation and errors

The graph error model separates authoring, compile, instantiation, recording,
and lowering. It preserves pass/resource/version IDs and covers stale handles,
wrong-pass authority, undeclared use, undefined read, checked range failures,
format/usage/route mismatch, overlapping subresources, cycle, missing root,
import semantic/lease mismatch, missing frame binding, frame double consume,
alias overlap, allocation refusal, and backend lowering failure.

Compile and report output must be deterministic for identical graph and target
facts. CPU-only tests cover every validation class; `TestRhi` proves the plan
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
  instantiate/     per-frame binding and preflight
  plan/            GraphExecutionPlan and RHI SPI
  trace/           deterministic report, visualization and FrozenGraphIR
  test_rhi/        CPU-only protocol implementation
```

The module shape may evolve. The ownership boundary and compilation phases may
not be collapsed merely to mirror one backend.

## Deferred graph features

The following remain evidence-gated after `0.19`: a general Host pass, typed
blackboard, reusable subgraph/templates, conditional passes, history/temporal
rings, multi-device and external-memory/sync nodes, XR compositor nodes,
sparse/residency and video nodes, adaptive memory/cost scheduling, and debugger
step/checkpoint features. They cannot reserve empty public traits or handles.

The graph phase is “complete” when the scoped `0.18`/`0.19` gates pass on the
full backend matrix, not when every deferred domain has been guessed in advance.
