# Fluxel Rendering workspace architecture

> Architecture status: this is the post-foundation workspace/layer target. The
> active implementation order is [versions 0.16-0.20](version-plan.md), and the
> RHI architecture source is [RHI design](../crates/rhi/documents/design-rhi.md);
> rustdoc and contract tests define descriptor-level detail. This document is
> cross-layer only. RHI must be
> completed on every declared backend before RenderGraph implementation begins;
> RenderGraph must then be completed before portable capture/replay. New
> renderer/scene/canvas/runtime work resumes only after the `0.20` gate.

Fluxel Rendering is a layered Rust workspace for turning renderer-selected
scene data into portable GPU work, then executing that work through a small,
safe native boundary. The layers are deliberately separate: the renderer
chooses *what a frame means*, RenderGraph derives *what work and ordering that
meaning requires*, and RHI performs *how a selected backend owns and executes
the work*.

This document is the architectural entry point for the workspace. It describes
the stable layer model and the higher-level system to reconnect after the
foundation train. During `0.16`-`0.20`, higher layers may be explicitly dormant
while their dependencies are replaced; that temporary build state does not
change ownership or authorize a second architecture. This document does
not replace the crate designs, which define each layer in detail, or the ADRs,
which record why durable choices were made.

## Foundation-first execution mode

The dependency order is also the delivery order:

```text
0.16-0.17  Portable RHI and all backend/profile evidence
0.18-0.19  RenderGraph correctness, allocation, scheduling, and trace
0.20       Portable capture/replay
after 0.20 reconnect and extend renderer/scene/runtime work
```

RHI definitions include canonical descriptor/command/submission observation,
reconstructable shader artifacts, typed identity, readback layout, and external
input classification from the start. Those are capture prerequisites, not an
early capture implementation or file-format freeze.

The common public model contains RenderGraph resources, RHI bindings, devices,
surfaces, submission and completion. Browser/GL objects are backend-private.
There is no public or internal resource architecture based on browser
“sessions” or “asset tokens”; the exact RHI identity, lifetime, and retirement
rules are defined only by the normative RHI API.

## Goals and non-goals

The workspace is designed to make GPU correctness inspectable rather than an
accident of callback order or one backend's behavior. Its central goals are:

- portable, explicit resource-access and synchronization semantics;
- safe ownership of device-affine native objects across asynchronous GPU work;
- one immutable execution plan that can be lowered by more than one backend;
- renderer policy that stays separate from resource planning and native API
  details; and
- evidence that distinguishes CPU protocol tests, compile checks, and real GPU
  conformance.

It is not a scene/asset database, shader authoring framework, or host runtime.
The `0.16`-`0.20` foundation train implements portable RHI/Graph contracts
without exposing native mechanisms or turning backend features into universal support.
General material authoring and stable public asset/cache ABI remain outside
that foundation. The following is retained `0.15` historical evidence only:
the proven Windows presentation slice supported DX12 and Vulkan through one
narrow RHI surface façade. It handled resize/minimize/restore as generation
changes and independent acquired-frame tickets, while the harness privately
proved bounded frames-in-flight. It was not a general platform API or public
frame scheduler.

## Layer model and dependency direction

```text
application / asset system
        |
        v
fluxel-renderer  ---->  fluxel-rendergraph  ---->  fluxel-rhi
 scene policy             portable plan              native execution
```

Dependencies point downward. The renderer may use RenderGraph declarations and
RHI's safe opaque objects; RenderGraph does not depend on RHI or renderer; RHI
implements the execution SPI defined by RenderGraph. Native HAL types never
travel upward, and renderer concepts such as assets, materials, or visibility
never become graph concepts.

| Layer | Owns | Explicitly does not own |
| --- | --- | --- |
| `fluxel-renderer` | Domain inputs, renderer policy, fixed per-device GPU residency, GPU snapshot publication, fixed frame coordination, closed recipes, and the visible fixed-frame transaction | Logical asset loading/cache identity, graph compilation, native handles, barriers, swapchains, or host/window policy |
| `fluxel-rendergraph` | Logical resources and versions, declared accesses, validation, dependencies, culling, transitions, and immutable execution plans | Scenes, asset handles, shader/pipeline policy, allocation, native handles, queue submission, or readback implementation |
| `fluxel-rhi` | Device-affine native resources, opaque artifacts/bindings, backend lowering, command recording, submission, completion, diagnostics, and presentation as defined by the normative RHI API | Scene selection, asset policy, host/window ownership, general renderer lowering, or a public general graphics API |

Assets cross a repository boundary without moving platform or GPU policy into
one shared crate. `fluxel-bases` owns durable logical identity, typed handles,
content generations, loading state, and logical caching/reuse contracts.
Platform readers live in `fluxel-host` or `fluxel-jsbridge`; this workspace
owns only renderer-private, per-device GPU residency: fixed upload, recreation,
and frame-safe retirement. The renderer resolves an appropriate GPU-ready
snapshot by `(AssetId, ContentGeneration, DeviceIdentity)` during preparation,
before it declares the frame. RenderGraph receives only the resulting physical
binding and its contract. The reason for this boundary is recorded in
[ADR-0001](adr/0001-assets-outside-rendergraph.md) and
[ADR-0010](adr/0010-renderer-private-fixed-asset-residency.md). The native
containment boundary is recorded in
[ADR-0002](adr/0002-rhi-unsafe-containment.md).

For layer-specific contracts, see [Renderer design](design-renderer.md),
[RenderGraph design](design-rendergraph.md), [RHI design](../crates/rhi/documents/design-rhi.md), and
[capture/replay design](design-capture-replay.md). RHI rustdoc and contract
tests define descriptor-level API detail; [RHI design](../crates/rhi/documents/design-rhi.md)
and its ADR sequence define architecture; this workspace document records only
cross-layer invariants and integration boundaries.

## Frame data flow

The intended frame path has a stable conceptual shape:

```text
application scene/domain data
  -> renderer preparation resolves/starts fixed GPU residency
  -> renderer polls and selects only committed GPU snapshots and frame policy
  -> ordered render packet or closed fixed recipe
  -> acquire one RHI presentable image when presentation is requested
  -> RenderGraph declarations
  -> target-aware immutable CompiledGraph
  -> per-frame GraphInstantiation and GraphExecutionPlan
  -> RHI RecordedWork and SubmissionPlan lowering
  -> serial-base or capability-routed submission
  -> completion, export semantic use, presentation, and retirement
```

### Renderer-side selection

Application code supplies domain data such as cameras, geometry, materials, and
an insertion-ordered `DrawList`. A renderer decides which snapshot generation,
material behavior, and ordering policy are legal for the frame. Persistent CPU
data is validated before it enters the asynchronous GPU path.

For the retained fixed residency domains, preparation receives an immutable
`AssetSnapshot` and keys its private entry by `(AssetId, ContentGeneration,
DeviceIdentity)`. Pending uploads are not drawable; only a committed entry is
bound as a concrete graph import. Supersession, eviction, or device recreation
moves an old entry to retirement, where its RHI leases retain it until terminal
submission completion. A pass never receives `AssetStore`, and neither
RenderGraph nor the general/native RHI gains an asset/cache handle or a shader
generalization. Browser residency is represented by the same private
device-generation entries and completion-retained leases as other backends; no
browser token or session becomes a resource model.

The current renderer has deliberately narrow fixed paths: immutable
indexed mesh, texture, normal, and vertex-color snapshots are published only
after their uploads complete, and a fixed-frame coordinator selects one of a
small set of private, closed raster recipes. A recipe jointly specifies the
vertex/texture domain, graph access declarations, RHI kernel and bindings,
reservation topology, and exports. The vertex-color recipe is a representative
closed multi-stream contract: position `f32x3` in slot zero, linear
`UNORM8x4` color in slot one, `u32` indices, and the camera/tint uniform all
move together from immutable snapshot through graph declaration to RHI
recording. It is an internal consistency mechanism, not a configurable material
or pipeline API; see [ADR-0007](adr/0007-closed-fixed-renderer-recipes.md).

For the legacy unlit indexed contract, the renderer lowers a `DrawList` and a
positionally matching sequence of ready indexed snapshots into an owned,
opaque, device-affine `RenderPacket`. Construction copies camera/material
values and each draw's affine model-to-world placement, retains snapshot
leases, and validates exact CPU snapshot metadata; it does not reserve a
generation or expose graph/native objects. The renderer computes column-major
`projection * view * model` into the existing legacy-unlit uniform ABI, so
placement is draw/packet policy rather than a mesh property or an RHI binding
change. Submission deduplicates repeated snapshot generations for graph imports
and reservations, but retains one ordered draw and uniform per list entry. The
real CPU preparation dependencies (shared scene input, per-object work, and
ordered assembly) are privately represented with `slot-graph`; it neither
creates dummy work nor replaces RenderGraph's GPU semantics. It compiles one
graph with one raster pass, advances uniform uploads serially without blocking,
then submits the raster work once. This placement currently
does not extend Lambert normal handling. General material/layout variation,
PBR, scene loading, culling, batching, and a general persistent scene system are
not implemented; the current deterministic three-object scene is only a closed
presentation and preparation proof.

### RenderGraph declaration and compilation

The renderer declares passes rather than recording opaque, unordered native
work. Each pass states which version and range of each logical buffer or
texture it reads, writes, samples, attaches, copies, or accesses read/write.
From those declarations RenderGraph derives the dependency DAG, validity and
initialization checks, dead-pass culling, required usage, and semantic-use/
memory-dependency requirements.

Compilation produces an immutable, target-aware `CompiledGraph`. The
snapshot contains portable semantics, not an encoder, command buffer, queue,
or native allocation, but it may be specialized to the selected capability and
opaque allocation-requirements profile. Imports are stable slots; each frame
binds concrete resources to them. Exports name roots and carry a portable final
semantic-use contract for the next graph or external consumer. More detail is in the
[RenderGraph design](design-rendergraph.md).

An import binding identifies a provider-selected physical object and generation,
its declared initial semantic use, Fluxel-known history, allowed usage, and a
completion-safe lease. The executor checks these facts; an export records the
actual final semantic use established by execution rather than a guessed native
state. Compatible compiled graphs may privately reuse a
completed transient allocation only when target allocation requirements,
logical lifetime, and semantic history allow it. Reuse is segregated by device
and compiled-graph generation; graph
or device invalidation prevents a new checkout while non-terminal work
continues to retain its old-generation lease. It is not public aliasing or a
caller-visible cache. See [ADR-0009](adr/0009-resource-floor-and-reuse-safety.md).

### RHI resolution, execution, and completion

RHI resolves plan resources against opaque native resources, verifies device
identity, descriptors, declared incoming semantic use against Fluxel-known
history, and actual allowed usage, then lowers only declared commands. The
precise resource-use, lane, submission, completion, presentation, and
retirement semantics are defined by [RHI design](../crates/rhi/documents/design-rhi.md) and its ADRs;
`rhi-design` modules; this overview does not define an alternate RHI state
machine.

An export reports the final portable semantic use established by the plan.
Readback or a later consumer must declare a compatible incoming use and bind it
against Fluxel-known history; it must not invent a more convenient native state.
Submission is not treated as completion. The v1 API distinguishes plan
acceptance, GPU completion, and presentation outcome, and retains ownership
until the relevant terminal outcome is established. The old
accepted-unknown/quarantine terminology is retained only as `0.15` historical
background and is not authority for the v1 API.

The serial lowering is a correctness strategy, not a claim that graph passes
are tied to one encoder, command buffer, or queue. Multi-queue scheduling,
parallel recording, command caching, and aliasing remain possible future
optimizations only after they have a concrete measured benefit and preserve the
same portable plan semantics. See [ADR-0003](adr/0003-serial-execution-lowering.md).

## Lifetime and ownership model

The workspace keeps three lifetimes separate.

| Lifetime | Examples | Owner and rule |
| --- | --- | --- |
| Persistent domain lifetime | Geometry, material inputs, logical asset identity, renderer-private GPU residency metadata | Application/asset/renderer policy; not graph state |
| Per-frame logical lifetime | Graph versions, imports, exports, frame inputs, retained pass recipes | RenderGraph declaration/instantiation; a compiled plan contains no native per-frame object |
| Native asynchronous lifetime | Device, buffers, textures, pipelines, bindings, staging data, command objects, leases, completion handles | RHI; objects survive until the relevant work is proven retired |

An immutable GPU snapshot is the bridge between the first and third rows. It
encapsulates a concrete native generation plus leases and reported state, but
does not expose raw native handles. Publication is atomic across the resources
that make up the snapshot. The retained `0.15` `RenderPacket` baseline bridged
the persistent borrowed draw-list input and one graph execution through its
own device-affine reservation and poisoning rules. Those fixed-path rules are
historical implementation detail, not future RHI semantics.

RHI resources are device-affine and retain the opened native device through
shared ownership. Cloning a lease extends the resource/native-device lifetime;
the final owner performs destruction. All unsafe and HAL interaction remain in
the private RHI implementation subtree, so higher layers cannot accidentally
outlive, alias, or misuse a raw native object.

## Portable semantics and backend facts

RenderGraph describes portable meaning: resource operations, ranges, ordering,
initial contents, required usages, and exported final semantic uses. It does not
pretend that portable states are direct DX12 or Vulkan enumerations.

RHI reports backend facts separately: adapter identity, limits, supported
features, actual allowed resource usage after native normalization, and
validation availability. During frame resolution, the portable requirement
must be a subset of the resource's actual allowed operations. This prevents an
allocator or import binding from merely echoing what the plan requested.

The same compatible `GraphExecutionPlan` is intended to execute with the same observable
semantics on supported backends. Backend lowering may use different barriers,
resource flags, encoders, or state representations, and a same-state memory
dependency need not imply an identical native barrier. Those are
implementation facts as long as ordering, visibility, lifetime, and results
match the plan contract.

## Validation, errors, and evidence

Validation is layered rather than collapsed into one “it worked” result:

1. **Domain validation** rejects invalid scene, geometry, image, uniform, and
   fixed-recipe inputs before a GPU operation begins.
2. **Graph validation** rejects invalid versions, ranges, initialization,
   capability requirements, resource conflicts, and undeclared recording.
3. **RHI boundary validation** rejects foreign devices, invalid descriptors,
   usage/state/alignment violations, unavailable backends, and unsupported
   native prerequisites before unsafe calls.
4. **Completion validation** distinguishes rejection, pending work, proven
   completion, and unproven/failed accepted work without fabricating state.
5. **Native diagnostics and conformance** collect backend validation output
   and compare hardware readback against an independent CPU oracle.

Errors are structured at the layer that owns the failed contract. Renderer
errors never need to expose native handles; graph errors explain declaration
semantics; RHI errors identify native/open/resource/recording/submission or
completion boundaries without leaking unsafe types.

CPU-only tests and `TestRhi` prove compiler and protocol behavior. Compilation,
linking, and CI prove build coverage. Neither proves native GPU correctness.
A native conformance claim requires the same compiled plan on each supported
backend, required validation where available, collected diagnostics, exact
inputs/outputs, and an independent CPU oracle. The evidence policy is in
[ADR-0005](adr/0005-gpu-conformance-evidence.md).

## Repository and module map

```text
crates/
  renderer/       application-facing domain types, snapshots, packets, fixed-frame policy
    src/upload/   immutable snapshot upload/publication domains
    src/fixed_frame/ closed recipes, owned packets, and fixed-frame submissions
    src/shader/   private fixed shader sources/selection
  rendergraph/    portable graph declaration, validation, compiler, execution SPI
    src/compile/  dependency, validation, culling, transition-plan compilation
    src/execution/ frame instantiation and portable execution protocol
    src/plan/     immutable plan data and contracts
    src/test_rhi/  deterministic CPU-only execution-protocol backend
  rhi/            safe native resource and execution facade
    src/resource/ owned resources, uploads, leases, shader artifacts
    src/execution/ plan providers, command recording, completion helpers
    src/imp/      private HAL/native implementation and platform stubs
documents/
  design-*.md     target layer designs plus explicitly marked historical baselines
  adr/            durable architectural decisions and alternatives
  draft/          local, uncommitted plan/review working material
```

Each source module has one clear responsibility. Composition modules expose
only declarations, narrow shared contracts, and re-exports; implementation is
split by independently changing concerns. This preserves the `imp` subtree as
the sole unsafe/native containment boundary and keeps a public facade stable
while internal implementation evolves.

## Platform boundary

The portable graph and default renderer-domain model are not tied to a native
API. The remainder of this paragraph records the retained `0.15` baseline, not
the completion state of the rewritten `0.16`-`0.20` foundation. That baseline's
native execution is Windows-focused, with headless DX12/Vulkan feature
selection and a backend-neutral DX12/Vulkan surface-generation/ticket path. The
Windows proof harness owns its bounded three-slot admission and back-pressure
policy; RHI only guards native image availability and completion-driven teardown.
Non-Windows native requests fail explicitly rather than silently emulating a
backend. Lost native surface/device recovery and a general cross-platform
surface API remain absent. The Stage 2.1 WebGL2 and Stage 2.2 WebGPU paths are
closed browser adapters, not a general web renderer; DOM lifecycle and RAF
ownership remain in `fluxel-jsbridge`. The common fixed resource floor is
available on DX12, Vulkan, WebGPU, and WebGL2. Compute/storage-buffer and
writable storage-texture recipes are present on DX12, Vulkan, and WebGPU;
readable storage texture is currently Vulkan-only. DX12 reports its observed
read limitation and fails closed; WebGPU read is not promised. WebGL2 rejects
compute and every storage operation with structured capability evidence before
context/resource side effects, with no emulation. WebGPU separately owns a
private device-generation/canvas-epoch state machine, per-key resource registry,
completion-held leases, asynchronous recovery, and terminal disposal.
In normal browser rendering, a committed resident mesh is reused only for its
current device generation. Replacing content does not mutate an old entry: its
lease remains safe for in-flight work within the same generation while the
replacement uses a new content-generation entry. Generation loss removes
old entries from lookup and reuploads retained CPU snapshots for the replacement
generation. An image registry may exist for the same residency bookkeeping, but
the legacy-unlit browser path does not claim to sample a resident image; that
browser-only contract does not broaden native Surface.

The target RHI matrix replaces that implementation description in `0.16` and
`0.17`: real DX12, Vulkan, and Metal first, followed by browser WebGPU and the
GL-family desktop GL/GLES/WebGL2 profiles under the single v1 device-identity,
capability, submission, completion, presentation, and retirement contract. The
exact API is defined by rustdoc and tests, with [RHI design](../crates/rhi/documents/design-rhi.md) recording its architectural boundary; the
exact gates are in [version-plan.md](version-plan.md), and prior baseline
evidence is not reused as proof of the replacement.

Windows MSVC is the primary Windows development/native test environment. Linux
must be tested natively (for example in WSL2/Ubuntu), because it exercises
`cfg(not(windows))`, fallback, and feature paths that a Windows cross-build
cannot prove. When Android support is introduced, it requires an explicit NDK
target gate and executable emulator/device coverage for platform code. Platform
stubs must return their documented structured errors, not merely compile. The
rationale and testing rule are in [ADR-0008](adr/0008-native-platform-test-gates.md).

## Extension principles

New work should extend the lowest layer that naturally owns the new fact and
must not use a convenient lower-level escape hatch to bypass an existing
contract.

- Add scene/asset policy, snapshot selection, transforms, materials, and
  culling in the renderer; resolve assets before graph binding.
- Add portable resource-access semantics, compiler validation, and immutable
  plan contracts in RenderGraph; do not add native handles, barriers, or queue
  operations there.
- Add backend resource creation, artifacts, recording, synchronization,
  submission, and completion in RHI; keep raw native types and `unsafe`
  private.
- Add a closed recipe when a vertical slice proves a new combination. Do not
  turn a single proven combination into a general pipeline/material API without
  evidence for its ownership, reflection, layout, and portability contracts;
  see [ADR-0006](adr/0006-no-general-pipeline-yet.md).
- Treat performance work as a lowering optimization with benchmark and profiling
  evidence, never as a change to portable graph meaning.
- Record a short ADR when a decision will constrain more than the immediate
  implementation; keep current behavior in a design document and release
  investigation in local plan/review material.

This separation lets the rendering kernel grow from fixed headless
evidence-backed slices toward a general embeddable renderer without making
application code depend on backend accidents or transient implementation
policy.
