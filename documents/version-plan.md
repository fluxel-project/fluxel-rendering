# Fluxel Rendering Implementation Plan

> Status: active target plan; implementation migration is pending. Stage labels
> are work packages, not release numbers or completion claims. The organization
> [roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md) owns
> order and authorization; this document defines rendering scope and acceptance.

## Stages and dependencies

Every stage delivers an executable consumer with normal, boundary, and failure
cases. A crate, trait, or independently green library test is not stage closure.

| Stage | Libraries/modules and concepts | Required outcome |
| --- | --- | --- |
| S0 Foundation | Existing Bases `fluxel-assets`; stable independent RHI; asset identity, content generations, snapshots, device provenance, submission/completion | A pinned integration artifact creates, uploads, uses, and safely retires resources from real asset snapshots |
| S1 RenderGraph | `fluxel-renderer::graph`; passes, versions, definedness, DAG, hazards, compiled plans, frame bindings | Fixed-shader multi-pass execution, compatible plan reuse, safe retirement across 2–3 overlapping frames |
| S2 Shader Core | New `fluxel-shader`; composition request, WGSL/Naga validation, reflection, logical source artifacts, cache identity | Material-independent raster and supported compute programs execute through Graph and selected RHI shader routes |
| S3 Material Core | New `fluxel-material`; functions/fragments, parameters/resources, minimal composition IR, evaluation WGSL, instances/variants | Several functions and parameters compose into an executable material through the same Shader path |
| S4 RenderScene | Renderer scene storage, generational RenderIds, views, transforms/bounds, batch changes, snapshots | Incremental multi-object/view data and stable frame snapshots work without an ECS dependency |
| S5 Forward integration | Renderer pipeline/preparation/culling/sorting; public FramePipeline facade, phases, resolvers, residency and frame data | Multiple meshes/materials render through Scene -> FramePipeline -> Graph -> RHI; an early external consumer exercises extension seams |
| S6 Extension MVP | Renderer feature registration/storage/phase contribution; independent Rust Feature and FramePipeline | External Feature, external FramePipeline, and custom material cooperate in the same frame; first product MVP closes |
| S7 Rendering expansion | Material/Shader/Renderer and selected Features; Code Nodes, PBR, Forward+/Deferred, shadows/PostFX | A second pipeline shares scene/material contracts; materially different Features validate broader extension seams |
| S8 Measured strategies | Renderer resource/execution optimization; selected LOD, streaming, GPU culling, indirect work, batching/aliasing | Representative workloads demonstrate improvement with bounded memory and preserved lifecycle correctness |

```text
S0 baseline selection -> S0 asset integration + S1 Graph -> foundation closure
    -> S2 Shader -> S3 Material + S4 RenderScene -> S5 Forward -> S6 product MVP
    -> S7 selected expansion -> S8 measured optimization
```

S0 and S1 form one foundation closure. After dependency selection and assignment
of the integration artifact, independent tasks may overlap. S3 and S4 may overlap
after S2; scene storage can use logical material references while S3 finishes,
but S5 acceptance consumes both actual implementations. Feature storage and phase
seams begin in S4/S5 with a small external prototype. S6 integrates and validates
them; do not freeze Forward APIs and then retrofit extensibility at S6.

Measurements start with the first runnable workload. S8 selects optimizations
supported by those measurements; basic batching, bounded frame storage, and
lifecycle correctness must not wait until S8. Later stages are refined around
actual consumers rather than implemented as an exhaustive feature list.

## S0. Foundation and migration baseline

**Owners:** Rendering integration; Bases and RHI own their contracts.

Reuse `AssetId<K>`, immutable `ContentGeneration`, snapshots, producer coordination,
and CPU cache policy in `fluxel-assets`. Callers retain source lookup, loading,
decoding, retry, cancellation, hot reload, and streaming policy; platform services
supply I/O. Do not build another asset core.

Select an exact stable RHI revision and features, plus exact Bases and applicable
Host/JSBridge revisions. Inventory affected manifests, locks, examples, WASM
capsule, public consumers, and retained evidence. The current `0.16` dependency
and fixed recipes are migration inputs. Migrate affected consumers together or
record an explicit reviewed support decision; silently dropping an existing
target is not a migration strategy. RHI maintenance stays in its independent repo.

Rendering owns the planned evolving artifact `examples/rendering-integration/`
and its own `Cargo.lock`. It starts as the S0 producer/consumer fixture and grows
through S6. The planned independent consumer `examples/external-renderer/` has
its own manifest/lock and imports only the public Rust facade. These paths are
implementation destinations, not claims that examples exist. Reuse available
provenance helpers; retain evidence at
`target/integration/<stage>/<rendering-sha>/manifest.json`. Record participating
revisions, lock hashes, commands, inputs, targets, outcomes, and artifact paths.
A new generic evidence framework is not an entry gate.

Before changing the pin, record the selected revisions, runnable artifact, owner,
and target matrix in its tracked documentation. Distinguish RHI device provenance
from surface configuration and backend-exposed generations. Use the selected RHI's
actual portable types; do not invent a public device-generation field or copy old
`DeviceIdentity`, `FrameAttachment`, or browser-session APIs into the new path.

**Acceptance:** build the real producer and consumer from pinned inputs; replace
a snapshot under one asset identity, upload/use its contents, and observe the
completion permitting retirement. Reject foreign/stale resources and failed
uploads structurally. Record unsupported/untested routes. The migration inventory
must explicitly retire the public browser session/token resource path as affected
consumers migrate. This plan does not claim that migration is already complete.

## S1. Renderer-internal Graph and RHI execution

**Owner:** `fluxel-renderer::graph`.

Move useful standalone Graph implementation, tests, and diagnostics into Renderer
without maintaining two engines. Preserve typed pass-local access, resource
versions, definedness, hazards, deterministic DAG ordering/culling, transient
lifetimes, and diagnostics. Graph does not interpret scenes or materials.

Use direct RHI resources and fixture shaders. Exercise raster, copy/clear, and
compute where supported; unavailable compute/storage returns structured refusal
on the selected GL-family route. Full Shader/Material authoring is unnecessary.
Use RHI's portable recording, explicit transitions, dependencies, allocation,
submission, and completion. Native command storage and native barriers remain
backend-owned. Dedicated allocations and one ordered lane are valid baselines.

**Acceptance:**

- Run the fixture on DX12, Vulkan, and a named GL-family route, retaining inspected
  output and validation evidence. Re-run affected browser consumers under S0's
  matrix. One GL route does not certify another.
- Reuse a compiled plan over 2–3 outstanding frames with distinct completion
  evidence and safe per-frame storage. Dropping caller handles or replacing assets
  cannot cause early reuse. Per-frame device-idle is not this proof.
- Separate stable graph definition, device-specific plan, and per-frame bindings.
  Change resource instances and compatible material-like read sets while reusing
  a plan. Revalidate ranges, aliases, imported states, and producer dependencies.
  A new hazard/dependency or incompatible layout/access requires recompilation or
  refusal before recording. Include a newly aliased binding that invalidates the
  plan. Changed resource identity alone does not imply changed graph topology.
- Cover clean rejection, partial acceptance, partial immediate issue, and
  indeterminate native effects using the selected RHI outcomes. Finish execution
  as required even after a recording error. Retain original outcome payloads and
  affected resources; publish outgoing states only where evidence establishes
  them. Unknown effects prohibit reuse until the RHI contract supplies a safe
  disposition. Never retry a complete plan as though its accepted prefix had no
  effect. Deterministic fault injection supplements real-target execution.
- Reject foreign-device or stale resource/surface bindings, invalid imports, undeclared
  resources, incompatible uses, and unavailable capabilities before their backend
  effects. Device replacement invalidates device-specific plans/realizations while
  preserving logical asset snapshots.

Missing portable lowering facts trigger a bounded consumer-led RHI request,
rather than a duplicate capability model. Optional aliasing and multi-queue
optimization do not block foundation closure.

## S2. Independent Shader Core

**Owner:** new `fluxel-shader`; see [Shader design](design-shader.md).

First run complete WGSL vertex/fragment and compute programs without Material.
Shader owns composition, logical Naga validation/reflection, interfaces, source
artifacts, requirements, provenance/diagnostics, and compilation identity. A
standalone caller can assemble its generic request. Neither Material nor Renderer
is a dependency.

The initial artifact contains validated composed WGSL and logical metadata. Adapt
it to the selected portable RHI shader/pipeline API. RHI owns the opened device's
shader route, target ABI, backend Naga lowering, native compilation, and executable
objects. Do not reproduce its backend compiler in Shader. Distinguish source/
interface caches from device pipelines; source dependencies, entry points, static
choices, composition options, and applicable target/ABI assumptions determine
invalidation at the owning layer.

**Acceptance:** execute a textured raster fixture and supported compute fixture
through S1. Retain output/readback, reflected resource interfaces, and compiler/
runtime diagnostics. Cover generic source-unit composition, metadata/reflection
mismatch, missing entry points, invalid bindings, unsupported routes, and cache
invalidation. Record actual RHI routes, features, and compiler inputs; successful
Naga validation alone is not GPU execution evidence.

## S3. Minimal Material Core

**Owner:** new `fluxel-material`; see [Material design](design-material.md).

Implement Rust-defined functions/fragments, typed parameters and uniform/resource
schemas, a minimal call graph/Material IR, selected surface outputs, evaluation
WGSL, material asset references, instances, and static variants. Reuse Bases
asset identity plus content generation. Dynamic values share compiled variants.

Material emits generic Shader-owned source/interface contributions. The consumer
assembles the complete `ShaderCompositionRequest` from those contributions and
its pass template, entry points, geometry interface, and target/static requirements.
Renderer/FramePipeline does this in the rendering path; the S3 fixture can do so
independently. Material never needs Forward/Deferred knowledge, and Shader never
imports Material IR. WGSL/Naga remain the executable language/IR.

**Acceptance:** several material functions and parameters produce evaluation WGSL
that S2 composes and executes through S1. A custom function visibly affects output.
Reject invalid calls, cycles, metadata/type/binding mismatches, unsupported outputs,
and missing fragments. Prove static invalidation, dynamic sharing, and replacement
under a new content generation.

Full authoring graph breadth, Code Node authoring experience, PBR, and importers
are not prerequisites. Code Nodes are planned S7 functionality; Blender tooling
remains undecided. A small validated call graph suffices for S3.

## S4. RenderScene and frame snapshots

**Owner:** Renderer `render_scene` and initial typed Feature storage seams.

Implement generational RenderIds, dense render-only records, views/cameras,
resolved transforms/bounds, logical geometry/material references, typed batch
changes, and a frame snapshot boundary. Adapters own external identity mappings.
Do not introduce a gameplay hierarchy, ECS query language, universal object layout,
or general system scheduler.

**Acceptance:** run multi-object/view batch fixtures; verify incremental updates,
snapshot stability, empty batches, stale IDs, removal/reuse, conflicting changes,
and capacity refusal. Establish deterministic update/error semantics and dirty
ranges without copying the full world every frame. Retain ordinary and ECS-style
adapters without an ECS dependency. Prototype external typed Feature input before
freezing storage APIs. A runnable CPU consumer closes this data contract; GPU
integration closes at S5/S6.

## S5. FramePipeline and Forward integration

**Owner:** Renderer `pipeline`, `preparation`, `culling`, `sorting`, and resource
preparation modules. This produces an MVP candidate, not the product closure.

Expose a public Rust `FramePipeline` trait and controlled context. Graph compiler,
allocator/executor internals and independent submission remain private. The
pipeline selects phases/topology, resolves material/shader/pipeline requirements
through renderer services, and builds passes. Graph schedules declared work.
Basic visibility, deterministic phase ordering, and compatible batching operate
on typed batches, not per-object virtual draw calls.

Develop minimal material-backed Forward and the external Feature phase prototype
together. Check unavailable phases and incompatible shader requirements explicitly.
Do not force GPU-generated work to become a final CPU DrawList. Implement only
seams exercised by the selected consumers.

| Data/work | Owner and lifetime |
| --- | --- |
| Static mesh, texture, streamed terrain tile | Persistent per-device residency keyed by asset/content identity |
| Per-frame transforms, bone palettes, particle input updates | Frame Data Upload and frame-owned ranges retained through last use; not durable residency assets |
| GPU skinning, simulation, culling, indirect generation | Declared Graph passes |

The first two may share low-level upload allocation/copy machinery. Uploads outside
Graph are limited to renderer-controlled CPU-data preparation and export concrete
ranges, incoming states, producer dependencies, and completion into Graph/RHI before
consumption. Order overwrites after prior uses or allocate fresh storage. Waits do
not replace required memory visibility/state transitions. Pending content may be
usable by a particular dependent graph without being globally completed. Features
request preparation and never submit.

**Acceptance:** multiple meshes/materials render through actual Scene ->
FramePipeline -> Graph -> RHI. Exercise incremental transforms, persistent asset
replacement, frame-data reuse, overlapping frames, and failure/loss paths. Validate
every material binding against Graph declarations. Preserve plan reuse for
compatible bindings and reject/recompile new hazards. An independent crate compiles
the public pipeline/Feature prototype; the combined runtime proof is required at S6.

## S6. External extension and product MVP closure

**Owner:** Rendering integration and an independently built Rust consumer.

Use `examples/external-renderer/` with its own manifest/lock, normal public
dependencies, and no workspace-private hooks. It supplies an external FramePipeline,
typed RenderFeature, and custom material contribution. The integration artifact
must run all three together with the built-in mesh path in one frame. Separate
successful demos do not establish their composition.

**Acceptance:**

- The Feature owns typed storage, accepts batches, joins a selected phase, and
  declares all accesses without a core object-type switch or independent submission.
  Its custom material uses normal composition; the external pipeline controls policy.
- Retain inspected multi-frame output, exact revisions/locks, real producer/consumer
  builds, and the affected target matrix. Exercise visibility, ordering/batching,
  compatible binding refresh, and safe retirement.
- Reject unavailable phases, incompatible shader/layout/range/access requirements,
  missing declarations, foreign/stale resources, and preparation failure. Demonstrate
  a pending upload with an enforceable dependency and frame data that cannot be
  recycled before its final consumer completes.
- Stabilize only the demonstrated extension subset. Full terrain, water, particles,
  skinning, general cross-pass extensions, dynamic DLL loading, and a stable binary
  plugin ABI are not MVP gates.

## S7. Material, pipeline, and Feature expansion

**Owners:** Material/Shader semantics; Renderer and selected Feature algorithms.

Extend Code Nodes and source diagnostics, material lowering/variants, PBR,
Forward+, shadows, and PostFX incrementally. Prove Deferred as a second
FramePipeline sharing scene/material contracts. Code Nodes use the same WGSL
composition/reflection path; they are planned functionality.

Select a second materially different external Feature, such as skinning or terrain,
to exercise cross-pass and shader/material extensions before those stabilize.
Lighting extensions must affect shading through the defined seams; graph-only
hooks cannot establish that capability. Water, particles, vegetation, and area
lights are candidate workloads, not an exhaustive stage-exit checklist.

**Acceptance:** shared scene/material inputs work under Forward and Deferred; the
selected external Feature works without changing core object dispatch. Declare
compatibility/refusal per pipeline/backend and rerun affected external consumers
when their extension contracts change.

## S8. Measured performance and resource strategies

**Owner:** Renderer integration; callers/platform services keep source policy.

Use retained workloads to select residency budgets, streaming/LOD, GPU culling,
indirect work, batching, aliasing, multi-queue execution, or parallel recording.
Each needs its own evidence. Include a material-heavy workload (for example,
10,000 draws over 1,000 textures) to measure manifest validation, binding refresh,
plan reuse, preparation cost, and allocation/upload volume.

**Acceptance:** record baseline/candidate inputs, CPU/GPU measurements where
available, memory/upload/allocation counts, compiler/runtime identity, raw results,
and declared thresholds. Demonstrate bounded memory, replacement/eviction while
old data remains in flight, cancellation, upload failure, and device recreation.
Keep CPU asset caches, GPU residency, frame data, and graph transients distinct.
Retain the simpler implementation when an optimization has no measured gain.

## Undecided scope and future engine runtime

Capture/replay, Canvas 2D/basic text, declarative UI, Blender tooling, and JavaScript
wrappers remain undecided, with no delivery order or retained API design. A future
Game Scene, TypeScript gameplay runtime, game loop, Rust/TS bindings, and script
hot reload need separate planning. Preserve the possible adapter boundary:

```text
Gameplay / Game Scene -> Scene Adapter -> Batch ChangeSets -> RenderScene
```

`fluxel-jsbridge` adapts browser/mini-game platform lifecycle; it does not become a
gameplay VM. No future engine crate is authorized or required by this plan.

## Evidence and completion

Follow the ecosystem [evidence policy](https://github.com/fluxel-project/.github/blob/main/EVIDENCE_POLICY.md).
Run focused checks during implementation and the complete affected target matrix
at closure. There is no fixed quota on full verification runs. Tests, compilation,
documentation alignment, and old captures cannot certify a new GPU path.

RHI's [open work](https://github.com/fluxel-project/fluxel-rhi/blob/main/TODO.md) and
[verification limits](https://github.com/fluxel-project/fluxel-rhi/blob/main/documents/verification.md)
remain explicit. Missing Apple hardware evidence leaves the applicable Metal claim
open without blocking independent work. Record actual support/refusal per route;
do not inherit a general claim from a backend name or old RHI construction draft.
