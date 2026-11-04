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
External adapters -> RenderScene snapshot + typed Feature data
        -> FramePipeline policy + Feature-local preparation
           (geometry / Material / Shader resolution)
        -> prepared phase work + complete resource-access declarations
        -> internal RenderGraph -> portable RHI recording / submission
        -> private backend execution -> completion-driven retirement
```

- `RenderScene` is render-domain scene data: no gameplay logic or GPU objects.
- `FramePipeline` decides how this frame renders: Forward, Deferred, tile,
  path-tracing, or a custom strategy.
- `RenderFeature` contributes batches and GPU work using locally selected
  algorithms. Preparation resolves geometry/program interfaces, phase needs,
  concrete bindings and accesses before Graph compilation.
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
        -> material evaluation WGSL + generic source/interface contribution
        -> FramePipeline adds pass template, entry points, geometry and target facts
        -> ShaderCompositionRequest -> Shader System
        -> ShaderArtifact / Pipeline requirements
```

This is a compilation and generation relationship, not the frame-execution
path above. A compiled material variant is runtime input to the pipeline; its
authoring graph is not walked by a pass.

Material owns material function definitions/references, typed parameters,
uniform schemas, resource requirements, and function-call composition. It
generates evaluation WGSL and generic Shader-owned source/interface vocabulary
from its own IR. `FramePipeline` combines that contribution with its pass
template, entry points, geometry interface, static choices, and target
requirements to make one `ShaderCompositionRequest`. Shader owns complete
program composition, logical validation/reflection, source artifacts, and their
cache identity; it never receives `MaterialIr` or depends on Material. The
opened RHI device then selects `ShaderRoute`/`ShaderTarget` and performs any
backend Naga or native lowering, compilation, and ABI validation. This is a
conceptual compilation flow, not the per-frame call chain.

A shader module may contain several entry points; a GPU pipeline selects the
applicable stages and fixed-function state. A Graph pass is an execution node,
not a program: raster work contains draws with VS and optional FS and may switch
graphics pipelines; compute work contains CS dispatches; copy work needs no
shader. `VS + CS + FS` is not one ordinary GPU pipeline. MaterialGraph describes
value/function composition; RenderGraph describes GPU pass/resource dependencies.

## Ownership DAG and portable contracts

The architecture is an ownership DAG, not a single crate dependency chain.
The target has three principal crates: `fluxel-shader`, `fluxel-material`, and
`fluxel-renderer`. RenderGraph and the `FramePipeline` SPI are internal renderer
modules. The standalone graph crate and old WASM capsule have been removed;
their absence does not establish completion of the replacement. The intended
semantic ownership relationships are:

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
| `fluxel-renderer` | RenderScene, views, Feature/FramePipeline facade and reference consumers, preparation protocols and resolvers, residency, frame data, retirement, internal Graph | Mandatory visibility/sorting/geometry algorithms, gameplay/ECS, platform I/O, backend objects, host policy |
| Renderer `graph` module | Logical resources/versions, dependencies, definedness, hazards, lifetimes, transitions, deterministic plans | Scene/material semantics, asset policy, backend allocation mechanisms |
| `fluxel-material` | MaterialGraph, material functions and later code nodes, typed composition IR, evaluation-code generation, parameter/uniform schemas, instances, material variants, generic Shader contributions | Renderer dependency, graph passes, submission, executable shader IR |
| `fluxel-shader` | Generic source/function interfaces, complete program composition, logical WGSL/Naga validation/reflection, artifacts, variants, source-level compilation caches | MaterialGraph/MaterialIr semantics, scenes, rendering strategy, backend-private ABI or compiler ownership |
| `fluxel-rhi` | Portable GPU resources, capabilities, device shader-route selection and target ABI, backend lowering/compilation, recording, submission, completion, presentation, device lifecycle | Renderer policy, graph declarations, gameplay/asset policy, native objects in its public API |

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

The core is trait-extensible at batch and policy boundaries. Applications compose
Feature-local visibility, geometry, ordering, batching and deformation providers;
there is no required global algorithm singleton or per-object virtual draw loop.
Different Features can use BVH, quadtree and GPU visibility in the same frame.
Typed Feature storage participates in the snapshot boundary without a universal
Mesh record or an ECS dependency. SoA, AoS and AoSoA remain independent storage
choices; the public exchange contract does not freeze those layouts.
Extensibility uses static dispatch exclusively: generic concrete providers and
typed Feature sets are assembled at compile time. No trait-object registry,
`dyn Trait` or runtime type erasure is permitted. Runtime enablement uses state;
implementation switching uses explicit enum branches over compiled types.
Algorithm traits normally belong to Features or external crates; Core fixes
only necessary cross-Feature protocols. See
[ADR-0026](adr/0026-static-extension-composition.md). Runtime scene data and
Graph topology remain data, not per-object type parameters.

Geometry describes streams, attributes, indices and draw ranges. Preparation
matches those inputs with Shader requirements, including interleaved, separate
and mixed streams. External animation produces evaluated pose/transform data;
deformation Features consume it through frame inputs and declared GPU work.
Phase policy coordinates ordering across Features: separately sorted transparent
lists cannot simply be concatenated. GPU products are consumed through Graph
resources and dependencies, with no mandatory CPU visibility-list conversion.
See [the Renderer design](design-renderer.md) and
[ADR-0025](adr/0025-composable-rendering-policies.md) for these contracts.

Features operate on batches and own typed data. FramePipeline selects phases
and strategy; Features contribute packets, shader/material requirements, and
graph-visible GPU work. Object sorting differs from Graph dependency scheduling.
GPU-generated indirect work need not become a final CPU DrawList. All frame
rendering, simulation, and compute accesses stay under Graph authority.
Persistent residency uploads realize immutable meshes, textures, and streamed
tiles per device. Per-frame transforms, bone palettes, and similar frame inputs
use a separate frame-data upload lifetime, although both paths may share a
low-level transfer implementation. Each path exposes resource ranges, incoming
states, producer dependencies, visibility requirements, and retention evidence
to Graph/RHI. A pending producer may feed a specific consumer only through an
enforceable RHI dependency; it is not globally complete. Features request this
preparation rather than submit independently. GPU skinning, simulation, culling,
and indirect generation are Graph passes. See [the upload boundary](design-renderer.md#persistent-residency-frame-data-and-frame-work).

Reflection describes shader binding usage; preparation resolves it to concrete
resources, ranges, and access modes. Pass declarations must include the complete
binding resource-use manifest, not just a binding-object handle. Graph validates
coverage before recording and includes those resources in dependency/lifetime
analysis, including when dynamic bindings change.

Renderer owns the `FramePipeline` module. S5 supplies a bounded public Rust
trait/context facade, exercised by built-in Forward and independent concrete
pipeline/provider implementations. Contributions declare accesses on the real
Graph on every frame, including cache hits; compatible plans may be reused only
after the current declarations and actual bindings are validated. Graph compiler,
scheduler, allocator and submission internals remain private. This bounded seam
does not yet establish S6's heterogeneous GPU Feature composition.

The MVP closes only when the built-in mesh path and an independent Rust crate
use the same frame: a custom material, an external `RenderFeature`, and an
external `FramePipeline`. The public trait, context facade, handles, and errors
used by that crate are the supported extension surface; graph compiler,
scheduler, allocator, RHI submission, and backend objects remain private.
Feature/phase seams are shaped and compiled against a small external prototype
during renderer work, rather than added after Forward is complete. Cross-pass
and shader-extension contracts require a second materially different Feature
consumer before stabilization. Plugins are concrete Rust crate implementations
assembled at compile time; runtime loading of unknown Feature types is excluded.

## Migration and delivery

The workspace contains `fluxel-renderer`, `fluxel-shader`, and `fluxel-material`.
Shader and Material remain scaffolds. Renderer now has initial Copy, offscreen
and Surface Raster, supported Compute fixtures, a private buffer/texture Graph,
bounded completion-owned submissions, and an independent consumer. Scene and
extension surfaces remain unimplemented. Graph is private
to Renderer; the standalone Graph and WASM capsule directories are removed.
The reset inventory records current progress. Every support claim must prove
the selected composition rather than treating removed commands or evidence as
current.

Reuse Bases, establish internal Graph/RHI execution, then prove a minimal
WGSL/Naga-to-RHI shader compilation path before completing material authoring.
Deliver the Material-driven Renderer MVP with a custom material, externally
implemented FramePipeline, and external Feature. Material/pipeline
expansion and measured performance/resource strategies complete the active scope.
The five undecided features are listed in the product plan without design or
sequencing commitments. The
[implementation plan](version-plan.md) supplies the evidence gates.

## Graph and RHI integration

```text
FramePipeline
        -> RenderGraph definition / compile
CompiledGraph + per-frame RHI bindings
        -> RHI SubmissionPlan + explicit transitions
        -> RHI execution: deferred recording or immediate issue
        -> finish outcomes + completion evidence
```

RenderGraph compiles against the current RHI device and its authoritative
portable capability facts. If a device is lost or rebuilt, recompile the graph.
Avoid speculative compatibility fingerprints and cross-device reuse until
profiling proves their value.

A stable Graph definition describes topology and resource-use contracts;
`CompiledGraph` caches a device-specific execution plan; per-frame bindings
provide actual resources and dynamic values. Replacing many material textures
or buffers can reuse that plan when layouts, declared access modes/ranges,
aliases, and dependency contracts remain compatible. Instantiation revalidates
each concrete binding manifest. A topology, dependency edge, incompatible
resource use/range, or binding-layout change recompiles or fails structurally;
resource-instance churn alone does not. This makes no complexity promise and
does not require frame topology to stay stable.

Graph imports may be logical slots for reusable definitions, but they bind
directly to portable RHI resources. RHI/graph validation checks the selected
RHI device provenance, descriptor compatibility, allowed usage, known state,
pipeline/binding compatibility, and presentation lifetime. A renderer cache
may add a private generation only where the selected RHI exposes one; it must
not invent a public `DeviceGeneration` contract. An acquired `SurfaceImage` is
an exclusive presentation lease; `AttachmentView` distinguishes its surface
variant from an ordinary texture view.

RenderGraph owns logical lifetime, alias eligibility, portable transitions, and
GPU dependency planning. It may choose a portable allocation placement only
through the RHI allocation contract. RHI supplies allocation requirements and
realizes the selected placement or dedicated fallback; each backend selects and
lowers its native barriers, alias operations, heaps, and synchronization. A
wait alone does not establish memory visibility. Dedicated transient allocation
is the correct fallback; aliasing is optional.

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
[material design](design-material.md), [Shader design](design-shader.md), and
[RHI design](https://github.com/fluxel-project/fluxel-rhi/blob/main/crates/core/DESIGN.md) for layer-specific detail.
