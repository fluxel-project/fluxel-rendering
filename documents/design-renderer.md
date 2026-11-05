# Fluxel Renderer Design

> Status: target architecture with a closed bounded S4 CPU scene and feature
> data boundary. The current S0/S1 consumer uses fixed fixtures. `RenderScene`
> changes, immutable snapshots, and the provisional CPU-only `CpuFeature` facade
> are independently evidenced. S5 closes bounded Forward preparation, residency,
> frame data and a public FramePipeline facade for the named Windows DX12/WGL
> matrix; see the [S5 closure record](s5-forward-closure.md). S6 closes the
> demonstrated typed opaque-Forward Feature composition subset; S7 additionally
> closes a named Deferred one-mesh subset on that matrix. S8 closes one private
> validation-index optimization on a material-heavy Forward workload, while
> preserving Graph and owner-validation authority. General GPU products
> and cross-pass Features remain consumer-gated.
> The organization roadmap
> owns delivery order; [the implementation plan](version-plan.md) owns local
> scope and acceptance. This design does not authorize implementing later-stage
> APIs before their consumer gates.

## Core decision

Renderer is a data-oriented rendering framework with Rust trait extension
boundaries. Applications compose algorithms; Features process batches;
FramePipeline coordinates frame policy; the internal RenderGraph plans GPU work.
Core fixes the exchange contracts, authority, and lifetime rules. It does not fix
an object model, vertex structure, spatial index, animation system, or renderer
algorithm. [ADR-0025](adr/0025-composable-rendering-policies.md) records this decision.
The extension architecture uses static dispatch exclusively, as specified in
[ADR-0026](adr/0026-static-extension-composition.md).

```text
External world / ECS / scene graph / animation system
    -> batched scene and Feature changes
    -> RenderScene frame snapshot + typed Feature storage
                         |
       FramePipeline <-> Feature preparation
       phases / targets   local algorithms + Material / Shader resolution
                         |
           prepared phase contributions + GPU producer declarations
                         |
               internal RenderGraph
               accesses / dependencies / transitions / execution plan
                         |
                  portable RHI recording and submission
                         |
                  completion -> safe reuse / retirement
```

Preparation and frame policy cooperate before Graph compilation. This is not a
mandatory linear sequence of CPU culling, CPU sorting, then CPU draw-list
construction. A Feature may describe GPU work whose results never return to CPU.
Material compilation is a separate path that supplies program inputs to this
process; MaterialGraph is not the frame graph.

## Workspace and module boundaries

The target keeps three crates. No separate RenderGraph, FramePipeline, plugin
registry, or general ECS crate is needed to establish these boundaries.

| Crate | Owns |
| --- | --- |
| `fluxel-shader` | Generic source composition, logical validation/reflection, interfaces, variants, artifacts and their identity |
| `fluxel-material` | Material functions/graphs, private composition IR, parameters, instances and generic Shader contributions |
| `fluxel-renderer` | Rendering integration, scene/Feature exchange, preparation, frame policy, private Graph, residency and frame lifetime |

Material depends on Shader vocabulary; neither depends on Renderer. Renderer
combines their products with geometry and pass requirements. RHI owns opened
device routes, native lowering/compilation, executable pipeline validation,
resources, recording, submission, completion, and presentation.

| Renderer module | Stable responsibility | Replaceable choices |
| --- | --- | --- |
| `renderer` | Entry facade, service ownership, lifecycle and error coordination | Application configuration |
| `render_scene` | Render identities, batched updates, views and snapshot boundary | External world adapter and Feature storage layouts |
| `feature` | Typed Feature registration, preparation and phase contributions | Mesh, terrain, particles, skinning and other contributors |
| `pipeline` | Controlled public facade, phases, attachments and frame outputs | Forward, Deferred or application frame policy |
| `preparation` | Batch preparation protocols, resource and interface resolution | Visibility, ordering, batching, geometry realization strategies |
| `graph` | Versions, definedness, hazards, plans, pass-local access and RHI execution | Internal optimizations, subject to correctness evidence |
| `residency` | Per-device persistent realization and last-use retention | Requested content and caller streaming policy |
| `frame_data` | Frame-owned input ranges and completion-safe reuse | Transform, palette, instance and other frame data formats |

These are ownership boundaries, not eight mandatory plugin objects. Modules
remain private; implemented public traits, contexts, handles and errors are
re-exported through the crate facade. Additional files or crates require an
actual consumer. Unit tests belong in `src/tests/`; public integration tests
belong in crate-root `tests/`.

## Trait and plugin policy

There are two levels of composition:

1. Renderer-owned extension contracts allow external Rust crates to implement
   `RenderFeature` and `FramePipeline`, exchange typed batches, and declare work
   through controlled contexts.
2. A Feature or Pipeline composes local algorithm providers using Rust generics,
   traits and monomorphization. Visibility, geometry access, sorting, deformation
   and batching traits normally belong to their Feature or external crate.
   Core admits a shared protocol only when cross-Feature cooperation actually
   requires it and consumers prove its input, output, lifetime and refusal rules.
   Multiple users of an algorithm do not by themselves make its trait a Core API.

This is a commitment to replaceable local algorithms, not permission to make
algorithms inseparable from core. A user must be able to replace a provider from
another crate without editing Graph or core object dispatch. It also does not
require every implementation detail to have a public trait.

| Candidate seam | Scope and exchange | Admission rule |
| --- | --- | --- |
| Scene adapter | External changes -> typed batch updates | Plain slices/ChangeSets suffice initially; add a source trait only for needed lifecycle behavior |
| Geometry provider | Batch logical references -> immutable stream/interface descriptions | Batch resolution through concrete provider types |
| Visibility provider | Feature/view candidates -> CPU batches or declared GPU products | Multiple providers may coexist in one frame |
| Ordering / batching policy | Phase-compatible contributions -> ordered/merged work | Preserve phase-wide ordering, especially transparency |
| Deformation provider | Geometry + pose/morph inputs -> vertex-stage inputs or declared CPU/GPU products | Animation simulation stays outside Renderer |
| RenderFeature | Typed data + views + allowed services -> phase/work contributions | External crate exercises the public surface |
| FramePipeline | Views + Feature requirements -> phases, targets and graph composition | Shared machinery, no private execution escape |

These names describe responsibilities, not approved Rust signatures. Calls use
static dispatch at Feature, view, phase or batch boundaries; no `Mesh::draw()`
or `RenderObject::cull/sort/render()` object loop is required. Renderer does not
need a distinct generic type for every object, mesh layout or material value.

Plugins here are ordinary Rust crate implementations assembled by the caller.
Registration and dependencies are explicit and deterministic. Missing or
incompatible phase/resource/interface requirements return structured errors;
registration order must not silently decide correctness. A dynamic DLL ABI,
hot-unloading protocol, global service locator and arbitrary string-keyed data
bus are not part of this design.

Providers exchange logical descriptions, prepared products and controlled
resource requirements. Public opaque Graph references are scoped to their
definition/frame; they are not durable scene identities or permission to retain
internal frame bindings. Context borrows and surface leases must not escape their
declared lifetime. Backend-private resources are never provider inputs. Persistent
portable resources enter through the validated import/residency contract.

## Static extension composition

All RenderFeature, FramePipeline, geometry, visibility, sorting, batching and
deformation providers use Rust generics, trait bounds and monomorphization.
There is no `dyn Trait`, `Box<dyn Trait>`, `Vec<Box<dyn Trait>>`, trait-object
registry or runtime type erasure in the extensibility architecture. This applies
to both setup and frame processing, not merely to individual draw calls.

External crates supply concrete implementations. The application assembles a
concrete pipeline and Feature set at compile time. Ordinary structs, tuples,
traits and necessary associated types are sufficient starting mechanisms.
A Feature may itself be generic over its local providers. Heterogeneous Features
do not require an erased collection: a typed tuple or concrete aggregate can
implement the batch traversal contract through direct calls.

Runtime enable/disable is ordinary state in an existing concrete Feature. When
runtime implementation selection is required, an explicit enum selects among
known compiled implementations with statically resolved branches. This does not
permit arbitrary unknown Feature types to be loaded at runtime. Pipeline or
provider replacement still preserves all in-flight owners and snapshot rules.

Static types do not mean static scene contents or a fixed GPU graph. Object
counts, attribute descriptors, resource identities, material parameters, enabled
passes and resulting Graph topology can remain runtime data. Do not make each
asset, attribute combination or draw a new generic instantiation.

This guarantees no trait-object dispatch in Rendering's extension calls, not
zero execution cost or guaranteed inlining. Code size, compilation time,
allocations, enum branches and actual throughput remain measurable tradeoffs.
Share non-policy implementation where useful without erasing plugin types.
Do not introduce tuple metaprogramming frameworks, generated public APIs or a
general type-level scheduler before consumers prove a need.

Illustrations such as `Renderer<P, F>` and `FeatureSet` describe this composition
pattern, not final public signatures. Backend parameters, borrowing, errors and
GPU-product lifetimes must be established by implemented consumers. The rule
governs Rendering extensibility; it does not authorize changes to the selected
RHI or platform contracts.

## Scene data and ECS integration

RenderScene is a render-domain snapshot boundary. It contains common view and
identity facts plus typed Feature data; it is not a universal array of Mesh
objects. The reference mesh path may use a record containing a render ID,
transform, bounds, logical geometry/material references, visibility and ordering
hints. Other Features need not use that record or own conventional geometry.

The current public S4 subset is deliberately narrower: `RenderScene<R>` stores
consumer-chosen cloneable resource bundles beside records. `RenderResources<G, M>`
is a convenience bundle for pinned geometry/material asset references, rather
than a per-scene type restriction. The scene exposes generational record/view
identities, transactional `RenderChangeSet` updates, finite transforms/bounds,
exact dirty ranges, and immutable `FrameSnapshot` values. Its mutable records
are dense and removals compact them while the stable generational IDs and dirty
slot ranges remain meaningful. This does not claim that heterogeneous Feature
storage has stabilized.

Adapters push typed creates, updates and removals. They own the mapping from
external entities or scene nodes to generational renderer IDs. Renderer does not
query an ECS, require entity/component traits, walk gameplay hierarchies, or own
a general system scheduler. Render IDs are distinct from asset IDs, device IDs,
GPU addresses and dense storage indices. Removal/reuse must reject stale IDs.

Feature-owned typed storage participates in the same frame snapshot protocol.
Dense stores are the reference implementation, not a universal layout imposed on
an external Feature. Shared data uses explicit typed views and dependencies;
it does not require copying every field into both a core store and a Feature store.
Mutation enters at defined update boundaries. Snapshots retain their owners or
use bounded borrowing so later changes cannot invalidate in-flight preparation.
Changing a provider or removing a Feature does not retire its GPU resources.
Submitted resources remain retained independently until completion evidence.

The snapshot protocol specifies atomic/conflicting changes, errors, dirty ranges,
and identity mapping before public APIs stabilize. Incremental updates should not
require cloning or re-uploading the entire world. Parallel preparation, when
introduced, must have explicit read/write ownership; external ECS scheduling
must not leak into the rendering contract.

The independent `examples/external-renderer/` consumer currently proves this
subset on CPU: ordinary object-map and ECS-shaped adapters produce the same
public scene observations, an old snapshot remains stable through an update, and
two differently shaped application-owned feature snapshots are associated with
the corresponding scene versions. It implements the Renderer-exported static
`CpuFeature` trait with renderer-issued opaque handles, issuer-bound typed
storage, controlled `FeatureContext`, and `CpuPhase::Reference` contributions.
The facade checks scene, issuer, feature-handle, snapshot-version, collector
binding, and replay identity through structured refusal. It has no device,
Graph, submission, culling, indirect-work, `RenderFeature`, or `FramePipeline`
claim. Its compile-fail negative also checks that the external dependency cannot
name Renderer-private Graph state.

## Data-oriented layout

Data orientation means organizing work around the fields processed together.
It does not mean that SoA is always the correct representation.

| Access pattern | Useful initial representation | Constraint |
| --- | --- | --- |
| Visibility scanning bounds and flags | Dense columns / SoA | Avoid loading material and animation metadata for every bounds test |
| Processing most fields of a small prepared record | Compact AoS | Keep records contiguous and separate cold metadata |
| A measured vectorized chunk kernel | AoSoA | Choose chunk width from a real workload, not a public ABI assumption |
| Geometry on the GPU | Interleaved, separate or mixed streams | Match Shader input and selected RHI capabilities |
| GPU-driven visibility/draws | Buffers and indirect ranges | No mandatory CPU materialization or readback |

Public exchanges use typed batches, slices, immutable descriptors or opaque
resource products with explicit ownership. They do not promise a particular
container or expose a general `dyn Any` object world. CPU scene layout, CPU
asset layout and GPU vertex layout are separate decisions. Conversion/packing
has a measurable cost and should happen at a bounded preparation boundary.

## Geometry and material references

A mesh instance refers to geometry and material; it does not contain a required
vertex struct. Geometry describes data and interfaces. The reference geometry
path must express:

- one or more vertex streams, with ranges, stride and vertex/instance step rate;
- attributes with semantic identity, format and byte offset;
- optional indices and index format, primitive/submesh draw ranges;
- separate material assignments where a geometry has several parts; and
- optional deformation inputs without making all geometry skeletal.

A layout can be Position-only, Position/Normal/UV, or Position with custom
attributes. Position/Normal may share an interleaved stream while UV and
Joints/Weights live in separate streams. This is not permission to accept any
arbitrary format: validate ranges, strides, conversions, duplicate semantics and
the selected RHI's supported input limits before recording.

Semantic identity is not a shader location. Preparation matches geometry
attributes with the selected program interface, then produces a concrete vertex
input layout and pipeline selection. Custom semantics need explicit stable
identity; ad hoc names must not silently alias each other. Missing attributes
cause a structured incompatibility or an explicitly selected supported variant.
Do not silently invent normals, UVs or reinterpret formats.

Fixed-function vertex input is one consumption route. Later Features may use
vertex pulling, procedural generation, storage buffers or indirect work when
supported. They declare those accesses through Graph rather than pretending all
geometry is a fixed vertex buffer. These routes require their own consumers and
do not expand the RHI interface as part of this design exercise.

Asset-backed references preserve `AssetId<K>` and immutable `ContentGeneration`.
A mutable material instance may have a renderer-domain identity but retains its
material asset/content reference. Renderer-owned procedural inputs instead have
an explicit owner and revision/lifetime; they must not invent parallel durable
asset identities. RenderScene remains independent of RHI resources.

## Animation, skeletons and deformation

Animation is split by responsibility, not by one global `Animation` trait.

| Work/data | Owner |
| --- | --- |
| Clip sampling, state machines, blend trees, hierarchy and pose evaluation | External animation/application crates |
| Current/previous transforms, evaluated pose and morph weights | Batched input snapshots with explicit revision/identity |
| Static joints/weights, inverse-bind data, morph deltas | Logical geometry/deformation assets owned by the relevant Feature |
| Current palette, instance/deformation parameters | Frame data with retained lifetime |
| CPU deformation | Replaceable Feature preparation provider; output uses the normal upload contract |
| Vertex-shader skinning | Feature-selected program/interface with declared palette/geometry reads |
| Compute skinning/morphing | Graph passes producing streams/buffers consumed by later passes |

A higher-level crate can assemble different animation and deformation providers
for different objects. The selected deformation contract must state pose space,
joint mapping, palette format, previous-frame requirements and bounds validity.
Culling needs conservative valid bounds or a declared dependency on updated
bounds; a stale pre-deformation bound is not automatically safe. Renderer
coordinates these inputs and GPU work without owning a game animation runtime.

## Visibility can be local and heterogeneous

A frame may contain a static-mesh Feature using BVH, terrain using a quadtree,
particles using GPU culling and UI using 2D clipping. There is no mandatory global
`CullingService` that owns all of them. Providers may be shared explicitly by
compatible Features without forcing a single global algorithm or spatial index.

The common boundary describes a view, candidates and consumption requirements.
A CPU provider can return typed visible candidates/ranges. A GPU provider declares
passes producing visibility/compaction buffers, counts or indirect arguments.
The declaration exists during preparation; the values exist only after GPU
execution. The consuming phase uses graph resources and producer dependencies,
not a promised immediate CPU `VisibleSet`.

A GPU product names its graph version/instantiation, producer and consumers.
Keeping its handle is not evidence for cross-frame reuse. History or persistent
products require explicit import/export, dependencies and completion-safe owners.

The design does not freeze a universal visibility-result enum now. A supported
product protocol must identify its representation, scope, capacity, consumer,
accesses and capability requirements. Unsupported combinations are refused or
use an explicitly selected fallback. GPU-generated indices/counts need bounded
producer/consumer rules; CPU preflight cannot prove data-dependent shader results.

Visibility culling selects objects. Graph dead-pass culling removes unneeded GPU
work from declared roots. These are different operations with different owners.

## Ordering and batching are phase contracts

Features can select local ordering and batching providers, but a phase defines
correct composition across Features. Opaque work may permit state/material/depth
reordering. A conventional alpha-blended phase needs a shared depth-order policy
and deterministic ties across contributions. Appending separately sorted mesh,
water and particle lists is not generally a correct global order.

The phase owner therefore chooses a compatible merge/order protocol or an
explicit alternative such as separate compositing stages. A GPU-only contribution
cannot promise exact interleaving with a CPU transparent list without a supported
shared ordering/composition path. Unsupported combinations fail explicitly.
Feature-local transparent sorting is only a pre-sort; the phase owns the final
merge or explicitly chosen compositing order.

Batch compatibility includes selected geometry/program layouts, material/binding
requirements and phase semantics. Ordering draw work and scheduling GPU passes
remain distinct. Graph does not infer transparent sorting from resource hazards.

## Preparation and frame policy

Preparation is the integration boundary, not one hard-coded algorithm chain.
Its outputs describe:

- prepared phase contributions: CPU batches, or supported GPU product references;
- resolved geometry/program/pipeline interfaces and concrete bindings;
- required stages, attachments and semantic inputs such as scene color;
- complete buffer/texture access manifests and GPU producer declarations; and
- retained owners, frame ranges and external upload dependencies.

FramePipeline chooses rendering policy, phases, attachments and semantic resource
producers. A Feature contributes requirements and work within that policy. For
example, a material-backed water Feature may require scene color and depth;
the pipeline supplies supported versions or rejects the request. Material code
neither owns those resources nor decides where a copy/reflection pass runs.

Construction must have a bounded contract: collect requirements against a known
phase policy, resolve/validate the selected composition, finalize declarations,
then compile. Concrete passes and resources may reference producers declared
later in this construction, but nothing may add hidden work after compilation.
Do not introduce unbounded mutual callbacks or implicit iterative negotiation
between material selection and pipeline topology.

The following are hard boundaries rather than a prescribed implementation order:
each submission contributes to the actual Graph transaction exactly once,
including a compatible compiled-plan cache hit; declarations and the actual
owner/binding facts are validated before the upload, recording, or submission
operation they protect; and Graph remains the sole authority for GPU accesses,
dependencies, planning, and execution. A coordinator may prepare or allocate
resources before Graph declaration when CPU facts have been validated. It is not
required to build the whole Graph before every allocation. Private descriptors,
cache organization, callback signatures, and the exact preparation order remain
implementation choices until exercised consumers establish a public contract.

CPU preparation need not have a general graph scheduler. Callers can use ordinary
batch calls or their own CPU task orchestration. No `slot-graph` node identity or
ECS scheduling concept crosses the public renderer boundary. All GPU products
still enter the single RenderGraph.

## RenderFeature and FramePipeline authority

`FramePipeline` owns frame policy and composition. The bounded S5 Forward
consumer has passed its acceptance; S6 adds one independently built typed
`RenderFeature` consumer for an external Forward draw batch alongside built-in
meshes. The retained DX12/WGL matrix covers eleven offscreen submissions and a
two-frame native Surface resize workload, including pending replacement and
named refusal boundaries; see the [S6 closure record](s6-typed-forward-closure.md),
[measurement baseline](s6-measurement-baseline.md), and
[evidence index](evidence/s6-typed-forward-candidate/index.json). This closes
only that opaque composition subset. It does not establish general GPU products
or cross-pass Feature support.

`FeatureBinding` borrows a concrete Feature, its issuer and typed input.
Submission checks scene/issuer/handle/version before calling Feature preparation.
The Feature emits logical `ForwardDrawRequest`s with its own local identities,
asset generations, transforms and layer values, rather than synthetic scene
record IDs. Both sources use the same material/geometry validation and resource
realization. The pipeline orders the combined `ForwardDrawOrder` sequence; the
renderer checks that the result is an exact permutation before allocating its
GPU products. Adjacent compatible draws may then share a raster program without
changing that final sequence.

`ForwardCompositionContext` invokes the pipeline on the real opaque Graph pass.
The pipeline owns the common upload dependency and output write. Mesh and
Feature callbacks receive separate `ForwardFeaturePass` read scopes derived
from their actual resolved draws. Both callbacks must declare vertex, uniform
and texture reads, including empty batches. Scope completion and full draw
coverage are required before upload mapping, recording and submission. This
declaration runs again before plan selection on every frame, including cache
hits; no separate preflight graph or declaration replay is involved.

CPU Feature input and prepared values need to remain borrowed only through
preparation and declaration. Resolved asset snapshots, copied transforms/tints
and frame-owned upload bytes carry the data needed by GPU work. Those actual
owners use the existing pending/completion lifecycle; the Feature's CPU types
need no artificial `Send + Sync + 'static` requirement. The current composed
entry supports opaque Forward only and rejects an incompatible topology before
Feature preparation or GPU resource realization.

S4 currently exports a narrower, provisional CPU-only facade: `CpuFeature<R>`,
`FeatureIssuer`, opaque `FeatureHandle`, `FeatureContext`,
`TypedFeatureStorage<T>`, `TypedFeatureSnapshot<T>`, `CpuPhase`, and controlled
`CpuPhaseContributions`. A scene issues handles and contexts; storage retains a
typed payload for its issuing scene, issuer and snapshot version. The active
context/binding separately validates the consuming Feature handle. Same-typed
immutable payloads may be intentionally reused by Features of that issuer at
the same scene version; reuse does not transfer contribution or submission
authority. This establishes the exercised ownership
and structured-error boundary without exposing Graph, RHI, backend objects,
submission, or a mutable scheduler. It is not a `RenderFeature` or FramePipeline
contract and does not supply GPU work authority.

The shared `FramePipeline` facade now has a Deferred topology as well as the
bounded Forward topology. `ExternalDeferredPipeline` uses its controlled
Deferred context to make genuine Graph callbacks; it does not receive a mutable
Graph, allocator, backend object, encoder, or submission capability. The public
`submit_deferred` and `submit_deferred_with_deformation` entries retain the
same scene/material preparation contract as Forward. General imports,
raster/compute/copy work, roots, semantic outputs, and broader Feature authority
remain consumer-gated rather than implied by this facade.

This facade currently selects three built-in products: opaque Forward,
clear-only, and Deferred. It does not yet let an external pipeline define its
own pass chain, intermediate resources, or new GPU pass behavior. Forward and
Deferred integration evidence therefore does not establish a general
programmable render pipeline. A proposed next boundary probe is an independent
consumer that renders a scene, runs a custom post-processing pass, and produces
the final output without changing renderer internals. Such a probe must exercise
real Graph declarations, output pixels, reuse, and failure cleanup before a
broader public extension contract is selected; it is not an added release
commitment.

During recording, a pass uses scoped access to already declared resources and
portable RHI recording operations. Resource/binding validation must precede the
backend operations it protects, including on immediate routes. A pass may not
load assets, create undisclosed accesses, or discover a new graph dependency.
Prepared draw/dispatch descriptions are batch data, not a second portable command
language, replay interpreter or generated RHI source-code layer.

Reference mesh and Forward implementations use the same facade as external
crates. Feature contributions may expand to several ordinary passes; they do not
create hidden runtime subgraphs. Contexts grant only the authority needed for
the current stage. A missing phase, incompatible product or failed preparation
returns a structured owner-specific error before affected recording begins.
The coordinator must not substitute a validation-only callback or replay a
guessed internal topology on a cache hit: the caller's declaration itself is the
evidence being validated.

## Material and Shader integration

The shared Deferred route reuses the exact `FrameSnapshot<ForwardResources>` and
`ForwardAssets` consumed by Forward. `ExternalDeferredPipeline` contributes
through the real Deferred Graph context. The public `submit_deferred` entry
uses that pipeline policy; `submit_deferred_with_deformation` additionally
accepts one statically dispatched `DeferredPositionDeformer` with issuer- and
snapshot-validated typed input, a named WGSL `SourceUnit`, and four finite
parameters. The Feature cannot own GPU objects, Graph state, an encoder, or
submission.

The selected one-mesh, position-only product records five actual passes:
upload, compute deformation, Deferred geometry, lighting, and readback. The
renderer owns packed `array<f32>` source and product storage rather than using a
`vec3` storage stride; compute writes the 12-byte-per-vertex product consumed by
the later vertex pass, with no CPU deformation intermediate. Material continues
through MaterialCompiler and ShaderComposer, while lighting uses the same Shader
composition/admission path. Owners retain allocation descriptors, incoming
states, and device provenance through the existing completion lifecycle.

Focused witness tests
replace actual product owners with another same-device allocation and a foreign
device allocation, proving refusal before mapping or recording.

The Color route remains single-mesh and position-only. Its derivative normal is
in local/object space and its bounded lighting formula is not PBR. The separate
metallic/roughness consumer below does not change that Color behavior. Deferred
caching, Surface coverage, and broader material or Feature contracts remain open.
The C1 Material/Shader consumer is a bounded Code Node path. A typed
`MaterialCodeNode`, added with `MaterialDefinition::with_code_node`, joins the
same private function graph as legacy fragments. Identical named node and
definition resource declarations merge; incompatible declarations refuse. Source
units are immutable authoritative inputs: `source_units()` preserves each
author's identity, while derived `source()` and `source_id()` compatibility
getters cannot become an independent source of truth. `prepare_material` passes
those units to Shader composition and rejects invalid WGSL or callable signatures
before GPU work.

The external `ChannelMapVariant` exercised actual BGR×factor source variants
`[0.5, 1.0, 1.0]` and `[1.0, 0.5, 1.0]`. For each variant, both routes consume
the same exact scene and assets. Development attempts `dx12-20261104T162347864453Z-6652` and
`native-gl-wgl-20261104T162456696039Z-17624` retain 454 identical source inputs,
20 raw outputs, and actually viewed montages. Code Node cases observed
`0 -> 1 -> 0` pending ownership on each route, alongside deformation's
`0 -> 1 -> 2 -> 1 -> 0`; no pending native diagnostic remains. These positive
native observations and focused witness negatives have different scopes. They
do not establish PBR, a Deferred cache or Surface matrix, or S7 closure.

### Selected metallic/roughness consumer

C2 adds an opaque `SurfaceOutput::MetallicRoughness` consumer through the same
public submission and material preparation paths. Material supplies a zero-input
Vec4 base-color evaluator and a zero-input Vec2 metallic/perceptual-roughness
evaluator. Renderer consumes their results; Material does not select a pipeline.
The selected external material reads base RGB and metallic from its texture and
roughness from its dynamic tint alpha. These are author functions using the
existing uniform/texture schema, not hidden Renderer reads of material channels.

Both routes require exactly POSITION and NORMAL Float32x3 attributes in vertex
streams. Preparation validates actual CPU payloads, finite nonzero normals, and
a finite invertible affine world-from-local transform. The normal matrix is the
inverse transpose of the model's linear part; projection is never used as the
world normal transform. The transform binding contains two column-major matrices
(128 bytes), while Color retains its 64-byte binding. Uniform packing, layout,
program identity and Graph physical access ranges consume the same byte count.
Interpolation can cancel opposing normals; the shader maps a zero interpolated
normal to the unlit back-facing direction on both routes.

The private `pbr` module shares one direct-light calculation: GGX distribution,
height-correlated Smith visibility, Schlick Fresnel and Lambert diffuse weighted
by the remaining dielectric energy. The specular equations follow the
[Filament material model](https://google.github.io/filament/main/filament.html).
This selected policy uses world-space light direction `(0.6, 0, 0.8)`, radiance
`3`, and fixed orthographic view direction `(0, 0, 1)`. Base color and metallic
are clamped to `[0,1]`, perceptual roughness to `[0.04,1]`; alpha is opaque.
Inputs and output are linear, with no tone mapping or sRGB conversion. This is
not a perspective-view, configurable-light, IBL, normal-map or emissive model.

Deferred writes base RGB/metallic to its first RGBA8 attachment and encoded
world normal/perceptual roughness to its second, alongside Depth32Float. Lighting
reads those actual attachments, renormalizes the decoded normal, and calls the
same BRDF. Its UNORM quantization is part of the independently predicted result;
it need not match unquantized Forward pixels exactly. The position-only compute
deformer cannot supply the required normals and is refused for this surface.

Focused tests and separate DX12/WGL development attempts now validate the bounded
implementation. Five PBR cases change base color, metallic, roughness and a
rotated nonuniform model transform while sharing the exact material asset and
static program. Independent Rust and Python numeric oracles account for each
route's quantization; the representative native images have also been opened.
Actual short/retagged transform ranges and invalid normal inputs refuse before
protected work. C1 Color evidence alone cannot prove these results. Exact
attempts, evidence limits and remaining stage work are in `version-plan.md`.

Deferred now keeps device-local program and completed resident caches. Every
submission authors its current Graph declaration before a compatible compiled
plan can be selected; actual owner descriptors, provenance and incoming states
are checked again before mapping or recording. A partial resident hit uploads
only the missing products. An omitted upload requires the corresponding owner
to carry its actual initialized vertex-read or sampled-read state.

The accepted S8 private indices only accelerate lookup of a candidate binding
manifest and pass range for the material-heavy Forward consumer. They are not a
second scheduler or a substitute for Graph declaration. The Graph remains
authoritative for this frame's accesses and dependencies; the owner's saved
allocation descriptor, actual incoming state, and selected RHI device provenance
remain the facts checked before each protected allocation, upload, recording, or
submission operation. The [S8 measured closure](s8-measured-closure.md) records
the bounded DX12/WGL lifecycle and CPU/API elapsed evidence; it does not make
this private arrangement a public optimization contract.

Offscreen frame and G-buffer allocations enter the bounded slot pool after
completion and successful readback. Completion failure retains pending owners;
known device loss clears reusable caches and rejects new work. Program eviction
does not release cloned RHI executable handles held by pending submissions.
The settled cache path avoids static uploads but still allocates a small unused
resident staging buffer; allocation counters include it. The selected S7 closure also covers Surface and device isolation; see the
[acceptance record](s7-deferred-closure.md) for its exact source and evidence boundary.

Deferred also has a transient Surface output route. Both routes now author the
same upload, optional deformation, geometry and lighting Graph body; the final
output selects a texture/readback pair or a Surface attachment followed by an
explicit presentation step. A Surface lease never enters resident or plan
binding caches. Admission checks its device, current generation and view extent
before dependent device work, and the lighting executable uses its actual format.
Retirement requires completion evidence covering the accepted presentation.

The combined archived Surface baseline independently validates Color initial and
resized windows, two PBR roughness cases and two external position-deformation
parameters on DX12 and WGL. Each presented frame has a separate offscreen
companion using the same immutable scene, assets and applicable typed input.
Retained actual inputs support independent full stable-region pixel oracles for
both captures and raw readbacks. The later [S7 acceptance record](s7-deferred-closure.md) adds the formal
device/failure matrix and identifies the six exact native attempts.

The later formal matrix closes only the bounded S7 Deferred contract recorded in
the [S7 Deferred acceptance record](s7-deferred-closure.md): shared scene and
material inputs, optional statically composed deformation, selected CodeNode and
metallic/roughness cases, and cache, Surface/resize, device-isolation,
generation-replacement, pending, and refusal evidence on DX12/WGL. It does not
stabilize general PBR, arbitrary material/pipeline chains, or broader Feature
products.

```text
MaterialGraph -> private Material IR -> generic source/interface contribution
                                             +
Feature geometry/deformation interface + pipeline pass template / entry points
                                             |
                                  ShaderCompositionRequest
                                             |
                            composed program + logical reflection
                                             |
                       preparation: resources, RHI pipeline, access manifest
                                             |
                                          Graph
```

Material owns material semantics, functions, parameters and instances. Shader
owns complete logical composition, reflection and program identity. Renderer
owns their integration with geometry, pass policy and actual resources. RHI
owns executable device objects and backend lowering. Compute or post-processing
can use Shader without Material.

Material evaluation, program composition and preparation are not necessarily
repeated each frame. Cached products are reused according to their own identity.
A raster pass may draw several materials/pipelines; a material may appear in
several passes. No pass-per-material or pass-per-shader assumption is permitted.

| Change | Invalidation responsibility |
| --- | --- |
| Dynamic material value or resource instance | Refresh frame data/bindings and validate accesses; reuse compatible plans |
| Program/static variant or geometry interface | Re-evaluate Shader and executable pipeline identity; Graph depends on resulting access/attachment changes |
| Vertex stream packing only | Re-evaluate input layout and executable pipeline; logical Shader can remain identical if its interface is unchanged |
| New scene-color dependency, producer pass, attachment or incompatible access | Rebuild/reject Graph structure and revalidate the frame |
| Device replacement | Recreate device-local realizations/plans; retain old affected owners until their lifecycle is resolved |

Shader Reflection supplies logical requirements. Preparation resolves actual
resource identities, ranges, layouts, stages/accesses and RHI device provenance.
Graph validates complete concrete manifests before recording; it never guesses
shader accesses. Samplers are validated inputs but not memory hazards. A manual
manifest is sufficient for foundation fixtures.

## Graph definition, bindings and execution

Graph is a GPU-work compiler within Renderer. It owns logical resource versions,
definedness, RAW/WAR/WAW dependencies, explicit ordering, dead-pass culling,
transient lifetimes, portable transitions and deterministic execution plans.
It knows neither vertex semantics nor material, scene or Feature algorithms.

Keep definition, compiled device plan and per-frame bindings distinct. Every
frame validates current resources, ranges, aliases, incoming states and layout
contracts. Resource-instance churn alone does not force recompilation. Compatible
dynamic ranges can reuse a plan only when access/dependency semantics remain
valid. Structural, alias or incompatible layout/access changes recompile or fail;
they cannot be hidden behind a stale cached plan.

The validation source is the resource owner captured at actual allocation:
descriptor, incoming state, device provenance, and physical accessed range.
Graph declarations express requirements but cannot be copied back as those facts.
This validation happens before each protected upload map, recording, or
submission operation; allocation by itself is not submitted GPU work and creates
no completion ownership.

Pass-local authority restricts recording to declared resources and access ranges.
Graph uses the selected RHI's portable resource, capability, use and submission
vocabulary directly. Native barriers/queues/allocations remain backend concerns.
Dedicated allocations and one ordered lane are valid initial strategies. Aliasing,
cross-graph pools and multi-queue strategies require separate safety/evidence.

## Persistent residency, frame data, and frame work

Bases owns durable asset identity, content generations, CPU cache policy and
producer coordination. Callers own loading/decoding/streaming policy; platform
services own I/O. Renderer owns per-device GPU realizations and last-use tracking.
A realization key includes asset identity, content generation and selected RHI
provenance (`DeviceId` in the current baseline). Add no mandatory public device
generation absent an actual RHI contract.

Persistent residency uploads realize immutable or streamed content. Frame-data
uploads serve transforms, bone palettes and transient instance parameters with
a separate lifetime. They can share lower-level staging/copy machinery. Features
request preparation; the renderer coordinator submits it and retains its owners.
GPU skinning, particle simulation, culling and indirect generation are frame
Graph passes, not tasks hidden inside an upload service.

An upload exports ranges, incoming states, visibility and producer dependencies
to Graph/RHI. Pending content can serve a named consumer through an enforceable
GPU dependency; it is not thereby globally complete or reusable. A wait alone is
not a memory visibility transition. Overwrites must follow prior readers/writers
or use a fresh realization; staging and destination owners survive their last
uses. A CPU-ready flag is insufficient evidence.

## Completion, failure and platform lifecycle

Renderer retains original RHI outcomes and all potentially affected resources
through partial acceptance, issued prefixes, cancellation and indeterminate
results. It retires/reuses them only with covering completion or an authoritative
terminal resolution under the selected RHI contract. Returning from a function,
dropping a graph, replacing an asset or receiving a generic loss error does not
alone prove safe retirement.

A Feature cannot bypass this lifetime by freeing a buffer or switching providers.
Foreign/stale resources are rejected before protected backend work. Shader,
residency, scene and Graph errors preserve their owner and structured cause.
Unsupported capability or composition is an explicit result, not silent fallback.

Host and JSBridge own window/canvas lifecycle adaptation. RHI owns surface
configuration, acquisition, presentation and device lifecycle. Acquired surface
images are exclusive leases, not ordinary persistent textures. Browser sessions,
native objects and backend leases never enter this public architecture. Renderer
uses the same RHI evidence model on native and browser targets.

## Composition example

An application can assemble the following without making them core object kinds:

| Contribution | Application-selected providers | Shared products |
| --- | --- | --- |
| Static geometry | Mesh Feature + BVH visibility + opaque batching | Geometry/material bindings and opaque phase work |
| Terrain | Terrain Feature + quadtree selection | Patch/height data, phase work and any declared producer passes |
| Animated actors | External animation + Skinning Feature + chosen deformation | Pose inputs, palette frame data, shader inputs or produced streams |
| Particles | Particle Feature + GPU simulation/visibility | Graph buffers and supported indirect/transparent contributions |
| Frame | Forward or project FramePipeline | Phase policy, attachments and composition of all declared work |

This is a design example, not a promise that these Features are implemented or
that every combination is valid. Each phase must accept the products supplied;
transparency, capability and resource requirements still require validation.

## Delivery and validation of the design

Keep the organization sequence. Architectural extensibility is tested during
normal delivery, not retrofitted after a closed Forward renderer:

| Stage | Architectural proof |
| --- | --- |
| S0/S1 | Resource/access/plan/completion foundation; provisional fixtures can be replaced without a second Graph engine |
| S2/S3 | Generic Shader interfaces and Material contributions; preparation can consume their requirements |
| S4 | Dense RenderScene batches, immutable snapshots, and a public CPU-only typed feature facade; the bounded data/privacy/consumer acceptance is closed by the independently reviewed [S4 CPU closure record](s4-cpu-closure.md), without GPU or FramePipeline scope |
| S5 | Reference Forward plus external prototype; geometry layouts and a local policy can change without core edits |
| S6 | Bounded external typed opaque-Forward Feature, FramePipeline and custom material run with the reference mesh in one Graph pass; broader GPU products remain consumer-gated |
| S7 | A materially different consumer proves broader deformation, cross-pass or GPU-product seams before stabilization |
| S8 | Profile-led layout, streaming, batching and execution optimizations; the current acceptance covers only private validation indices for one material-heavy Forward workload |

S0/S1 has a separately bounded foundation closure. Its current fixed Copy/Raster/Compute entry points and
fixed vertex formats are fixtures, not the future public renderer model. Retain
the validated ownership/access algorithms and replace the adapters as real
consumers arrive; do not promise that implementation details will never change.
Do not introduce public trait scaffolds or change RHI to implement this document.

The exact acceptance matrix is maintained once in
[version-plan.md](version-plan.md), with Shader, Material and Graph details in
[their](design-shader.md) [own](design-material.md) [designs](design-rendergraph.md).
