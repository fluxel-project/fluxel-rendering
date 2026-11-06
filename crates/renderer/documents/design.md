# Fluxel Renderer Design

`fluxel-renderer` integrates scene data, material and shader products, frame
policy, resource residency, and a private RenderGraph. It is the only layer in
this workspace that turns prepared rendering work into RHI recording and
submission. It does not own application state, an ECS, a host event loop,
native windows, browser sessions, or backend-native GPU objects.

The crate depends on [Shader](../../shader/documents/design.md) and
[Material](../../material/documents/design.md). Shader owns logical WGSL
composition and reflection; Material owns authoring semantics and generic
contributions. The selected RHI owns physical resource realization, recording,
submission, completion, presentation, device loss, and backend lowering.
Renderer owns the integration and the lifetime decisions that consume those RHI
facts.

## Scope and implementation boundary

The implemented surface is deliberately narrower than the target architecture.

| Area | Current implemented subset | Not established by the subset |
| --- | --- | --- |
| Scene | Generational render and view identities; transactional batch changes; immutable snapshots; typed CPU feature storage | A general ECS integration or heterogeneous runtime registry |
| Forward | Multi-mesh preparation over geometry, materials and textures; controlled opaque and clear policy; residency and frame data | General SRP, transparent composition, arbitrary external GPU products |
| External composition | One statically typed opaque Forward Feature and external pipeline compose through the actual Graph | Dynamic plugins, arbitrary Graph access, a plugin ABI, general cross-pass Features |
| Deferred | One mesh, fixed lighting, selected metallic/roughness material path, and optional position-only GPU deformation | Configurable lighting, IBL, normal maps, arbitrary deformation, general Deferred scenes |
| Optimisation | Private validation indices for the exercised Forward path | A public cache/index API, or claims about GPU duration, VRAM, or whole-frame speedup |

Future work starts from an actual producer and consumer. A new public extension
contract requires a demonstrated input/output shape, Graph declaration, binding
validation, failure cleanup, and completion ownership; architectural preference
alone is insufficient.

## Ownership model

```text
application / ECS / scene graph
  -> RenderChangeSet + feature-owned typed inputs
  -> RenderScene immutable FrameSnapshot
  -> FramePipeline and Feature preparation
  -> private Graph declarations and compiled device plan
  -> RHI recording, submission, Completion
  -> frame-data and residency retirement/reuse
```

`RenderScene` is a rendering-domain snapshot boundary, not an application object
world. It stores consumer-chosen cloneable resource bundles beside records,
finite transforms and bounds. Applications map their own entities to stable,
generational `RenderId` and `RenderViewId`; dense storage indices, asset IDs,
device IDs, and GPU addresses are different identities. Changes apply as one
transaction, reject stale/foreign IDs and conflicting writes, report exact dirty
ranges, and publish snapshots that remain valid after later mutation.

The application owns gameplay ECS scheduling, animation evaluation, source I/O,
and policy for selecting visible content. Renderer owns neither their thread
model nor a universal scene layout. Features may use dense columns, compact
records, GPU buffers, or another layout appropriate to their access pattern.
CPU scene layout, logical asset layout, and GPU vertex layout are independent.

Persistent asset identity is supplied by Bases as asset identity plus immutable
content generation. Per-device GPU residency is Renderer-owned and keyed by
actual content and device provenance. A cache hit still consumes this frame's
real Graph declaration and verifies the concrete owner before dependent GPU
work.

## Static extension composition

Extensions are ordinary Rust implementations assembled at compile time. All
`RenderFeature`, `FramePipeline`, local geometry, visibility, ordering,
batching, and deformation providers use generics, traits, and monomorphization.
There is no trait-object dispatch, erased plugin registry, hand-written vtable,
runtime-loaded unknown Feature, or global service locator in this architecture.

Typed tuples, ordinary aggregates, and associated types are sufficient. Runtime
enablement is state in a known concrete Feature; a runtime choice among known
implementations uses an explicit enum. Static implementation types do not make
scene objects, assets, materials, or Graph topology static types.

Public contexts are controlled capabilities. An external pipeline can choose one
of the Renderer-owned products and make the allowed phase declarations; an
external Feature can prepare its typed batch and contribute to the allowed
phase. Neither receives Graph construction, allocator, command encoder,
submission, surface lease, native handle, or scheduling authority. Opaque
handles and contexts validate scene, issuer, feature, snapshot version, phase,
and replay identity before use.

The stable direction is batch-oriented: a Feature exchanges typed input,
prepared products, and bounded access requirements. It is not a per-object
virtual `draw`, `cull`, or `sort` model. Cross-Feature ordering, particularly
for a future transparent phase, belongs to a frame-wide composition rule rather
than incidental Feature registration order.

## Geometry, materials, and preparation

Geometry describes streams and attributes, not one mandatory Rust vertex type.
Streams carry range, stride, step rate, and physical owner data; attributes carry
semantic identity, format, and offset. A mesh may have separate vertex streams,
indices, submesh ranges, material assignments, and feature-specific deformation
inputs. Preparation resolves semantic-to-location mappings against Shader
reflection and creates concrete bindings. It rejects missing or duplicate
semantics, invalid physical ranges, unsupported formats, and incompatible
material/layout requirements before recording.

Binding views and Graph access footprints are distinct. The RHI vertex binding
keeps the index origin required by the draw, while Graph declares the actual
physical fetch footprint. Dynamic uniform offsets and texture staging likewise
declare their actual physical ranges and layouts. Texture row pitch and texture
placement alignment are separate RHI constraints.

Material and Shader integration follows this path:

```text
Material definition and instance
  -> generic WGSL contribution and interfaces
  -> Renderer pass template, entry points, and vertex requirements
  -> Shader composition and reflection
  -> prepared concrete bindings and Graph access declarations
  -> RHI pipeline/resource realization
```

Renderer does not duplicate Material IR or Shader composition. Logical shader
validity does not prove hardware capability: the selected RHI route must admit
the exact requirements and device facts before realization and execution.
An optional capability is not a partial promise: creation, binding, recording,
and the selected backend lowering must all support the complete portable
contract before Renderer advertises or consumes it.

## Private RenderGraph

Graph is the single core path for resource access, dependency construction,
planning, validation, recording, and execution. A Graph definition, a
device-specific compiled plan, and a frame-specific binding set are separate
products. Caching a compatible plan never authorizes an empty preflight, a
second scheduler, or replay of internal policy: every frame contributes its
actual declarations and revalidates its bindings.

The Graph tracks imports and exports, logical resource versions, definedness,
pass-local accesses, dead-pass culling, hazards, dependencies, state transitions,
deterministic ordering, transient
lifetime, and diagnostics. It validates concrete resource owners against the
descriptor saved at allocation, incoming state, and selected RHI device
provenance. A requested graph descriptor is not evidence about an actual owner.

Logical transient lifetime is a planning fact, distinct from physical allocation
reuse. Execution remains serial until a real RHI consumer proves a safe ordering,
queue, ownership, and completion contract for additional parallelism. Renderer
uses only portable RHI contracts; Host owns window/canvas lifecycle, while RHI
owns device and Surface realization. Assets remain outside Graph policy: they
provide logical identity and content, while Renderer decides when an asset
becomes a device-local realization.

Access validation protects the operation it precedes. Buffer, uniform, vertex,
and texture ranges must cover the physical footprint. Texture coverage respects
aspects, mip ranges, and array layers; disjoint declarations may cover distinct
requested regions but cannot fill a missing region. A plan is reusable only when
topology, hazards, layouts, and access contracts stay compatible with the
concrete frame bindings.

Recording callbacks receive only the resources declared for their pass through
the held RHI recording scope. They use RHI's command vocabulary directly;
Renderer must not create a second command/replay model. Errors retain their
owning layer: Renderer input/preparation errors, Graph declaration/binding/plan
errors and RHI execution errors are not replaced by an invented shared hierarchy.

## Residency, frame data, and completion

Renderer separates three owner classes:

* CPU asset caches retain logical source content.
* Per-device residency retains persistent realized geometry, textures, programs,
  and their descriptor/provenance facts.
* Frame data and Graph transients retain input ranges and intermediate resources
  only for the submissions that consume them.

Resources submitted to the GPU retire only after observing a completion that
covers their last use. Completion wait failure retains pending owners. Known
device loss clears non-reusable caches and refuses new work; it does not invent
retirement evidence. Conversely, a later CPU map/readback failure does not undo
a completion that was already observed for the GPU work.

Surface images are leases, not cacheable texture views. Presentation, resize,
loss, and host lifecycle remain bounded by RHI and Host contracts. Renderer must
not let a surface lease escape a frame or become persistent residency.

Partial recording or execution failures retain any owner whose GPU effect is
unknown until completion or a safe shutdown establishes otherwise. Explicit
pre-device refusal may release local temporary owners. Device provenance is part
of cache isolation: resources prepared for one device cannot bind to another;
returning to an earlier device may reuse only that device's compatible owners.

## Pipeline direction

`FramePipeline` owns phase policy and Renderer-owned output selection; a
`RenderFeature` owns its local preparation algorithms and contributes through
the controlled phase interface. The current topology is intentionally limited to
opaque Forward, clear-only, and the selected Deferred product. This does not
freeze a universal render-pass taxonomy.

An eventual GPU product produced by a Feature must be a first-class Graph
resource with an explicit producer, consumer, version, access, and lifetime.
Renderer must not force it through a CPU draw list or readback merely because
the current Forward path is CPU-prepared. A future transparent phase needs an
explicit final ordering/composition policy across Features.

Animation sampling, blend trees, pose evaluation, and gameplay state remain
outside Renderer. A Feature may consume evaluated pose/morph input and select
CPU deformation, vertex-stage deformation, or Graph compute production. Bounds
used for culling must be conservative for the selected deformation or depend on
updated bounds; stale pre-deformation bounds are not accepted implicitly.

## Failure boundaries and invariants

* Reject stale scene/view/feature identities, mismatched snapshot versions, and
  incompatible phase contributions before backend work.
* Reject missing declarations, incompatible access contracts, actual-owner
  descriptor/state/device mismatches, and insufficient physical ranges before
  allocation, upload, recording, or submission as appropriate.
* Do not infer support from a backend name or logical shader validation. Consume
  the same selected device facts at route, creation, binding, and execution.
* Preserve submitted owners through failure and completion boundaries. Device
  loss is terminal for reuse until the relevant renderer state is rebuilt.
* Keep Graph and RHI authority private. External code cannot construct phase
  contexts or name private Graph modules.

## Source layout and verification

The public facade is `src/lib.rs`. Scene contracts live in `src/render_scene/`;
controlled extension and pipeline surfaces in `src/feature/` and `src/pipeline/`;
preparation in `src/preparation/`; private Graph in `src/graph/`; realization
and transient ownership in `src/residency/` and `src/frame_data/`; coordinated
submission in `src/renderer/`.

Unit tests reside in `src/tests/`, grouped with their production responsibility.
Public integration tests reside in crate-root `tests/`. The independent
`../examples/external-renderer` consumer verifies that external Rust code can use
the bounded public scene and extension contracts while private Graph and phase
construction remain inaccessible. Native graphical behaviour requires separate
real-window and RHI readback evidence; CPU tests and compilation do not replace
that evidence. Native evidence records the actual target, inputs, outputs,
diagnostics, and host/window facts; a successful offscreen readback does not
stand in for a visible presentation claim.
