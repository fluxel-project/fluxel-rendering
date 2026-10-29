# Fluxel Renderer Design

> Status: **Post-foundation target architecture**
>
> Scope: the data-oriented `RenderScene`, renderer-side preparation services,
> built-in reference pipelines and feature extension seams, asset-to-render
> preparation boundaries, and the renderer-internal RenderGraph.
>
> This document replaces the retained `0.15` fixed-renderer architecture as the
> renderer design target built on the stable independent RHI. The checked-out
> renderer's older `0.16` integration still requires migration and evidence.
> Portable RHI API and backend-SPI behavior are authoritative in the current
> [RHI Core design](https://github.com/fluxel-project/fluxel-rhi/blob/main/crates/core/DESIGN.md);
> this document does not freeze historical RHI example type names.
>
> Historical fixed recipes, `FixedFrameRenderer`, legacy snapshot gates, and old
> execution-plan terminology are implementation history only and are not part of
> the future renderer contract.
>
> The stable RHI is the foundation. The first renderer work establishes its
> internal RenderGraph and direct integration with that contract, then proves
> material/shader semantics and a small Renderer MVP. The MVP includes the
> built-in mesh path and one independent Rust-crate feature. It must prove
> batch scene input, phase participation, and graph-resource declaration before
> broader features or pipeline APIs are treated as stable.

---

## 1. Core Model

The target provides a complete, extensible real-time renderer. Its built-in reference
pipelines prove shared scene, material, shader, feature, RenderGraph, and RHI
contracts; they do not make Forward or Deferred the only rendering strategy.

```text
Game / Engine Scene
        |
        | extract rendering-only state
        v
RenderScene
        |
        v
FramePipeline SPI
        |
        | cull / sort / prepare
        | resolve material variants and shader/pipeline requirements
        | declare logical resources and passes
        v
RenderGraph
        |
        v
Shader / Pipeline + Material Runtime
        |
        v
RHI
```

The central contract is:

> A `FramePipeline` converts `RenderScene + RenderView` into a `RenderGraph`.

`fluxel-renderer` is one of the three planned rendering crates, beside
`fluxel-material` and `fluxel-shader`. `RenderGraph` and `FramePipeline` are
internal renderer modules, not independent crates. Renderer consumes the
material and shader crates and uses RHI portable contracts directly where their
meaning requires it. Normal frame GPU work still enters through RenderGraph,
whose pass-local authority controls graph resources.

---

## 2. Layer Ownership

### 2.1 Game / Engine Scene

Owned outside `fluxel-renderer`.

It may contain:

```text
gameplay
physics
AI
navigation
scripts
animation state
network state
editor state
save state
```

None of these concepts belong to the renderer contract.

### 2.2 RenderScene

`RenderScene` is the rendering-domain projection of the game scene.

It contains only data required to decide what can be rendered:

```text
render object identity
transform
bounds
geometry reference
material reference
visibility flags
render layers
render ordering hints
light data
renderer extensions
```

It is not a gameplay scene graph.

### 2.3 FramePipeline

`FramePipeline` owns frame rendering policy.

Built-in reference implementations include Forward and, after the shared path
is proved, Deferred. A Rust crate may provide another implementation through
the stable extension boundary. The SPI remains an internal renderer module;
it is not a separate Fluxel crate.

Possible implementations include:

```text
ForwardPipeline
DeferredPipeline
TilePipeline
PathTracingPipeline
project-specific pipelines
```

### 2.4 RenderGraph

RenderGraph receives the logical work already chosen by the pipeline.

It owns:

```text
passes
logical resources
resource versions
reads / writes
dependencies
definedness
lifetime
culling
execution planning
instantiation
```

It does not know:

```text
RenderScene
Mesh
Material
Forward
Deferred
Shadow
Bloom
TAA
```

### 2.5 RHI

RHI owns portable GPU execution:

```text
resources
shader artifacts
bindings
pipelines
recording
submission
completion
presentation
```

Native handles, barriers, queues, heaps, descriptors, and backend-specific
objects remain private to RHI/backend implementations.

---

## 3. Renderer Core Responsibility

`fluxel-renderer` owns:

```text
RenderScene vocabulary
RenderObject vocabulary
RenderView vocabulary

renderer-domain identities

CullingService
SortingService

material-variant and shader/pipeline resolution service seams

evidence-driven draw preparation service seams

FramePipeline SPI and built-in reference pipelines
FramePipelineContext

its internal RenderGraph, graph authoring, and direct RHI integration

feature registration, phase contribution, and typed feature-owned storage

renderer-level errors and results
```

It does not own gameplay world state, a mandatory ECS, or one fixed rendering
strategy. There must be no mandatory:

```text
Forward
Deferred
PBR
Lambert
Shadow
Bloom
TAA
fixed raster recipe for every consumer
```

inside the renderer core.

---

## 4. RenderScene

### 4.1 Purpose

`RenderScene` is a rendering-only scene representation.

`RenderScene` is data-oriented: it stores typed render-domain data in dense
storage and processes batches, rather than storing one dynamically dispatched
object per draw. The exact Rust layout is profile-led, but the initial model is
stable IDs with generation checks, a slot-to-dense map, and typed dense stores
for mesh instances, lights, and each registered feature. Hot fields such as
transforms, bounds, material references, and flags may use SoA or AoSoA; rare
metadata stays separate.

`RenderScene` should be:

- backend-independent;
- independent of RenderGraph resources;
- independent of native GPU objects;
- a read-mostly frame snapshot during frame construction;
- suitable for multiple views;
- independent of gameplay scene layout.

### 4.2 RenderObject and identity

Conceptual shape:

```rust
pub struct MeshInstanceRecord {
    id: RenderId,

    transform: RenderTransform,
    bounds: RenderBounds,

    geometry: GeometryHandle,
    material: MaterialHandle,

    visibility: VisibilityFlags,
    layers: RenderLayerMask,
    queue: RenderQueue,
}
```

This describes responsibility, not frozen Rust ABI. It is a logical record, not
a per-object trait object and not the required physical storage layout.

### 4.3 Identity

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderId {
    slot: u32,
    generation: u32,
}
```

`RenderId` is not required to equal:

```text
ECS Entity
GameObject ID
AssetId
RHI ObjectId
GPU address
```

The integration layer owns the mapping from game objects to render IDs. Dense
storage may use swap removal only when it preserves this generation rule.

### 4.3 Scene updates and adapters

Renderer never queries an ECS or a scene graph. An adapter pushes typed,
batched creates, updates, and removals into `RenderScene`; it can be backed by
an ECS, a scene graph, or application-specific data without exposing any of
those models to the renderer.

```text
external world -> adapter -> typed ChangeSets -> RenderScene snapshot
```

Updates should be split by field where that avoids needless work, for example
transform, bounds, material, and visibility streams. Large producers may submit
chunk or slice updates. A frame boundary snapshots accepted changes, so
concurrent world mutation never races a renderer read. GPU uploads use dirty
ranges and per-frame resource retention; they do not re-upload the full scene
on every frame.

---

## 5. Transform

The renderer consumes already-resolved transform data.

Conceptually:

```rust
pub struct RenderTransform {
    pub model_to_world: Mat4,
    pub previous_model_to_world: Option<Mat4>,
}
```

Hierarchy evaluation does not belong to the renderer.

Systems such as:

```text
scene hierarchy
animation state and blend trees
physics
constraints
```

must resolve their current rendering transform before or during `RenderScene`
extraction/update. GPU skinning, bone-palette uploads, and skinning passes are
renderer features; game animation state and its simulation remain outside.

---

## 6. Bounds

Bounds are explicit renderer-domain data.

Conceptually:

```rust
pub enum RenderBounds {
    Sphere {
        center: Vec3,
        radius: f32,
    },

    Aabb {
        min: Vec3,
        max: Vec3,
    },
}
```

The implementation may initially standardize on one canonical representation.

Culling operates on render bounds and never traverses gameplay scene structure.

---

## 7. Geometry and Material References

`RenderScene` stores logical references:

```rust
pub struct GeometryHandle {
    /* opaque */
}

pub struct MaterialHandle {
    pub asset: MaterialAssetRef,
    /* optional renderer-domain instance selection/state */
}
```

These are not RHI buffers, textures, bind groups, or pipelines.

`GeometryHandle` and `MaterialHandle` are conceptual typed logical references,
not independent asset identity domains. When they are backed by Fluxel assets,
their durable identity is the corresponding `AssetId<K>` plus
`ContentGeneration`, for example:

```rust
type GeometryAssetId = AssetId<GeometryAsset>;
type MaterialAssetId = AssetId<MaterialAsset>;

pub struct MaterialAssetRef {
    pub id: MaterialAssetId,
    pub generation: ContentGeneration,
}
```

A renderer may assign a separate identity to a `MaterialInstance`, because an
instance is mutable renderer-domain state rather than the material asset. That
instance and `MaterialHandle` must retain `MaterialAssetRef`; a
`MaterialInstanceId` must not replace or mint a parallel durable material
identity. In particular, `AssetId<MaterialAsset>` alone is insufficient after
content replacement or hot reload.

Resolution happens later:

```text
GeometryHandle
    -> render-ready geometry representation

MaterialHandle
    -> material variant
    -> shader / pipeline requirements
```

This keeps `RenderScene` device-independent.

---

## 8. RenderView

One frame may contain one or more rendering views.

Conceptually:

```rust
pub struct RenderView {
    pub id: RenderViewId,

    pub view: Mat4,
    pub projection: Mat4,

    pub viewport: Viewport,
    pub layers: RenderLayerMask,

    pub camera_position: Vec3,
}
```

Later additions may include:

```text
previous matrices
jitter
exposure
LOD parameters
stereo / XR metadata
view-family identity
```

Only real requirements should add fields.

---

## 9. Visibility and Render Layers

Conceptually:

```rust
pub struct VisibilityFlags(u32);
pub struct RenderLayerMask(u64);
```

Possible portable semantics:

```text
visible
casts shadow
receives shadow
```

Project-specific gameplay categories must not leak into the renderer core.

---

## 10. Render Queue

RenderScene may provide coarse ordering hints:

```rust
pub enum RenderQueue {
    Opaque,
    AlphaTest,
    Transparent,
    Overlay,
    Custom(i32),
}
```

This metadata does not force a pipeline policy.

The active `FramePipeline` decides how queues participate in passes and sorting.

---

## 11. Culling Service

Culling is reusable renderer infrastructure.

It is not a built-in pipeline.

Conceptually:

```rust
pub trait CullingService {
    fn cull(
        &self,
        scene: &RenderScene,
        view: &RenderView,
        request: &CullingRequest,
    ) -> Result<VisibleSet, CullingError>;
}
```

```rust
pub struct CullingRequest {
    pub layers: RenderLayerMask,
}
```

```rust
pub struct VisibleSet {
    objects: Vec<RenderId>,
}
```

A minimal implementation may provide:

```text
visibility flags
layer filtering
frustum culling
```

Future implementations may add:

```text
LOD
occlusion
HZB
portal visibility
GPU-driven visibility
```

without changing the meaning of `FramePipeline`.

---

## 12. Sorting Service

Sorting is another reusable mechanism.

Conceptually:

```rust
pub trait SortingService {
    fn sort(
        &self,
        scene: &RenderScene,
        view: &RenderView,
        visible: &VisibleSet,
        policy: SortPolicy,
    ) -> Result<SortedRenderList, SortError>;
}
```

Possible initial policies:

```rust
pub enum SortPolicy {
    FrontToBack,
    BackToFront,
    StableSceneOrder,
}
```

Batching and pipeline/material-aware sorting use a real shader/material key
model. Object sorting and graph scheduling are different operations: an opaque
phase may order packets by pipeline, material, and depth while transparent work
orders primarily back-to-front; RenderGraph orders passes from resource
dependencies. Neither order substitutes for the other.

---

## 13. Draw Preparation

The conceptual preparation pipeline is:

```text
RenderScene
    |
    v
VisibleSet
    |
    v
SortedRenderList
    |
    v
DrawPackets / DrawItems
```

Culling answers:

> Which objects matter for this view?

Sorting answers:

> In what logical order should they be considered?

Future draw preparation answers:

> Which concrete geometry, material variant, shader program, and pipeline
> requirements does each draw use?

Preparation also turns shader reflection and a material variant's parameter,
uniform, texture, buffer, and sampler schema into a concrete binding-use
manifest. For every buffer or texture it names the RHI resource, relevant
range, access, shader stage, and RHI device provenance. The active
pipeline declares that manifest to Graph before recording; binding a material
object does not bypass graph resource declaration or dependency analysis.
Samplers remain validated binding inputs but are not memory accesses.

This layer must not be invented prematurely before the material/shader contracts
are ready. GPU-driven paths may instead produce candidate data for GPU culling,
compaction, sorting, and indirect arguments; they are not forced to create a
complete CPU `DrawList`.

---

## 14. RenderFeature extension model

A feature extends renderer capability at batch granularity. It may own typed
storage and contribute scene extraction, preparation, packets, material/shader
requirements, and graph passes for one or more pipeline phases. Examples are
static mesh, GPU skinning, terrain, water, particles, vegetation, decals, and
area-light implementations.

The exact traits are established by the MVP rather than frozen from this design,
but their boundary is fixed:

```text
external scene data -> feature-owned typed storage -> batch prepare/queue
    -> declared phase contribution and RenderGraph passes
```

Dynamic dispatch, if used, occurs once per feature stage, never once per render
object or draw. Features may add object types and passes without a growing core
`match` over object kinds. A feature that affects shading must use the
material/shader extension contract; graph-only hooks are insufficient for, for
example, area lights.

Every feature's frame GPU operation, resource access, and queue dependency is
declared through the renderer's RenderGraph. A feature cannot submit frame work
directly to RHI or hide a side graph. `FramePipeline` owns phase policy;
features contribute content to the selected phases. Persistent residency upload
is a renderer preparation service with the limited contract described in section
21; a feature supplies CPU data and requirements to that service, never an
independent submission path.

The first extensibility gate is deliberately small: the renderer's built-in mesh
feature plus one feature implemented in a separate Rust crate. The public trait,
context facade, handles, and errors used by that crate are the supported surface;
graph compiler, scheduler, allocator, RHI submission, and backend objects are
not. A small external compile/run prototype shapes that surface while the
Forward path is built, before the final integration closure. The Feature must
exercise
batched incremental input, feature-owned storage, a declared phase, and graph
resource/pass contribution. This proves only the exercised subset of the
extension interface. Stabilizing cross-pass contribution or shader/material
extension contracts waits for a second materially different feature that uses
those contracts. Terrain, water, particles, and full skeletal animation remain
later consumers of this boundary.

---

## 15. FramePipeline SPI

The core SPI is intentionally small. `pipeline/` remains a renderer-owned
module and hides graph compilation, executor state, and renderer internals. The
public extension surface is the `FramePipeline` trait plus its controlled
context, handles, and error types. That permits an external Rust crate to
implement a pipeline without exposing mutable graph compiler state or making
the module a separate crate or a stable plugin ABI.

Conceptually:

```rust
pub trait FramePipeline {
    fn build(
        &mut self,
        ctx: &mut FramePipelineContext<'_>,
    ) -> Result<FramePipelineOutput, FramePipelineError>;
}
```

The context initially provides:

```rust
pub struct FramePipelineContext<'a> {
    pub scene: &'a RenderScene,
    pub views: &'a [RenderView],

    pub culling: &'a dyn CullingService,
    pub sorting: &'a dyn SortingService,

    pub graph: &'a mut RenderGraphBuilder,
}
```

The initial minimum context may omit material-backed services while the scene
and SPI shape are being established. Once a material-backed Forward pipeline
is implemented, however, `FramePipelineContext` must receive real renderer
services for material-variant resolution and shader/pipeline resolution. The
pipeline uses those services before pass declaration to establish its shader,
pipeline, binding, and resource requirements; it must not bypass the renderer
by consulting global material, shader, or pipeline state directly.

Additional services may be introduced when their contracts exist, for example:

```text
draw-packet builder
history service
shader composition/compiler (when it is distinct from shader/pipeline resolution)
```

Draw-packet preparation and history remain evidence-driven additions. No
placeholder service is required before its contract exists.

---

## 16. Policy vs Mechanism

`FramePipeline` owns policy.

Common renderer services provide mechanism.

For example:

```rust
let visible = ctx.culling.cull(
    ctx.scene,
    view,
    &request,
)?;

let ordered = ctx.sorting.sort(
    ctx.scene,
    view,
    &visible,
    SortPolicy::FrontToBack,
)?;
```

A different pipeline may choose:

```text
different culling request
different sorting
different pass partitioning
different material variants
different graph topology
```

Therefore:

```text
CullingService != pipeline policy
SortingService != pipeline policy
```

---

## 17. RenderGraph Authoring

`FramePipeline` builds GPU work through RenderGraph.

It may:

```text
create logical resources
import external resources
add raster passes
add compute passes
add copy passes
declare reads/writes
declare frame outputs
declare presentation roots
```

It may not:

```text
access native command encoders
record native barriers
submit native queues
touch descriptor heaps
unwrap native handles
bypass RenderGraph validation
```

There must be one normal rendering path:

```text
FramePipeline
    -> RenderGraph
    -> RHI
```

FramePipeline and Features cannot submit outside RenderGraph. The explicit
residency upload service in section 21 shares the same RHI synchronization and
resource-lifetime model.

Graph authoring has three distinct products. A definition records the selected
passes, topology, logical resources, and declared resource uses. Compilation
turns a definition into a device-specific plan. Per-frame instantiation binds
the current imports, attachments, prepared material bindings, and dynamic
values. The last step may refresh a binding only when it preserves the compiled
layout and compatible concrete resource-use, alias, and dependency contracts.
Resource-instance churn, including large read-only material sets, does not by
itself make a new graph. Each concrete range and access is revalidated. A
topology, binding-layout, incompatible access/range, alias relationship, or
resource-edge change makes a new definition and recompiles or fails
structurally; it must not be hidden in a cached plan.

---

## 18. FramePipeline Output

The graph remains the main product.

The pipeline may additionally return semantic frame outputs.

Conceptually:

```rust
pub struct FramePipelineOutput {
    pub color: Option<GraphTexture>,
    pub depth: Option<GraphTexture>,
}
```

The public shape should remain narrow until concrete consumers require more.

---

## 19. Multi-Pass Features

A frame pipeline may use reusable graph-authoring helpers:

```text
ShadowFeature
BloomFeature
TaaFeature
DepthPyramidFeature
```

A feature may expand into several RenderGraph passes.

Example:

```text
BloomFeature
    -> Downsample Pass
    -> Blur H Pass
    -> Blur V Pass
    -> Upsample Pass
```

These helpers are composition units, not nested runtime graphs.

The actual RenderGraph scheduling unit remains the pass.

---

## 20. Relationship to slot-graph

`slot-graph` remains a CPU-side dependency/composition utility.

It may be used for:

```text
scene extraction
per-object preparation
culling preparation
draw preparation
material/shader preparation
higher-level feature preparation
```

It does not model:

```text
GPU resource hazards
GPU submission
GPU completion
GPU synchronization
RenderGraph execution
```

Its future `Subgraph` concept may package reusable CPU preparation/features, but
all frame GPU work still becomes ordinary RenderGraph passes. CPU preparation
may request residency uploads through the service below.

No `slot-graph` node or subgraph identity needs to cross the renderer public API.

---

## 21. Asset and GPU Residency Boundary

Persistent logical asset identity, loading, caching, hot reload, and eviction
remain outside RenderGraph.

The durable flow is:

```text
AssetStore
    |
    | immutable asset/content snapshot
    v
renderer preparation
    |
    | resolve / upload / reuse device-local representation
    v
render-ready geometry/material resources
    |
    v
FramePipeline / RenderGraph declaration
```

A RenderGraph pass must not:

```text
load an asset
query AssetStore
start an asset upload
own persistent asset lifetime
```

It consumes already-resolved render/GPU resources through explicit graph
imports/bindings.

### 21.1 Persistent residency, frame data, and frame work

Persistent residency upload is distinct from frame-graph work. The renderer's
residency coordinator realizes CPU-prepared immutable or streaming content,
including meshes, textures, and terrain tiles, through RHI before graph
recording. It owns the persistent realization's asset/content identity, accepted
producer submission, completion evidence, and failure/loss outcome.

Frame-data upload has a separate lifetime for current-frame transforms, bone
palettes, and similar per-frame inputs. It may reuse the same low-level staging
and transfer mechanism, but it is not a long-lived residency asset cache.
Particle seed/input data follows the lifetime selected by its consumer rather
than making the residency coordinator a general frame scheduler.

Both paths record a completed or pending producer as an imported resource state
and dependency for Graph. A consuming pass may use a pending producer only when
the shared RHI plan has an enforceable dependency and matching visibility/state
transition; it is not globally complete merely because CPU preparation ended.

The coordinator also orders upload writes after any previous readers/writers of
the destination range, or allocates a fresh realization while the old one is in
flight. Staging storage and destination resources remain retained until their
respective last-use completion. Readiness requires an observed producer
completion or an enforceable RHI submission dependency; a CPU-ready flag alone
is insufficient. A single-queue implementation may use queue ordering and a
bounded completion wait where required; multiple queues are not a prerequisite.

Per-frame GPU work belongs in RenderGraph: GPU skinning, particle simulation,
culling, indirect-argument generation, and other dispatch/draw/copy work are
ordinary graph passes with declared reads and writes. The coordinator is not a
second frame scheduler and cannot create invisible resource accesses. A feature
may request preparation, but cannot submit it; wrong-device resources,
unsupported uses, failed uploads, and unavailable
completion dependencies return structured preparation/graph/RHI errors before
the consuming pass records.

A device-local cache, if used, must key by enough identity to prevent stale or
cross-device reuse, conceptually including:

```text
logical asset identity
content generation
the selected RHI DeviceId/provenance
any private generation exposed by the selected RHI contract
```

Exact cache ownership belongs to the renderer/resource preparation layer, not
RenderGraph.

---

## 22. Resource Lifetime Boundary

Renderer preparation may retain device-local resources needed by a frame.

However, RHI reports whether work was rejected, accepted, partially accepted,
failed with an issued prefix, or indeterminate, as well as any completion or
loss facts. The renderer preserves every resource that could be affected by a
partial or indeterminate outcome and refuses reuse until terminal evidence or
the selected recovery path makes its state trustworthy. The obsolete
"accepted-unknown" protocol is not retained, but uncertainty is not erased.

The authoritative facts include:

```text
submission acceptance
GPU completion
device-loss outcome
```

Renderer/RenderGraph retain submitted resources and decide reuse or retirement
only from these RHI facts.

Renderer code must not guess that a resource is safe to recycle merely because:

```text
the frame function returned
the graph object was dropped
the scene changed
the asset was replaced
```

Accepted GPU work remains retained by the renderer/executor until the
corresponding RHI completion evidence proves retirement is safe.

---

## 23. Material and Shader Boundary

The renderer core must not assume one material model.

It must not hard-code:

```text
PBR
Metallic/Roughness
Lambert
Phong
Substrate
```

The future material/shader subsystem will produce renderer-consumable results
such as:

```text
material variant
shader artifact(s)
shader interface
parameter schema
pipeline requirements
```

`FramePipeline` decides how those variants participate in passes.

Material contributes evaluation WGSL and generic Shader-owned source/interface
information. Before compilation, `FramePipeline` combines that contribution with
the selected pass template, entry points, geometry interface, static choices,
and target requirements to make a complete `ShaderCompositionRequest`. A Shader
consumer that has no material uses the same request without a material
contribution. Shader composes and validates the portable source/artifact;
the opened RHI device chooses its supported shader route and keeps backend
lowering, native compilation, and ABI details private.

RenderGraph does not compile material graphs or compose shader source.

RHI does not know materials.

---

## 24. Built-in reference pipelines

The renderer contains built-in reference pipelines that exercise the shared
contracts. The first is Forward; Deferred follows only after the retained scene
and feature paths prove it can consume the same materials, shaders, and graph
model. These implementations are reference consumers, not a hidden mandatory
graph for every application.

The renderer must not silently impose a fixed graph such as:

```text
Depth
-> GBuffer
-> Lighting
-> Transparent
-> Bloom
-> Tonemap
```

Nor should it silently choose:

```text
Forward
Deferred
Mobile
Path Tracing
```

An application or an external Rust crate may provide another `FramePipeline`
implementation through the proved feature boundary. Future ecosystem crates may
include:

```text
fluxel-renderer-forward
fluxel-renderer-deferred
fluxel-renderer-2d
project-specific renderer crates
```

This packaging is optional and does not turn `FramePipeline` into its own crate.

---

## 25. Minimal Example

Conceptually:

```rust
pub struct ProjectPipeline;

impl FramePipeline for ProjectPipeline {
    fn build(
        &mut self,
        ctx: &mut FramePipelineContext<'_>,
    ) -> Result<FramePipelineOutput, FramePipelineError> {
        let view = &ctx.views[0];

        let visible = ctx.culling.cull(
            ctx.scene,
            view,
            &CullingRequest {
                layers: view.layers,
            },
        )?;

        let ordered = ctx.sorting.sort(
            ctx.scene,
            view,
            &visible,
            SortPolicy::FrontToBack,
        )?;

        let color = ctx.graph.create_texture(/* ... */)?;

        ctx.graph.add_raster_pass(
            "main",
            /* resource declarations */,
            move |pass| {
                // consume already-prepared draw/material/shader data
                // encode portable RHI commands through the graph execution seam
                Ok(())
            },
        )?;

        Ok(FramePipelineOutput {
            color: Some(color),
            depth: None,
        })
    }
}
```

The example intentionally leaves material/shader resolution abstract until that
subsystem is designed.

---

## 26. Error Model

Renderer-level errors should distinguish at least:

```text
RenderScene invalid data
Culling failure
Sorting failure
Preparation failure
Material/Shader resolution failure
RenderGraph authoring failure
Unsupported pipeline requirements
```

RHI execution/submission/completion failures remain structured RHI/Graph
failures and must not be flattened into strings.

---

## 27. Target crate structure

```text
crates/renderer/src/
    lib.rs

    render_scene/
        mod.rs
        storage.rs
        changes.rs
        snapshot.rs
        view.rs
        bounds.rs

    feature/
        mod.rs
        registry.rs
        phase.rs
        context.rs

    culling/
        mod.rs

    sorting/
        mod.rs

    preparation/
        mod.rs

    graph/
        builder.rs
        compiler.rs
        scheduler.rs
        allocator.rs
        executor.rs

    pipeline/
        mod.rs
        context.rs
        output.rs
        error.rs
        forward.rs
        deferred.rs
```

`graph/` and `pipeline/` are internal `fluxel-renderer` modules. During
migration, the existing `fluxel-rendergraph` crate is an implementation
boundary, not the target public architecture; its contracts and evidence move
into the renderer without losing their validation coverage.

The renderer should not contain:

```text
gameplay_ecs/
per_object_draw_trait/
dynamic_dll_plugin_loader/
one universal feature data layout/
```

Historical fixed-renderer code may remain temporarily during migration, but it
must not define the post-foundation public architecture.

---

## 28. Relationship to Unity SRP

Conceptually:

```text
Unity
--------------------------------
renderable scene / CullingResults
ScriptableRenderContext services
RenderPipeline
RenderGraph / renderer features

Fluxel
--------------------------------
RenderScene / VisibleSet
renderer services + RenderGraphBuilder
FramePipeline
RenderGraph feature/subgraph builders
```

The important common idea is:

> A project-defined pipeline decides how renderable scene data becomes frame work.

Fluxel keeps one additional hard boundary:

> Normal GPU work must be represented through RenderGraph; a custom pipeline
> does not gain a parallel native-command escape path.

---

## 29. Frame Flow

The intended future frame path is:

```text
Game / Engine Scene
        |
        v
RenderScene extraction/update
        |
        v
RenderView(s)
        |
        v
FramePipeline::build
        |
        +-- CullingService
        +-- SortingService
        +-- material-variant and shader/pipeline resolution
        +-- feature builders
        |
        v
RenderGraph declaration
        |
        v
RenderGraph compile / instantiate
        |
        v
pass-local graph authority
        |
        v
Shader / Pipeline + Material Runtime binding
        |
        v
RHI execution
        |
        v
completion / present / frame result
```

---

## 30. Historical 0.15 Renderer

The previous `0.15` renderer provided:

```text
fixed Camera / Geometry / Mesh / BasicMaterial
fixed snapshot upload paths
fixed RenderPacket
seven closed raster recipes
FixedFrameRenderer
fixed graph declarations
legacy snapshot reservation/gate behavior
```

Those paths were valuable correctness proofs for the old foundation.

They are not the future renderer architecture.

During migration:

- retain tests/evidence where they still prove lower-level behavior;
- migrate useful resource/preparation invariants into the new architecture;
- remove fixed recipe assumptions from the future public API;
- do not carry obsolete execution-plan, browser-token, or the old
  accepted-unknown protocol into the new foundation; preserve current RHI
  partial/indeterminate outcome handling and retention requirements.

---

## 31. Final Contract

```text
RenderScene
    = data-oriented, rendering-only snapshot fed by batched external adapters

Culling / Sorting
    = reusable renderer mechanisms

RenderFeature
    = feature-owned typed storage and batch work that contributes to pipeline phases

FramePipeline
    = renderer-owned public Rust SPI and built-in/external policy that converts
      RenderScene + RenderView into RenderGraph

RenderGraph
    = frame work, resource dependencies, and execution planning

Shader / Pipeline + Material Runtime
    = prepared execution objects and per-instance binding data selected by a pass

RHI
    = portable GPU execution
```

The defining rule is:

> `fluxel-renderer` provides a complete extensible renderer: common scene and
> preparation mechanisms, reference pipelines, feature extension seams, and
> one internal graph/RHI execution path.
