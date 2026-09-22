# Fluxel Renderer Design

> Status: **Post-foundation target architecture**
>
> Scope: `RenderScene`, renderer-side preparation services, custom `FramePipeline` SPI,
> asset-to-render preparation boundaries, and RenderGraph authoring.
>
> This document replaces the retained `0.15` fixed-renderer architecture as the
> renderer design target built on the completed `0.16` RHI baseline.
>
> Historical fixed recipes, `FixedFrameRenderer`, legacy snapshot gates, and old
> execution-plan terminology are implementation history only and are not part of
> the future renderer contract.
>
> Delivery order: the minimal scene milestone freezes only `RenderScene`,
> `RenderObject`, `RenderView`, the `FramePipeline` SPI, and one Forward proof.
> The complete scene-preparation milestone adds culling and proves the SPI again
> with Deferred. The renderer authors RenderGraph work using RHI's portable
> contract; it owns scene policy and ordinary integration glue, not a Graph/RHI
> translation layer or a bypass around graph pass-local resource authority.

---

## 1. Core Model

Fluxel does not ship a mandatory rendering strategy.

The renderer layer defines the data model and services needed to build one:

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

The diagram is an execution path, not a rule that forces crate dependencies
into a chain. Renderer consumes RenderGraph and material/shader services;
RenderGraph, shader/pipeline code, and material runtime may each use RHI's
portable contracts directly. Normal frame GPU work still enters through
RenderGraph, whose pass-local authority controls graph resources.

`fluxel-renderer` defines the language and common services for building a
renderer. It does **not** define the renderer itself.

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

Possible external implementations include:

```text
ForwardPipeline
DeferredPipeline
TilePipeline
PathTracingPipeline
2DPipeline
project-specific pipelines
```

Fluxel core does not require or provide one of these as the default.

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

The future `fluxel-renderer` core owns only:

```text
RenderScene vocabulary
RenderObject vocabulary
RenderView vocabulary

renderer-domain identities

CullingService
SortingService

material-variant and shader/pipeline resolution service seams

evidence-driven draw preparation service seams

FramePipeline SPI
FramePipelineContext

RenderGraph authoring and direct RHI integration

renderer-level errors and results
```

It does not own a built-in rendering strategy.

There must be no mandatory:

```text
Forward
Deferred
PBR
Lambert
Shadow
Bloom
TAA
fixed FrameGraph
fixed raster recipe
```

inside the renderer core.

---

## 4. RenderScene

### 4.1 Purpose

`RenderScene` is a rendering-only scene representation.

Conceptually:

```rust
pub struct RenderScene {
    objects: Vec<RenderObject>,
    lights: Vec<RenderLight>,
}
```

The exact storage strategy is not frozen here.

`RenderScene` should be:

- backend-independent;
- independent of RenderGraph resources;
- independent of native GPU objects;
- read-mostly during frame construction;
- suitable for multiple views;
- independent of gameplay scene layout.

### 4.2 RenderObject

Conceptual shape:

```rust
pub struct RenderObject {
    id: RenderObjectId,

    transform: RenderTransform,
    bounds: RenderBounds,

    geometry: GeometryHandle,
    material: MaterialHandle,

    visibility: VisibilityFlags,
    layers: RenderLayerMask,
    queue: RenderQueue,
}
```

This describes responsibility, not frozen Rust ABI.

### 4.3 Identity

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderObjectId(u64);
```

`RenderObjectId` is not required to equal:

```text
ECS Entity
GameObject ID
AssetId
RHI ObjectId
GPU address
```

The integration layer owns the mapping from game objects to render objects.

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
animation
bones
physics
constraints
```

must resolve their current rendering transform before or during `RenderScene`
extraction/update.

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
    /* opaque */
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
```

A renderer may assign a separate identity to a `MaterialInstance`, because an
instance is mutable renderer-domain state rather than the material asset. That
instance must retain its material asset reference; it must not replace or mint a
parallel durable material identity.

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
    objects: Vec<RenderObjectId>,
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

Batching and pipeline/material-aware sorting should be added only after the
material/shader system provides a real key model.

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

This layer must not be invented prematurely before the material/shader contracts
are ready.

---

## 14. FramePipeline SPI

The core SPI is intentionally small.

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

## 15. Policy vs Mechanism

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

## 16. RenderGraph Authoring

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

No hidden RHI submission path is allowed beside RenderGraph.

---

## 17. FramePipeline Output

The graph remains the main product.

The pipeline may additionally return semantic frame outputs.

Conceptually:

```rust
pub struct FramePipelineOutput {
    pub color: Option<GraphTexture>,
    pub depth: Option<GraphTexture>,
}
```

or later:

```rust
pub struct FramePipelineOutput {
    pub exports: FrameExports,
}
```

The public shape should remain narrow until concrete consumers require more.

---

## 18. Multi-Pass Features

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

## 19. Relationship to slot-graph

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
all GPU work still becomes ordinary RenderGraph passes.

No `slot-graph` node or subgraph identity needs to cross the renderer public API.

---

## 20. Asset and GPU Residency Boundary

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

A device-local cache, if used, must key by enough identity to prevent stale or
cross-device reuse, conceptually including:

```text
logical asset identity
content generation
DeviceIdentity
```

Exact cache ownership belongs to the renderer/resource preparation layer, not
RenderGraph.

---

## 21. Resource Lifetime Boundary

Renderer preparation may retain device-local resources needed by a frame.

However:

```text
GPU completion
submission lifetime
retirement
device loss
```

remain authoritative at RHI.

Renderer code must not guess that a resource is safe to recycle merely because:

```text
the frame function returned
the graph object was dropped
the scene changed
the asset was replaced
```

Accepted GPU work remains alive until the corresponding RHI completion semantics
prove retirement is safe.

---

## 22. Material and Shader Boundary

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

RenderGraph does not compile material graphs or compose shader source.

RHI does not know materials.

---

## 23. No Built-In Pipeline

The renderer core must not contain a hidden mandatory graph such as:

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

Those are external `FramePipeline` implementations.

Possible ecosystem crates may later include:

```text
fluxel-renderer-forward
fluxel-renderer-deferred
fluxel-renderer-2d
project-specific renderer crates
```

This packaging is optional and does not affect the core SPI.

---

## 24. Minimal Example

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

## 25. Error Model

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

## 26. Suggested Crate Structure

```text
crates/renderer/src/
    lib.rs

    render_scene/
        mod.rs
        object.rs
        view.rs
        bounds.rs

    culling/
        mod.rs

    sorting/
        mod.rs

    preparation/
        mod.rs

    pipeline/
        mod.rs
        context.rs
        output.rs
        error.rs
```

The core renderer crate should not contain:

```text
fixed_frame/
fixed_recipe/
built_in_forward/
built_in_deferred/
default_frame_graph/
```

Historical fixed-renderer code may remain temporarily during migration, but it
must not define the post-foundation public architecture.

---

## 27. Relationship to Unity SRP

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

## 28. Frame Flow

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
completion / present / exported result
```

---

## 29. Historical 0.15 Renderer

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
- do not carry obsolete execution-plan, browser-token, or accepted-unknown
  semantics into the new foundation.

---

## 30. Final Contract

```text
RenderScene
    = rendering-only projection of the game/engine scene

Culling / Sorting
    = reusable renderer mechanisms

FramePipeline
    = project-defined policy that converts
      RenderScene + RenderView into RenderGraph

RenderGraph
    = frame work, resource dependencies, and execution planning

Shader / Pipeline + Material Runtime
    = prepared execution objects and per-instance binding data selected by a pass

RHI
    = portable GPU execution
```

The defining rule is:

> `fluxel-renderer` provides the scene vocabulary, preparation services, and SPI
> required to build a renderer; it does not provide a renderer strategy itself.
