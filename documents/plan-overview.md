# Fluxel Rendering Roadmap and System Plan

> Status: active product direction
>
> This is the high-level plan for the Fluxel rendering ecosystem. Normative
> low-level contracts remain in `design-rhi.md`, the RenderGraph design, the
> ADRs, and `version-plan.md`.

## 1. Product direction

Fluxel is a Rust rendering framework whose primary authoring experience is
native Blender integration. Rust owns rendering semantics, shader assembly,
material runtime, RenderGraph, and renderer execution. Blender is the editor
and authoring environment, not a second rendering authority.

The acceptance criterion is:

```text
Blender-authored material -> Fluxel material/shader semantics
    -> Fluxel renderer -> Blender preview and exported runtime
```

Preview and runtime consume the same Material IR, shader variant, parameters,
textures, and renderer path:

```text
Fluxel in Blender == Fluxel in the standalone runtime
```

The goal is not to reproduce EEVEE or Cycles.

## 2. Main architecture

```text
Blender native add-on and tools
  material translation / preview / export / diagnostics
                |
                v
fluxel-material: MaterialGraph -> Material IR -> assets/instances
                |
                v
fluxel-shader: modules/templates -> features -> ShaderArtifact/variant
                |
                v
fluxel-renderer: RenderScene -> preparation -> FramePipeline
  custom pipeline SPI + built-in Forward + built-in Deferred
                |
                v
fluxel-rendergraph: logical passes/resources -> execution plan
                |
                v
fluxel-rhi: device identity -> bindings -> commands -> completion
```

The public architecture contains RenderGraph resources, RHI bindings,
Device/Surface/Completion, and generation-aware ownership. WebGPU, WebGL,
native handles, and reference-counted leases remain backend-private. No public
or backend resource model uses browser sessions or asset tokens.

## 3. Workspace and ecosystem ownership

```text
fluxel-rendering/
  crates/rhi/          portable RHI and backend execution
  crates/rendergraph/  graph authoring, compilation, and execution plans
  crates/material/     material graph and Material IR
  crates/shader/       shader assembly, reflection, and variants
  crates/renderer/     scene preparation and FramePipeline contracts
  tools/blender/       native Blender add-on/tooling package
```

`material`, `shader`, and Blender tooling become public boundaries only after
working vertical slices prove their need. A proposed crate is not itself a
commitment to an API.

`fluxel-bases` remains the home of shared platform-neutral mechanisms;
`fluxel-host` owns native host lifecycle; `fluxel-jsbridge` remains a platform
and JavaScript adapter. None of them is the semantic authority for rendering.

## 4. Delivery sequence

The current `0.16` release is the completed RHI baseline. It remains the
lower-layer contract for the next train:

```text
0.16  completed RHI protocol and backend baseline
```

The next product train is:

```text
0.17  Shader assembly + material system
  -> MaterialGraph, Material IR, runtime instances
  -> shader modules, composition, reflection, variants, and cache
0.18  RenderGraph + Renderer Framework
  -> custom FramePipeline SPI
  -> built-in Forward and Deferred pipelines
  -> renderer lowering into RenderGraph
0.19  RenderScene
  -> scene/object/view model, preparation, culling, and ordering
  -> material/shader selection and frame construction
0.20  Blender-native authoring and preview loop
  -> native add-on/tools, viewport preview, export, runtime equivalence
0.21  Preview/runtime equivalence, export, and integration hardening
0.22  RenderScene recording and replay
0.23  JavaScript API interface
0.24  Declarative Vue-like UI framework + Canvas 2D API
```

These versions describe sequencing, not a promise that every item fits one
release. Each train requires an end-to-end evidence slice before expansion.

## 5. Next version: shader assembly and material system

This is the immediate priority after the completed `0.16` RHI baseline.

### 5.1 Material system

Define and implement:

```text
MaterialDomain::Surface
MaterialValueType and typed values
MaterialGraph and typed node handles
graph validation and diagnostics
Material IR and graph lowering
static versus dynamic parameters
MaterialAsset / MaterialInstance
SurfaceOutput contract
```

The first vocabulary is small: constants, parameters, UVs, texture sampling,
arithmetic, vector operations, normal mapping, and surface outputs for base
color, metallic, roughness, normal, emissive, and opacity. Blender compatibility
maps into these semantics; it does not define them.

### 5.2 Shader assembly

Define and implement:

```text
shader module and pass-template contracts
geometry/material/pass interfaces
feature analysis and canonical variant keys
shader composition and source/IR generation
reflection into ShaderInterface
ShaderArtifact production and validation
portable shader cache identity
```

Dynamic values update bindings or instance data. Static choices affect the
variant key. Identical material structure with different runtime values shares
one shader variant.

### 5.3 Acceptance

The first proof is Rust-only:

```text
Rust MaterialGraph -> Material IR -> composed artifact
    -> reflected interface -> compiled material variant
    -> renderer-facing material instance
```

Blender integration is not part of `0.17`; it begins in `0.21`. The `0.17`
proof is Rust-only. Unsupported material features fail with structured
diagnostics; they are never silently approximated.

## 6. RenderGraph and renderer direction

`fluxel-renderer` is a framework for renderer policy, not one fixed recipe. It
provides `RenderScene`, object/view preparation, culling, deterministic
ordering, material/shader variant selection, a `FramePipeline` SPI, RenderGraph
lowering, and renderer diagnostics.

The custom SPI allows applications and tools to define pipelines while keeping
the same scene, material, shader, RenderGraph, and RHI contracts. Fluxel also
ships two initial built-in pipelines:

```text
Forward:  depth, ordering, lighting, and material evaluation
Deferred: G-buffer, lighting, material evaluation, and composition
```

Both consume the same compiled material variants and lower through RenderGraph.

## 7. RenderScene and recording/replay

`0.19` defines the complete renderer-facing scene and frame-preparation model:

```text
RenderScene -> RenderView/RenderObject -> culling/order
            -> material/shader selection -> FramePipeline -> RenderGraph
```

`0.20` applies recording and replay to the whole RenderScene-driven path. The
recording boundary includes scene inputs, material/shader decisions,
frame/pipeline configuration, and portable execution evidence. Replay uses the
normal renderer, RenderGraph, and RHI path rather than a second implementation.

## 8. Blender-native editor/tooling direction

Starting in `0.20`, Blender integration is a first-class product path, not a late JavaScript
facade. The native add-on/tool package progressively provides:

```text
Blender shader-node -> Fluxel MaterialGraph translation
scene/mesh/material extraction
Fluxel material and shader diagnostics in Blender
Fluxel viewport preview through the Fluxel renderer
asset export and rebuild reporting
preview/runtime equivalence fixtures
```

The initial supported nodes are Material Output, Principled BSDF, Image
Texture, Texture Coordinate, Normal Map, RGB, Value, Add, Multiply, and Mix.
Coverage expands only when a Fluxel semantic and validation test exists.

## 9. Later product route

After the Blender loop, the roadmap continues with preview/runtime equivalence,
export and integration hardening, then complete RenderScene recording/replay,
the JavaScript API interface, a declarative Vue-like UI framework, and a Canvas
2D API. These layers consume the established Rust rendering contracts; they do
not redefine material semantics, renderer pipelines, RenderGraph, or RHI.

## 10. Product milestone

```text
Create a cube and material in Blender
  -> import supported nodes
  -> build Material IR and shader variant
  -> preview through a Fluxel built-in pipeline
  -> export Fluxel assets
  -> render the same scene in a standalone Rust runtime
  -> compare preview and runtime output
```

After this succeeds, expand node coverage, renderer features, and custom
pipeline examples incrementally. Do not create a second shader implementation
for Blender or for either built-in renderer.
