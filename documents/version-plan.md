# Fluxel rendering version plan

> Status: active execution plan from the completed `0.16` baseline.
>
> `0.16` is complete: the declared RHI contracts and backend work are closed
> for the current release. This plan does not reopen that foundation. Detailed
> RHI behavior remains specified by `crates/rhi/documents/design-rhi.md`, its
> ADRs, and contract tests.

## 1. Delivery model

The plan becomes intentionally broader as it moves away from the current
release:

```text
0.16  completed RHI baseline
  |
0.17  shader assembly + material system        detailed
  |
0.18  RenderGraph + renderer framework         detailed
  |
0.19  RenderScene                              detailed
  |
0.20  RenderScene recording + replay           detailed
  |
0.21  Blender-native plugin/tooling loop       detailed
  |
0.22+ ecosystem extensions and candidates      coarse
```

Every version must still have a named owner, structured refusal behavior,
tests, retained evidence, and an ecosystem-level acceptance slice. A version
is not closed by a single crate's green test suite.

## 2. Version 0.16 — completed RHI baseline

### Scope

The portable RHI contract, device/context identity and generation rules,
resource ownership, bindings, command recording, submission/completion, loss
behavior, presentation boundary, and declared backend evidence.

### Status

Complete for the current release. Do not add new RHI concepts merely to start
the higher-level work. Any contract change requires a new consumer, ADR,
migration plan, and affected backend evidence.

## 3. Version 0.17 — shader assembly and material system

This is the next implementation priority. Keep the scope narrow enough to
produce one complete Rust-to-shader vertical slice.

### Material work

- Define `MaterialDomain::Surface`, typed material values, typed graph handles,
  diagnostics, and graph validation.
- Implement the minimal `MaterialGraph` node vocabulary: constants,
  parameters, UVs, texture sampling, arithmetic, vector operations, and normal
  mapping.
- Define `SurfaceOutput` for base color, metallic, roughness, normal, emissive,
  and opacity.
- Lower a validated graph into a semantic `Material IR`.
- Separate dynamic instance data from static variant decisions.
- Define `MaterialAsset`, `MaterialInstance`, parameter schema, and cache
  identity without making GPU objects part of the material model.

### Shader work

- Define shader modules, pass templates, geometry/material/pass interfaces, and
  target profiles.
- Analyze required features and generate canonical material variant keys.
- Assemble source/IR into a validated `ShaderArtifact` and reflected
  `ShaderInterface` accepted by the RHI.
- Establish portable artifact and variant cache identity.
- Prove that materials with identical structure but different dynamic values
  share one variant.

### Acceptance

```text
Rust MaterialGraph
  -> Material IR
  -> shader composition
  -> ShaderArtifact + ShaderInterface
  -> CompiledMaterialVariant
  -> MaterialInstance consumed by renderer preparation
```

Blender import is not part of this version. The complete proof is Rust-only;
Blender translation begins in `0.21`.

## 4. Version 0.18 — RenderGraph and renderer framework

### Scope

Build on the `0.17` material/shader contracts and connect renderer policy to
RenderGraph:

- a custom `FramePipeline` SPI for application-defined pipelines;
- a built-in Forward pipeline;
- a built-in Deferred pipeline;
- RenderGraph lowering for all pipeline routes.

### Acceptance

The custom SPI and both built-in pipelines lower representative work to
RenderGraph using the same compiled material variant and shader interface. The
full retained scene model is defined in `0.19`.

## 5. Version 0.19 — RenderScene

### Scope

Define the renderer-facing scene and frame-preparation model:

- `RenderScene`, `RenderObject`, `RenderView`, and stable scene identities;
- culling, deterministic ordering, visibility, and frame preparation;
- material/shader variant selection;
- scene-to-`FramePipeline`-to-RenderGraph construction;
- renderer diagnostics and retained scene fixtures.

### Acceptance

One complete Rust scene can be prepared, lowered through either built-in
pipeline or the custom SPI, and executed through RenderGraph and RHI.

## 6. Version 0.20 — RenderScene recording and replay

Record and replay the complete RenderScene-driven path, including scene inputs,
view/frame configuration, material/shader decisions, pipeline selection,
RenderGraph inputs, portable execution evidence, and output observations.
Replay uses the normal renderer, RenderGraph, and RHI contracts.

## 7. Version 0.21 — Blender-native editor and tooling

Start Blender integration only after the Rust rendering path and its
RenderScene recording/replay evidence are complete:

- ShaderNodeTree to `MaterialGraph` translation;
- mesh, transform, camera, light, material assignment, and texture extraction;
- structured unsupported-node and shader/material diagnostics in Blender;
- Fluxel viewport preview through the renderer;
- initial export of material, mesh, texture, and scene assets.

The first supported node subset is Material Output, Principled BSDF, Image
Texture, Texture Coordinate, Normal Map, RGB, Value, Add, Multiply, and Mix.
Each supported node requires a Fluxel semantic and a validation fixture.

## 8. Version 0.22 and later — broad extension direction

Keep later planning coarse until the first product loop is proven. Possible
extensions include additional Blender semantics, more renderer features,
custom-pipeline samples, stronger shader caching, animation/lighting breadth,
and broader backend evidence.

JavaScript material APIs, CSS-like styling, Canvas/UI frameworks, browser-first
editor UX, and general DevTools remain candidates. They are not release goals
in this plan and require a demonstrated consumer after the Blender/material/
renderer route is stable.

## 8. Cross-version invariants

- Rust owns material and shader semantics; Blender is an authoring frontend.
- Renderers lower through RenderGraph and execute through the same RHI model.
- The custom pipeline SPI and built-in pipelines share material/shader
  contracts.
- Device/context identity plus generation is the only public resource-affinity
  model; browser sessions/tokens never enter the architecture.
- Unsupported features fail explicitly and do not silently approximate.
- Cross-repository claims require evidence from the integrated artifact, not
  only a local library test.
