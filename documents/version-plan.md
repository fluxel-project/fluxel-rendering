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
0.18  minimal scene + Forward framework        detailed
  |
0.19  RenderScene                              detailed
  |
0.20  Blender-native plugin/tooling loop       detailed
  |
0.21  Preview/runtime equivalence and export   detailed
  |
0.22  RenderScene recording + replay           detailed
  |
0.23  JavaScript API interface                coarse
  |
0.24  Canvas 2D + minimal text                coarse
  |
0.25  Declarative Vue-like UI                  coarse
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

Before feature work closes, device construction must finalize and validate all
finite capability-query domains so no successfully published snapshot can
reach a query-time missing-entry panic. The renderer-owned private graph/RHI
bridge is the only integration owner; neither public crate depends on the
other.

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
Blender translation begins in `0.20`.

## 4. Version 0.18 — minimal scene and Forward framework

### Scope

Build on the `0.17` material/shader contracts and connect renderer policy to
RenderGraph:

- minimal `RenderScene`, `RenderObject`, and `RenderView` vocabulary sufficient
  to freeze the consumer shape;
- a custom `FramePipeline` SPI for application-defined pipelines;
- one built-in Forward pipeline proof; and
- renderer-owned, workspace-private lowering from graph IR to RHI
  `RecordedWork` and `SubmissionPlan`.

### Acceptance

The minimal scene and custom SPI lower representative Forward work through the
same compiled material variant and shader interface. RenderGraph remains a pure
compiler/IR and RHI remains pure execution; neither depends on the other.

## 5. Version 0.19 — RenderScene

### Scope

Complete the renderer-facing scene and frame-preparation model:

- `RenderScene`, `RenderObject`, `RenderView`, and stable scene identities;
- culling, deterministic ordering, visibility, and frame preparation;
- material/shader variant selection;
- scene-to-`FramePipeline`-to-RenderGraph construction;
- renderer diagnostics and retained scene fixtures.
- Deferred as a second, independent `FramePipeline` proof after Forward.

### Acceptance

One complete Rust scene can be prepared, lowered through either built-in
pipeline or the custom SPI, and executed through RenderGraph and RHI.

## 6. Version 0.20 — Blender-native editor and tooling

Start Blender integration after the Rust rendering path and RenderScene model
are complete:

- ShaderNodeTree to `MaterialGraph` translation;
- mesh, transform, camera, light, material assignment, and texture extraction;
- structured unsupported-node and shader/material diagnostics in Blender;
- Fluxel viewport preview through the renderer.

## 7. Version 0.21 — Preview/runtime equivalence and export

Close the authoring-to-runtime loop:

```text
Blender scene -> Fluxel import -> viewport preview
             -> export -> standalone runtime -> equivalence comparison
```

Add deterministic comparison fixtures, asset-version handling, and integration
diagnostics. The first supported node subset is Material Output, Principled
BSDF, Image Texture, Texture Coordinate, Normal Map, RGB, Value, Add,
Multiply, and Mix.

## 8. Version 0.22 — RenderScene recording and replay

Record and replay the complete RenderScene-driven path, including scene inputs,
view/frame configuration, material/shader decisions, pipeline selection,
RenderGraph inputs, portable execution evidence, and output observations.
Replay uses the normal renderer, RenderGraph, and RHI contracts.

## 9. Version 0.23 — JavaScript API interface

Expose a deliberately narrow JavaScript API over the established Rust
contracts. JavaScript is an integration surface; it does not own GPU resources,
shader semantics, material identity, or the RenderScene model.

## 10. Version 0.24 — Canvas 2D and minimal text

Define Canvas 2D and minimal text on top of the Fluxel renderer and prepared
resources. They remain consumers of the same RenderGraph/RHI architecture and
must not introduce DOM, CSS, browser session, or token-based resource ownership
into the core.

## 11. Version 0.25 — Declarative UI

Build the declarative Vue-like UI layer as a consumer of the established Canvas
and text services. It does not redefine rendering, resource ownership, or the
graph/RHI boundary.

## 12. Later evolution

Later work expands Blender semantics, renderer features, custom-pipeline
examples, shader caching, animation, lighting, and backend evidence as the
completed route requires.

## 13. Cross-version invariants

- Rust owns material and shader semantics; Blender is an authoring frontend.
- RenderGraph is a pure graph compiler/IR and RHI is pure execution. Their
  bridge/lowering is renderer-owned and workspace-private.
- The custom pipeline SPI and built-in pipelines share material/shader
  contracts.
- Device/context identity plus generation is the only public resource-affinity
  model; browser sessions/tokens never enter the architecture.
- Unsupported features fail explicitly and do not silently approximate.
- Cross-repository claims require evidence from the integrated artifact, not
  only a local library test.
