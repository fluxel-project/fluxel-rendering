# Fluxel Rendering Plan

> Status: active execution plan. The completed RHI baseline is the foundation;
> its detailed behavior remains specified by the RHI design, ADRs, rustdoc,
> and contract tests. This plan sequences future work without assigning it to
> specific release numbers.

## Delivery model

Work moves from a narrow vertical proof toward broader product integration:

```text
completed RHI baseline
  -> material and shader
  -> minimal scene and Forward
  -> scene preparation and Deferred
  -> Blender authoring
  -> preview/runtime equivalence and export
  -> capture and replay
  -> JavaScript
  -> Canvas 2D and text
  -> declarative UI
```

Every plan needs a named owner, structured refusal behavior, retained tests and
evidence, and an ecosystem-level acceptance slice. A single crate's green test
suite cannot close a plan.

## Material and shader plan

Define `MaterialDomain::Surface`, typed material values and graph handles,
validation/diagnostics, Material IR, static versus dynamic parameters,
`MaterialAsset`, and `MaterialInstance`. The initial node vocabulary is kept
small: constants, parameters, UVs, texture sampling, arithmetic, vector
operations, normal mapping, and surface outputs for base color, metallic,
roughness, normal, emissive, and opacity.

Define shader modules, pass templates, geometry/material/pass interfaces,
target profiles, feature analysis, canonical variant keys, source/IR assembly,
reflection into `ShaderInterface`, and portable artifact/cache identity.
Dynamic values update instance data or bindings; static choices affect variant
identity. Identical material structure with different dynamic values shares one
variant.

The acceptance proof is Rust-only:

```text
MaterialGraph -> Material IR -> shader composition
  -> ShaderArtifact + ShaderInterface
  -> compiled material variant -> renderer preparation
```

Unsupported material features return structured diagnostics rather than silent
approximations. Blender translation begins only after this proof.

## Minimal scene and Forward plan

Introduce the smallest `RenderScene`, `RenderObject`, and `RenderView`
vocabulary sufficient to freeze the consumer shape, a custom `FramePipeline`
SPI, and one built-in Forward proof.

The completion gate is an end-to-end Forward frame that resolves a material
variant, builds and compiles RenderGraph work through the direct portable RHI
contract, records/submits it, and retains deterministic validation evidence.
The exact layer contracts remain in the renderer, RenderGraph, material, and
RHI design documents.

## Scene preparation and Deferred plan

Complete renderer-facing scene preparation: stable scene identities,
visibility, culling, deterministic ordering, material/shader variant selection,
diagnostics, and retained fixtures. Add Deferred as a second, independent
`FramePipeline` proof after Forward. Both pipelines consume the same material
and shader contracts and execute through the same graph/RHI boundary.

## Blender authoring, equivalence, and export plans

The Blender plan provides ShaderNodeTree-to-MaterialGraph translation, mesh,
transform, camera, light, material assignment, and texture extraction;
structured unsupported-node/material/shader diagnostics; and Fluxel viewport
preview. Initial supported nodes are Material Output, Principled BSDF, Image
Texture, Texture Coordinate, Normal Map, RGB, Value, Add, Multiply, and Mix.
Coverage expands only with a Fluxel semantic and validation test.

The equivalence plan closes the authoring-to-runtime loop:

```text
Blender scene -> Fluxel import -> viewport preview
             -> export -> standalone runtime -> equivalence comparison
```

It retains deterministic comparison fixtures, content-generation handling, and
integration diagnostics.

## Capture and replay plan

Capture and replay the dependency-closed RenderScene path, including scene
inputs, view/frame configuration, material/shader decisions, pipeline choice,
RenderGraph inputs, portable execution evidence, and output observations.
Replay uses the normal renderer, RenderGraph, and RHI contracts; it is not a
second renderer. See [capture/replay design](design-capture-replay.md) for the
artifact and replay boundary.

## JavaScript, Canvas/text, and declarative UI plans

The JavaScript plan exposes a deliberately narrow integration surface over
established Rust contracts. JavaScript does not own GPU resources, shader
semantics, material identity, or the RenderScene model. A language-level SDK
core may be extracted once an established Rust contract and at least one real
adapter prove a stable boundary; future adapters must not push platform-specific
behavior into that core.

Canvas 2D and minimal text consume the renderer and prepared-resource model.
Declarative UI is built above those services. It may use Vue-inspired
ergonomics, but it is not Vue-compatible and does not import Vue
component/runtime semantics.

## Cross-plan invariants

- Rust owns material and shader semantics; Blender is an authoring frontend.
- The custom pipeline SPI and built-in pipelines share material/shader
  contracts.
- Unsupported features fail explicitly and never silently approximate.
- Cross-repository claims require evidence from an integrated artifact, not
  only a local library test.
