# Fluxel Rendering Product Plan

> Status: active product direction. This document sequences product outcomes;
> it does not freeze public API detail. Normative low-level contracts remain in
> the RHI design, RenderGraph design, ADRs, and the
> [implementation plan](version-plan.md).

## Product direction

Fluxel is a Rust rendering framework whose primary authoring experience is
native Blender integration. Rust owns rendering semantics, shader assembly,
material runtime, RenderGraph, and renderer execution. Blender is an authoring
and diagnostic frontend, not a second rendering authority.

The intended authoring loop is:

```text
Blender-authored material -> Fluxel material/shader semantics
    -> Fluxel renderer -> Blender preview and exported runtime
```

Preview and runtime consume the same Material IR, shader variant, parameters,
textures, and renderer path. Fluxel does not aim to reproduce EEVEE or Cycles.

Architecture and ownership are specified by the
[workspace architecture](design-overview.md), [renderer](design-renderer.md),
[RenderGraph](design-rendergraph.md), [material](design-material.md), and
[RHI](../crates/rhi/documents/design-rhi.md) designs. This product plan only
orders the outcomes that exercise those contracts.

## Delivery order

The completed RHI baseline is the foundation. Future work proceeds through
named plans rather than release-number promises:

1. **Material and shader plan.** Define MaterialGraph, Material IR, runtime
   instances, shader modules, composition, reflection, variants, and cache
   identity; prove one Rust-only material-to-shader vertical slice.
2. **Minimal scene and Forward plan.** Establish the smallest
   RenderScene/RenderView/RenderObject vocabulary, the `FramePipeline` SPI,
   direct RenderGraph/RHI integration, and one Forward implementation.
3. **Scene preparation and Deferred plan.** Complete culling, deterministic
   ordering, material selection, diagnostics, and Deferred as an independent
   pipeline proof.
4. **Blender authoring plan.** Translate supported Blender data, provide
   diagnostics and viewport preview, and establish the initial export route.
5. **Equivalence and export plan.** Prove preview/runtime equivalence with
   retained fixtures and integration diagnostics.
6. **Capture and replay plan.** Record and replay dependency-closed
   RenderScene execution through the normal renderer, graph, and RHI path.
7. **JavaScript plan.** Expose a narrow language-level integration surface over
   established Rust contracts. A shared SDK core may be extracted once an
   established Rust contract and a real adapter prove a stable language boundary;
   later adapters must not force platform behavior into that core.
8. **Canvas and text plan.** Add Canvas 2D and minimal text as consumers of the
   same renderer and prepared-resource model.
9. **Declarative UI plan.** Build declarative UI on Canvas and text. It may seek
   Vue-inspired ergonomics, but it is neither Vue compatibility nor a DOM/CSS
   runtime.

Each plan closes only with an integrated, end-to-end evidence slice. A green
test suite in one crate does not establish an ecosystem contract.

## Product milestone

```text
Create a cube and material in Blender
  -> import supported nodes
  -> build Material IR and shader variant
  -> preview through a Fluxel built-in pipeline
  -> export Fluxel assets
  -> render the same scene in a standalone Rust runtime
  -> compare preview and runtime output
```

After this works, expand node coverage, renderer features, and custom pipeline
examples incrementally. Do not create a second shader implementation for
Blender or for individual built-in renderers.
