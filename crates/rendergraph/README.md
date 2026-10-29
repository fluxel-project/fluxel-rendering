# fluxel-rendergraph

> Current package and migration guide. The target moves Graph into
> `fluxel-renderer::graph` and retires this standalone package as real consumers
> migrate. Preserve graph semantics and evidence; do not build a second engine.
> This document does not claim the move has already happened.

`fluxel-rendergraph` plans portable GPU work for Fluxel. A frame pipeline
declares passes, logical resources, versions, and accesses; the graph derives
dependencies, validates definedness, culls dead work, analyzes lifetimes, and
produces a deterministic plan.

It directly depends on [`fluxel-rhi`](https://github.com/fluxel-project/fluxel-rhi) for portable GPU
vocabulary. RHI supplies descriptors, formats, usage flags, resource uses,
capability facts, recording scopes, resources, pipelines, bindings,
submission, and presentation. RenderGraph adds only graph-specific meaning; it
does not duplicate those APIs or access backend-private native objects.

Read [the RenderGraph design](../../documents/design-rendergraph.md) for the
target architecture and [the implementation plan](../../documents/version-plan.md)
for delivery order. This README also records the historical `0.15` examples and
fixed execution evidence; they do not freeze the next public graph API.

## Target frame integration

```text
RenderScene -> FramePipeline -> RenderGraph
        -> Shader / Pipeline + Material Runtime -> RHI

CompiledGraph + per-frame RHI bindings
        -> RHI SubmissionPlan + explicit transitions
        -> deferred recording / immediate issue -> outcomes + completion
```

Material and shader/pipeline requirements are resolved by `FramePipeline`
before pass declaration. At execution, a pass selects the prepared
shader/pipeline and binds material-runtime data through graph pass-local
authority. Imports may be reusable logical slots, but each frame can bind
them directly to the selected RHI's buffers, textures, pipelines, bindings,
and acquired surface-image attachment API. The current independent RHI uses
`SurfaceImage` and `AttachmentView`; its presentation lease is not an ordinary
owned texture. Historical `FrameAttachment` names do not define that new API.

`CompiledGraph` is associated with the current device and capability
environment. On device loss or replacement, compile again. Do not build a
second capability profile or promise cross-device graph reuse until evidence
shows that the optimization is needed.

## What the graph owns

- logical resources, versions, definedness, and typed pass-local access;
- declared reads/writes, explicit dependencies, and RAW/WAR/WAW analysis;
- roots, dead-pass culling, deterministic scheduling, and diagnostics;
- logical lifetimes and alias opportunities.

Graph supplies portable RHI transition, dependency, and allocation decisions.
RHI realizes allocation and lowers native synchronization privately. Dedicated
transient allocation is valid baseline behavior; aliasing is optional.

## Pass recording

Each pass records through graph pass-local authority, including its pass-local
resolver, not a bare mutable RHI recorder. The authority holds the
corresponding RHI raster, compute, or copy scope, enforces the pass's declared
resource access and command family, and accepts only resources, pipelines, and
bindings resolved for that pass. Its callback-facing operations reuse RHI's
draw/dispatch/copy/pipeline/binding command vocabulary and semantics directly;
the graph must not create a second command language for later translation.

## Historical examples and verification

The numbered examples remain useful implementation evidence for the retained
declaration/compiler contract:

```sh
cargo run -p fluxel-rendergraph --example 00_minimal_compile
cargo test -p fluxel-rendergraph
```

Current fixed native evidence is intentionally limited. It is not a general
shader or pipeline promise; see RHI documentation and release conformance
evidence for supported backend details.

## License

Licensed under either [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
