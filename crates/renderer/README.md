# Fluxel Renderer

`fluxel-renderer` owns render-domain scene storage, preparation, frame policy,
the private RenderGraph, concrete binding validation, residency, transient frame
data, recording, submission coordination, and completion-owned retirement. It
uses `fluxel-material` for authoring semantics, `fluxel-shader` for logical
program assembly, and `fluxel-rhi` for native resource realization, command
recording, submission, completion, and presentation.

The public `RenderScene` API provides generational render/view identities,
transactional change sets, immutable snapshots, finite transforms and bounds.
It performs no GPU allocation or Graph work by itself. Public CPU Feature and
FramePipeline seams are controlled: they prepare typed data and select supported
products, but cannot obtain private Graph state, command encoders, allocators,
or submission authority. Compile-fail consumers enforce that boundary.

Frame work has one route:

```text
scene and feature data -> preparation and frame policy -> Graph declaration
-> concrete owner/binding validation -> RHI recording and submission
-> completion evidence -> frame-data and residency retirement
```

Graph declarations remain required even when a compatible device plan is cached.
Before allocation, upload, recording, or submission, the current owner
descriptor, range, incoming state, and RHI device provenance are validated.
Persistent residency and per-frame data remain separate; submitted owners retire
only after covering completion evidence. Device loss rejects new work and clears
objects that cannot safely be reused.

Current supported consumers include bounded Forward preparation for multiple
meshes, layouts, materials, and textures; a controlled opaque typed Feature in
the same pass; and a selected Deferred path with one mesh, fixed lighting,
optional position deformation, and selected metallic/roughness material output.
The external position deformer supplies a compute function and typed parameters;
Graph orders its GPU position-buffer production before geometry vertex fetch,
and Renderer retains the product until completion.
They do not form a general programmable pipeline, arbitrary cross-pass Feature
model, runtime plugin system, transparent compositor, complete PBR system,
Forward+, shadow, PostFX, streaming, LOD, or GPU-culling API.

Features and frame policies are statically composed with Rust generics and
concrete types. Runtime data may vary, but trait-object dispatch, erased
registries, and a second scheduler are excluded.

Read the durable design in [documents/design.md](documents/design.md). Unit
tests live under `src/tests/`; crate-root `tests/` is reserved for public
integration contracts. Independent consumers are in `examples/` and exercise
the public boundary without accessing Graph internals.
