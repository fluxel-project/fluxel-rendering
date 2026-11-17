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
the same pass; and a selected Deferred scene path with fixed lighting and
UV0-normal metallic/roughness draws over distinct geometry, material, and
texture assets. Legacy single-mesh material routes and optional position
deformation remain available with their existing scope.
The external position deformer supplies a compute function and typed parameters;
Graph orders its GPU position-buffer production before geometry vertex fetch,
and Renderer retains the product until completion.
The bounded captured-color consumer also renders a producer and a Material
receiver in one Graph submission. The receiver samples the producer's actual
GPU target rather than a readback upload; completion retains both frame owners.
Its Surface route adds a GPU composite into the acquired presentation lease in
the same Graph, retaining the composite and lease through covering completion.
The `02-texture-material`, `04-dynamic-uniform-material` and
`07-offscreen-material` examples provide bounded numerical and real-window
consumers. The cube grid selects an optional Renderer-owned Depth32Float
attachment; disabled Forward consumers retain their existing requirements.
RGBA8 assets explicitly select linear or sRGB input encoding. Actual texture
and view formats preserve that choice; sRGB RGB decoding precedes filtering,
while alpha remains linear. The 02 consumer's `--srgb-filter` control exercises
this boundary separately from output transfer or tone mapping.
The `pbr-environment-material` consumer separately exercises selected split-sum
specular IBL using a complete RGBA16F radiance mip chain, BRDF LUT, and explicit
shading camera. Its orthographic plane controls compare independent numeric
predictions and actual window captures on both pipelines and native backends;
they do not prove environment prefiltering or physical LUT generation.
`pbr-sphere-material` extends this to a perspective six-sphere mesh with packed
ORM, actual tangent-space normals, and AO/camera controls. Its independent CPU
triangle oracle accounts for each pipeline's G-buffer precision, while its real
window captures compare densely with same-input readbacks. `--multi-records`
uses six independent local meshes, textures, and model matrices, including a
material with neutral surface effects. This exercises the supported shared
G-buffer family; arbitrary mixed lighting families remain outside that scope.
Broader material and pipeline coverage still needs its own evidence.
Materials can supply multiple `ForwardTextureBinding` entries by resource name.
The four-texture PBR consumer combines base, normal, ORM and emissive images,
shared assets and an alias in two real scene records. Bindings and Graph reads
follow the prepared schema, while physical residency uses exact asset keys.
The same example's `--mixed-capture` mode combines receiver base/normal/ORM
assets with a same-frame GPU capture as emissive, through Forward. The producer
slot is independently named; Graph imports shared owners once and checks every
owner before binding deduplication. Completed assets can be reused; capture
products remain frame-owned. `submit_deferred_capture` and
`submit_deferred_capture_surface` support a Deferred capture producer with an
opaque/masked G-buffer prefix and an optional sorted transparent overlay.
`submit_deferred_oit_capture` and `submit_deferred_oit_capture_surface` select
independent producer and receiver peeling, followed by excluded sorted blends. The receiver
samples the actual latest scene-color version or independent final OIT output
alongside immutable assets. Opaque/masked receivers retain their Deferred path;
Blend/MaskBlend receivers use an independent Forward target and depth attachment
in the same Graph, with optional receiver peeling.
Both sides execute in one Graph submission. Offscreen reads the receiver final
color; Surface composites to the actual lease and presents without readback.
All frames, optional OIT products, and failed/unhanded leases remain pending
through receiver-terminal completion. No peel candidates means no OIT admission.
A producer without opaque/masked draws uses its already-prepared Forward
namespace and actual ordinary or OIT final product. Opaque/masked receivers use
a Deferred namespace; blend receivers retain the existing two-Forward route.
Typed capture Features and behavior acceptance remain open. Shared immutable and captured inputs resolve to their
validated canonical Graph owners during OIT recording.
The explicit `ForwardCpuPipeline::with_sorted_transparency()` product places
opaque/masked draws before one globally sorted transparent range. Material
`SurfacePolicy` selects alpha masking, blending, culling, and two-sided lighting;
transparent draws use Babylon-compatible alpha composition and depth testing
without depth writes. A perspective camera supplies the sorting eye;
`ShadingCamera::orthographic_at` supplies it for parallel lighting. The selected
Deferred `submit_deferred` route partitions the same definitions: opaque/masked
draws create and light the G-buffer, while Blend/MaskBlend draws use the global
sorted Forward overlay with borrowed scene color (ColorRW) and G-buffer depth.
An empty opaque partition is a meaningful ordinary Forward path; no transparent
partition retains the existing Deferred route. The overlay neither copies nor
allocates replacement color/depth attachments; offscreen reads its final color,
and the Surface route composites/presents in the same Graph without a readback
pass. `submit_deferred_oit` and `submit_deferred_oit_surface` extend that same
boundary when an opaque prefix and peel candidates coexist: G-buffer and
lighting remain the prefix, scene color is a sampled read-only OIT input, and
the original G-buffer depth is read-only. Peeling writes its independent final
owner, then any excluded sorted blend suffix loads that owner before the normal
offscreen or Surface terminal. Empty opaque input takes the already-prepared
Forward route; zero peel candidates do not admit OIT. The combined submission
retains opaque and overlay frames, OIT owners, composite, publication and an
unhanded/failed presentation binding through covering completion. Opaque
transmission/scene-color refraction and capture typed Features remain open. All of this awaits milestone behavior
acceptance.
`Renderer::submit_forward_oit` and `submit_forward_oit_surface` provide scene
CPU Forward entries with `DualDepthPeelingConfig` (five rounds by default).
They keep opaque depth, peeling and excluded sorted draws in one Graph. The
terminal either reads back the final output or composites it to the acquired
Surface and presents in the same submission without CPU readback. Owners remain
retained through covering completion. These routes have static compilation
coverage; behavior acceptance remains open.
`Renderer::submit_forward_oit_capture` and
`Renderer::submit_forward_oit_capture_surface` keep producer and receiver in
one Graph submission with independent OIT selection on each side. An OIT
producer aliases its final composited output—not its opaque attachment or an
intermediate—to the receiver. The receiver's final output is either read back
or composited into the acquired Surface image and presented in that submission;
the latter path has no readback pass or ticket. It reuses the offscreen frame
allocation, so unused readback-buffer capacity can remain, but is never bound,
copied or mapped; it retains both frames, OIT owners, composite and an
unhanded/failed present binding through completion.
The composite consumes the actual Surface format, including BGRA/sRGB variants.
This has static code coverage only. OIT capture typed Features and behavior
acceptance remain open.
The `submit_forward_oit_with_feature` / `submit_forward_oit_with_features` APIs
and their Surface counterparts compose scene draws with one or two concrete
typed features. Each actual opaque, initial-depth, peel and excluded sorted
pass invokes `RenderFeature::contribute` in that feature's stage intersection.
Missing, repeated or out-of-range reads refuse the Graph declaration. The
excluded transparent suffix is globally sorted across every contributor.
Forward UV0-normal PBR materials can request environment refraction and
thickness-dependent Beer-Lambert attenuation through `TransmissionExtensions`.
This requires the selected specular radiance/LUT and camera, and consumes that
environment directly. Scene-color refraction and Deferred transmission still
need their dedicated composition paths; behavior acceptance is pending.
The pair APIs `submit_forward_with_features` and
`submit_forward_surface_with_features` accept two separately typed feature
bindings. They validate each feature identity and snapshot, merge their draws
with scene draws before ordering, and require independent Graph read scopes.
Their behavior acceptance remains part of the material milestone.
They do not form a general programmable pipeline, arbitrary cross-pass Feature
model, runtime plugin system, general transparent compositor, complete PBR system,
Forward+, shadow, PostFX, streaming, mesh LOD, or GPU-culling API.

Features and frame policies are statically composed with Rust generics and
concrete types. Runtime data may vary, but trait-object dispatch, erased
registries, and a second scheduler are excluded.

Read the durable design in [documents/design.md](documents/design.md). Unit
tests live under `src/tests/`; crate-root `tests/` is reserved for public
integration contracts. Independent consumers are in `examples/` and exercise
the public boundary without accessing Graph internals.

The next material milestone plans shared node-authored PBR definitions across
Forward and Deferred, including specialized transparent and refractive passes.
Its first acceptance matrix is DX12/WGL. Full delivery and acceptance remain open;
see [execution design](documents/design.md#planned-material-effects-and-composition)
and the [ecosystem roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md).

Material textures can request a generated linear or sRGB RGBA8 mip chain through immutable
`mip_levels`. Graph declares the base upload and every level's blit before full
view sampling, and completion includes the complete chain. The four-texture
example's `--mip-generation` slice exercises explicit LOD and trilinear sampling
through both pipelines. sRGB generation shares the format-preserving Graph
path; its behavior acceptance and the complete tunnel/filtering matrix remain
open.
