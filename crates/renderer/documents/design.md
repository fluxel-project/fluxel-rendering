# Fluxel Renderer Design

`fluxel-renderer` integrates scene data, material and shader products, frame
policy, resource residency, and a private RenderGraph. It is the only layer in
this workspace that turns prepared rendering work into RHI recording and
submission. It does not own application state, an ECS, a host event loop,
native windows, browser sessions, or backend-native GPU objects.

The crate depends on [Shader](../../shader/documents/design.md) and
[Material](../../material/documents/design.md). Shader owns logical WGSL
composition and reflection; Material owns authoring semantics and generic
contributions. The selected RHI owns physical resource realization, recording,
submission, completion, presentation, device loss, and backend lowering.
Renderer owns the integration and the lifetime decisions that consume those RHI
facts.

## Scope and implementation boundary

`src/pbr/direct_light.rs` shares the selected Babylon directional-light
calculation across Forward, Deferred, and environment transmission. It uses
EON with default zero diffuse roughness, legacy reflectivity energy allocation,
glTF Schlick Fresnel, and height-correlated GGX visibility. The renderer supplies
a fixed light direction and color. Forward and Deferred consume the view's
perspective eye or orthographic direction. Basic PBR uses the historical +Z
direction when the view omits a shading camera. Deferred reconstructs world
position from depth for perspective lighting and reads the camera range from
its frame uniform buffer. Specular IBL supplies its camera through the existing
environment uniform.
Environment transmission selects a prefiltered 2D latitude-longitude source.

| Area | Current implemented subset | Not established by the subset |
| --- | --- | --- |
| Scene | Generational render and view identities; transactional batch changes; immutable snapshots; typed CPU feature storage | A general ECS integration or heterogeneous runtime registry |
| Forward | Multi-mesh preparation over geometry, materials and textures; controlled opaque and clear policy; residency and frame data | General SRP, transparent composition, arbitrary external GPU products |
| External composition | One statically typed opaque Forward Feature and external pipeline compose through the actual Graph | Dynamic plugins, arbitrary Graph access, a plugin ABI, general cross-pass Features |
| Deferred | Multiple UV0-normal metallic/roughness records with fixed direct lighting, selected diffuse/specular environment inputs, and compatible G-buffer packing; legacy single-mesh paths and optional position-only GPU deformation | Arbitrarily mixed lighting families, configurable direct lighting, general environment generation, arbitrary normal-map conventions or deformation |
| Optimisation | Private validation indices for the exercised Forward path | A public cache/index API, or claims about GPU duration, VRAM, or whole-frame speedup |

Future work starts from an actual producer and consumer. A new public extension
contract requires a demonstrated input/output shape, Graph declaration, binding
validation, failure cleanup, and completion ownership; architectural preference
alone is insufficient.

## Ownership model

```text
application / ECS / scene graph
  -> RenderChangeSet + feature-owned typed inputs
  -> RenderScene immutable FrameSnapshot
  -> FramePipeline and Feature preparation
  -> private Graph declarations and compiled device plan
  -> RHI recording, submission, Completion
  -> frame-data and residency retirement/reuse
```

`RenderScene` is a rendering-domain snapshot boundary, not an application object
world. It stores consumer-chosen cloneable resource bundles beside records,
finite transforms and bounds. Applications map their own entities to stable,
generational `RenderId` and `RenderViewId`; dense storage indices, asset IDs,
device IDs, and GPU addresses are different identities. Changes apply as one
transaction, reject stale/foreign IDs and conflicting writes, report exact dirty
ranges, and publish snapshots that remain valid after later mutation.

The application owns gameplay ECS scheduling, animation evaluation, source I/O,
and policy for selecting visible content. Renderer owns neither their thread
model nor a universal scene layout. Features may use dense columns, compact
records, GPU buffers, or another layout appropriate to their access pattern.
CPU scene layout, logical asset layout, and GPU vertex layout are independent.

Persistent asset identity is supplied by Bases as asset identity plus immutable
content generation. Per-device GPU residency is Renderer-owned and keyed by
actual content and device provenance. A cache hit still consumes this frame's
real Graph declaration and verifies the concrete owner before dependent GPU
work.

## Static extension composition

Extensions are ordinary Rust implementations assembled at compile time. All
`RenderFeature`, `FramePipeline`, local geometry, visibility, ordering,
batching, and deformation providers use generics, traits, and monomorphization.
There is no trait-object dispatch, erased plugin registry, hand-written vtable,
runtime-loaded unknown Feature, or global service locator in this architecture.

Typed tuples, ordinary aggregates, and associated types are sufficient. Runtime
enablement is state in a known concrete Feature; a runtime choice among known
implementations uses an explicit enum. Static implementation types do not make
scene objects, assets, materials, or Graph topology static types.

Public contexts are controlled capabilities. An external pipeline can choose one
of the Renderer-owned products and make the allowed phase declarations; an
external Feature can prepare its typed batch and contribute to the allowed
phase. Neither receives Graph construction, allocator, command encoder,
submission, surface lease, native handle, or scheduling authority. Opaque
handles and contexts validate scene, issuer, feature, snapshot version, phase,
and replay identity before use.

The stable direction is batch-oriented: a Feature exchanges typed input,
prepared products, and bounded access requirements. It is not a per-object
virtual `draw`, `cull`, or `sort` model. Cross-Feature ordering, particularly
for a future transparent phase, belongs to a frame-wide composition rule rather
than incidental Feature registration order.

## Geometry, materials, and preparation

Geometry describes streams and attributes, not one mandatory Rust vertex type.
Streams carry range, stride, step rate, and physical owner data; attributes carry
semantic identity, format, and offset. A mesh may have separate vertex streams,
indices, submesh ranges, material assignments, and feature-specific deformation
inputs. Preparation resolves semantic-to-location mappings against Shader
reflection and creates concrete bindings. It rejects missing or duplicate
semantics, invalid physical ranges, unsupported formats, and incompatible
material/layout requirements before recording.

Binding views and Graph access footprints are distinct. The RHI vertex binding
keeps the index origin required by the draw, while Graph declares the actual
physical fetch footprint. Dynamic uniform offsets and texture staging likewise
declare their actual physical ranges and layouts. Texture row pitch and texture
placement alignment are separate RHI constraints.

Material and Shader integration follows this path:

```text
Material definition and instance
  -> generic WGSL contribution and interfaces
  -> Renderer pass template, entry points, and vertex requirements
  -> Shader composition and reflection
  -> prepared concrete bindings and Graph access declarations
  -> RHI pipeline/resource realization
```

Renderer does not duplicate Material IR or Shader composition. Logical shader
validity does not prove hardware capability: the selected RHI route must admit
the exact requirements and device facts before realization and execution.
An optional capability is not a partial promise: creation, binding, recording,
and the selected backend lowering must all support the complete portable
contract before Renderer advertises or consumes it.

### Diffuse irradiance input

UV0 normal materials may supply optional scalar occlusion and linear emissive
evaluators. Forward evaluates them directly; Deferred uses a conditional
RGBA16F color attachment (emissive RGB, occlusion alpha), declared and bound
through the same Graph on offscreen and surface paths. Only those effect
materials require a third color attachment and device support for that exact
renderable/sampled format. Materials without effects retain the original
attachment requirements. Occlusion affects diffuse irradiance only; emission
adds independently of the environment. Binary16 storage and current RGBA8
composition have distinct precision/range limits, so this does not establish
HDR presentation or all reference AO modes.

`Renderer::set_environment_irradiance` selects an exact immutable asset snapshot
for subsequent submissions. Each submitted frame retains its own device-local
texture, sampler and staging owner until the completion covering its last use;
changing the selection does not mutate an in-flight frame. `None` selects the
direct-lighting route without an environment upload or sampled access.

The current input is linear RGBA8 latitude-longitude diffuse irradiance already
divided by pi. RGB contributes `E_over_pi * base_color * (1 - metallic)`;
alpha is ignored. Positive Y is the north pole, U repeats at the longitude seam,
and V clamps. This input does not implement prefiltered specular radiance.
Forward scene and Feature PBR consumers use the same selected environment.
Forward and Deferred retain concrete owners for offscreen and surface paths;
Graph declares their upload and fragment reads, and bindings validate actual
allocation descriptors, sampling compatibility and device provenance.

### Selected specular environment input

`Renderer::set_environment_specular_ibl` selects an immutable
`EnvironmentSpecularIblData` snapshot independently of diffuse irradiance.
The selected contract uses a complete linear `RGBA16Float` latitude-longitude
radiance chain down to 1x1 and a separate linear `RGBA16Float` BRDF LUT.
Inputs must contain finite binary16 values and nonnegative RGB. This is a
consumer of externally prefiltered radiance and an externally integrated LUT;
it does not generate either product or establish HDR presentation.

The LOD rule selects Babylon's `LINEARSPECULARREFLECTION` branch:
`clamp(max_lod * perceptual_roughness * scale + offset, 0, max_lod)`.
Here `max_lod` is the authored chain's final mip index, rather than the
latitude-longitude width treated as a cube face size. Scale and offset are
finite signed metadata. Radiance repeats U, clamps V, and uses linear spatial
and mip filtering. The LUT clamps both axes at LOD zero. Its UV is
`(NdotV, perceptual_roughness)` and its height-correlated channels contribute
`(F90 - F0) * lut.x + F0 * lut.y`. The selected Babylon legacy branch
uses the dielectric F0 and metallic weight for diffuse albedo. The
height-correlated multiscatter correction scales direct specular and reflected
radiance by `1 + F0 * (1 / lut.y - 1)`. A zero Y coefficient in the authored
LUT uses unit correction. Environment transmission uses one minus the maximum
component of the colored reflectance before multiscatter correction.

Specular consumers require an explicit `ShadingCamera`: a finite perspective
eye or a normalized orthographic direction toward the eye. Forward carries
actual world position; Deferred reconstructs it from the pixel center, depth,
and inverse clip-from-world transform. The shared 112-byte uniform carries
that inverse, camera, target extent, and LOD metadata. Missing cameras and
noninvertible view transforms reject preparation before dependent uploads and
recording. Legacy consumers without specular IBL do not require this camera.

Each frame owns the radiance, LUT, their distinct samplers, uniform buffer,
and aligned staging bytes. Graph declares every actual mip copy, LUT copy,
uniform copy, and fragment read; binding checks consume saved allocation
descriptors, incoming states, layouts, copy footprints, and device provenance.
The owners follow the same pending/completion lifetime as other frame inputs;
selection changes do not mutate a submitted frame. The
`pbr-environment-material` consumer verifies the selected orthographic plane
controls through Forward/Deferred on DX12/WGL, with independent payload-derived
numeric probes and actual window captures compared densely with same-input
offscreen companions. The `pbr-sphere-material` consumer extends this evidence
to perspective curved surfaces and packed ORM: one combined six-sphere mesh,
AO-only and camera changes, and independent triangle interpolation/PBR probes.
The oracle accounts separately for UNORM base/metal/normal/roughness and
binary16 effects precision. Both consumers use the same actual Graph and
completion ownership on their offscreen and Surface routes. Cube/RGBD inputs, runtime prefiltering,
additional reflection lobes, and unrestricted mixed Deferred surface families remain
outside this input contract.

### Bounded Deferred scene batches

Multiple visible records currently share the UV0-normal metallic/roughness
lighting and packing family. Other established single-mesh routes remain
available; an incompatible multi-record family is refused before allocation.
The position-deformation entry retains its single-draw contract.

One frame aggregates concrete resident streams and textures, deduplicating by
exact asset identity, generation, and device. Every draw retains its own stream
mapping, texture index, pipeline, binding groups, and aligned transform/tint
ranges in the shared frame uniform buffer. The geometry pass clears once and
records each selected draw in order. G-buffer executables are cached as the
current frame's active set; pending draws retain their own native handles.

PBR uses a linear RGBA16Float base-color/metallic attachment, preserving
nonnegative base-color values above one up to the format's representable range.
Unlit uses the RGBA8 attachment. Format admission checks the actual sampled
and render-attachment usages. Pipeline and frame reuse consume that format.
The normal/roughness attachment remains RGBA8. The lighting output is still
RGBA8, so the wider G-buffer preserves input to lighting while HDR output
requires the later output-composition work.

Effects selection is frame-wide. If any material needs effects, every draw
uses the third-output template, including neutral `AO=1` and `emissive=0` for
materials without extensions. A two-output draw cannot rely on a cleared effects
attachment: its zero alpha would incorrectly remove diffuse irradiance.
Without effects, multiple draws retain the two-color-attachment requirement.

Graph imports actual owners once and declares each draw's physical fetch,
uniform, and sampled-image accesses. A cached plan consumes this frame's full
declaration and binding set. Completed publication records map every geometry
key to its real streams and every texture key to its real owner; pending or
failed-completion work is not published. The sphere consumer's multi-record
mode verifies different model matrices and resource owners through both
pipelines and native backends, including independent readbacks and actual
window capture. The four-texture consumer extends the same path to multiple
named image bindings per material.

### Material binding locations

The bounded material schema resolves one dynamic vec4 parameter, one or more 2D
float textures, and zero or one ordinary sampler in material group zero. Names and
locations come from Material contribution metadata and its parameter mappings.
Actual reflected resource shapes, read/query access, and fragment visibility
must agree. Schema admission adds no unused-declaration restriction beyond the
existing Shader contract. This admission belongs
to the Forward/Deferred consumer; generic material preparation remains broader.

Forward, Deferred, and capture derive layouts and concrete entries from that
same schema, ordered by binding number as required by RHI. Deferred recomposes
and revalidates against its actual G-buffer shader. Transform group one and
environment group two remain Renderer contracts. A dense location permutation
(texture zero, sampler one, tint two) exercises real submissions and reuse;
zero-sampler textureLoad definitions use the same schema path.

`ForwardResources::textures` supplies `ForwardTextureBinding` entries by exact
material resource name. Preparation rejects missing, duplicate, or extra names
and resolves exact asset generations in schema order. Submission rejects a
capture source without its producer context. Capture resources share this model.
The capture entry takes an explicit producer slot independently of the receiver
resource names; multiple logical bindings may alias that one GPU product while
other bindings resolve immutable assets. Missing producer names are refused
before dependent GPU allocation. Mixed-source acceptance requires its own
end-to-end consumer and source-bound evidence.

Each draw retains a slot for every logical texture binding. Residency deduplicates
physical owners by exact asset identity, generation, and device, including two
names referencing the same image. Every view is bound and every actual sampled
footprint enters Graph; duplicate logical slots do not create duplicate uploads.
Completed publication covers every physical source. Reuse validates current
declarations and all saved/actual owner device facts before recording.

`pbr-four-texture-material` uses two transformed records, six nonconstant source
images, four authored roles, reordered locations and source lists, isolated image
channel controls, and an alias case. Its independent byte-based plane/PBR oracle
models sRGB sampling and each pipeline's storage precision. This consumer fixes
single mip, UV0, and one shared sampler; it does not establish per-slot samplers,
UV transforms, texture arrays, or a general parameter-block ABI.

### RGBA8 input transfer encoding

`ColorTextureData::encoding` is immutable asset data. `TextureEncoding::Linear`
selects `Rgba8Unorm`; `Srgb` selects `Rgba8UnormSrgb`. RGB is decoded by the
texture format before filtering, while alpha remains linear. Upload bytes are
unchanged. The allocated texture and view use the same selected format, and
admission queries that exact descriptor on the actual device. A material using
`textureLoad` still consumes the encoding contract without requiring filtering
support; an ordinary filtering sampler additionally requires filterability.

Asset identity, content generation, and device remain the residency key.
Changing bytes or encoding requires a new generation. On reuse, the current
source encoding and extent are checked against the saved actual descriptor
before creating bindings; Graph validation also checks the current owner against the planned
descriptor. A captured scene-color product retains its linear target format.
Neither the material compiler nor the Renderer adds a shader gamma conversion.

The bounded consumer uses a nonconstant sRGB image through the same ColorUv0
definition in both pipelines. Its independent oracle decodes individual source
texels before interpolation, and contrasts filtering encoded values before
decoding and decoding twice at the same acceptance tolerance. This input
contract does not infer color roles from names. The four-texture consumer checks
its explicit base/emissive color and normal/ORM data-source contract before use.
Output exposure, tone mapping, transfer encoding, and HDR presentation
remain separate work; the current raw linear UNORM output is retained.

## Private RenderGraph

Graph is the single core path for resource access, dependency construction,
planning, validation, recording, and execution. A Graph definition, a
device-specific compiled plan, and a frame-specific binding set are separate
products. Caching a compatible plan never authorizes an empty preflight, a
second scheduler, or replay of internal policy: every frame contributes its
actual declarations and revalidates its bindings.

The Graph tracks imports and exports, logical resource versions, definedness,
pass-local accesses, dead-pass culling, hazards, dependencies, state transitions,
deterministic ordering, transient
lifetime, and diagnostics. It validates concrete resource owners against the
descriptor saved at allocation, incoming state, and selected RHI device
provenance. A requested graph descriptor is not evidence about an actual owner.

Logical transient lifetime is a planning fact, distinct from physical allocation
reuse. Execution remains serial until a real RHI consumer proves a safe ordering,
queue, ownership, and completion contract for additional parallelism. Renderer
uses only portable RHI contracts; Host owns window/canvas lifecycle, while RHI
owns device and Surface realization. Assets remain outside Graph policy: they
provide logical identity and content, while Renderer decides when an asset
becomes a device-local realization.

Access validation protects the operation it precedes. Buffer, uniform, vertex,
and texture ranges must cover the physical footprint. Texture coverage respects
aspects, mip ranges, and array layers; disjoint declarations may cover distinct
requested regions but cannot fill a missing region. A plan is reusable only when
topology, hazards, layouts, and access contracts stay compatible with the
concrete frame bindings.

Recording callbacks receive only the resources declared for their pass through
the held RHI recording scope. They use RHI's command vocabulary directly;
Renderer must not create a second command/replay model. Errors retain their
owning layer: Renderer input/preparation errors, Graph declaration/binding/plan
errors and RHI execution errors are not replaced by an invented shared hierarchy.

## Residency, frame data, and completion

Renderer separates three owner classes:

* CPU asset caches retain logical source content.
* Per-device residency retains persistent realized geometry, textures, programs,
  and their descriptor/provenance facts.
* Frame data and Graph transients retain input ranges and intermediate resources
  only for the submissions that consume them.

Resources submitted to the GPU retire only after observing a completion that
covers their last use. Completion wait failure retains pending owners. Known
device loss clears non-reusable caches and refuses new work; it does not invent
retirement evidence. Conversely, a later CPU map/readback failure does not undo
a completion that was already observed for the GPU work.

Pending submissions retain concrete CPU content owners as well as their typed
GPU workload. Byte fixtures retain the exact immutable `Arc<Vec<u8>>` after
upload; their logical asset handle and generation are no longer used by execution
or retirement. Forward and Deferred retain the concrete scene snapshot and
resolved draw inputs, including their exact asset generations. This ownership
uses a closed enum rather than runtime type erasure, and follows the same
completion and failure boundaries as the GPU owners.

Surface images are leases, not cacheable texture views. Presentation, resize,
loss, and host lifecycle remain bounded by RHI and Host contracts. Renderer must
not let a surface lease escape a frame or become persistent residency.

Partial recording or execution failures retain any owner whose GPU effect is
unknown until completion or a safe shutdown establishes otherwise. Explicit
pre-device refusal may release local temporary owners. Device provenance is part
of cache isolation: resources prepared for one device cannot bind to another;
returning to an earlier device may reuse only that device's compatible owners.

## Pipeline direction

`FramePipeline` owns phase policy and Renderer-owned output selection; a
`RenderFeature` owns its local preparation algorithms and contributes through
the controlled phase interface. The current topology is intentionally limited to
opaque Forward, clear-only, and the selected Deferred product. This does not
freeze a universal render-pass taxonomy.

The current `DeferredPositionDeformer` seam already exercises a bounded GPU
product: the external `BendFeature` supplies typed parameters and WGSL, a
Renderer-owned compute pass writes a position buffer, and the following geometry
pass consumes that buffer as a vertex stream. The private Graph declares the
producer/consumer accesses and dependency; actual-owner validation and pending
submission ownership cover the buffer through completion. Both offscreen and
Surface consumers exist in `examples/rendering-integration`; this is one
controlled product, not an arbitrary external pass-chain interface.

Any additional GPU product produced by a Feature must likewise be a first-class
Graph resource with an explicit producer, consumer, version, access, and lifetime.
Renderer must not force it through a CPU draw list or readback merely because
the current Forward path is CPU-prepared. A future transparent phase needs an
explicit final ordering/composition policy across Features.

Animation sampling, blend trees, pose evaluation, and gameplay state remain
outside Renderer. A Feature may consume evaluated pose/morph input and select
CPU deformation, vertex-stage deformation, or Graph compute production. Bounds
used for culling must be conservative for the selected deformation or depend on
updated bounds; stale pre-deformation bounds are not accepted implicitly.

## Failure boundaries and invariants

* Reject stale scene/view/feature identities, mismatched snapshot versions, and
  incompatible phase contributions before backend work.
* Reject missing declarations, incompatible access contracts, actual-owner
  descriptor/state/device mismatches, and insufficient physical ranges before
  allocation, upload, recording, or submission as appropriate.
* Do not infer support from a backend name or logical shader validation. Consume
  the same selected device facts at route, creation, binding, and execution.
* Preserve submitted owners through failure and completion boundaries. Device
  loss is terminal for reuse until the relevant renderer state is rebuilt.
* Keep Graph and RHI authority private. External code cannot construct phase
  contexts or name private Graph modules.

## Source layout and verification

The public facade is `src/lib.rs`. Scene contracts live in `src/render_scene/`;
controlled extension and pipeline surfaces in `src/feature/` and `src/pipeline/`;
preparation in `src/preparation/`; private Graph in `src/graph/`; realization
and transient ownership in `src/residency/` and `src/frame_data/`; coordinated
submission in `src/renderer/`.

Unit tests reside in `src/tests/`, grouped with their production responsibility.
Public integration tests reside in crate-root `tests/`. The independent
`../examples/external-renderer` consumer verifies that external Rust code can use
the bounded public scene and extension contracts while private Graph and phase
construction remain inaccessible. Native graphical behaviour requires separate
real-window and RHI readback evidence; CPU tests and compilation do not replace
that evidence. Native evidence records the actual target, inputs, outputs,
diagnostics, and host/window facts; a successful offscreen readback does not
stand in for a visible presentation claim.

## Planned material effects and composition

The next material delivery is planned, not implemented support. The same
node-authored material must execute through Forward and Deferred on Windows
DX12 and WGL, with transparent blending, refraction/transmission and
thickness-based absorption included in the first full milestone. The
[Material design](../../material/documents/design.md#babylon-aligned-material-direction-planned)
owns authoring semantics; the [ecosystem roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md)
owns delivery order. Browser routes have separate capability and evidence gates.

Use Babylon.js revision `daae8d9483dcbb6308ebd0f6ca5b218783614771` for the
selected programs. Start with `packages/dev/core/src/ShadersWGSL/pbr.fragment.fx`
and follow the active includes and the host code that selects them.

| Behavior | Upstream source in the selected revision | Renderer adaptation |
| --- | --- | --- |
| Direct and environment lighting | `pbrDirectLightingFunctions.fx`, `pbrBRDFFunctions.fx`, `pbrIBLFunctions.fx` | Share selected WGSL functions between Forward shading and Deferred lighting. Supply the light, view and environment inputs and document any G-buffer quantization. |
| Refraction and attenuation | `pbrBlockSubSurface.fx` and `pbrSubSurfaceConfiguration.ts` | Preserve thickness and IOR branch selection and sampling order. Bind the declared environment or scene-color source and retain its owner. |
| Sorted transparency and finite OIT | `renderingGroup.ts`, `thinDepthPeelingRenderer.pure.ts`, `oitFragment.fx`, `oitBackBlend.fragment.fx`, `oitFinal.fragment.fx` | Preserve sort, peel and blend rules. Declare attachment versions and final consumption in the Graph. |
| Exposure and output | `imageProcessingFunctions.fx::applyImageProcessing` and its material/post-process callers | Keep lighting linear through the selected composition. Apply exposure, tone mapping and output transfer at the selected boundary, accounting for the actual target format. |

Compare the same inputs under the same effective branches at milestone
acceptance. Record which differences come from resource representation, such as
an environment projection or G-buffer format, and which change the computation.
Keep applicable notices and source attribution for copied or translated code.

A Deferred pipeline may combine opaque G-buffer lighting with specialized
transparent and refractive passes in the same Graph. Requiring every effect to
be evaluated from a single G-buffer is not a contract. Transparent ordering and
compositing must merge built-in and external Feature contributions globally for
the phase; independently sorted Feature lists are insufficient. Graph declares
all scene-color/depth, environment, peel and accumulation products, their
versions, consumers and physical footprints. Sampling a target being written
requires a valid separate version/resolve or a proven RHI contract; implicit
framebuffer feedback is forbidden.

### Reference behavior and proposed realization

The Forward environment-refraction route consumes Material's optional
`TransmissionExtensions` on UV0-normal metallic/roughness surfaces. It requires
the selected complete specular radiance/LUT owner and an explicit shading
camera. Its refracted input is that same linear, prefiltered latitude-longitude
environment, rather than a scene-color capture or cube map. It reuses the
existing environment bindings, Graph declarations and completion retention.
Capture receivers may prepare this required logical interface before a Renderer
is available, but submission re-prepares against the actual selected environment
and refuses a missing owner before allocation. Ordinary Deferred submissions
route environment transmission into the Forward overlay after G-buffer lighting.
Opaque and masked transmissive draws write the original depth attachment before
the sorted blend segment. The Graph declares that depth version read/write only
when an overlay pipeline writes depth. With no G-buffer draws, the selected
inputs use the normal Forward submission path. Scene-color refraction still
requires its explicit capture input.

Surface IOR determines dielectric F0; optional volume IOR controls
`refract(-view, normal, 1 / volume_ior)` and falls back to the surface IOR.
The selected Babylon linear-refraction-LOD branch uses
`alphaG = roughness * roughness + 0.0005`, adjusted toward zero by
`clamp(eta * 3 - 2, 0, 1)`. Here eta is reciprocal volume IOR when the authored
thickness interval is nonzero, and reciprocal surface IOR otherwise. This is
distinct from using IOR itself in that expression or using reflection's
perceptual-roughness LOD unchanged.

Thickness is multiplied by the largest world-matrix column length. CPU checks
use f64 and Shader uses scaled lengths to avoid a spurious f32 dot-product
overflow. Static default thickness that cannot be represented after scaling is
refused before dependent backend work. Custom callable outputs are not treated
as statically known or clamped to a guessed interval. Beer-Lambert attenuation
uses `sigma = -log(color) / max(distance, 0.00001)` and
`exp(-sigma * thickness_world)`. The same authored intensity reduces direct and
environment diffuse while weighting refraction; direct and environment specular
remain. Without the extension, the original literal dielectric F0 is retained.
All sampling finishes before alpha rejection. These are implemented shader and
admission contracts; behavior acceptance remains open.

The scene `Renderer::submit_forward_oit` and `submit_forward_oit_surface`
entries implement finite dual depth peeling for the built-in CPU Forward pipeline. `DualDepthPeelingConfig`
defaults to five rounds and explicitly excludes named draw sources. The route
admits seven intermediate targets from exact
device support, retain actual owner descriptors and provenance, and declare
initialization, min/max reduction, five dual-ended rounds, back accumulation and
final composition. MAX blending requires attachment read/write access even for
the min/max and peel targets. An OIT declaration error aborts its containing
Graph declaration; the caller must discard that Graph. Renderer-owned fragment
shading now has ordinary, initial-depth and peel entries, with RG16 depth
quantization before comparisons. Dedicated unfilterable group-three layouts
and MRT programs reuse the ordinary material, transform and environment
layouts, checking their saved program identity and actual device capabilities.
Babylon's peeling renderer temporarily disables culling, while PBR enables
two-sided final-normal handling from the independent requested lighting fact.
Accordingly, ordinary Fluxel entries retain the Material helper's
`!back_face_culling && two_sided_lighting` result, but OIT entries receive the
raw requested `two_sided_lighting` only to specialize the Renderer-owned PBR
surface-normal unit. Thus an authoring policy with culling enabled and lighting
requested remains ordinary-unlit on its culled back face, yet flips the final
normal when OIT's NativeCullNone route actually rasterizes that face. This does
not mutate Material compilation or tangent handedness.
Frame resource assembly owns the seven intermediates, a separate final output,
and both sides of every ping-pong sampled binding. Binding creation records
actual sampled owner identities; checked recording validates the selected side
and authorizes material, environment and full radiance-mip footprints through
the current Graph pass. Recording consumes one supplied Graph pass at a time.
The shared offscreen import/upload/mip prefix feeds one Graph containing opaque
color/depth, initial reduction, all peel/accumulation passes, final composition,
excluded sorted transparency and the selected terminal. The opaque pass stores depth
for later read-only tests. Only excluded blend draws require distance sorting;
OIT candidates do not acquire a sorting-eye requirement. An empty candidate set
uses ordinary sorted Forward before OIT admission; selected-route failures
remain errors. One submission retains all intermediate/output owners, pipelines
and groups in its pending Forward workload until covering completion.
Its sampled opaque target does not enter the ordinary CopySrc frame pool.
Final alpha follows Babylon's `oitFinal` source: `front.a + back.a`.
The offscreen terminal reads back the final output. The Surface terminal samples
that same output into an actual acquired image, followed by native presentation;
it declares no readback pass and does not bind the unused readback buffer.
The sampled linear intermediate format is independent of the acquired format:
the composite pipeline and Graph surface binding use the actual image format.
Unconsumed or failed-present bindings remain in the pending workload.
Completion checks cover every compiled step through final readback or all
command steps plus accepted presentation and its covering frontier. Surface
frames retire through `complete`, and offscreen frames through `readback`.
OIT plans are currently compiled per submission. The
`submit_forward_oit_capture` and `submit_forward_oit_capture_surface` routes
compose producer and receiver with independent zero/nonzero-peel selection in
one Graph: an OIT producer aliases only its final output into the receiver.
The receiver final output is either read back or composited into the actual
acquired image and presented in the same submission. The Surface terminal has
no readback pass or ticket; its Graph binding and composite pipeline consume
the actual image format, including BGRA/sRGB variants. It currently reuses the
offscreen frame allocation, so an unused readback-buffer capacity may remain,
but that buffer is neither bound, copied nor mapped. Pending work retains both
sides, optional OIT owners, composite and an unhanded/failed present binding
until completion covers every command step and the accepted present. It does
not add a CPU transfer or scheduler. Capture typed Feature remains unfinished.
Shader composition and compilation
are static evidence only; the local actual-constructor Naga matrix accepts
1360 ordinary/OIT combinations, including every independent cull/lighting
pair, but behavior acceptance remains open.

The additional `submit_forward_oit_with_feature` and
`submit_forward_oit_with_features` entries, with corresponding Surface methods,
retain independently typed feature preparations during declaration and check
issuer, scene, snapshot version and handle namespaces before GPU allocation.
All sources are checked for duplicates before stable opaque/peel/excluded
partitioning. Only the ordinary excluded suffix enters distance sorting.
Every actual opaque, initial-depth, peel and excluded pass creates a bounded
`ForwardOitReadContext`; feature indices intersect its range and map to local
draw indices while physical stream/texture slots remain unchanged. Existing
`RenderFeature::contribute` calls consume real `ForwardReadScope` authorities,
including empty intersections. The renderer requires complete read coverage
and preserves contribution failures as pipeline errors. No callback authorizes
attachments, and no global automatic reads replace feature contributions.
An empty opaque range omits environment/IBL reads, matching its recorder.
The zero-peel route uses the ordinary real composed Forward scope. All these
entries share frame assembly, upload, execution and completion ownership.

Material's static `SurfacePolicy` carries `Opaque`, `Mask`, `Blend`, or
`MaskBlend`, a finite alpha cutoff, back-face culling, and two-sided lighting.
Defaults follow Babylon: opaque, cutoff 0.4, back-face culling enabled, and
two-sided lighting disabled. The double-sided authoring convenience disables
culling and enables two-sided lighting together. Cutoff is not clamped to
the unit interval: mask rejection uses the strict authored-alpha comparison.
These facts belong to the compiled material variant; Material does not create
GPU state or choose Graph topology.

Renderer evaluates a complete base RGBA value and all texture/derivative
work before applying alpha rejection in the fragment entry. Material
evaluators remain pure. Two-sided PBR flips the final mapped world normal on
back faces. For explicit tangents, the Babylon branch constructs the local
TBN from normalized vertex normal and tangent and uses tangent W for the
bitangent. It transforms all three columns by the world matrix before
interpolation. Fragment shading applies and normalizes the decoded normal in
that interpolated basis.
The effective raster front face follows the sign of the actual draw's linear
world-transform determinant. Both executable program keys and adjacent-draw
compatibility include the resulting raster state. A blend request is refused
by an opaque-only consumer before its dependent GPU allocation until the
transparent phase supplies correct ordering and composition. This refusal
records an unfinished route and does not count as transparency support.

`ForwardCpuPipeline::with_sorted_transparency()` selects `SortedForward` and
the shared Depth32Float attachment. Ordinary opaque/masked draws run first;
transparent draws from the scene and the typed Feature then merge into one
global list. They run in the same real Graph raster pass because the attachment
owners and resource footprints are shared. This is an explicit sorted product:
an opaque contribution callback does not authorize it.

The selected Deferred `submit_deferred` path applies the same alpha policy at a
separate ownership boundary. Opaque/Mask records produce the actual G-buffer
and lighting scene color; Blend/MaskBlend records form a globally alpha-index
and eye-sorted Forward overlay. That overlay borrows the actual RGBA8 scene
color as a ColorRW attachment and the actual G-buffer depth as read-only depth;
it neither copies nor allocates replacement opaque attachments. Exact
scene-color blend support is admitted before dependent allocation. The recorder
uses the original opaque owners, rather than cloned pointer identities, when
validating bindings. An empty opaque partition deliberately becomes ordinary
Forward, not an error fallback; a missing transparent partition keeps the prior
Deferred route. The offscreen terminal reads the overlay final color. The
Surface terminal composites that final owner to the actual lease and presents in
the same Graph without a readback pass, while pending work retains both frame
owners, inputs and any unhanded/failed Present binding through completion.
Deferred finite OIT now uses this same boundary when its opaque prefix and peel
partition are both nonempty. It preserves the actual G-buffer/lighting prefix,
samples the unchanged lighting color, reads the original G-buffer depth, and
writes independent peeling output before the excluded sorted suffix and final
terminal. Empty opaque input stays on the prepared Forward path, while zero
peel candidates stay on the ordinary Deferred/Forward routes without OIT
admission. The ordinary Deferred overlay also composes environment transmission.
The explicit capture routes and their remaining combinations are described below.

The existing opaque-order hook applies to the opaque segment only. The
`transparent_alpha_index` hook supplies each transparent draw's alpha index
once; scene draws default to zero and Feature draws default to their explicit
order. Renderer then sorts by ascending alpha index, descending distance from
the world-transformed local geometry-bounds center to the actual camera eye,
with equal keys preserving the received draw order. It never batches across an intervening draw to
improve resource reuse. CPU geometry supplies the bounds in this product;
GPU-generated geometry requires its own declared bounds product rather than a
forced GPU readback into this list.

`submit_forward_with_features` and its Surface counterpart accept a concrete
pair `(FeatureBinding<F>, FeatureBinding<G>)`. Each feature independently
validates its issuer, scene and snapshot version and retains its own associated
preparation type through declaration. Duplicate feature handles are refused;
different feature namespaces may reuse local draw IDs. Scene and both feature
draw lists are merged before the single final ordering step. The pair's default
`FramePipeline::contribute_composed_pair` callback consumes three disjoint real
Graph read scopes (meshes, first feature, second feature), each exactly once.
Neither feature receives another scope or attachment-write authority. This
concrete two-feature consumer does not introduce an erased list or registry;
it uses the existing frame and completion owners for all submitted resources.

Perspective `ShadingCamera` already supplies its world eye. For parallel
lighting, `ShadingCamera::orthographic_at` supplies an explicit sorting eye
without changing the direction-only lighting uniform ABI. A parallel direction
alone cannot establish Babylon's camera-distance order and is refused for a
transparent draw. Invalid or projective world transforms and unavailable CPU
position data are also refused before dependent backend operations.

The sorted pipeline uses Babylon's actual `ALPHA_COMBINE` factors: RGB is
`src.rgb * src.a + dst.rgb * (1 - src.a)`, while alpha uses `src.a + dst.a`.
Transparent draws keep LessEqual depth comparison and disable depth writes;
opaque/masked draws retain writes. The Graph color contract includes destination
read and color write, and its depth contract covers the opaque writes and
transparent reads. Clearing at the beginning of this same pass establishes the
initial attachment contents. The actual selected color descriptor must report
blend support before dependent GPU allocations. Cache keys include fixed blend,
depth and raster facts; completed-resource publication and pending retention
continue to use the covering RHI completion.

At Babylon revision `daae8d9483dcbb6308ebd0f6ca5b218783614771`,
`Rendering/renderingGroup.ts` sorts ordinary transparency by alphaIndex then
back-to-front distance. Optional `Rendering/thinDepthPeelingRenderer.pure.ts`
performs a configurable finite number of peel iterations (default five) and
then composites. Iteration count is not a detected maximum scene-layer count.
Excluded meshes return to ordinary transparent rendering. Its not-ready branch
is different and must not be described as that fallback. These are checked
source behaviors, not measured accuracy or universal device-support claims.

`Materials/PBR/pbrSubSurfaceConfiguration.ts` selects a supplied refraction
texture or, when enabled, the environment texture. The WGSL
`ShadersWGSL/ShadersInclude/pbrBlockSubSurface.fx` evaluates thickness-dependent
refraction and absorption. Environment-texture refraction, captured scene-color
refraction and screen-space tracing are distinct routes; do not claim arbitrary
recursive ray transport from the existence of any one of them.

The proposed implementation uses existing Graph/RHI execution and typed Feature
composition for sorted transparency, optional depth peeling and refraction
inputs. A reference-matched route must expose its finite quality and source
limitations. Before broad implementation, prove format renderability and
sampleability, MRT counts, blend equations (including per-attachment needs),
depth/read-only depth behavior, resolve/copy rules, texture filtering, shader
requirements and memory costs against the selected RHI and actual DX12/WGL
devices. Public RHI changes require prior user consultation; do not silently
introduce a private native shortcut. Optional OIT capability refusal and any
explicit sorted-mode selection need separate tests; neither certifies OIT.

### Ownership and protected operations

Material instances retain logical references. Renderer residency owns realized
material and environment resources. Frame owners retain captured scene inputs,
peel/accumulation buffers and final products through the completion covering
last use; caches publish only compatible completed owners. Resize, source
replacement and device replacement must not reuse stale captures or foreign
bindings. Cache hits consume this frame's real declarations and owner facts.

Check scene/version identities, required geometry semantics, material inputs and
known device incompatibilities before dependent allocation/upload. Check saved
actual descriptors, provenance, incoming state and physical ranges before
recording. Failure after possible GPU effects retains pending owners; device
loss refuses new work. A budget or quality setting must not turn an allocation,
recording or completion failure into an apparent successful approximation.

### Selected opaque Forward depth

`FramePipeline::forward_depth_mode` selects the current frame's bounded depth
policy once. `ForwardCpuPipeline::with_depth32_less_equal` requests a single
`Depth32Float` attachment cleared to one, with depth writes and `LessEqual`
comparison. The default remains disabled and adds no depth allocation or format
requirement. Clear-only topology refuses enabled depth before GPU allocation.

Renderer owns the actual depth texture, view, descriptor, device identity and
incoming state. Its pipeline key includes depth selection. The real opaque
Graph contribution must explicitly declare `write_selected_depth`; missing
selected-depth access fails before recording. Ordinary, composed and capture
Forward routes bind the same owner in the actual raster pass. Depth is retained
with frame work until covering completion; the first implementation discards its
contents and releases it at retirement rather than publishing an unproven depth
reuse state through the color frame pool. This is not a depth-sampling, shadow,
transparency or depth-resolve interface.

### Per-draw ordinary sampler policy

`ForwardResources::sampler` optionally supplies one `ForwardSamplerBinding` by
its material-declared resource name and a backend-neutral RHI `SamplerDesc`.
CPU preparation rejects unexpected names, comparison/border policies, nonfinite
or reversed LOD ranges, zero anisotropy, and incompatible anisotropic filters.
It does not guess a device's anisotropy limit: the selected RHI admits the exact
descriptor at sampler creation. Omission retains automatic sampler selection
from that draw's requested image mips. Generic Material preparation is broader
than this consumer's current one-sampler schema.

Sampling state belongs to a draw binding, independently of immutable image
bytes, encoding and generation. Equal image references share their texture/view
owner across different policies. Each frame retains native samplers deduplicated
by all semantic descriptor fields; debug labels do not affect equality. Both
pipelines bind each draw's exact descriptor and device, including on warm image
reuse. Capture preparation preserves the explicit policy through lighting
recomposition and checks the sampler resource name in the complete schema.

Samplers have no Graph memory footprint. The resident frame retains them through
covering completion alongside their bind groups; they do not become image-cache
owners or Graph resources. Policy changes do not invalidate image residency or
silently choose a frame-wide maximum/first sampler. Scene-color composite
samplers keep their separate strict linear contract.

### Generated material mip chains

`ColorTextureData::mip_levels` is immutable texture-generation metadata. One
level preserves the base-only path. A larger valid count requests generation
from the supplied base RGBA8 payload; it does not claim that the caller supplied
the remaining levels. Linear and sRGB RGBA8 retain their exact formats throughout
the chain. For sRGB, each RHI blit decodes source RGB before linear filtering and
encodes destination RGB again; alpha stays linear. Neither averaging encoded
bytes nor relabeling the destination as linear satisfies this contract. The
selected RHI implements this format-preserving path; actual cross-backend
behavior still requires milestone acceptance evidence for the current source.

Fresh generated textures allocate the exact mip count, sampled/copy source/copy
destination usage, and a view covering the entire chain. Their Graph resources
start undefined. Upload defines mip zero; each next level is a separate Graph
pass that reads its predecessor in CopySrc and writes its own range in CopyDst.
The final material read covers every mip exposed by the view. A missing producer
therefore cannot authorize sampling an undefined tail. Checked recording resolves
both blit regions through those same pass contracts before any transition/blit.

Backend-private first-use attachment initialization belongs to RHI. The Renderer
keeps the declared Undefined-to-CopyDst boundary and does not inject fake base
uploads or attachment clears to hide a native blit initialization failure.

Actual texture support, including the reported maximum mip count, is consumed
before texture creation. The selected RHI exposes no separate blit-support query;
a backend blit refusal remains a recording failure with pending owners retained
when native effects are uncertain. Completion alone cannot publish a chain whose
recording or execution failed. Completed full-chain assets are reusable without
repeating upload/generation; cache hits still consume the current declaration and
revalidate actual owner facts.

Forward, Deferred and the Forward capture routes insert these passes into their
existing Graph execution. Completion schedules include every generated level.
The automatic per-draw sampler enables linear mip filtering and the full LOD
range when an input has mips; single-level defaults retain their previous
descriptor. Explicit named sampler policy is independent of this default.
Texture mip generation does not introduce a geometry LOD system.

The four-texture example's separate `--mip-generation` slice uses a nonconstant
4x4 base image, three GPU-generated levels, explicit LOD 0/1/1.5/2 and a source
control. Its CPU oracle independently averages and quantizes each level, then
models bilinear/trilinear sampling. It precedes the full moving-tunnel consumer,
implicit-derivative/minification coverage and broader filtering controls.

### Deferred composition boundary

The Deferred declaration builds one Graph prefix through upload, mip generation,
optional deformation, geometry and lighting. Its latest scene-color and G-buffer
versions remain in that Graph; `gbuffer[2]` is the depth version. The ordinary
consumer immediately appends its existing composite and output terminal.

The current lighting color is RGBA8 with render-attachment and sampled usage.
A sorted transparent consumer can blend into that color after admitting the
actual descriptor's blendability, using the existing depth as a read-only
attachment. OIT can instead sample the unchanged lighting color and produce an
independent final color before the output terminal. Neither path needs an extra
opaque-color copy merely to enable ordinary blending.

The ordinary transparent consumer now uses this boundary through
`DeferredTransparentPlan`: the original opaque draws still produce and light
the G-buffer, and only blend draws use the Forward material overlay. Its raster
loads the lighting color and uses the original G-buffer depth read-only. The
owner binding set installs those attachments once; recording passes their
original owner references rather than the Forward frame's cloned handles.

The two frame owners remain pending through the complete combined Graph and,
for Surface work, native presentation. Hybrid frames do not enter the ordinary
Deferred pool because the final depth state differs. Surface composition uses
the actual lease format and has no readback pass. Current assembly reuses
offscreen frame allocations, so unused target/readback capacity may remain
allocated without being bound, copied or mapped.

`DeferredOitPlan` now selects that independent final Graph version while
retaining its actual native owner and view. It imports the seven product
owners and final output once, keeps scene-color's binding tied to the lighting
owner, validates actual owner/device/descriptor facts before recording, and
retains all prefix, overlay, product, composite and presentation owners until
the covering completion. Its terminal samples the latest output version after
the optional excluded sorted suffix. Scene-color refraction separately still
requires feedback isolation; sampling and attachment writes cannot target the
same unprotected physical resource in one pass.

### Captured color as a material input

The bounded `submit_forward_capture` consumer prepares an ordinary Forward
producer and a `PreparedCaptureReceiver` without a CPU texture asset for its
capture slot. It executes producer upload/raster, receiver upload/raster and
receiver readback in one Graph submission. The producer target is imported
once; the receiver's sampled access uses that same resource version and its
binding view comes from the actual producer owner. Asset bindings retain their
own uploads; any upload into a capture slot, self-feedback, mismatched extents
and incompatible owners are refused. Logical source markers enter the draw
and reusable Graph shape, so actual owner identity cannot redefine the expected
source. Warm shared resident owners must reuse one Graph import with consistent
descriptor, device and incoming-state facts across producer and receiver.

`submit_forward_capture_with_feature` and
`submit_forward_capture_surface_with_feature` include one concrete Feature
in the ordinary Forward producer. Scene and Feature draws share the selected
ordering policy. The producer invokes `contribute_composed` on the same Graph
that exposes the capture target to the receiver.

`submit_forward_oit_capture_with_feature` and its Surface counterpart declare
one Feature's draw reads in the producer's opaque, peel and excluded draw
scopes. When it has no peel candidates, the producer invokes the typed sorted
contribution on the same capture Graph. The receiver independently selects
its ordinary or OIT product.

The scene-color transmission receiver is prepared with
`prepare_forward_scene_refraction_receiver`. It names a material texture bound
to the producer's capture slot and uses the material's ordinary sampler. The
producer and receiver share the projection and shading camera. The selected
Babylon two-dimensional thickness branch projects the displaced world point
into capture coordinates and samples the resolved scene image at level zero.
The specular environment remains the source of the surface reflectance.

The same Graph copies the producer color to a distinct receiver target before
the refractor raster loads it. The refractor samples the original producer
version. The copy preserves color outside the refractor's coverage, and only
this receiver target needs copy-destination usage. The producer stores its
actual depth for the scene-refraction route. The receiver loads that same
attachment read-only for foreground occlusion and does not allocate another
depth texture. The capture pipeline must select depth before submission.

The capture target adds sampled usage only on this route. Ordinary Forward
targets keep their existing capability requirements. Device format facts and
the allocation owner's saved descriptor, identity, incoming state and layout
govern admission and recording. The receiver does not publish its capture slot
as a persistent asset texture or upload a readback copy.

Receiver PBR preparation uses the same material request and normal-transform
validation as ordinary draws, with the actual Renderer diffuse/specular options.
Pending work retains both frames, exact prepared inputs and both asset
publication lists through completion covering the receiver. Only completed
assets become reusable; capture targets never enter the persistent asset cache.
A failed or indeterminate submission cannot release possibly submitted owners.

`submit_forward_capture_surface` extends this Forward consumer with a GPU
passthrough composite into the actual acquired Surface lease, followed by
presentation in the same Graph execution. The receiver target is sampleable on
this route, and the composite pipeline uses the actual Surface format. Pending
work retains both frames, the composite owner and any unhanded present binding
until covering completion or explicit no-effect evidence. Recording failure and
rejected presentation preserve the lease while potentially submitted work is
unresolved. Fragment position already includes the pixel-centre offset; the
one-to-one composite must not add another half texel.

Deferred capture retains its real opaque/masked G-buffer and lighting prefix.
An optional producer composition appends the same sorted transparency or finite
OIT Graph operations used by standalone Deferred submissions. Ordinary blending
updates scene color; OIT keeps lighting color read-only and writes an independent
output, followed by any excluded sorted blends. The receiver aliases that actual
final version and native owner alongside immutable assets. Opaque/masked
receivers keep their own G-buffer and lighting; Blend/MaskBlend receivers append
a Forward namespace with their own color/depth owners and optional finite OIT.
Only sampled capture bindings alias the producer, so the receiver never clears
or writes its sampled source. A scene-refraction receiver uses the Forward
refraction program for opaque, masked, and blended materials. Its Graph copy
reads the final producer color after Deferred lighting and any sorted or peeling
suffix, and initializes a separate receiver color target. The receiver samples
the producer color and loads the original G-buffer depth attachment read-only,
using the same camera and projection. This preserves foreground occlusion and
the background outside coverage. With no opaque producer prefix, a sorted
producer and this receiver use the existing Forward refraction capture path.
Scene-refraction receivers that require peeling and all-transparent peeling
producers for scene refraction still require their composition paths.
One shared resident import set covers Forward and Deferred producer,
overlay and receiver cache hits. Completed allocations reuse one Graph handle
only after comparing actual device, descriptor and incoming-state facts;
resources with pending uploads or fresh mip writes cannot reuse an old version. A producer has no output terminal; only the
receiver reads back or composites/presents to the acquired Surface. Shared imports
and checked bindings retain physical identity and reject incompatible owner facts.
Receiver-terminal completion covers all Deferred/Forward frames, optional
overlay/OIT owners, publications, and failed presentation bindings. Zero peel candidates take
the ordinary route before OIT admission. Without an opaque/masked producer
prefix, the already-prepared inputs create a real Forward producer namespace.
Its ordinary target or OIT final output feeds either a Deferred opaque/masked
receiver or the existing Forward blend receiver. This does not allocate a dummy
G-buffer or prepare the producer again. Both terminals retain one Graph and
submission. Typed capture Features remain unfinished. Shared read-only resources are
merged only after complete owner-fact comparison; OIT resolves their canonical
installed references for native recording, preserving pointer-exact guards on
attachments, depth, and product targets.
Capture blur and the wider PBR reflection and refraction combinations still
require their own Graph paths and acceptance evidence. Deferred OIT has its real Graph path but
still requires the M1 behavior matrix. The Deferred transparency seam reuses the
actual RGBA8 scene-color target and G-buffer depth: ordinary blend is ColorRW
without an extra copy, while OIT treats scene color as a read-only source and
writes a separate output. Extracting authoring seams without a real mixed
consumer is not feature completion. Native example entry points and their
backend-specific window evidence remain separate from this ownership contract.

### Acceptance cases

Renderer-owned examples are real consumers of Material authoring and Shader
composition through this same Renderer/Graph path. Scenes and assets copied
from RHI examples retain their original provenance and notices; RHI examples
remain intact. A copied low-level submission loop is not a Renderer material
acceptance case. Capability-specific Features, capture/resolve products and
multi-pass effects keep their actual resource owners and Graph access contracts.
Each scene pairs a visible, non-activating window mode with deterministic
evidence inputs and independent numerical checks of the same material semantics.
Complex scenes supplement focused probes and must make ignored effects
observable through controlled parameter changes. The staged scene inventory
and delivery gates belong to the
[ecosystem roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md#renderer-examples-as-staged-acceptance-consumers).

Use opaque/masked controls, overlapping transparent objects from multiple
Features, camera/order changes, refractive objects with different IOR/roughness,
zero/nonzero thickness and known attenuation, plus finite-peeling stress cases.
Include empty scenes, missing textures/UVs/tangents, invalid numbers, insufficient
capabilities, replaced actual owners, resize, multiple in-flight frames and
partial failure. Define finite-peel limitations against the fixed reference;
there is no new requirement to detect and reject every scene exceeding its
iteration budget. Independent numerical checks should cover alpha composition,
color conversion and attenuation separately from reference screenshots.

Retain identical material inputs across both pipelines and both native targets,
with source/lock identity, expected results, raw readbacks, diagnostics and
real-window images. Inspect representative images and independently validate
readbacks. No CPU-only result, empty graph preflight or old S0-S8 archive closes
this delivery. Shader legality alone does not prove any of these effects.
