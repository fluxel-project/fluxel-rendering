# Fluxel Material Design

`fluxel-material` owns material authoring semantics, deterministic structural
validation, static variants, dynamic instances, and generic Shader
contributions. It is independent of Renderer policy and RenderGraph execution.

The durable dependency and ownership direction is:

```text
assets: identity + content generation
    ↓
material: authoring definition + normalized composition + instances
    ↓ generic source and interface contribution
shader: whole-program composition + logical validation/reflection
    ↓ selected device route
renderer: preparation, concrete binding ownership, Graph declaration and execution
```

Material depends on `fluxel-assets` for durable asset identity and on
`fluxel-shader` for generic source/interface vocabulary. It must not depend on
Renderer or RHI. See [Shader design](../../shader/documents/design.md) and
[Renderer design](../../renderer/documents/design.md).

## Implementation boundaries

The source tree mirrors the durable ownership model. `authoring/` owns asset
identity, definitions, typed values, callable metadata, and typed node topology.
`compiler/` owns normalization, authoring validation, canonical identity,
generated evaluators, source-name scanning, and generic Shader contributions.
`runtime/` owns compiled variants and validated dynamic instances. `lib.rs`
only assembles these modules and re-exports the stable public API.

Fields that represent authoring or compiled state remain private to their
owner. Compiler and runtime coordination uses narrow crate-private constructors
and read-only accessors; it must not expose fields merely because two sibling
modules currently need the same representation. This keeps representation
changes local while preserving the public Material contract.

## Scope and non-goals

Material describes what a surface or another material domain means. A consuming
pipeline decides how to realize it: Forward, Deferred, depth-only, shadow, or a
project-specific pass. Material does not create Graph passes, allocate GPU
objects, record commands, submit work, choose backend shader syntax, own native
objects, or own a general material editor.

The author-facing executable language is WGSL. Naga is the executable logical
IR and validation authority used by Shader. Material's normalized representation
is a typed composition/link manifest, never a second instruction language that
duplicates WGSL or Naga.

UV0 normal surfaces can declare occlusion and emissive evaluators. The selected
Forward and Deferred consumers clamp scalar occlusion to [0, 1] and apply it to
diffuse irradiance, leaving direct light unchanged; nonnegative linear emission
is additive with or without an environment. This subset does not implement the
reference's analytical-light AO control or bent-normal specular occlusion.
Deferred transports emission RGB and occlusion alpha in an optional RGBA16F
attachment, subject to binary16 quantization and range. Current final RGBA8
outputs clamp the composed result; this is not HDR output acceptance. Logical
WGSL validation does not establish finite runtime evaluator results. Authors
must supply finite, representable effect values; arbitrary Code Node arithmetic
can still exceed those resource-format limits.

The public API currently implements a compact, selected subset. The broader
authoring model in this document remains target direction until a real consumer
and validation close each addition. Current limits must not be mistaken for a
ceiling on domains, resource kinds, parameter types, or editor/import tooling.

## Conceptual model

Material separates static structure from dynamic calculation:

| Layer | Owns | Does not own |
| --- | --- | --- |
| MaterialGraph / authoring definition | nodes, typed ports, connections, parameters, resources, static choices, outputs, provenance | GPU objects, passes, native ABI |
| Material IR | normalized type-checked composition, output mapping, source provenance, variant-relevant metadata | shader instructions or backend lowering |
| WGSL fragments and Code Nodes | callable implementation, expressions, sampling, control flow | scene or pass scheduling |
| Material instance | dynamic values and logical resource selections validated against one static variant | compilation identity or GPU residency |

The conceptual compilation flow is not a per-frame call chain:

```text
authoring definition
  -> validate and normalize
  -> evaluation WGSL + generic contribution
  -> consumer adds template, entries, geometry and target facts
  -> Shader composes and validates one complete program
  -> selected RHI route realizes device-local executable objects
```

At frame time, Renderer preparation resolves the already compiled material
variant to concrete bindings and access ranges. Graph then declares and validates
the real GPU work. A pass never interprets the authoring graph.

## Authoring structure and callable graph

The target `MaterialGraph` is a typed data-flow authoring AST. It may be built
by Rust builders, procedural generators, importers, or a future editor. It owns
nodes, typed ports, connections, parameters, resources, static features, domain
and output selection, and author/source metadata. It does not require every
asset, draw, or runtime topology to become a Rust type.

One fragment or Code Node supplies a WGSL callable plus explicit typed metadata:

```text
name, typed inputs, typed output, explicit call edges,
resource declarations, exact WGSL source and diagnostic identity
```

Lowering may normalize structure, eliminate dead nodes, propagate constants and
deduplicate compatible declarations while preserving author provenance. It must
not translate WGSL into a backend language, allocate GPU resources or submit work.

The material compiler validates deterministic graph facts before asking Shader
to compose a whole program:

- fragment and Code Node names, callable input-port names, parameter names,
  resource names, source-unit identities, and binding locations are unique;
- every declared call resolves, has matching argument types, and the call graph
  has no cycle;
- dynamic parameters refer to declared compatible uniform resources, with one
  writer per such resource;
- equal resource declarations can merge, incompatible declarations fail;
- the selected root and domain output have the required types and inputs;
- generated declarations, parameters, resources, fragments, Code Nodes, and
  static options share one WGSL top-level namespace.

When a typed node topology is lowered, the generated evaluator binds each
reachable callable result once in dependency order. An edge can read the
entire result or a named vector output such as RGB and alpha from one texture
sample. The compiler checks the port against the upstream vector width and the
receiving input type. Color and legacy metallic-roughness evaluators take
no arguments. UV0 contexts supply `MaterialNodeInputSource::Uv0`
as a `vec2<f32>`. A UV0 metallic-roughness graph can connect `WorldPosition`,
`WorldNormal` and `CameraPosition` to `vec3<f32>` ports. Renderer supplies
the interpolated position, normalized geometric mesh normal and actual eye
in Fluxel world space. The geometric normal precedes normal mapping and
sidedness. A graph that asks for the eye requires a view with an explicit
position, including for a parallel lighting camera. Other surface contexts
reject these connections. A UV0 metallic-roughness definition can select a
vertex-stage UV callable with `with_vertex_uv0` or a typed node graph
with `with_vertex_uv0_graph`. The vertex evaluator transforms mesh UV0
before interpolation. The vertex graph has its own node identities, can
consume mesh UV0, and returns a `vec2<f32>`. A fragment graph connects
the interpolated result through `VertexUv0` and can still connect raw UV0
independently.
A material can additionally select node or node-output results with
`with_vertex_varyings`. Material-local varying IDs are unique, values selected
from the typed `with_vertex_uv0_graph` vertex graph that the vertex path
evaluates and interpolates for fragment-graph
`MaterialNodeInputSource::VertexVarying(id)` inputs. Their value type is
preserved in the contribution for the consumer template; callable roots and
non-graph sources are intentionally not accepted.
A vertex UV callable may also read material uniforms. To apply a texture
matrix with the Babylon.js UV convention, read a `mat4x4<f32>` parameter and
return `(texture_transform * vec4<f32>(uv, 1.0, 0.0)).xy`. Keep texture
sampling in the fragment graph so it can use derivatives. Author source and
declared callable edges may not reference generated wrappers: every
author-to-node edge must be represented by a typed port connection. This
prevents hidden cross-graph recursion before a variant is published.

The top-level namespace check scans actual emitted declarations, including
functions, types, globals, overrides and generated names. Metadata-only name
comparison is insufficient. A scan that cannot safely establish the namespace
fails closed; Shader then independently validates actual WGSL and reflected
callable/binding interfaces.

Metadata is an admission contract, not a substitute for WGSL truth. Shader must
compare callable and binding declarations against Naga reflection before device
work. Material must not infer a successful whole-program interface by trusting
author metadata alone.

## Code Nodes and source provenance

`MaterialCodeNode` is a first-class author-provided WGSL callable, not an escape
hatch outside normal validation. It enters the same call graph as Rust-defined
fragments and declares its inputs, result type, call edges, resources, exact
`SourceUnit`, and source identity.

Compiled contributions retain ordered source units as their authority: generated
material glue is first, then exact author Code Node units. Consumers use
`source_units()` for composition and diagnostics. Aggregate `source()` and
`source_unit()` are compatibility views derived from those units; they must not
become an independently mutable source representation.

Code Node source text and identity participate in static material identity.
Dynamic instance values do not. An author node may implement project-specific
BRDF, procedural, sampling, or experimental logic without expanding Material's
built-in node vocabulary. That openness does not give it Graph, submission, or
backend authority.

## Domains and surface semantics

A domain defines legal semantic inputs, legal outputs, required output shape,
and the pass contexts that may consume it. The target begins with `Surface` and
can later add domains such as post-processing without embedding a pipeline in
the material definition.

The target standard Surface semantics include base color, metallic, roughness,
normal, emissive and opacity. These describe material meaning independently of
the lighting pipeline; they are not all implemented by the current evaluator.

The current surface subset has:

```rust
SurfaceOutput::Color
SurfaceOutput::ColorUv0
SurfaceOutput::MetallicRoughness { properties }
SurfaceOutput::MetallicRoughnessUv0 { properties }
SurfaceOutput::MetallicRoughnessUv0Normal { properties, normal }
SurfaceOutput::MetallicRoughnessUv0Graph { metallic, roughness, occlusion, normal, world_normal }
```

Every surface also has a static `SurfacePolicy`. Its alpha mode is `Opaque`,
`Mask`, `Blend`, or `MaskBlend`; its default cutoff is Babylon's `0.4`.
Cutoffs must be finite but are not clamped to `[0, 1]`, matching Babylon's
direct comparison behavior. The policy separately records back-face culling
and two-sided lighting. An optional side orientation overrides the winding of
the mesh using the material. Defaults follow Babylon: opaque, back-face culling, and
no two-sided normal reversal. `SurfacePolicy::double_sided()` explicitly turns
off culling and turns on two-sided lighting together. Material emits pure WGSL
helpers for alpha rejection, output alpha, and two-sided-lighting selection;
the Renderer applies `discard` only after all material evaluations that can use
texture derivatives, and it owns culling, blending, ordering, and Graph work.

`SurfacePolicy::with_separate_culling_pass` opts into two ordered raster
draws when ordinary back-face culling is disabled. It is a static variant
fact. The consumer retains ownership of both raster states and their order.

`TransmissionExtensions` is a separate optional contract for
`MetallicRoughnessUv0Normal`. Its optional UV0 scalar callables provide
transmission intensity and normalized thickness. Static fields carry surface
and optional volume IOR, attenuation color and distance, and a non-negative
thickness interval. A missing extension emits neutral helpers: zero
transmission and thickness, IOR 1.5, and white attenuation at distance one.
With an extension, a missing intensity callable evaluates to one and a missing
thickness callable evaluates to `maximum_thickness - minimum_thickness`, as in
Babylon's no-thickness-map path. Material validates finite IORs of at least one,
strictly positive finite attenuation color/distance for `-log(color)/distance`,
and a finite non-negative ordered thickness interval. It does not allocate a
refraction source or select environment versus scene capture; those are
Renderer/Graph choices.

`RefractionEnvironment` is an optional material-local alternative to the
Renderer scene-environment fallback. It requires transmission extensions and
names an ordinary group-zero linear `Texture2dF32` resource and non-comparison
sampler. Its finite `lod_scale`, `lod_offset`, and `direction_from_world`
matrix are static variant facts. The compiler adds the implicit linear
texture-encoding requirement and rejects incompatible explicit metadata.
Material neither allocates nor prefilters this environment; Renderer retains
source residency, Graph reads, mip availability, sampling support, and fallback
selection.

When present, the thickness callable is used directly in
`minimum + (maximum - minimum) * sample`. Material does not clamp it to
`[0, 1]`: imported texture data is conventionally normalized, while a Code Node
may intentionally extend that range. Renderer may validate only static bounds
and transforms known before recording; dynamic callable output remains a
milestone-end behavior-validation boundary.

`Color` and `MetallicRoughness` require a zero-argument `vec4<f32>` root and generate
`evaluate_material()`. `MetallicRoughness` additionally requires a named,
zero-argument `vec2<f32>` callable that returns metallic then perceptual
roughness and generates `evaluate_metallic_roughness()`. Those generated names
are reserved. Missing, wrongly typed, argument-taking, or colliding roots fail
structural validation before Shader composition.

`ColorUv0` requires a `vec2<f32> -> vec4<f32>` root and generates
`evaluate_material_uv0(uv)`. Its typed graph can consume UV0 without requiring
metallic/roughness or normal evaluators. The same authored definition serves
the Forward and Deferred textured-color consumers and captured-color receivers.
Its output domain does not select lighting or render-target topology; alpha and
raster intent are carried separately by `SurfacePolicy`.

`MetallicRoughnessUv0` supplies the PBR evaluator context. A direct root and
the named property callable take one `vec2<f32>` UV0 argument. The Renderer
calls `evaluate_pbr_vertex_uv0(uv)` at the vertex entry and passes its
interpolated result to the fragment entry as the fifth argument of
`evaluate_pbr_surface_uv0(uv, world_position, world_normal, camera_position, vertex_uv0)`.
UV-only roots also expose `evaluate_material_uv0(uv)`. Existing
property callables still expose `evaluate_metallic_roughness_uv0(uv)`.
Renderer reads both values from one `evaluate_pbr_surface_uv0` call.
`MetallicRoughnessUv0Graph` connects metallic and roughness to separate
`f32` node values or vector output ports within the same fragment graph. Its
optional `occlusion` source selects another `f32` graph value; absence means
unit occlusion. The graph's occlusion is multiplied by any valid legacy
occlusion effect, so either source can remain neutral without changing the
other.
The graph's selected output supplies the `vec4<f32>` base color. The
compiler evaluates the union of their dependencies once, so multiple ports
of a texture sample share its result. Each root can consume the same UV0,
vertex-derived UV0, world-space position, geometric normal, and camera
eye inputs. An optional normal callable supplies a tangent-space normal. A
`MetallicRoughnessUv0Graph::world_normal` source instead selects one `vec3<f32>`
node or node-output value in the same graph. It is a world-space,
pre-orientation direction: Renderer applies its established orientation and
sidedness rules after evaluating it. `world_normal` and the legacy tangent-space
normal callable are mutually exclusive. Renderer owns the mesh ABI and
must provide UV0 at its documented vertex location; Material does not invent
normal, view, tangent, or filtering facts.

`MetallicRoughnessUv0Normal` keeps that UV0 base/property contract and adds a
named `normal` callable with the exact signature `vec2<f32> -> vec3<f32>`. It
generates `evaluate_normal_uv0(uv)`. The returned value is an author-defined
tangent-space direction: Material intentionally does not prescribe a texture
binding name, sampling operation, `[0, 1]` decoding, green-channel convention,
strength, or default normal. A consumer must require and validate a compatible
mesh tangent basis, build the TBN frame, and apply its actual normal transform.
The neutral returned direction is `(0, 0, 1)`. An author sampling an encoded
`(0.5, 0.5, 1)` texel must perform its intended decoding in that callable;
neither the compiler nor the Renderer applies an implicit texture decode.
Derivative fallback, normal blending, parallax, and layered normal policy are
not part of this first contract.

This expresses material semantics, not a general PBR implementation. Lighting,
normal transformation, G-buffer layout, fixed-function state, and the decision
to consume surface semantics in Forward or Deferred remain Renderer policy.
The current subset does not imply full PBR, arbitrary domains, derivative
normal fallback, normal blending, general environment generation, shadows, transparency, or an editor
graph.

The Renderer `pbr-sphere-material` consumer maps a single linear RGBA8 image
to packed ORM: R is occlusion, G is perceptual roughness, and B is metallic.
Its authored properties and occlusion evaluators sample the same physical
resource; this does not add a special ORM surface type or a general multi-texture
binding API. Renderer applies the selected AO semantic to diffuse irradiance,
including its specular-energy conservation factor, independently of direct
and specular lighting.

The Renderer four-texture consumer uses the same surface contracts with
independent base-color, normal, ORM, and emissive image resources. Material
declares their logical names and locations; Renderer resolves named sources,
physical residency, actual transfer formats, and Graph access. Two logical
names may reference one physical image without changing the authored shader
interface. Image color roles remain explicit consumer facts, rather than
special resource kinds inferred from names.

`PbrSurfaceFactors` is an opt-in UV0 metallic/roughness layer over the
existing authored evaluator ABI. It never changes the meaning of legacy root,
metallic/roughness, normal, or optional occlusion callables. Packed occlusion
requires the UV0 normal-surface interface. The compiler emits
the same `evaluate_material_uv0`, `evaluate_metallic_roughness_uv0`, and
`evaluate_occlusion_uv0` entry points, with authored values multiplied by the
selected standard factors. An optional base texture is hardware-sRGB and is
multiplied with an authored `vec4` base and optional `Vec4` base factor. An
optional packed texture is linear and multiplies authored metallic and
perceptual roughness through explicit channels. `gltf_orm` selects occlusion
red, roughness green, metallic blue; `babylon_metallic_roughness` selects
metallic red and roughness alpha. Other explicit channel assignments remain
available because Babylon exposes those controls independently.

When a packed occlusion channel is selected, the generated scalar relation is
`mix(authored_occlusion_or_one, packed_channel, occlusion_strength_or_one)`.
This follows Babylon's ambient-strength blend while retaining an existing
custom occlusion callable as the previous value. Factors require correctly
typed declared parameters and texture/sampler resources. Their fixed sRGB or
linear role is added to material texture-encoding requirements and to the
variant identity; conflicting explicit metadata is rejected before publication.
No renderer pass, coordinate transform, binding owner, or texture allocation is
introduced by this configuration.

## Resources, parameters, and instances

RGB transfer encoding is an image-source fact resolved by Renderer, rather than
a different Material resource kind. A `Texture2dF32` can sample an sRGB-format
color source or a linear-format data source. Authored functions consume the
resulting linear values and must not decode an already decoded color sample.
Normal-map unpacking remains an independent authored operation. Renderer owns
actual format support, source-generation changes, view creation, and binding
validation; the material compiler does not infer color roles from resource names.

An author may attach an optional `MaterialTextureEncodingRequirement` to a
declared texture resource. It becomes part of the static variant identity and
requires an asset binding with the selected linear, hardware-sRGB, or
sRGB-encoded-UNORM encoding. A
same-frame capture is linear. Unannotated resources retain the generic contract
for callers that intentionally support either source representation.

`EmissiveConfiguration` is the standard UV0-normal-PBR emissive path. It emits
`color * sampled_linear_rgb * texture_level * intensity`, matching Babylon's
PBR emissive composition. Its color, intensity, and texture-level references
are optional, typed material parameters (`Vec3`, `F32`, and `F32`); absent
references lower to black color and unit factors. The optional texture names a
declared `Texture2dF32` resource and non-comparison sampler, plus an explicit
linear, hardware-sRGB, or sRGB-encoded-UNORM encoding. That encoding is added
to the variant's required texture metadata. Hardware `Srgb` is decoded before
filtering by its sampled texture format. `SrgbEncodedUnorm` instead samples an
ordinary UNORM view and performs Babylon's default post-filter `pow(rgb, 2.2)`
decode in the evaluator. `EmissiveSrgbDecode::Exact` selects the IEC sRGB
piecewise inverse instead. This distinction is intentional because filtering
before versus after transfer decoding is observably different. The author can
supply a custom `SurfaceExtensions::emissive` callable as an exclusive
alternative.
Renderer selects its effects route through
`MaterialDefinition::requires_surface_effects()`, which includes either
emissive form, legacy occlusion, packed occlusion, and graph-selected
occlusion.

Material resources describe generic shader-visible requirements. The current
public subset has f32, i32, u32, float vector and column-major
`mat4x4<f32>` uniforms, read-only `Texture2dF32`, `Texture2dArrayF32`,
`TextureCubeF32`, `Texture3dF32`, `Texture2dDepth`, `Texture2dArrayDepth`, and
`StorageTexture2dRgba8UnormRead`,
and `Sampler { comparison }`, each with a stable name and bind-group location.
A `Texture2dArrayF32` resource uses a discrete layer argument in WGSL.
A `TextureCubeF32` resource uses a direction coordinate; a `Texture3dF32`
resource uses a three-dimensional coordinate. `Texture2dDepth` reads raw depth
and has no RGB transfer-encoding metadata.
`Texture2dArrayDepth` likewise reads raw depth through an `i32` layer coordinate
and has no RGB transfer-encoding metadata.
`StorageTexture2dRgba8UnormRead` is a raw `textureLoad`-only
`rgba8unorm` storage image with exact read access; it likewise has no RGB
transfer-encoding metadata. A material can specify the RGB transfer encoding
for any of the float texture shapes. `comparison` is a WGSL fact; filtering is
not, so Renderer validates it from the actual sampler and texture descriptors
before recording. Future kinds must carry equally explicit logical semantics
and must be proven by a consumer before their public contract freezes. The
retained resource direction includes uniform data and read-only storage
alongside textures, without granting Material GPU ownership.

Parameters divide into two lifetimes:

- Static options change program/material structure and enter the canonical
  variant identity. They are not preprocessor macros; specialization belongs to
  Material/Shader composition semantics.
- Dynamic values change per-instance uniform data or logical resource selection
  and do not compile a new variant. Signed integers and matrices correspond to
  Babylon Node Material's `InputBlock` uploads through `setInt` and `setMatrix`.

`MaterialInstance::new` requires the exact asset identity and content generation
when its variant is asset-bound. It rejects unknown or missing dynamic parameters
and external resources, and parameter type mismatches. The value of a
parameter-owned uniform comes from the parameters map. That uniform's resource
alias is optional. This prevents an old compiled
variant from being accidentally used with replacement material content.

Instances carry logical resource selections only. Renderer resolves those to
actual device-owned buffers, textures, bindings, physical ranges, incoming state
and Graph accesses. Material does not treat a declared descriptor as proof of a
concrete owner or of device compatibility.

Generated PBR samples can select UV0 or UV1 independently by logical texture
resource through `MaterialDefinition::with_texture_uv`. This covers the
generated base color, packed metallic/roughness/occlusion and emissive samples.
The transform stores three finite columns for the projected two-dimensional
affine map. The compiler evaluates the same projected coordinate as Babylon's
texture matrix applied to `(u, v, 1, 0)`. A texture without metadata samples
UV0 directly. Authored WGSL nodes retain their declared inputs and sampling
expressions. The coordinate set and coefficient bits participate in static
material identity.

## Static identity and caching

The canonical `MaterialVariantKey` includes static options, normalized material
structure, output-domain facts, required property-root identity, sources and
source identities, binding declarations, and (when applicable) the exact asset
identity plus content generation. The full canonical key remains cache identity.

The generated glue `SourceUnit` may use a compact digest-derived identifier so
diagnostics remain practical, but a digest is not a cache key and cannot create
aliasing: composition and cache comparison retain full canonical source/content
facts. A replacement asset generation necessarily isolates the resulting
variant. Renderer device caches add device provenance and completion ownership;
Material never assumes logical identity alone permits GPU reuse.

Compilation can be cached at Material or Shader's appropriate logical layer, but
source-level artifacts and device pipelines have distinct owners and invalidation
inputs. A Material cache never makes a Graph declaration optional: every frame's
consumer still declares its actual use and validates current concrete bindings.

## Contribution contract with Shader

`CompiledMaterialVariant` exposes a `MaterialShaderContribution` containing:

- ordered source units and generated glue;
- generic binding declarations and dynamic parameter-to-uniform associations;
- callable declarations to compare with actual WGSL reflection;
- typed material-local vertex varying identifiers for vertex/fragment templates;
- canonical static choices needed by complete program composition.

The consumer supplies the pass template, selected entry points, vertex/geometry
interface, render targets, and any pass-level static requirements to construct a
complete `ShaderCompositionRequest`. Material never creates a complete pipeline
program by itself, and Shader never imports Material IR.

Logical WGSL acceptance does not prove selected-device support. The device-side
consumer must use the same complete descriptor and RHI capability facts before
creation, binding, or execution. A material contribution with undeclared or
incompatible concrete resources must be refused before the backend operation it
would otherwise protect.

## Failure boundaries

Material errors are structured and should identify the violated authoring or
instance contract. Validation must reject before dependent Shader, RHI, or GPU
work when it can do so. Important failure classes include duplicate or
conflicting declarations, invalid or cyclic calls, missing roots, unsupported
output shapes, Code Node identity/source problems, namespace collisions,
binding conflicts, asset-generation mismatch, and unknown/missing/type-mismatched
dynamic values.

If a later whole-program check fails, diagnostics retain the generated or author
`SourceUnit` identity so the failure can be attributed without reinterpreting the
material graph. Failure must not publish a partial variant, mutate a prior
compiled variant, or authorize a fallback binding/pipeline.

## Extension direction

Built-in nodes may cover common values, texture sampling, arithmetic, vector
operations and portable semantic helpers. They are not the semantic ceiling;
Code Nodes preserve open-ended authoring. Future MaterialGraph breadth, typed
ports/connections, domain-specific semantic sets, importer/editor metadata,
additional resource classes, and more surface outputs must be added only with a
real consumer and a narrow testable contract.

Material's extension model remains ordinary Rust data, generic traits, and
monomorphized composition where type-level extensibility is needed. It must not
introduce trait-object plugin registries, runtime type erasure, a second render
scheduler, or public access to Renderer Graph/RHI internals.

## Verification expectations

Changes to this crate require tests that exercise the actual compiler and
contribution path, not only data constructors. Cover valid composition and its
corresponding refusal: callable signatures, generated namespace collisions,
resource/binding ownership contracts, static invalidation, dynamic sharing,
asset-generation replacement, and source provenance. A real consumer must also
verify that the exact material contribution reaches Shader and that concrete
bindings/accesses are validated before recording.

Tests establish the contracts they execute. GPU presentation or a new backend
claim requires an independently retained real-target attempt; successful logical
composition or Naga validation alone is not such evidence.

## Babylon-aligned material direction (planned)

The target is Babylon-style material functionality with an authoring boundary
suitable for a future Blender adapter. This section is a target contract, not a
claim that the current compact API implements it. Delivery order and milestone
acceptance belong to the [ecosystem roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md).
No material editor, Blender plugin, serialized file format, or file parser is
part of this work. A future framework adapter maps external authoring semantics
into ordinary Rust material data; glTF fields do not define the internal model.

Functional equivalence means explicit parameter meanings, supported feature
combinations and reproducible rendering behavior. It does not require Babylon
API compatibility or pixel-identical output. Compare scenes under recorded
geometry, lighting, environment, exposure, transfer functions and reference
versions; define tolerances before examining results. Blender compatibility
classifies mappings as exact, approximate, bake-required or unsupported, with
named reasons. Do not promise arbitrary Blender graphs or a specific Principled
version before its semantics have been checked.

### Porting the selected material behavior

For each selected effect, trace the authored value from the Babylon host
configuration through the effective defines and shader inputs to the WGSL
consumer at the revision below. Material carries channel meanings, units,
defaults, texture interpretation and static choices. Renderer combines those
choices with scene and device facts to select an executable program. Shader
composes the source units and reflects their interfaces.

Port the selected functions with their constants, input domains, branch
conditions and evaluation order. Keep the upstream file, symbol, revision and
applicable notices with adapted code. Record resource or coordinate adaptations
and the input that exposes their effect. An algorithmic difference needs a
specific reason and a comparison case. Forward and Deferred consume the same
material definition and share the selected shading functions where their
inputs are equivalent.

### Authoring and consumer contract

The first real consumer must construct a representative PBR material through
typed node/function connections, including a Code Node, and execute the same
material definition in both Forward and Deferred pipelines. Node topology,
material values and resource selections remain runtime data. Extension
implementation types remain statically dispatched. Reuse the existing compiler,
WGSL callable composition, source provenance and Naga validation; a second
instruction IR or runtime node interpreter is not required.

A candidate design adds named typed surface channels and material evaluation
context for UVs, normal/tangent basis and view-dependent inputs. Exact Rust
signatures, packing and optional-channel defaults remain provisional until both
consumers work. Start with base color, metallic/perceptual roughness, normal,
emission, occlusion, opacity/cutoff, IOR, transmission and thickness/attenuation
requirements exercised by those consumers. Specify attenuation color, its reference distance, units, texture interpretation
and no-absorption defaults before freezing the contract. Distinguish spaces, linear
versus encoded colors, and geometric versus shading normals. An unused channel
may be omitted only according to an explicit default; a consumer must not silently
drop a required effect. Keep current Color and MetallicRoughness entry points
working through the transition or document and test any deliberate migration.

Texture semantics include UV selection/transform, sampler behavior, channel
selection, color-space interpretation and normal conventions. Material owns the
logical meaning and instance values; Shader owns declarations/reflection;
Renderer resolves actual image/sampler owners and matching binding requirements.
Dynamic values and resource replacement must not create structural variants;
source, static choices, required channels and binding layout must invalidate
incompatible artifacts. Variant growth and cache lifetime require bounded,
measured policies rather than an eager Cartesian product of feature switches.

### Effects and failure boundaries

Transparent blending, refraction/transmission and thickness-based volume
absorption are required in the first full PBR delivery. The Deferred consumer
may combine opaque G-buffer lighting with dedicated transparent/refraction
passes. Material requests these effects without selecting pass topology or
owning scene-color textures. See [Renderer execution design](../../renderer/documents/design.md#planned-material-effects-and-composition).

Babylon's PBR thickness/attenuation model is the reference for volume absorption;
it does not imply a general heterogeneous volume or arbitrary nested-media
solver. OIT is a rendering policy with finite quality/cost, not a material node
or a guarantee of unlimited transparent layers. Geometry, lights, environment
capture/filtering and final compositing remain Renderer or Feature concerns.

Reject malformed connections, cycles, incompatible port types, missing required
inputs, non-finite/invalid physical parameters and conflicting declarations
before publishing a compiled variant. Whole-program failures preserve author
source identity and do not mutate a usable prior variant. Device-specific
unsupported resources or sampling requirements are rejected by the consumer
before the operation they protect. Reference-specific defaults and rendering
approximations must not disguise allocation failure, device loss or unsafe
resource lifetime as a successful degraded frame.

### Reproducible reference

The inspected Babylon checkout is `daae8d9483dcbb6308ebd0f6ca5b218783614771`.
Its source is Apache-2.0 licensed; direct code adaptation requires preserving
applicable notices and reviewing translated algorithms. It is a reference, not
a new runtime dependency. Primary entry points under `packages/dev/core/src/`
are `Materials/PBR/pbrBaseMaterial.pure.ts`,
`Materials/PBR/pbrSubSurfaceConfiguration.ts`, `Materials/Node/nodeMaterial.pure.ts`,
`Materials/shaderMaterial.pure.ts` and `Materials/standardMaterial.pure.ts`.
Surface contracts and static extension constraints take precedence over copying
Babylon's class or plugin organization. Broader node catalogs, advanced PBR,
Standard/SpecGloss and specialized effects require separate executable cases;
this reference inventory is not proof of complete parity.
