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

- fragment and Code Node names, parameter names, resource names, source-unit
  identities, and binding locations are unique;
- every declared call resolves, has matching argument types, and the call graph
  has no cycle;
- dynamic parameters refer to declared compatible uniform resources, with one
  writer per such resource;
- equal resource declarations can merge, incompatible declarations fail;
- the selected root and domain output have the required types and inputs;
- generated declarations, parameters, resources, fragments, Code Nodes, and
  static options share one WGSL top-level namespace.

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
SurfaceOutput::MetallicRoughness { properties }
```

Both require a zero-argument `vec4<f32>` root and generate
`evaluate_material()`. `MetallicRoughness` additionally requires a named,
zero-argument `vec2<f32>` callable that returns metallic then perceptual
roughness and generates `evaluate_metallic_roughness()`. Those generated names
are reserved. Missing, wrongly typed, argument-taking, or colliding roots fail
structural validation before Shader composition.

This expresses material semantics, not a general PBR implementation. Lighting,
normal transformation, G-buffer layout, fixed-function state, and the decision
to consume surface semantics in Forward or Deferred remain Renderer policy.
The current subset does not imply full PBR, arbitrary domains, normal mapping,
IBL, shadows, transparency, or an editor graph.

## Resources, parameters, and instances

Material resources describe generic shader-visible requirements. The current
public subset has `UniformVec4` and read-only `Texture2dF32`, each with a stable
name and bind-group location. Future kinds must carry equally explicit logical
semantics and must be proven by a consumer before their public contract freezes.
The retained resource direction includes samplers, uniform data and read-only
storage alongside textures, without granting Material GPU ownership.

Parameters divide into two lifetimes:

- Static options change program/material structure and enter the canonical
  variant identity. They are not preprocessor macros; specialization belongs to
  Material/Shader composition semantics.
- Dynamic values change per-instance uniform data or logical resource selection
  and do not compile a new variant.

`MaterialInstance::new` requires the exact asset identity and content generation
when its variant is asset-bound. It rejects unknown or missing dynamic parameters
and resources, and parameter type mismatches. This prevents an old compiled
variant from being accidentally used with replacement material content.

Instances carry logical resource selections only. Renderer resolves those to
actual device-owned buffers, textures, bindings, physical ranges, incoming state
and Graph accesses. Material does not treat a declared descriptor as proof of a
concrete owner or of device compatibility.

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
