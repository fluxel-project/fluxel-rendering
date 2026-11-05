# Fluxel Shader System Design

> Status: target architecture with an accepted bounded DX12/WGL consumer subset.
> This document defines the `fluxel-shader` boundary; the selected stable RHI
> revision remains the authority for device routes, portable shader creation,
> and backend lowering. The accepted subset does not stabilize every future
> consumer or target.

`fluxel-shader` turns portable WGSL source contributions into a complete logical
program. It is usable by materials, compute, post-processing, and custom
pipelines. It does not own material semantics, graph construction, frame policy,
or native backend compilation. Consumer extension hooks follow
[ADR-0026](adr/0026-static-extension-composition.md): concrete generic composition,
without trait-object registries. Source units and interface descriptors remain
runtime data; they are not a Rust type per shader or vertex layout.

## Composition boundary

```text
Material lowering                FramePipeline / another consumer
  source units + interface           pass template, entries, geometry,
  contribution                        static choices, requirements
          \                              /
           +---- ShaderCompositionRequest
                          |
                          v
             fluxel-shader composition and validation
                          |
                          v
          composed WGSL + logical reflection + requirements
                          |
                          v
                   selected fluxel-rhi Device
                          |
                          v
       device route and private backend compilation/lowering
```

`ShaderCompositionRequest` is a Shader-owned, generic input. It contains source
units, callable and resource-interface declarations, a pass template, requested
entry points and stages, geometry interfaces, static specialization values, and
portable requirements. It contains neither `MaterialIr` nor material-domain
meaning.

Material lowering produces a generic contribution: its source units, generated
evaluation functions, and interface metadata. Every surface contribution has
`evaluate_material`; the bounded metallic-roughness surface output also has
`evaluate_metallic_roughness`. It cannot assemble a
complete request because it does not know whether a Forward, Deferred, shadow,
compute, or project pass will consume the function. The active `FramePipeline`
or another Shader consumer combines that contribution with pass-specific inputs
and submits the complete request. A Shader-only consumer can submit a complete
request without Material or Renderer.

## Ownership

Shader owns:

- deterministic source-unit composition, symbol isolation, and source maps;
- logical WGSL and Naga validation, reflection, and validation of supplied
  interface metadata against the composed program;
- logical artifacts, interfaces, portable pipeline requirements, variants, and
  caches for those products; and
- diagnostics remapped to source units and their contributors.

RHI owns the facts of an opened device and creation of executable objects. The
current RHI contract selects `ShaderRoute::Direct`, `ShaderRoute::Naga`, or
`Unsupported` for a `ShaderDesc`; its `ShaderTarget` includes the backend format,
environment, and `abi_key`. The selected backend performs any Naga or native
lowering and compilation. Shader passes the selected portable source and entry
metadata to that contract. It neither selects a backend dialect nor depends on
backend-private ABI types.

The initial `ShaderArtifact` is therefore a validated composed WGSL program plus
logical reflection, source provenance, and portable requirements. It is not a
precompiled DXIL, SPIR-V, MSL, or backend ABI. Renderer preparation adapts it to
the current RHI portable shader and pipeline creation API. Future retained target
artifacts require an explicit consumer, exact target-route contract, and proof;
they do not change this default boundary.

Names such as `ShaderArtifact`, `ShaderInterface`, and
`PipelineRequirements` in this document describe Shader-owned target types. They
do not freeze or imply historical RHI types with the same names.

## Identity and caches

Logical composition caching keys the complete request, including content hashes
of source dependencies, generated source, callable/resource interfaces, pass
template, entry-point and geometry interfaces, static choices, and Shader
compiler options. Dynamic uniform values and concrete texture or buffer instances
do not enter this key.

Executable RHI shader and pipeline objects are device-local. Their cache identity
also includes the selected device identity, capability facts that affect the
request, the selected `ShaderRoute` and `ShaderTarget::abi_key`, and any RHI
compiler/lowerer version exposed by the selected integration. Cache entries never
cross a lost or replaced device. Backend-native reflection may validate its own
ABI but is not a second public material or Shader semantic interface.

## Binding and frame boundary

Geometry semantics and shader locations are distinct. Renderer preparation
matches the selected logical input interface against geometry stream descriptors
and creates the corresponding vertex layout/pipeline. Shader does not impose a
fixed Position/Normal/UV vertex struct. Interleaved, separate and mixed streams
can share a logical program when their resolved input interface is compatible;
their physical packing still participates in executable pipeline identity.
Missing inputs require explicit refusal or an explicitly supported variant.
Storage-based/procedural geometry and deformation follow their selected program
interfaces and resource requirements, not an assumed conventional Mesh shape.

Logical reflection describes binding slots, stages, and access classes. Renderer
preparation resolves each slot to a concrete RHI resource, range, access, and
device provenance. The resulting binding-use manifest is declared to
RenderGraph, which validates it before recording and includes buffer and texture
uses in dependency and lifetime analysis. A sampler remains a validated binding
input, not a memory hazard.

A changed material resource instance does not by itself require a new compiled
graph. The cached plan may be instantiated after manifest revalidation when its
access contract and dependency edges remain compatible. A changed access mode,
range contract, binding layout, or dependency edge requires a new graph
definition and compilation.

Persistent textures, meshes, and streaming tiles use renderer residency. Per-frame
transforms, bone palettes, and similar transient inputs use frame-data upload;
both may share a low-level upload mechanism and RHI completion tracking, but
frame data is not a long-lived residency asset. GPU skinning, particle
simulation, culling, and indirect work are RenderGraph passes.

## Delivery gates

## Current implementation status

The current S2 implementation provides a logical WGSL-only boundary in
`fluxel-shader`. A caller supplies named source units, selected entry points,
optional binding and vertex-input declarations, callable declarations, static
options, and portable requirements. The composer concatenates the source units,
retains source spans, parses and validates the resulting WGSL with Naga, and
returns a portable `ShaderArtifact` containing the composed source and logical
reflection.

Logical validation records a sufficient set of Naga capabilities for the full
module. Primitive index, early depth test, and ray query requirements are also
merged with caller-supplied portable RHI feature names. The consumer must check
all recorded capability requirements against the selected device and its
actual pipeline before creating backend objects; requirements it cannot
express or check must be reported explicitly. This is not a claim that any
particular device or backend lacks those capabilities. Capability validation
currently covers the whole module, so an unselected entry can conservatively
cause an additional requirement.

Reflection is limited to the requested entry points. Binding visibility and
read, write, query, and atomic use are derived from Naga's per-entry-point
analysis rather than from caller metadata. Declared bindings and reflected
reachable bindings are checked in both directions. Vertex locations are
reflected only from selected vertex entry-point arguments (including struct
members); built-ins and fragment inputs are not vertex attributes. The current
request accepts at most one vertex entry because its vertex-input declaration
is shared; multiple compute entry points remain supported. The current public
vertex type vocabulary is scalar or two-, three-, and four-component
F16, F32, I32, or U32 values.

The logical cache is keyed by the complete structural composition request, so
source, declarations, selected entries, static options, and requirements remain
distinct. This is a logical artifact cache only; it does not cache RHI shader or
pipeline objects and does not establish device-local reuse.

Logical validation is not sufficient execution evidence by itself. The named
S2/S3 and later DX12/WGL consumer records establish their respective selected
routes, but each new Shader consumer still must check requirements against the
selected device and retain its own Graph/RHI evidence. No unsupported backend or
future Shader feature is inferred from those bounded results.

After the S0/S1 foundation closure, S2 creates `fluxel-shader` and proves an
independent complete WGSL raster program and, where supported, a compute program
through composition, reflection, selected RHI creation, and real Graph/RHI
execution. S3 creates the smallest `fluxel-material` consumer: Rust-defined
function fragments, parameter/resource schemas, static/dynamic choices, and a
multi-fragment `evaluate_material` contribution. It does not require a complete
editor graph or Code Node authoring experience.

S4 scene work may proceed alongside S3 after S2 establishes the Shader
contract. S5 proves the material-backed Forward path. S6 closes the renderer MVP
only when an external Rust Feature, an external FramePipeline, and a custom
material contribution run together through the normal Graph/RHI path. Code Nodes
and broader authoring experience are selected S7 consumers, not an MVP gate.
The first C1 consumer is now implemented: Material supplies ordered immutable
source units, including each `MaterialCodeNode` author's exact `SourceUnit`, to
the same `ShaderCompositionRequest`; the composer preserves source spans and
checks the node's declared callable against actual WGSL. The legacy aggregate
source is derived only for compatibility, so it cannot replace author-attributed
units.

The current C2 extension adds a bounded `MetallicRoughness` surface domain to
that same contribution path. Material structurally validates its zero-argument
`Vec4` base root and zero-argument `Vec2` metallic/perceptual-roughness root,
then Shader composes pass templates that consume both generated evaluators.
This validates source attribution, callable signatures, and complete logical
WGSL before GPU work. It is not GPU PBR evidence, a complete MaterialGraph, or
general material-domain support; the selected private renderer consumer and its
separate bounded DX12/WGL GPU validation are described in
[the Renderer design](design-renderer.md) and [version-plan.md](version-plan.md).
