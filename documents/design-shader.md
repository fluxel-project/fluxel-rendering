# Fluxel Shader System Design

> Status: post-foundation target architecture. This document defines the
> `fluxel-shader` boundary; the selected stable RHI revision remains the
> authority for device routes, portable shader creation, and backend lowering.

`fluxel-shader` turns portable WGSL source contributions into a complete logical
program. It is usable by materials, compute, post-processing, and custom
pipelines. It does not own material semantics, graph construction, frame policy,
or native backend compilation.

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
`evaluate_material` function, and interface metadata. It cannot assemble a
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
