# Fluxel Material System Design

## 1. Purpose

This document defines the architecture and implementation plan for
`fluxel-material`. It is paired with the independent `fluxel-shader` crate:
material owns authoring semantics and lowering to Material IR, while Shader owns
complete-program assembly, logical validation/reflection, variants, artifacts,
and cache identity. Material supplies functions, parameter/uniform and resource
schemas, call topology, and the generated `evaluate_material` source needed by a
material. It emits a generic Shader contribution, rather than a complete program:
the active pipeline supplies the pass template and other pass-specific inputs
before Shader composes the program. Neither crate owns renderer policy or
RenderGraph construction. The precise handoff is defined in
[the Shader design](design-shader.md).

The material system is designed around four explicit boundaries:

```text
MaterialGraph      = static material structure
Material Fragment  = dynamic shader process
Material IR        = normalized composition/link metadata
Naga IR            = executable shader program IR
```

Fluxel does not implement its own material editor or shader compiler frontend.

The intended Rust toolchain is:

```text
Rust builder / procedural generator / tests
  -> Fluxel MaterialGraph
  -> Material IR
  -> Material-generated evaluation WGSL + generic interface contribution
  -> FramePipeline assembles ShaderCompositionRequest
  -> Shader composition
  -> WGSL ShaderModule
  -> Naga
  -> Shader System
       -> ShaderArtifact / ShaderInterface / Pipeline Requirements
       -> RHI shader/pipeline creation
```

WGSL is the Fluxel-facing shader source language for custom material code, shader fragments, and custom pipeline shader modules.

Naga owns the executable WGSL IR and validation used by Shader. The selected RHI
backend owns its device-specific route, target lowering, and native compilation.

Fluxel's Shader system owns logical `ShaderArtifact`, `ShaderInterface`, pipeline
requirements, and their cache identities. RHI consumes the selected portable
source and entry metadata to create device-local executable shader and pipeline
objects using its advertised route. Fluxel owns material semantics, composition,
resource contracts, variant semantics, and integration with `FramePipeline`,
`RenderGraph`, and RHI.

---

## 2. Architectural Position

Material compilation and frame execution are related but distinct flows.
The compilation diagram is a conceptual data flow, not a per-frame call chain:
the pipeline template provider may assemble and cache a complete request before a
frame, and a `FramePipeline` normally reuses the resulting artifact.

### Material compilation

```text
Rust builder / procedural generator / tests
        ↓
Fluxel MaterialGraph
        ↓
Material IR
        ↓
Material-generated evaluation WGSL + generic interface contribution
        ↓
FramePipeline or another consumer assembles ShaderCompositionRequest
        ↓
Shader Composer
        ↓
WGSL ShaderModule
        ↓
Naga
        ↓
Shader System
        ├─ ShaderArtifact
        ├─ ShaderInterface
        └─ Pipeline Requirements
                ↓
               RHI shader/pipeline creation
```

### Frame execution

```text
RenderScene
        ↓
FramePipeline
        ↓
RenderGraph
        ↓
Shader / Pipeline + Material Runtime
        ↓
RHI
```

The authoring graph is never interpreted during frame execution.

At runtime, `FramePipeline` selects prepared material/shader variants, declares graph work, and pass execution binds material runtime data through RHI portable contracts.

---

## 3. Core Mental Model

The material system is split into:

```text
static structure
+
dynamic process
```

### Static structure

Static structure answers:

```text
What nodes/fragments exist?
What are their typed inputs and outputs?
How are they connected?
Which parameters exist?
Which GPU resources are required?
Which static features are enabled?
Which final material semantics are produced?
```

This belongs to:

```text
MaterialGraph
Material IR
```

### Dynamic process

Dynamic process answers:

```text
Given these inputs, how is the value computed?
Which expressions run?
Which functions are called?
Which textures are sampled?
Which branches or loops execute?
What value is returned?
```

This belongs to:

```text
WGSL Material Fragments
Naga IR
```

Fluxel must not duplicate Naga by inventing another executable shader instruction language inside Material IR.

---

## 4. MaterialGraph

`MaterialGraph` is the editable authoring representation of a material.

It represents the static topology of the material.

Conceptually:

```text
UV0
  ↓
TextureSample ────┐
                  Multiply ───→ Surface.BaseColor
TintColor ────────┘
```

A `MaterialGraph` owns:

```text
nodes
typed ports
connections
parameters
resource references
static feature values
material domain
material outputs
source provenance
editor/tool metadata
```

It does not own:

```text
native GPU objects
RenderGraph passes
command recording
pipeline submission
backend-specific shader code
Naga IR
```

### 4.1 MaterialGraph as authoring AST

A useful analogy is:

```text
MaterialGraph ≈ typed authoring AST / data-flow graph
```

Rust builders, tests, procedural material generation, and future importers may
all construct the same Fluxel `MaterialGraph`.

---

## 5. Material Fragment

A `Material Fragment` is a reusable dynamic shader implementation with a typed interface.

Conceptually:

```wgsl
fn procedural_color(
    uv: vec2f,
    time: f32,
) -> vec3f {
    let x = sin(uv.x * 20.0 + time);
    return vec3f(x, 1.0 - x, 0.5);
}
```

The fragment defines:

```text
function implementation
typed inputs
typed outputs
required resources
optional helper functions
optional static requirements
semantic role
source provenance
```

A fragment may implement:

```text
multiply
normal mapping
texture sampling helper
BRDF term
procedural noise
custom user code
project-specific shading logic
```

### 5.1 Fragment metadata

Fluxel stores metadata describing the authoring exposure and semantic names by
which a fragment participates in material composition. It is not a second
executable interface definition.

Conceptually:

```rust
pub struct ShaderFragment {
    pub module: ShaderModuleId,
    pub function: String,

    pub inputs: Vec<FragmentInput>,
    pub outputs: Vec<FragmentOutput>,

    pub resources: Vec<FragmentResource>,
    pub semantic: FragmentSemantic,
}
```

The executable function body and its exact callable/resource interface remain
WGSL. WGSL plus Naga reflection are the executable interface truth.

During composition, the shader system must compare the metadata with Naga's
reflected WGSL interface exactly, then produce one `NormalizedFragmentInterface`
for Material IR and subsequent composition. Names, parameter directions, types,
returns, binding classes, and resource requirements must agree. A mismatch (for
example metadata `vec3f` versus WGSL `vec4f`) is a compilation error; Fluxel must
not continue by trusting either declaration opportunistically.

Fluxel does not translate the function body into a parallel `MaterialInstruction` language.

---

## 6. Code Nodes Are First-Class

Custom code is a core material capability, not an optional escape hatch.

A material system with custom rendering pipelines must allow user-defined shader logic without requiring Fluxel to add a built-in node for every operation.

The material graph therefore supports a first-class code/fragment node.

Conceptually:

```text
Fluxel Code Node

inputs:
    uv       : vec2f
    strength : f32

outputs:
    color    : vec3f

resources:
    noise_tex : Texture2D

implementation:
    WGSL function/module
```

Example:

```wgsl
fn evaluate_noise(
    uv: vec2f,
    strength: f32,
) -> vec3f {
    let v = sin(uv.x * 32.0) * cos(uv.y * 32.0);
    return vec3f(v * strength);
}
```

The graph treats this node like any other typed node.

Its implementation is supplied by WGSL rather than a built-in Fluxel node kind.

### 6.1 Why Code Nodes Matter

Built-in nodes are useful for:

```text
common authoring operations
portable semantic helpers
diagnostics
discoverability
```

Code nodes provide:

```text
open-ended expressiveness
project-specific shading
custom BRDF logic
procedural effects
research/experimental shading
custom FramePipeline integration
```

Fluxel must not require the core material node enum to grow indefinitely in order to express new shader logic.

---

## 7. WGSL as the Shader Source Language

Fluxel uses WGSL as its author-facing shader source language.

WGSL is used for:

```text
Material Fragment implementations
Material Code Nodes
shader templates
custom FramePipeline shader modules
shared shader modules/helpers
```

Fluxel does not expose GLSL, HLSL, MSL, or backend-native shader syntax as the primary authoring contract.

Those may exist as backend/output forms where required.

### 7.1 Static features are not preprocessor macros

Fluxel does not depend on a C/GLSL-style preprocessor model such as:

```c
#define USE_NORMAL_MAP
#ifdef USE_NORMAL_MAP
#endif
```

Static configuration belongs to Fluxel variant semantics.

Conceptually:

```text
static feature:
    use_normal_map = true
    alpha_test = false
    skinning = true
```

The shader composer specializes the material/shader composition before final WGSL validation and compilation.

WGSL `override` values may be used where appropriate, but Fluxel's material variant system is not defined by WGSL preprocessor behavior.

---

## 8. Material IR

`Material IR` is not an executable shader language.

It is the normalized, editor-independent composition description of one material configuration.

A useful analogy is:

```text
Material IR ≈ typed link/composition manifest
```

rather than:

```text
Material IR ≈ LLVM IR
```

### 8.1 Material IR responsibilities

Material IR records:

```text
material domain
typed parameters
GPU resource requirements
shader fragment references
fragment input/output contracts
connections between fragment ports
static feature decisions
material output mapping
source provenance
variant-relevant metadata
```

Conceptually:

```rust
pub struct MaterialIr {
    pub domain: MaterialDomain,

    pub parameters: Vec<MaterialParameter>,
    pub resources: Vec<MaterialResource>,

    pub fragments: Vec<ShaderFragmentRef>,
    pub connections: Vec<MaterialConnection>,

    pub static_features: Vec<StaticFeature>,
    pub outputs: MaterialOutputs,

    pub provenance: MaterialProvenance,
}
```

### 8.2 Material IR does not contain shader instructions

Material IR must not define an executable instruction set such as:

```text
LoadParameter
TextureSample
Add
Multiply
Branch
Normalize
Return
```

Those are shader-program semantics and belong to WGSL/Naga.

If a built-in material node performs multiplication, its lowered Material IR references the appropriate fragment/function and records its connections.

For example:

```text
Fragment F0:
    function = sample_base_color

Fragment F1:
    function = multiply_color

Connections:
    UV0                -> F0.uv
    BaseTexture        -> F0.texture
    F0.color           -> F1.a
    TintColor          -> F1.b
    F1.result          -> Surface.BaseColor
```

The actual multiplication implementation lives in WGSL.

---

## 9. MaterialGraph to Material IR Lowering

Lowering removes authoring/editor structure that is irrelevant to shader composition and normalizes the material into a deterministic composition model.

```text
MaterialGraph
        ↓
validate
        ↓
resolve static features
        ↓
normalize nodes/fragments
        ↓
collect parameters/resources
        ↓
normalize connections
        ↓
Material IR
```

Lowering may perform:

```text
graph validation
type validation
cycle detection
domain validation
static-feature resolution
dead-node elimination
constant propagation where useful
resource collection
parameter collection
fragment deduplication
connection normalization
output validation
provenance retention
```

Lowering must not:

```text
create RHI objects
create RenderGraph passes
perform GPU submission
translate to backend-native shader syntax
duplicate Naga's executable IR
```

---

## 10. Material Domain

A material belongs to an explicit domain.

Conceptually:

```rust
#[non_exhaustive]
pub enum MaterialDomain {
    Surface,
    PostProcess,
}
```

The first required domain is:

```text
Surface
```

A domain defines:

```text
legal semantic inputs
legal material outputs
required output shape
which FramePipeline/pass contexts may consume the material
```

A material graph does not own pass construction.

---

## 11. Surface Material Contract

The standard surface contract expresses material semantics rather than a particular shading pipeline.

Conceptually:

```rust
pub struct SurfaceOutput {
    pub base_color: Color3,
    pub metallic: Float,
    pub roughness: Float,
    pub normal: Normal3,
    pub emissive: Color3,
    pub opacity: Float,
}
```

The exact representation may use graph-local typed handles.

The important rule is:

```text
Material describes surface semantics.
FramePipeline decides how those semantics are consumed.
```

The same material may therefore participate in:

```text
Forward shading
Deferred GBuffer generation
Depth-only pass
Shadow pass
custom project passes
```

without embedding a Forward or Deferred renderer inside the material definition.

---

## 12. Built-In Material Nodes

Fluxel may provide a deliberately small built-in node vocabulary for common
authoring operations.

Examples:

```text
Constant
Parameter
TexCoord
VertexColor
Texture2DParameter
SampleTexture2D

Add
Subtract
Multiply
Divide
Min
Max
Clamp
Lerp

Dot
Normalize
Combine
Split

NormalMap

Code / ShaderFragment
```

Built-in nodes are not the semantic ceiling of the system.

The code/fragment node guarantees extensibility.

---

## 13. Parameters

Material parameters are explicitly divided into dynamic and static categories.

### Dynamic parameters

Examples:

```text
base color
roughness
texture binding
emissive intensity
animation time
per-instance values
```

Dynamic changes update runtime data or bindings and should not require shader recompilation.

### Static parameters

Examples:

```text
normal map enabled
alpha test enabled
shading mode
skinning enabled
optional expensive branch
```

Static changes may change:

```text
MaterialVariantKey
fragment selection
shader composition
pipeline requirements
```

---

## 14. Material Resources

Material resource declarations describe required portable GPU resources.

Examples:

```text
Texture2D
Sampler
uniform data
storage/read-only data where supported
```

The material layer owns the logical material requirement.

RHI owns actual GPU resources and binding objects.

Renderer/material runtime resolves a material resource requirement to a concrete per-device GPU resource before/during frame preparation.

The shader reflection and compiled material variant expose a binding-use
manifest. For every declared binding it identifies the logical material slot,
binding class, shader stages, abstract access (sampled/read, uniform read,
storage read/write), and any subresource or buffer-range requirement available
from the interface. Preparation resolves each slot to its concrete portable RHI
texture, buffer, sampler, and binding, preserving the selected RHI's device
provenance, applicable exposed generations, and concrete access ranges. A sampler is listed separately even when
it is paired with a texture in WGSL.

Before a pass is accepted, `FramePipeline` maps that concrete manifest to the
pass's Graph declarations. Graph validates that every texture, buffer, sampler,
and declared range/access used by the bound material is present and compatible,
then includes buffer/texture accesses in dependency and hazard analysis.
Samplers are retained and validated binding inputs, with no memory hazards of
their own. Declaring only a
bind group or material instance is insufficient. Missing, stale, cross-device,
or incompatible bindings fail before recording; Graph never needs to interpret
material semantics to perform this validation.

Changing a concrete texture or buffer instance does not itself change Graph
topology. A cached graph plan may be instantiated after this manifest is
revalidated when access modes, ranges, and dependency edges remain compatible.
Changing any of those contracts requires a new graph definition and compilation.

Material IR must not contain:

```text
VkImage
ID3D12Resource
MTLTexture
descriptor heap index
native binding handles
```

---

## 15. Shader Composition

`fluxel-shader` owns a deliberately neutral `ShaderCompositionRequest`. It
contains generic WGSL source units, declared callable and resource interfaces,
requested entry points/stages, a pass template, geometry interfaces, target
facts, and static specialization values. It has no `MaterialIr`,
`MaterialGraph`, or material-domain field.

Material lowers its private `MaterialIr` into a Shader-owned generic source and
interface contribution. That contribution selects and names WGSL fragment
functions, emits the material parameter/uniform and resource declarations, and
generates `evaluate_material`. It deliberately omits the pass template, entry
points, geometry interface, and final target choices. The active
`FramePipeline` gathers those pass-specific inputs, combines them with the
material contribution, and creates the complete `ShaderCompositionRequest`.
Shader then owns complete-program assembly. Thus `fluxel-material` may depend
on Shader's generic contribution vocabulary, Shader never depends back on
Material, and no third IR or crate is introduced.

The Shader Composer combines the request's:

```text
generic WGSL source units, including material-generated evaluation code
+
FramePipeline-provided pass shader template
+
geometry interface requirements
+
target capability facts
+
static variant decisions
```

into one complete WGSL `ShaderModule`.

Conceptually, the private Material-to-Shader handoff is:

```text
Material IR -> material WGSL + generic interface contribution
                         + FramePipeline-provided pass template,
                           entries, geometry interface, and target facts
                                      ↓
                       ShaderCompositionRequest
                                      ↓
                          Shader Composer
              ↓
       complete WGSL module
```

### 15.1 Example

Material IR may establish:

```text
Surface.BaseColor = fragment_A.result
Surface.Normal    = fragment_B.result
Surface.Roughness = parameter_roughness
```

A Forward template may consume it as:

```text
surface = evaluate_material(material_input)
color = evaluate_lighting(surface, lighting_context)
return color
```

A Deferred template may consume the same material composition as:

```text
surface = evaluate_material(material_input)
write_gbuffer(surface)
```

Material semantics remain shared while pass behavior differs.

---

## 16. Naga Boundary

Naga is the executable shader IR and logical-validation boundary.

Fluxel supplies complete WGSL source/module composition.

Naga, as used by Shader, owns:

```text
WGSL parsing
WGSL validation
executable shader IR
control-flow validation
type validation
resource validation
portable IR generation and reflection
```

Conceptually:

```text
WGSL ShaderModule
        ↓
Naga frontend
        ↓
Naga IR
        ↓
validated logical interface and source artifact
```

Fluxel does not define another executable shader IR above Naga. The selected RHI
device then chooses `Direct`, `Naga`, or `Unsupported` shader routing and its
backend owns any native target lowering or compilation. Shader does not choose a
backend dialect or recreate that compiler boundary.

---

## 17. Shader Artifact

The Shader system owns the conversion of a validated composed program into a
renderer-consumable logical `ShaderArtifact`, together with its
`ShaderInterface` and `PipelineRequirements`. These are Shader-system products,
not historical RHI semantic types or precompiled native binaries.

Associated information may include:

```text
ShaderInterface
stage entry points
resource/binding requirements
vertex input requirements
render-target requirements
variant identity
reflection
cache identity
```

The exact public shape reuses RHI portable shader/pipeline vocabulary where
appropriate. Renderer preparation adapts the artifact's composed WGSL and entry
metadata to the selected RHI shader/pipeline creation API. RHI owns the
device-specific route, ABI, lowering, and executable object cache; it does not
create or own the logical `ShaderArtifact`.

---

## 18. Material Variant

A compiled material variant is the runtime representation of one static material configuration.

Material asset identity is durable only as the pair of its asset ID and content
generation. This prevents a hot reload or replacement under the same
`AssetId<MaterialAsset>` from reusing a variant compiled for old content.

```rust
pub type MaterialAssetId = AssetId<MaterialAsset>;

pub struct MaterialAssetRef {
    pub id: MaterialAssetId,
    pub generation: ContentGeneration,
}
```

Conceptually:

```rust
pub struct CompiledMaterialVariant {
    pub material: MaterialAssetRef,
    pub variant_key: MaterialVariantKey,

    pub shader_artifacts: Vec<ShaderArtifactRef>,
    pub parameter_schema: MaterialParameterSchema,
    pub resource_schema: MaterialResourceSchema,

    pub pipeline_requirements: PipelineRequirements,
}
```

A variant is affected by static choices.

Dynamic material-instance values are not part of shader compilation identity unless they change static behavior.

---

## 19. Material Instance

`MaterialInstance` is runtime parameter/resource state over a compiled material family.

Conceptually:

```rust
pub struct MaterialInstance {
    pub material: MaterialAssetRef,
    pub static_variant: MaterialVariantKey,
    pub parameters: MaterialParameterValues,
    pub resources: MaterialResourceValues,
}
```

`MaterialInstanceId` may identify mutable renderer-domain instance state, but it
cannot replace `MaterialAssetRef`: every instance and every variant retains the
asset ID plus generation it represents.

It does not own:

```text
MaterialGraph
Naga IR
RenderGraph passes
native descriptor handles
```

Renderer/material runtime resolves the instance into the bindings required by the selected compiled variant.

---

## 20. FramePipeline Integration

`FramePipeline` owns rendering policy.

During frame preparation/building it may:

```text
select material variant
select shader/pipeline
resolve material runtime bindings
decide pass topology
declare RenderGraph resources
declare RenderGraph passes
```

Conceptually:

```text
RenderObject
    ↓
MaterialInstance
    ↓
Material Variant Resolver
    ↓
CompiledMaterialVariant
    ↓
FramePipeline
    ↓
RenderGraph pass declaration
```

### 21.1 Preparation vs execution

During graph construction:

```text
FramePipeline selects/resolves:
    shader
    pipeline
    material bindings
    resource requirements
```

During graph execution:

```text
RenderGraph pass uses:
    selected shader/pipeline
    material runtime data
    every concrete resource access in the resolved binding-use manifest
```

Preparation may resolve residency and create/update persistent material
bindings, but it cannot hide an access from the frame graph. Persistent textures,
static meshes, and streaming terrain tiles use residency upload. Per-frame
transforms, bone palettes, and similar short-lived input use frame-data upload.
Both paths may share staging and RHI completion tracking, but frame data is not a
long-lived residency asset and its coordinator cannot become a second frame
scheduler. GPU skinning, particle simulation, culling, indirect-argument work,
and all other frame GPU work are Graph passes. If an upload's visibility is
required by a frame pass, its enforceable producer dependency and incoming state
are imported into Graph/RHI submission planning before that pass records. A
pending upload can therefore be usable by that declared consumer without being
globally complete; a CPU-ready flag alone is insufficient.

---

## 21. RenderGraph Boundary

Material code cannot:

```text
create or order RenderGraph passes
own SceneColor
own Depth
own History
submit GPU work
control graph scheduling
own presentation
```

RenderGraph cannot:

```text
compile MaterialGraph
interpret Material IR semantics
compile WGSL
own material assets
```

The boundary is:

```text
CompiledMaterialVariant
        ↓
FramePipeline chooses how to use it
        ↓
RenderGraph pass
```

A Graph pass is one execution node, not one shader program. A raster pass may
record many draws and select multiple graphics pipelines, each with a vertex
stage and optional fragment stage. A compute pass records one or more dispatches using compute
pipelines. A copy pass requires no shader. Raster and compute work are separate
pass declarations unless an RHI contract explicitly supports a combined scope.

---

## 22. RHI Boundary

RHI is the shared portable GPU foundation.

Material runtime, shader system, renderer, and RenderGraph may each reuse RHI portable contracts where required.

RHI owns:

```text
device identity
portable shader/pipeline objects
buffers/textures/samplers
bindings
recording
submission
completion
presentation
physical allocation
backend realization
```

RHI does not own:

```text
MaterialGraph
Material IR
material semantics
material variants
ShaderArtifact
ShaderInterface
PipelineRequirements
FramePipeline policy
```

No upper layer may depend on backend-private native API objects.

---

## 23. Identity and Caching

Material/shader compilation must distinguish identities rather than use one universal material hash.

Relevant identities may include:

```text
MaterialGraph identity
Material IR identity
Material static variant identity
MaterialAssetRef (AssetId + ContentGeneration)
Shader composition identity
WGSL module identity
Naga compilation identity
ShaderArtifact identity
ShaderInterface compatibility
Pipeline identity
```

Dynamic instance parameter changes must not invalidate shader compilation identity.

---

## 24. Source Provenance and Diagnostics

Diagnostics must remain traceable to authoring sources.

Material IR and shader composition should preserve enough provenance to report errors back to:

```text
source node
source port
Code Node
WGSL source span
generated fragment
material parameter
material output
```

Naga diagnostics should be remapped through retained source provenance where possible.

---

## 25. Error Model

Errors should remain owned by the subsystem that understands them.

Material graph errors:

```text
type mismatch
cycle
missing connection
invalid domain operation
invalid parameter
```

Material composition errors:

```text
missing fragment
port mismatch
resource conflict
invalid output mapping
static feature conflict
```

WGSL/Naga errors:

```text
syntax
type
control flow
resource/interface validation
```

Renderer errors:

```text
variant not available
material instance/schema mismatch
resource resolution failure
unsupported pass requirement
```

RHI errors:

```text
shader/pipeline creation
binding incompatibility
resource/device mismatch
submission/completion/presentation
```

---

## 26. Relationship to slot-graph

`slot-graph` may be reused by `MaterialGraph` where its generic typed DAG mechanics are useful.

Potential reuse includes:

```text
stable node IDs
typed slots
connection validation
topological traversal
cycle detection
subgraph composition
```

MaterialGraph semantics remain owned by the material subsystem.

RenderGraph remains independent from `slot-graph`; its GPU resource-version and
hazard model is specialized enough to remain the renderer's own graph system.

---

## 27. Authoring and Runtime Asset Forms

The authoring form may contain:

```text
MaterialGraph
Code Node WGSL
source provenance
parameter defaults
resource references
```

The normalized/cooked form may contain:

```text
Material IR
normalized fragment references
static variant configuration
parameter/resource schemas
compiled/cached shader variants
source/compiler version metadata
```

Exact serialization format is a separate asset-format decision.

---

## 28. Initial Supported Surface

The first implementation should prove architecture rather than node breadth.
The vocabulary below describes the initial direction, not a requirement to
implement every node and surface output before textured output or the MVP.
Start with the fixed material functions and outputs required by that consumer;
broader nodes and Code Nodes belong to material expansion.

### Material domain

```text
Surface
```

### Built-in nodes

```text
Constant
Parameter
TexCoord
Texture2DParameter
SampleTexture2D
Add
Multiply
Lerp
Normalize
NormalMap
Code / ShaderFragment
```

### Standard outputs

```text
BaseColor
Metallic
Roughness
Normal
Emissive
Opacity
```

## 29. Implementation Plan

The authoritative sequence is [the implementation plan](version-plan.md) and
the ecosystem roadmap. This section states Material's contribution to that
sequence rather than creating a second schedule.

| Stage | Material/Shader work and acceptance |
| --- | --- |
| S0/S1 foundation | Select exact Bases/RHI revisions, then establish the renderer-internal Graph, bounded multi-frame execution, and completion-based retirement. Material runtime contracts do not freeze before this closure. |
| S2 Shader | Create `fluxel-shader`; prove a standalone complete WGSL raster program and, where supported, compute through logical validation/reflection, selected RHI shader/pipeline creation, and real Graph/RHI execution. |
| S3 Material | Create the smallest `fluxel-material` consumer: Rust-defined fragments, typed parameter/resource schemas, static and dynamic choices, deterministic Material IR, and a multi-fragment `evaluate_material` contribution. Metadata must exactly agree with Shader reflection. A complete editor graph and Code Node UX are not prerequisites. |
| S4 scene | RenderScene batch/snapshot work may proceed alongside S3 after S2 establishes the Shader boundary. |
| S5 Forward | FramePipeline combines the material contribution with the selected pass template and creates a material-backed Forward path. It resolves concrete bindings and declares their complete Graph manifests. |
| S6 MVP closure | An external Rust Feature, external FramePipeline, and custom material contribution execute together through the normal Graph/RHI path. Unsupported requirements and binding incompatibilities fail structurally. |
| S7 expansion | Selected consumers extend Code Nodes, authoring UX, PBR, additional material domains, and pipeline/Feature contracts. A second materially different Feature validates cross-pass and shader-extension seams before they stabilize. |

The S3 acceptance proof is deterministic: the same material input produces the
same Material IR and contribution; dynamic values do not recompile a Shader
variant; a static choice does. The S5/S6 proofs retain source provenance,
reflection diagnostics, the exact RHI revision, real target output, and the
binding manifest for every participating pass.

---

## 30. Non-Goals

Do not make the first implementation depend on:

```text
MaterialX compatibility
custom shader language
custom executable shader IR
GLSL/HLSL authoring
backend-native shader code in material assets
full shader debugger
all possible material domains
```

---

## 31. Final Architecture

```text
AUTHORING

Rust builder / procedural generator / tests
        ↓
MaterialGraph
(static structure)
        ↓
Material IR
(normalized composition/link metadata)
        ↓
Material lowering
        ↓
generic material WGSL + interface contribution
        ↓
FramePipeline / another Shader consumer adds pass template,
entry points, geometry interface, and target requirements
        ↓
ShaderCompositionRequest -> Shader Composer
        ↓
Complete WGSL ShaderModule
        ↓
Naga
(executable shader IR and logical validation)
        ↓
ShaderArtifact / ShaderInterface / Pipeline Requirements
 (logical products owned by Shader System)
        ↓
selected RHI route -> backend-native lowering and compilation


RUNTIME

RenderScene
        ↓
FramePipeline
        ↓
resolve MaterialInstance
        ↓
CompiledMaterialVariant
        ↓
RenderGraph
        ↓
use Shader / Pipeline + Material Runtime bindings
        ↓
RHI
        ↓
private GPU backend
```

The defining rules are:

> `MaterialGraph` describes static material structure.

> `Material Fragment` describes dynamic shader computation.

> `Material IR` describes how fragments, parameters, resources, and material outputs are composed.

> Material lowers private composition into a generic Shader contribution.
> `FramePipeline` or another Shader consumer creates the complete
> `ShaderCompositionRequest`; Shader never receives `MaterialIr`.

> WGSL is Fluxel's shader source language.

> Naga owns executable shader IR and logical validation. RHI owns selected-device
> route selection and backend-native lowering or compilation.

> Code Nodes are first-class material nodes.

> `FramePipeline` decides how compiled material semantics participate in rendering.

> RenderGraph owns GPU-work dependencies and lifetimes, not material semantics.

> A bound material contributes its reflected, concrete binding-use manifest to
> every Graph pass that uses it.

> RHI owns portable GPU execution, not material or renderer policy.
