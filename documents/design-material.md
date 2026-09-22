# Fluxel Material System Design

## 1. Purpose

This document defines the architecture and implementation plan for the Fluxel material system.

The material system is designed around four explicit boundaries:

```text
MaterialGraph      = static material structure
Material Fragment  = dynamic shader process
Material IR        = normalized composition/link metadata
Naga IR            = executable shader program IR
```

Fluxel does not implement its own material editor or shader compiler frontend.

The intended toolchain is:

```text
Blender
  -> Fluxel MaterialGraph
  -> Material IR
  -> Shader Composition
  -> WGSL ShaderModule
  -> Naga
  -> Shader System
       -> ShaderArtifact / ShaderInterface / Pipeline Requirements
       -> RHI shader/pipeline creation
```

Blender is the primary visual authoring environment.

WGSL is the Fluxel-facing shader source language for custom material code, shader fragments, and custom pipeline shader modules.

Naga owns shader parsing, validation, executable shader IR, and target translation.

Fluxel's shader system owns `ShaderArtifact`, `ShaderInterface`, pipeline
requirements, and their cache identities. RHI consumes those portable products to
create shader and pipeline objects; it is not their semantic owner. Fluxel owns
material semantics, composition, resource contracts, variant semantics, and
integration with `FramePipeline`, `RenderGraph`, and RHI.

---

## 2. Architectural Position

Material compilation and frame execution are related but distinct flows.

### Material compilation

```text
Blender Shader Nodes
        ↓
Fluxel MaterialGraph
        ↓
Material IR
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

Blender nodes, a Rust builder, tests, procedural material generation, or future importers may all construct the same Fluxel `MaterialGraph`.

The graph is not Blender-specific.

Blender is one frontend.

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
Blender translation
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
    Ui,
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

## 12. Blender Integration

Blender is the primary material authoring UI.

The intended path is:

```text
Blender ShaderNodeTree
        ↓
Fluxel Blender Adapter
        ↓
Fluxel MaterialGraph
        ↓
Material IR
        ↓
Shader Composer
        ↓
WGSL
        ↓
Naga
        ↓
Fluxel Renderer / RHI
```

The Blender adapter owns:

```text
Blender node mapping
Blender socket mapping
parameter extraction
texture/resource references
source provenance
unsupported-node diagnostics
Code Node editing/integration
incremental recompile triggers
asset export
```

The material core does not import Blender APIs.

### 12.1 Principled BSDF

Blender `Principled BSDF` is not the fundamental Fluxel material model.

It is translated into Fluxel surface semantics.

Example:

```text
Principled Base Color  -> Surface.BaseColor
Principled Metallic    -> Surface.Metallic
Principled Roughness   -> Surface.Roughness
Principled Normal      -> Surface.Normal
Principled Emission    -> Surface.Emissive
Principled Alpha       -> Surface.Opacity
```

Unsupported Principled features must fail explicitly rather than silently approximate behavior.

---

## 13. Built-In Material Nodes

Fluxel may provide a deliberately small built-in node vocabulary for common authoring operations and Blender translation.

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

## 14. Parameters

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

## 15. Material Resources

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

Material IR must not contain:

```text
VkImage
ID3D12Resource
MTLTexture
descriptor heap index
native binding handles
```

---

## 16. Shader Composition

The Shader Composer combines:

```text
Material IR
+
WGSL Material Fragments
+
FramePipeline / pass shader template
+
geometry interface requirements
+
target capability facts
+
static variant decisions
```

into one complete WGSL `ShaderModule`.

Conceptually:

```text
Material IR
        \
WGSL fragments
          \
Forward/Deferred/custom pass template
            \
geometry/pass interface
              ↓
          Shader Composer
              ↓
       complete WGSL module
```

### 16.1 Example

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

## 17. Naga Boundary

Naga is the executable shader compiler/IR boundary.

Fluxel supplies complete WGSL source/module composition.

Naga owns:

```text
WGSL parsing
WGSL validation
executable shader IR
control-flow validation
type validation
resource validation
target translation where supported
```

Conceptually:

```text
WGSL ShaderModule
        ↓
Naga frontend
        ↓
Naga IR
        ↓
SPIR-V / MSL / HLSL / GLSL / WGSL output as required
```

Fluxel does not define another executable shader IR above Naga.

---

## 18. Shader Artifact

The shader system owns the conversion of validated shader output into a
renderer/RHI-consumable `ShaderArtifact`, together with its `ShaderInterface` and
`PipelineRequirements`. These are shader-system compilation products, not RHI
semantic objects.

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

The exact public shape should reuse RHI portable shader/pipeline vocabulary where
appropriate. RHI consumes an artifact and requirements to create its portable
shader/pipeline objects and execute GPU work; it does not create, own, or cache
the `ShaderArtifact` itself.

---

## 19. Material Variant

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

## 20. Material Instance

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
Blender nodes
Naga IR
RenderGraph passes
native descriptor handles
```

Renderer/material runtime resolves the instance into the bindings required by the selected compiled variant.

---

## 21. FramePipeline Integration

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
    declared resources
```

---

## 22. RenderGraph Boundary

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
interpret Blender nodes
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

---

## 23. RHI Boundary

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
Blender translation
```

No upper layer may depend on backend-private native API objects.

---

## 24. Identity and Caching

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

## 25. Source Provenance and Diagnostics

Diagnostics must remain traceable to authoring sources.

Material IR and shader composition should preserve enough provenance to report errors back to:

```text
Blender node
Blender socket
Code Node
WGSL source span
generated fragment
material parameter
material output
```

Naga diagnostics should be remapped through retained source provenance where possible.

---

## 26. Error Model

Errors should remain owned by the subsystem that understands them.

Material graph errors:

```text
type mismatch
cycle
missing connection
invalid domain operation
invalid parameter
unsupported Blender translation
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

## 27. Relationship to slot-graph

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

RenderGraph remains independent from `slot-graph`; its GPU resource-version and hazard model is specialized enough to remain its own graph system.

---

## 28. Authoring and Runtime Asset Forms

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

Shipping runtime does not require Blender.

Exact serialization format is a separate asset-format decision.

---

## 29. Initial Supported Surface

The first implementation should prove architecture rather than node breadth.

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

### Blender translation subset

```text
Material Output
Principled BSDF subset
Image Texture
Texture Coordinate
Normal Map
RGB
Value
Add
Multiply
Mix
Code / custom WGSL node
```

Unsupported nodes fail with structured diagnostics.

---

## 30. Implementation Plan

### Phase A — Semantic substrate

Implement:

```text
MaterialDomain
MaterialValueType
typed ports
MaterialParameterSchema
MaterialResourceSchema
SurfaceOutput
MaterialGraph identity/provenance
```

### Phase B — Fragment model

Implement:

```text
ShaderFragment metadata
authoring exposure/semantic names (not executable interface truth)
WGSL module/source representation
typed fragment inputs/outputs
resource requirements
Code Node
built-in nodes backed by fragments
```

### Phase C — Material IR

Implement:

```text
MaterialGraph validation
static feature resolution
fragment normalization
parameter/resource collection
connection normalization
dead-node elimination
Material IR serialization/debug dump
```

Acceptance:

```text
Material IR contains composition metadata only
no executable shader instruction language exists in Material IR
deterministic graph input produces deterministic Material IR
```

### Phase D — Shader composition

Implement:

```text
WGSL fragment composition
symbol/name isolation
resource binding reconciliation
material evaluation function generation
pass-template integration
source map/provenance retention
```

Acceptance:

```text
Material IR + pass template -> complete WGSL module
```

### Phase E — Naga integration

Implement:

```text
WGSL parse
validation
Naga IR generation
reflection/interface extraction
exact metadata-to-reflection validation and NormalizedFragmentInterface creation
shader-system artifact/interface/requirement creation
diagnostic source remapping
```

### Phase F — Material variants/runtime

Implement:

```text
MaterialVariantKey
CompiledMaterialVariant
dynamic/static parameter split
runtime parameter/resource binding
pipeline requirement extraction
cache identity
```

### Phase G — FramePipeline integration

Implement:

```text
material variant resolver
shader/pipeline resolver
FramePipeline material services
RenderGraph pass declaration using resolved shader/material requirements
```

### Phase H — Blender integration

Implement:

```text
Blender node translation
Code Node UX/integration
source provenance
preview recompile
unsupported-node diagnostics
```

Acceptance:

```text
Blender material -> Fluxel preview
same exported material -> Fluxel runtime
equivalent Fluxel rendering semantics
```

---

## 31. Non-Goals

Do not make the first implementation depend on:

```text
hundreds of Blender nodes
full Principled BSDF coverage
MaterialX compatibility
custom Fluxel material editor
custom shader language
custom executable shader IR
GLSL/HLSL authoring
backend-native shader code in material assets
full shader debugger
all possible material domains
```

---

## 32. Final Architecture

```text
AUTHORING

Blender Shader Editor
        ↓
MaterialGraph
(static structure)
        ↓
Material IR
(normalized composition/link metadata)
        ↓
Shader Composer
        ↑
Material Fragments / Code Nodes
(dynamic WGSL processes)
        ↓
Complete WGSL ShaderModule
        ↓
Naga
(executable shader IR/compiler)
        ↓
ShaderArtifact / ShaderInterface / Pipeline Requirements
 (owned by Shader System; consumed by RHI creation)


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

> WGSL is Fluxel's shader source language.

> Naga owns executable shader IR and shader-language compilation.

> Code Nodes are first-class material nodes.

> Blender is the primary visual material editor.

> `FramePipeline` decides how compiled material semantics participate in rendering.

> RenderGraph owns GPU-work dependencies and lifetimes, not material semantics.

> RHI owns portable GPU execution, not material or renderer policy.
