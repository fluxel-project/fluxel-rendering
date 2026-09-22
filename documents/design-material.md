# Fluxel Material System Design

> Status: **Active next-version architecture**
>
> Scope: Material authoring graph, material semantic model, Material IR, lowering boundary,
> Blender integration boundary, and renderer-facing compiled material contract.
>
> This document is the primary semantic contract for the next version's shader
> assembly and material work. It does not define the full Blender add-on UI or
> the implementation details of the two built-in renderer pipelines.

---

## 1. Core Decision

Fluxel should not design its material system as a clone of Blender's node editor.

Blender is the primary authoring environment.

Fluxel owns the rendering semantics.

The intended flow is:

```text
Blender Shader Nodes
        |
        | Blender -> Fluxel translation
        v
Fluxel MaterialGraph
        |
        v
Material IR
        |
        v
Shader Composition / Compiler
        |
        v
ShaderArtifact + ShaderInterface
        |
        v
CompiledMaterialVariant
        |
        v
FramePipeline / Renderer
        |
        v
RenderGraph
        |
        v
RHI
```

The key consistency rule is:

> Blender preview and game runtime must execute the same Fluxel material semantics.

The target is therefore:

```text
Fluxel in Blender == Fluxel in Game
```

not:

```text
EEVEE == Fluxel
Cycles == Fluxel
```

Blender provides the editing UX.

Fluxel provides the material semantics and the actual preview renderer.

---

## 2. Crate Ownership

The material system should **not** live inside `fluxel-renderer`.

Recommended workspace structure:

```text
fluxel-rendering/
    crates/
        rhi/
        rendergraph/
        material/
        shader/
        renderer/
```

Suggested responsibilities:

```text
fluxel-material
    MaterialGraph
    material node vocabulary
    MaterialDomain
    MaterialValueType
    material parameters
    Material IR
    graph validation
    graph -> Material IR lowering

fluxel-shader
    shader fragments/modules
    shader composition
    shader target profile
    permutation/variant resolution
    source generation / compiler frontend
    reflection
    ShaderArtifact production

fluxel-renderer
    RenderScene
    Culling / Sorting
    FramePipeline SPI
    material-variant selection
    RenderGraph construction

fluxel-rendergraph
    pass/resource dependency graph

fluxel-rhi
    ShaderArtifact acceptance
    bindings
    pipelines
    commands
    submission
```

The dependency direction should be approximately:

```text
material
    |
    v
shader substrate

renderer
    |
    +---- consumes material compiled output
    |
    v
rendergraph
    |
    v
rhi
```

`fluxel-material` must not depend on `fluxel-renderer`.

---

## 3. First Implementation Priority

Do **not** begin with a large list of material nodes.

The first thing to define is the stable semantic substrate:

```text
1. MaterialDomain
2. MaterialValueType
3. Material parameter model
4. Material output contract
5. Material IR
6. MaterialGraph -> Material IR lowering
7. Minimal MaterialGraph node API
8. Blender-node mapping
```

The reason is simple:

> UI nodes are replaceable frontends; Material IR is the semantic contract.

If the public API is designed directly around Blender node classes, Fluxel will inherit
Blender's UI vocabulary and compatibility burden.

---

## 4. MaterialGraph vs Material IR

These are different objects.

### 4.1 MaterialGraph

MaterialGraph is an editable authoring representation.

It contains:

```text
nodes
typed input/output ports
connections
parameters
domain
material outputs
source/editor metadata
```

It is suitable for:

```text
Blender import
visual editing
serialization
diagnostics
hot reload
tooling
```

### 4.2 Material IR

Material IR is the compiler-facing semantic representation.

It contains:

```text
typed values
typed operations
resource reads
parameters
domain outputs
static decisions
normalized semantic operations
```

It does not preserve editor layout or Blender UI concepts unless retained as optional provenance.

The pipeline is:

```text
MaterialGraph
    |
    | validate + lower
    v
Material IR
```

---

## 5. Material Domains

The initial design should make domain explicit.

Conceptually:

```rust
#[non_exhaustive]
pub enum MaterialDomain {
    Surface,
    PostProcess,
    Ui,
}
```

Only `Surface` needs to be implemented first.

Domain determines:

```text
which inputs exist
which outputs are legal
which semantic operations are legal
which renderer passes may consume the material
```

A material graph is never allowed to create RenderGraph passes.

---

## 6. Material Value Types

Material graph ports must be strongly typed.

Conceptually:

```rust
#[non_exhaustive]
pub enum MaterialValueType {
    Bool,
    Float,
    Vec2,
    Vec3,
    Vec4,

    Color3,
    Color4,

    Normal3,

    Texture2D,
    Sampler,
}
```

The exact set should remain small initially.

Semantic types such as:

```text
Color3
Normal3
```

are intentionally distinct from plain vectors where useful.

This lets validation reject meaningless connections before shader compilation.

---

## 7. Typed Value Handles

The Rust authoring API should expose typed handles rather than stringly-typed ports.

Conceptually:

```rust
pub struct Value<T> {
    node: MaterialNodeId,
    output: MaterialPortId,
    _marker: PhantomData<T>,
}
```

Marker types may include:

```rust
pub struct Float;
pub struct Vec2;
pub struct Vec3;
pub struct Vec4;
pub struct Color3;
pub struct Color4;
pub struct Normal3;
```

This makes:

```rust
let roughness: Value<Float>;
let base_color: Value<Color3>;
```

different at compile time.

---

## 8. MaterialGraph

Conceptual API:

```rust
pub struct MaterialGraph {
    /* editable typed DAG */
}
```

Creation:

```rust
let mut graph = MaterialGraph::new(MaterialDomain::Surface);
```

The graph owns:

```text
MaterialNodeId
MaterialPortId
connections
parameters
surface output
diagnostic provenance
```

---

## 9. Node Identity

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialNodeId(u64);
```

Node identity is graph-local authoring identity.

It is not:

```text
ShaderArtifact identity
pipeline identity
GPU object identity
Blender node pointer
```

Blender integration may retain a source mapping:

```text
Blender node identifier
    -> MaterialNodeId
```

for diagnostics and hot reload.

---

## 10. Minimal Node Vocabulary

The first version should intentionally support only a small semantic subset.

### Sources

```text
Constant
Parameter
TexCoord
VertexColor
Texture2DParameter
```

### Texture

```text
SampleTexture2D
```

### Arithmetic

```text
Add
Subtract
Multiply
Divide
Min
Max
Clamp
Lerp
```

### Vector

```text
Dot
Normalize
Combine
Split
```

### Semantic helpers

```text
NormalMap
```

This is enough to prove the architecture.

Do not start by reproducing hundreds of Blender nodes.

---

## 11. Node API

The recommended public Rust API is a typed builder over an internal node representation.

Example:

```rust
let mut material = MaterialGraph::new(MaterialDomain::Surface);

let uv = material.tex_coord(0)?;

let albedo = material.texture2d_parameter(
    "base_color_texture",
)?;

let sampled = material.sample_texture2d(
    albedo,
    uv,
)?;

let tint = material.color3_parameter(
    "base_color",
    [1.0, 1.0, 1.0],
)?;

let base_color = material.mul(
    sampled.rgb(),
    tint,
)?;

material.set_surface_output(SurfaceOutput {
    base_color,
    roughness: material.float_constant(0.5)?,
    metallic: material.float_constant(0.0)?,
    normal: material.default_normal()?,
    emissive: material.color3_constant([0.0, 0.0, 0.0])?,
    opacity: material.float_constant(1.0)?,
})?;
```

This API is intended for:

```text
Blender translator
tests
procedural material generation
other future authoring tools
```

Game runtime code normally consumes compiled material assets instead.

---

## 12. Raw Node Representation

Internally, each authoring operation lowers to an explicit node kind.

Conceptually:

```rust
#[non_exhaustive]
pub enum MaterialNodeKind {
    FloatConstant(f32),
    Vec2Constant([f32; 2]),
    Vec3Constant([f32; 3]),
    Vec4Constant([f32; 4]),

    Parameter(MaterialParameterId),

    TexCoord {
        channel: u8,
    },

    SampleTexture2D,

    Add,
    Subtract,
    Multiply,
    Divide,

    Min,
    Max,
    Clamp,
    Lerp,

    Dot,
    Normalize,

    NormalMap,
}
```

The public API should prefer typed helpers.

The raw enum exists for:

```text
serialization
tooling
compiler inspection
graph import
```

It should not become a giant Blender-node compatibility enum.

---

## 13. Connections

MaterialGraph is a typed DAG.

Conceptually:

```rust
pub fn connect<T>(
    &mut self,
    output: Value<T>,
    input: Input<T>,
) -> Result<(), MaterialGraphError>;
```

Validation rejects:

```text
wrong type
foreign graph
stale handle
multiple source for single input
cycle
invalid domain operation
missing required input
```

No implicit numeric/vector conversion should be assumed initially.

Conversions should be explicit operations.

---

## 14. Parameters

Static and dynamic values must be separated.

Conceptually:

```rust
pub enum MaterialParameterKind {
    Dynamic,
    Static,
}
```

Dynamic examples:

```text
base color
roughness
texture
normal intensity
emissive color
```

Static examples:

```text
feature switch
shading mode
alpha-test enabled
optional expensive branch
```

The rule is:

```text
dynamic parameter
    -> runtime binding / uniform data

static parameter
    -> material variant identity / shader compilation
```

Changing a dynamic scalar must not compile a new shader.

---

## 15. Parameter Schema

Conceptually:

```rust
pub struct MaterialParameterSchema {
    parameters: Vec<MaterialParameter>,
}
```

```rust
pub struct MaterialParameter {
    pub id: MaterialParameterId,
    pub name: String,
    pub value_type: MaterialValueType,
    pub kind: MaterialParameterKind,
    pub default_value: MaterialValue,
}
```

Material instances later reference this schema.

---

## 16. Surface Output Contract

The first supported domain should define one explicit semantic output.

Conceptually:

```rust
pub struct SurfaceOutput {
    pub base_color: Value<Color3>,

    pub metallic: Value<Float>,
    pub roughness: Value<Float>,

    pub normal: Value<Normal3>,

    pub emissive: Value<Color3>,
    pub opacity: Value<Float>,
}
```

This is intentionally a semantic structure, not a shader function.

The renderer/shader system decides how these values participate in:

```text
Forward shading
Deferred GBuffer
Depth-only pass
Shadow pass
other pipeline variants
```

MaterialGraph itself does not know those passes.

---

## 17. Standard Surface vs Blender Principled BSDF

Blender's `Principled BSDF` should not become the foundational Fluxel node type.

Instead:

```text
Blender Principled BSDF
        |
        | translator
        v
Fluxel SurfaceOutput / StandardSurface semantics
```

Only the subset whose semantics Fluxel explicitly supports is translated.

Unsupported inputs fail with structured diagnostics.

Example:

```text
Principled Base Color
    -> SurfaceOutput.base_color

Principled Metallic
    -> SurfaceOutput.metallic

Principled Roughness
    -> SurfaceOutput.roughness

Principled Normal
    -> SurfaceOutput.normal

Principled Emission
    -> SurfaceOutput.emissive

Principled Alpha
    -> SurfaceOutput.opacity
```

Other Principled features may initially return:

```text
UnsupportedMaterialFeature
```

rather than silently approximate them.

---

## 18. Material IR

The compiler-facing IR should be smaller and more normalized than MaterialGraph.

Conceptually:

```rust
pub struct MaterialModule {
    pub domain: MaterialDomain,
    pub parameters: MaterialParameterSchema,
    pub instructions: Vec<MaterialInstruction>,
    pub outputs: MaterialOutputs,
}
```

Example instruction family:

```rust
#[non_exhaustive]
pub enum MaterialInstruction {
    Constant {
        result: MaterialValueId,
        value: MaterialValue,
    },

    LoadParameter {
        result: MaterialValueId,
        parameter: MaterialParameterId,
    },

    TextureSample2D {
        result: MaterialValueId,
        texture: MaterialValueId,
        uv: MaterialValueId,
    },

    Add {
        result: MaterialValueId,
        lhs: MaterialValueId,
        rhs: MaterialValueId,
    },

    Multiply {
        result: MaterialValueId,
        lhs: MaterialValueId,
        rhs: MaterialValueId,
    },

    Lerp {
        result: MaterialValueId,
        a: MaterialValueId,
        b: MaterialValueId,
        t: MaterialValueId,
    },

    Normalize {
        result: MaterialValueId,
        value: MaterialValueId,
    },
}
```

This is conceptual shape only.

The real IR may use dense IDs and typed instruction tables.

---

## 19. Graph Lowering

Compilation begins:

```rust
pub fn lower_material_graph(
    graph: &MaterialGraph,
    target: &MaterialLoweringTarget,
) -> Result<MaterialModule, MaterialCompileError>;
```

Lowering performs:

```text
graph validation
domain validation
static-switch resolution
type validation
constant folding
dead-node elimination
parameter collection
resource collection
topological ordering
normalization into Material IR
```

It must not create RHI objects.

---

## 20. Material Compile Target

Material compilation consumes normalized renderer/shader target facts.

Conceptually:

```rust
pub struct MaterialLoweringTarget {
    pub profile: ShaderTargetProfile,
    pub features: MaterialFeatureSet,
}
```

It must not contain:

```text
VkDevice
ID3D12Device
MTLDevice
WebGPUDevice
native descriptor
native compiler pointer
```

The target is canonicalizable and suitable for cache keys.

---

## 21. Compiled Material Variant

The material system's renderer-facing product is not the graph.

Conceptually:

```rust
pub struct CompiledMaterialVariant {
    pub domain: MaterialDomain,

    pub variant_key: MaterialVariantKey,

    pub shader_requirements: MaterialShaderRequirements,

    pub parameter_schema: MaterialParameterSchema,

    pub interface_requirements: MaterialInterfaceRequirements,
}
```

After shader composition/compiler work, this may reference or contain:

```text
ShaderArtifact(s)
ShaderInterface
PipelineInterface requirements
binding map
vertex requirements
render-target requirements
```

The exact cross-crate shape should be finalized together with `fluxel-shader`.

---

## 22. Renderer Boundary

Renderer consumes compiled material semantics.

It does not interpret the editable graph.

Correct:

```text
MaterialGraph
    -> Material IR
    -> CompiledMaterialVariant
    -> Renderer
```

Incorrect:

```text
Renderer
    -> walk MaterialGraph nodes every frame
```

The renderer may select among already compiled variants based on:

```text
FramePipeline
pass kind
mesh/vertex requirements
target profile
enabled renderer features
```

---

## 23. RenderGraph Boundary

MaterialGraph cannot:

```text
create passes
order passes
own SceneColor
own Depth
own History
submit work
create presentation targets
```

RenderGraph cannot:

```text
interpret material nodes
compile Material IR
understand Blender nodes
```

The boundary is:

```text
CompiledMaterialVariant
        |
        v
FramePipeline chooses usage
        |
        v
RenderGraph pass
```

---

## 24. Blender Integration

Blender is an authoring frontend.

The integration flow is:

```text
Blender ShaderNodeTree
        |
        v
Fluxel Blender Adapter
        |
        | supported-node translation
        v
MaterialGraph
        |
        v
Material IR
        |
        v
Fluxel Shader Compiler
        |
        v
Fluxel Renderer
        |
        v
Blender Viewport
```

The Blender adapter owns:

```text
mapping Blender node types
mapping Blender sockets
mapping Blender parameters
source-location/provenance
unsupported-node diagnostics
incremental recompile trigger
asset export
```

The material core does not import Blender APIs.

---

## 25. Blender Support Policy

Fluxel supports an explicit Blender node subset.

For example, an initial subset may include:

```text
Material Output
Principled BSDF subset

Image Texture
Texture Coordinate
Normal Map

RGB / Value

Math subset
Vector Math subset
Mix

Separate / Combine
```

A node is supported only when Fluxel defines equivalent semantics.

Unsupported behavior must be explicit.

Do not silently substitute an approximate node.

---

## 26. Preview Consistency

WYSIWYG depends on sharing the same execution semantics.

Blender editing mode should eventually use:

```text
Fluxel MaterialGraph
Fluxel Material IR
Fluxel Shader Composition
Fluxel FramePipeline
Fluxel RHI
```

for viewport rendering.

Game runtime uses the same layers.

Therefore the preview does not attempt to reproduce EEVEE.

It runs Fluxel itself.

---

## 27. Exported Material Asset

The runtime asset should not require Blender.

Conceptually:

```text
Blender file
    |
    v
Fluxel exporter
    |
    v
Material asset
```

The asset may contain:

```text
MaterialGraph or normalized Material IR
parameter schema
static variant configuration
texture/resource references
source provenance
compiler/version metadata
```

For shipping builds, a cooked form may instead contain precompiled/cached variants.

Exact serialization belongs to a later asset-format design.

---

## 28. Error Model

Material errors should be structured.

Possible families:

```rust
#[non_exhaustive]
pub enum MaterialErrorKind {
    InvalidGraph,
    TypeMismatch,
    Cycle,
    MissingInput,
    InvalidDomainOperation,
    UnsupportedNode,
    UnsupportedFeature,
    InvalidParameter,
    CompileFailure,
}
```

Diagnostics should carry enough source information for Blender to highlight the offending node/socket.

Conceptually:

```rust
pub struct MaterialDiagnostic {
    pub kind: MaterialErrorKind,
    pub node: Option<MaterialNodeId>,
    pub port: Option<MaterialPortId>,
    pub message: String,
    pub source: Option<MaterialSourceRef>,
}
```

---

## 29. Source Provenance

Tool integrations need source mapping.

Conceptually:

```rust
pub enum MaterialSourceRef {
    Blender {
        node_tree: String,
        node: String,
        socket: Option<String>,
    },

    Generated {
        label: String,
    },
}
```

The exact representation should not leak Blender into the core semantic model.

A generic provenance payload or adapter-owned mapping may be preferable in the final implementation.

---

## 30. Caching Identity

Do not use one material hash for every purpose.

At minimum, distinguish:

```text
MaterialGraph identity
Material IR identity
Material static variant identity
Shader composition identity
ShaderArtifact identity
ShaderInterface compatibility
pipeline identity
```

Dynamic instance parameter changes must not invalidate shader compilation identity.

---

## 31. Material Instance

A material instance is runtime parameter data over a compiled material family.

Conceptually:

```rust
pub struct MaterialInstance {
    material: MaterialAssetId,
    parameters: MaterialParameterValues,
}
```

It does not contain:

```text
MaterialGraph editor nodes
RHI descriptor indices
native texture handles
compiled RenderGraph passes
```

The renderer resolves instance data against the compiled parameter schema.

---

## 32. MaterialGraph API Summary

The initial Rust authoring API should approximately expose:

```rust
pub struct MaterialGraph;

impl MaterialGraph {
    pub fn new(domain: MaterialDomain) -> Self;

    pub fn float_constant(&mut self, value: f32) -> Result<Value<Float>, MaterialGraphError>;

    pub fn color3_constant(
        &mut self,
        value: [f32; 3],
    ) -> Result<Value<Color3>, MaterialGraphError>;

    pub fn float_parameter(
        &mut self,
        name: impl Into<String>,
        default: f32,
    ) -> Result<Value<Float>, MaterialGraphError>;

    pub fn color3_parameter(
        &mut self,
        name: impl Into<String>,
        default: [f32; 3],
    ) -> Result<Value<Color3>, MaterialGraphError>;

    pub fn texture2d_parameter(
        &mut self,
        name: impl Into<String>,
    ) -> Result<Value<Texture2D>, MaterialGraphError>;

    pub fn tex_coord(
        &mut self,
        channel: u8,
    ) -> Result<Value<Vec2>, MaterialGraphError>;

    pub fn sample_texture2d(
        &mut self,
        texture: Value<Texture2D>,
        uv: Value<Vec2>,
    ) -> Result<Value<Color4>, MaterialGraphError>;

    pub fn add<T>(
        &mut self,
        lhs: Value<T>,
        rhs: Value<T>,
    ) -> Result<Value<T>, MaterialGraphError>
    where
        T: AddMaterialValue;

    pub fn mul<A, B>(
        &mut self,
        lhs: Value<A>,
        rhs: Value<B>,
    ) -> Result<Value<<A as MulMaterialValue<B>>::Output>, MaterialGraphError>
    where
        A: MulMaterialValue<B>;

    pub fn lerp<T>(
        &mut self,
        a: Value<T>,
        b: Value<T>,
        t: Value<Float>,
    ) -> Result<Value<T>, MaterialGraphError>
    where
        T: LerpMaterialValue;

    pub fn set_surface_output(
        &mut self,
        output: SurfaceOutput,
    ) -> Result<(), MaterialGraphError>;

    pub fn validate(
        &self,
    ) -> Result<(), MaterialGraphError>;

    pub fn lower(
        &self,
        target: &MaterialLoweringTarget,
    ) -> Result<MaterialModule, MaterialCompileError>;
}
```

This is a design target, not frozen ABI.

---

## 33. What Not to Freeze Yet

Do not freeze yet:

```text
hundreds of node kinds
full Blender compatibility
full Principled BSDF
Subsurface
Hair
Volume
Coat
Sheen
Anisotropy
procedural noise library
MaterialX compatibility
shader source language
shader cache file format
GPU binding ABI
material asset binary format
```

Those require real implementation evidence.

---

## 34. Initial Vertical Slice

The first useful end-to-end proof should be intentionally small.

### Supported Blender nodes

```text
Material Output
Principled BSDF:
    Base Color
    Metallic
    Roughness
    Normal
    Emission
    Alpha

Image Texture
Texture Coordinate
Normal Map
Value
RGB
Multiply
Add
Mix
```

### Fluxel semantic output

```text
SurfaceOutput
```

### Proof

The same material is:

```text
edited in Blender
translated to MaterialGraph
lowered to Material IR
compiled by Fluxel
previewed by Fluxel inside Blender
exported
rendered by Fluxel runtime
```

and both Fluxel outputs match within the declared rendering tolerance.

---

## 35. Recommended Implementation Order

```text
Step 1
    MaterialDomain
    MaterialValueType
    typed Value<T>

Step 2
    SurfaceOutput
    parameter schema

Step 3
    minimal MaterialGraph
    typed arithmetic / texture nodes

Step 4
    Material IR
    graph lowering
    validation / diagnostics

Step 5
    shader-composition boundary

Step 6
    one fixed Fluxel surface shader consumer

Step 7
    Blender translator for the supported subset

Step 8
    Fluxel Blender viewport integration

Step 9
    material asset export/import

Step 10
    grow the supported Blender subset only from real needs
```

---

## 36. Final Architecture

```text
Blender / procedural frontend
        |
        v
MaterialGraph
        |
        v
Material IR
        |
        v
Shader Composition / Compiler
        |
        v
CompiledMaterialVariant
        |
        v
FramePipeline
        |
        v
RenderGraph
        |
        v
RHI
```

The defining rules are:

> Blender is the primary material authoring UI.

> Fluxel owns the material semantics.

> MaterialGraph is an authoring frontend, not a renderer.

> Material IR is the stable compiler-facing semantic representation.

> `fluxel-renderer` consumes compiled material variants but does not own or interpret the material graph.

> Blender preview and game runtime use the same Fluxel material/shader/rendering path.
