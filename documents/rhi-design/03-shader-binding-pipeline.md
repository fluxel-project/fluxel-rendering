# RHI API freeze v13. Shader, binding, and pipeline

> Normative module of [Fluxel RHI API freeze v13](../design-rhi.md). Read the root
> specification and this module in full before implementation. No other
> document may redefine the interfaces in this module.

# 19. Shader model

This chapter is **FROZEN / P0**.

The RHI Shader contract has three layers:

```text
ShaderCode
    = a code form consumable by the current target Device/backend

ShaderInterface
    = the portable semantics of this entry point's resources and stage IO

ShaderProvenance
    = whether Capture/Replay/toolchain can regenerate code for other backends
```

These three layers must no longer be conflated into one “portable shader blob”.

---

## 19.1 Shader stage / stage mask

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
    Task,
    Mesh,
    RayGeneration,
    Miss,
    ClosestHit,
    AnyHit,
    Intersection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderStages(u16);

impl ShaderStages {
    pub const VERTEX: Self = Self(1 << 0);
    pub const FRAGMENT: Self = Self(1 << 1);
    pub const COMPUTE: Self = Self(1 << 2);
    pub const TASK: Self = Self(1 << 3);
    pub const MESH: Self = Self(1 << 4);
    pub const RAY_GENERATION: Self = Self(1 << 5);
    pub const MISS: Self = Self(1 << 6);
    pub const CLOSEST_HIT: Self = Self(1 << 7);
    pub const ANY_HIT: Self = Self(1 << 8);
    pub const INTERSECTION: Self = Self(1 << 9);

    pub fn contains(self, other: Self) -> bool;
    pub fn union(self, other: Self) -> Self;
    pub fn is_empty(self) -> bool;
}
```

Vertex + Fragment are Base graphics stages.

Compute is legal only when:

```text
OptionalFeature::Compute enabled
```

---

## 19.2 Backend-consumable ShaderCode

The RHI does not provide a shader cross-compiler.

It accepts only a code/source form that the current Device backend can consume:

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlslProfile {
    Core,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum ShaderCode {
    /// Canonical source form for the WebGPU backend.
    Wgsl(std::sync::Arc<str>),

    /// Canonical binary/module form for the Vulkan backend.
    SpirV(std::sync::Arc<[u32]>),

    /// Canonical compiled form for the DX12 backend.
    Dxil(std::sync::Arc<[u8]>),

    /// Source form that the Metal backend may compile at runtime.
    Msl(std::sync::Arc<str>),

    /// Compiled library/function provenance for the Metal backend.
    Metallib(std::sync::Arc<[u8]>),

    /// Desktop OpenGL source.
    Glsl {
        version: u16,
        profile: GlslProfile,
        source: std::sync::Arc<str>,
    },

    /// OpenGL ES / WebGL2 source.
    GlslEs {
        version: u16,
        source: std::sync::Arc<str>,
    },
}
```

There is no:

```text
ShaderCode::Portable
```

because:

```text
SPIR-V may be toolchain-portable provenance
    !=
a WebGPU browser can directly execute SPIR-V
```

Whether the current Device accepts it is determined by:

```rust
device
    .capabilities()
    .shader_acceptance(&artifact)
```

### Trusted native passthrough

Accepting a native form is not permission to trust arbitrary caller-provided
bytecode or reflection. Ordinary `ShaderArtifact::new` always follows normal
artifact/interface validation. The only bypass of frontend reflection is the
explicit unsafe admission boundary:

```rust
let artifact = unsafe {
    artifact.assume_trusted_passthrough(
        PassthroughShaderProvenance::new(
            "producer identity",
            "verification of this exact immutable code payload",
        ),
    )
};
```

This additionally requires `OptionalFeature::PassthroughShaders`; code-form
acceptance alone is insufficient. Safety requires the caller to establish that
the exact code payload, stage, entry point, `ShaderInterface`, and
`ShaderRequirements` agree. The RHI still validates the supplied declaration and
device capability, but cannot discover semantics omitted by dishonest reflection.
Provenance is retained for capture and diagnostics, not treated as authority.

---

## 19.3 Fluxel Shader ABI

The logical:

```text
group
slot
vertex location
fragment location
```

is **not equal to**:

```text
Vulkan DescriptorSet / Binding
HLSL register / register space
Metal buffer/texture/sampler index
GL binding point
```

The Shader toolchain may lower the Fluxel logical interface to each backend's own binding ABI.

Therefore, an artifact must identify its lowering ABI:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderAbiVersion {
    pub major: u16,
    pub minor: u16,
}
```

When the ABI version changes, an old executable must not be silently interpreted using new rules.

`ArtifactAcceptance` must be able to report:

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactAcceptance {
    Accepted,
    UnsupportedCodeFormat,
    UnsupportedAbi,
    MissingFeature,
    LimitExceeded,
    InterfaceUnsupported,
}
```

Specifically:

```text
logical group/slot -> native register/index
location -> native semantic
argument-buffer/root-signature strategy
```

remain backend/toolchain-private and do not enter the portable API.

---

## 19.4 Shader IO types

P0 freezes only 32-bit numeric stage IO.

In the future, f16 / packed inter-stage IO may be added as capability-gated enum variants.

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderNumericType {
    Float32,
    Sint32,
    Uint32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderLocation(u32);

impl ShaderLocation {
    pub fn new(value: u32) -> Self;
    pub fn get(self) -> u32;
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationMode {
    Perspective,
    Linear,
    Flat,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationSampling {
    Center,
    Centroid,
    Sample,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderInterpolation {
    pub mode: InterpolationMode,
    pub sampling: InterpolationSampling,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderLocationInterface {
    pub location: ShaderLocation,
    pub numeric_type: ShaderNumericType,

    /// 1..=4
    pub components: u8,

    /// Usually None for vertex inputs / fragment outputs.
    ///
    /// Vertex outputs / fragment inputs must be canonicalized by the artifact producer
    /// to explicit interpolation.
    pub interpolation: Option<ShaderInterpolation>,
}
```

---

## 19.5 Shader resource requirement

An entry point's resource requirements directly reuse the RHI Binding vocabulary.

Do not maintain a second:

```text
ShaderBindingKind
```

to avoid shader reflection and BindGroupLayout types slowly diverging into two systems.

```rust
#[derive(Clone, Debug)]
pub struct ShaderResourceRequirement {
    pub group: BindGroupIndex,
    pub slot: BindingSlotId,

    /// The resource semantics actually required by the Shader.
    pub kind: BindingKind,

    /// The fixed resource count of this logical binding in Shader code.
    pub count: BindingCount,
}
```

`dynamic_offset` **is not a shader semantic**, so it does not appear here.

Whether a dynamic offset is used is determined by BindGroupLayout.

---

## 19.6 Entry-point ShaderInterface

```rust
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct ShaderInterface {
    resources: Vec<ShaderResourceRequirement>,
    inputs: Vec<ShaderLocationInterface>,
    outputs: Vec<ShaderLocationInterface>,

    /// Must be true for a Vertex entry point.
    writes_position: bool,

    /// Optional for a Fragment entry point.
    writes_frag_depth: bool,
    writes_sample_mask: bool,
}

impl ShaderInterface {
    pub fn new() -> Self;

    pub fn with_resource(
        mut self,
        requirement: ShaderResourceRequirement,
    ) -> Self;

    pub fn with_input(
        mut self,
        input: ShaderLocationInterface,
    ) -> Self;

    pub fn with_output(
        mut self,
        output: ShaderLocationInterface,
    ) -> Self;

    pub fn with_writes_position(
        mut self,
        value: bool,
    ) -> Self;

    pub fn with_writes_frag_depth(
        mut self,
        value: bool,
    ) -> Self;

    pub fn with_writes_sample_mask(
        mut self,
        value: bool,
    ) -> Self;

    pub fn resources(&self) -> &[ShaderResourceRequirement];
    pub fn inputs(&self) -> &[ShaderLocationInterface];
    pub fn outputs(&self) -> &[ShaderLocationInterface];

    pub fn writes_position(&self) -> bool;
    pub fn writes_frag_depth(&self) -> bool;
    pub fn writes_sample_mask(&self) -> bool;
}
```

### Canonical interface representation

`ShaderInterface` is a canonical, duplicate-free semantic description. Its
builder methods may collect entries, but `create_shader()` must reject an
artifact unless all of the following are true:

```text
resources:
    (group, slot) is unique
    ordered lexicographically by (group, slot)

inputs:
    location is unique
    ordered by ascending location

outputs:
    location is unique
    ordered by ascending location
```

An input location and an output location are separate namespaces and may have
the same numeric value. Duplicate entries, or entries in a non-canonical order,
are `InvalidUsage`; the RHI must not silently sort, merge, or choose one.

Stage-specific validation:

All `components` must be `1..=4`; `Sint32/Uint32` inter-stage IO must use `Flat` interpolation.

```text
Vertex:
    writes_position = true
    inputs  = vertex attributes
    outputs = inter-stage locations

Fragment:
    inputs  = inter-stage locations
    outputs = color target locations

Compute:
    inputs / outputs must be empty
    writes_* must be false
```

Built-ins:

```text
vertex_index
instance_index
front_facing
sample_index
global_invocation_id
...
```

do not occupy a ShaderLocation and are not required to enter the pipeline layout.

If a future built-in changes the pipeline contract, add metadata for it separately.

---

## 19.7 ShaderRequirements

```rust
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct ShaderRequirements {
    required_features: Vec<OptionalFeature>,
    limit_requirements: Vec<LimitRequirement>,
}

impl ShaderRequirements {
    pub fn new() -> Self;

    pub fn require_feature(
        mut self,
        feature: OptionalFeature,
    ) -> Self;

    pub fn require_limit(
        mut self,
        requirement: LimitRequirement,
    ) -> Self;

    pub fn required_features(&self) -> &[OptionalFeature];
    pub fn limit_requirements(&self) -> &[LimitRequirement];
}
```

Do not repeat binding capability here.

It is validated by:

```text
ShaderInterface.resources
    ->
BindingSupportQuery
```

---

## 19.8 Artifact provenance / identity

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactHash(pub [u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactProducerVersion {
    pub major: u16,
    pub minor: u16,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum ShaderProvenance {
    /// The Toolchain can regenerate ShaderCode for other backends from this.
    PortableSource {
        language: PortableShaderLanguage,
        bytes: std::sync::Arc<[u8]>,

        /// The producer must canonicalize:
        /// - keys are unique;
        /// - sorted lexicographically by key;
        /// - contains no temporary absolute paths/process addresses.
        compiler_options: Vec<(String, String)>,
    },

    /// Contains only the current executable/code, with no cross-backend source/IR.
    ExecutableOnly,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortableShaderLanguage {
    Wgsl,
    SpirV,
    FluxelIr,
}
```

`ArtifactHash` is the content-address/provenance key supplied by the artifact
producer.

Freeze rule:

> The RHI may use it as a candidate cache key, but **correctness must not rely only on equal hashes**.

Object/interface compatibility must still use complete canonical semantic validation.

---

## 19.9 ShaderArtifact

```rust
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ShaderArtifact {
    pub label: Label,

    pub stage: ShaderStage,
    pub entry_point: String,

    pub code: ShaderCode,

    pub abi_version: ShaderAbiVersion,

    pub interface: ShaderInterface,
    pub requirements: ShaderRequirements,

    pub provenance: ShaderProvenance,

    pub content_hash: ArtifactHash,
    pub producer_version: ArtifactProducerVersion,
}

impl ShaderArtifact {
    pub fn new(
        stage: ShaderStage,
        entry_point: impl Into<String>,
        code: ShaderCode,
        abi_version: ShaderAbiVersion,
        interface: ShaderInterface,
        requirements: ShaderRequirements,
        content_hash: ArtifactHash,
        producer_version: ArtifactProducerVersion,
    ) -> Self;

    pub fn with_label(
        mut self,
        label: impl Into<String>,
    ) -> Self;

    pub fn with_provenance(
        mut self,
        provenance: ShaderProvenance,
    ) -> Self;
}
```

A P0 ShaderArtifact must be an artifact with **pipeline specialization closed**.

That is:

```text
WGSL required override without default
Vulkan unresolved specialization constant
Metal required function constant
```

Pipeline specialization is deliberately closed before `create_shader()` in v13:
the artifact is the exact compiled input to its pipeline. Fluxel does not expose
a second native-specialization map because it would be an additional public
shader ABI whose cross-backend validation has not been specified. This is a
semantic boundary, not a deferred or partially implemented capability: callers
that need specialization produce a distinct `ShaderArtifact` first, and cache
identity is consequently unambiguous.

---

## 19.10 ShaderModule

```rust
#[derive(Clone)]
pub struct ShaderModule {
    /* opaque, exactly one artifact entry point */
}

impl Device {
    pub async fn create_shader(
        &self,
        artifact: &ShaderArtifact,
    ) -> RhiResult<ShaderModule>;
}

impl ShaderModule {
    pub fn id(&self) -> ObjectId;
    pub fn device_identity(&self) -> DeviceIdentity;
    pub fn artifact(&self) -> &ShaderArtifact;
    pub fn stage(&self) -> ShaderStage;
}
```

`create_shader()` must first validate:

```text
ArtifactAcceptance
ShaderAbiVersion
stage
entry point
ShaderRequirements
ShaderInterface resource binding support
ShaderInterface stage IO shape
```

Only then may it enter backend shader/module creation.

Compilation errors from GLSL/MSL runtime compilers must be returned through:

```text
RhiError + DiagnosticEvent
```

and must not panic.

### Capture/Replay rule

The RHI does not cross-compile.

ReplayRuntime/toolchain uses:

```text
ShaderProvenance
+
target Device capability
```

to determine whether it can generate target ShaderCode.

---

# 20. Binding model

This chapter is now **FROZEN / P0**.

Core definition:

> `BindGroup` is an immutable resource packet validated against a logical layout.

It does not promise:

```text
VkDescriptorSet
D3D12 descriptor table
Metal argument buffer
GL texture-unit packet
```

These are only backend lowerings.

---

## 20.1 Group / slot identity

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindGroupIndex(u32);

impl BindGroupIndex {
    pub fn new(value: u32) -> Self;
    pub fn get(self) -> u32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindingSlotId(u32);

impl BindingSlotId {
    pub fn new(value: u32) -> Self;
    pub fn get(self) -> u32;
}
```

They are Fluxel logical indices.

They are not native registers/sets/indices.

---

## 20.2 BindingCount

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingCount {
    One,

    /// A fixed-length resource binding array.
    ///
    /// value >= 2.
    Fixed(u32),

    /// The binding packet supplies a non-zero active element count.
    RuntimeSized,
}

impl BindingCount {
    pub fn elements(self) -> u32;
}
```

Array shape is capability-gated vocabulary. `Fixed(n)` is exact-length;
`RuntimeSized` has a non-zero active packet length and uses the device's per-stage
binding-array ceiling as its maximum.

It **does not imply**:

```text
runtime-sized
partially-bound
update-after-bind
non-uniform arbitrary descriptor indexing
```

These are separate descriptor-indexing features. They must never be inferred from
the existence of a Rust binding-array type: a device reports
`RuntimeSizedBindingArrays`, `PartiallyBoundBindingArrays`, and the appropriate
non-uniform-indexing facts independently. The current binding packet models exact
fixed arrays and a contiguous active runtime array; a future sparse packet must be
a separately reviewed API rather than interpreting a missing element implicitly.

WebGPU core does not require a backend to support this vocabulary;
the WebGPU backend may correctly return `BindingSupport::Unsupported`.

---

## 20.3 Binding kinds

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureSampleType {
    Float,
    UnfilterableFloat,
    Sint,
    Uint,
    Depth,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerKind {
    Filtering,
    NonFiltering,
    Comparison,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferBindingAccess {
    ReadOnly,
    ReadWrite,
}

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindingKind {
    UniformBuffer {
        /// The minimum visible byte range required by the shader/layout.
        min_size: u64,
    },

    StorageBuffer {
        access: BufferBindingAccess,
        min_size: u64,
    },

    SampledTexture {
        dimension: TextureViewDimension,
        sample_type: TextureSampleType,
        multisampled: bool,
    },

    StorageTexture {
        dimension: TextureViewDimension,
        format: TextureFormat,
        access: StorageAccess,
    },

    Sampler {
        kind: SamplerKind,
    },
}
```

`min_size > 0`.

If a future requirement needs zero to mean a different contract, namely “determined by runtime binding size,” design it separately; do not use magic zero.

---

## 20.4 Binding capability

Binding support does not depend only on “the Device supports textures.”

For example:

```text
StorageTexture + Cube
StorageTexture + ReadWrite
StorageBuffer in Vertex stage
Fixed resource array
dynamic buffer offset
```

can each have independent limitations.

Therefore, add the formal capability query:

```rust
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BindingSupportQuery {
    pub visibility: ShaderStages,
    pub kind: BindingKind,
    pub count: BindingCount,

    /// Valid only for UniformBuffer / StorageBuffer.
    pub dynamic_offset: bool,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingSupport {
    Unsupported,
    Supported,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BindingLimitClass {
    UniformBuffers,
    StorageBuffers,
    SampledTextures,
    StorageTextures,
    Samplers,
}
```

The query entry points are frozen as:

```rust
AvailableCapabilities::binding_support(...)
EnabledCapabilities::binding_support(...)

AvailableCapabilities::binding_limit(stage, class)
EnabledCapabilities::binding_limit(stage, class)
```

This avoids recreating infinitely many booleans:

```text
supports_storage_texture_cube
supports_vertex_storage_buffer
supports_texture_binding_array
...
```

---

## 20.5 BindingSlot

```rust
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BindingSlot {
    pub slot: BindingSlotId,
    pub visibility: ShaderStages,
    pub kind: BindingKind,
    pub count: BindingCount,

    /// Valid only for buffer bindings.
    pub dynamic_offset: bool,
}

impl BindingSlot {
    pub fn new(
        slot: BindingSlotId,
        visibility: ShaderStages,
        kind: BindingKind,
    ) -> Self;

    pub fn with_count(
        mut self,
        count: BindingCount,
    ) -> Self;

    pub fn with_dynamic_offset(
        mut self,
        enabled: bool,
    ) -> Self;
}
```

Creating a BindGroupLayout must validate:

```text
visibility is non-empty
entry count <= MaxBindingsPerGroup
every slot < MaxBindingsPerGroup
slots are unique within the layout
BindingSupportQuery == Supported

dynamic_offset:
    allowed only for UniformBuffer / StorageBuffer

Fixed(n):
    n >= 2
    capability support is present
```

**Aggregate limits such as per-stage resource count and dynamic-buffers-per-pipeline-layout cannot be determined at an individual BindGroupLayout stage**,
because they span multiple groups; they are uniformly validated when `PipelineInterface` is created.

---

# 21. BindGroupLayout

## 21.1 Exact compatibility token

Old draft:

~~~rust
CompatibilityId(pub u128)
~~~

This could easily be misused as “equal 128-bit hash => equal correctness”.

Freeze v13 explicitly separates:

~~~rust
/// Device-scoped exact compatibility token.
///
/// Produced by Device interning canonical layout descriptors.
/// A caller cannot construct it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindGroupLayoutCompatibilityId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PipelineInterfaceCompatibilityId(u64);

/// Canonical descriptor fingerprint.
///
/// Used for cache / diagnostics / Capture provenance.
/// Equal fingerprints cannot alone replace correctness validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayoutFingerprint(pub [u8; 32]);
~~~

On the same Device:

~~~text
two canonical BindGroupLayoutDescriptors are completely identical
    ->
BindGroupLayoutCompatibilityId must be identical
~~~

Different Device objects must still first pass:

~~~text
DeviceIdentity validation
~~~

A compatibility token cannot make objects interoperable across Devices.

---

## 21.2 Descriptor

~~~rust
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BindGroupLayoutDescriptor {
    pub label: Label,
    pub entries: Vec<BindingSlot>,
}

impl BindGroupLayoutDescriptor {
    pub fn new(
        entries: Vec<BindingSlot>,
    ) -> Self;

    pub fn with_label(
        mut self,
        label: impl Into<String>,
    ) -> Self;
}
~~~

When Device creates a layout it must canonicalize:

~~~text
entries in ascending BindingSlotId order
~~~

and reject duplicate slots.

label does not participate in compatibility.

---

## 21.3 Object

~~~rust
#[derive(Clone)]
pub struct BindGroupLayout {
    /* opaque */
}

impl Device {
    pub fn create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDescriptor,
    ) -> RhiResult<BindGroupLayout>;
}

impl BindGroupLayout {
    pub fn id(&self) -> ObjectId;
    pub fn device_identity(&self) -> DeviceIdentity;

    /// Returns the canonicalized descriptor.
    pub fn descriptor(&self) -> &BindGroupLayoutDescriptor;

    pub fn compatibility_id(&self) -> BindGroupLayoutCompatibilityId;
    pub fn fingerprint(&self) -> LayoutFingerprint;

    /// Number of dynamic buffer elements in the current layout.
    pub fn dynamic_offset_count(&self) -> u32;
}
~~~

Dynamic-offset order is frozen as:

~~~text
ascending BindingSlotId
    ->
ascending element index within the same Fixed(n) binding
~~~

This order is shared by:

~~~text
RasterScope::set_bind_group
ComputeScope::set_bind_group
Capture tooling IR
backend lowering
~~~

Vulkan dynamic descriptor offsets are likewise uint32_t[] and consumed in binding/array-element order; WebGPU setBindGroup also provides Uint32Array form, so the portable type is frozen as u32.

---

# 22. BindGroup

## 22.1 BindingResource

~~~rust
#[non_exhaustive]
#[derive(Clone)]
pub enum BindingResource {
    Buffer(BufferBinding),
    Texture(TextureView),
    Sampler(Sampler),

    BufferArray(Vec<BufferBinding>),
    TextureArray(Vec<TextureView>),
    SamplerArray(Vec<Sampler>),
}

#[derive(Clone)]
pub struct BindGroupEntry {
    pub slot: BindingSlotId,
    pub resource: BindingResource,
}

impl BindGroupEntry {
    pub fn new(
        slot: BindingSlotId,
        resource: BindingResource,
    ) -> Self;
}
~~~

Rules:

~~~text
BindingCount::One
    -> scalar BindingResource

BindingCount::Fixed(n)
    -> corresponding Array variant
    -> array.len() == n
~~~

An Array of length 1 cannot stand in for One.

---

## 22.2 Descriptor / object

~~~rust
#[non_exhaustive]
#[derive(Clone)]
pub struct BindGroupDescriptor {
    pub label: Label,
    pub layout: BindGroupLayout,
    pub entries: Vec<BindGroupEntry>,
}

impl BindGroupDescriptor {
    pub fn new(
        layout: BindGroupLayout,
    ) -> Self;

    pub fn with_entry(
        mut self,
        entry: BindGroupEntry,
    ) -> Self;

    pub fn with_entries(
        mut self,
        entries: impl IntoIterator<Item = BindGroupEntry>,
    ) -> Self;

    pub fn with_label(
        mut self,
        label: impl Into<String>,
    ) -> Self;
}

#[derive(Clone)]
pub struct BindGroup {
    /* opaque immutable logical packet */
}

impl Device {
    pub fn create_bind_group(
        &self,
        desc: &BindGroupDescriptor,
    ) -> RhiResult<BindGroup>;
}

impl BindGroup {
    pub fn id(&self) -> ObjectId;
    pub fn device_identity(&self) -> DeviceIdentity;
    pub fn layout(&self) -> &BindGroupLayout;
    pub fn descriptor(&self) -> &BindGroupDescriptor;
}
~~~

When BindGroup is created, entries must be canonicalized by BindingSlotId; duplicate slot -> InvalidUsage.

A P0 BindGroup is immutable after creation.

---

## 22.3 Resource validation

BindGroup creation must validate every resource:

### Buffer

~~~text
DeviceIdentity
range bounds

UniformBuffer:
    BufferUsage::UNIFORM
    range.size >= min_size
    range.size <= MaxUniformBufferBindingSize
    base offset satisfies MinUniformBufferOffsetAlignment

StorageBuffer:
    BufferUsage::STORAGE
    range.size >= min_size
    range.size <= MaxStorageBufferBindingSize
    base offset satisfies MinStorageBufferOffsetAlignment
~~~

### Sampled Texture

~~~text
TextureUsage::SAMPLED
view dimension matches
Color sample type -> view aspects == COLOR
Depth sample type -> view aspects == DEPTH
P0 forbids sampled STENCIL
sample type matches FormatFacts
multisampled matches underlying texture sample_count
~~~

### Storage Texture

~~~text
TextureUsage::STORAGE
view aspects == COLOR
view dimension matches
view format == BindingKind.format
FormatFacts.storage_access supports required access
~~~

### Sampler

~~~text
SamplerKind is semantically compatible with Sampler descriptor
~~~

Here:

~~~text
Filtering sampler
    !=
every paired texture is necessarily filterable
~~~

Whether a sampler and sampled texture are legal as a **paired use** is finally checked by ShaderInterface + Pipeline validation.

---

## 22.4 Dynamic offset validation

At set_bind_group():

~~~text
dynamic_offsets.len() == layout.dynamic_offset_count()

each offset:
    u32
    validated against corresponding buffer-kind alignment

effective_offset =
    BufferBinding.range.offset
    + dynamic_offset

effective_offset + range.size
    <= Buffer.size
~~~

A dynamic offset does not modify the BindGroup object.

It is only part of current command binding state.

---

# 23. PipelineInterface

PipelineInterface is:

> the logical layout contract between shader entry points and BindGroup runtime packets.

It is not:

~~~text
VkPipelineLayout
D3D12 RootSignature
Metal argument-buffer layout
GL uniform-location table
~~~

---

## 23.1 Descriptor

The vector index of groups is BindGroupIndex.

If these are needed:

~~~text
group 0
group 2
~~~

an empty group 1 layout must be explicitly supplied.

This gives every backend stable, simple logical group numbering.

~~~rust
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PipelineInterfaceDescriptor {
    pub label: Label,
    pub groups: Vec<BindGroupLayout>,
}

impl PipelineInterfaceDescriptor {
    pub fn new(
        groups: Vec<BindGroupLayout>,
    ) -> Self;

    pub fn with_label(
        mut self,
        label: impl Into<String>,
    ) -> Self;
}
~~~

It must validate:

~~~text
groups.len() <= MaxBindGroups
all BindGroupLayout DeviceIdentity values are identical

for each ShaderStage + BindingLimitClass:
    aggregate all visible group entries
    count Fixed(n) as n binding elements
    <= binding_limit(stage, class)

aggregate dynamic uniform/storage buffer elements:
    <= MaxDynamicUniformBuffersPerPipelineLayout
    <= MaxDynamicStorageBuffersPerPipelineLayout
~~~

RasterPipeline creation additionally validates:

~~~text
if Device exposes MaxBindGroupsPlusVertexBuffers:
    interface.groups.len() + vertex_input.buffers.len() <= limit
~~~

---

## 23.2 Object / compatibility

~~~rust
#[derive(Clone)]
pub struct PipelineInterface {
    /* opaque */
}

impl Device {
    pub fn create_pipeline_interface(
        &self,
        desc: &PipelineInterfaceDescriptor,
    ) -> RhiResult<PipelineInterface>;
}

impl PipelineInterface {
    pub fn id(&self) -> ObjectId;
    pub fn device_identity(&self) -> DeviceIdentity;

    pub fn descriptor(&self) -> &PipelineInterfaceDescriptor;

    pub fn compatibility_id(&self) -> PipelineInterfaceCompatibilityId;
    pub fn fingerprint(&self) -> LayoutFingerprint;

    pub fn group(
        &self,
        index: BindGroupIndex,
    ) -> Option<&BindGroupLayout>;
}
~~~

PipelineInterface compatibility is obtained by canonicalizing/interning the complete ordered group-layout sequence.

Likewise:

~~~text
PipelineInterfaceCompatibilityId
    = exact same-Device token

LayoutFingerprint
    = cache/tooling hint
~~~

not the reverse.

---

## 23.3 Shader interface compatibility

When Raster/Compute pipelines are created, merge the participating stages':

~~~text
ShaderInterface.resources
~~~

For the same:

~~~text
(group, slot)
~~~

multiple stages must require:

~~~text
the same BindingKind
the same BindingCount
~~~

The corresponding BindingSlot in PipelineInterface must then satisfy:

~~~text
layout visibility
    ⊇ actually used stages

layout kind
    compatible with shader requirement

layout count
    == shader requirement count
~~~

For Buffer min_size:

~~~text
layout min_size >= shader required min_size
~~~

Storage-access merge is frozen as a small lattice:

~~~text
StorageBuffer:
    ReadOnly + ReadOnly   -> ReadOnly
    any ReadWrite         -> ReadWrite

StorageTexture:
    same access           -> that access
    mixed different access -> requires ReadWrite
    if BindingSupport does not support ReadWrite -> pipeline Unsupported
~~~

Other BindingKind/dimension/format/sample-type/count values must match exactly.

PipelineInterface may contain extra bindings unused by a shader.

This lets Renderer share one interface among multiple Pipelines.

---

## 23.4 Inline parameters

`PipelineInterface` declares portable `ImmediateData` byte ranges and their
stage visibility. Raster, compute, and ray scopes set a byte subrange by offset.
`Immediates`, `MaxImmediateSize`, and `ImmediateDataAlignment` are negotiated
capability/limit facts. The public semantic is not called push/root constants
and a backend without it returns `Unsupported`; it must not silently bind an
ordinary uniform buffer. See module 09.

---

# 24. Vertex input

Vertex input must be verifiable against ShaderInterface.inputs at pipeline creation.

---

## 24.1 VertexFormat

~~~rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexFormat {
    Uint8, Uint8x2, Uint8x4,
    Sint8, Sint8x2, Sint8x4,
    Unorm8, Unorm8x2, Unorm8x4, Unorm8x4Bgra,
    Snorm8, Snorm8x2, Snorm8x4,
    Uint16, Uint16x2, Uint16x4,
    Sint16, Sint16x2, Sint16x4,
    Unorm16, Unorm16x2, Unorm16x4,
    Snorm16, Snorm16x2, Snorm16x4,
    Float16, Float16x2, Float16x4,
    Float32, Float32x2, Float32x3, Float32x4,
    Uint32, Uint32x2, Uint32x3, Uint32x4,
    Sint32, Sint32x2, Sint32x3, Sint32x4,
    Float64, Float64x2, Float64x3, Float64x4,
    Unorm10_10_10_2,
}

impl VertexFormat {
    pub fn byte_size(self) -> u32;

    /// Portable numeric type sent to shader location after vertex fetch.
    pub fn shader_numeric_type(self) -> ShaderNumericType;

    pub fn components(self) -> u8;
}
~~~

`Float64*` requires `VertexAttribute64Bit`; every variant has exact byte size,
numeric type, component count, capture representation and native conversion.

---

## 24.2 Layout

~~~rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexStepMode {
    Vertex,
    Instance,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct VertexAttribute {
    pub location: ShaderLocation,
    pub format: VertexFormat,
    pub offset: u64,
}

impl VertexAttribute {
    pub fn new(
        location: ShaderLocation,
        format: VertexFormat,
        offset: u64,
    ) -> Self;
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct VertexBufferLayout {
    pub stride: u64,
    pub step_mode: VertexStepMode,
    pub attributes: Vec<VertexAttribute>,
}

impl VertexBufferLayout {
    pub fn new(
        stride: u64,
        step_mode: VertexStepMode,
    ) -> Self;

    pub fn with_attribute(
        mut self,
        attribute: VertexAttribute,
    ) -> Self;
}

#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct VertexInputState {
    pub buffers: Vec<VertexBufferLayout>,
}

impl VertexInputState {
    pub fn new() -> Self;

    pub fn with_buffer(
        mut self,
        layout: VertexBufferLayout,
    ) -> Self;
}
~~~

Validation:

~~~text
buffer count <= MaxVertexBuffers
attribute count <= MaxVertexAttributes
stride <= MaxVertexBufferArrayStride

ShaderLocation unique
attribute.offset + VertexFormat.byte_size <= stride
~~~

And:

~~~text
each Vertex Shader location input
    must have a matching attribute

numeric_type
components
    must be compatible
~~~

VertexInputState may contain extra attributes not consumed by the shader, provided the backend contract accepts them.

---

# 25. Raster fixed state

This chapter owns portable raster state. Polygon line/point modes, depth clip
control, conservative rasterization, depth-bias clamp, dual-source blend,
independent blend, and multisampled shading are present as independently
capability-gated semantics. Depth bounds, programmable sample positions and
VRS require an equally complete admission package before they are introduced;
they are not represented as placeholder fields.

---

## 25.1 Primitive

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontFace {
    Ccw,
    Cw,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CullMode {
    None,
    Front,
    Back,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthBiasState {
    pub constant: i32,
    pub slope_scale: f32,
}

impl DepthBiasState {
    pub fn new(constant: i32, slope_scale: f32) -> Self;
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PrimitiveState {
    pub topology: PrimitiveTopology,
    pub front_face: FrontFace,
    pub cull_mode: CullMode,

    /// Constant, slope, and capability-gated clamp semantics are validated
    /// against the selected raster route.
    pub depth_bias: Option<DepthBiasState>,

    /// Legal only for LineStrip / TriangleStrip.
    ///
    /// If Some, the IndexFormat of indexed strip draws must match,
    /// and the corresponding fixed primitive-restart value is enabled.
    pub strip_index_format: Option<IndexFormat>,
}

impl PrimitiveState {
    pub fn new(
        topology: PrimitiveTopology,
    ) -> Self;

    pub fn with_front_face(
        mut self,
        front_face: FrontFace,
    ) -> Self;

    pub fn with_cull_mode(
        mut self,
        cull_mode: CullMode,
    ) -> Self;

    pub fn with_depth_bias(
        mut self,
        bias: DepthBiasState,
    ) -> Self;

    pub fn with_strip_index_format(
        mut self,
        format: IndexFormat,
    ) -> Self;
}
```

WebGPU itself requires the indexed draw of strip topology to determine the strip index format in the pipeline;
D3D12 PSO also has strip-cut value, so it can't be left to be guessed at draw.

Depth bias validates topology and the selected route; `slope_scale` must be
finite. Clamp and line/point variants are capability-gated rather than silently
omitted; see module 09.

---

## 25.2 Blend

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    One,

    Src,
    OneMinusSrc,
    SrcAlpha,
    OneMinusSrcAlpha,

    Dst,
    OneMinusDst,
    DstAlpha,
    OneMinusDstAlpha,

    SrcAlphaSaturated,

    /// Uses the current dynamic blend constant.
    Constant,
    OneMinusConstant,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendOperation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlendComponent {
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
    pub operation: BlendOperation,
}

impl BlendComponent {
    pub fn new(
        src_factor: BlendFactor,
        dst_factor: BlendFactor,
        operation: BlendOperation,
    ) -> Self;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlendState {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}

impl BlendState {
    pub fn new(
        color: BlendComponent,
        alpha: BlendComponent,
    ) -> Self;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorWriteMask(u8);

impl ColorWriteMask {
    pub const NONE: Self = Self(0);
    pub const RED: Self = Self(1 << 0);
    pub const GREEN: Self = Self(1 << 1);
    pub const BLUE: Self = Self(1 << 2);
    pub const ALPHA: Self = Self(1 << 3);
    pub const ALL: Self = Self(0x0f);

    pub fn contains(self, other: Self) -> bool;
    pub fn union(self, other: Self) -> Self;
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ColorTargetState {
    pub format: TextureFormat,
    pub blend: Option<BlendState>,
    pub write_mask: ColorWriteMask,
}

impl ColorTargetState {
    pub fn new(
        format: TextureFormat,
    ) -> Self;

    pub fn with_blend(
        mut self,
        blend: BlendState,
    ) -> Self;

    pub fn with_write_mask(
        mut self,
        mask: ColorWriteMask,
    ) -> Self;
}
```

if:

```text
blend != None
```

but:

```text
FormatFacts.color_attachment = true
FormatFacts.blendable = true
```

must be established.

Blend validation：

```text
operation == Min / Max
    -> src_factor == One
    -> dst_factor == One
```

`Constant / OneMinusConstant` corresponds to the Recording layer:

```text
set_blend_constant()
```

No more dead interfaces like "the API has a blend constant command, but no blend factor can use it".

---

## 25.3 Depth / stencil

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StencilOperation {
    Keep,
    Zero,
    Replace,
    Invert,
    IncrementClamp,
    DecrementClamp,
    IncrementWrap,
    DecrementWrap,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilFaceState {
    pub compare: CompareFunction,
    pub fail_op: StencilOperation,
    pub depth_fail_op: StencilOperation,
    pub pass_op: StencilOperation,
}

impl StencilFaceState {
    pub fn new(
        compare: CompareFunction,
    ) -> Self;
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilState {
    pub front: StencilFaceState,
    pub back: StencilFaceState,
    pub read_mask: u32,
    pub write_mask: u32,
}

impl StencilState {
    pub fn new(
        front: StencilFaceState,
        back: StencilFaceState,
    ) -> Self;
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthState {
    pub write_enabled: bool,
    pub compare: CompareFunction,
}

impl DepthState {
    pub fn new(
        compare: CompareFunction,
    ) -> Self;

    pub fn with_write_enabled(
        mut self,
        enabled: bool,
    ) -> Self;
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DepthStencilState {
    pub format: TextureFormat,
    pub depth: Option<DepthState>,
    pub stencil: Option<StencilState>,
}

impl DepthStencilState {
    pub fn new(
        format: TextureFormat,
    ) -> Self;

    pub fn with_depth(
        mut self,
        depth: DepthState,
    ) -> Self;

    pub fn with_stencil(
        mut self,
        stencil: StencilState,
    ) -> Self;
}
```

Must be consistent with `FormatFacts.aspects()`.

For example:

```text
Depth32Float
    -> stencil must be None

Depth24PlusStencil8
    -> depth / stencil can be enabled/disabled independently
```

---

## 25.4 Multisample

```rust
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultisampleState {
    pub count: u32,

    /// The portable sample mask aligns with the core mask width of Vulkan/WebGPU/D3D12.
    pub mask: u32,

    pub alpha_to_coverage_enabled: bool,
}

impl MultisampleState {
    pub fn new(
        count: u32,
    ) -> Self;

    pub fn with_mask(
        mut self,
        mask: u32,
    ) -> Self;

    pub fn with_alpha_to_coverage(
        mut self,
        enabled: bool,
    ) -> Self;
}
```

`count` must be consistent with all active render target attachment sample counts.

`alpha_to_coverage_enabled` is only valid when `count > 1`.

---

# 26. Pipeline target signature

Multiview is no longer half-frozen.

Color locations allow holes, so:

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderTargetSignature {
    /// Vector index = fragment output / color attachment location.
    ///
    /// None = this location has no color target.
    pub color_formats: Vec<Option<TextureFormat>>,

    pub depth_stencil_format: Option<TextureFormat>,

    pub sample_count: u32,
}
```

This is consistent with the WebGPU fragment target sequence model that allows `null/undefined` slots,
At the same time, it can also correctly express the sparse MRT location of Vulkan/D3D12/Metal.

Trailing `None` must canonicalize/remove:

```text
[RGBA8, None, None]
    =>
[RGBA8]
```

Avoid multiple representations of the same signature.

---

# 27. RasterPipeline

## 27.1 Descriptor

```rust
#[non_exhaustive]
#[derive(Clone)]
pub struct RasterPipelineDescriptor {
    pub label: Label,

    pub vertex: ShaderModule,
    pub fragment: Option<ShaderModule>,

    pub interface: PipelineInterface,

    pub vertex_input: VertexInputState,
    pub primitive: PrimitiveState,

    pub depth_stencil: Option<DepthStencilState>,
    pub multisample: MultisampleState,

    /// Vector index = color output location.
    pub color_targets: Vec<Option<ColorTargetState>>,
}

impl RasterPipelineDescriptor {
    /// Creates a minimal graphics descriptor.
    ///
    /// defaults:
    /// - no fragment stage
    /// - empty vertex input
    /// - TriangleList
    /// - no depth/stencil
    /// - sample count 1
    /// - no color targets
    pub fn new(
        vertex: ShaderModule,
        interface: PipelineInterface,
    ) -> Self;

    pub fn with_label(
        mut self,
        label: impl Into<String>,
    ) -> Self;

    pub fn with_fragment(
        mut self,
        fragment: ShaderModule,
    ) -> Self;

    pub fn with_vertex_input(
        mut self,
        state: VertexInputState,
    ) -> Self;

    pub fn with_primitive(
        mut self,
        state: PrimitiveState,
    ) -> Self;

    pub fn with_depth_stencil(
        mut self,
        state: DepthStencilState,
    ) -> Self;

    pub fn with_multisample(
        mut self,
        state: MultisampleState,
    ) -> Self;

    /// Automatically extends the vector; intermediate locations are filled with None.
    pub fn with_color_target(
        mut self,
        location: ShaderLocation,
        target: ColorTargetState,
    ) -> Self;

    pub fn target_signature(&self) -> RenderTargetSignature;
}
```

---

## 27.2 Pipeline object

```rust
#[derive(Clone)]
pub struct RasterPipeline {
    /* opaque */
}

impl Device {
    pub async fn create_raster_pipeline(
        &self,
        desc: &RasterPipelineDescriptor,
    ) -> RhiResult<RasterPipeline>;
}

impl RasterPipeline {
    pub fn id(&self) -> ObjectId;
    pub fn device_identity(&self) -> DeviceIdentity;

    pub fn descriptor(&self) -> &RasterPipelineDescriptor;
    pub fn interface(&self) -> &PipelineInterface;
    pub fn target_signature(&self) -> &RenderTargetSignature;
}
```

---

## 27.3 Raster pipeline validation

Complete at least: before entering backend:

### Device / stage

```text
vertex / fragment / PipelineInterface
    DeviceIdentity matches exactly

vertex.stage == Vertex

fragment:
    None
    or stage == Fragment
```

### Shader resources

Press Chapter 23:

```text
merge resource requirements
validate PipelineInterface
validate binding visibility/kind/count/min_size
```

### Vertex input

```text
Vertex Shader inputs
    are covered by VertexInputState

numeric_type / components compatible

location unique
limits/alignment are legal
```

### Vertex -> Fragment inter-stage linkage

If there is a Fragment:

```text
each Fragment input location
    must be provided by a Vertex output

numeric_type
components
interpolation
    must be compatible
```

Additional Vertex output not consumed by Fragment is legal.

### Fragment outputs

Fragment shaders can output pipeline unbound locations; this output is ignored.

For each active `color_targets[location] = Some(target)`:

```text
if Fragment has an output at that location:
    ShaderNumericType must match FormatFacts.color_output_type()

if Fragment has no output at that location:
    target.write_mask must == ColorWriteMask::NONE
```

If there is no fragment:

```text
all color_targets must be None
```

If `alpha_to_coverage_enabled == true`:

```text
fragment must be present
location 0 target must be Some
fragment must output a Float32 vec4 at location 0
current target format must have_alpha_channel()
writes_sample_mask must be false
```

These constraints ensure common executable semantics for WebGPU/Metal and other backends.

### Fragment depth

if:

```text
fragment.interface.writes_frag_depth == true
```

then there must be:

```text
depth_stencil != None
and the format contains the Depth aspect
```

### Target facts

Each active color target:

```text
FormatFacts.color_attachment == true

blend != None
    -> FormatFacts.blendable == true

TextureSupport:
    D2
    COLOR_ATTACHMENT
    sample_count
    Supported
```

Depth/stencil is the same.

All active targets must use:

```text
the same MultisampleState.count
```

### Limits

verify:

```text
MaxColorAttachments
MaxColorAttachmentBytesPerSample (if exposed by Device)
MaxVertexBuffers
MaxVertexAttributes
MaxVertexBufferArrayStride
MaxInterStageShaderVariables
MaxBindGroupsPlusVertexBuffers (if present)
```

### Strip topology

```text
non-strip topology:
    strip_index_format must be None

LineStrip / TriangleStrip:
    strip_index_format may be None
    but before an Indexed Draw it must be present and match the IndexBuffer format
```

---

# 28. ComputePipeline

API shape frozen, but capability-gated.

```rust
#[non_exhaustive]
#[derive(Clone)]
pub struct ComputePipelineDescriptor {
    pub label: Label,
    pub shader: ShaderModule,
    pub interface: PipelineInterface,
}

impl ComputePipelineDescriptor {
    pub fn new(
        shader: ShaderModule,
        interface: PipelineInterface,
    ) -> Self;

    pub fn with_label(
        mut self,
        label: impl Into<String>,
    ) -> Self;
}

#[derive(Clone)]
pub struct ComputePipeline {
    /* opaque */
}

impl Device {
    pub async fn create_compute_pipeline(
        &self,
        desc: &ComputePipelineDescriptor,
    ) -> RhiResult<ComputePipeline>;
}

impl ComputePipeline {
    pub fn id(&self) -> ObjectId;
    pub fn device_identity(&self) -> DeviceIdentity;
    pub fn descriptor(&self) -> &ComputePipelineDescriptor;
    pub fn interface(&self) -> &PipelineInterface;
}
```

Must verify when creating:

```text
OptionalFeature::Compute enabled

shader / interface DeviceIdentity are the same
shader.stage == Compute

ShaderInterface:
    location inputs/outputs are empty
    resource requirements are compatible with PipelineInterface

ShaderRequirements:
    features / limits satisfied
```

does not exist:

```rust
trait ComputeApi;
```

---

## 28.1 Pipeline / shader cache rule

v13 freezes an opaque, device-scoped `PipelineCache`; it deliberately does not
freeze a backend pipeline-binary file format. All pipeline descriptors remain
completely describable by:

```text
ShaderArtifact
PipelineInterface canonical descriptors
fixed state
target signature
```

The cache descriptor optionally carries opaque serialized bytes together with a
`PipelineCacheValidationKey`. These two values are supplied together or not at
all. The key is persisted beside the blob and compared as an opaque backend /
adapter / device contract; applications do not interpret it. Cache creation has
an explicit invalid-data policy:

```text
RejectInvalidData
IgnoreInvalidData and create an empty cache
```

`PipelineCache` and restoration/serialization have distinct capability facts.
The object may be named by shader/raster/compute/mesh/ray pipeline descriptors,
and wrong-device use is rejected before backend creation. Serialization returns
opaque bytes plus the cache object's validation key. Device loss terminates all
later cache operations. You cannot use a native PSO/pipeline binary as a
portable correctness source.

### Backend cache implementation route

The public cache is a performance input only. A miss or an ignored invalid blob
falls back to ordinary asynchronous pipeline creation without changing shader,
layout, fixed-state, target-signature, or capability validation. Backends that
do not implement the object keep both capability facts false and return
structured `Unsupported` before native entry; no reachable path may terminate
in `todo!()` or `unimplemented!()`.

---
