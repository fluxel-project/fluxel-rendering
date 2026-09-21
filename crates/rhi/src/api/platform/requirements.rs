//! What a caller asks of a device before it exists (specification section 5.7).
//!
//! This module owns the *request* layer of the three the specification keeps
//! apart:
//!
//! ```text
//! AvailableOnAdapter   what an adapter can do          (capability module)
//!         |
//! Required / Preferred what a caller asks for          (here)
//!         |
//! EnabledOnDevice      what the device actually got    (capability module)
//! ```
//!
//! Collapsing any two of those is the mistake this module exists to prevent: a
//! requirement is not a fact, and satisfying a requirement is not the same as
//! enabling a feature.

use crate::api::binding::BindingSupportQuery;
use crate::api::format::TextureSupportQuery;
use crate::api::resource::{BufferSupportQuery, RouteQuery};

/// A device capability that is optional, and therefore must be asked for.
///
/// Section 0 requires that capability never be inferred from the presence of a
/// Rust trait, so a backend does not become "compute-capable" by implementing
/// something. It reports the fact, and the caller requests it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OptionalFeature {
    /// Compute pipeline and dispatch vocabulary.
    Compute,
    /// Anisotropic sampler filtering.
    ///
    /// Not inferable from a limit: WebGL2 needs an extension and Vulkan needs
    /// the corresponding feature, so `MaxSamplerAnisotropy > 1` does not imply
    /// that anisotropic filtering may be used.
    SamplerAnisotropy,
    /// Fixed-length arrays of buffers, textures, samplers, or acceleration
    /// structures. Runtime-sized and partially-bound forms have distinct feature
    /// variants because a fixed array alone does not imply descriptor indexing.
    BindingArrays,
    /// Samplers that compare a sampled depth value to a reference.
    ComparisonSamplers,
    /// `ClampToBorder` sampler addressing.
    SamplerClampToBorder,
    /// `ClampToBorder` with an all-zero integer border value.
    SamplerClampToZero,
    /// Non-fill polygon rasterization.  Backends report the line and point
    /// variants separately because their native support is independent.
    PolygonModeLine,
    /// Point polygon rasterization.
    PolygonModePoint,
    /// Disables depth clipping where the native API can express it.
    DepthClipControl,
    /// Conservative rasterization.
    ConservativeRasterization,
    /// A non-zero depth-bias clamp.
    DepthBiasClamp,
    /// Blend factors reading the second fragment output.
    DualSourceBlending,
    /// Per-target blend state for MRT pipelines.
    IndependentBlend,
    /// Per-sample fragment shading.
    MultisampledShading,
    /// A raster pipeline can narrow the default all-samples-enabled mask.
    MultisampleMask,
    /// 64-bit vertex attributes.
    VertexAttribute64Bit,
    /// Occlusion queries can be recorded and resolved.
    OcclusionQuery,
    /// Timestamp queries and timestamp-result resolution.
    TimestampQuery,
    /// Timestamp writes outside pass scopes.
    TimestampInsideEncoder,
    /// Timestamp writes in raster scopes.
    TimestampInsideRasterScope,
    /// Timestamp writes in compute scopes.
    TimestampInsideComputeScope,
    /// Pipeline-statistics queries.
    PipelineStatisticsQuery,
    /// Query result resolution into a buffer.
    QueryResolve,
    /// Direct and indexed indirect draws.
    IndirectDraw,
    /// Indirect compute dispatch.
    IndirectDispatch,
    /// More than one indirect draw in one command.
    MultiDrawIndirect,
    /// GPU indirect arguments may contain a non-zero `first_instance` field.
    ///
    /// The portable recorder cannot inspect GPU-provided argument bytes. A
    /// backend may advertise executable raster indirect draws only when its
    /// native route guarantees this semantic, rather than deferring failure to
    /// the driver.
    IndirectFirstInstance,
    /// A count buffer determines an indirect draw count.
    MultiDrawIndirectCount,
    /// Indexed draws accept a non-zero base vertex.
    BaseVertex,
    /// Direct draws accept a non-zero first instance.
    ///
    /// This is separate from [`Self::IndirectFirstInstance`]: direct command
    /// arguments are visible to the recorder and can fail closed before native
    /// work, while indirect argument bytes are GPU-owned.
    BaseInstance,
    /// Native clear-buffer lowering.
    ClearBuffer,
    /// Native clear-texture lowering.
    ClearTexture,
    /// A map usage may coexist with broader primary GPU usages on one buffer.
    ///
    /// This does not gate ordinary staging maps: `MAP_READ | COPY_DST` and
    /// `MAP_WRITE | COPY_SRC` are decided by the exact buffer-support answer
    /// and the mapping lease. This feature lets a requirement distinguish a
    /// backend that additionally admits combinations such as map-plus-vertex,
    /// uniform, or storage usage.
    MappablePrimaryBuffers,
    /// A mapped lease may remain open across submissions when the backend's
    /// memory model permits it.  Absence does not remove ordinary map/unmap.
    PersistentMapping,
    /// Host-visible mappings are coherent; explicit flush/invalidate is not
    /// required for visibility (though it remains a legal no-op).
    CoherentMapping,
    /// Immediate constant data declared by a pipeline interface.
    Immediates,
    /// Bindings whose element count is selected at runtime.
    RuntimeSizedBindingArrays,
    /// Binding arrays may leave elements unbound.
    PartiallyBoundBindingArrays,
    /// Non-uniform indexing of sampled textures and storage buffers.
    NonUniformSampledTextureAndStorageBufferIndexing,
    /// Non-uniform indexing of storage textures.
    NonUniformStorageTextureIndexing,
    /// External-video/image texture bindings.
    ExternalTexture,
    /// Multiview rasterization.
    Multiview,
    /// Selective multiview rasterization.
    SelectiveMultiview,
    /// Multisampled array textures.
    MultisampleArray,
    /// Task/mesh shader pipelines.
    MeshShader,
    /// Point primitive output from mesh shaders.
    MeshShaderPoints,
    /// Mesh shaders used with multiview.
    MeshShaderMultiview,
    /// Acceleration-structure bindings and ray queries.
    RayQuery,
    /// Ray-hit vertex return from ray queries.
    RayHitVertexReturn,
    /// Extended acceleration-structure vertex formats.
    ExtendedAccelerationStructureVertexFormats,
    /// In-place compatible acceleration-structure updates.
    AccelerationStructureUpdate,
    /// Acceleration-structure compact-size query and compaction copies.
    AccelerationStructureCompaction,
    /// Ray-tracing pipelines.
    RayTracingPipeline,
    /// Cooperative-matrix shader operations.
    CooperativeMatrix,
    /// Half precision floating-point shader operations.
    ShaderF16,
    /// Double precision floating-point shader operations.
    ShaderF64,
    /// Signed/unsigned 16-bit integer shader operations.
    ShaderI16,
    /// Signed/unsigned 64-bit integer shader operations.
    ShaderInt64,
    /// Float32 atomic shader operations.
    ShaderFloat32Atomic,
    /// Int64 atomic min/max shader operations.
    ShaderInt64AtomicMinMax,
    /// All supported int64 atomic shader operations.
    ShaderInt64AtomicAllOps,
    /// Texture atomic shader operations.
    TextureAtomic,
    /// Int64 texture atomic shader operations.
    TextureInt64Atomic,
    /// Explicit early depth testing in shaders.
    ShaderEarlyDepthTest,
    /// Subgroup operations.
    Subgroup,
    /// Subgroup operations in vertex-stage shaders.
    SubgroupVertex,
    /// Subgroup barrier operations.
    SubgroupBarrier,
    /// Fragment barycentric built-ins.
    ShaderBarycentrics,
    /// Per-vertex shader built-ins.
    ShaderPerVertex,
    /// Draw-index shader builtin.
    ShaderDrawIndex,
    /// Primitive-index shader builtin.
    PrimitiveIndex,
    /// Clip-distance shader outputs.
    ClipDistances,
    /// Coherent shader memory decoration.
    MemoryDecorationCoherent,
    /// Volatile shader memory decoration.
    MemoryDecorationVolatile,
    /// f16 values represented through an f32 interface.
    ShaderF16InF32,
    /// Caller supplied, ABI-checked reflection for trusted native code.
    PassthroughShaders,
    /// Native pipeline-cache object creation.
    PipelineCache,
    /// Pipeline-cache serialization and restoration.
    PipelineCacheSerialization,
    /// Integration with a native graphics debugger capture.
    NativeGraphicsCapture,
    /// Native allocator/memory diagnostics.
    AllocatorReport,
    /// Import of platform external-memory handles through the extension SPI.
    ExternalMemory,
    /// Copying an opaque external image source into a texture.
    ExternalImageCopy,
}

/// A portable device limit a caller may require.
///
/// The keys split into two directions, which is why
/// [`LimitRequirement`] has two variants rather than one:
///
/// ```text
/// MaxFoo                 a larger value is more capable   -> AtLeast
/// MinFooAlignment        a smaller value is more capable  -> AtMost
/// ```
///
/// Section 7.4 forbids collapsing those into one `minimum_limit()`-shaped verb,
/// because doing so silently inverts the meaning for the alignment keys.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LimitKey {
    /// Largest single buffer, in bytes.
    MaxBufferSize,
    /// Legacy required alignment for both the offset and size of mapped buffer
    /// ranges.
    ///
    /// New backends must report [`Self::MapOffsetAlignment`] and
    /// [`Self::MapSizeAlignment`] instead. This key remains as a compatibility
    /// fallback for already-established backends whose native contract uses one
    /// common alignment; it must never be used to approximate two different
    /// constraints.
    MapAlignment,
    /// Required alignment of a mapped buffer range's starting offset.
    ///
    /// This is deliberately distinct from [`Self::MapSizeAlignment`]. For
    /// example, WebGPU requires an 8-byte offset but only a 4-byte size.
    MapOffsetAlignment,
    /// Required alignment of a mapped buffer range's size.
    MapSizeAlignment,
    /// Largest 1D texture dimension.
    MaxTexture1dDimension,
    /// Largest 2D texture dimension.
    MaxTexture2dDimension,
    /// Largest 3D texture dimension.
    MaxTexture3dDimension,
    /// Largest number of texture array layers.
    MaxTextureArrayLayers,
    /// Largest number of bind groups in a pipeline layout.
    MaxBindGroups,
    /// Largest number of bindings in one bind group.
    MaxBindingsPerGroup,
    /// Largest combined number of bind groups and vertex buffers.
    ///
    /// Some backends constrain the sum rather than each part; a backend without
    /// that constraint reports no value for this key.
    MaxBindGroupsPlusVertexBuffers,
    /// Largest uniform buffer binding, in bytes.
    MaxUniformBufferBindingSize,
    /// Largest storage buffer binding, in bytes.
    MaxStorageBufferBindingSize,
    /// Largest number of dynamic uniform buffers in one pipeline layout.
    MaxDynamicUniformBuffersPerPipelineLayout,
    /// Largest number of dynamic storage buffers in one pipeline layout.
    MaxDynamicStorageBuffersPerPipelineLayout,
    /// Largest sampler anisotropy.
    ///
    /// Meaningful only when [`OptionalFeature::SamplerAnisotropy`] is enabled.
    MaxSamplerAnisotropy,
    /// Largest number of color attachments in one render pass.
    MaxColorAttachments,
    /// Largest number of color attachment bytes per sample.
    MaxColorAttachmentBytesPerSample,
    /// Largest number of vertex buffers.
    MaxVertexBuffers,
    /// Largest number of vertex attributes.
    MaxVertexAttributes,
    /// Largest vertex buffer array stride, in bytes.
    MaxVertexBufferArrayStride,
    /// Largest number of inter-stage shader variables.
    MaxInterStageShaderVariables,
    /// Largest number of compute invocations per workgroup.
    MaxComputeInvocationsPerWorkgroup,
    /// Largest compute workgroup size on X.
    MaxComputeWorkgroupSizeX,
    /// Largest compute workgroup size on Y.
    MaxComputeWorkgroupSizeY,
    /// Largest compute workgroup size on Z.
    MaxComputeWorkgroupSizeZ,
    /// Largest number of compute workgroups per dimension.
    MaxComputeWorkgroupsPerDimension,
    /// Largest compute workgroup storage, in bytes.
    MaxComputeWorkgroupStorageSize,
    /// Largest number of slots in one query set.
    ///
    /// A backend that enables any query-set feature must report this exact
    /// ceiling. Absence is not interpreted as an arbitrary implementation
    /// default because query allocation is observable resource creation.
    MaxQueriesPerQuerySet,
    /// Required alignment of a query-result resolve destination offset, in bytes.
    ///
    /// Meaningful only when [`OptionalFeature::QueryResolve`] is enabled. A
    /// backend must report a non-zero power of two; absence is fail-closed.
    QueryResolveBufferAlignment,
    /// Smallest uniform buffer offset alignment, in bytes.
    MinUniformBufferOffsetAlignment,
    /// Smallest storage buffer offset alignment, in bytes.
    MinStorageBufferOffsetAlignment,
    /// Largest number of binding-array elements visible to one shader stage.
    MaxBindingArrayElementsPerShaderStage,
    /// Largest number of acceleration structures in a binding array per stage.
    MaxBindingArrayAccelerationStructureElementsPerShaderStage,
    /// Largest number of samplers in a binding array per stage.
    MaxBindingArraySamplerElementsPerShaderStage,
    /// Largest number of non-sampler bindings.
    MaxNonSamplerBindings,
    /// Largest immediate-data payload in bytes.
    MaxImmediateSize,
    /// Required alignment of immediate-data offsets and sizes.
    ImmediateDataAlignment,
    /// Largest number of views addressed by a multiview mask.
    MaxMultiviewViewCount,
    /// Task workgroup total count.
    MaxTaskWorkgroupTotalCount,
    /// Task workgroups per dimension.
    MaxTaskWorkgroupsPerDimension,
    /// Mesh workgroup total count.
    MaxMeshWorkgroupTotalCount,
    /// Mesh workgroups per dimension.
    MaxMeshWorkgroupsPerDimension,
    /// Task invocations per workgroup.
    MaxTaskInvocationsPerWorkgroup,
    /// Task invocations per dimension.
    MaxTaskInvocationsPerDimension,
    /// Mesh invocations per workgroup.
    MaxMeshInvocationsPerWorkgroup,
    /// Mesh invocations per dimension.
    MaxMeshInvocationsPerDimension,
    /// Task payload size in bytes.
    MaxTaskPayloadSize,
    /// Mesh output vertices.
    MaxMeshOutputVertices,
    /// Mesh output primitives.
    MaxMeshOutputPrimitives,
    /// Mesh output layers.
    MaxMeshOutputLayers,
    /// Mesh multiview count.
    MaxMeshMultiviewViewCount,
    /// BLAS primitive count.
    MaxBlasPrimitiveCount,
    /// BLAS geometry count.
    MaxBlasGeometryCount,
    /// TLAS instance count.
    MaxTlasInstanceCount,
    /// Exact byte size of one backend-encoded raw TLAS instance record.
    ///
    /// This is a fact rather than a guessed ABI constant.  Code that uploads
    /// backend-native instance records must request it explicitly; ordinary
    /// portable [`crate::api::resource::TlasInstance`] construction never
    /// exposes that representation.
    RawTlasInstanceSize,
    /// Required alignment of acceleration-structure build scratch buffers.
    RayTracingScratchBufferAlignment,
    /// Acceleration structures visible to one shader stage.
    MaxAccelerationStructuresPerShaderStage,
    /// Combined buffers and acceleration structures visible to one shader stage.
    MaxBuffersAndAccelerationStructuresPerShaderStage,
    /// Ray dispatch count per dimension.
    MaxRayDispatchCount,
    /// Ray recursion depth.
    MaxRayRecursionDepth,
    /// Maximum bytes in one ray-tracing shader-table group record.
    MaxRayTracingPipelineGroupDataSize,
    /// Required alignment of one shader-table record's byte offset.
    RayTracingPipelineGroupDataAlignment,
    /// Required alignment of a shader-table region start.
    RayTracingPipelineGroupDataOffsetAlignment,
    /// Uniform binding bounds-check alignment.
    UniformBoundsCheckAlignment,
    /// Buffer binding size alignment.
    BufferBindingSizeAlignment,
}

impl LimitKey {
    /// Whether a larger value for this key is the more capable one.
    ///
    /// The specification states the convention once, in section 7.4, and it is
    /// easy to read past. Stating it as a total function over the keys — with no
    /// wildcard arm, so a new key fails to compile here until it is classified —
    /// is what keeps [`LimitRequirement`] from being built backwards.
    ///
    /// # Why this has no caller in the crate
    ///
    /// It was written for the requirement-versus-answer comparison, and that
    /// comparison does not need it: [`LimitRequirement`]'s variant already carries
    /// the direction of the bound (`AtLeast` reads the device's value from below,
    /// `AtMost` from above), so the key's own direction never has to be consulted
    /// to compare. What this classifies is which *spelling* expresses "at least
    /// this capable" for a given key — the knowledge a producer needs when it turns
    /// a reflection result into a requirement, and the reason section 7.4 refuses
    /// to collapse the two variants into one `minimum_limit()`.
    ///
    /// So it is a classifier with an audience outside this build, and it is kept
    /// rather than deleted for the property the no-wildcard match gives it: adding
    /// a limit key without deciding its direction is a compile error, and the
    /// direction is the one thing about a limit a caller cannot read off the name.
    /// The contract tests exercise it.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "a classifier for producers rather than a step of any comparison in this crate: the requirement's variant carries the direction. Exercised by the contract tests; see the note above"
        )
    )]
    pub(crate) fn larger_is_stronger(self) -> bool {
        match self {
            LimitKey::MaxBufferSize
            | LimitKey::MaxTexture1dDimension
            | LimitKey::MaxTexture2dDimension
            | LimitKey::MaxTexture3dDimension
            | LimitKey::MaxTextureArrayLayers
            | LimitKey::MaxBindGroups
            | LimitKey::MaxBindingsPerGroup
            | LimitKey::MaxBindGroupsPlusVertexBuffers
            | LimitKey::MaxUniformBufferBindingSize
            | LimitKey::MaxStorageBufferBindingSize
            | LimitKey::MaxDynamicUniformBuffersPerPipelineLayout
            | LimitKey::MaxDynamicStorageBuffersPerPipelineLayout
            | LimitKey::MaxSamplerAnisotropy
            | LimitKey::MaxColorAttachments
            | LimitKey::MaxColorAttachmentBytesPerSample
            | LimitKey::MaxVertexBuffers
            | LimitKey::MaxVertexAttributes
            | LimitKey::MaxVertexBufferArrayStride
            | LimitKey::MaxInterStageShaderVariables
            | LimitKey::MaxComputeInvocationsPerWorkgroup
            | LimitKey::MaxComputeWorkgroupSizeX
            | LimitKey::MaxComputeWorkgroupSizeY
            | LimitKey::MaxComputeWorkgroupSizeZ
            | LimitKey::MaxComputeWorkgroupsPerDimension
            | LimitKey::MaxComputeWorkgroupStorageSize
            | LimitKey::MaxQueriesPerQuerySet
            | LimitKey::MaxBindingArrayElementsPerShaderStage
            | LimitKey::MaxBindingArrayAccelerationStructureElementsPerShaderStage
            | LimitKey::MaxBindingArraySamplerElementsPerShaderStage
            | LimitKey::MaxNonSamplerBindings
            | LimitKey::MaxImmediateSize
            | LimitKey::MaxMultiviewViewCount
            | LimitKey::MaxTaskWorkgroupTotalCount
            | LimitKey::MaxTaskWorkgroupsPerDimension
            | LimitKey::MaxMeshWorkgroupTotalCount
            | LimitKey::MaxMeshWorkgroupsPerDimension
            | LimitKey::MaxTaskInvocationsPerWorkgroup
            | LimitKey::MaxTaskInvocationsPerDimension
            | LimitKey::MaxMeshInvocationsPerWorkgroup
            | LimitKey::MaxMeshInvocationsPerDimension
            | LimitKey::MaxTaskPayloadSize
            | LimitKey::MaxMeshOutputVertices
            | LimitKey::MaxMeshOutputPrimitives
            | LimitKey::MaxMeshOutputLayers
            | LimitKey::MaxMeshMultiviewViewCount
            | LimitKey::MaxBlasPrimitiveCount
            | LimitKey::MaxBlasGeometryCount
            | LimitKey::MaxTlasInstanceCount
            | LimitKey::RawTlasInstanceSize
            | LimitKey::MaxAccelerationStructuresPerShaderStage
            | LimitKey::MaxBuffersAndAccelerationStructuresPerShaderStage
            | LimitKey::MaxRayDispatchCount
            | LimitKey::MaxRayRecursionDepth
            | LimitKey::MaxRayTracingPipelineGroupDataSize => true,
            LimitKey::MinUniformBufferOffsetAlignment
            | LimitKey::MinStorageBufferOffsetAlignment
            | LimitKey::MapAlignment
            | LimitKey::MapOffsetAlignment
            | LimitKey::MapSizeAlignment
            | LimitKey::QueryResolveBufferAlignment
            | LimitKey::ImmediateDataAlignment
            | LimitKey::UniformBoundsCheckAlignment
            | LimitKey::BufferBindingSizeAlignment
            | LimitKey::RayTracingPipelineGroupDataAlignment
            | LimitKey::RayTracingPipelineGroupDataOffsetAlignment => false,
            LimitKey::RayTracingScratchBufferAlignment => false,
        }
    }
}

/// One limit the resulting device must satisfy.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitRequirement {
    /// At least this value is required, as in `MaxBufferSize >= value`.
    AtLeast {
        /// The limit being constrained.
        key: LimitKey,
        /// The bound the device must reach.
        value: u64,
    },
    /// At most this value is required, as in
    /// `MinUniformBufferOffsetAlignment <= value`.
    AtMost {
        /// The limit being constrained.
        key: LimitKey,
        /// The bound the device must stay within.
        value: u64,
    },
}

impl LimitRequirement {
    /// The limit being constrained.
    pub fn key(self) -> LimitKey {
        match self {
            LimitRequirement::AtLeast { key, .. } | LimitRequirement::AtMost { key, .. } => key,
        }
    }

    /// The bound the device must satisfy.
    pub fn value(self) -> u64 {
        match self {
            LimitRequirement::AtLeast { value, .. } | LimitRequirement::AtMost { value, .. } => {
                value
            }
        }
    }
}

/// Everything a caller asks of the device it is about to create.
///
/// A closing builder: every `require_*`/`prefer_*` verb consumes and returns the
/// value, so requirements can be written as one expression and cannot be
/// half-applied to a device that already exists.
///
/// The two feature lists are not a ranking. A required feature that cannot be
/// enabled makes the whole request fail; a preferred one that cannot be enabled
/// does not. Which of the preferred features were actually enabled is a question
/// for [`crate::api::platform::Device::capabilities`] afterwards, not something
/// this type can answer.
#[derive(Clone, Debug, Default)]
pub struct DeviceRequirements {
    required_features: Vec<OptionalFeature>,
    preferred_features: Vec<OptionalFeature>,
    limit_requirements: Vec<LimitRequirement>,
    required_buffers: Vec<BufferSupportQuery>,
    required_textures: Vec<TextureSupportQuery>,
    required_bindings: Vec<BindingSupportQuery>,
    required_routes: Vec<RouteQuery>,
}

impl DeviceRequirements {
    /// No requirements at all.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requires a feature. Its absence fails the entire device request.
    pub fn require_feature(mut self, feature: OptionalFeature) -> Self {
        self.required_features.push(feature);
        self
    }

    /// Asks for a feature. Its absence does not fail the request.
    pub fn prefer_feature(mut self, feature: OptionalFeature) -> Self {
        self.preferred_features.push(feature);
        self
    }

    /// Requires `key >= value`, for the keys where larger is more capable.
    pub fn require_limit_at_least(mut self, key: LimitKey, value: u64) -> Self {
        self.limit_requirements
            .push(LimitRequirement::AtLeast { key, value });
        self
    }

    /// Requires `key <= value`, for the keys where smaller is more capable.
    pub fn require_limit_at_most(mut self, key: LimitKey, value: u64) -> Self {
        self.limit_requirements
            .push(LimitRequirement::AtMost { key, value });
        self
    }

    /// Requires that the resulting device can create these buffer semantics.
    pub fn require_buffer_support(mut self, query: BufferSupportQuery) -> Self {
        self.required_buffers.push(query);
        self
    }

    /// Requires that the resulting device can create these texture semantics.
    ///
    /// Deliberately not a "require format": whether a format is usable depends on
    /// its dimension, usage, and sample count together with the view-compatibility
    /// intent at creation, so a format-only query would be ambiguous about what
    /// it promised.
    pub fn require_texture_support(mut self, query: TextureSupportQuery) -> Self {
        self.required_textures.push(query);
        self
    }

    /// Requires that the resulting device can express these binding semantics.
    pub fn require_binding_support(mut self, query: BindingSupportQuery) -> Self {
        self.required_bindings.push(query);
        self
    }

    /// Requires that the resulting device has these transfer, resolve, or blit
    /// routes.
    pub fn require_route(mut self, query: RouteQuery) -> Self {
        self.required_routes.push(query);
        self
    }

    /// The features whose absence fails the request.
    pub fn required_features(&self) -> &[OptionalFeature] {
        &self.required_features
    }

    /// The features asked for on a best-effort basis.
    pub fn preferred_features(&self) -> &[OptionalFeature] {
        &self.preferred_features
    }

    /// The limits the device must satisfy.
    pub fn limit_requirements(&self) -> &[LimitRequirement] {
        &self.limit_requirements
    }

    /// The buffer semantics the device must be able to create.
    pub fn required_buffer_support(&self) -> &[BufferSupportQuery] {
        &self.required_buffers
    }

    /// The texture semantics the device must be able to create.
    pub fn required_texture_support(&self) -> &[TextureSupportQuery] {
        &self.required_textures
    }

    /// The binding semantics the device must be able to express.
    pub fn required_binding_support(&self) -> &[BindingSupportQuery] {
        &self.required_bindings
    }

    /// The transfer routes the device must have.
    pub fn required_route_support(&self) -> &[RouteQuery] {
        &self.required_routes
    }
}

// ---------------------------------------------------------------------------
// Canonical capability encoding
// ---------------------------------------------------------------------------
//
// The rules of the encoding, and what it is for, are stated once in
// `api::capability::CapabilityFacts`. It lives here because these two
// vocabularies are declared here, and the limits map is keyed by one of them.

impl OptionalFeature {
    /// Writes this feature's canonical byte.
    ///
    /// A fieldless enum encodes as its discriminant; see
    /// [`crate::api::shader::ShaderStage::encode_into`] for why that dependency on
    /// declaration order is the intended one.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        out.push(*self as u8);
    }
}

impl LimitKey {
    /// Writes this limit key's canonical byte.
    ///
    /// A fieldless enum encodes as its discriminant, which is also what makes the
    /// limits section of the encoding orderable without `Ord`: section 7.4 fixes
    /// this type's derive list, and `Ord` is not on it, so the canonical order has
    /// to come from the encoded bytes rather than from the key.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        out.push(*self as u8);
    }
}
