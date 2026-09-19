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
    /// Fixed-length arrays of buffers, textures, or samplers.
    ///
    /// Runtime-sized, partially-bound, update-after-bind, and arbitrarily-indexed
    /// binding arrays remain future bindless/indexing extensions and are not
    /// covered by this variant.
    BindingArrays,
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
    /// Smallest uniform buffer offset alignment, in bytes.
    MinUniformBufferOffsetAlignment,
    /// Smallest storage buffer offset alignment, in bytes.
    MinStorageBufferOffsetAlignment,
}

impl LimitKey {
    /// Whether a larger value for this key is the more capable one.
    ///
    /// The specification states the convention once, in section 7.4, and it is
    /// easy to read past. Stating it as a total function over the keys — with no
    /// wildcard arm, so a new key fails to compile here until it is classified —
    /// is what keeps [`LimitRequirement`] from being built backwards.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "exercised by the contract tests; the requirement-versus-limit comparison that reads it is not written"
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
            | LimitKey::MaxComputeWorkgroupStorageSize => true,
            LimitKey::MinUniformBufferOffsetAlignment
            | LimitKey::MinStorageBufferOffsetAlignment => false,
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
