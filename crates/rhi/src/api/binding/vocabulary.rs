//! Section 20: the binding vocabulary.
//!
//! Logical group and slot identity, counts, kinds, and the capability *question*
//! one binding asks of a device, together with the validators of vocabulary
//! invariants that every other file in the module asks through: a kind that
//! states a usable size, a count the portable layer can ask about, and the class
//! a binding is counted under.
//!
//! Not owned here: the answers to those questions (they are the device's, and
//! arrive as parameters) and the layouts and packets that are written in this
//! vocabulary — those are sections 21 and 22.

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::TextureFormat;
use crate::api::resource::view::TextureViewDimension;
use crate::api::shader::ShaderStages;

/// The logical index of a bind group within a
/// [`crate::api::pipeline::PipelineInterface`].
///
/// A Fluxel logical index, not a Vulkan descriptor set number, not an HLSL
/// register space, not a Metal buffer index (section 20.1). The toolchain lowers
/// it; the lowering is not portable API (section 19.3).
///
/// Publicly constructible, unlike the identity tokens of section 3: a group index
/// is a logical position the caller chooses while writing a pipeline, and minting
/// one cannot forge an identity comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindGroupIndex(u32);

impl BindGroupIndex {
    /// Names one logical group.
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the logical value.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// The logical slot of one binding within a bind group.
///
/// Publicly constructible for the same reason as [`BindGroupIndex`], and a
/// distinct type so that a group index can never be passed where a slot was
/// meant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindingSlotId(u32);

impl BindingSlotId {
    /// Names one logical slot.
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the logical value.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// How many resource elements one logical binding holds.
///
/// `Fixed(n)` is capability-gated vocabulary, and section 20.2 is careful about
/// what it does *not* imply: not runtime-sized, not partially bound, not
/// update-after-bind, not non-uniform arbitrary descriptor indexing. Those remain
/// future bindless/indexing extensions, and a backend is free to answer
/// [`BindingSupport::Unsupported`] for the whole vocabulary — WebGPU core does
/// not require it.
///
/// `Hash` is required because a count is part of a [`BindingSupportQuery`], which
/// is a capability-cache key; the specification's derive list omits it (see
/// adjudication A25 in this crate's 0.16 series plan).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BindingCount {
    /// Exactly one resource element.
    One,

    /// A fixed-length resource binding array.
    ///
    /// `value >= 2`. There is no `Fixed(1)`: section 22.1 states outright that an
    /// array of length 1 cannot stand in for [`Self::One`], so a count of one
    /// element has exactly one legal spelling.
    Fixed(u32),
}

impl BindingCount {
    /// The number of resource elements this count requires.
    ///
    /// Total rather than fallible. [`Self::Fixed`] is documented as `>= 2`, and
    /// `validate_binding_count` is what refuses `Fixed(0)` and `Fixed(1)`; this
    /// accessor still answers for a value that never passed validation, because an
    /// accessor that panicked on bad input would be a second, invisible validation
    /// rule.
    pub fn elements(self) -> u32 {
        match self {
            Self::One => 1,
            Self::Fixed(elements) => elements,
        }
    }
}

/// What numeric type a shader may sample a texture as.
///
/// The distinction between [`Self::Float`] and [`Self::UnfilterableFloat`] is the
/// reason this is a separate question from "is it a float format": a float format
/// permits a filtering sampler and an unfilterable-float format does not, so
/// collapsing the two would make a shader that samples through a filtering
/// sampler unrepresentable.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureSampleType {
    /// Sampled as a float, filterable.
    Float,
    /// Sampled as a float, not filterable.
    UnfilterableFloat,
    /// Sampled as a signed integer.
    Sint,
    /// Sampled as an unsigned integer.
    Uint,
    /// Sampled as a depth value.
    Depth,
}

/// What a shader may do to a storage texture.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StorageAccess {
    /// Read-only access.
    ReadOnly,
    /// Write-only access.
    WriteOnly,
    /// Both reads and writes.
    ReadWrite,
}

/// What a shader expects of a sampler.
///
/// A property of the *binding*, not of the sampler object: the same
/// [`crate::api::resource::sampler::SamplerDescriptor`] may satisfy a
/// [`Self::Filtering`] interface in one place and fail a [`Self::Comparison`]
/// interface in another. Section 22.3 therefore checks only that the kind is
/// compatible with the descriptor and leaves the paired-use verdict to pipeline
/// validation.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SamplerKind {
    /// May be used with filtering.
    Filtering,
    /// Must not filter.
    NonFiltering,
    /// Compares against a reference instead of filtering.
    Comparison,
}

/// What a shader may do to a storage buffer.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferBindingAccess {
    /// Read-only access.
    ReadOnly,
    /// Both reads and writes.
    ReadWrite,
}

/// The resource semantics of one logical binding.
///
/// Section 19.5 makes the shader-side requirement reuse this enum directly rather
/// than maintain a parallel `ShaderBindingKind`, so that reflection and layout
/// cannot drift into two systems that disagree about what a storage texture is.
/// That is why [`crate::api::shader::ShaderResourceRequirement::kind`] is this
/// type and not a look-alike.
///
/// `min_size > 0` for both buffer kinds. Section 20.3 refuses a magic zero
/// explicitly: if a future requirement needs zero to mean "determined by runtime
/// binding size", that is a separately designed contract, not a value of this
/// field.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BindingKind {
    /// A uniform buffer binding.
    UniformBuffer {
        /// The minimum visible byte range required by the shader/layout.
        ///
        /// Also the field the fixed-array full-use rule is applied to: when a
        /// stage declares a `Fixed(n)` binding, every element counts as used, so
        /// the pipeline validator aggregates `min_size` over all of them rather
        /// than assuming the shader touches only some.
        min_size: u64,
    },

    /// A storage buffer binding.
    StorageBuffer {
        /// What the shader may do to the buffer.
        access: BufferBindingAccess,
        /// The minimum visible byte range required by the shader/layout.
        min_size: u64,
    },

    /// A sampled texture binding.
    SampledTexture {
        /// The view dimension the shader expects.
        dimension: TextureViewDimension,
        /// The numeric type the shader samples as.
        sample_type: TextureSampleType,
        /// Whether the shader samples a multisampled texture.
        multisampled: bool,
    },

    /// A storage texture binding.
    StorageTexture {
        /// The view dimension the shader expects.
        dimension: TextureViewDimension,
        /// The format the shader reads and writes.
        format: TextureFormat,
        /// What the shader may do to the texture.
        access: StorageAccess,
    },

    /// A sampler binding.
    Sampler {
        /// What the shader expects of the sampler.
        kind: SamplerKind,
    },
}

/// One question about whether, and how, a device can satisfy a binding.
///
/// Section 20.4 introduces this as a formal query because binding support does
/// not follow from "the device supports textures": `StorageTexture + Cube`,
/// `StorageTexture + ReadWrite`, `StorageBuffer` in the vertex stage, a fixed
/// resource array, and a dynamic buffer offset each have independent limitations.
///
/// The whole query is the key, which is what makes the answer cacheable and what
/// keeps the capability surface from growing one boolean per combination.
///
/// `PartialEq`, `Eq`, and `Hash` are required by
/// [`crate::api::capability::EnabledCapabilities`], which stores its answers in a
/// `HashMap` keyed by this type; the specification's derive list omits them. The
/// whole family of deviations this forces — six sibling enums inherit `Hash` from
/// this type being a map key — is adjudicated together as A25 in this crate's
/// 0.16 series plan.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BindingSupportQuery {
    /// The stages that will see the binding.
    pub visibility: ShaderStages,
    /// The resource semantics being asked about.
    pub kind: BindingKind,
    /// The resource count being asked about.
    pub count: BindingCount,

    /// Whether a dynamic offset will be applied.
    ///
    /// "Valid only for UniformBuffer / StorageBuffer", per section 20.4: for the
    /// other kinds it is not a question that has an answer, and the layout
    /// validator refuses a dynamic offset on them before any query is made.
    pub dynamic_offset: bool,
}

/// The device's answer to a [`BindingSupportQuery`].
///
/// Two members, and deliberately not more: section 20.4 freezes the query
/// vocabulary, and a third member such as "supported with a smaller maximum
/// count" would be a limit — a separate question, answered by
/// `binding_limit(stage, class)`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingSupport {
    /// The device cannot express this binding at all.
    Unsupported,
    /// The device can express this binding.
    Supported,
}

/// The resource class a binding-count limit is grouped under.
///
/// Separate from [`BindingKind`] because the limits are counted per class rather
/// than per kind: a device states one ceiling for every uniform buffer visible to
/// a stage, not one per `min_size`. Section 20.4 freezes the five members.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BindingLimitClass {
    /// Uniform buffer bindings.
    UniformBuffers,
    /// Storage buffer bindings.
    StorageBuffers,
    /// Sampled texture bindings.
    SampledTextures,
    /// Storage texture bindings.
    StorageTextures,
    /// Sampler bindings.
    Samplers,
}

/// The class one binding kind is counted under.
///
/// No wildcard arm: a sixth [`BindingKind`] variant fails to compile here until it
/// is classified, which is the point — an unclassified kind would silently escape
/// every aggregate limit of section 23.1.
pub(crate) fn binding_kind_class(kind: &BindingKind) -> BindingLimitClass {
    match kind {
        BindingKind::UniformBuffer { .. } => BindingLimitClass::UniformBuffers,
        BindingKind::StorageBuffer { .. } => BindingLimitClass::StorageBuffers,
        BindingKind::SampledTexture { .. } => BindingLimitClass::SampledTextures,
        BindingKind::StorageTexture { .. } => BindingLimitClass::StorageTextures,
        BindingKind::Sampler { .. } => BindingLimitClass::Samplers,
    }
}

/// Whether a binding kind states a usable size.
///
/// Section 20.3's `min_size > 0`, in one place because two chapters depend on it:
/// layout creation and shader-artifact validation both reject a zero-sized buffer
/// binding, and a second copy of the rule is how the two chapters start to
/// disagree.
pub(crate) fn validate_binding_kind(kind: &BindingKind) -> RhiResult<()> {
    let min_size = match kind {
        BindingKind::UniformBuffer { min_size } => Some(*min_size),
        BindingKind::StorageBuffer { min_size, .. } => Some(*min_size),
        BindingKind::SampledTexture { .. }
        | BindingKind::StorageTexture { .. }
        | BindingKind::Sampler { .. } => None,
    };
    if min_size == Some(0) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a buffer binding must require at least one byte; zero is not a size",
        ));
    }
    Ok(())
}

/// Whether a binding count is one the portable layer can ask about.
///
/// Section 20.5's `Fixed(n): n >= 2`. Refused rather than repaired, because
/// section 22.1 states that an array of length 1 cannot stand in for
/// [`BindingCount::One`] — so there is exactly one legal spelling of "one
/// element", and a `Fixed(1)` is a producer that has not decided which it means.
pub(crate) fn validate_binding_count(count: BindingCount) -> RhiResult<()> {
    let elements = match count {
        BindingCount::One => return Ok(()),
        BindingCount::Fixed(elements) => elements,
    };
    if elements < 2 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "a fixed binding array holds at least 2 elements, not {elements}; one element \
                 is BindingCount::One"
            ),
        ));
    }
    Ok(())
}

/// Whether a binding kind is one a dynamic offset has any meaning for.
///
/// Section 20.4 says the `dynamic_offset` field is "valid only for
/// UniformBuffer / StorageBuffer". Stated once so that the layout validator and
/// any future caller of it cannot disagree.
pub(crate) fn is_buffer_kind(kind: &BindingKind) -> bool {
    match kind {
        BindingKind::UniformBuffer { .. } | BindingKind::StorageBuffer { .. } => true,
        BindingKind::SampledTexture { .. }
        | BindingKind::StorageTexture { .. }
        | BindingKind::Sampler { .. } => false,
    }
}
