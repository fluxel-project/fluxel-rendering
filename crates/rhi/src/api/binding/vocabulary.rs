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
/// `Hash` is *not* derived, and it used to be. A count was part of the capability
/// cache key while the whole [`BindingSupportQuery`] was that key; the key is now
/// the narrowed `BindingSupportKey`, which records whether a binding is an array
/// and not how many elements it holds, so nothing asks a count to be hashable. The
/// specification's derive list omits `Hash` here and the deviation is withdrawn
/// rather than kept (adjudication A25 in this crate's 0.16 series plan).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
///
/// `Hash` is not derived, for the reason given on [`BindingCount`]: this type is no
/// longer a capability-cache key. The four payload enums above are — they are
/// fields of `BindableKind`, which is — see the note there.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
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
/// The whole query used to be the capability cache key, and it is not any more: the
/// key is `BindingSupportKey`, narrowed to the fields an answer actually depends
/// on, because the query's two magnitudes have owners elsewhere in the
/// specification and section 7.3 allows a fact one canonical source. See that
/// key's doc for why, and for what a caller loses by it — nothing observable, since
/// the accessor is a function of the query either way.
///
/// `PartialEq` and `Eq` stay: comparing two requirements is a caller need that owes
/// nothing to the cache, and `Hash` does not, so `Hash` is not derived here and the
/// deviation the specification's derive list would have been charged with does not
/// arise. What *is* still derived against that list is `Hash` on the four payload
/// enums of [`BindingKind`], which `BindableKind` does need; the whole family is
/// adjudicated together as A25 in this crate's 0.16 series plan.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
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

/// [`BindingKind`] with its two magnitudes removed.
///
/// Lives beside the type it mirrors, and next to [`binding_kind_class`], because
/// both answer the same shape of question — "what part of a binding kind does
/// *this* rule care about" — and a mirror kept away from its original is how the
/// two start to disagree about a variant. No wildcard arm, so a sixth
/// [`BindingKind`] variant fails to compile here until its magnitudes are
/// identified.
///
/// # Why a mirror rather than a canonicalized [`BindingKind`]
///
/// The alternative is to store a `min_size` no caller asked about — the smallest
/// legal one, say — and a fabricated field in a key is a claim about a query that
/// the query did not make. Section 20.3 refuses that shape of shortcut for this
/// very field ("if a future requirement needs zero to mean a different contract …
/// do not use magic zero"), and the reasoning covers a magic one as well.
///
/// # Why the magnitude is not a support question
///
/// `min_size`'s *magnitude* is section 22.3's, measured against
/// `MaxUniformBufferBindingSize` / `MaxStorageBufferBindingSize` when a BindGroup
/// is created, where the range's actual size is what the rule is about.
/// [`BindingCount::Fixed`]'s magnitude is section 23.1's, aggregated per stage and
/// class against `binding_limit(stage, class)`. Both answers already have exactly
/// one owner, and section 7.3's closing rule — a fact can only have one canonical
/// source — is what keeps a support table from becoming a second one. The
/// *kind*-shaped residue is what this type keeps, which is the same line
/// [`crate::api::capability::CapabilityFacts`] draws when it says its accessor
/// answers whether the kind of binding is expressible.
///
/// # Why four public enums carry `Hash` for a crate-private type's sake
///
/// This type is a field of the capability cache key, so it must be hashable, and
/// so must everything it holds: [`BufferBindingAccess`], [`TextureSampleType`],
/// [`StorageAccess`], and [`SamplerKind`] — plus `TextureViewDimension` and
/// `TextureFormat`, which derive it for their own reasons. The specification's
/// derive lists omit `Hash` on the four, and the deviation is one deviation with
/// one root rather than four, since a type that must satisfy a declared
/// containment relation cannot decline the traits the container needs. A25 in this
/// crate's 0.16 series plan adjudicates the family; the reader who finds a
/// "surplus" derive should read that entry before removing it, because removing
/// one of these stops this module compiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum BindableKind {
    /// A uniform buffer binding, at any declared minimum size.
    UniformBuffer,
    /// A storage buffer binding, at any declared minimum size.
    StorageBuffer {
        /// The access the shader declares.
        access: BufferBindingAccess,
    },
    /// A sampled texture binding.
    SampledTexture {
        /// The dimension the shader views it as.
        dimension: TextureViewDimension,
        /// The numeric type the shader samples as.
        sample_type: TextureSampleType,
        /// Whether the shader samples a multisampled texture.
        multisampled: bool,
    },
    /// A storage texture binding.
    StorageTexture {
        /// The dimension the shader views it as.
        dimension: TextureViewDimension,
        /// The format the shader reads and writes.
        format: TextureFormat,
        /// What the shader may do to it.
        access: StorageAccess,
    },
    /// A sampler binding.
    Sampler {
        /// What the shader expects of the sampler.
        kind: SamplerKind,
    },
}

impl BindableKind {
    /// The magnitude-free part of `kind`.
    pub(crate) fn of(kind: &BindingKind) -> Self {
        match kind {
            BindingKind::UniformBuffer { .. } => Self::UniformBuffer,
            BindingKind::StorageBuffer { access, .. } => Self::StorageBuffer { access: *access },
            BindingKind::SampledTexture {
                dimension,
                sample_type,
                multisampled,
            } => Self::SampledTexture {
                dimension: *dimension,
                sample_type: *sample_type,
                multisampled: *multisampled,
            },
            BindingKind::StorageTexture {
                dimension,
                format,
                access,
            } => Self::StorageTexture {
                dimension: *dimension,
                format: *format,
                access: *access,
            },
            BindingKind::Sampler { kind } => Self::Sampler { kind: *kind },
        }
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

// ---------------------------------------------------------------------------
// Canonical capability encoding
// ---------------------------------------------------------------------------
//
// The rules of the encoding, and what it is for, are stated once in
// `api::capability::CapabilityFacts`. It lives here because every field read
// below is private to this module.

/// The encoding of the five fieldless vocabularies this module declares.
///
/// One macro rather than five hand-written methods: the bodies would be
/// character-for-character identical, and the only thing distinguishing them is
/// the type. The doc comment each expansion carries is deliberately generic,
/// because the reasoning really is the same for all five — a fieldless enum
/// encodes as its discriminant, and the dependency on declaration order is the
/// intended one (see [`crate::api::shader::ShaderStage::encode_into`]).
macro_rules! fieldless_encoding {
    ($($type:ty),+ $(,)?) => {
        $(
            impl $type {
                /// Writes this value's canonical byte.
                pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
                    out.push(*self as u8);
                }
            }
        )+
    };
}

fieldless_encoding!(
    TextureSampleType,
    BufferBindingAccess,
    SamplerKind,
    StorageAccess,
    BindingLimitClass,
);

impl BindingKind {
    /// Writes this kind as a tag, then its fields in declaration order.
    ///
    /// Every field is written, including the ones a given variant makes
    /// interchangeable with a sibling. `UniformBuffer { min_size: 64 }` and
    /// `UniformBuffer { min_size: 128 }` are different bindings, and a device that
    /// can express one is not thereby stating it can express the other; a query
    /// keyed on the first must not intern to the same contract as one keyed on the
    /// second.
    ///
    /// No wildcard arm: a sixth binding kind must state its encoding before this
    /// compiles.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            Self::UniformBuffer { min_size } => {
                out.push(0);
                out.extend_from_slice(&min_size.to_le_bytes());
            }
            Self::StorageBuffer { access, min_size } => {
                out.push(1);
                access.encode_into(out);
                out.extend_from_slice(&min_size.to_le_bytes());
            }
            Self::SampledTexture {
                dimension,
                sample_type,
                multisampled,
            } => {
                out.push(2);
                dimension.encode_into(out);
                sample_type.encode_into(out);
                out.push(u8::from(*multisampled));
            }
            Self::StorageTexture {
                dimension,
                format,
                access,
            } => {
                out.push(3);
                dimension.encode_into(out);
                format.encode_into(out);
                access.encode_into(out);
            }
            Self::Sampler { kind } => {
                out.push(4);
                kind.encode_into(out);
            }
        }
    }
}

impl BindableKind {
    /// Writes this kind's canonical bytes, in [`BindingKind`]'s tag space.
    ///
    /// The three variants both types spell identically are *delegated* rather than
    /// re-encoded: this pushes tags 2, 3 and 4 by handing a `BindingKind` to the
    /// encoder above, so the two encodings cannot drift for them. The two buffer
    /// variants are the only ones written by hand, and only because their payload
    /// here is shorter by the `min_size` that `BindingKind` writes — the tag is the
    /// same number and the remaining payload the same order, so a reader comparing
    /// the two methods sees one vocabulary rather than two.
    ///
    /// No wildcard arm: a sixth binding kind must state its magnitude-free
    /// encoding before this compiles.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            Self::UniformBuffer => out.push(0),
            Self::StorageBuffer { access } => {
                out.push(1);
                access.encode_into(out);
            }
            Self::SampledTexture {
                dimension,
                sample_type,
                multisampled,
            } => BindingKind::SampledTexture {
                dimension: *dimension,
                sample_type: *sample_type,
                multisampled: *multisampled,
            }
            .encode_into(out),
            Self::StorageTexture {
                dimension,
                format,
                access,
            } => BindingKind::StorageTexture {
                dimension: *dimension,
                format: *format,
                access: *access,
            }
            .encode_into(out),
            Self::Sampler { kind } => BindingKind::Sampler { kind: *kind }.encode_into(out),
        }
    }
}

impl BindingSupport {
    /// Writes this answer's canonical byte.
    ///
    /// Fieldless, and written as a match rather than a discriminant cast so that a
    /// third member is a compile error here until it is encoded — the encoding
    /// carries a meaning, so it should not be inherited by accident.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            Self::Unsupported => out.push(0),
            Self::Supported => out.push(1),
        }
    }
}
