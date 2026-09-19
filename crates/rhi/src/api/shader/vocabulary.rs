//! Sections 19.1-19.4: the shader's stage, code form, and location vocabulary.
//!
//! Stages and stage sets, the GLSL profile and the code forms a backend can
//! consume, the lowering ABI version, the device's acceptance verdict, the
//! numeric type of an inter-stage value, and the location/interpolation pair.
//! Everything here is plain data a caller states; nothing here asks a device
//! anything, and nothing here is validated.
//!
//! Not owned here: what an entry point *requires* (section 19.5, in
//! `requirements.rs`) and the rules that refuse an inconsistent one (in
//! `validation.rs`).

use std::sync::Arc;

/// The stage an entry point belongs to.
///
/// P0 freezes three: two base graphics stages plus compute. Section 19.1 makes
/// compute legal only when `OptionalFeature::Compute` is enabled, which is a
/// device fact rather than a property of this enum — the same enum value is legal
/// on one device and not on another.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    /// The vertex stage, which also owns the vertex-input contract.
    Vertex,
    /// The fragment stage, which owns the color-target and depth outputs.
    Fragment,
    /// The compute stage, which owns the workgroup contract.
    Compute,
}

/// A set of [`ShaderStage`] values.
///
/// A newtype over the bits of its members rather than a `Vec` or a `HashSet`,
/// because it appears in a capability query key
/// ([`BindingSupportQuery::visibility`](crate::api::binding::BindingSupportQuery::visibility))
/// that must be cheap to copy and to compare, and because the set is closed at
/// three members for as long as P0 freezes three stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderStages(u8);

impl ShaderStages {
    /// The vertex stage alone.
    pub const VERTEX: Self = Self(1 << 0);
    /// The fragment stage alone.
    pub const FRAGMENT: Self = Self(1 << 1);
    /// The compute stage alone.
    pub const COMPUTE: Self = Self(1 << 2);

    /// Whether every bit set in `other` is set in `self`.
    ///
    /// An empty `other` is contained in everything, which is why the "visibility
    /// is non-empty" rule is checked where a layout entry is validated rather
    /// than being implied here.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two stage sets.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether no stage bit is set.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// The set of stages one [`ShaderStage`] denotes.
///
/// A free function rather than a method on [`ShaderStage`] or [`ShaderStages`]:
/// it is the mapping between two types that section 19.1 declares separately,
/// and putting it here keeps both of their declared surfaces exactly as the
/// specification writes them.
///
/// No wildcard arm, so a fourth stage fails to compile here until it is mapped.
pub(crate) fn stage_mask(stage: ShaderStage) -> ShaderStages {
    match stage {
        ShaderStage::Vertex => ShaderStages::VERTEX,
        ShaderStage::Fragment => ShaderStages::FRAGMENT,
        ShaderStage::Compute => ShaderStages::COMPUTE,
    }
}

/// The GLSL dialect a desktop GLSL source is written in.
///
/// One member in P0. It is an enum rather than a boolean because the next
/// dialects that matter — compatibility profiles, ES-with-extensions — are
/// additional variants of the same question, and section 19.2 makes the source
/// form carry its dialect explicitly rather than have the backend guess it from
/// the `#version` line.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlslProfile {
    /// Core profile, without the deprecated compatibility features.
    Core,
}

/// A code form that the current device backend can consume.
///
/// The variants name *forms*, not backends, even though each is the canonical
/// form for one: a device is what decides acceptability, and it does so through
/// `EnabledCapabilities::shader_acceptance`. Two devices of the same backend
/// family can therefore answer differently about the same value.
///
/// The payloads are reference-counted and immutable so that an artifact can be
/// cloned, cached, and handed to capture without copying shader text or bytecode,
/// and so that nothing here can be mutated after an artifact's content hash was
/// computed from it.
///
/// This enum is not `Copy`: the payloads are shared pointers, and a caller that
/// wants a second handle wants a `Clone` of the pointer, not a new allocation.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum ShaderCode {
    /// Canonical source form for the WebGPU backend.
    Wgsl(Arc<str>),

    /// Canonical binary/module form for the Vulkan backend.
    SpirV(Arc<[u32]>),

    /// Canonical compiled form for the DX12 backend.
    Dxil(Arc<[u8]>),

    /// Source form that the Metal backend may compile at runtime.
    Msl(Arc<str>),

    /// Compiled library/function provenance for the Metal backend.
    Metallib(Arc<[u8]>),

    /// Desktop OpenGL source.
    Glsl {
        /// The `#version` number the source declares.
        version: u16,
        /// The dialect the source is written in.
        profile: GlslProfile,
        /// The source text.
        source: Arc<str>,
    },

    /// OpenGL ES / WebGL2 source.
    ///
    /// No profile field: ES has no core/compatibility split, and adding one
    /// would invite a caller to set a value the target cannot honour.
    GlslEs {
        /// The `#version` number the source declares.
        version: u16,
        /// The source text.
        source: Arc<str>,
    },
}

/// The Fluxel logical-to-native lowering ABI an artifact was produced against.
///
/// The portable interface names a logical `group`/`slot` and a vertex or fragment
/// `location`; every backend lowers those to its own registers, sets, or indices,
/// and the *strategy* for doing so — argument buffer, root signature, binding
/// table — is backend/toolchain-private (section 19.3). This version identifies
/// which lowering contract the artifact was built against.
///
/// Section 19.3 makes it a hard boundary: when the ABI version changes, an old
/// executable must not be silently interpreted using new rules, so acceptance is
/// an explicit device decision rather than an assumption.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderAbiVersion {
    /// Breaking component.
    pub major: u16,
    /// Compatible-extension component.
    pub minor: u16,
}

/// The device's verdict on one shader artifact.
///
/// A richer answer than a boolean because the refusals are not interchangeable to
/// a caller: a code format this device cannot consume, an ABI this device does not
/// implement, a missing optional feature, a limit that was exceeded, and an
/// interface the device cannot express each point at a different remedy.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactAcceptance {
    /// The device can create a module from this artifact.
    Accepted,
    /// The device cannot consume this [`ShaderCode`] form.
    UnsupportedCodeFormat,
    /// The artifact was lowered against a [`ShaderAbiVersion`] this device does
    /// not implement.
    UnsupportedAbi,
    /// An optional feature the artifact requires is not enabled on this device.
    MissingFeature,
    /// A limit the artifact requires exceeds this device's.
    LimitExceeded,
    /// The device cannot express the artifact's entry-point interface.
    InterfaceUnsupported,
}

/// The numeric type of a shader stage input or output.
///
/// P0 freezes 32-bit numeric stage IO only. Section 19.4 reserves f16 and packed
/// inter-stage IO for a future capability-gated addition to *this* enum, because a
/// separate type or trait would move a capability into the type system, where a
/// caller cannot ask about it at run time.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderNumericType {
    /// 32-bit IEEE float.
    Float32,
    /// 32-bit signed integer.
    Sint32,
    /// 32-bit unsigned integer.
    Uint32,
}

/// A logical vertex or fragment location.
///
/// Not a native semantic, register, or attribute index: the toolchain lowers it,
/// and the lowering is not portable API (section 19.3).
///
/// A public constructor, unlike the identity tokens of section 3: a location is a
/// logical index the *caller* chooses when it writes a pipeline, not a value only
/// the RHI may mint. Minting one cannot forge an identity comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderLocation(u32);

impl ShaderLocation {
    /// Names one logical location.
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the logical value.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// How a value is interpolated across a primitive.
///
/// `Flat` is not a hint: section 19.6 requires integer inter-stage IO to be flat,
/// because there is no meaningful interpolation between two integers, and a
/// backend that received `Perspective` for a `Sint32` location would either
/// refuse it or produce a value that no other backend reproduces.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Perspective-correct interpolation.
    Perspective,
    /// Screen-linear interpolation.
    Linear,
    /// No interpolation: the provoking vertex's value reaches every fragment.
    Flat,
}

/// Where within a fragment the interpolation is sampled.
///
/// Separate from [`InterpolationMode`] because the two questions are independent:
/// a value can be flat *and* sampled at the centroid, and a backend can support
/// one combination while refusing another.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationSampling {
    /// At the fragment center.
    Center,
    /// At a covered sample inside the fragment.
    Centroid,
    /// At the sample being shaded.
    Sample,
}

/// The interpolation of one inter-stage location.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderInterpolation {
    /// How the value is interpolated.
    pub mode: InterpolationMode,
    /// Where the interpolated value is taken.
    pub sampling: InterpolationSampling,
}

/// One entry-point location: its numeric type, width, and interpolation.
///
/// The same struct describes a vertex attribute, a vertex output, a fragment
/// input, and a fragment output, because in every one of those roles the four
/// facts are the same four facts. This is what lets the vertex-to-fragment
/// linkage rule of section 27.3 be a comparison between two of these rather than a
/// table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderLocationInterface {
    /// The logical location.
    pub location: ShaderLocation,
    /// The numeric type carried at this location.
    pub numeric_type: ShaderNumericType,

    /// How many components the value has, `1..=4`.
    ///
    /// The upper bound is what makes this type describe P0's 32-bit IO: a wider
    /// value is a second location, not a wider one.
    pub components: u8,

    /// How the value is interpolated, when it crosses stages.
    ///
    /// Usually `None` for vertex inputs and fragment outputs, which do not cross a
    /// stage boundary. Vertex outputs and fragment inputs must be canonicalized
    /// by the artifact producer to explicit interpolation, because the pipeline
    /// has to compare the two sides and `None` on one side against `Some` on the
    /// other is not a comparison any backend could act on.
    pub interpolation: Option<ShaderInterpolation>,
}
