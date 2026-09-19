//! Shader stages, code forms, artifacts, and entry-point interfaces.
//!
//! This module owns rhi-design section 19.
//!
//! # What it is
//!
//! The RHI provides no shader cross-compiler. It accepts only a code form the
//! current backend can consume, and it decides acceptance through
//! [`super::EnabledCapabilities::shader_acceptance`] rather than through any
//! Rust type. A [`ShaderArtifact`] therefore carries the lowering ABI it was
//! built against, the canonical semantic interface it exposes, and the
//! requirements it places on the device, so that a mismatch is a structured
//! refusal before any backend shader object is created.
//!
//! # What it deliberately does not own
//!
//! The Fluxel logical `(group, slot, location)` triple is not a Vulkan
//! descriptor set/binding, an HLSL register or register space, a Metal
//! buffer/texture/sampler index, or a GL binding point. Lowering that interface
//! to a native binding ABI, and choosing an argument-buffer or root-signature
//! strategy, stays backend- and toolchain-private. Specialization constants are
//! a future P2 API; a P0 artifact must already have pipeline specialization
//! closed, so no half-complete constants map is pre-created here.

use std::sync::Arc;

use super::binding::{BindGroupIndex, BindingCount, BindingKind, BindingSlotId};
use super::platform::{
    BackendKind, Label, LimitKey, LimitRequirement, ObjectId, OptionalFeature, RhiError,
    RhiErrorKind, RhiResult,
};

/// A programmable shader stage.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShaderStage {
    /// The vertex stage.
    Vertex,
    /// The fragment stage.
    Fragment,
    /// The compute stage. Legal only when [`OptionalFeature::Compute`] is
    /// enabled.
    Compute,
}

/// A mask of shader stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShaderStages(u8);

impl ShaderStages {
    /// The vertex stage bit.
    pub const VERTEX: Self = Self(1 << 0);
    /// The fragment stage bit.
    pub const FRAGMENT: Self = Self(1 << 1);
    /// The compute stage bit.
    pub const COMPUTE: Self = Self(1 << 2);

    /// The bit for one stage.
    pub fn from_stage(stage: ShaderStage) -> Self {
        match stage {
            ShaderStage::Vertex => Self::VERTEX,
            ShaderStage::Fragment => Self::FRAGMENT,
            ShaderStage::Compute => Self::COMPUTE,
        }
    }

    /// Whether every bit of `other` is present.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two masks.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether no stage is set.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The raw bits, for canonical hashing and diagnostics.
    pub fn bits(self) -> u8 {
        self.0
    }

    /// Whether this mask covers the graphics stages only.
    pub fn is_graphics_only(self) -> bool {
        self.0 & !(Self::VERTEX.0 | Self::FRAGMENT.0) == 0
    }
}

/// The GLSL profile for a desktop OpenGL source artifact.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GlslProfile {
    /// The core profile.
    Core,
}

/// A backend-consumable shader code form.
///
/// There is deliberately no `Portable` variant: SPIR-V may be portable
/// toolchain provenance, but that does not mean a WebGPU browser can execute it,
/// so a code form is either directly consumable by one backend family or it is
/// not part of this vocabulary.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShaderCode {
    /// Canonical source form for the WebGPU backend.
    Wgsl(Arc<str>),

    /// Canonical binary module form for the Vulkan backend.
    SpirV(Arc<[u32]>),

    /// Canonical compiled form for the DX12 backend.
    Dxil(Arc<[u8]>),

    /// Source form the Metal backend may compile at runtime.
    Msl(Arc<str>),

    /// Compiled library/function provenance for the Metal backend.
    Metallib(Arc<[u8]>),

    /// Desktop OpenGL source.
    Glsl {
        /// The GLSL version, for example `450`.
        version: u16,
        /// The GLSL profile.
        profile: GlslProfile,
        /// The source text.
        source: Arc<str>,
    },

    /// OpenGL ES / WebGL2 source.
    GlslEs {
        /// The GLSL ES version, for example `310`.
        version: u16,
        /// The source text.
        source: Arc<str>,
    },
}

impl ShaderCode {
    /// The backend kind this code form is directly consumable by.
    ///
    /// [`ShaderCode::Metallib`] and [`ShaderCode::SpirV`] are both provenance
    /// for one family; the mapping here is the acceptance entry condition, not
    /// a claim that the device can link the code.
    pub fn backend_affinity(&self) -> BackendKind {
        match self {
            Self::Wgsl(_) => BackendKind::WebGpu,
            Self::SpirV(_) => BackendKind::Vulkan,
            Self::Dxil(_) => BackendKind::Dx12,
            Self::Msl(_) | Self::Metallib(_) => BackendKind::Metal,
            Self::Glsl { .. } => BackendKind::OpenGl,
            Self::GlslEs { .. } => BackendKind::WebGl2,
        }
    }
}

/// The lowering ABI a shader artifact was built against.
///
/// When the ABI version changes, an old executable must not be silently
/// interpreted with new rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShaderAbiVersion {
    /// The breaking ABI component.
    pub major: u16,
    /// The compatible ABI component.
    pub minor: u16,
}

impl ShaderAbiVersion {
    /// The ABI version this crate implements.
    pub const CURRENT: Self = Self { major: 1, minor: 0 };

    /// Whether an artifact built against this version may be used.
    ///
    /// A different major is always refused. A newer minor is refused because
    /// this crate cannot know the newer rules.
    pub fn accepts(&self, artifact: Self) -> bool {
        self.major == artifact.major && artifact.minor <= self.minor
    }
}

/// The device's verdict on one shader artifact.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ArtifactAcceptance {
    /// The artifact may be used.
    Accepted,
    /// The code form is not consumable by this backend.
    UnsupportedCodeFormat,
    /// The artifact's ABI version is not the one this device speaks.
    UnsupportedAbi,
    /// A required optional feature is not enabled.
    MissingFeature,
    /// A required limit is not met.
    LimitExceeded,
    /// The declared interface cannot be served by this device.
    InterfaceUnsupported,
}

/// A 32-bit numeric shader IO class.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShaderNumericType {
    /// A 32-bit float.
    Float32,
    /// A 32-bit signed integer.
    Sint32,
    /// A 32-bit unsigned integer.
    Uint32,
}

/// A Fluxel logical shader location.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShaderLocation(u32);

impl ShaderLocation {
    /// A location index.
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    /// The index.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// How a value is interpolated between vertices.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InterpolationMode {
    /// Perspective-correct interpolation.
    Perspective,
    /// Linear interpolation.
    Linear,
    /// No interpolation; the provoking vertex value is used.
    Flat,
}

/// Where within a pixel the interpolation sample is taken.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InterpolationSampling {
    /// The pixel center.
    Center,
    /// The centroid of the covered area.
    Centroid,
    /// The sample position.
    Sample,
}

/// A complete interpolation specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShaderInterpolation {
    /// The interpolation mode.
    pub mode: InterpolationMode,
    /// The interpolation sampling position.
    pub sampling: InterpolationSampling,
}

/// One location in a shader's input or output interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShaderLocationInterface {
    /// The Fluxel logical location.
    pub location: ShaderLocation,
    /// The numeric class carried at this location.
    pub numeric_type: ShaderNumericType,

    /// The component count, in `1..=4`.
    pub components: u8,

    /// The interpolation, canonicalized to explicit values for vertex outputs
    /// and fragment inputs. Usually `None` for vertex inputs and fragment
    /// outputs.
    pub interpolation: Option<ShaderInterpolation>,
}

/// One resource an entry point requires.
///
/// This reuses the binding vocabulary rather than defining a parallel
/// `ShaderBindingKind`, so shader reflection and bind-group layouts cannot drift
/// into two systems. `dynamic_offset` is not a shader semantic and therefore
/// does not appear here; it is a bind-group layout decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderResourceRequirement {
    /// The logical bind group this resource belongs to.
    pub group: BindGroupIndex,
    /// The logical slot within that group.
    pub slot: BindingSlotId,
    /// The resource semantics the shader requires.
    pub kind: BindingKind,
    /// The fixed resource count in shader code.
    pub count: BindingCount,
}

/// A canonical, duplicate-free description of one entry point's interface.
///
/// Builder methods collect entries; validation of uniqueness and canonical
/// ordering happens when a device creates the shader. The RHI never silently
/// sorts, merges, or picks one of two duplicates.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderInterface {
    resources: Vec<ShaderResourceRequirement>,
    inputs: Vec<ShaderLocationInterface>,
    outputs: Vec<ShaderLocationInterface>,
    writes_position: bool,
    writes_frag_depth: bool,
    writes_sample_mask: bool,
}

impl ShaderInterface {
    /// An empty interface.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one resource requirement.
    pub fn with_resource(mut self, requirement: ShaderResourceRequirement) -> Self {
        self.resources.push(requirement);
        self
    }

    /// Adds one input location.
    pub fn with_input(mut self, input: ShaderLocationInterface) -> Self {
        self.inputs.push(input);
        self
    }

    /// Adds one output location.
    pub fn with_output(mut self, output: ShaderLocationInterface) -> Self {
        self.outputs.push(output);
        self
    }

    /// Declares whether this entry point writes clip position.
    pub fn with_writes_position(mut self, value: bool) -> Self {
        self.writes_position = value;
        self
    }

    /// Declares whether this entry point writes fragment depth.
    pub fn with_writes_frag_depth(mut self, value: bool) -> Self {
        self.writes_frag_depth = value;
        self
    }

    /// Declares whether this entry point writes a sample mask.
    pub fn with_writes_sample_mask(mut self, value: bool) -> Self {
        self.writes_sample_mask = value;
        self
    }

    /// The declared resource requirements, in declaration order.
    pub fn resources(&self) -> &[ShaderResourceRequirement] {
        &self.resources
    }

    /// The declared inputs, in declaration order.
    pub fn inputs(&self) -> &[ShaderLocationInterface] {
        &self.inputs
    }

    /// The declared outputs, in declaration order.
    pub fn outputs(&self) -> &[ShaderLocationInterface] {
        &self.outputs
    }

    /// Whether this entry point writes clip position.
    pub fn writes_position(&self) -> bool {
        self.writes_position
    }

    /// Whether this entry point writes fragment depth.
    pub fn writes_frag_depth(&self) -> bool {
        self.writes_frag_depth
    }

    /// Whether this entry point writes a sample mask.
    pub fn writes_sample_mask(&self) -> bool {
        self.writes_sample_mask
    }

    /// The interface in canonical order, or the first violation found.
    ///
    /// Canonical form means resources sorted by `(group, slot)` with unique
    /// pairs, and inputs and outputs each sorted by ascending location with
    /// unique locations. Input and output locations are separate namespaces and
    /// may legitimately share a numeric value.
    pub fn canonical(&self) -> RhiResult<CanonicalShaderInterface<'_>> {
        for window in self.resources.windows(2) {
            let previous = (window[0].group.get(), window[0].slot.get());
            let current = (window[1].group.get(), window[1].slot.get());
            if current == previous {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "shader interface declares resource {}:{} twice",
                        current.0, current.1
                    ),
                ));
            }
            if current < previous {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "shader interface resources are not ordered by (group, slot)",
                ));
            }
        }
        check_locations(&self.inputs, "input")?;
        check_locations(&self.outputs, "output")?;
        Ok(CanonicalShaderInterface { interface: self })
    }
}

fn check_locations(locations: &[ShaderLocationInterface], what: &'static str) -> RhiResult<()> {
    for window in locations.windows(2) {
        if window[0].location == window[1].location {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "shader interface declares {} location {} twice",
                    what,
                    window[0].location.get()
                ),
            ));
        }
        if window[1].location < window[0].location {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("shader interface {what}s are not ordered by ascending location"),
            ));
        }
    }
    Ok(())
}

/// Proof that a [`ShaderInterface`] is in canonical form.
///
/// Holding this value is the only way to enter interface comparison, so a
/// non-canonical interface can never be compared two different ways.
#[derive(Clone, Copy, Debug)]
pub struct CanonicalShaderInterface<'a> {
    interface: &'a ShaderInterface,
}

impl<'a> CanonicalShaderInterface<'a> {
    /// The validated interface.
    pub fn interface(self) -> &'a ShaderInterface {
        self.interface
    }
}

/// The workgroup shape and shared-memory demand of a compute entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeWorkgroupRequirements {
    /// The workgroup X dimension.
    pub x: u32,
    /// The workgroup Y dimension.
    pub y: u32,
    /// The workgroup Z dimension.
    pub z: u32,
    /// The total invocation count, which must equal `x * y * z` without
    /// overflow.
    pub total_invocations: u32,
    /// The workgroup or shared memory bytes this entry point needs.
    pub workgroup_storage_bytes: u64,
}

impl ComputeWorkgroupRequirements {
    /// Builds the requirements, deriving nothing.
    pub fn new(
        x: u32,
        y: u32,
        z: u32,
        total_invocations: u32,
        workgroup_storage_bytes: u64,
    ) -> Self {
        Self {
            x,
            y,
            z,
            total_invocations,
            workgroup_storage_bytes,
        }
    }

    /// Whether the invocation count is consistent with the dimensions.
    ///
    /// Checked before pipeline creation so a reflection mismatch is a portable
    /// refusal rather than a backend or validation-layer surprise.
    pub fn is_consistent(&self) -> bool {
        self.x
            .checked_mul(self.y)
            .and_then(|value| value.checked_mul(self.z))
            == Some(self.total_invocations)
    }
}

/// What a shader entry point requires from the device.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderRequirements {
    required_features: Vec<OptionalFeature>,
    limit_requirements: Vec<LimitRequirement>,
    compute_workgroup: Option<ComputeWorkgroupRequirements>,
}

impl ShaderRequirements {
    /// No requirements.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requires `feature`.
    pub fn require_feature(mut self, feature: OptionalFeature) -> Self {
        self.required_features.push(feature);
        self.required_features.sort_unstable();
        self.required_features.dedup();
        self
    }

    /// Requires the stated limit.
    pub fn require_limit(mut self, requirement: LimitRequirement) -> Self {
        self.limit_requirements.push(requirement);
        self.limit_requirements.sort_unstable();
        self.limit_requirements.dedup();
        self
    }

    /// Records the compute workgroup shape. Required for a compute entry point
    /// and absent for vertex and fragment entry points.
    pub fn with_compute_workgroup(mut self, requirements: ComputeWorkgroupRequirements) -> Self {
        self.compute_workgroup = Some(requirements);
        self
    }

    /// The required optional features, sorted and deduplicated.
    pub fn required_features(&self) -> &[OptionalFeature] {
        &self.required_features
    }

    /// The limit requirements, sorted and deduplicated.
    pub fn limit_requirements(&self) -> &[LimitRequirement] {
        &self.limit_requirements
    }

    /// The compute workgroup requirements, when this is a compute entry point.
    pub fn compute_workgroup(&self) -> Option<ComputeWorkgroupRequirements> {
        self.compute_workgroup
    }

    /// Validates the stage-specific presence rule and the workgroup arithmetic.
    pub fn validate_for(&self, stage: ShaderStage) -> RhiResult<()> {
        match (stage, self.compute_workgroup) {
            (ShaderStage::Compute, None) => Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "compute entry point must declare workgroup requirements",
            )),
            (ShaderStage::Vertex | ShaderStage::Fragment, Some(_)) => Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "only a compute entry point may declare workgroup requirements",
            )),
            (ShaderStage::Compute, Some(requirements)) => {
                if requirements.x == 0 || requirements.y == 0 || requirements.z == 0 {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "compute workgroup dimensions must be non-zero",
                    ));
                }
                if !requirements.is_consistent() {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "compute total_invocations must equal x * y * z",
                    ));
                }
                Ok(())
            }
            (ShaderStage::Vertex | ShaderStage::Fragment, None) => Ok(()),
        }
    }

    /// Whether every limit requirement holds for `limit`.
    pub(crate) fn limit_satisfied_by(&self, limit: impl Fn(LimitKey) -> Option<u64>) -> bool {
        self.limit_requirements.iter().all(|requirement| {
            let (key, value, at_least) = match *requirement {
                LimitRequirement::AtLeast { key, value } => (key, value, true),
                LimitRequirement::AtMost { key, value } => (key, value, false),
            };
            match limit(key) {
                Some(actual) if at_least => actual >= value,
                Some(actual) => actual <= value,
                None => false,
            }
        })
    }
}

/// The content-address and provenance key supplied by an artifact producer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtifactHash(pub [u8; 32]);

/// The version of the artifact producer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtifactProducerVersion {
    /// The breaking producer component.
    pub major: u16,
    /// The compatible producer component.
    pub minor: u16,
}

/// The stable identity of the toolchain that produced an artifact.
///
/// It must not contain a temporary path, a process address, or a build-directory
/// identity.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtifactProducerId(pub String);

/// Where an executable-only artifact may be replayed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExecutableReplayAcceptanceScope {
    /// The artifact is not an acceptable replay input.
    Denied,

    /// Replay may accept the executable only on this backend kind.
    SameBackend(BackendKind),
}

/// A language from which the toolchain can regenerate code for other backends.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PortableShaderLanguage {
    /// WGSL source.
    Wgsl,
    /// SPIR-V module.
    SpirV,
    /// A Fluxel intermediate representation.
    FluxelIr,
}

/// Where an artifact's code came from and how far it can travel.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShaderProvenance {
    /// The toolchain can regenerate shader code for other backends from this.
    PortableSource {
        /// The retained source language.
        language: PortableShaderLanguage,
        /// The retained source or IR bytes.
        bytes: Arc<[u8]>,

        /// Compiler options that affect semantics.
        ///
        /// Keys must be unique, sorted lexicographically, and free of temporary
        /// absolute paths and process addresses.
        compiler_options: Vec<(String, String)>,
    },

    /// Only the current executable or code, with no cross-backend source.
    ExecutableOnly {
        /// The explicit replay acceptance scope for this artifact.
        replay_acceptance: ExecutableReplayAcceptanceScope,
    },
}

impl ShaderProvenance {
    /// Whether the compiler options are in canonical form.
    pub fn compiler_options_are_canonical(&self) -> bool {
        match self {
            Self::PortableSource {
                compiler_options, ..
            } => {
                let keys_are_sorted_and_unique = compiler_options
                    .windows(2)
                    .all(|window| window[0].0 < window[1].0);
                let keys_are_not_build_local = compiler_options.iter().all(|(key, value)| {
                    !looks_build_local(key) && !looks_build_local(value)
                });
                keys_are_sorted_and_unique && keys_are_not_build_local
            }
            Self::ExecutableOnly { .. } => true,
        }
    }
}

/// A conservative check for values that would make an artifact hash
/// machine-local, and would therefore break provenance and replay equality.
fn looks_build_local(value: &str) -> bool {
    let bytes = value.as_bytes();
    let has_windows_root = bytes.len() >= 3 && bytes[1] == b':' && (bytes[2] == b'\\' || bytes[2] == b'/');
    has_windows_root
        || value.starts_with("\\\\")
        || value.starts_with("0x")
        || value.contains("/tmp/")
        || value.contains("/home/")
        || value.contains("/Users/")
}

/// A shader artifact with pipeline specialization closed.
///
/// A P0 artifact may not require a value at pipeline creation time: a WGSL
/// override without a default, an unresolved Vulkan specialization constant, or
/// a required Metal function constant belongs to a future specialization API.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderArtifact {
    /// A diagnostic label. It never participates in the content hash.
    pub label: Label,

    /// The stage this entry point belongs to.
    pub stage: ShaderStage,
    /// The entry point name.
    pub entry_point: String,

    /// The consumable code.
    pub code: ShaderCode,

    /// The lowering ABI this code was built against.
    pub abi_version: ShaderAbiVersion,

    /// The canonical semantic interface.
    pub interface: ShaderInterface,
    /// What the entry point requires from the device.
    pub requirements: ShaderRequirements,

    /// Where the code came from.
    pub provenance: ShaderProvenance,

    /// The content-address key supplied by the producer.
    pub content_hash: ArtifactHash,
    /// The producing toolchain.
    pub producer: ArtifactProducerId,
    /// The producing toolchain version.
    pub producer_version: ArtifactProducerVersion,
}

impl ShaderArtifact {
    /// Builds an artifact with a fail-closed default provenance.
    ///
    /// The default is [`ShaderProvenance::ExecutableOnly`] with
    /// [`ExecutableReplayAcceptanceScope::Denied`], so an artifact that never
    /// declares provenance can never be replayed on another backend by
    /// accident.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stage: ShaderStage,
        entry_point: impl Into<String>,
        code: ShaderCode,
        abi_version: ShaderAbiVersion,
        interface: ShaderInterface,
        requirements: ShaderRequirements,
        content_hash: ArtifactHash,
        producer: ArtifactProducerId,
        producer_version: ArtifactProducerVersion,
    ) -> Self {
        Self {
            label: Label::none(),
            stage,
            entry_point: entry_point.into(),
            code,
            abi_version,
            interface,
            requirements,
            provenance: ShaderProvenance::ExecutableOnly {
                replay_acceptance: ExecutableReplayAcceptanceScope::Denied,
            },
            content_hash,
            producer,
            producer_version,
        }
    }

    /// Sets the diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label::new(label);
        self
    }

    /// Declares the artifact's provenance.
    pub fn with_provenance(mut self, provenance: ShaderProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    /// Whether the artifact's stage, ABI, and provenance rules are internally
    /// consistent, independent of any device.
    pub fn is_well_formed(&self) -> RhiResult<()> {
        if self.entry_point.is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "shader artifact entry point must be non-empty",
            ));
        }
        if self.producer.0.is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "shader artifact producer id must be non-empty",
            ));
        }
        self.requirements.validate_for(self.stage)?;
        self.interface.canonical()?;
        for location in self.interface.inputs() {
            validate_location(location)?;
        }
        for location in self.interface.outputs() {
            validate_location(location)?;
        }
        match self.stage {
            ShaderStage::Vertex => {
                if !self.interface.writes_position() {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "vertex entry point must write clip position",
                    ));
                }
            }
            ShaderStage::Fragment => {
                if self.interface.writes_position() {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "fragment entry point must not write clip position",
                    ));
                }
            }
            ShaderStage::Compute => {
                if !self.interface.inputs().is_empty() || !self.interface.outputs().is_empty() {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "compute entry point must not declare stage IO",
                    ));
                }
                if self.interface.writes_position()
                    || self.interface.writes_frag_depth()
                    || self.interface.writes_sample_mask()
                {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "compute entry point must not declare fragment outputs",
                    ));
                }
            }
        }
        if !self.provenance.compiler_options_are_canonical() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "shader artifact compiler options are not canonical",
            ));
        }
        Ok(())
    }
}

fn validate_location(location: &ShaderLocationInterface) -> RhiResult<()> {
    if location.components == 0 || location.components > 4 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "shader location components must be in 1..=4",
        ));
    }
    if location.numeric_type != ShaderNumericType::Float32 {
        let is_flat = matches!(
            location.interpolation,
            Some(ShaderInterpolation {
                mode: InterpolationMode::Flat,
                ..
            })
        );
        if !is_flat {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "integer shader stage IO must use flat interpolation",
            ));
        }
    }
    Ok(())
}

/// A validated entry point on a device.
#[derive(Clone)]
pub struct ShaderModule {
    inner: Arc<dyn ShaderModuleBackend>,
}

impl core::fmt::Debug for ShaderModule {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ShaderModule")
            .field("id", &self.id())
            .field("stage", &self.stage())
            .finish_non_exhaustive()
    }
}

impl ShaderModule {
    pub(crate) fn new(inner: Arc<dyn ShaderModuleBackend>) -> Self {
        Self { inner }
    }

    /// The backing this handle retains, for the device's retirement registry.
    pub(crate) fn backing(&self) -> Arc<dyn ShaderModuleBackend> {
        Arc::clone(&self.inner)
    }

    /// This module's object id.
    pub fn id(&self) -> ObjectId {
        self.inner.id()
    }

    /// The device identity this module belongs to.
    pub fn device_identity(&self) -> super::DeviceIdentity {
        self.inner.device_identity()
    }

    /// The artifact this module was created from.
    pub fn artifact(&self) -> &ShaderArtifact {
        self.inner.artifact()
    }

    /// The stage of this module's single entry point.
    pub fn stage(&self) -> ShaderStage {
        self.artifact().stage
    }
}

/// The backend half of a [`ShaderModule`].
pub(crate) trait ShaderModuleBackend: Send + Sync + 'static {
    /// This module's object id.
    fn id(&self) -> ObjectId;

    /// The device identity this module belongs to.
    fn device_identity(&self) -> super::DeviceIdentity;

    /// The artifact this module was created from.
    fn artifact(&self) -> &ShaderArtifact;
}
