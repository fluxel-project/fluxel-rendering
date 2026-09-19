//! Sections 19.8-19.9: provenance, the artifact, and the created module.
//!
//! What an artifact may be regenerated from, the content hash the producer
//! computed, the toolchain identity, and the finished [`ShaderArtifact`] —
//! entry point, code, ABI, interface, provenance and hash in one value. The
//! created [`ShaderModule`] is here too, because it is an artifact plus the
//! device identity that accepted it.
//!
//! Not owned here: the interface and requirements the artifact carries (section
//! 19.5-19.7, in `requirements.rs`) and the rule that decides whether an artifact
//! is internally consistent (in `validation.rs`).

use core::fmt;
use std::sync::Arc;

use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::platform::provider::BackendKind;

use super::requirements::{ShaderInterface, ShaderRequirements};
use super::vocabulary::{ShaderAbiVersion, ShaderCode, ShaderStage};

/// The content-address/provenance key computed by the artifact producer.
///
/// A newtype with a public array field, unlike the identity tokens of section 3,
/// and section 19.8 says why in the same breath as it defines it: it is supplied
/// by the producer, it is a *candidate cache key*, and "correctness must not rely
/// only on equal hashes". Object and interface compatibility therefore still use
/// complete canonical semantic validation, which is what keeps this field from
/// being an identity a caller could forge.
///
/// The canonical content domain is fixed by section 19.8:
///
/// ```text
/// ArtifactProducerId + ArtifactProducerVersion
/// stage + entry_point + ShaderCode + ShaderAbiVersion
/// canonical ShaderInterface + ShaderRequirements
/// canonical ShaderProvenance
/// ```
///
/// `label` is explicitly excluded from it, as are temporary paths, process
/// addresses, and other build-local values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactHash(pub [u8; 32]);

/// The version of the toolchain that produced an artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactProducerVersion {
    /// Breaking component of the toolchain's own version.
    pub major: u16,
    /// Compatible-extension component of the toolchain's own version.
    pub minor: u16,
}

/// Stable identity of the toolchain that produced an artifact.
///
/// Section 19.8 requires this to be stable: it must not contain a temporary path,
/// a process address, or a build-directory identity. Together with
/// [`ArtifactProducerVersion`] it identifies the lowering contract that the
/// artifact was built against, which is what Replay needs in order to decide
/// whether the provenance can be regenerated.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactProducerId(pub String);

/// What a replay runtime may do with an executable-only artifact.
///
/// Section 19.10 requires the scope to be explicit, because a current device
/// accepting an executable says nothing about another backend being able to
/// replay it: an executable-only artifact carries no cross-backend source or IR to
/// regenerate from.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutableReplayAcceptanceScope {
    /// The artifact is not an acceptable replay input.
    Denied,

    /// Replay may accept the executable only on this backend kind.
    SameBackend(BackendKind),
}

/// Where the code in an artifact came from, and what may be regenerated from it.
///
/// The third of the three layers section 19 opens with. The question it answers is
/// not "is this good code" but "can a toolchain produce code for a *different*
/// device from this artifact".
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum ShaderProvenance {
    /// The toolchain can regenerate [`ShaderCode`] for other backends from this.
    PortableSource {
        /// The language the portable source is written in.
        language: PortableShaderLanguage,
        /// The portable source or IR itself.
        bytes: Arc<[u8]>,

        /// The compiler options the artifact was built with.
        ///
        /// A producer must canonicalize this list: keys unique, sorted
        /// lexicographically by key, and free of temporary absolute paths and
        /// process addresses. Section 19.8 makes a non-canonical list a
        /// `create_shader` rejection rather than something the RHI repairs,
        /// because the list feeds the canonical provenance encoding.
        compiler_options: Vec<(String, String)>,
    },

    /// Contains only the current executable/code, with no cross-backend source.
    ExecutableOnly {
        /// The explicit replay acceptance scope for this artifact.
        replay_acceptance: ExecutableReplayAcceptanceScope,
    },
}

/// A language a [`ShaderProvenance::PortableSource`] may be written in.
///
/// Distinct from [`ShaderCode`]'s variants even where a spelling repeats: this
/// names something a toolchain can *regenerate from*, while `ShaderCode` names
/// something the current device can *consume*. SPIR-V appears in both, which is
/// exactly the asymmetry section 19.2 warns about rather than a redundancy.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortableShaderLanguage {
    /// WGSL source.
    Wgsl,
    /// SPIR-V module.
    SpirV,
    /// Fluxel's own intermediate representation.
    FluxelIr,
}

/// One entry point, fully described, ready to be accepted or refused.
///
/// Every field is public because an artifact is data the toolchain produces and
/// the RHI consumes; there is no device fact in it, so there is nothing a caller
/// could claim that the RHI would have to verify against hardware.
///
/// A P0 artifact must have **pipeline specialization closed**: no WGSL required
/// override without a default, no unresolved Vulkan specialization constant, no
/// Metal required function constant (section 19.9). Supplying values at pipeline
/// creation belongs to a future capability family, and P0 does not carry a
/// half-complete constants map in anticipation of it.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ShaderArtifact {
    /// Diagnostic label. Excluded from the canonical hash domain (section 19.8).
    pub label: Label,

    /// The stage of this entry point.
    pub stage: ShaderStage,
    /// The entry-point name within the code.
    pub entry_point: String,

    /// The code, in a form the target device must be able to consume.
    pub code: ShaderCode,

    /// The lowering ABI this code was produced against.
    pub abi_version: ShaderAbiVersion,

    /// The portable semantics of this entry point.
    pub interface: ShaderInterface,
    /// What this entry point needs from the device.
    pub requirements: ShaderRequirements,

    /// What may be regenerated from this artifact.
    pub provenance: ShaderProvenance,

    /// The producer-computed content hash.
    pub content_hash: ArtifactHash,
    /// The toolchain that produced the artifact.
    pub producer: ArtifactProducerId,
    /// That toolchain's version.
    pub producer_version: ArtifactProducerVersion,
}

impl ShaderArtifact {
    /// Assembles an artifact from the facts the producer knows.
    ///
    /// Checks nothing: `validate_shader_artifact` is the check, and it is run by
    /// `create_shader` before the artifact reaches a backend.
    ///
    /// The provenance is initialized to the most restrictive value,
    /// [`ShaderProvenance::ExecutableOnly`] with
    /// [`ExecutableReplayAcceptanceScope::Denied`]. Section 19.9's constructor
    /// list has no provenance parameter and supplies only
    /// [`Self::with_provenance`] to set one, so this constructor has to choose
    /// something; it fails closed, because the alternative would grant a replay
    /// permission that no producer asked for.
    // Section 19.9 freezes this constructor with its nine parameters and no
    // builder for the fields it sets, so the clippy suggestion to fold them into
    // an argument struct is declined: that struct would be public API the
    // specification did not declare. `expect` rather than `allow`, for the same
    // reason as `SamplerDescriptor`: a suppression that cannot expire is a
    // suppression nobody re-checks.
    #[expect(
        clippy::too_many_arguments,
        reason = "section 19.9 freezes this constructor with nine parameters and no builder; folding them into a struct would add public API the specification did not declare"
    )]
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
            label: Label::default(),
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

    /// Attaches a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }

    /// Declares what may be regenerated from this artifact.
    pub fn with_provenance(mut self, provenance: ShaderProvenance) -> Self {
        self.provenance = provenance;
        self
    }
}

/// A created entry point on one device.
///
/// Opaque, cloneable, and identified by [`ObjectId`] plus the
/// [`DeviceIdentity`] that created it. It owns its artifact rather than borrowing
/// it, because a module's interface and provenance outlive the call that created
/// it: section 28.1 requires everything a pipeline needs to be re-describable from
/// the artifacts and interfaces it was built from.
#[derive(Clone)]
pub struct ShaderModule {
    id: ObjectId,
    device: DeviceIdentity,
    artifact: ShaderArtifact,
}

impl ShaderModule {
    /// Assembles a created module.
    ///
    /// Crate-private: section 3 gives identity to the object that created it, so
    /// only `Device::create_shader` may produce one. The verb itself waits on
    /// `api::platform`, which is declared after the capability module it depends
    /// on.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Device::create_shader calls this once api::platform is declared"
        )
    )]
    pub(crate) fn new(id: ObjectId, device: DeviceIdentity, artifact: ShaderArtifact) -> Self {
        Self {
            id,
            device,
            artifact,
        }
    }

    /// This module's process-local object ID.
    pub fn id(&self) -> ObjectId {
        self.id
    }

    /// The device that created this module.
    ///
    /// Section 3.3 makes this the only answer to a cross-device use: there is no
    /// implicit recompile or module transfer, so a module from another device is a
    /// refusal.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The artifact this module was created from.
    ///
    /// Still needed after creation: a pipeline re-describes its stages from their
    /// artifacts (section 28.1), and Replay reads the provenance from here.
    pub fn artifact(&self) -> &ShaderArtifact {
        &self.artifact
    }

    /// The stage of this module's entry point.
    ///
    /// A convenience over `module.artifact().stage`, and the only place the stage
    /// is decided: a module is exactly one artifact entry point (section 19.10),
    /// so there is no second stage to disagree with.
    pub fn stage(&self) -> ShaderStage {
        self.artifact.stage
    }
}

/// Prints portable identity only.
///
/// Written by hand rather than derived: section 19.10 declares
/// `#[derive(Clone)]` and no `Debug` on this handle, while section 28 declares
/// descriptors that *contain* a module and do derive `Debug` — the specification
/// is internally inconsistent here, and the resolution (defect D6 of the 0.16
/// plan) is that every public opaque handle implements `Debug` portably.
///
/// It prints the identity rather than the artifact because section 7.1 describes
/// an object by its identity, because the backend port will add a native field
/// that has no reason to be `Debug`, and because printing a native handle into a
/// log would leak it. `finish_non_exhaustive()` is what makes it honest that the
/// artifact is not shown — a caller who needs it calls [`ShaderModule::artifact`].
impl fmt::Debug for ShaderModule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ShaderModule")
            .field("id", &self.id)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}
