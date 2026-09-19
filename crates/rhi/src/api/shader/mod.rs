//! Shader code, entry-point interface, requirements, provenance, and the
//! artifact a device accepts (specification section 19).
//!
//! Section 19 opens by refusing to conflate three layers that a "portable shader
//! blob" would merge, and this module is organised along those three:
//!
//! ```text
//! ShaderCode        a code form the current backend can consume
//! ShaderInterface   the portable semantics of one entry point
//! ShaderProvenance  whether Capture/Replay can regenerate code elsewhere
//! ```
//!
//! The RHI is not a shader cross-compiler. It never converts one of those code
//! forms into another, which is why there is no `ShaderCode::Portable`: SPIR-V
//! may be toolchain-portable *provenance* without any browser being able to
//! execute it. Whether the current device accepts a code form is a question for
//! `EnabledCapabilities::shader_acceptance`, not for a property of `ShaderCode`.
//!
//! # What this module owns
//!
//! - The stage vocabulary and its mask ([`ShaderStage`], [`ShaderStages`]).
//! - The code forms ([`ShaderCode`], [`GlslProfile`]) and the lowering ABI
//!   version an artifact declares ([`ShaderAbiVersion`]).
//! - The 32-bit stage IO vocabulary ([`ShaderNumericType`], [`ShaderLocation`],
//!   [`ShaderInterpolation`], [`ShaderLocationInterface`]).
//! - The entry-point description ([`ShaderInterface`]) and what it needs from the
//!   device ([`ShaderRequirements`], [`ComputeWorkgroupRequirements`]).
//! - Artifact provenance and identity ([`ArtifactHash`],
//!   [`ArtifactProducerId`], [`ArtifactProducerVersion`], [`ShaderProvenance`]).
//! - The artifact itself ([`ShaderArtifact`]) and the created module
//!   ([`ShaderModule`]).
//!
//! # What this module deliberately does not own
//!
//! - *Binding* vocabulary. Section 19.5 makes an entry point's resource
//!   requirements reuse [`crate::api::binding`]'s kinds and counts directly, so
//!   that reflection and layout cannot drift into two systems. There is no
//!   `ShaderBindingKind` here, and adding one would be the failure the section
//!   names.
//! - *Binding capability*. Section 19.7 forbids repeating it in
//!   [`ShaderRequirements`]: it is answered by asking
//!   [`BindingSupportQuery`](crate::api::binding::BindingSupportQuery) about each
//!   [`ShaderInterface`] resource.
//! - Logical group/slot to native register/index lowering, location to native
//!   semantic mapping, and the argument-buffer/root-signature strategy. All three
//!   stay backend/toolchain-private (section 19.3).
//! - Cross-compilation and specialization-constant resolution. A P0 artifact must
//!   have pipeline specialization *closed* (section 19.9); supplying values at
//!   pipeline creation belongs to a future capability family.
//!
//! # The rules this module decides
//!
//! `validate_shader_artifact` is the portable half of `create_shader`'s
//! validation list (section 19.10). It runs before any backend is touched, and it
//! refuses rather than normalizes: section 19.6 makes a duplicate or
//! non-canonical interface a rejection, explicitly *not* something the RHI may
//! silently sort, merge, or choose from.
//!
//! The remaining entries of that list are device facts — `ArtifactAcceptance`,
//! `ShaderAbiVersion` acceptance, and binding support — so they are answered by
//! the device façade and arrive here as parameters.
//!
//! # Files
//!
//! One part of the chapter per file, with this file holding only the declarations
//! and the re-exports of the *public* types:
//!
//! ```text
//! mod.rs          declarations and re-exports, no rule of its own
//! vocabulary.rs   sections 19.1-19.4, stages, code forms, and locations
//! requirements.rs section 19.5-19.7, what an entry point requires
//! artifact.rs     sections 19.8-19.9, provenance and the created module
//! validation.rs   sections 19.6-19.10, the portable validators
//! ```
//!
//! The re-export list is the module's public contract with the rest of the crate:
//! `crate::api::shader::X` names every public type this chapter defines.
//!
//! `validate_shader_artifact` and `stage_mask` are crate-private and are not
//! re-exported. Each submodule stays crate-visible rather than private because the
//! validators it owns are crate-private entry points of their own: a `pub(crate)
//! use` of one would be an unused import in a non-test build, since the device
//! verbs that call it are not written yet, and this module does not carry lint
//! attributes as a substitute for a caller — so a crate-internal caller names the
//! file that defines the item.

pub(crate) mod artifact;
pub(crate) mod requirements;
pub(crate) mod validation;
pub(crate) mod vocabulary;

pub use artifact::{
    ArtifactHash, ArtifactProducerId, ArtifactProducerVersion, ExecutableReplayAcceptanceScope,
    PortableShaderLanguage, ShaderArtifact, ShaderModule, ShaderProvenance,
};
pub use requirements::{
    ComputeWorkgroupRequirements, ShaderInterface, ShaderRequirements, ShaderResourceRequirement,
};
pub use vocabulary::{
    ArtifactAcceptance, GlslProfile, InterpolationMode, InterpolationSampling, ShaderAbiVersion,
    ShaderCode, ShaderInterpolation, ShaderLocation, ShaderLocationInterface, ShaderNumericType,
    ShaderStage, ShaderStages,
};
