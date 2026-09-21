//! Section 19.8's acceptance decision: may this device use this artifact?
//!
//! One rule, in one place, over device facts and nothing else. It answers the
//! question [`EnabledCapabilities::shader_acceptance`](crate::api::capability::EnabledCapabilities::shader_acceptance)
//! is declared to answer, and it is what `Device::create_shader` consults before a
//! backend ever sees the artifact (section 19.10).
//!
//! # What it is not
//!
//! Not the portable half of the check. `validate_shader_artifact` owns the
//! canonicality rules, and section 19.6 makes those a *producer's* mistake, which
//! is why they are an [`RhiError`](crate::api::error::RhiError) and this is a
//! verdict. The two overlap in exactly one place, deliberately: the binding
//! support of each interface resource appears on section 19.10's list *and* in
//! this rule, because a caller may ask this question on its own — the answer is a
//! device fact, and a caller deciding whether to build an artifact does not want a
//! `Result` about the artifact's spelling along with it.
//!
//! Nor is it a compile. Section 19.10 says a runtime GLSL/MSL compile error is
//! returned through `RhiError` plus a `DiagnosticEvent`; this rule decides whether
//! the device will *try*, and it cannot decide more than that, because for a source
//! form there is no compiler here to ask.
//!
//! # Why the backend kind is not consulted
//!
//! Section 19.2 names a canonical form per backend, which invites the shortest
//! possible implementation: remember the [`BackendKind`] and match the code form
//! against it. That is refused, and section 6.3 is why — the kind a device reports
//! is "diagnostics, selection provenance, and tooling UI only", explicitly **not**
//! a capability oracle. Two devices of one kind can differ about which forms they
//! consume (a desktop GL context and a GLES context are the same backend kind and
//! different answers), so the accepted forms are recorded facts, exactly like every
//! other capability, and this rule reads the record.
//!
//! [`BackendKind`]: crate::api::platform::provider::BackendKind

use crate::api::capability::CapabilityFacts;
use crate::api::platform::requirements::{LimitKey, LimitRequirement, OptionalFeature};
use crate::api::shader::artifact::ShaderArtifact;
use crate::api::shader::requirements::ShaderRequirements;
use crate::api::shader::vocabulary::{
    AcceptedCodeForm, ArtifactAcceptance, IMPLEMENTED_ABI, ShaderAbiVersion, ShaderStage,
};

/// Whether `facts` may use `artifact`.
///
/// The order is section 19.8's list, and it is not arbitrary: each step answers a
/// question that is *cheaper to state* than the next and that makes the next
/// meaningless if it fails. A code form the device does not consume cannot be
/// judged for ABI, a missing feature cannot be judged for limits, and the interface
/// is only worth asking about once the code is something the device would read at
/// all. Answering `LimitExceeded` for an artifact whose form is unsupported would
/// name a remedy the caller cannot use.
///
/// The first failing step is the answer, which is why the enum has six members and
/// not one boolean: the refusals point at different remedies.
pub(crate) fn decide(facts: &CapabilityFacts, artifact: &ShaderArtifact) -> ArtifactAcceptance {
    if !facts.accepts_code_form(AcceptedCodeForm::of(&artifact.code)) {
        return ArtifactAcceptance::UnsupportedCodeFormat;
    }

    if !abi_accepts(IMPLEMENTED_ABI, artifact.abi_version) {
        return ArtifactAcceptance::UnsupportedAbi;
    }

    // Section 19.1 makes compute legal only when the feature is enabled. This is a
    // separate question from the artifact's own `required_features`, because a
    // producer is not obliged to spell out a feature the *stage* already implies,
    // and a compute entry point that did not would otherwise be accepted on a
    // device with no compute.
    if artifact.stage == ShaderStage::Compute && !facts.has_feature(OptionalFeature::Compute) {
        return ArtifactAcceptance::MissingFeature;
    }

    // Native code form and trusted passthrough are intentionally separate: a
    // backend may consume SPIR-V/DXIL/etc. through its normal validated compiler
    // path without accepting caller-asserted reflection.  Only the explicit unsafe
    // admission boundary requires this feature.
    if artifact.is_trusted_passthrough() && !facts.has_feature(OptionalFeature::PassthroughShaders)
    {
        return ArtifactAcceptance::MissingFeature;
    }

    if artifact
        .requirements
        .required_features()
        .iter()
        .any(|feature| !facts.has_feature(*feature))
    {
        return ArtifactAcceptance::MissingFeature;
    }

    if artifact
        .requirements
        .cooperative_matrices()
        .iter()
        .any(|requirement| !facts.supports_cooperative_matrix(*requirement))
    {
        return ArtifactAcceptance::MissingFeature;
    }

    if artifact
        .requirements
        .builtins()
        .iter()
        .any(|builtin| !facts.has_feature(ShaderRequirements::builtin_feature(*builtin)))
        || (!artifact.requirements.cooperative_matrices().is_empty()
            && !facts.has_feature(OptionalFeature::CooperativeMatrix))
    {
        return ArtifactAcceptance::MissingFeature;
    }

    if artifact
        .requirements
        .limit_requirements()
        .iter()
        .any(|requirement| !limit_satisfied(facts, *requirement))
    {
        return ArtifactAcceptance::LimitExceeded;
    }

    // The local workgroup is part of the compute shader's executable contract,
    // not an advisory requirement a producer may omit.  Unlike a generic stated
    // limit, each of these four facts is mandatory once Compute is enabled: an
    // absent fact cannot prove that native lowering accepts this local shape.
    if artifact.stage == ShaderStage::Compute {
        let Some(shape) = artifact.interface.compute_workgroup_size() else {
            return ArtifactAcceptance::LimitExceeded;
        };
        let limits = [
            (LimitKey::MaxComputeWorkgroupSizeX, u64::from(shape.x)),
            (LimitKey::MaxComputeWorkgroupSizeY, u64::from(shape.y)),
            (LimitKey::MaxComputeWorkgroupSizeZ, u64::from(shape.z)),
            (
                LimitKey::MaxComputeInvocationsPerWorkgroup,
                shape.invocation_count(),
            ),
        ];
        if limits
            .into_iter()
            .any(|(key, required)| facts.limit(key).is_none_or(|actual| actual < required))
        {
            return ArtifactAcceptance::LimitExceeded;
        }
    }

    for resource in artifact.interface.resources() {
        if !facts
            .binding_support(&resource.binding_query(artifact.stage))
            .is_supported()
        {
            return ArtifactAcceptance::InterfaceUnsupported;
        }
    }

    ArtifactAcceptance::Accepted
}

/// Whether a device implementing `implemented` may use an artifact lowered against
/// `artifact`.
///
/// A different major is always refused, and so is a newer minor: this build cannot
/// know the rules of an ABI it does not implement, and section 19.3 makes that the
/// whole point of the version — an old executable must not be silently interpreted
/// using new rules. An older minor is accepted, which is what "compatible-extension
/// component" means.
///
/// A free function rather than a method on [`ShaderAbiVersion`]: section 19.3
/// freezes that type at a struct with two public fields and no methods, and the
/// comparison is the device layer's rule rather than the vocabulary's.
///
/// Crate-visible so that the contract tests can drive both directions of it. The
/// verdict can only show one: [`IMPLEMENTED_ABI`] is `1.0`, so "an older minor is
/// accepted" has no instance to observe through `decide` until this build's ABI
/// minor moves off zero.
pub(crate) fn abi_accepts(implemented: ShaderAbiVersion, artifact: ShaderAbiVersion) -> bool {
    implemented.major == artifact.major && artifact.minor <= implemented.minor
}

/// Whether the device's answer satisfies one stated requirement.
///
/// The requirement's *variant* carries the direction of the bound, per section
/// 7.4's two spellings: `AtLeast` compares the device's value from below, `AtMost`
/// from above. The key's own direction — which is what
/// [`LimitKey::larger_is_stronger`] classifies — therefore does not enter the
/// comparison; it is what tells a *producer* which spelling expresses "at least
/// this capable" for a given key, and a producer that spells it the other way has
/// asked for a device that is at most that capable, which is a statement this rule
/// takes at its word.
///
/// **A limit the contract does not define is not a refusal.** `facts.limit` answers
/// `None` there, and the comparison is skipped rather than failed: answering
/// `LimitExceeded` would assert that the device's value is below the requirement
/// when the device has no value at all — a false statement about the device, and a
/// false diagnosis for a caller, since the remedy it names is "use a smaller
/// requirement" when the real state is "this contract does not say". The direction
/// is the same one [`DeviceLimits::get`](crate::api::capability::DeviceLimits::get)
/// documents ("a limit the capability contract does not define imposes no
/// requirement") and the same one
/// `PipelineInterface`'s ceiling check reads for an absent
/// [`binding_limit`](crate::api::capability::EnabledCapabilities::binding_limit).
/// It has a real case: the DX12 port records 20 of the 27 keys and the other 7 are
/// limits Direct3D 12 does not state, so refusing on absence would refuse legal
/// artifacts for the shape of this backend's table rather than for anything about
/// the device.
fn limit_satisfied(facts: &CapabilityFacts, requirement: LimitRequirement) -> bool {
    let Some(actual) = facts.limit(requirement.key()) else {
        return true;
    };
    match requirement {
        LimitRequirement::AtLeast { value, .. } => actual >= value,
        LimitRequirement::AtMost { value, .. } => actual <= value,
    }
}
