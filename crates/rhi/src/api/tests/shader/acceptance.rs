//! Section 19.8 acceptance tests: the device's verdict on one artifact.
//!
//! [`decide`](crate::api::shader::acceptance) is a rule over two inputs — the
//! device's recorded facts and one artifact — so it is decidable without a
//! backend, and these tests hand it both. What they are for is the part of the
//! rule the specification leaves to the implementation:
//!
//! * **The order decides which of several true refusals is reported.** Section
//!   19.8 lists six verdicts and the first failing step is the answer, because each
//!   one names a different remedy. A test that only ever breaks one rule at a time
//!   cannot tell a rule that short-circuits from one that does not, so one test
//!   below breaks two.
//! * **A limit the contract does not define is not a refusal.** The specification
//!   does not say, and the two readings are not equally honest: `LimitExceeded`
//!   asserts a comparison result, and there is no device value to compare against.
//!   That decision is pinned here so that it cannot be reversed by accident.
//! * **The accepted forms are a set, and the device is what answers.** The
//!   shortest implementation of section 19.2 is "match the form against the backend
//!   kind", which section 6.3 forbids. A test that a device may accept two forms at
//!   once is what keeps that shortcut from being reintroduced unnoticed.
//!
//! None of this needs a GPU, and none of it is GPU evidence.

use std::sync::Arc;

use super::*;
use crate::api::binding::vocabulary::BindableKind;
use crate::api::capability::{BindingSupportKey, CapabilityFacts, EnabledCapabilities};
use crate::api::shader::ArtifactAcceptance;
use crate::api::shader::acceptance::{abi_accepts, decide};
use crate::api::shader::vocabulary::{AcceptedCodeForm, IMPLEMENTED_ABI};
use crate::api::submission::SubmissionCapabilities;

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

/// The same artifact with a different code form.
///
/// The content hash is left stale on purpose: it is producer data that the
/// acceptance rule never reads — an acceptance verdict is not a cache lookup,
/// which is exactly what section 19.8's freeze rule is about.
fn using(mut artifact: ShaderArtifact, code: ShaderCode) -> ShaderArtifact {
    artifact.code = code;
    artifact
}

fn dxil() -> ShaderCode {
    ShaderCode::Dxil(Arc::from([0u8; 4].as_slice()))
}

/// A device that consumes one code form and nothing else.
fn device_consuming(form: AcceptedCodeForm) -> CapabilityFacts {
    let mut facts = CapabilityFacts::empty();
    facts.record_code_form(form);
    facts
}

fn enabled(facts: CapabilityFacts) -> EnabledCapabilities {
    EnabledCapabilities::from_facts(facts, SubmissionCapabilities::new(Vec::new()))
}

/// The verdict a device with these facts gives this artifact.
fn verdict(facts: CapabilityFacts, artifact: &ShaderArtifact) -> ArtifactAcceptance {
    enabled(facts).shader_acceptance(artifact)
}

fn shader_requirements() -> crate::api::shader::ShaderRequirements {
    crate::api::shader::ShaderRequirements::new()
}

/// A device ready to accept anything except the rule under test: it consumes
/// WGSL, has compute, and states no limits of its own.
fn permissive_device() -> CapabilityFacts {
    let mut facts = device_consuming(AcceptedCodeForm::Wgsl);
    facts.record_feature(OptionalFeature::Compute);
    facts
}

// ---------------------------------------------------------------------------
// Trusted native passthrough.
// ---------------------------------------------------------------------------

/// Native bytecode is an accepted code *form*, not an implicit trust grant.  The
/// explicit unsafe boundary needs its own enabled feature even where DXIL itself
/// can be consumed by the backend.
#[test]
fn trusted_passthrough_requires_its_separate_capability() {
    let artifact = unsafe {
        using(
            artifact_with(
                ShaderStage::Vertex,
                vertex_interface(),
                shader_requirements(),
            ),
            dxil(),
        )
        .assume_trusted_passthrough(crate::api::shader::PassthroughShaderProvenance::new(
            "trusted-test-toolchain",
            "reflection matched this exact bytecode",
        ))
    };

    let mut only_dxil = device_consuming(AcceptedCodeForm::Dxil);
    assert_eq!(
        verdict(only_dxil.clone(), &artifact),
        ArtifactAcceptance::MissingFeature,
        "native form acceptance alone must not enable caller-asserted reflection"
    );
    only_dxil.record_feature(OptionalFeature::PassthroughShaders);
    assert_eq!(verdict(only_dxil, &artifact), ArtifactAcceptance::Accepted);
}

/// The normal native-bytecode path stays usable without the trusted boundary:
/// backends may validate bytecode themselves, and must not be forced to advertise
/// passthrough merely because they accept a native form.
#[test]
fn native_code_without_the_explicit_trust_marker_needs_no_passthrough_feature() {
    let artifact = using(
        artifact_with(
            ShaderStage::Vertex,
            vertex_interface(),
            shader_requirements(),
        ),
        dxil(),
    );
    assert_eq!(
        verdict(device_consuming(AcceptedCodeForm::Dxil), &artifact),
        ArtifactAcceptance::Accepted
    );
    assert!(artifact.passthrough_provenance().is_none());
}

// ---------------------------------------------------------------------------
// The code form.
// ---------------------------------------------------------------------------

/// The form is a recorded fact, so a device that did not record it refuses it —
/// and refusing costs nothing, because nothing was claimed about that form.
#[test]
fn a_device_that_did_not_record_the_form_refuses_it() {
    let artifact = artifact_with(
        ShaderStage::Vertex,
        vertex_interface(),
        shader_requirements(),
    );

    assert_eq!(
        verdict(CapabilityFacts::empty(), &artifact),
        ArtifactAcceptance::UnsupportedCodeFormat
    );
    assert_eq!(
        verdict(device_consuming(AcceptedCodeForm::Wgsl), &artifact),
        ArtifactAcceptance::Accepted
    );
}

/// Two forms, one device. This is the test that keeps section 6.3's rule from
/// being quietly replaced by a backend-kind match: a device that consumes both a
/// source form and a binary form is nonsense for a *kind*, and ordinary for the GL
/// family and for any backend that compiles at runtime.
#[test]
fn the_accepted_forms_are_a_set_and_not_a_single_form() {
    let mut facts = device_consuming(AcceptedCodeForm::Glsl);
    facts.record_code_form(AcceptedCodeForm::GlslEs);

    let desktop = using(
        artifact_with(
            ShaderStage::Vertex,
            vertex_interface(),
            shader_requirements(),
        ),
        ShaderCode::Glsl {
            version: 450,
            profile: crate::api::shader::GlslProfile::Core,
            source: Arc::from("#version 450\nvoid main() {}"),
        },
    );
    let es = using(
        artifact_with(
            ShaderStage::Vertex,
            vertex_interface(),
            shader_requirements(),
        ),
        ShaderCode::GlslEs {
            version: 300,
            source: Arc::from("#version 300 es\nvoid main() {}"),
        },
    );
    let dxil_artifact = using(
        artifact_with(
            ShaderStage::Vertex,
            vertex_interface(),
            shader_requirements(),
        ),
        dxil(),
    );

    assert_eq!(
        verdict(facts.clone(), &desktop),
        ArtifactAcceptance::Accepted
    );
    assert_eq!(verdict(facts.clone(), &es), ArtifactAcceptance::Accepted);
    assert_eq!(
        verdict(facts, &dxil_artifact),
        ArtifactAcceptance::UnsupportedCodeFormat
    );
}

// ---------------------------------------------------------------------------
// The ABI.
// ---------------------------------------------------------------------------

/// Section 19.3's rule, in both directions: the compatible-extension component
/// may be older than the device's, and a different major is refused because this
/// build cannot know another lowering's rules.
///
/// The comparison is tested against a device version this build does not implement
/// (`1.2`), because the rule's two halves are not both reachable through
/// [`IMPLEMENTED_ABI`]: that constant is `1.0`, so no *older* minor exists for it to
/// accept and the compatible-extension half would go untested. The second half of
/// the test drives the same rule through the verdict, where the constant is what it
/// compares against.
#[test]
fn the_abi_accepts_older_minors_and_refuses_other_majors() {
    let device = ShaderAbiVersion { major: 1, minor: 2 };

    assert!(abi_accepts(device, device));
    assert!(abi_accepts(device, ShaderAbiVersion { major: 1, minor: 0 }));
    assert!(!abi_accepts(
        device,
        ShaderAbiVersion { major: 1, minor: 3 }
    ));
    assert!(!abi_accepts(
        device,
        ShaderAbiVersion { major: 2, minor: 0 }
    ));

    let build = |abi_version| {
        let mut artifact = artifact_with(
            ShaderStage::Vertex,
            vertex_interface(),
            shader_requirements(),
        );
        artifact.abi_version = abi_version;
        artifact
    };

    assert_eq!(
        verdict(permissive_device(), &build(IMPLEMENTED_ABI)),
        ArtifactAcceptance::Accepted
    );
    assert_eq!(
        verdict(
            permissive_device(),
            &build(ShaderAbiVersion {
                major: IMPLEMENTED_ABI.major,
                minor: IMPLEMENTED_ABI.minor + 1,
            })
        ),
        ArtifactAcceptance::UnsupportedAbi
    );
    assert_eq!(
        verdict(
            permissive_device(),
            &build(ShaderAbiVersion {
                major: IMPLEMENTED_ABI.major + 1,
                minor: 0,
            })
        ),
        ArtifactAcceptance::UnsupportedAbi
    );
}

// ---------------------------------------------------------------------------
// Features.
// ---------------------------------------------------------------------------

/// Section 19.1's stage rule is the device's, not the artifact's: a compute entry
/// point is refused on a device with no compute even when the artifact asks for no
/// features at all, because a producer is not obliged to spell out what the stage
/// implies.
#[test]
fn a_compute_entry_point_needs_the_compute_feature_even_when_it_asks_for_nothing() {
    let artifact = artifact_with(
        ShaderStage::Compute,
        ShaderInterface::new(),
        shader_requirements(),
    );
    assert!(artifact.requirements.required_features().is_empty());

    assert_eq!(
        verdict(device_consuming(AcceptedCodeForm::Wgsl), &artifact),
        ArtifactAcceptance::MissingFeature
    );
    assert_eq!(
        verdict(permissive_device(), &artifact),
        ArtifactAcceptance::Accepted
    );
}

#[test]
fn a_required_feature_that_is_not_enabled_refuses_the_artifact() {
    let requirements = shader_requirements().require_feature(OptionalFeature::SamplerAnisotropy);
    let artifact = artifact_with(ShaderStage::Vertex, vertex_interface(), requirements);

    assert_eq!(
        verdict(permissive_device(), &artifact),
        ArtifactAcceptance::MissingFeature
    );

    let mut facts = permissive_device();
    facts.record_feature(OptionalFeature::SamplerAnisotropy);
    assert_eq!(verdict(facts, &artifact), ArtifactAcceptance::Accepted);
}

/// Ray-query intersection tests and returning the hit triangle's vertices are
/// distinct native capabilities. Naming the latter as a builtin keeps shader
/// reflection from silently accepting it merely because AS bindings work.
#[test]
fn ray_hit_vertex_return_builtin_requires_its_specific_feature() {
    let requirements = shader_requirements()
        .require_builtin(crate::api::shader::ShaderBuiltin::RayHitVertexPosition);
    let artifact = artifact_with(ShaderStage::Compute, ShaderInterface::new(), requirements);

    assert_eq!(
        verdict(permissive_device(), &artifact),
        ArtifactAcceptance::MissingFeature
    );

    let mut facts = permissive_device();
    facts.record_feature(OptionalFeature::RayQuery);
    assert_eq!(
        verdict(facts, &artifact),
        ArtifactAcceptance::MissingFeature,
        "ordinary ray-query support must not imply hit-vertex return"
    );

    let mut facts = permissive_device();
    facts.record_feature(OptionalFeature::RayHitVertexReturn);
    assert_eq!(verdict(facts, &artifact), ArtifactAcceptance::Accepted);
}

// ---------------------------------------------------------------------------
// Limits.
// ---------------------------------------------------------------------------

#[test]
fn a_stated_limit_below_the_requirement_refuses_the_artifact() {
    let requirements = shader_requirements().require_limit(LimitRequirement::AtLeast {
        key: LimitKey::MaxBufferSize,
        value: 1 << 20,
    });
    let artifact = artifact_with(ShaderStage::Vertex, vertex_interface(), requirements);

    let mut small = permissive_device();
    small.record_limit(LimitKey::MaxBufferSize, (1 << 20) - 1);
    assert_eq!(verdict(small, &artifact), ArtifactAcceptance::LimitExceeded);

    let mut large = permissive_device();
    large.record_limit(LimitKey::MaxBufferSize, 1 << 20);
    assert_eq!(verdict(large, &artifact), ArtifactAcceptance::Accepted);
}

/// The direction the variant carries is the direction the comparison uses, so an
/// `AtMost` requirement is satisfied by a *smaller* device value. This is section
/// 7.4's whole reason for having two variants, and it is the one place a test can
/// show the comparison is not simply "bigger is better".
#[test]
fn an_at_most_requirement_is_satisfied_from_above() {
    let requirements = shader_requirements().require_limit(LimitRequirement::AtMost {
        key: LimitKey::MinUniformBufferOffsetAlignment,
        value: 256,
    });
    let artifact = artifact_with(ShaderStage::Vertex, vertex_interface(), requirements);

    let mut tight = permissive_device();
    tight.record_limit(LimitKey::MinUniformBufferOffsetAlignment, 64);
    assert_eq!(verdict(tight, &artifact), ArtifactAcceptance::Accepted);

    let mut loose = permissive_device();
    loose.record_limit(LimitKey::MinUniformBufferOffsetAlignment, 512);
    assert_eq!(verdict(loose, &artifact), ArtifactAcceptance::LimitExceeded);
}

/// A limit the contract does not define is not a refusal.
///
/// The alternative reading — refuse, because the requirement cannot be verified —
/// would answer `LimitExceeded`, which asserts that the device's value is below the
/// requirement. There is no device value, so that answer is false about the device
/// and names a remedy the caller cannot use. It also has a live case in this tree:
/// the DX12 port records 20 of the 27 keys, so refusing on absence would refuse
/// legal artifacts for the shape of one backend's table.
#[test]
fn a_limit_the_contract_does_not_define_is_not_a_refusal() {
    let requirements = shader_requirements().require_limit(LimitRequirement::AtLeast {
        key: LimitKey::MaxInterStageShaderVariables,
        value: 32,
    });
    let artifact = artifact_with(ShaderStage::Vertex, vertex_interface(), requirements);

    assert_eq!(
        permissive_device().limit(LimitKey::MaxInterStageShaderVariables),
        None
    );
    assert_eq!(
        verdict(permissive_device(), &artifact),
        ArtifactAcceptance::Accepted
    );
}

// ---------------------------------------------------------------------------
// The interface.
// ---------------------------------------------------------------------------

#[test]
fn a_resource_the_device_cannot_express_refuses_the_artifact() {
    let interface = vertex_interface().with_resource(resource(0, 0));
    let artifact = artifact_with(ShaderStage::Vertex, interface, shader_requirements());

    let mut no_storage = permissive_device();
    no_storage.record_binding_support(
        BindingSupportKey {
            visibility: ShaderStages::VERTEX,
            kind: BindableKind::UniformBuffer,
            array: false,
            runtime_sized: false,
            dynamic_offset: false,
        },
        BindingSupport::Unsupported,
    );
    assert_eq!(
        verdict(no_storage, &artifact),
        ArtifactAcceptance::InterfaceUnsupported
    );

    let mut yes_storage = permissive_device();
    yes_storage.record_binding_support(
        BindingSupportKey {
            visibility: ShaderStages::VERTEX,
            kind: BindableKind::UniformBuffer,
            array: false,
            runtime_sized: false,
            dynamic_offset: false,
        },
        BindingSupport::Supported,
    );
    assert_eq!(
        verdict(yes_storage, &artifact),
        ArtifactAcceptance::Accepted
    );
}

// ---------------------------------------------------------------------------
// The order.
// ---------------------------------------------------------------------------

/// The first failing step is the answer, and this artifact fails three.
///
/// The order matters to a caller because the verdicts are remedies: told
/// `UnsupportedCodeFormat`, a caller re-lowers the artifact; told `MissingFeature`,
/// it would ask for a different device. Reporting the last failure instead of the
/// first would send it to the wrong remedy.
#[test]
fn the_first_failing_step_is_the_verdict() {
    let requirements = shader_requirements()
        .require_feature(OptionalFeature::SamplerAnisotropy)
        .require_limit(LimitRequirement::AtLeast {
            key: LimitKey::MaxBufferSize,
            value: 1 << 40,
        });
    let interface = vertex_interface().with_resource(resource(0, 0));
    let artifact = artifact_with(ShaderStage::Vertex, interface, requirements);

    // Nothing recorded at all: the code form is the first step, so it is the
    // answer even though the feature, the limit and the binding would each fail.
    assert_eq!(
        verdict(CapabilityFacts::empty(), &artifact),
        ArtifactAcceptance::UnsupportedCodeFormat
    );

    // With the form accepted, the missing feature precedes the limit.
    assert_eq!(
        verdict(device_consuming(AcceptedCodeForm::Wgsl), &artifact),
        ArtifactAcceptance::MissingFeature
    );

    // With the feature enabled, the limit precedes the interface. The limit has to
    // be *stated* and too small to refuse here: an unstated one is not a refusal,
    // so leaving it out would fall through to the interface and test nothing about
    // the order.
    let mut facts = device_consuming(AcceptedCodeForm::Wgsl);
    facts.record_feature(OptionalFeature::SamplerAnisotropy);
    facts.record_limit(LimitKey::MaxBufferSize, (1 << 40) - 1);
    assert_eq!(verdict(facts, &artifact), ArtifactAcceptance::LimitExceeded);

    // With a limit that fits, the interface is the last thing that can refuse.
    let mut facts = device_consuming(AcceptedCodeForm::Wgsl);
    facts.record_feature(OptionalFeature::SamplerAnisotropy);
    facts.record_limit(LimitKey::MaxBufferSize, 1 << 41);
    assert_eq!(
        verdict(facts, &artifact),
        ArtifactAcceptance::InterfaceUnsupported
    );
}

/// The rule is reached through the frozen accessor, not around it.
///
/// `decide` takes the facts directly so that it is testable without a device, and
/// a crate-private shortcut like that is exactly where a public verb stops agreeing
/// with its own documentation. The two are compared here rather than trusted.
#[test]
fn the_public_accessor_answers_what_the_rule_answers() {
    let artifact = artifact_with(
        ShaderStage::Vertex,
        vertex_interface(),
        shader_requirements(),
    );
    let facts = device_consuming(AcceptedCodeForm::Wgsl);

    assert_eq!(
        enabled(facts.clone()).shader_acceptance(&artifact),
        decide(&facts, &artifact)
    );
}
