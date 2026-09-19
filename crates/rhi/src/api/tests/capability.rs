//! Capability tests (specification section 7).
//!
//! These are the review instrument for the capability interface, not a
//! conformance suite: there is no hardware behind them and none of them may be
//! presented as GPU evidence. What they can answer is whether a caller can
//! compare what an adapter offered against what a device enabled without learning
//! two vocabularies, and whether the answers section 7.2 makes load-bearing —
//! `None` for an unavailable format, `None` for an inapplicable binding limit —
//! are expressible at all.

use crate::api::capability::{
    AvailableCapabilities, CapabilityCompatibilityId, CapabilityFingerprint, EnabledCapabilities,
};
use crate::api::format::{
    FormatFacts, StorageAccessSupport, TextureFormat, TextureSupport, TextureSupportLimits,
    TextureSupportQuery,
};
use crate::api::platform::requirements::{LimitKey, OptionalFeature};
use crate::api::resource::buffer::{BufferSupport, BufferSupportQuery, BufferUsage};
use crate::api::resource::route::{RouteCapabilities, RouteQuery, RouteSupport};
use crate::api::resource::texture::{Extent3d, TextureDimension, TextureUsage};

/// Both levels answer the same vocabulary, which is what lets a caller write one
/// comparison instead of two.
#[test]
fn available_and_enabled_answer_the_same_questions() {
    let mut available = AvailableCapabilities::new();
    let mut enabled = enabled_capabilities();

    available.record_feature(OptionalFeature::Compute);
    available.record_limit(LimitKey::MaxBufferSize, 1 << 28);

    enabled.record_feature(OptionalFeature::Compute);
    enabled.record_limit(LimitKey::MaxBufferSize, 1 << 26);

    assert!(available.supports_feature(OptionalFeature::Compute));
    assert!(enabled.supports_feature(OptionalFeature::Compute));
    assert!(!available.supports_feature(OptionalFeature::BindingArrays));

    assert_eq!(available.limit(LimitKey::MaxBufferSize), Some(1 << 28));
    assert_eq!(enabled.limit(LimitKey::MaxBufferSize), Some(1 << 26));

    // A key neither level defines is `None` on both — a fact, not a failure.
    assert_eq!(available.limit(LimitKey::MaxTexture2dDimension), None);
    assert_eq!(enabled.limit(LimitKey::MaxTexture2dDimension), None);
}

/// Section 7.2's WebGPU case, which is why `format` returns `Option` where the
/// support queries return an enum: the adapter may say `Some` while the device
/// says `None`, and correctness follows the device.
#[test]
fn a_format_available_on_the_adapter_can_be_unavailable_on_the_device() {
    let mut available = AvailableCapabilities::new();
    let enabled = enabled_capabilities();

    available.record_format(
        TextureFormat::R8Unorm,
        FormatFacts::new(
            TextureFormat::R8Unorm,
            StorageAccessSupport::new(true, true, true),
        ),
    );

    assert!(available.format(TextureFormat::R8Unorm).is_some());
    assert!(
        enabled.format(TextureFormat::R8Unorm).is_none(),
        "a format the adapter reported may still be unavailable to the device"
    );
}

/// The two completeness rules from the module docs, side by side: an absent
/// *format* is a fact, an absent *support* record is a snapshot bug.
#[test]
fn an_absent_format_is_an_answer_and_an_absent_support_record_is_not() {
    let enabled = enabled_capabilities();

    // Absent format: an answer. `FormatFacts` is opaque and carries no
    // `PartialEq`, so the question is asked as "is it there", not "does it equal".
    assert!(enabled.format(TextureFormat::R8Unorm).is_none());

    // Absent route record: not an answer, and neither variant would be honest.
    let outcome = std::panic::catch_unwind(|| {
        let _ = enabled.route(&RouteQuery::BufferToBuffer);
    });
    assert!(
        outcome.is_err(),
        "a route query enumeration never answered must not produce a value"
    );
}

/// A recorded negative answer is distinguishable from an unrecorded question.
#[test]
fn a_recorded_negative_route_is_answered_without_panicking() {
    let mut enabled = enabled_capabilities();
    enabled.record_route(RouteQuery::BufferToBuffer, RouteSupport::Unsupported);

    assert!(!enabled.route(&RouteQuery::BufferToBuffer).is_supported());
}

/// A recorded positive answer carries its facts back out, so a caller that asked
/// once does not have to ask again per size.
#[test]
fn a_recorded_positive_route_answer_carries_its_capabilities() {
    let mut enabled = enabled_capabilities();
    enabled.record_route(
        RouteQuery::BufferToBuffer,
        RouteSupport::Supported(RouteCapabilities::new(None, None)),
    );

    let answer = enabled.route(&RouteQuery::BufferToBuffer);
    assert!(answer.is_supported());
    assert!(answer.capabilities().is_some());
}

/// The same, for a texture query whose answer is a ceiling.
#[test]
fn a_recorded_positive_texture_answer_carries_its_maxima() {
    let mut enabled = enabled_capabilities();
    let query = TextureSupportQuery::new(
        TextureDimension::D2,
        TextureFormat::R8Unorm,
        TextureUsage::SAMPLED,
        1,
    );
    enabled.record_texture_support(
        query.clone(),
        TextureSupport::Supported(TextureSupportLimits::new(Extent3d::d1(8192), 1, 1)),
    );

    let answer = enabled.texture_support(&query);
    assert!(answer.is_supported());
    assert_eq!(
        answer.limits().map(|l| l.max_extent().max_component()),
        Some(8192)
    );
}

/// A buffer query recorded as unsupported reports exactly that, which is the
/// negative answer the enum exists to carry.
#[test]
fn a_buffer_query_can_be_recorded_as_unsupported() {
    let mut enabled = enabled_capabilities();
    let query = BufferSupportQuery::new(BufferUsage::STORAGE);
    enabled.record_buffer_support(query, BufferSupport::Unsupported);

    assert!(!enabled.buffer_support(&query).is_supported());
}

/// `limits()` and `limit()` are two views of one set, not two answers.
#[test]
fn the_limits_view_agrees_with_the_single_key_query() {
    let mut enabled = enabled_capabilities();
    enabled.record_limit(LimitKey::MaxBufferSize, 4096);
    enabled.record_limit(LimitKey::MinUniformBufferOffsetAlignment, 256);

    assert_eq!(enabled.limits().get(LimitKey::MaxBufferSize), Some(4096));
    assert_eq!(
        enabled.limits().get(LimitKey::MaxBufferSize),
        enabled.limit(LimitKey::MaxBufferSize)
    );
    assert_eq!(enabled.limits().keys().count(), 2);
}

/// The compatibility token is evidence of equality and not an ordinal, and it is
/// not interchangeable with the fingerprint.
#[test]
fn the_compatibility_token_and_the_fingerprint_are_not_interchangeable() {
    let a = enabled_capabilities();
    let b = enabled_capabilities();

    assert_eq!(a.compatibility_id(), b.compatibility_id());

    // A device created under a different capability contract yields a different
    // token, and the fingerprint is a separate value no correctness path reads.
    let c = EnabledCapabilities::new(
        CapabilityCompatibilityId::new(7),
        CapabilityFingerprint([9u8; 32]),
    );
    assert_ne!(a.compatibility_id(), c.compatibility_id());
    assert_ne!(c.fingerprint().0, a.fingerprint().0);
}

/// A device built under a fixed contract, for the tests above.
fn enabled_capabilities() -> EnabledCapabilities {
    EnabledCapabilities::new(
        CapabilityCompatibilityId::new(1),
        CapabilityFingerprint([0u8; 32]),
    )
}

/// Section 8.5's warning, stated as a test: two formats of equal byte size are
/// not thereby interchangeable as base and view.
///
/// This is the case that makes the verb necessary rather than convenient. A
/// caller that reasoned from texel size would create a view the driver rejects,
/// and the rejection would arrive from a backend — which is the class of failure
/// section 3.1 exists to keep in the portable layer.
#[test]
fn equal_byte_size_does_not_imply_view_compatibility() {
    let mut caps = enabled_capabilities();

    assert!(
        !caps.texture_view_format_compatible(
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb
        ),
        "an unrecorded pair is not compatible, whatever its texel size"
    );

    caps.record_view_compatibility(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb);

    assert!(
        caps.texture_view_format_compatible(
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb
        )
    );
}

/// A recorded pair answers for that pair and no other.
///
/// The relation is per-pair device data, not a property of a format: recording
/// one compatible pair must not make an unrelated pair compatible, or the
/// function would be answering a question enumeration never asked.
#[test]
fn recording_one_view_pair_does_not_answer_for_another() {
    let mut caps = enabled_capabilities();
    caps.record_view_compatibility(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb);

    assert!(
        caps.texture_view_format_compatible(
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb
        )
    );
    assert!(
        !caps.texture_view_format_compatible(
            TextureFormat::Bgra8Unorm,
            TextureFormat::Bgra8UnormSrgb
        ),
        "an unrelated pair must not inherit the first pair's answer"
    );
}

// ---------------------------------------------------------------------------
// Shape tests.
//
// Compiled, never called. They keep the parts of section 7.2 whose vocabulary
// belongs to modules 03 and 05 compiling as realistic call sites while those
// modules are still being written: they name the parameter types rather than
// constructing values, so they prove the call shape is usable without guessing a
// variant name to do it.
// ---------------------------------------------------------------------------

#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_per_stage_binding_count(
    caps: &EnabledCapabilities,
    stage: crate::api::shader::ShaderStage,
    class: crate::api::binding::BindingLimitClass,
) {
    // Section 7.3 makes this the canonical source for a per-stage resource
    // count, and `None` means "inapplicable", which a caller has to be able to
    // tell apart from zero.
    if let Some(limit) = caps.binding_limit(stage, class) {
        let _ = limit;
    }
}

#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_binding_legality(
    caps: &EnabledCapabilities,
    query: &crate::api::binding::BindingSupportQuery,
) {
    let _ = caps.binding_support(query);
}

#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_shader_acceptance(
    caps: &EnabledCapabilities,
    artifact: &crate::api::shader::ShaderArtifact,
) {
    let _ = caps.shader_acceptance(artifact);
}

/// A shape test: the submission-capability accessor.
///
/// Restored once `api::submission::SubmissionCapabilities` existed. It was
/// commented rather than deleted while module 05 was unwritten, because the
/// accessor's *shape* — takes `&self`, returns a borrow, needs no extra
/// construction — is exactly what this test exists to check, and a caller
/// cannot check that against a type that does not compile.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_submission_lanes(caps: &EnabledCapabilities) {
    let _ = caps.submission();
}
