//! Capability tests (specification section 7).
//!
//! These are the review instrument for the capability interface, not a
//! conformance suite: there is no hardware behind them and none of them may be
//! presented as GPU evidence. What they can answer is whether a caller can
//! compare what an adapter offered against what a device enabled without learning
//! two vocabularies, and whether the answers section 7.2 makes load-bearing —
//! `None` for an unavailable format, `None` for an inapplicable binding limit —
//! are expressible at all.
//!
//! The interning tests at the end are a second kind of instrument: they check the
//! canonical encoding by its two stated properties — equal facts give equal ids,
//! and *different* facts give different ones — rather than by comparing bytes
//! against a golden value. A golden value would pin the encoding without checking
//! either property, and would be rewritten to match whatever the code did.

use crate::api::binding::{BindingCount, BindingKind, BindingSupport, BindingSupportQuery};
use crate::api::capability::{
    AvailableCapabilities, CapabilityFacts, CapabilityFingerprint, EnabledCapabilities,
};
use crate::api::format::{
    FormatFacts, StorageAccessSupport, TextureFormat, TextureSupport, TextureSupportLimits,
    TextureSupportQuery,
};
use crate::api::platform::requirements::{LimitKey, OptionalFeature};
use crate::api::resource::buffer::{BufferSupport, BufferSupportQuery, BufferUsage};
use crate::api::resource::route::{RouteCapabilities, RouteQuery, RouteSupport};
use crate::api::resource::texture::{Extent3d, TextureDimension, TextureUsage};
use crate::api::shader::ShaderStages;
use crate::api::submission::{
    LaneDependencyRoute, LaneWorkDomains, SubmissionCapabilities, SubmissionLaneClass,
    SubmissionLaneId, SubmissionLaneInfo,
};

/// Both levels answer the same vocabulary, which is what lets a caller write one
/// comparison instead of two.
#[test]
fn available_and_enabled_answer_the_same_questions() {
    let mut facts = CapabilityFacts::empty();
    facts.record_feature(OptionalFeature::Compute);
    facts.record_limit(LimitKey::MaxBufferSize, 1 << 28);
    let available = AvailableCapabilities::from_facts(facts);

    let mut facts = CapabilityFacts::empty();
    facts.record_feature(OptionalFeature::Compute);
    facts.record_limit(LimitKey::MaxBufferSize, 1 << 26);
    let enabled = enabled_from(facts);

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
    let mut facts = CapabilityFacts::empty();
    facts.record_format(
        TextureFormat::R8Unorm,
        FormatFacts::new(
            TextureFormat::R8Unorm,
            StorageAccessSupport::new(true, true, true),
        ),
    );
    let available = AvailableCapabilities::from_facts(facts);
    let enabled = enabled_capabilities();

    assert!(available.format(TextureFormat::R8Unorm).is_some());
    assert!(
        enabled.format(TextureFormat::R8Unorm).is_none(),
        "a format the adapter reported may still be unavailable to the device"
    );
}

/// The shapes of an absent record, side by side.
///
/// This is the review instrument for the rule in the module documentation, and it
/// is deliberately one test rather than three: the difference between the shapes
/// *is* the rule, and a reader who meets only one of them in isolation will read
/// the other two as inconsistencies in the code rather than as a decision.
#[test]
fn an_absent_record_answers_according_to_whether_its_key_space_is_enumerable() {
    let enabled = enabled_capabilities();

    // `format` answers `Option`, and `None` is a real answer. `FormatFacts` is
    // opaque and carries no `PartialEq`, so the question is asked as "is it
    // there", not "does it equal".
    assert!(enabled.format(TextureFormat::R8Unorm).is_none());

    // `buffer_support` is keyed on the sixty-four usage masks, so enumeration can
    // be complete, so an absent entry is a hole in it — and a hole has no honest
    // answer. Answering `Supported` would permit an operation the device cannot
    // perform and answering `Unsupported` would hide the bug behind a
    // driver-shaped symptom, so neither variant is produced and the query says so.
    let outcome = std::panic::catch_unwind(|| {
        let _ = enabled.buffer_support(&BufferSupportQuery::new(BufferUsage::STORAGE));
    });
    assert!(
        outcome.is_err(),
        "a buffer query over an enumerable key space must not invent an answer"
    );

    // The other three carry an unbounded component in their key, so no
    // enumeration could have been complete and an absent entry cannot be a hole.
    // Each answers its own negative.
    assert!(
        !enabled
            .texture_support(&TextureSupportQuery::new(
                TextureDimension::D2,
                TextureFormat::R8Unorm,
                TextureUsage::SAMPLED,
                1,
            ))
            .is_supported()
    );
    assert_eq!(
        enabled.binding_support(&BindingSupportQuery {
            visibility: ShaderStages::FRAGMENT,
            kind: BindingKind::Sampler {
                kind: crate::api::binding::SamplerKind::Filtering,
            },
            count: BindingCount::One,
            dynamic_offset: false,
        }),
        BindingSupport::Unsupported
    );
    assert!(!enabled.route(&RouteQuery::BufferToBuffer).is_supported());
}

/// The price of the unbounded-key shape, asserted rather than left implied.
///
/// Where the key space cannot be enumerated, a backend that forgot to record a
/// route and a device that genuinely lacks it produce the same answer. That is the
/// trade the module documentation states, and this test exists so that it stays a
/// stated trade: if a future change makes the two distinguishable again — by
/// narrowing the key until it *is* enumerable — this test fails and the change
/// gets read as the improvement it would be.
#[test]
fn a_recorded_negative_and_an_unrecorded_question_are_one_answer_where_the_key_is_unbounded() {
    let mut facts = CapabilityFacts::empty();
    facts.record_route(RouteQuery::BufferToBuffer, RouteSupport::Unsupported);
    let recorded = enabled_from(facts);
    let unrecorded = enabled_capabilities();

    assert!(!recorded.route(&RouteQuery::BufferToBuffer).is_supported());
    assert!(!unrecorded.route(&RouteQuery::BufferToBuffer).is_supported());
}

/// A texture is supported only when its shape *and* its declared views are.
///
/// The two halves of the answer come from different places — the recorded table
/// and the pairwise relation — so this is the test that would catch one of them
/// being dropped from the conjunction.
#[test]
fn a_texture_answer_is_the_conjunction_of_its_shape_and_its_views() {
    let shape = TextureSupportQuery::new(
        TextureDimension::D2,
        TextureFormat::R8Unorm,
        TextureUsage::SAMPLED,
        1,
    );
    let mut facts = CapabilityFacts::empty();
    facts.record_texture_support(
        &shape,
        TextureSupport::Supported(TextureSupportLimits::new(Extent3d::d1(4096), 1, 1)),
    );
    facts.record_view_compatibility(TextureFormat::R8Unorm, TextureFormat::R8Uint);
    let enabled = enabled_from(facts);

    // Shape alone: supported.
    assert!(enabled.texture_support(&shape).is_supported());

    // A declared view format the pairwise relation does not list takes the whole
    // answer down, rather than being reported separately: section 13.2 makes the
    // view intent a creation-time fact, so a texture that cannot be viewed as
    // asked is a texture that cannot be created as asked.
    assert!(
        !enabled
            .texture_support(&shape.clone().with_view_format(TextureFormat::Rgba8Unorm))
            .is_supported()
    );

    // A declared view format the relation does list leaves it standing.
    assert!(
        enabled
            .texture_support(&shape.with_view_format(TextureFormat::R8Uint))
            .is_supported()
    );
}

/// A sample count the device did not record is an answer, not an absent fact.
///
/// The count is a `u32`, so it is the clearest case of why the texture key cannot
/// be enumerated: no backend can record a row for every integer a caller might
/// name. `texture_support` is where the number lives, and it must refuse rather
/// than panic.
#[test]
fn an_unrecorded_sample_count_is_refused_rather_than_asserted() {
    let mut facts = CapabilityFacts::empty();
    facts.record_texture_support(
        &TextureSupportQuery::new(
            TextureDimension::D2,
            TextureFormat::R8Unorm,
            TextureUsage::SAMPLED,
            1,
        ),
        TextureSupport::Supported(TextureSupportLimits::new(Extent3d::d1(4096), 1, 1)),
    );
    let enabled = enabled_from(facts);

    assert!(
        !enabled
            .texture_support(&TextureSupportQuery::new(
                TextureDimension::D2,
                TextureFormat::R8Unorm,
                TextureUsage::SAMPLED,
                7,
            ))
            .is_supported()
    );
}

/// A recorded positive answer carries its facts back out, so a caller that asked
/// once does not have to ask again per size.
#[test]
fn a_recorded_positive_route_answer_carries_its_capabilities() {
    let mut facts = CapabilityFacts::empty();
    facts.record_route(
        RouteQuery::BufferToBuffer,
        RouteSupport::Supported(RouteCapabilities::new(None, None)),
    );
    let enabled = enabled_from(facts);

    let answer = enabled.route(&RouteQuery::BufferToBuffer);
    assert!(answer.is_supported());
    assert!(answer.capabilities().is_some());
}

/// The same, for a texture query whose answer is a ceiling.
#[test]
fn a_recorded_positive_texture_answer_carries_its_maxima() {
    let query = TextureSupportQuery::new(
        TextureDimension::D2,
        TextureFormat::R8Unorm,
        TextureUsage::SAMPLED,
        1,
    );

    let mut facts = CapabilityFacts::empty();
    facts.record_texture_support(
        &query,
        TextureSupport::Supported(TextureSupportLimits::new(Extent3d::d1(8192), 1, 1)),
    );
    let enabled = enabled_from(facts);

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
    let query = BufferSupportQuery::new(BufferUsage::STORAGE);

    let mut facts = CapabilityFacts::empty();
    facts.record_buffer_support(query.usage(), BufferSupport::Unsupported);
    let enabled = enabled_from(facts);

    assert!(!enabled.buffer_support(&query).is_supported());
}

/// `limits()` and `limit()` are two views of one set, not two answers.
#[test]
fn the_limits_view_agrees_with_the_single_key_query() {
    let mut facts = CapabilityFacts::empty();
    facts.record_limit(LimitKey::MaxBufferSize, 4096);
    facts.record_limit(LimitKey::MinUniformBufferOffsetAlignment, 256);
    let enabled = enabled_from(facts);

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

    // Note how the equality is obtained: not by both call sites passing the same
    // constant, but by both handing over facts that encode identically and letting
    // the interning table decide. The constant form this test used to have would
    // have passed even if interning did nothing at all.
    assert_eq!(a.compatibility_id(), b.compatibility_id());
    assert_eq!(a.fingerprint(), b.fingerprint());

    // A device created under a different capability contract yields a different
    // token and a different fingerprint: the two are computed from the same bytes,
    // so they move together and neither can stand in for the other.
    let c = enabled_from(a_different_contract());
    assert_ne!(a.compatibility_id(), c.compatibility_id());
    assert_ne!(c.fingerprint(), a.fingerprint());
}

/// A device built under a fixed contract, for the tests above.
fn enabled_capabilities() -> EnabledCapabilities {
    enabled_from(CapabilityFacts::empty())
}

/// A contract that differs from [`enabled_capabilities`] in exactly one fact.
fn a_different_contract() -> CapabilityFacts {
    let mut facts = CapabilityFacts::empty();
    facts.record_feature(OptionalFeature::Compute);
    facts
}

/// Wraps a filled record the way a completed device request would.
///
/// The empty lane set is the caller's, not the constructor's: section 7.2's base
/// lane guarantee is checked by `SubmissionCapabilities`' own validator, which the
/// device-request path calls, and a constructor that also enforced it would put
/// the rule in two places. These tests are asking about capability facts, not
/// about lanes, so they hand over none.
fn enabled_from(facts: CapabilityFacts) -> EnabledCapabilities {
    EnabledCapabilities::from_facts(facts, SubmissionCapabilities::new(Vec::new()))
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
    let caps = enabled_capabilities();

    assert!(
        !caps.texture_view_format_compatible(
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb
        ),
        "an unrecorded pair is not compatible, whatever its texel size"
    );

    let mut facts = CapabilityFacts::empty();
    facts.record_view_compatibility(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb);
    let caps = enabled_from(facts);

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
    let mut facts = CapabilityFacts::empty();
    facts.record_view_compatibility(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb);
    let caps = enabled_from(facts);

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
// Interning.
//
// Section 7.1 gives the compatibility id two properties, and these are them.
// They are contract tests with no GPU behind them: passing them says nothing
// about any backend's enumeration, only about what the portable layer does with
// the record a backend hands it.
// ---------------------------------------------------------------------------

/// Insertion order must not reach the id.
///
/// This is the property the sort in `CapabilityFacts::canonical_bytes` exists for,
/// and it is not hypothetical. The facts live in `HashMap`s and `HashSet`s, whose
/// iteration order is unspecified and differs between runs, so two devices that
/// recorded exactly the same facts can walk them in different orders. Without the
/// sort their ids would differ across runs of the *same* program — a failure no
/// single run can observe, which is why it is tested by construction here rather
/// than left to a real-device test to happen to catch.
#[test]
fn facts_recorded_in_different_orders_intern_to_the_same_id() {
    let mut forwards = CapabilityFacts::empty();
    forwards.record_feature(OptionalFeature::Compute);
    forwards.record_feature(OptionalFeature::SamplerAnisotropy);
    forwards.record_limit(LimitKey::MaxBufferSize, 4096);
    forwards.record_limit(LimitKey::MinUniformBufferOffsetAlignment, 256);
    forwards.record_route(RouteQuery::BufferToBuffer, RouteSupport::Unsupported);
    forwards.record_view_compatibility(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb);

    let mut backwards = CapabilityFacts::empty();
    backwards.record_view_compatibility(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb);
    backwards.record_route(RouteQuery::BufferToBuffer, RouteSupport::Unsupported);
    backwards.record_limit(LimitKey::MinUniformBufferOffsetAlignment, 256);
    backwards.record_limit(LimitKey::MaxBufferSize, 4096);
    backwards.record_feature(OptionalFeature::SamplerAnisotropy);
    backwards.record_feature(OptionalFeature::Compute);

    assert_eq!(
        enabled_from(forwards).compatibility_id(),
        enabled_from(backwards).compatibility_id(),
        "the id must depend on the facts, not on the order a backend recorded them in"
    );
}

/// A difference anywhere in the facts must reach the id.
///
/// This is the opposite failure to the one above, and the more dangerous one: an
/// encoding that drops a field makes two genuinely different contracts
/// indistinguishable, so a compiled graph keyed on the id runs against a device it
/// was not compiled for. Each case below differs from the empty contract in
/// exactly one place, and every section of the record is covered by one of them —
/// a section nobody exercises is a section whose omission goes unnoticed.
#[test]
fn a_difference_anywhere_in_the_facts_yields_a_different_id() {
    let baseline = enabled_capabilities().compatibility_id();
    let differ = |mutate: &dyn Fn(&mut CapabilityFacts)| {
        let mut facts = CapabilityFacts::empty();
        mutate(&mut facts);
        enabled_from(facts).compatibility_id()
    };

    assert_ne!(
        baseline,
        differ(&|facts| facts.record_feature(OptionalFeature::Compute)),
        "a feature"
    );
    assert_ne!(
        baseline,
        differ(&|facts| facts.record_limit(LimitKey::MaxBufferSize, 4096)),
        "a limit value"
    );
    assert_ne!(
        baseline,
        differ(&|facts| facts.record_format(
            TextureFormat::R8Unorm,
            FormatFacts::new(
                TextureFormat::R8Unorm,
                StorageAccessSupport::new(true, false, false)
            )
        )),
        "a format fact"
    );
    assert_ne!(
        baseline,
        differ(
            &|facts| facts.record_buffer_support(BufferUsage::STORAGE, BufferSupport::Unsupported)
        ),
        "a buffer support answer"
    );
    assert_ne!(
        baseline,
        differ(&|facts| facts.record_texture_support(
            &TextureSupportQuery::new(
                TextureDimension::D2,
                TextureFormat::R8Unorm,
                TextureUsage::SAMPLED,
                1
            ),
            TextureSupport::Unsupported
        )),
        "a texture support answer"
    );
    assert_ne!(
        baseline,
        differ(&|facts| facts.record_binding_support(
            BindingSupportQuery {
                visibility: ShaderStages::FRAGMENT,
                kind: BindingKind::Sampler {
                    kind: crate::api::binding::SamplerKind::Filtering,
                },
                count: BindingCount::One,
                dynamic_offset: false,
            },
            BindingSupport::Unsupported
        )),
        "a binding support answer"
    );
    assert_ne!(
        baseline,
        differ(&|facts| facts.record_binding_limit(
            crate::api::shader::ShaderStage::Fragment,
            crate::api::binding::BindingLimitClass::Samplers,
            16
        )),
        "a binding-count ceiling"
    );
    assert_ne!(
        baseline,
        differ(&|facts| facts.record_route(RouteQuery::BufferToBuffer, RouteSupport::Unsupported)),
        "a recorded negative route, which is a fact and not an absence"
    );
    assert_ne!(
        baseline,
        differ(&|facts| facts
            .record_view_compatibility(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb)),
        "a view-compatible pair"
    );
}

/// The same, for a difference *inside* a key rather than between keys.
///
/// `min_size` is the case that would be easiest to lose: two queries with the same
/// visibility, kind tag, and count differ only in a payload field of the kind, and
/// an encoding that wrote the discriminant and stopped would conflate them. A
/// device that can express a 64-byte uniform binding is not thereby stating it can
/// express a 128-byte one.
#[test]
fn two_queries_differing_only_inside_a_key_yield_different_ids() {
    let with_min_size = |min_size: u64| {
        let mut facts = CapabilityFacts::empty();
        facts.record_binding_support(
            BindingSupportQuery {
                visibility: ShaderStages::FRAGMENT,
                kind: BindingKind::UniformBuffer { min_size },
                count: BindingCount::One,
                dynamic_offset: false,
            },
            BindingSupport::Supported,
        );
        enabled_from(facts).compatibility_id()
    };

    assert_ne!(with_min_size(64), with_min_size(128));
}

/// A key's *set-valued* field must not depend on the order the caller listed it
/// in either.
///
/// `TextureSupportQuery` is the one key that carries a collection, and its
/// `view_formats` are a `Vec` the caller builds, so two callers asking the same
/// question with the same formats in different orders are asking the same
/// question. This is the same property as the first test above, one level down.
/// A query's alternate view formats are not part of what enumeration records, so
/// two queries that differ only in that list describe one contract.
///
/// This replaces a test that asserted the *encoder* sorted the list. That sort
/// existed because the whole query was the recorded key and a `Vec` in a key makes
/// two spellings of one question compare unequal; the field is not in the key any
/// more, so the property holds by construction and there is no sort left to test.
/// What is worth asserting is the property itself, and it is asserted from the
/// outside — through the id — rather than by reading the key type.
#[test]
fn a_queries_view_format_list_does_not_reach_the_contract() {
    let query = |first: TextureFormat, second: TextureFormat| {
        TextureSupportQuery::new(
            TextureDimension::D2,
            TextureFormat::R8Unorm,
            TextureUsage::SAMPLED,
            1,
        )
        .with_view_format(first)
        .with_view_format(second)
    };
    let id_of = |query: &TextureSupportQuery| {
        let mut facts = CapabilityFacts::empty();
        facts.record_texture_support(query, TextureSupport::Unsupported);
        enabled_from(facts).compatibility_id()
    };
    let without = {
        let mut facts = CapabilityFacts::empty();
        facts.record_texture_support(
            &TextureSupportQuery::new(
                TextureDimension::D2,
                TextureFormat::R8Unorm,
                TextureUsage::SAMPLED,
                1,
            ),
            TextureSupport::Unsupported,
        );
        enabled_from(facts).compatibility_id()
    };

    assert_eq!(
        id_of(&query(
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb
        )),
        id_of(&query(
            TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Rgba8Unorm
        )),
        "the same two view formats in the other order are the same question"
    );
    assert_eq!(
        id_of(&query(
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb
        )),
        without,
        "declaring view formats asks the same shape question as declaring none"
    );
}

/// The fingerprint is a 32-byte digest of the same bytes the id is interned
/// under, so it changes exactly when the id does.
///
/// Stated as a test because the two are easy to accidentally decouple — computing
/// the fingerprint over a `Debug` rendering, say, or over the facts in iteration
/// order — and a fingerprint that tracked something other than the contract would
/// be worse than useless as artifact provenance: it would look authoritative and
/// be wrong.
#[test]
fn the_fingerprint_moves_exactly_when_the_id_does() {
    let with = |mutate: &dyn Fn(&mut CapabilityFacts)| {
        let mut facts = CapabilityFacts::empty();
        mutate(&mut facts);
        let caps = enabled_from(facts);
        (caps.compatibility_id(), caps.fingerprint())
    };

    let plain = with(&|_| {});
    let featured = with(&|facts| facts.record_feature(OptionalFeature::Compute));

    assert_ne!(plain.0, featured.0);
    assert_ne!(plain.1, featured.1);
    assert_eq!(with(&|_| {}), plain, "the same facts give the same pair");
    assert_ne!(
        plain.1,
        CapabilityFingerprint([0u8; 32]),
        "an empty contract still has a digest of its own, not a zeroed one"
    );
}

/// Section 7.1 calls the token the interning of "canonical *EnabledCapabilities*
/// semantics", and `submission()` is declared on `EnabledCapabilities` and on
/// nothing else — so a lane layout is part of the contract the id stands for.
///
/// The consequence is what makes this worth a test rather than a reading: section
/// 7.1 keys `CompiledGraph` correctness reuse on the id, and a compiled plan names
/// the lanes it submits to. Two devices that agreed on every query fact but laid
/// out lanes differently would otherwise intern to one id, and a plan interned
/// under the first would be reused against a device that cannot accept its
/// batches.
///
/// The same facts are used on both sides, so the only thing the two ids can differ
/// by is the lane snapshot.
#[test]
fn two_devices_differing_only_in_their_lanes_have_different_ids() {
    let one_lane = enabled_over(lanes(&[(
        SubmissionLaneId::new(0),
        SubmissionLaneClass::General,
        LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY),
    )]));
    let two_lanes = enabled_over(lanes(&[
        (
            SubmissionLaneId::new(0),
            SubmissionLaneClass::General,
            LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY),
        ),
        (
            SubmissionLaneId::new(1),
            SubmissionLaneClass::Compute,
            LaneWorkDomains::COMPUTE,
        ),
    ]));

    assert_ne!(
        one_lane.compatibility_id(),
        two_lanes.compatibility_id(),
        "a device offering a second lane is not the same contract as one that does not"
    );
    assert_ne!(
        one_lane.fingerprint(),
        two_lanes.fingerprint(),
        "the fingerprint covers the same bytes as the id, so it moves with it"
    );
}

/// A lane's *identity* is its [`SubmissionLaneId`], not its position in the list
/// the backend happened to enumerate in.
///
/// `SubmissionCapabilities::lanes` documents itself as reporting "the order it
/// reported them", so the order is observable — which is exactly why the encoding
/// must not inherit it. Two devices offering the same lanes in a different
/// reported order offer the same contract.
#[test]
fn lanes_reported_in_a_different_order_intern_to_the_same_id() {
    let raster = (
        SubmissionLaneId::new(0),
        SubmissionLaneClass::General,
        LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY),
    );
    let compute = (
        SubmissionLaneId::new(1),
        SubmissionLaneClass::Compute,
        LaneWorkDomains::COMPUTE,
    );

    assert_eq!(
        enabled_over(lanes(&[raster, compute])).compatibility_id(),
        enabled_over(lanes(&[compute, raster])).compatibility_id(),
    );
}

/// The cross-lane routes are a reported fact like any other, and they are not
/// derivable from the lane list: two lanes say which domains each accepts, not
/// whether a native dependency primitive exists between them. An id that ignored
/// them would call a device that cannot order two lanes equivalent to one that can.
#[test]
fn a_reported_dependency_route_changes_the_id() {
    let plain = two_split_lanes();
    let mut routed = plain.clone();
    routed.record_dependency_route(
        SubmissionLaneId::new(0),
        SubmissionLaneId::new(1),
        LaneDependencyRoute::Gpu,
    );

    assert_ne!(
        enabled_over(plain).compatibility_id(),
        enabled_over(routed).compatibility_id(),
    );
}

/// The mirror of the lane-order test, for the route table.
///
/// `record_dependency_route` retains-then-pushes, so the table's order is the
/// order a backend discovered routes in. That is a discovery artifact like the
/// lane list's order, and the encoding must not inherit it.
#[test]
fn routes_recorded_in_a_different_order_intern_to_the_same_id() {
    let mut forward = two_split_lanes();
    forward.record_dependency_route(
        SubmissionLaneId::new(0),
        SubmissionLaneId::new(1),
        LaneDependencyRoute::Gpu,
    );
    forward.record_dependency_route(
        SubmissionLaneId::new(1),
        SubmissionLaneId::new(0),
        LaneDependencyRoute::Collapse,
    );

    let mut backward = two_split_lanes();
    backward.record_dependency_route(
        SubmissionLaneId::new(1),
        SubmissionLaneId::new(0),
        LaneDependencyRoute::Collapse,
    );
    backward.record_dependency_route(
        SubmissionLaneId::new(0),
        SubmissionLaneId::new(1),
        LaneDependencyRoute::Gpu,
    );

    assert_eq!(
        enabled_over(forward).compatibility_id(),
        enabled_over(backward).compatibility_id(),
    );
}

/// Two lanes split by domain, with no route recorded between them.
fn two_split_lanes() -> SubmissionCapabilities {
    lanes(&[
        (
            SubmissionLaneId::new(0),
            SubmissionLaneClass::Graphics,
            LaneWorkDomains::RASTER,
        ),
        (
            SubmissionLaneId::new(1),
            SubmissionLaneClass::Transfer,
            LaneWorkDomains::COPY,
        ),
    ])
}

/// Builds a lane set from `(id, class, domains)` triples.
fn lanes(
    entries: &[(SubmissionLaneId, SubmissionLaneClass, LaneWorkDomains)],
) -> SubmissionCapabilities {
    SubmissionCapabilities::new(
        entries
            .iter()
            .map(|(id, class, domains)| SubmissionLaneInfo::new(*id, *class, *domains))
            .collect(),
    )
}

/// An enabled contract over empty facts and the given lanes.
///
/// Empty facts on purpose in the lane tests: the question is whether the lane
/// snapshot reaches the encoding, and holding the query facts at "nothing" is what
/// makes the lane snapshot the only difference between the two sides.
fn enabled_over(submission: SubmissionCapabilities) -> EnabledCapabilities {
    EnabledCapabilities::from_facts(CapabilityFacts::empty(), submission)
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
