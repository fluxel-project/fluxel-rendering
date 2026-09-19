//! Tests for the compute domain: the image-unit mirror, what an invalidation
//! must forget, and what a refused call carries out.
//!
//! Every test runs the domain against the mock provider rather than a
//! hand-written double, because what is being tested is the *trace* -- which
//! calls were emitted, how many, and in what order -- and a double written for
//! these tests would answer with whatever trace the test expected.  The mock is
//! the same provider the Layer 1 contract suites run against, so a change in
//! Layer 1's own call surface shows up here as a failing trace rather than as a
//! mirror that quietly agrees with itself.
//!
//! The domain is small enough that one file pins all three contracts; the three
//! sections below are still independently reviewable, and each says which one it
//! is.

use super::*;
use crate::webgl2::api::tests::compute_storage_snapshot;
use crate::webgl2::api::{
    GlExtent3d, GlFormat, GlResourceApi, GlStorageImageAccess, GlTextureDesc, GlTextureDimension,
    GlTextureUsage, MockCall, MockComputeStorageApi, MockGlFamilyApi, TextureId,
};
use crate::webgl2::state::counters::DomainCounters;
use crate::webgl2::state::event::ScopedRawAccess;

/// The optional-domain recorder, an invalidation sink, and two storage images.
///
/// The verb this domain mirrors lives on a trait the command-backend bound does
/// not include, so the reconcile runs against the optional-domain wrapper.  The
/// sink is a second recorder rather than the same one, so that "forgetting a
/// binding is not a driver call" is read off a recorder nothing else writes to
/// instead of inferred from a shared trace.  Both are built from the same
/// snapshot and allocate in the same order, so the *n*th texture in each carries
/// the same identity; the coincidence is asserted at construction rather than
/// assumed.
struct Fixture {
    api: MockComputeStorageApi,
    sink: MockGlFamilyApi,
    first: TextureId,
    second: TextureId,
}

impl Fixture {
    fn new() -> Self {
        let mut api = MockGlFamilyApi::from_discovery(compute_storage_snapshot(true));
        let first = create_image(&mut api);
        let second = create_image(&mut api);
        let mut sink = MockGlFamilyApi::from_discovery(compute_storage_snapshot(true));
        assert_eq!(first, create_image(&mut sink), "the recorders share ids");
        assert_eq!(second, create_image(&mut sink), "the recorders share ids");
        Self {
            api: MockComputeStorageApi::new(api).expect("the snapshot proved storage images"),
            sink,
            first,
            second,
        }
    }

    /// Where the recorded trace currently ends.
    fn mark(&self) -> usize {
        self.api.calls().len()
    }

    /// Every storage-image binding word recorded since `mark`, in order.
    fn since(&self, mark: usize) -> Vec<MockCall> {
        self.api.calls()[mark..]
            .iter()
            .filter(|call| matches!(call, MockCall::BindStorageImage { .. }))
            .cloned()
            .collect()
    }

    fn counters_for(&self, counters: &StateCounters) -> DomainCounters {
        *counters.domain_counts(StateDomain::Compute)
    }
}

/// One storage image: single level, single sample, both accesses certified.
fn create_image(backend: &mut MockGlFamilyApi) -> TextureId {
    backend
        .create_texture_resource(GlTextureDesc {
            dimension: GlTextureDimension::D2,
            extent: GlExtent3d {
                width: 1,
                height: 1,
                depth_or_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            format: GlFormat::Rgba8Unorm,
            usage: GlTextureUsage::STORAGE_BINDING,
        })
        .expect("a storage image")
}

/// One image-unit binding word, for an exact trace comparison.
///
/// The word carries the unit and the texture and nothing else, because that is
/// what Layer 1's recorded call carries; the rest of the binding is compared by
/// the mirror, not by the trace.  The two tests below therefore read as a pair:
/// the same binding twice emits once, and a binding whose access changed emits
/// again -- and since the trace word is identical in both, what the pair pins is
/// that the *domain's* comparison, not the trace's, is what decides.
fn bound(binding: u32, texture: TextureId) -> MockCall {
    MockCall::BindStorageImage { binding, texture }
}

/// One binding of `texture` at `binding`, with the whole-request access chosen by
/// the caller so that a changed access is a changed request.
fn image(texture: TextureId, access: GlStorageImageAccess) -> GlStorageImageBinding {
    GlStorageImageBinding {
        texture,
        level: 0,
        sample_count: 1,
        layered: false,
        layer: Some(0),
        format: GlFormat::Rgba8Unorm,
        access,
    }
}

// ---------------------------------------------------------------------------
// The mirror: which requests are the same request.
// ---------------------------------------------------------------------------

#[test]
fn the_first_request_for_a_unit_emits_and_a_repeated_one_is_skipped() {
    let mut fixture = Fixture::new();
    let mut state = ComputeState::new(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();

    state.bind_storage_image(
        0,
        image(fixture.first, GlStorageImageAccess::ReadWrite),
        &mut counters,
    );
    let mark = fixture.mark();
    state
        .reconcile(&mut fixture.api, &mut counters)
        .expect("an unset unit emits the call that establishes it");
    assert_eq!(
        fixture.since(mark),
        vec![bound(0, fixture.first)],
        "an unknown unit never agrees, so the first request is always emitted"
    );
    assert_eq!(fixture.counters_for(&counters).unknown_recoveries, 1);

    // The same request again.  The mirror is known to agree, so the call is
    // proved redundant and nothing is emitted.
    state.bind_storage_image(
        0,
        image(fixture.first, GlStorageImageAccess::ReadWrite),
        &mut counters,
    );
    let mark = fixture.mark();
    state
        .reconcile(&mut fixture.api, &mut counters)
        .expect("a redundant request cannot fail");
    assert!(
        fixture.since(mark).is_empty(),
        "the same whole binding is the same request"
    );
    assert_eq!(fixture.counters_for(&counters).skipped, 1);
}

#[test]
fn the_whole_binding_is_the_request_so_a_changed_access_emits_again() {
    let mut fixture = Fixture::new();
    let mut state = ComputeState::new(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();

    state.bind_storage_image(
        0,
        image(fixture.first, GlStorageImageAccess::ReadWrite),
        &mut counters,
    );
    state
        .reconcile(&mut fixture.api, &mut counters)
        .expect("the first application emits");
    let mark = fixture.mark();

    // The access is a fact of the binding, not of the texture: a shader that will
    // write needs the read-write view even when the image and the unit are the
    // ones already bound.
    state.bind_storage_image(
        0,
        image(fixture.first, GlStorageImageAccess::ReadOnly),
        &mut counters,
    );
    state
        .reconcile(&mut fixture.api, &mut counters)
        .expect("a changed binding emits");
    assert_eq!(fixture.since(mark), vec![bound(0, fixture.first)]);
}

#[test]
fn two_units_are_independent_slots() {
    let mut fixture = Fixture::new();
    let mut state = ComputeState::new(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();

    // Recorded out of order on purpose: the settle order is the mirror's own
    // contract, because a trace whose order depended on a hash would make the
    // differential comparison against the oracle meaningless.
    state.bind_storage_image(
        1,
        image(fixture.second, GlStorageImageAccess::ReadOnly),
        &mut counters,
    );
    state.bind_storage_image(
        0,
        image(fixture.first, GlStorageImageAccess::ReadWrite),
        &mut counters,
    );
    let mark = fixture.mark();
    state
        .reconcile(&mut fixture.api, &mut counters)
        .expect("both units are new");
    assert_eq!(
        fixture.since(mark),
        vec![bound(0, fixture.first), bound(1, fixture.second),]
    );
    assert_eq!(fixture.counters_for(&counters).emitted, 2);
}

#[test]
fn the_oracle_emits_a_redundant_request_the_optimized_domain_skips() {
    let mut optimized = Fixture::new();
    let mut state = ComputeState::new(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();
    for _ in 0..2 {
        state.bind_storage_image(
            0,
            image(optimized.first, GlStorageImageAccess::ReadOnly),
            &mut counters,
        );
        state
            .reconcile(&mut optimized.api, &mut counters)
            .expect("the optimized domain settles");
    }
    let mark = optimized.mark();
    state.bind_storage_image(
        0,
        image(optimized.first, GlStorageImageAccess::ReadOnly),
        &mut counters,
    );
    state
        .reconcile(&mut optimized.api, &mut counters)
        .expect("a redundant request cannot fail");
    assert!(optimized.since(mark).is_empty());

    // The oracle runs the same request against the same mirror and still emits,
    // which is the whole reason it exists: a skipping oracle would produce a
    // trace no mirror-free machine could have produced.
    let mut oracle = Fixture::new();
    let mut state = ComputeState::new(ExecutionMode::Oracle);
    let mut counters = StateCounters::default();
    let mark = oracle.mark();
    for _ in 0..2 {
        state.bind_storage_image(
            0,
            image(oracle.first, GlStorageImageAccess::ReadOnly),
            &mut counters,
        );
        state
            .reconcile(&mut oracle.api, &mut counters)
            .expect("the oracle emits what was asked for");
    }
    assert_eq!(
        oracle.since(mark),
        vec![bound(0, oracle.first), bound(0, oracle.first),]
    );
    assert_eq!(oracle.counters_for(&counters).skipped, 0);
    assert_eq!(state.mode(), ExecutionMode::Oracle);
}

// ---------------------------------------------------------------------------
// Refusals.
// ---------------------------------------------------------------------------

#[test]
fn an_image_unit_outside_the_index_space_is_refused_by_layer_one() {
    let mut fixture = Fixture::new();
    let mut state = ComputeState::new(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();

    // The fixture proves four image units, so unit 4 does not name one.  The
    // domain asks, Layer 1 refuses, and the domain reports the refusal rather
    // than inventing a second bounds check that could drift from this one.
    state.bind_storage_image(
        4,
        image(fixture.first, GlStorageImageAccess::ReadWrite),
        &mut counters,
    );
    let mark = fixture.mark();
    let error = state
        .reconcile(&mut fixture.api, &mut counters)
        .expect_err("the unit is outside the discovered count");

    assert_eq!(error.domain(), StateDomain::Compute);
    assert_eq!(error.operation(), "bind-storage-image");
    assert!(
        !error.is_pre_side_effect(),
        "the backend reported it, so this layer cannot claim no call was made"
    );
    assert_eq!(counters.lifecycle.driver_errors, 1);
    assert!(fixture.since(mark).is_empty());
    assert_eq!(fixture.counters_for(&counters).emitted, 0);
}

// ---------------------------------------------------------------------------
// Invalidations.
// ---------------------------------------------------------------------------

#[test]
fn deleting_a_texture_forgets_every_unit_that_named_it() {
    let mut fixture = Fixture::new();
    let mut state = ComputeState::new(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();

    state.bind_storage_image(
        0,
        image(fixture.first, GlStorageImageAccess::ReadWrite),
        &mut counters,
    );
    state.bind_storage_image(
        1,
        image(fixture.first, GlStorageImageAccess::ReadWrite),
        &mut counters,
    );
    state.bind_storage_image(
        2,
        image(fixture.second, GlStorageImageAccess::ReadWrite),
        &mut counters,
    );
    state
        .reconcile(&mut fixture.api, &mut counters)
        .expect("all three units are new");
    let mark = fixture.mark();

    // The deletion is dispatched to the mirror before the backend is asked to
    // delete, which is the ordering that keeps a reused name from hitting a
    // record that still means the old occupant.
    state.invalidate(
        &mut fixture.sink,
        &StateEvent::TextureDeleted(fixture.first),
        &mut counters,
    );

    assert!(
        fixture.since(mark).is_empty(),
        "forgetting a binding is not a driver call"
    );
    assert_eq!(
        counters.lifecycle.domain_invalidations, 0,
        "one texture is not a whole-domain invalidation"
    );
    assert_eq!(
        fixture.counters_for(&counters).emitted,
        3,
        "only the three earlier applications emitted"
    );

    // Nothing to re-apply: the two units that named the deleted texture are no
    // longer wanted, and the unit naming the live one is still believed.  The
    // rule is [`super::super::binding`]'s and this domain follows it rather than
    // having one of its own -- re-emitting a binding to a deleted texture could
    // only be refused, and the refusal would be reported as a failure of this
    // reconcile rather than of any request the caller made.
    let desired_before = state.images.desired_len();
    state
        .reconcile(&mut fixture.api, &mut counters)
        .expect("a deletion leaves nothing to re-apply");
    assert!(
        fixture.since(mark).is_empty(),
        "a want that named the deleted texture is dropped, not re-emitted"
    );
    assert_eq!(
        desired_before, 1,
        "the two units that named the deleted texture lost their wants"
    );

    // A unit re-requested afterwards is a fresh want and emits, because the unit
    // is unknown again rather than believed to hold the deleted texture.
    state.bind_storage_image(
        0,
        image(fixture.second, GlStorageImageAccess::ReadWrite),
        &mut counters,
    );
    state
        .reconcile(&mut fixture.api, &mut counters)
        .expect("the unit is unknown, so the new want emits");
    assert_eq!(
        fixture.since(mark),
        vec![bound(0, fixture.second)],
        "only the unit that was re-requested emits"
    );
}

#[test]
fn a_declared_scope_only_forgets_when_it_names_this_domain() {
    let mut fixture = Fixture::new();
    let mut state = ComputeState::new(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();

    state.bind_storage_image(
        0,
        image(fixture.first, GlStorageImageAccess::ReadWrite),
        &mut counters,
    );
    state
        .reconcile(&mut fixture.api, &mut counters)
        .expect("the first application emits");

    // A scope that named another domain leaves this one believing what it
    // applied, which is the entire reason a scope declares its domains.
    state.invalidate(
        &mut fixture.sink,
        &StateEvent::ScopedRawAccess(ScopedRawAccess::declaring(DirtyDomains::of(
            StateDomain::Textures,
        ))),
        &mut counters,
    );
    let mark = fixture.mark();
    state
        .reconcile(&mut fixture.api, &mut counters)
        .expect("nothing to re-apply");
    assert!(fixture.since(mark).is_empty());
    assert_eq!(counters.lifecycle.domain_invalidations, 0);

    // A scope that named this domain forgets what the driver holds, and the want
    // survives to be re-applied.
    state.invalidate(
        &mut fixture.sink,
        &StateEvent::ScopedRawAccess(ScopedRawAccess::declaring(DirtyDomains::of(
            StateDomain::Compute,
        ))),
        &mut counters,
    );
    let mark = fixture.mark();
    state
        .reconcile(&mut fixture.api, &mut counters)
        .expect("the want is re-applied");
    assert_eq!(fixture.since(mark).len(), 1);
    assert_eq!(counters.lifecycle.domain_invalidations, 1);

    // A scope that declared nothing is assumed to have touched everything.
    state.invalidate(
        &mut fixture.sink,
        &StateEvent::ScopedRawAccess(ScopedRawAccess::all()),
        &mut counters,
    );
    let mark = fixture.mark();
    state
        .reconcile(&mut fixture.api, &mut counters)
        .expect("the want is re-applied");
    assert_eq!(fixture.since(mark).len(), 1);
    assert_eq!(
        counters.lifecycle.domain_invalidations, 2,
        "one undeclared scope is still one domain invalidation, not one per event"
    );
}

#[test]
fn a_whole_mirror_event_keeps_the_wants_and_forgets_the_beliefs() {
    let mut fixture = Fixture::new();
    let mut state = ComputeState::new(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();

    state.bind_storage_image(
        0,
        image(fixture.first, GlStorageImageAccess::ReadWrite),
        &mut counters,
    );
    state
        .reconcile(&mut fixture.api, &mut counters)
        .expect("the first application emits");

    state.invalidate(&mut fixture.sink, &StateEvent::ContextLost, &mut counters);

    assert_eq!(counters.lifecycle.domain_invalidations, 1);
    let mark = fixture.mark();
    state
        .reconcile(&mut fixture.api, &mut counters)
        .expect("the want is re-applied after restoration");
    assert_eq!(
        fixture.since(mark),
        vec![bound(0, fixture.first)],
        "the caller's want is re-established rather than assumed to be gone"
    );
    assert_eq!(fixture.counters_for(&counters).unknown_recoveries, 2);
}
