//! Tests for the bound vertex input and the derived vertex-array records.
//!
//! Every test here runs the domain against the shared recorder rather than a
//! hand-written double, because what is being tested is the *trace* -- which
//! calls were emitted, in what order, and how many.  A double written for these
//! tests would answer with whatever trace the test expected; the recorder is the
//! same provider the Layer 1 contract suites run against, so a change in Layer
//! 1's own call surface shows up here as a failing trace instead of as a mirror
//! that quietly agrees with itself.
//!
//! The recorders here count calls and check identities rather than inspecting
//! the buffer bindings a bind carried: the recorder's trace deliberately names
//! the object a call was made on and nothing else, so the bindings are only
//! observable through the object they were emitted into.  What the tests can
//! assert is therefore the part the mirror owns -- *when* a bind is emitted and
//! *which* array it names.
//!
//! # Why this is one file
//!
//! Every case below shares the same four trace counters and the same three
//! fixture builders, and each one reads as a short sequence of setup, event and
//! trace assertion.  Splitting the suite by contract would separate the counters
//! from their only callers and make a reader follow two files to see what one
//! trace proves.  The file is long because there are many independent traces to
//! pin down, not because it has more than one responsibility.

use super::*;
use crate::webgl2::api::tests::snapshot;
use crate::webgl2::api::{
    GlBufferDesc, GlBufferUsage, GlError, GlFamilyProfile, GlIndexFormat, GlResourceApi,
    GlVertexApi, GlVertexAttribute, GlVertexBufferLayout, GlVertexFormat, GlVertexStepMode,
    MockCall, MockGlFamilyApi,
};
use crate::webgl2::state::{
    CacheBudget, DEFAULT_BUDGET, DirtyDomains, DomainCounters, ScopedRawAccess,
};

fn backend() -> MockGlFamilyApi {
    MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2))
}

fn buffer_with(backend: &mut MockGlFamilyApi, usage: GlBufferUsage) -> BufferId {
    backend
        .create_buffer_resource(GlBufferDesc { size: 64, usage })
        .expect("the recorder allocates the buffer")
}

fn vertex_buffer(backend: &mut MockGlFamilyApi) -> BufferId {
    buffer_with(backend, GlBufferUsage::VERTEX)
}

fn index_buffer(backend: &mut MockGlFamilyApi) -> BufferId {
    buffer_with(backend, GlBufferUsage::INDEX)
}

/// A one-slot, one-attribute layout: `Float32x3` at offset 0 of the slot.
fn layout(stride: u32, step_mode: GlVertexStepMode) -> GlVertexLayout {
    GlVertexLayout {
        buffers: vec![GlVertexBufferLayout {
            slot: 0,
            stride,
            step_mode,
        }],
        attributes: vec![GlVertexAttribute {
            location: 0,
            buffer_slot: 0,
            format: GlVertexFormat::Float32x3,
            offset: 0,
        }],
    }
}

fn input(layout: GlVertexLayout, buffer: BufferId, offset: u64) -> VertexInput {
    VertexInput {
        layout,
        bindings: vec![GlVertexBufferBinding {
            slot: 0,
            buffer,
            offset,
        }],
        index: None,
    }
}

fn indexed(layout: GlVertexLayout, buffer: BufferId, source: BufferId) -> VertexInput {
    VertexInput {
        index: Some(GlIndexBinding {
            buffer: source,
            format: GlIndexFormat::Uint16,
            offset: 0,
        }),
        ..input(layout, buffer, 0)
    }
}

fn geometry(mode: CacheMode) -> GeometryState {
    GeometryState::new(DEFAULT_BUDGET, mode)
}

fn calls_of(backend: &MockGlFamilyApi, matched: fn(&MockCall) -> bool) -> usize {
    backend.calls().iter().filter(|call| matched(call)).count()
}

fn created(backend: &MockGlFamilyApi) -> usize {
    calls_of(backend, |call| {
        matches!(call, MockCall::CreateVertexArray(_))
    })
}

fn destroyed(backend: &MockGlFamilyApi) -> usize {
    calls_of(backend, |call| {
        matches!(call, MockCall::DestroyVertexArray(_))
    })
}

fn binds(backend: &MockGlFamilyApi) -> usize {
    calls_of(backend, |call| matches!(call, MockCall::BindVertexArray(_)))
}

fn counts(counters: &StateCounters) -> &DomainCounters {
    counters.domain_counts(StateDomain::Geometry)
}

/// Reconciles after a setup step whose own effect the test does not assert.
///
/// The reported effect must agree with the trace that produced it: a result
/// claiming a bind that no call in the trace shows would be exactly the mirror
/// bug these tests exist to catch, so every caller of this helper checks it
/// rather than restating the assertion.
fn apply(state: &mut GeometryState, backend: &mut MockGlFamilyApi, counters: &mut StateCounters) {
    let before = binds(backend);
    let effects = state
        .reconcile(backend, counters)
        .expect("the input applies");
    assert_eq!(
        effects.vertex_array_bound,
        binds(backend) > before,
        "the reported effect must describe the trace it came from"
    );
}

#[test]
fn a_reconcile_with_nothing_asked_for_emits_nothing() {
    let mut backend = backend();
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();

    apply(&mut state, &mut backend, &mut counters);

    assert_eq!(binds(&backend), 0);
    assert_eq!(created(&backend), 0);
    assert_eq!(counts(&counters).requests, 1);
    assert_eq!(counts(&counters).skipped, 1);
}

#[test]
fn an_unchanged_request_is_not_bound_twice() {
    let mut backend = backend();
    let buffer = vertex_buffer(&mut backend);
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();
    let wanted = input(layout(12, GlVertexStepMode::Vertex), buffer, 0);

    state.set_vertex_input(wanted.clone());
    let first = state
        .reconcile(&mut backend, &mut counters)
        .expect("the first bind emits");
    assert_eq!(
        first,
        GeometryEffects {
            vertex_array_bound: true,
            array_buffer_binding_unknown: true,
        }
    );
    assert_eq!(state.applied_input(), Some(&wanted));
    assert_eq!(
        counts(&counters).unknown_recoveries,
        1,
        "nothing was known about the driver's vertex input, so this bind recovered it"
    );

    // Asking again for what is already bound is the common case in a renderer
    // that re-enters the same draw setup.
    let second = state
        .reconcile(&mut backend, &mut counters)
        .expect("an unchanged input is not a failure");
    assert_eq!(second, GeometryEffects::NONE);
    assert_eq!(binds(&backend), 1);
    assert_eq!(created(&backend), 1);
    assert_eq!(
        counts(&counters).emitted,
        2,
        "two driver calls on the first request -- the derivation and the bind -- and none on the second"
    );
    assert_eq!(counts(&counters).skipped, 1);
    assert_eq!(
        counts(&counters).unknown_recoveries,
        1,
        "a skipped call is not a recovery: the value was already known"
    );
}

#[test]
fn a_changed_request_binds_again_through_the_same_array() {
    let mut backend = backend();
    let buffer = vertex_buffer(&mut backend);
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();
    let vertex_layout = layout(12, GlVertexStepMode::Vertex);

    state.set_vertex_input(input(vertex_layout.clone(), buffer, 0));
    apply(&mut state, &mut backend, &mut counters);
    // One field of the input changed: the slot now starts 16 bytes in.  The
    // layout did not, so the same array can be pointed at it.
    state.set_vertex_input(input(vertex_layout, buffer, 16));
    let effects = state
        .reconcile(&mut backend, &mut counters)
        .expect("the changed binding binds");

    assert!(effects.vertex_array_bound);
    assert_eq!(binds(&backend), 2);
    assert_eq!(
        created(&backend),
        1,
        "one layout needs one array however many times it is bound"
    );
    assert_eq!(state.vertex_array_records(), 1);
    assert_eq!(counters.caches.hits, 1);
    assert_eq!(counters.caches.created, 1);
}

#[test]
fn a_second_request_for_the_same_record_reuses_it_without_binding_it() {
    let mut backend = backend();
    let buffer = vertex_buffer(&mut backend);
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();
    let wanted = input(layout(12, GlVertexStepMode::Vertex), buffer, 0);

    // Two derivations of one input are one object: a caller that wants an
    // identity to name in a pipeline does not pay for a second array.
    let (first, owned) = state
        .vertex_array_for(&mut backend, &wanted, &mut counters)
        .expect("the first derivation builds the array");
    assert!(!owned, "a retained array belongs to the cache");
    let (second, owned) = state
        .vertex_array_for(&mut backend, &wanted, &mut counters)
        .expect("the second derivation is served from the cache");
    assert_eq!(first, second);
    assert!(!owned);

    assert_eq!(created(&backend), 1);
    assert_eq!(binds(&backend), 0, "deriving is not binding");
    assert_eq!(state.applied_vertex_array(), None);
    assert_eq!(counters.caches.hits, 1);
    assert_eq!(counters.caches.misses, 1);
    assert_eq!(counters.caches.created, 1);
    assert_eq!(counters.caches.live_entries, 1);
    assert_eq!(state.vertex_array_records(), 1);
}

#[test]
fn two_layouts_differing_in_one_field_do_not_share_an_array() {
    let mut backend = backend();
    let buffer = vertex_buffer(&mut backend);
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();

    // A stride is one input of the derivation, and a step mode is another.
    // Either one alone describes a different layout, and an array reused across
    // them would re-emit the wrong description at the next bind with nothing
    // anywhere reporting it.
    let mut stride_changed = layout(12, GlVertexStepMode::Vertex);
    stride_changed.buffers[0].stride = 16;
    let stepped = layout(12, GlVertexStepMode::Instance);

    state.set_vertex_input(input(layout(12, GlVertexStepMode::Vertex), buffer, 0));
    apply(&mut state, &mut backend, &mut counters);
    state.set_vertex_input(input(stride_changed, buffer, 0));
    apply(&mut state, &mut backend, &mut counters);
    state.set_vertex_input(input(stepped, buffer, 0));
    let effects = state
        .reconcile(&mut backend, &mut counters)
        .expect("a different layout binds");

    assert!(effects.vertex_array_bound);
    assert_eq!(created(&backend), 3);
    assert_eq!(state.vertex_array_records(), 3);
    assert_eq!(counters.caches.misses, 3);
    assert_eq!(counters.caches.hits, 0);
}

#[test]
fn a_layout_with_no_slots_does_not_reach_the_array_buffer_point() {
    let mut backend = backend();
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();

    // A layout with no slots is legal, and a bind of one emits no attribute
    // description at all, so the generic array-buffer binding point is the one
    // thing that bind did not touch.
    state.set_vertex_input(VertexInput {
        layout: GlVertexLayout {
            buffers: vec![],
            attributes: vec![],
        },
        bindings: vec![],
        index: None,
    });
    let effects = state
        .reconcile(&mut backend, &mut counters)
        .expect("an empty layout binds");

    assert_eq!(
        effects,
        GeometryEffects {
            vertex_array_bound: true,
            array_buffer_binding_unknown: false,
        }
    );
    assert_eq!(binds(&backend), 1);
    assert_eq!(state.vertex_array_records(), 1);
}

#[test]
fn an_indexed_input_depends_on_its_index_buffer() {
    let mut backend = backend();
    let buffer = vertex_buffer(&mut backend);
    let source = index_buffer(&mut backend);
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();

    state.set_vertex_input(indexed(
        layout(12, GlVertexStepMode::Vertex),
        buffer,
        source,
    ));
    apply(&mut state, &mut backend, &mut counters);
    assert_eq!(state.vertex_array_records(), 1);

    // The array's recorded index binding names this buffer, so the record goes
    // before the caller deletes it.
    state.invalidate(
        &mut backend,
        &StateEvent::BufferDeleted(source),
        &mut counters,
    );

    assert_eq!(state.vertex_array_records(), 0);
    assert_eq!(destroyed(&backend), 1);
    assert_eq!(state.applied_input(), None);
    assert_eq!(counters.caches.invalidated, 1);
}

#[test]
fn deleting_a_buffer_the_record_reads_drops_the_record() {
    let mut backend = backend();
    let buffer = vertex_buffer(&mut backend);
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();

    state.set_vertex_input(input(layout(12, GlVertexStepMode::Vertex), buffer, 0));
    apply(&mut state, &mut backend, &mut counters);
    assert_eq!(state.vertex_array_records(), 1);

    state.invalidate(
        &mut backend,
        &StateEvent::BufferDeleted(buffer),
        &mut counters,
    );

    assert_eq!(state.vertex_array_records(), 0);
    assert_eq!(destroyed(&backend), 1);
    assert_eq!(state.applied_input(), None);
    assert_eq!(
        counters.lifecycle.domain_invalidations, 0,
        "one deletion is not a whole-domain invalidation"
    );

    // The record is dropped rather than repaired, so the next request derives a
    // new array instead of handing back one that reads a dead buffer.
    let replacement = vertex_buffer(&mut backend);
    state.set_vertex_input(input(layout(12, GlVertexStepMode::Vertex), replacement, 0));
    let effects = state
        .reconcile(&mut backend, &mut counters)
        .expect("a new buffer binds");
    assert!(effects.vertex_array_bound);
    assert_eq!(created(&backend), 2);
    assert_eq!(binds(&backend), 2);
}

#[test]
fn a_buffer_only_the_claim_names_still_forces_a_rebind() {
    let mut backend = backend();
    let first = vertex_buffer(&mut backend);
    let second = vertex_buffer(&mut backend);
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();
    let vertex_layout = layout(12, GlVertexStepMode::Vertex);

    state.set_vertex_input(input(vertex_layout.clone(), first, 0));
    apply(&mut state, &mut backend, &mut counters);
    // The record was created from the first buffer, so its dependencies name
    // that one; the bind now in force points at the second.
    state.set_vertex_input(input(vertex_layout.clone(), second, 0));
    apply(&mut state, &mut backend, &mut counters);
    assert_eq!(state.vertex_array_records(), 1);

    state.invalidate(
        &mut backend,
        &StateEvent::BufferDeleted(second),
        &mut counters,
    );

    // No record depended on the deleted buffer, so none was dropped -- but the
    // claim described an input whose attribute source is going away, and the
    // array's live description still points at it.
    assert_eq!(state.vertex_array_records(), 1);
    assert_eq!(destroyed(&backend), 0);
    assert_eq!(state.applied_input(), None);

    state.set_vertex_input(input(vertex_layout, second, 0));
    let effects = state
        .reconcile(&mut backend, &mut counters)
        .expect("the claim is rebuilt");
    assert!(effects.vertex_array_bound);
    assert_eq!(binds(&backend), 3);
    assert_eq!(
        created(&backend),
        1,
        "the record survived, so the rebind reuses the same array"
    );
}

#[test]
fn a_refused_bind_is_not_mirrored_as_an_applied_input() {
    let mut backend = backend();
    let buffer = vertex_buffer(&mut backend);
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();
    let vertex_layout = layout(12, GlVertexStepMode::Vertex);

    state.set_vertex_input(input(vertex_layout.clone(), buffer, 0));
    apply(&mut state, &mut backend, &mut counters);
    let binds_before = binds(&backend);

    state.set_vertex_input(input(vertex_layout, buffer, 16));
    backend.fail_next(GlError::Driver {
        operation: "injected",
        message: "the driver refused the bind".into(),
    });
    let error = state
        .reconcile(&mut backend, &mut counters)
        .expect_err("the driver refuses the bind");

    assert_eq!(error.domain(), StateDomain::Geometry);
    assert_eq!(error.operation(), "bind-vertex-array");
    assert!(!error.is_pre_side_effect());
    match &error {
        StateError::Backend { applied, .. } => {
            assert_eq!(applied.emitted, 0, "no call reached the driver");
        }
        other => panic!("a driver refusal is a backend failure, not {other:?}"),
    }
    assert_eq!(counters.lifecycle.driver_errors, 1);
    assert_eq!(
        state.applied_input(),
        None,
        "a failed bind leaves the driver's input unknown rather than unchanged"
    );
    assert_eq!(binds(&backend), binds_before);

    // The mirror must not treat the refusal as a success: the retry emits the
    // bind the driver never accepted.
    let effects = state
        .reconcile(&mut backend, &mut counters)
        .expect("the bind is retried");
    assert!(effects.vertex_array_bound);
    assert_eq!(binds(&backend), binds_before + 1);
    assert_eq!(counts(&counters).unknown_recoveries, 2);
}

#[test]
fn a_refused_creation_is_reported_against_the_derivation() {
    let mut backend = backend();
    let buffer = vertex_buffer(&mut backend);
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();
    let wanted = input(layout(12, GlVertexStepMode::Vertex), buffer, 0);

    backend.fail_next(GlError::Driver {
        operation: "injected",
        message: "the driver refused the array".into(),
    });
    let error = state
        .vertex_array_for(&mut backend, &wanted, &mut counters)
        .expect_err("the driver refuses the array");

    assert_eq!(error.domain(), StateDomain::Geometry);
    assert_eq!(error.operation(), "create-vertex-array");
    assert_eq!(counters.lifecycle.driver_errors, 1);
    assert_eq!(created(&backend), 0);
    assert_eq!(state.vertex_array_records(), 0);
    assert_eq!(counters.caches.created, 0);

    // Nothing was retained, so the next request is a fresh derivation rather
    // than a hit on an object that was never built.
    let (_, owned) = state
        .vertex_array_for(&mut backend, &wanted, &mut counters)
        .expect("the retry derives the array");
    assert!(!owned);
    assert_eq!(created(&backend), 1);
    assert_eq!(state.vertex_array_records(), 1);
}

#[test]
fn a_whole_mirror_event_purges_without_destroying_and_keeps_the_request() {
    let mut backend = backend();
    let buffer = vertex_buffer(&mut backend);
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();
    let wanted = input(layout(12, GlVertexStepMode::Vertex), buffer, 0);

    state.set_vertex_input(wanted.clone());
    apply(&mut state, &mut backend, &mut counters);
    let destroyed_before = destroyed(&backend);

    state.invalidate(&mut backend, &StateEvent::ContextLost, &mut counters);

    assert_eq!(state.vertex_array_records(), 0);
    assert_eq!(
        destroyed(&backend),
        destroyed_before,
        "an identity from a lost epoch is not callable, so nothing may be destroyed through it"
    );
    assert_eq!(counters.caches.purged, 1);
    assert_eq!(counters.lifecycle.domain_invalidations, 1);
    assert_eq!(state.applied_input(), None);
    assert_eq!(
        state.desired_input(),
        Some(&wanted),
        "the caller's request outlives the epoch; restoration rebinds it"
    );
}

#[test]
fn a_scope_that_declares_this_domain_forgets_the_claim_and_keeps_the_records() {
    let mut backend = backend();
    let buffer = vertex_buffer(&mut backend);
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();
    let wanted = input(layout(12, GlVertexStepMode::Vertex), buffer, 0);

    state.set_vertex_input(wanted.clone());
    apply(&mut state, &mut backend, &mut counters);

    // A scope that declared another domain leaves this one's claim alone: the
    // declaration is what makes the scope precise enough to ignore.
    state.invalidate(
        &mut backend,
        &StateEvent::ScopedRawAccess(ScopedRawAccess::declaring(DirtyDomains::of(
            StateDomain::Textures,
        ))),
        &mut counters,
    );
    assert_eq!(state.applied_input(), Some(&wanted));

    state.invalidate(
        &mut backend,
        &StateEvent::ScopedRawAccess(ScopedRawAccess::declaring(DirtyDomains::of(
            StateDomain::Geometry,
        ))),
        &mut counters,
    );

    assert_eq!(state.applied_input(), None);
    assert_eq!(
        state.vertex_array_records(),
        1,
        "the arrays are objects this layer created, and no scope reported deleting one"
    );
    assert_eq!(counters.lifecycle.domain_invalidations, 0);

    let effects = state
        .reconcile(&mut backend, &mut counters)
        .expect("the claim is rebuilt");
    assert!(effects.vertex_array_bound);
    assert_eq!(binds(&backend), 2);
    assert_eq!(created(&backend), 1);
}

#[test]
fn a_failed_buffer_domain_drops_the_claim_but_not_the_records() {
    let mut backend = backend();
    let buffer = vertex_buffer(&mut backend);
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();
    let wanted = input(layout(12, GlVertexStepMode::Vertex), buffer, 0);

    state.set_vertex_input(wanted.clone());
    apply(&mut state, &mut backend, &mut counters);

    state.invalidate(
        &mut backend,
        &StateEvent::DomainFailed(StateDomain::Buffers),
        &mut counters,
    );

    assert_eq!(state.applied_input(), None);
    // The record is keyed by a structure that names no buffer, so a binding
    // failure cannot make it describe the wrong layout; only the claim, which is
    // built from buffer bindings, is gone.
    assert_eq!(state.vertex_array_records(), 1);
    assert_eq!(destroyed(&backend), 0);

    apply(&mut state, &mut backend, &mut counters);
    assert_eq!(state.applied_input(), Some(&wanted));

    // A failure in a domain this one does not read is not this one's to react
    // to, which is why the event names the domain instead of dirtying a mask.
    state.invalidate(
        &mut backend,
        &StateEvent::DomainFailed(StateDomain::Textures),
        &mut counters,
    );
    assert_eq!(state.applied_input(), Some(&wanted));
}

#[test]
fn deleting_the_bound_array_drops_the_record_without_destroying_it_again() {
    let mut backend = backend();
    let buffer = vertex_buffer(&mut backend);
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();

    state.set_vertex_input(input(layout(12, GlVertexStepMode::Vertex), buffer, 0));
    apply(&mut state, &mut backend, &mut counters);
    let vertex_array = state
        .applied_vertex_array()
        .expect("the bind names an array");
    backend
        .destroy_vertex_array(vertex_array)
        .expect("the owner deletes it");
    let destroyed_before = destroyed(&backend);

    state.invalidate(
        &mut backend,
        &StateEvent::VertexArrayDeleted(vertex_array),
        &mut counters,
    );

    assert_eq!(state.vertex_array_records(), 0);
    assert_eq!(
        destroyed(&backend),
        destroyed_before,
        "the object is already gone, so asking again would be a call on an identity the backend refuses"
    );
    assert_eq!(state.applied_input(), None);
    assert_eq!(counters.lifecycle.driver_errors, 0);

    let effects = state
        .reconcile(&mut backend, &mut counters)
        .expect("a fresh array is derived");
    assert!(effects.vertex_array_bound);
    assert_eq!(created(&backend), 2);
    assert_ne!(state.applied_vertex_array(), Some(vertex_array));
}

#[test]
fn a_disabled_cache_never_reuses_and_hands_ownership_back() {
    let mut backend = backend();
    let buffer = vertex_buffer(&mut backend);
    let mut state = geometry(CacheMode::Disabled);
    let mut counters = StateCounters::default();
    let wanted = input(layout(12, GlVertexStepMode::Vertex), buffer, 0);

    let (first, owned) = state
        .vertex_array_for(&mut backend, &wanted, &mut counters)
        .expect("the oracle still derives a usable array");
    assert!(
        owned,
        "the oracle retains nothing, so the caller destroys it"
    );
    let (second, owned) = state
        .vertex_array_for(&mut backend, &wanted, &mut counters)
        .expect("the oracle derives again");
    assert!(owned);
    assert_ne!(first, second);

    assert_eq!(created(&backend), 2);
    assert_eq!(state.vertex_array_records(), 0);
    assert_eq!(counters.caches.hits, 0);
    assert_eq!(counters.caches.created, 0);
    backend
        .destroy_vertex_array(first)
        .expect("the caller destroys its own array");
    backend
        .destroy_vertex_array(second)
        .expect("the caller destroys its own array");
}

#[test]
fn an_array_the_budget_could_not_keep_is_destroyed_when_its_claim_is_replaced() {
    // Neither bound can be met, so every derivation is refused and the array it
    // built belongs to the domain alone.
    let mut backend = backend();
    let buffer = vertex_buffer(&mut backend);
    let mut state = GeometryState::new(CacheBudget::new(0, 0), CacheMode::Enabled);
    let mut counters = StateCounters::default();
    let vertex_layout = layout(12, GlVertexStepMode::Vertex);

    state.set_vertex_input(input(vertex_layout.clone(), buffer, 0));
    apply(&mut state, &mut backend, &mut counters);
    assert_eq!(state.vertex_array_records(), 0);
    assert_eq!(destroyed(&backend), 0);

    state.set_vertex_input(input(vertex_layout, buffer, 16));
    apply(&mut state, &mut backend, &mut counters);

    assert_eq!(
        destroyed(&backend),
        1,
        "the array the previous claim owned is nobody else's to destroy"
    );
    assert_eq!(created(&backend), 2);
    assert_eq!(
        counters.caches.evicted, 0,
        "a refused insert is not an eviction: nothing was in the way"
    );

    state.shutdown(&mut backend, &mut counters);
    assert_eq!(
        destroyed(&backend),
        2,
        "shutdown destroys the array the live claim owns as well"
    );
    assert_eq!(state.applied_input(), None);
    assert_eq!(counters.lifecycle.driver_errors, 0);
}

#[test]
fn shutdown_destroys_what_the_domain_derived_and_forgets_the_request() {
    let mut backend = backend();
    let buffer = vertex_buffer(&mut backend);
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();

    state.set_vertex_input(input(layout(12, GlVertexStepMode::Vertex), buffer, 0));
    apply(&mut state, &mut backend, &mut counters);
    assert_eq!(destroyed(&backend), 0);

    state.shutdown(&mut backend, &mut counters);

    assert_eq!(destroyed(&backend), 1);
    assert_eq!(state.vertex_array_records(), 0);
    assert_eq!(state.applied_input(), None);
    assert_eq!(state.desired_input(), None);
    assert_eq!(counters.lifecycle.driver_errors, 0);
}

#[test]
fn an_emitted_bind_counts_one_allocation_and_a_skipped_one_counts_none() {
    let mut backend = backend();
    let buffer = vertex_buffer(&mut backend);
    let mut state = geometry(CacheMode::Enabled);
    let mut counters = StateCounters::default();

    state.set_vertex_input(input(layout(12, GlVertexStepMode::Vertex), buffer, 0));
    let before = counters.steady_state_allocations;
    apply(&mut state, &mut backend, &mut counters);

    // One for the record's key and one for the claim, both on the path that
    // emits.
    assert_eq!(counters.steady_state_allocations, before + 2);
    let after = counters.steady_state_allocations;
    apply(&mut state, &mut backend, &mut counters);
    assert_eq!(
        counters.steady_state_allocations, after,
        "a skipped call allocates nothing"
    );
}
