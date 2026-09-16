//! What an invalidation must forget, and what it must not.
//!
//! One row of [`super::super::super::event`]'s matrix is this domain's: a deleted
//! buffer is forgotten by every role that still names it, before the backend is
//! asked to delete the name.  The other two tests here pin the boundary of that
//! obligation -- a scope that named a different domain must not disturb this one,
//! and no invalidation may drop what the caller asked for.

use super::super::super::event::ScopedRawAccess;
use super::*;

#[test]
fn deleting_a_buffer_forgets_it_in_every_role() {
    let mut fixture = two_role_backend();
    let mut state = BuffersState::new(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();

    state.bind_uniform_buffer(0, Some(fixture.first), 0, 0, &mut counters);
    state.bind_uniform_buffer(1, Some(fixture.second), 0, 0, &mut counters);
    state.bind_storage_buffer(0, range(fixture.first), &mut counters);
    state.bind_storage_buffer(1, range(fixture.second), &mut counters);
    state
        .reconcile(&mut fixture.uniform, &mut counters)
        .expect("both uniform slots emit");
    state
        .reconcile_storage(&mut fixture.storage, &mut counters)
        .expect("both storage slots emit");
    let applied = storage_mark(&fixture.storage);
    fixture.uniform.clear_calls();

    // The deletion is dispatched to the mirror before the backend is asked to
    // delete, which is the ordering that keeps a reused name from hitting a
    // record that still means the old occupant.
    state.invalidate(
        &mut fixture.uniform,
        &StateEvent::BufferDeleted(fixture.first),
        &mut counters,
    );

    assert!(
        fixture.uniform.calls().is_empty(),
        "forgetting a binding is not a driver call"
    );
    assert_eq!(
        storage_mark(&fixture.storage),
        applied,
        "a binding is not an object, so a deletion has nothing to destroy"
    );
    assert_eq!(
        counters.lifecycle.domain_invalidations, 0,
        "one buffer is not a whole-domain invalidation"
    );

    state
        .reconcile(&mut fixture.uniform, &mut counters)
        .expect("the uniform want is re-applied");
    state
        .reconcile_storage(&mut fixture.storage, &mut counters)
        .expect("the storage want is re-applied");

    assert_eq!(
        uniform_calls(&fixture.uniform),
        vec![uniform(0, Some(fixture.first), 0, 0)],
        "only the slot that named the deleted buffer is re-applied"
    );
    assert_eq!(
        storage_calls_since(&fixture.storage, applied),
        vec![storage(0, fixture.first, 0, 256)],
        "the storage role forgot it too, and the other slot was left alone"
    );
}

#[test]
fn a_declared_scope_only_forgets_when_it_names_this_domain() {
    let (mut backend, buffer) = uniform_backend();
    let mut state = BuffersState::new(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();

    state.bind_uniform_buffer(3, Some(buffer), 0, 0, &mut counters);
    state
        .reconcile(&mut backend, &mut counters)
        .expect("the first application emits");
    backend.clear_calls();

    // A scope that named another domain leaves this one believing what it
    // applied, which is the entire reason a scope declares its domains.
    state.invalidate(
        &mut backend,
        &StateEvent::ScopedRawAccess(ScopedRawAccess::declaring(DirtyDomains::of(
            StateDomain::Textures,
        ))),
        &mut counters,
    );
    state
        .reconcile(&mut backend, &mut counters)
        .expect("nothing to re-apply");
    assert!(uniform_calls(&backend).is_empty());
    assert_eq!(counters.lifecycle.domain_invalidations, 0);

    // A scope that named this domain forgets what the driver holds, and the want
    // survives to be re-applied.
    state.invalidate(
        &mut backend,
        &StateEvent::ScopedRawAccess(ScopedRawAccess::declaring(DirtyDomains::of(
            StateDomain::Buffers,
        ))),
        &mut counters,
    );
    state
        .reconcile(&mut backend, &mut counters)
        .expect("the want is re-applied");
    assert_eq!(uniform_calls(&backend).len(), 1);
    assert_eq!(counters.lifecycle.domain_invalidations, 1);

    // A scope that declared nothing is assumed to have touched everything.
    backend.clear_calls();
    state.invalidate(
        &mut backend,
        &StateEvent::ScopedRawAccess(ScopedRawAccess::all()),
        &mut counters,
    );
    state
        .reconcile(&mut backend, &mut counters)
        .expect("the want is re-applied");
    assert_eq!(uniform_calls(&backend).len(), 1);
    assert_eq!(
        counters.lifecycle.domain_invalidations, 2,
        "one undeclared scope is still one domain invalidation, not one per event"
    );
}

#[test]
fn a_whole_mirror_event_keeps_the_want_and_forgets_the_belief() {
    let (mut backend, buffer) = uniform_backend();
    let mut state = BuffersState::new(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();

    state.bind_uniform_buffer(1, Some(buffer), 256, 0, &mut counters);
    state
        .reconcile(&mut backend, &mut counters)
        .expect("the first application emits");
    backend.clear_calls();

    state.invalidate(&mut backend, &StateEvent::ContextLost, &mut counters);

    assert!(
        backend.calls().is_empty(),
        "there is no context left to emit into"
    );
    assert_eq!(counters.lifecycle.domain_invalidations, 1);
    assert_eq!(state.uniform.desired.len(), 1, "the caller still wants it");

    state
        .reconcile(&mut backend, &mut counters)
        .expect("the want is re-applied after restoration");
    assert_eq!(
        uniform_calls(&backend),
        vec![uniform(1, Some(buffer), 256, 0)]
    );
}
