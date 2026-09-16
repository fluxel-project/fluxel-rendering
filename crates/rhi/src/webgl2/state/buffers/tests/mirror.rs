//! Which requests are the same request.
//!
//! The whole redundancy rule is one predicate, [`DriverKnowledge::agrees`], and
//! these tests pin what it compares: the whole entry the verb carried, per role.
//! A request that differs in any part of that entry is a different request, and a
//! slot the two roles number the same way is two slots.

use super::*;

#[test]
fn a_vacuous_reconcile_emits_nothing() {
    let (mut backend, _) = uniform_backend();
    let mut state = BuffersState::new();
    let mut counters = StateCounters::default();

    state
        .reconcile(&mut backend, &mut counters)
        .expect("settling nothing cannot fail");

    assert!(uniform_calls(&backend).is_empty());
    assert_eq!(
        counters_for(&counters),
        DomainCounters {
            requests: 1,
            emitted: 0,
            skipped: 1,
            unknown_recoveries: 0,
        },
        "the domain was asked to settle and settled without a driver call"
    );
}

#[test]
fn a_redundant_request_emits_nothing() {
    let (mut backend, buffer) = uniform_backend();
    let mut state = BuffersState::new();
    let mut counters = StateCounters::default();

    state.bind_uniform_buffer(2, Some(buffer), 0, 0, &mut counters);
    state
        .reconcile(&mut backend, &mut counters)
        .expect("the first request emits");
    assert_eq!(
        uniform_calls(&backend),
        vec![uniform(2, Some(buffer), 0, 0)]
    );

    // The caller restating the binding point it already installed is the common
    // case in a renderer that re-submits the same group every frame.
    state.bind_uniform_buffer(2, Some(buffer), 0, 0, &mut counters);
    state
        .reconcile(&mut backend, &mut counters)
        .expect("a redundant request is not a failure");

    assert_eq!(
        uniform_calls(&backend),
        vec![uniform(2, Some(buffer), 0, 0)],
        "the driver must not be told twice"
    );
    assert_eq!(
        counters_for(&counters),
        DomainCounters {
            requests: 2,
            emitted: 1,
            skipped: 1,
            unknown_recoveries: 1,
        }
    );
}

#[test]
fn the_same_buffer_at_a_different_range_is_not_redundant() {
    let (mut backend, buffer) = uniform_backend();
    let mut state = BuffersState::new();
    let mut counters = StateCounters::default();

    // Three requests that name one buffer and three different ranges.  None is
    // redundant with another: the verb carries the offset and the size the caller
    // wrote, and `size == 0` is "through the allocation end" rather than
    // "unspecified".
    for (offset, size) in [(0, 0), (0, 256), (256, 0)] {
        state.bind_uniform_buffer(2, Some(buffer), offset, size, &mut counters);
        state
            .reconcile(&mut backend, &mut counters)
            .expect("each range is its own request");
    }

    assert_eq!(
        uniform_calls(&backend),
        vec![
            uniform(2, Some(buffer), 0, 0),
            uniform(2, Some(buffer), 0, 256),
            uniform(2, Some(buffer), 256, 0),
        ]
    );
    assert_eq!(counters_for(&counters).emitted, 3);
    assert_eq!(counters_for(&counters).skipped, 0);
}

#[test]
fn an_unbind_is_a_request_of_its_own() {
    let (mut backend, buffer) = uniform_backend();
    let mut state = BuffersState::new();
    let mut counters = StateCounters::default();

    state.bind_uniform_buffer(0, Some(buffer), 0, 0, &mut counters);
    state
        .reconcile(&mut backend, &mut counters)
        .expect("the bind emits");
    state.bind_uniform_buffer(0, None, 0, 0, &mut counters);
    state
        .reconcile(&mut backend, &mut counters)
        .expect("the unbind emits");

    assert_eq!(
        uniform_calls(&backend),
        vec![uniform(0, Some(buffer), 0, 0), uniform(0, None, 0, 0)],
        "a slot the caller unbound is not a slot the mirror also holds a buffer in"
    );

    // And the unbind is itself a value: restating it is redundant, which is what
    // separates "the driver holds this slot unbound" from "the driver's value
    // here is not known".
    state.bind_uniform_buffer(0, None, 0, 0, &mut counters);
    state
        .reconcile(&mut backend, &mut counters)
        .expect("a restated unbind is redundant");
    assert_eq!(uniform_calls(&backend).len(), 2);
    assert_eq!(counters_for(&counters).skipped, 1);
}

#[test]
fn the_two_roles_do_not_share_one_mirror_entry() {
    let mut fixture = two_role_backend();
    let mut state = BuffersState::new();
    let mut counters = StateCounters::default();
    let mark = storage_mark(&fixture.storage);

    // One slot number, one buffer, two roles whose index spaces are unrelated.
    // If the roles shared a table keyed by slot, the second reconcile below would
    // skip and the storage binding point would hold a uniform range.
    state.bind_uniform_buffer(0, Some(fixture.first), 0, 0, &mut counters);
    state.bind_storage_buffer(0, range(fixture.first), &mut counters);

    state
        .reconcile_storage(&mut fixture.storage, &mut counters)
        .expect("the storage role emits");
    assert_eq!(
        storage_calls_since(&fixture.storage, mark),
        vec![storage(0, fixture.first, 0, 256)]
    );

    state
        .reconcile(&mut fixture.uniform, &mut counters)
        .expect("the uniform role still owes its own call");
    assert_eq!(
        uniform_calls(&fixture.uniform),
        vec![uniform(0, Some(fixture.first), 0, 0)]
    );

    // Each role recorded its own want, and each reconcile counted its own
    // request: nothing was skipped, because nothing was already settled.
    assert_eq!(state.uniform.desired.len(), 1);
    assert_eq!(state.storage.desired.len(), 1);
    assert_eq!(
        counters_for(&counters),
        DomainCounters {
            requests: 2,
            emitted: 2,
            skipped: 0,
            unknown_recoveries: 2,
        }
    );
}

#[test]
fn a_redundant_storage_request_emits_nothing() {
    let mut fixture = two_role_backend();
    let mut state = BuffersState::new();
    let mut counters = StateCounters::default();
    let mark = storage_mark(&fixture.storage);
    let wanted = range(fixture.first);

    state.bind_storage_buffer(2, wanted, &mut counters);
    state
        .reconcile_storage(&mut fixture.storage, &mut counters)
        .expect("the first request emits");
    state.bind_storage_buffer(2, wanted, &mut counters);
    state
        .reconcile_storage(&mut fixture.storage, &mut counters)
        .expect("a redundant request is not a failure");

    assert_eq!(
        storage_calls_since(&fixture.storage, mark),
        vec![storage(2, fixture.first, 0, 256)]
    );
    assert_eq!(
        counters_for(&counters),
        DomainCounters {
            requests: 2,
            emitted: 1,
            skipped: 1,
            unknown_recoveries: 1,
        }
    );
}

#[test]
fn a_storage_request_that_differs_only_in_usage_is_re_emitted() {
    let mut fixture = two_role_backend();
    let mut state = BuffersState::new();
    let mut counters = StateCounters::default();
    let mark = storage_mark(&fixture.storage);

    state.bind_storage_buffer(0, range(fixture.first), &mut counters);
    state
        .reconcile_storage(&mut fixture.storage, &mut counters)
        .expect("the first request emits");

    // The declared usage is validated against the allocation and never reaches a
    // binding point, but the entry holds it and the skip predicate compares whole
    // entries, so this re-emits.  That is the conservative direction: one call
    // that was not strictly needed, never a skipped call that was needed.
    let mut read_write = range(fixture.first);
    read_write.usage = GlStorageBufferUsage::ReadWrite;
    state.bind_storage_buffer(0, read_write, &mut counters);
    state
        .reconcile_storage(&mut fixture.storage, &mut counters)
        .expect("the changed entry emits");

    assert_eq!(
        storage_calls_since(&fixture.storage, mark),
        vec![
            storage(0, fixture.first, 0, 256),
            storage(0, fixture.first, 0, 256)
        ],
        "the descriptor changed, so the binding point must be told again"
    );
    assert_eq!(counters_for(&counters).emitted, 2);
    assert_eq!(counters_for(&counters).skipped, 0);
}
