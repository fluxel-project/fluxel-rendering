//! What a refused call carries out.
//!
//! Two refusals reach this domain and both must be reported as the backend's
//! own, because neither this layer nor the caller can tell them apart from the
//! outside: Layer 1 rejecting a binding point outside its index space, and the
//! driver refusing a call it accepted the shape of.  The second test is the one
//! the domain contract is judged on -- a failure leaves the applied state
//! *unknown* rather than unchanged.

use super::*;

#[test]
fn a_binding_point_outside_the_index_space_is_refused_by_layer_one() {
    let (mut backend, buffer) = uniform_backend();
    let mut state = BuffersState::new(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();

    // The WebGL2 fixture proves 24 indexed uniform binding points, so index 24
    // does not name one.  The domain asks, Layer 1 refuses, and the domain reports
    // the refusal rather than inventing a second bounds check that could drift
    // from this one.
    state.bind_uniform_buffer(24, Some(buffer), 0, 0, &mut counters);
    let error = state
        .reconcile(&mut backend, &mut counters)
        .expect_err("the index is outside the discovered count");

    assert_eq!(error.domain(), StateDomain::Buffers);
    assert_eq!(error.operation(), "bind-uniform-buffer");
    assert!(
        !error.is_pre_side_effect(),
        "the backend reported it, so this layer cannot claim no call was made"
    );
    assert_eq!(counters.lifecycle.driver_errors, 1);
    assert!(uniform_calls(&backend).is_empty());
    assert_eq!(counters_for(&counters).emitted, 0);
}

#[test]
fn a_storage_binding_point_outside_its_own_space_is_refused_too() {
    let mut fixture = two_role_backend();
    let mut state = BuffersState::new(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();
    let mark = storage_mark(&fixture.storage);

    // The storage index space is narrower than the uniform one, which is why the
    // two roles cannot share a bounds check or a table.
    state.bind_storage_buffer(8, range(fixture.first), &mut counters);
    let error = state
        .reconcile_storage(&mut fixture.storage, &mut counters)
        .expect_err("the binding is outside the discovered storage count");

    assert_eq!(error.domain(), StateDomain::Buffers);
    assert_eq!(error.operation(), "bind-storage-buffer");
    assert_eq!(counters.lifecycle.driver_errors, 1);
    assert!(storage_calls_since(&fixture.storage, mark).is_empty());
    assert_eq!(counters_for(&counters).emitted, 0);
}

#[test]
fn a_driver_failure_leaves_the_whole_group_unknown() {
    let (mut backend, buffer) = uniform_backend();
    let mut state = BuffersState::new(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();

    state.bind_uniform_buffer(0, Some(buffer), 0, 0, &mut counters);
    state
        .reconcile(&mut backend, &mut counters)
        .expect("slot 0 is established");
    backend.clear_calls();

    // Slot 0 is restated and settles for free; slot 1 is new and is the call the
    // driver refuses.  The failure therefore happens after a skip, which is what
    // makes `applied` report a group of one expected call with none emitted.
    state.bind_uniform_buffer(0, Some(buffer), 0, 0, &mut counters);
    state.bind_uniform_buffer(1, Some(buffer), 0, 0, &mut counters);
    let injected = GlError::Driver {
        operation: "bind-uniform-buffer",
        message: "injected".into(),
    };
    backend.fail_next(injected.clone());
    let error = state
        .reconcile(&mut backend, &mut counters)
        .expect_err("the driver refuses the second slot");

    assert_eq!(error.domain(), StateDomain::Buffers);
    assert_eq!(error.operation(), "bind-uniform-buffer");
    assert!(!error.is_pre_side_effect());
    match &error {
        StateError::Backend {
            applied, source, ..
        } => {
            assert_eq!(*applied, PartialApplication::new(0, 1));
            assert_eq!(source, &injected);
        }
        other => panic!("a refused driver call is a backend failure, not {other:?}"),
    }
    assert_eq!(counters.lifecycle.driver_errors, 1);
    assert_eq!(
        counters_for(&counters).emitted,
        1,
        "only the first reconcile emitted"
    );
    assert_eq!(counters_for(&counters).skipped, 1);
    assert_eq!(
        counters_for(&counters).unknown_recoveries,
        1,
        "a refused call recovered nothing and is not counted as one"
    );

    // The group is unknown *as a whole*, so the next reconcile re-applies every
    // slot -- including the one the driver had already accepted.  A mirror that
    // kept believing slot 0 would be building a later skip on a belief the failure
    // may have invalidated.
    backend.clear_calls();
    state
        .reconcile(&mut backend, &mut counters)
        .expect("the group is re-applied in full");

    assert_eq!(
        uniform_calls(&backend),
        vec![
            uniform(0, Some(buffer), 0, 0),
            uniform(1, Some(buffer), 0, 0)
        ],
        "the whole group is re-applied, in ascending slot order"
    );
    assert_eq!(counters_for(&counters).unknown_recoveries, 3);
}
