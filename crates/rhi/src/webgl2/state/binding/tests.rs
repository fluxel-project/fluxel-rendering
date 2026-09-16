//! Tests for the shared binding-point mirror: the half of its contract that no
//! domain can reach.
//!
//! Each domain's own collection runs this machinery through a real provider and
//! a real verb, which is where its behaviour under Layer 1's validation and the
//! mock's tracing is pinned.  What is pinned here is what only this module owns:
//! the settle order, the exact tally one request produces, and the failure
//! rule's whole-map clearing.  The closure below is the only "driver" involved,
//! which is what makes those three visible without a provider in the way.

use super::*;
use crate::webgl2::api::GlError;
use crate::webgl2::state::counters::DomainCounters;

/// One domain's tallies, for an exact comparison.
fn counts(counters: &StateCounters, domain: StateDomain) -> DomainCounters {
    *counters.domain_counts(domain)
}

/// An emit that records the slot it was handed and accepts the call.
fn recording(emitted: &mut Vec<u32>) -> impl FnMut(u32, &u32) -> Result<(), GlError> + '_ {
    move |slot, _| {
        emitted.push(slot);
        Ok(())
    }
}

#[test]
fn slots_settle_in_ascending_index_order() {
    let mut points = BindingPoints::default();
    let mut counters = StateCounters::default();
    // Recorded out of order on purpose: the order is this module's contract, not
    // the caller's, because a trace whose call order depended on a hash would
    // make the differential comparison against the oracle meaningless.
    points.record(2, 20, &mut counters);
    points.record(0, 0, &mut counters);
    points.record(1, 10, &mut counters);

    let mut emitted = Vec::new();
    points
        .settle(
            StateDomain::Buffers,
            "bind",
            ExecutionMode::Optimized,
            &mut counters,
            recording(&mut emitted),
        )
        .expect("every slot is new, so every emit is accepted");

    assert_eq!(emitted, vec![0, 1, 2]);
    assert_eq!(
        counts(&counters, StateDomain::Buffers),
        DomainCounters {
            requests: 1,
            emitted: 3,
            unknown_recoveries: 3,
            ..DomainCounters::default()
        }
    );
}

#[test]
fn a_settle_with_nothing_desired_emits_nothing_and_counts_a_skip() {
    let mut points: BindingPoints<u32> = BindingPoints::default();
    let mut counters = StateCounters::default();
    let mut emitted = Vec::new();

    points
        .settle(
            StateDomain::Buffers,
            "bind",
            ExecutionMode::Optimized,
            &mut counters,
            recording(&mut emitted),
        )
        .expect("a settle with no slots cannot fail");

    assert!(emitted.is_empty());
    assert_eq!(
        counts(&counters, StateDomain::Buffers),
        DomainCounters {
            requests: 1,
            skipped: 1,
            ..DomainCounters::default()
        },
        "the request settled without a driver call, which is a skip in both modes"
    );
}

#[test]
fn a_redundant_slot_is_skipped_only_where_the_mode_allows_it() {
    let mut optimized = BindingPoints::default();
    let mut counters = StateCounters::default();
    optimized.record(0, 7, &mut counters);
    let mut emitted = Vec::new();
    optimized
        .settle(
            StateDomain::Buffers,
            "bind",
            ExecutionMode::Optimized,
            &mut counters,
            recording(&mut emitted),
        )
        .expect("the first request emits");
    assert_eq!(emitted, vec![0]);

    // The same request again.  The mirror is known to agree, so the optimized
    // machine proves it redundant and emits nothing.
    optimized.record(0, 7, &mut counters);
    emitted.clear();
    optimized
        .settle(
            StateDomain::Buffers,
            "bind",
            ExecutionMode::Optimized,
            &mut counters,
            recording(&mut emitted),
        )
        .expect("a redundant request cannot fail");
    assert!(
        emitted.is_empty(),
        "the optimized machine proved the request redundant"
    );
    assert_eq!(counts(&counters, StateDomain::Buffers).skipped, 1);

    // The oracle runs the same request against the same mirror and still emits,
    // which is the whole reason it exists: a skipping oracle would produce a
    // trace no mirror-free machine could have produced.
    let mut oracle = BindingPoints::default();
    let mut counters = StateCounters::default();
    let mut emitted = Vec::new();
    for _ in 0..2 {
        oracle.record(0, 7, &mut counters);
        oracle
            .settle(
                StateDomain::Buffers,
                "bind",
                ExecutionMode::Oracle,
                &mut counters,
                recording(&mut emitted),
            )
            .expect("the oracle emits what was asked for");
    }
    assert_eq!(emitted, vec![0, 0]);
    assert_eq!(counts(&counters, StateDomain::Buffers).skipped, 0);
    assert_eq!(counts(&counters, StateDomain::Buffers).emitted, 2);
}

#[test]
fn a_refused_emit_leaves_every_slot_unknown_rather_than_unchanged() {
    let mut points = BindingPoints::default();
    let mut counters = StateCounters::default();
    points.record(0, 0, &mut counters);
    points.record(1, 1, &mut counters);
    let mut emitted = Vec::new();
    points
        .settle(
            StateDomain::Compute,
            "bind-compute",
            ExecutionMode::Optimized,
            &mut counters,
            recording(&mut emitted),
        )
        .expect("both slots are accepted");
    assert_eq!(emitted, vec![0, 1]);
    let before = counts(&counters, StateDomain::Compute);

    // Slots 0 and 1 are restated and settle for free; slot 2 is new and is the
    // call that is refused.  The failure therefore happens after two skips,
    // which is what makes `applied` report a group of one expected call with
    // none emitted.
    let injected = GlError::Driver {
        operation: "bind-compute",
        message: "injected".into(),
    };
    points.record(0, 0, &mut counters);
    points.record(1, 1, &mut counters);
    points.record(2, 2, &mut counters);
    let refusing = injected.clone();
    let error = points
        .settle(
            StateDomain::Compute,
            "bind-compute",
            ExecutionMode::Optimized,
            &mut counters,
            move |slot, _| {
                if slot == 2 {
                    return Err(refusing.clone());
                }
                Ok(())
            },
        )
        .expect_err("the third slot is refused");

    match &error {
        StateError::Backend {
            domain,
            operation,
            applied,
            source,
        } => {
            assert_eq!(*domain, StateDomain::Compute);
            assert_eq!(*operation, "bind-compute");
            assert_eq!(*applied, PartialApplication::new(0, 1));
            assert_eq!(source, &injected);
        }
        other => panic!("a refused driver call is a backend failure, not {other:?}"),
    }
    assert_eq!(
        counts(&counters, StateDomain::Compute).emitted,
        before.emitted,
        "the refused call emitted nothing, so the tally the failure ran under is unchanged"
    );
    assert_eq!(
        counts(&counters, StateDomain::Compute).skipped,
        before.skipped + 2
    );
    assert_eq!(counters.lifecycle.driver_errors, 1);
    assert_eq!(
        counts(&counters, StateDomain::Buffers),
        DomainCounters::default(),
        "the tallies land in the domain the caller named and nowhere else"
    );

    // The group is unknown *as a whole*, so the next settle re-applies every
    // slot -- including the two the mirror was sure of.  A mirror that kept
    // believing them would be building a later skip on a belief the failure may
    // have invalidated.
    let mut emitted = Vec::new();
    points
        .settle(
            StateDomain::Compute,
            "bind-compute",
            ExecutionMode::Optimized,
            &mut counters,
            recording(&mut emitted),
        )
        .expect("the whole group is re-applied");
    assert_eq!(emitted, vec![0, 1, 2]);
    assert_eq!(
        counts(&counters, StateDomain::Compute).unknown_recoveries,
        5
    );
}

#[test]
fn forgetting_the_matching_entries_keeps_the_others_and_every_want() {
    let mut points = BindingPoints::default();
    let mut counters = StateCounters::default();
    for slot in 0..4 {
        points.record(slot, slot, &mut counters);
    }
    let mut emitted = Vec::new();
    points
        .settle(
            StateDomain::Buffers,
            "bind",
            ExecutionMode::Optimized,
            &mut counters,
            recording(&mut emitted),
        )
        .expect("every slot is new");
    assert_eq!(emitted, vec![0, 1, 2, 3]);

    // Only the even slots are forgotten, and only their *applied* half is: the
    // wants survive, so what settles again is the same request rather than a
    // default this module would have had to assume.
    points.forget_where(|value| value % 2 == 0);
    let mut re_emitted = Vec::new();
    points
        .settle(
            StateDomain::Buffers,
            "bind",
            ExecutionMode::Optimized,
            &mut counters,
            |slot, value| {
                re_emitted.push((slot, *value));
                Ok(())
            },
        )
        .expect("the forgotten slots are re-established");

    assert_eq!(
        re_emitted,
        vec![(0, 0), (2, 2)],
        "the odd slots were still known and the even ones carried their own value back"
    );
}

#[test]
fn forgetting_everything_keeps_the_wants_and_clears_only_the_beliefs() {
    let mut points = BindingPoints::default();
    let mut counters = StateCounters::default();
    points.record(0, 7, &mut counters);
    let mut emitted = Vec::new();
    points
        .settle(
            StateDomain::Buffers,
            "bind",
            ExecutionMode::Optimized,
            &mut counters,
            recording(&mut emitted),
        )
        .expect("the first request emits");

    points.forget_applied();
    let mut re_emitted = Vec::new();
    points
        .settle(
            StateDomain::Buffers,
            "bind",
            ExecutionMode::Optimized,
            &mut counters,
            |slot, value| {
                re_emitted.push((slot, *value));
                Ok(())
            },
        )
        .expect("a cleared mirror re-establishes what the caller asked for");

    assert_eq!(re_emitted, vec![(0, 7)]);
}
