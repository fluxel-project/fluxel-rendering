//! Trace tests for the texture state domain.
//!
//! Every test runs the domain against the Layer 1 recorder and asserts the *trace*
//! -- which calls were emitted, how many, and in what order -- because the mirror's
//! own fields are what the domain believes, and a belief is exactly what a wrong
//! skip is made of.  A double written for these tests would answer with whatever
//! trace the test expected; the recorder is the same provider the Layer 1 contract
//! suites run against, so a change in Layer 1's call surface shows up here as a
//! failing trace instead of as a mirror that quietly agrees with itself.
//!
//! Not covered here: the counter report as a whole (the counters module owns its
//! own tests) and the machine's dispatch of an invalidation to this domain (the
//! machine's test module owns that).

use super::*;
use crate::webgl2::state::event::ScopedRawAccess;
use crate::webgl2::state::knowledge::DirtyDomains;

/// The stage these tests run on, in its own file so that this one stays a list of
/// assertions rather than a list of assertions plus a construction manual.
mod fixture;
use fixture::Fixture;

#[test]
fn a_fresh_domain_applies_nothing_until_it_is_asked() {
    let mut fixture = Fixture::new(ExecutionMode::Optimized);

    fixture.apply();

    assert!(
        fixture.trace_len() == 0,
        "a first reconcile must not establish a unit, a target or a sampler: every one of those would be a guess about the driver"
    );
    assert_eq!(
        fixture.domain.applied_active_unit(),
        DriverKnowledge::Unknown
    );
    assert_eq!(fixture.counts().skipped, 1);
    assert_eq!(fixture.counts().emitted, 0);
}

#[test]
fn a_redundant_bind_emits_nothing_and_leaves_the_active_unit_alone() {
    let mut fixture = Fixture::new(ExecutionMode::Optimized);

    fixture
        .domain
        .bind_texture(3, GlTextureTarget::D2, Some(fixture.first));
    fixture.apply();
    assert_eq!(
        fixture.domain.applied_active_unit(),
        DriverKnowledge::Known(3),
        "the bind selected the unit, so the mirror has to record the move"
    );

    fixture.domain.active_texture(7);
    fixture.apply();
    assert_eq!(
        fixture.domain.applied_active_unit(),
        DriverKnowledge::Known(7)
    );

    let emitted = fixture.trace_len();
    fixture
        .domain
        .bind_texture(3, GlTextureTarget::D2, Some(fixture.first));
    fixture.apply();

    assert_eq!(
        fixture.trace_len(),
        emitted,
        "the slot already names that texture, so the whole verb is skipped"
    );
    assert_eq!(
        fixture.domain.applied_active_unit(),
        DriverKnowledge::Known(7),
        "a skipped bind emitted nothing, so it cannot have moved the unit it names"
    );
    assert_eq!(fixture.counts().requests, 3);
    assert_eq!(fixture.counts().emitted, 2);
    assert_eq!(fixture.counts().skipped, 1);
}

#[test]
fn a_bind_that_ran_makes_a_later_selection_of_that_unit_redundant() {
    let mut fixture = Fixture::new(ExecutionMode::Optimized);

    fixture
        .domain
        .bind_texture(4, GlTextureTarget::Cube, Some(fixture.first));
    fixture.apply();
    fixture.domain.active_texture(4);
    fixture.apply();

    assert!(
        fixture.selected_units().is_empty(),
        "the bind already selected unit 4, so the explicit selection proves redundant and emits nothing"
    );
    assert_eq!(fixture.counts().skipped, 1);
    assert_eq!(fixture.counts().unknown_recoveries, 1);
}

#[test]
fn a_skipped_bind_must_not_make_the_mirror_claim_the_unit_it_names() {
    let mut fixture = Fixture::new(ExecutionMode::Optimized);

    fixture
        .domain
        .bind_texture(3, GlTextureTarget::D2, Some(fixture.first));
    fixture.apply();
    fixture.domain.active_texture(5);
    fixture.apply();

    // Redundant, and the trap: the driver is still on unit 5 because nothing was
    // emitted, so the mirror must not pick the unit up from the request.
    fixture
        .domain
        .bind_texture(3, GlTextureTarget::D2, Some(fixture.first));
    fixture.apply();
    assert_eq!(
        fixture.domain.applied_active_unit(),
        DriverKnowledge::Known(5)
    );

    fixture.domain.active_texture(3);
    fixture.apply();

    assert_eq!(
        fixture.selected_units(),
        vec![5, 3],
        "the selection has to be emitted: the skipped bind left the driver on unit 5"
    );
    assert_eq!(
        fixture.domain.applied_active_unit(),
        DriverKnowledge::Known(3)
    );
}

#[test]
fn the_target_is_part_of_the_binding_identity() {
    let mut fixture = Fixture::new(ExecutionMode::Optimized);

    fixture
        .domain
        .bind_texture(0, GlTextureTarget::D2, Some(fixture.first));
    fixture.apply();
    fixture
        .domain
        .bind_texture(0, GlTextureTarget::D2Array, Some(fixture.first));
    fixture.apply();

    assert_eq!(
        fixture.bound_textures(),
        vec![
            (0, GlTextureTarget::D2, Some(fixture.first)),
            (0, GlTextureTarget::D2Array, Some(fixture.first)),
        ],
        "one unit holds one binding per target, so the second request is not a repeat"
    );

    // A slot this domain has never applied is unknown even when the unit and the
    // target are otherwise described, so an unbind there is emitted rather than
    // assumed.
    fixture.domain.bind_texture(0, GlTextureTarget::D3, None);
    fixture.apply();
    assert_eq!(fixture.bound_textures().len(), 3);
}

#[test]
fn binding_a_sampler_does_not_move_the_active_unit() {
    let mut fixture = Fixture::new(ExecutionMode::Optimized);

    fixture
        .domain
        .bind_texture(3, GlTextureTarget::D2, Some(fixture.first));
    fixture.apply();
    fixture.domain.active_texture(6);
    fixture.apply();

    fixture.domain.bind_sampler(6, Some(fixture.sampler));
    fixture.apply();
    assert_eq!(
        fixture.bound_samplers(),
        vec![(6, Some(fixture.sampler))],
        "the sampler verb is addressed by unit index and shares the unit index space"
    );
    assert_eq!(
        fixture.domain.applied_active_unit(),
        DriverKnowledge::Known(6),
        "the unit index it is addressed by is not a selection of that unit"
    );

    fixture.domain.active_texture(6);
    fixture.apply();
    assert_eq!(
        fixture.selected_units(),
        vec![6],
        "the selection after the sampler bind is still redundant, because the sampler bind did not move the unit"
    );
}

#[test]
fn the_texture_and_sampler_slots_of_one_unit_are_independent() {
    let mut fixture = Fixture::new(ExecutionMode::Optimized);

    fixture
        .domain
        .bind_texture(2, GlTextureTarget::D2, Some(fixture.first));
    fixture.apply();
    fixture.domain.bind_sampler(2, Some(fixture.sampler));
    fixture.apply();

    // An unbind of the texture slot must not disturb the sampler slot, and a
    // repeat of the sampler bind must still be skipped after it.
    fixture.domain.bind_texture(2, GlTextureTarget::D2, None);
    fixture.apply();
    fixture.domain.bind_sampler(2, Some(fixture.sampler));
    fixture.apply();

    assert_eq!(
        fixture.bound_textures(),
        vec![
            (2, GlTextureTarget::D2, Some(fixture.first)),
            (2, GlTextureTarget::D2, None),
        ]
    );
    assert_eq!(
        fixture.bound_samplers(),
        vec![(2, Some(fixture.sampler))],
        "both verbs address unit 2, and neither request disturbs the other's slot"
    );
    assert_eq!(
        fixture.domain.applied_texture(2, GlTextureTarget::D2),
        DriverKnowledge::Known(None)
    );
    assert_eq!(
        fixture.domain.applied_sampler(2),
        DriverKnowledge::Known(Some(fixture.sampler))
    );
}

#[test]
fn deleting_a_bound_texture_clears_every_slot_that_named_it() {
    let mut fixture = Fixture::new(ExecutionMode::Optimized);

    fixture
        .domain
        .bind_texture(0, GlTextureTarget::D2, Some(fixture.first));
    fixture.apply();
    fixture
        .domain
        .bind_texture(2, GlTextureTarget::D2Array, Some(fixture.first));
    fixture.apply();
    fixture
        .domain
        .bind_texture(1, GlTextureTarget::D2, Some(fixture.second));
    fixture.apply();

    fixture.invalidate(StateEvent::TextureDeleted(fixture.first));

    assert_eq!(
        fixture.domain.applied_texture(0, GlTextureTarget::D2),
        DriverKnowledge::Unknown
    );
    assert_eq!(
        fixture.domain.applied_texture(2, GlTextureTarget::D2Array),
        DriverKnowledge::Unknown
    );
    assert_eq!(
        fixture.domain.applied_texture(1, GlTextureTarget::D2),
        DriverKnowledge::Known(Some(fixture.second)),
        "a slot that never named the deleted texture keeps what it knows"
    );
    assert_eq!(fixture.counts().skipped, 0, "a deletion is not a request");
    assert_eq!(
        fixture.counters.lifecycle.domain_invalidations, 0,
        "one deletion is not a whole-domain invalidation"
    );

    // The cleared slots are applied again rather than skipped, and the untouched
    // slot is still skipped.
    let emitted = fixture.trace_len();
    let recoveries = fixture.counts().unknown_recoveries;
    fixture.domain.bind_texture(0, GlTextureTarget::D2, None);
    fixture.apply();
    assert_eq!(fixture.trace_len(), emitted + 1);
    fixture
        .domain
        .bind_texture(1, GlTextureTarget::D2, Some(fixture.second));
    fixture.apply();
    assert_eq!(fixture.trace_len(), emitted + 1);
    assert_eq!(
        fixture.counts().unknown_recoveries,
        recoveries + 1,
        "the cleared slot is re-applied from unknown; the surviving one is not"
    );
}

#[test]
fn a_request_naming_a_deleted_texture_is_forgotten_rather_than_emitted() {
    let mut fixture = Fixture::new(ExecutionMode::Optimized);

    fixture
        .domain
        .bind_texture(4, GlTextureTarget::D2, Some(fixture.first));
    // The deletion is dispatched before the backend deletes, which is the window a
    // request can be stranded in.
    fixture.invalidate(StateEvent::TextureDeleted(fixture.first));

    let emitted = fixture.trace_len();
    fixture.apply();

    assert_eq!(
        fixture.trace_len(),
        emitted,
        "the identity is gone, so its bind must not be emitted"
    );
    assert_eq!(
        fixture.domain.applied_texture(4, GlTextureTarget::D2),
        DriverKnowledge::Unknown
    );
}

#[test]
fn deleting_a_bound_sampler_clears_its_unit() {
    let mut fixture = Fixture::new(ExecutionMode::Optimized);

    fixture.domain.bind_sampler(2, Some(fixture.sampler));
    fixture.apply();

    fixture.invalidate(StateEvent::SamplerDeleted(fixture.sampler));
    assert_eq!(fixture.domain.applied_sampler(2), DriverKnowledge::Unknown);
    assert_eq!(fixture.domain.applied_sampler(3), DriverKnowledge::Unknown);

    // The same two rules as a texture's deletion: the cleared slot is applied
    // again, and a request naming the deleted sampler is forgotten.
    fixture.domain.bind_sampler(2, None);
    fixture.apply();
    assert_eq!(
        fixture.bound_samplers(),
        vec![(2, Some(fixture.sampler)), (2, None)]
    );

    fixture.domain.bind_sampler(5, Some(fixture.sampler));
    fixture.invalidate(StateEvent::SamplerDeleted(fixture.sampler));
    let emitted = fixture.trace_len();
    fixture.apply();
    assert_eq!(fixture.trace_len(), emitted);
}

#[test]
fn a_context_loss_forgets_the_mirror_without_emitting_anything() {
    let mut fixture = Fixture::new(ExecutionMode::Optimized);

    fixture
        .domain
        .bind_texture(1, GlTextureTarget::D2, Some(fixture.first));
    fixture.apply();
    fixture.domain.bind_sampler(1, Some(fixture.sampler));
    fixture.apply();

    let emitted = fixture.trace_len();
    fixture.invalidate(StateEvent::ContextLost);

    assert_eq!(
        fixture.trace_len(),
        emitted,
        "an epoch that is gone is not callable, so nothing may be emitted through it"
    );
    assert_eq!(
        fixture.domain.applied_active_unit(),
        DriverKnowledge::Unknown
    );
    assert_eq!(
        fixture.domain.applied_texture(1, GlTextureTarget::D2),
        DriverKnowledge::Unknown
    );
    assert_eq!(fixture.domain.applied_sampler(1), DriverKnowledge::Unknown);
    assert_eq!(fixture.counters.lifecycle.domain_invalidations, 1);

    // The request survives the loss: the caller asked for it, and the next
    // reconcile after restoration is what applies it again.
    fixture.apply();
    assert_eq!(
        fixture.bound_samplers(),
        vec![(1, Some(fixture.sampler)), (1, Some(fixture.sampler))],
        "the last request is re-applied against the restored context"
    );
    assert_eq!(
        fixture.domain.applied_sampler(1),
        DriverKnowledge::Known(Some(fixture.sampler))
    );
}

#[test]
fn a_declared_scope_forgets_only_the_domain_it_declared() {
    let mut fixture = Fixture::new(ExecutionMode::Optimized);

    fixture
        .domain
        .bind_texture(0, GlTextureTarget::D2, Some(fixture.first));
    fixture.apply();

    fixture.invalidate(StateEvent::ScopedRawAccess(ScopedRawAccess::declaring(
        DirtyDomains::of(StateDomain::Pipeline),
    )));
    assert_eq!(
        fixture.domain.applied_texture(0, GlTextureTarget::D2),
        DriverKnowledge::Known(Some(fixture.first)),
        "a scope that named other domains left this one alone, which is the point of declaring them"
    );
    assert_eq!(fixture.counters.lifecycle.domain_invalidations, 0);

    fixture.invalidate(StateEvent::ScopedRawAccess(ScopedRawAccess::declaring(
        DirtyDomains::of(StateDomain::Textures),
    )));
    assert_eq!(
        fixture.domain.applied_texture(0, GlTextureTarget::D2),
        DriverKnowledge::Unknown
    );
    assert_eq!(fixture.counters.lifecycle.domain_invalidations, 1);

    // A scope that declared nothing must be assumed to have touched everything.
    fixture.apply();
    fixture.invalidate(StateEvent::ScopedRawAccess(ScopedRawAccess::all()));
    assert_eq!(
        fixture.domain.applied_texture(0, GlTextureTarget::D2),
        DriverKnowledge::Unknown
    );
    assert_eq!(fixture.counters.lifecycle.domain_invalidations, 2);
}

#[test]
fn a_refused_bind_is_reported_against_this_domain_and_leaves_no_claim() {
    let mut fixture = Fixture::new(ExecutionMode::Optimized);

    fixture.backend.fail_next(GlError::Validation {
        operation: "bind-texture",
        message: "injected refusal".into(),
    });
    fixture
        .domain
        .bind_texture(3, GlTextureTarget::D2, Some(fixture.first));
    let error = fixture.reconcile().expect_err("the refusal surfaces");

    assert_eq!(error.domain(), StateDomain::Textures);
    assert_eq!(error.operation(), "bind-texture");
    assert!(
        !error.is_pre_side_effect(),
        "the backend refused a call this domain had already made"
    );
    assert_eq!(
        fixture.domain.applied_texture(3, GlTextureTarget::D2),
        DriverKnowledge::Unknown,
        "a refused bind may have selected the unit before it failed, so the mirror claims nothing at all"
    );
    assert_eq!(
        fixture.domain.applied_active_unit(),
        DriverKnowledge::Unknown
    );
    assert_eq!(fixture.counts().emitted, 0);
    assert_eq!(fixture.counters.lifecycle.driver_errors, 1);
    assert!(
        fixture.bound_textures().is_empty(),
        "a failed call is not a call the trace shows"
    );

    // The request survives the failure and the next reconcile applies it in full.
    fixture.apply();
    assert_eq!(
        fixture.bound_textures(),
        vec![(3, GlTextureTarget::D2, Some(fixture.first))]
    );
    assert_eq!(
        fixture.domain.applied_active_unit(),
        DriverKnowledge::Known(3)
    );

    // The unit selection reports its own operation, which is the name a caller
    // needs to tell the two failure paths apart.
    fixture.backend.fail_next(GlError::Validation {
        operation: "active-texture",
        message: "injected refusal".into(),
    });
    fixture.domain.active_texture(9);
    let error = fixture.reconcile().expect_err("the refusal surfaces");
    assert_eq!(error.domain(), StateDomain::Textures);
    assert_eq!(error.operation(), "active-texture");
    assert_eq!(fixture.counters.lifecycle.driver_errors, 2);
}

#[test]
fn the_oracle_emits_every_request_and_still_keeps_the_mirror() {
    let mut fixture = Fixture::new(ExecutionMode::Oracle);

    for _ in 0..2 {
        fixture
            .domain
            .bind_texture(0, GlTextureTarget::D2, Some(fixture.first));
        fixture.apply();
    }

    assert_eq!(
        fixture.bound_textures().len(),
        2,
        "an oracle skips nothing, so its trace is the one a mirror-free machine would have emitted"
    );
    assert_eq!(fixture.counts().skipped, 0);
    assert_eq!(
        fixture.domain.applied_texture(0, GlTextureTarget::D2),
        DriverKnowledge::Known(Some(fixture.first)),
        "the mirror is still kept, because it is what the differential test compares"
    );
}

#[test]
fn the_oracle_never_selects_a_unit_it_has_not_been_asked_for() {
    let mut fixture = Fixture::new(ExecutionMode::Oracle);

    // Emitting everything must not become emitting more than a mirror-free machine
    // would: the mode disables skipping, not the request model.
    fixture.apply();
    assert_eq!(fixture.trace_len(), 0, "there was no request to emit");
    assert_eq!(
        fixture.domain.desired_active_unit(),
        None,
        "no request about the active unit has been made"
    );
}
