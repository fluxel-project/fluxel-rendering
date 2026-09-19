//! The optional command domains at the integration point.
//!
//! The sibling file exercises the machine through one required domain, which is
//! enough to pin the domain contract.  It is not enough for the two domains whose
//! entry points carry a bound: those four methods and the `compute` arm of the
//! machine's invalidation dispatch had no caller and no test at all until the
//! wrapper under [`MockComputeStorageApi`] was completed (the plan's P1-18).  This
//! file is that test, and it is written against a machine over the *wrapper*
//! rather than over the plain command recorder, because the wrapper is the only
//! mock that satisfies the optional bound.
//!
//! What can and cannot be observed here is worth stating, because two of the
//! three contracts could be asserted either way and only one of them is
//! discriminating:
//!
//! - A **scoped** invalidation naming exactly one domain is fully observable: only
//!   that domain counts it, and only its want is re-established, so the count and
//!   the trace together say which domain the machine's dispatch reached.  This is
//!   the discriminator for the dispatch line itself.
//! - A **whole-mirror** event is observable through the re-emit: every domain
//!   forgets its beliefs and keeps its wants, so a machine that failed to dispatch
//!   to one of them would settle that domain silently instead of re-establishing
//!   it.
//! - A **deletion** is *not* observable from out here, and that is why the domain
//!   tests own it.  Whether the deleted object's want was dropped or merely left
//!   believed, the next settle emits nothing either way -- so an assertion written
//!   here would pass in both worlds.  The two domain suites reach their mirrors
//!   directly and pin it exactly; duplicating it here would be a test that proves
//!   nothing while reading as if it did.

use super::*;
use crate::webgl2::api::tests::compute_storage_snapshot;
use crate::webgl2::api::{
    GlBufferDesc, GlBufferUsage, GlExtent3d, GlFormat, GlResourceApi, GlStorageBufferRange,
    GlStorageBufferUsage, GlStorageImageAccess, GlStorageImageBinding, GlTextureDesc,
    GlTextureDimension, GlTextureUsage, MockComputeStorageApi, TextureId,
};
use crate::webgl2::state::counters::DomainCounters;
use crate::webgl2::state::event::ScopedRawAccess;
use crate::webgl2::state::knowledge::{DirtyDomains, StateDomain};

/// A machine over the optional-domain wrapper, and the two objects it binds.
///
/// The machine is built over [`MockComputeStorageApi`], which is the whole point:
/// the same value carries the required command domains and the three optional
/// ones, so one machine reaches every entry point this layer has.  A real
/// provider has that shape too -- one type serves the context, and only the
/// capability row decides which verbs it will accept.
struct Fixture {
    machine: GlStateMachine<MockComputeStorageApi>,
    buffer: BufferId,
    image: TextureId,
}

impl Fixture {
    fn new(mode: ExecutionMode) -> Self {
        let inner = MockGlFamilyApi::from_discovery(compute_storage_snapshot(true));
        let mut machine = GlStateMachine::with_mode(
            MockComputeStorageApi::new(inner).expect("the snapshot proved compute and storage"),
            mode,
        );
        let buffer = machine
            .backend()
            .create_buffer_resource(GlBufferDesc {
                size: 512,
                usage: GlBufferUsage::UNIFORM | GlBufferUsage::STORAGE,
            })
            .expect("a buffer both roles can bind");
        let image = machine
            .backend()
            .create_texture_resource(image_desc())
            .expect("a storage image");
        Self {
            machine,
            buffer,
            image,
        }
    }

    /// Where the recorded trace currently ends.
    fn mark(&mut self) -> usize {
        self.machine.backend().calls().len()
    }

    /// Every binding word recorded since `mark`, in order.
    ///
    /// Filtered to the two roles under test rather than read whole, so that a
    /// call some other part of the machine makes cannot make an exact trace
    /// comparison fail for a reason that has nothing to do with this file.
    fn binds_since(&mut self, mark: usize) -> Vec<MockCall> {
        self.machine.backend().calls()[mark..]
            .iter()
            .filter(|call| {
                matches!(
                    call,
                    MockCall::BindStorageBuffer { .. } | MockCall::BindStorageImage { .. }
                )
            })
            .cloned()
            .collect()
    }

    fn domain(&self, domain: StateDomain) -> DomainCounters {
        *self.machine.counters().domain_counts(domain)
    }

    /// The three reconciles in the order the machine's domains run in.
    fn apply(&mut self) {
        self.machine
            .apply_buffers()
            .expect("the required role has nothing wanted and settles");
        self.machine
            .apply_storage_buffers()
            .expect("the storage role is new");
        self.machine
            .apply_storage_images()
            .expect("the image unit is new");
    }

    fn range(&self) -> GlStorageBufferRange {
        GlStorageBufferRange {
            buffer: self.buffer,
            offset: 0,
            size: 256,
            usage: GlStorageBufferUsage::ReadOnly,
        }
    }

    fn image(&self) -> GlStorageImageBinding {
        GlStorageImageBinding {
            texture: self.image,
            level: 0,
            sample_count: 1,
            layered: false,
            layer: Some(0),
            format: GlFormat::Rgba8Unorm,
            access: GlStorageImageAccess::ReadWrite,
        }
    }
}

/// One storage image: single level, single sample, the one certified format.
fn image_desc() -> GlTextureDesc {
    GlTextureDesc {
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
    }
}

// ---------------------------------------------------------------------------
// The bounded entry points reach their domains.
// ---------------------------------------------------------------------------

#[test]
fn the_machine_settles_both_optional_roles_through_the_wrapper() {
    let mut fixture = Fixture::new(ExecutionMode::Optimized);
    fixture.machine.bind_storage_buffer(0, fixture.range());
    fixture.machine.bind_storage_image(0, fixture.image());
    let mark = fixture.mark();
    fixture.apply();

    assert_eq!(
        fixture.binds_since(mark),
        vec![
            MockCall::BindStorageBuffer {
                binding: 0,
                buffer: fixture.buffer,
                offset: 0,
                size: 256,
            },
            MockCall::BindStorageImage {
                binding: 0,
                texture: fixture.image,
            },
        ],
        "each bounded entry point reached the role it names, in the machine's domain order"
    );

    // The two roles are accounted against their own domains, which is what the
    // counter domain being a *parameter* of the shared settle pass is for: one
    // implementation of the redundancy rule, two sets of tallies, neither able to
    // report the other's work.
    let buffers = fixture.domain(StateDomain::Buffers);
    let compute = fixture.domain(StateDomain::Compute);
    assert_eq!(
        buffers.emitted, 1,
        "the storage range was the buffer domain's one call"
    );
    assert_eq!(compute.emitted, 1, "the image unit was compute's one call");
    assert_eq!(
        buffers.unknown_recoveries, 1,
        "the storage role's first application is a recovery, not a change"
    );
    assert_eq!(compute.unknown_recoveries, 1);
    assert_eq!(
        buffers.requests, 2,
        "the required and the optional role each asked the buffer domain to settle"
    );
    assert_eq!(
        compute.requests, 1,
        "nothing but the image unit asked the compute domain to settle"
    );
    assert_eq!(
        buffers.skipped, 1,
        "the required role settled with nothing wanted, without a driver call"
    );
    assert_eq!(compute.skipped, 0);
}

// ---------------------------------------------------------------------------
// The dispatch reaches the right domain, and the right hook.
// ---------------------------------------------------------------------------

#[test]
fn the_machine_dispatches_a_scoped_invalidation_to_the_optional_domain_it_named() {
    let mut fixture = Fixture::new(ExecutionMode::Optimized);
    fixture.machine.bind_storage_buffer(0, fixture.range());
    fixture.machine.bind_storage_image(0, fixture.image());
    fixture.apply();

    // A scope declaring exactly one domain is the event that makes the machine's
    // dispatch line itself observable.  Only the domain named counts it, and only
    // that domain re-establishes its want -- so the count of one says no *other*
    // domain reacted, and the single re-emitted word says which one did.  If the
    // machine's `compute` arm were missing, the count would be zero and the
    // compute belief would have survived to skip the next settle.
    let mark = fixture.mark();
    fixture
        .machine
        .invalidate(StateEvent::ScopedRawAccess(ScopedRawAccess::declaring(
            DirtyDomains::of(StateDomain::Compute),
        )));
    assert_eq!(fixture.machine.counters().lifecycle.domain_invalidations, 1);
    fixture
        .machine
        .apply_storage_images()
        .expect("the want survived and is re-established");
    fixture
        .machine
        .apply_storage_buffers()
        .expect("the buffer domain was not named by the scope");
    assert_eq!(
        fixture.binds_since(mark),
        vec![MockCall::BindStorageImage {
            binding: 0,
            texture: fixture.image,
        }],
        "the compute want is re-established and the buffer want is left believed"
    );

    // The same event naming the buffer role reaches the other domain and leaves
    // the compute mirror believing what it applied, which is the whole reason a
    // scope declares its domains.
    let mark = fixture.mark();
    fixture
        .machine
        .invalidate(StateEvent::ScopedRawAccess(ScopedRawAccess::declaring(
            DirtyDomains::of(StateDomain::Buffers),
        )));
    assert_eq!(fixture.machine.counters().lifecycle.domain_invalidations, 2);
    fixture
        .machine
        .apply_storage_buffers()
        .expect("the want survived and is re-established");
    fixture
        .machine
        .apply_storage_images()
        .expect("the compute domain was not named by the scope");
    assert_eq!(
        fixture.binds_since(mark),
        vec![MockCall::BindStorageBuffer {
            binding: 0,
            buffer: fixture.buffer,
            offset: 0,
            size: 256,
        }],
        "the buffer want is re-established and the compute want is left believed"
    );

    // The two roles are separate mirrors and neither scope disturbed the other
    // one's belief, so both are still known to the mirror after both events.
    assert_eq!(fixture.domain(StateDomain::Compute).emitted, 2);
    assert_eq!(fixture.domain(StateDomain::Buffers).emitted, 2);
}

#[test]
fn the_machine_dispatches_a_whole_mirror_event_to_both_optional_domains() {
    let mut fixture = Fixture::new(ExecutionMode::Optimized);
    fixture.machine.bind_storage_buffer(0, fixture.range());
    fixture.machine.bind_storage_image(0, fixture.image());
    fixture.apply();

    fixture.machine.invalidate(StateEvent::ContextLost);
    let mark = fixture.mark();
    fixture
        .machine
        .apply_storage_buffers()
        .expect("the want is re-established after restoration");
    fixture
        .machine
        .apply_storage_images()
        .expect("the want is re-established after restoration");
    assert_eq!(
        fixture.binds_since(mark),
        vec![
            MockCall::BindStorageBuffer {
                binding: 0,
                buffer: fixture.buffer,
                offset: 0,
                size: 256,
            },
            MockCall::BindStorageImage {
                binding: 0,
                texture: fixture.image,
            },
        ],
        "a lost context leaves every want standing and no belief, in both roles"
    );
    assert!(
        fixture.machine.counters().lifecycle.domain_invalidations > 1,
        "a whole-mirror event is counted once per domain that owns it, not once"
    );
}

// ---------------------------------------------------------------------------
// The oracle, through the bounded entry points.
// ---------------------------------------------------------------------------

/// The mode's contract has to hold for the optional domains too.
///
/// The sibling file asserts it for the session domain, which is required; this is
/// the same property reached through the two bounded entry points.  It matters
/// more here than there: a machine over a compute-capable profile is exactly where
/// the differential test runs, so an oracle that skipped in these two roles would
/// produce a trace no mirror-free machine could have produced in the roles the
/// comparison is most sensitive to.
#[test]
fn an_oracle_machine_emits_the_optional_requests_the_optimized_one_skips() {
    let mut optimized = Fixture::new(ExecutionMode::Optimized);
    let mark = optimized.mark();
    for _ in 0..2 {
        optimized.machine.bind_storage_buffer(0, optimized.range());
        optimized.machine.bind_storage_image(0, optimized.image());
        optimized.apply();
    }
    assert_eq!(
        optimized.binds_since(mark).len(),
        2,
        "the second request for each role is the same request and is skipped"
    );
    assert_eq!(optimized.domain(StateDomain::Compute).skipped, 1);
    assert_eq!(optimized.machine.mode(), ExecutionMode::Optimized);

    let mut oracle = Fixture::new(ExecutionMode::Oracle);
    let mark = oracle.mark();
    for _ in 0..2 {
        oracle.machine.bind_storage_buffer(0, oracle.range());
        oracle.machine.bind_storage_image(0, oracle.image());
        oracle.apply();
    }
    assert_eq!(
        oracle.binds_since(mark).len(),
        4,
        "the oracle re-establishes both roles instead of skipping them"
    );
    assert_eq!(oracle.domain(StateDomain::Compute).skipped, 0);
    assert_eq!(
        oracle.domain(StateDomain::Buffers).skipped,
        2,
        "only the required role's empty settle is skipped, once per transition"
    );
    assert_eq!(oracle.machine.mode(), ExecutionMode::Oracle);
}
