//! The indexed binding-point mirror: what the caller asked each slot of one
//! index space to hold, what the driver is known to hold there, and the pass
//! that emits exactly the calls closing the difference.
//!
//! Responsibility: hold those two maps and settle them, for one binding role.
//!
//! Not owned here: what a slot's value *means*.  A value is an opaque `Copy`
//! payload to this module -- a uniform binding, a storage range, a storage-image
//! binding -- and the only question asked of it is whether two of them are the
//! same request, which is its own `PartialEq`.  Nothing here validates a value,
//! resolves an identity or reads a capability: those are Layer 1's checks, made
//! on every path including the ones this layer skips, and a second definition of
//! validity here would drift from that one and win by being cheaper.
//!
//! Not owned here either: which index space a set of slots belongs to, and which
//! domain a settle is accounted against.  Both are the caller's, because both
//! are facts about *which verb* is being mirrored rather than about mirroring.
//! Two roles with unrelated index spaces get two [`BindingPoints`], which is
//! what makes an index legal for one role impossible to use for the other; the
//! counter domain is a parameter so that one implementation serves two domains
//! without either owning the other's tallies.
//!
//! # The one predicate a skip may be built on
//!
//! A call is skipped only when [`DriverKnowledge::agrees`] says the driver is
//! *known* to hold exactly the requested value.  There is deliberately no
//! structural shortcut and no comparison against an assumed default: a slot this
//! mirror has never set is `Unknown`, and `Unknown` never agrees, so the first
//! request for any slot always emits the call that establishes it.
//!
//! # Why the settle order is fixed
//!
//! Slots settle in ascending index order, which is the order a `BTreeMap`
//! iterates in.  A trace whose call order depended on a hash would make the
//! differential comparison against the oracle meaningless, so the order is part
//! of this module's contract rather than an implementation detail.
//!
//! # The two maps, and which invalidation may touch which
//!
//! `desired` is what the caller asked for; `applied` is what the driver was last
//! told.  Two kinds of invalidation reach them and they must not be conflated,
//! because the difference is whether the want is still *satisfiable*.
//!
//! [`BindingPoints::forget_object_where`] is the **object-scoped** one: an object
//! a slot named is gone.  It forgets both maps for every slot whose entry matched,
//! so the want goes with the belief.  That want can no longer be satisfied by any
//! call, and keeping it does not surface anything to the caller -- the caller made
//! no request -- it makes *this layer* re-emit a request of its own and then
//! report Layer 1's refusal as a failure of the caller's transition, on every
//! settle, until the caller happens to rebind that slot.  Dropping it costs
//! nothing that the belief did not already cost: the slot is left `Unknown` in
//! `applied`, so nothing believes the driver still holds the dead object, and a
//! later request for the slot emits normally.
//!
//! [`BindingPoints::forget_applied`] is the **mirror-wide** one: the identities
//! in the mirror belong to a context or an epoch the backend no longer accepts,
//! or a raw scope declared this domain.  Every want here is still satisfiable --
//! its object exists -- so every want stays, and the next settle re-establishes
//! it.  This is the same rule the session domain gives a context loss.
//!
//! The rule that covers both: a want is dropped only when the object it names is
//! gone, never because a belief was.  `textures` is the domain that states it in
//! those terms (a request naming a deleted identity cannot be satisfied); P1-17 in
//! the 0.15 plan is the finding that made this module's `forget_where` follow it,
//! having previously kept the want and argued that re-applying it was always
//! better than dropping it.

use std::collections::BTreeMap;

use crate::webgl2::api::GlError;

use super::counters::StateCounters;
use super::error::{PartialApplication, StateError};
use super::knowledge::{DriverKnowledge, ExecutionMode, StateDomain};

/// One index space's binding points: what the caller asked for, and what the
/// driver was last told.
///
/// A slot with no entry in `applied` is [`DriverKnowledge::Unknown`], which is
/// the state a slot starts in and the state an invalidation returns it to.  The
/// emit path inserts the entry it is about to establish, so "never asked about"
/// and "asked and then forgotten" need no second representation.
#[derive(Debug)]
pub(super) struct BindingPoints<T> {
    /// What the caller last asked each slot to hold.
    desired: BTreeMap<u32, T>,
    /// What the driver is known to hold, per slot.
    applied: BTreeMap<u32, DriverKnowledge<T>>,
}

impl<T> Default for BindingPoints<T> {
    fn default() -> Self {
        // Written out rather than derived: a derived `Default` would carry a
        // `T: Default` bound that no entry type has, and the empty maps are the
        // only value a binding-point set is ever defaulted to.
        Self {
            desired: BTreeMap::new(),
            applied: BTreeMap::new(),
        }
    }
}

impl<T: Copy + PartialEq> BindingPoints<T> {
    /// Records that the caller wants `slot` to hold `value`.
    ///
    /// Nothing is emitted here.  The next [`BindingPoints::settle`] decides
    /// whether a call is needed, which is what keeps the redundancy check in one
    /// place instead of once per entry point.
    pub(super) fn record(&mut self, slot: u32, value: T, counters: &mut StateCounters) {
        if self.desired.insert(slot, value).is_none() {
            // The map node a slot costs the first time it is named.  Counted
            // here rather than in the settle pass because this is where it is
            // allocated: the counter reports heap traffic, not one phase.
            counters.allocated();
        }
    }

    /// How many slots the caller has asked about.
    ///
    /// A mirror-wide invalidation never reduces this -- it clears beliefs about
    /// the driver, not the wants those beliefs were established for.  An
    /// object-scoped one does, for exactly the slots whose object died, because
    /// those wants are the unsatisfiable ones.
    pub(super) fn desired_len(&self) -> usize {
        self.desired.len()
    }

    /// Whether the driver is known to hold exactly `value` at `slot`.
    ///
    /// This is the whole redundancy rule.  An unknown slot never agrees, so the
    /// first request for a slot always emits and the mirror never has to assume
    /// a default the driver may not have.
    fn agrees(&self, slot: u32, value: &T) -> bool {
        self.applied
            .get(&slot)
            .is_some_and(|known| known.agrees(value))
    }

    /// How many driver calls settling these slots would take.
    fn pending(&self) -> u32 {
        self.desired
            .iter()
            .filter(|(slot, value)| !self.agrees(**slot, value))
            .count() as u32
    }

    /// Makes the driver's slots agree with the caller's, through `emit`.
    ///
    /// `domain` is the domain the request, emit, skip and recovery tallies are
    /// accounted against, and `operation` is the name a refused call carries
    /// out.  Both are the caller's to state: this module knows how to settle a
    /// binding role but not which verb it is settling for.
    ///
    /// One request is counted for the whole settle, and then one emit or one
    /// skip per slot decision.  The two tallies are not required to be equal,
    /// because one request can settle several slots; what they do say is what
    /// this layer exists to report, which is how many driver calls a transition
    /// cost and how many of them it proved redundant.
    pub(super) fn settle<E>(
        &mut self,
        domain: StateDomain,
        operation: &'static str,
        mode: ExecutionMode,
        counters: &mut StateCounters,
        mut emit: E,
    ) -> Result<(), StateError>
    where
        E: FnMut(u32, &T) -> Result<(), GlError>,
    {
        counters.domain(domain).request();

        if self.desired.is_empty() {
            // Nothing has been asked for, so no slot can be settled and no call
            // can be proved redundant -- but the caller did ask this domain to
            // settle, and it settled without a driver call.  That is what the
            // skip tally counts, and it is the accounting the session domain
            // uses for a boundary that never changed.
            counters.domain(domain).skip();
            return Ok(());
        }

        // Taken before anything is emitted, because a failure reports how much
        // of the group had been intended rather than how much of it is left.
        let expected = self.pending();
        let mut emitted = 0_u32;
        for (slot, value) in &self.desired {
            let known = self.applied.get(slot);
            let redundant = known.is_some_and(|known| known.agrees(value));
            if mode.may_skip() && redundant {
                counters.domain(domain).skip();
                continue;
            }
            // The slot was never established, or a deletion or an invalidation
            // took the belief away: this call recovers state rather than changing
            // a value the mirror was sure of.
            let recovering = !known.is_some_and(DriverKnowledge::is_known);
            if let Err(source) = emit(*slot, value) {
                // The domain contract's failure rule: this domain's applied state
                // is left *unknown* rather than unchanged.  A provider may refuse
                // a call after the driver has already seen part of its effect,
                // and a mirror that kept believing the previous value here is
                // exactly the stale belief a later skip would be built on.
                self.applied.clear();
                counters.lifecycle.driver_errors += 1;
                return Err(StateError::backend(
                    domain,
                    operation,
                    PartialApplication::new(emitted, expected),
                    source,
                ));
            }
            if self
                .applied
                .insert(*slot, DriverKnowledge::Known(*value))
                .is_none()
            {
                counters.allocated();
            }
            if recovering {
                // Counted only after the call succeeded: a refused call recovered
                // nothing, and reporting it as a recovery would make the tally
                // claim progress the driver never made.
                counters.domain(domain).recover();
            }
            counters.domain(domain).emit();
            emitted += 1;
        }
        Ok(())
    }
}

impl<T> BindingPoints<T> {
    /// Forgets every slot whose entry satisfies `matches`, in *both* maps.
    ///
    /// This is the object-scoped invalidation: the entry named an object that is
    /// about to be deleted or has been retired, so the want is dropped with the
    /// belief and the slot returns to the state it had before it was ever named.
    /// The module doc gives the reason a want goes here and stays under
    /// [`BindingPoints::forget_applied`], and the distinction is the whole reason
    /// the two hooks exist rather than one.
    ///
    /// The predicate is supplied by the caller because only the caller knows
    /// which object an entry names -- a storage range names a buffer by field, a
    /// storage-image binding names a texture -- and a mirror that could not
    /// answer that question would be a binding point it could not clear on a
    /// deletion at all.
    ///
    /// Removing a want frees a `desired` node but is not counted: the allocation
    /// tally reports heap traffic the layer caused, and a freed node was already
    /// counted when it was inserted.  A slot named again afterwards counts a new
    /// allocation, which is honest -- it is a new node.
    pub(super) fn forget_object_where(&mut self, matches: impl Fn(&T) -> bool) {
        self.applied
            .retain(|_, known| !known.get().is_some_and(&matches));
        self.desired.retain(|_, value| !matches(value));
    }

    /// Forgets what the driver holds everywhere, keeping every want.
    ///
    /// The mirror-wide invalidation: a context loss, a device replacement, or a
    /// raw scope that declared this domain.  Every want survives because every
    /// object it names still exists, so the next settle re-establishes what the
    /// caller asked for instead of assuming the request went away with the
    /// beliefs.
    pub(super) fn forget_applied(&mut self) {
        // Clearing rather than marking every entry unknown: the two say the same
        // thing about the driver, and clearing also gives the map back, which a
        // lost context makes worthless anyway.
        self.applied.clear();
    }
}

#[cfg(test)]
mod tests;
