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
//! # The two maps, and what an invalidation may touch
//!
//! `desired` is what the caller asked for; `applied` is what the driver was last
//! told.  An invalidation may only ever clear the second, and the asymmetry is
//! the point.  Backing a want with a belief the mirror no longer has is wrong:
//! the next settle re-emits it and Layer 1 refuses a deleted or stale-epoch
//! object with a structured error, so the caller learns.  Dropping the want
//! instead would leave the binding point holding whatever the driver last had,
//! with no error anywhere and a wrong image as the only symptom -- the one
//! outcome strictly worse than a failure.
//!
//! ⚠️ **That paragraph is the subject of an open finding, and the reader should
//! not act on it as settled.** The plan's P1-17 disputes the last sentence: this
//! module's own `forget_where` clears the applied entry too, so the slot is left
//! honestly `Unknown` and the "wrong image as the only symptom" cannot occur --
//! what keeping the want produces instead is a *doomed re-emit*, and with it a
//! failure the caller never asked for, on every reconcile until it rebinds the
//! slot. `textures` already drops the want for the object that died and states
//! that reason. The two policies are expected to converge on `textures`', which
//! moves this module's `forget_where` to filter `desired` by the same predicate;
//! the split that remains is deletion (unsatisfiable, so the want goes) versus a
//! whole-mirror event (still satisfiable, so it stays). Recorded here rather than
//! only in the plan because this is the paragraph a domain author copies.

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
    /// An invalidation never reduces this: it clears beliefs about the driver,
    /// not the wants those beliefs were established for.
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
    /// Forgets what the driver holds for every slot whose entry satisfies
    /// `matches`.
    ///
    /// The predicate is supplied by the caller because only the caller knows
    /// which object an entry names -- a storage range names a buffer by field, a
    /// storage-image binding names a texture -- and a mirror that could not
    /// answer that question would be a binding point it could not clear on a
    /// deletion, which is the defect this hook exists to prevent.
    ///
    /// The desired entries stay: see the module doc for why re-applying a want is
    /// always better than silently dropping it -- and for the open finding that
    /// disputes it, which is why this is the hook a fix would widen rather than a
    /// second hook a caller would have to remember.
    pub(super) fn forget_where(&mut self, matches: impl Fn(&T) -> bool) {
        self.applied
            .retain(|_, known| !known.get().is_some_and(&matches));
    }

    /// Forgets what the driver holds everywhere, keeping the caller's wants.
    pub(super) fn forget_applied(&mut self) {
        // Clearing rather than marking every entry unknown: the two say the same
        // thing about the driver, and clearing also gives the map back, which a
        // lost context makes worthless anyway.
        self.applied.clear();
    }
}

#[cfg(test)]
mod tests;
