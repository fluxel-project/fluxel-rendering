//! The counters Layer 2 must be able to report.
//!
//! These are the plan's required counters reduced to the ones a state layer can
//! actually observe: attempted logical requests, emitted driver calls, skipped
//! calls, unknown-state recoveries, cache traffic, submissions, barriers, and
//! the two steady-state costs the performance funnel attributes first —
//! allocations and bytes copied while applying a group.
//!
//! # Why the per-domain tally is an array
//!
//! Counters are read per frame, per pass and cumulatively.  Rather than five
//! named fields per domain, every domain gets one [`DomainCounters`] entry in a
//! fixed-size array indexed by [`StateDomain::index`], so a report enumerates
//! [`StateDomain::ALL`] and a new domain cannot be added without appearing in
//! every report.  Nothing here allocates.
//!
//! # What is deliberately not counted
//!
//! Wall-clock time is not recorded here.  The plan asks for CPU time per phase,
//! and that belongs to whichever harness does the measuring, on hardware this
//! layer cannot see; a per-call `Instant::now` in a state machine would be a
//! cost the counters themselves introduce.  What this layer can honestly report
//! is call counts, which is what the causal attribution in the performance
//! funnel is required to use first.

use super::knowledge::StateDomain;

/// Call traffic for one state domain.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DomainCounters {
    /// Logical requests this domain was asked to serve.
    pub requests: u64,
    /// Driver calls this domain emitted.
    pub emitted: u64,
    /// Calls this domain proved redundant and did not emit.
    pub skipped: u64,
    /// Times a value the domain needed was `Unknown` and had to be applied.
    pub unknown_recoveries: u64,
}

impl DomainCounters {
    /// Records one logical request.
    pub(crate) fn request(&mut self) {
        self.requests += 1;
    }

    /// Records one emitted driver call.
    pub(crate) fn emit(&mut self) {
        self.emitted += 1;
    }

    /// Records one call the mirror proved redundant.
    pub(crate) fn skip(&mut self) {
        self.skipped += 1;
    }

    /// Records one value recovered from `Unknown`.
    pub(crate) fn recover(&mut self) {
        self.unknown_recoveries += 1;
    }
}

/// Derived-cache traffic, kept apart from the state domains.
///
/// A cache lookup is not a state transition: it neither emits nor skips a
/// driver call by itself, and folding the two together would make a cache that
/// hits but still emits look like a redundant-call win.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CacheCounters {
    /// Lookups answered from an existing entry.
    pub hits: u64,
    /// Lookups that found nothing and had to create.
    pub misses: u64,
    /// Entries created.
    pub created: u64,
    /// Entries evicted to respect a budget.
    pub evicted: u64,
    /// Entries dropped because a dependency of theirs was deleted or an epoch
    /// changed.
    pub invalidated: u64,
    /// Entries whose retained bytes are currently accounted for.
    pub live_entries: u64,
    /// Estimated retained bytes across all live entries.
    pub live_bytes: u64,
    /// The high-water mark of [`CacheCounters::live_bytes`].
    pub peak_bytes: u64,
    /// Entries dropped without touching a backend, because the context was lost
    /// and its backend values are no longer callable.
    pub purged_on_loss: u64,
}

/// Command submission tallies, which belong to no single state domain.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SubmissionCounters {
    /// Render passes begun.
    pub passes: u64,
    /// Load/clear operations executed, which must be one per pass that asks.
    pub pass_loads: u64,
    /// Store/resolve/invalidate operations executed at pass end.
    pub pass_stores: u64,
    /// Ordinary and instanced draws issued.
    pub draws: u64,
    /// Multi-draw commands issued.
    pub multi_draws: u64,
    /// Indirect draw commands issued.
    pub indirect_draws: u64,
    /// Direct dispatches issued.
    pub dispatches: u64,
    /// Indirect dispatches issued.
    pub indirect_dispatches: u64,
    /// Clear commands issued.
    pub clears: u64,
    /// Copy commands issued.
    pub copies: u64,
    /// Presentation operations performed.
    pub presents: u64,
    /// Memory barriers emitted, by the category set they covered.
    pub barriers: u64,
}

/// Lifecycle and failure tallies.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct LifecycleCounters {
    /// Requests rejected because an object belonged to another epoch.
    pub stale_epoch_rejections: u64,
    /// Whole-domain invalidation passes performed.
    pub domain_invalidations: u64,
    /// Groups whose application failed partway and were marked unknown.
    pub poisoned_groups: u64,
    /// Times the context was observed lost.
    pub context_losses: u64,
    /// Times the context was restored onto a newer epoch.
    pub context_restorations: u64,
    /// Times the backend reported a driver or browser error.
    pub driver_errors: u64,
}

/// The complete counter set Layer 2 maintains.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct StateCounters {
    domains: [DomainCounters; StateDomain::COUNT],
    /// Derived-cache traffic.
    pub caches: CacheCounters,
    /// Command submission tallies.
    pub submissions: SubmissionCounters,
    /// Lifecycle and failure tallies.
    pub lifecycle: LifecycleCounters,
    /// Steady-state heap allocations made while applying state groups.
    ///
    /// A domain that clones a binding list, builds a `Vec` of calls, or copies
    /// binding contents must record it here.  This is the counter the third
    /// optimization candidate is judged by, so an implementation that hides an
    /// allocation behind a `SmallVec` spill still has to account for it.
    pub steady_state_allocations: u64,
    /// Bytes copied while applying state groups.
    pub binding_bytes_copied: u64,
}

impl StateCounters {
    /// The tallies for one domain.
    pub(crate) fn domain(&mut self, domain: StateDomain) -> &mut DomainCounters {
        &mut self.domains[domain.index()]
    }

    /// The recorded tallies for one domain, for a report.
    pub(crate) const fn domain_counts(&self, domain: StateDomain) -> &DomainCounters {
        &self.domains[domain.index()]
    }

    /// Every domain's tallies with its name, for a report.
    pub(crate) fn report(&self) -> impl Iterator<Item = (StateDomain, &DomainCounters)> {
        StateDomain::ALL
            .into_iter()
            .map(|domain| (domain, self.domain_counts(domain)))
    }

    /// Records one steady-state allocation.
    pub(crate) fn allocated(&mut self) {
        self.steady_state_allocations += 1;
    }

    /// Records `bytes` copied while applying state.
    pub(crate) fn copied(&mut self, bytes: u64) {
        self.binding_bytes_copied += bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::StateCounters;
    use crate::webgl2::state::knowledge::StateDomain;

    #[test]
    fn per_domain_tallies_are_independent() {
        let mut counters = StateCounters::default();
        counters.domain(StateDomain::Textures).emit();
        counters.domain(StateDomain::Textures).skip();
        counters.domain(StateDomain::Pipeline).request();

        let textures = counters.domain_counts(StateDomain::Textures);
        assert_eq!(textures.emitted, 1);
        assert_eq!(textures.skipped, 1);
        assert_eq!(textures.requests, 0);
        assert_eq!(counters.domain_counts(StateDomain::Pipeline).requests, 1);
        assert_eq!(counters.domain_counts(StateDomain::Sync).emitted, 0);
    }

    #[test]
    fn a_report_names_every_domain_in_a_stable_order() {
        let mut counters = StateCounters::default();
        counters.domain(StateDomain::Groups).recover();
        let named: Vec<_> = counters
            .report()
            .map(|(domain, counts)| (domain.name(), counts.unknown_recoveries))
            .collect();
        assert_eq!(named.len(), StateDomain::COUNT);
        assert_eq!(named[0].0, "session");
        assert_eq!(named[StateDomain::Groups.index()].1, 1);
    }
}
