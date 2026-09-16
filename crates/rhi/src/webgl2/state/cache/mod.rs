//! Derived-state caches: budgets, deterministic eviction, and reverse
//! dependency invalidation.
//!
//! A derived cache holds a backend object Layer 2 created so a later frame can
//! reuse it -- a vertex-array object, a framebuffer, a linked program -- keyed
//! by the complete structural value it was derived from.  Three properties make
//! one safe to keep:
//!
//! 1. **Structural keys.**  A key carries every input that changes the value,
//!    including the context stamp.  The keys are Layer 1's own descriptor types,
//!    which is why this table is bounded on `Eq + Hash + Clone` and not on
//!    `Ord`: those types deliberately derive the former and not the latter, and
//!    the alternative -- a hand-written total order over every descriptor -- is a
//!    second definition of a key that can disagree with the first.  A hash may
//!    accelerate a lookup; it may never replace equality, because the reference
//!    implementation this series was written against used hash-only geometry
//!    identity and the plan records that as a weakness not to reproduce.
//! 2. **Reverse dependencies.**  Every entry records the resources it was
//!    derived from, so deleting a buffer can invalidate the geometry records
//!    built from it *before* the backend deletes the name and a new object
//!    reuses it.
//! 3. **Deterministic eviction.**  Eviction is a function of the access trace
//!    alone -- never wall-clock age, and never weak-reference death, neither of
//!    which proves the GPU is done with an object.  Order is deterministic
//!    regardless of the table's iteration order: access ticks are unique, and
//!    removals that happen in one call report in creation order.
//!
//! # What eviction does not know
//!
//! An entry whose object is still named by work the driver has not finished is
//! *not* protected here, and this module deliberately does not pretend
//! otherwise.  Deciding that would need a completion signal that lives above
//! this layer, and the two mechanisms that actually make an early drop safe are
//! already in force below it: Layer 1 defers the deletion of an object that
//! still has a name until it is unbound, and every entry carries the generation
//! it was derived under, so a record whose object was deleted is dropped by
//! invalidation rather than reused.  So an eviction can make a frame slower; it
//! cannot make it wrong.
//!
//! # Who destroys what
//!
//! A cached value is normally a backend object this layer created, so a cache
//! must never drop one silently.  Every call that can remove an entry hands the
//! removed pairs back for the caller to destroy, and the two teardown paths are
//! kept apart because they are not the same act: [`StructuralCache::drain`]
//! returns everything so a machine shutting down destroys what it made, while
//! [`StructuralCache::purge`] destroys nothing, because the identities in those
//! entries belong to a context epoch the backend no longer accepts.
//!
//! # Where the typed caches live
//!
//! This module holds the machinery and nothing else: budgets, the entry table,
//! eviction, and reverse-dependency invalidation.  Each domain's cache
//! lives inside that domain's own module, because a cache's key, value and
//! per-entry cost are that domain's knowledge: a key that omitted one of the
//! domain's inputs is a defect only the domain can notice, and a cost estimate
//! that is wrong in the direction of "too small" is what turns a bounded cache
//! into unbounded memory.  A domain owns its directory end to end, which is what
//! lets its cache, its tests and its mirror be reviewed without reading the
//! layer's other domains.

use std::collections::HashMap;

use crate::webgl2::api::{
    BufferId, FramebufferId, ProgramId, QueryId, RenderbufferId, SamplerId, ShaderId,
    SurfaceImageId, SyncId, TextureId, VertexArrayId,
};

use super::counters::CacheCounters;

/// Whether derived state may be retained between operations.
///
/// The oracle mode runs the same domains with this set to `Disabled`, so a
/// differential test compares an optimized trace against the trace of a machine
/// that never retained anything.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CacheMode {
    /// Retain and reuse derived backend state.
    Enabled,
    /// Create everything fresh and retain nothing.
    Disabled,
}

impl CacheMode {
    /// Whether a lookup may be answered from an existing entry.
    pub(crate) const fn may_reuse(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

/// The size a cache is allowed to reach before it evicts.
///
/// Both bounds are required.  An entry bound alone lets a cache of large
/// framebuffer records exceed any byte budget it was given, and a byte bound
/// alone lets a cache of tiny program records spend the byte budget on
/// bookkeeping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CacheBudget {
    /// Maximum live entries.
    pub max_entries: usize,
    /// Maximum estimated retained bytes.
    pub max_bytes: u64,
}

impl CacheBudget {
    /// A budget with both bounds.
    pub(crate) const fn new(max_entries: usize, max_bytes: u64) -> Self {
        Self {
            max_entries,
            max_bytes,
        }
    }
}

/// A resource a derived entry was built from.
///
/// The variants are the identity types Layer 1 hands out.  A raw name is
/// deliberately absent: a derived entry must be invalidated by Fluxel identity,
/// so that a driver reusing an integer name cannot keep a stale entry alive.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum ResourceRef {
    /// A buffer object.
    Buffer(BufferId),
    /// A texture object.
    Texture(TextureId),
    /// A sampler object.
    Sampler(SamplerId),
    /// A shader object.
    Shader(ShaderId),
    /// A linked program.
    Program(ProgramId),
    /// A vertex-array object.
    VertexArray(VertexArrayId),
    /// A framebuffer object.
    Framebuffer(FramebufferId),
    /// A renderbuffer object.
    Renderbuffer(RenderbufferId),
    /// A query object.
    Query(QueryId),
    /// A sync object.
    Sync(SyncId),
    /// A presentation surface image.
    SurfaceImage(SurfaceImageId),
}

/// The resources one entry depends on, kept ordered for deterministic reports.
pub(crate) type DependencySet = std::collections::BTreeSet<ResourceRef>;

/// One retained derived object.
#[derive(Clone, Debug, Eq, PartialEq)]
struct CacheEntry<V> {
    value: V,
    /// Estimated retained bytes, chosen by the cache that owns the entry.
    bytes: u64,
    /// The cache's own monotonic access tick, never a wall-clock instant.
    ///
    /// Unique per entry: every lookup and every insert takes a fresh tick, so
    /// the least-recently-used search below always has a single answer and never
    /// needs a tie-break that would depend on iteration order.
    last_used: u64,
    /// The tick this entry was created at, so removals can be reported in
    /// creation order without the key type being ordered.
    sequence: u64,
    /// Resources this value was derived from.
    dependencies: DependencySet,
}

/// What one cache call removed, and whether it kept what it was given.
///
/// `removed` is in creation order, so a caller that destroys each value makes
/// the same sequence of backend calls on every run of the same trace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CacheMutation<K, V> {
    /// Entries this call took out of the table, for the caller to destroy.
    pub removed: Vec<(K, V)>,
    /// Whether the value the call was given is now retained by the cache.
    ///
    /// `false` means the value does not fit the budget even with the cache
    /// emptied -- a record larger than `max_bytes`, or a budget of zero entries.
    /// Nothing was inserted, the caller still owns the value it passed, and it
    /// must destroy that value when it is done with it.
    pub retained: bool,
}

impl<K, V> CacheMutation<K, V> {
    /// A call that removed nothing.
    fn untouched(retained: bool) -> Self {
        Self {
            removed: Vec::new(),
            retained,
        }
    }
}

/// A structurally keyed cache with deterministic eviction.
#[derive(Debug)]
pub(crate) struct StructuralCache<K, V> {
    budget: CacheBudget,
    entries: HashMap<K, CacheEntry<V>>,
    live_bytes: u64,
    peak_bytes: u64,
    tick: u64,
}

impl<K, V> StructuralCache<K, V>
where
    K: Clone + Eq + std::hash::Hash,
{
    /// An empty cache with this budget.
    pub(crate) fn new(budget: CacheBudget) -> Self {
        Self {
            budget,
            entries: HashMap::new(),
            live_bytes: 0,
            peak_bytes: 0,
            tick: 0,
        }
    }

    /// The number of live entries.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// The estimated retained bytes.
    pub(crate) const fn live_bytes(&self) -> u64 {
        self.live_bytes
    }

    /// The high-water mark of [`StructuralCache::live_bytes`].
    pub(crate) const fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    /// Advances the access clock and returns the new tick.
    fn advance(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Looks an entry up, recording the access.
    ///
    /// A miss is counted; the caller is responsible for counting a create when
    /// it inserts the value it had to build.
    pub(crate) fn get(&mut self, key: &K, counters: &mut CacheCounters) -> Option<&V> {
        let tick = self.advance();
        match self.entries.get_mut(key) {
            Some(entry) => {
                entry.last_used = tick;
                counters.hits += 1;
                Some(&entry.value)
            }
            None => {
                counters.misses += 1;
                None
            }
        }
    }

    /// Looks an entry up without recording an access.
    ///
    /// For a question a caller asks about state it is not about to use -- a
    /// report, or a check before a deletion -- so that observing the cache does
    /// not change what eviction would do next.
    pub(crate) fn peek(&self, key: &K) -> Option<&V> {
        self.entries.get(key).map(|entry| &entry.value)
    }

    /// Inserts an entry, evicting entries until it fits.
    ///
    /// The evicted values are returned in [`CacheMutation::removed`] for the
    /// caller to destroy.  A key already present is replaced, and its previous
    /// value is reported the same way.
    ///
    /// The victims are chosen before anything is removed, so a refused insert
    /// leaves the cache exactly as it was: no entry is destroyed for an insert
    /// that did not happen.
    pub(crate) fn insert(
        &mut self,
        key: K,
        value: V,
        bytes: u64,
        dependencies: DependencySet,
        counters: &mut CacheCounters,
    ) -> CacheMutation<K, V> {
        // Take the entry this key replaces out of the table whole, so a refused
        // insert can put it back with its own bytes, sequence and dependencies
        // rather than with the new ones.
        let previous = self.entries.remove(&key);
        let victims = match self.plan_eviction(bytes, previous.as_ref()) {
            Some(victims) => victims,
            None => {
                if let Some(entry) = previous {
                    self.entries.insert(key, entry);
                }
                return CacheMutation::untouched(false);
            }
        };

        let mut removed = Vec::with_capacity(victims.len() + 1);
        if let Some(entry) = previous {
            self.live_bytes -= entry.bytes;
            removed.push((key.clone(), entry.value));
        }
        for victim in victims {
            if let Some(entry) = self.entries.remove(&victim) {
                self.live_bytes -= entry.bytes;
                removed.push((victim, entry.value));
                counters.evicted += 1;
            }
        }

        let tick = self.advance();
        self.entries.insert(
            key,
            CacheEntry {
                value,
                bytes,
                last_used: tick,
                sequence: tick,
                dependencies,
            },
        );
        self.live_bytes += bytes;
        self.peak_bytes = self.peak_bytes.max(self.live_bytes);
        counters.created += 1;
        counters.live_entries = self.entries.len() as u64;
        counters.live_bytes = self.live_bytes;
        counters.peak_bytes = self.peak_bytes;
        CacheMutation {
            removed,
            retained: true,
        }
    }

    /// Chooses which entries an insert of `bytes` would have to evict.
    ///
    /// `replaced` is the entry this key is replacing, already out of the table.
    /// Returns `None` when the value cannot fit the budget even with the cache
    /// emptied.  Nothing is removed here: a refusal must cost nothing.
    fn plan_eviction(&self, bytes: u64, replaced: Option<&CacheEntry<V>>) -> Option<Vec<K>> {
        // The key's previous entry is already out of the table, so the entry
        // count needs no adjustment -- but `live_bytes` has not been reduced for
        // it yet, because a refused insert must be able to put it back with its
        // own accounting intact.
        let mut entries = self.entries.len();
        let mut total = self.live_bytes;
        if let Some(entry) = replaced {
            total -= entry.bytes;
        }

        let fits = |entries: usize, total: u64| {
            entries < self.budget.max_entries && total + bytes <= self.budget.max_bytes
        };
        if fits(entries, total) {
            return Some(Vec::new());
        }

        let mut candidates: Vec<(u64, K)> = self
            .entries
            .iter()
            .map(|(key, entry)| (entry.last_used, key.clone()))
            .collect();
        // Access ticks are unique, so this order is total and needs no
        // tie-break that would depend on the table's iteration order.
        candidates.sort_by_key(|(last_used, _)| *last_used);

        let mut victims = Vec::new();
        for (_, key) in candidates {
            if fits(entries, total) {
                break;
            }
            total -= self.entries.get(&key).map_or(0, |entry| entry.bytes);
            entries -= 1;
            victims.push(key);
        }
        fits(entries, total).then_some(victims)
    }

    /// Drops every entry that names `resource`, reporting what was dropped.
    pub(crate) fn invalidate_resource(
        &mut self,
        resource: ResourceRef,
        counters: &mut CacheCounters,
    ) -> CacheMutation<K, V> {
        let mut doomed: Vec<(u64, K)> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.dependencies.contains(&resource))
            .map(|(key, entry)| (entry.sequence, key.clone()))
            .collect();
        // Creation order, so a caller destroying each value makes a
        // reproducible sequence of backend calls.
        doomed.sort_by_key(|(sequence, _)| *sequence);

        let mut removed = Vec::with_capacity(doomed.len());
        for (_, key) in doomed {
            if let Some(entry) = self.entries.remove(&key) {
                self.live_bytes -= entry.bytes;
                removed.push((key, entry.value));
                counters.invalidated += 1;
            }
        }
        counters.live_entries = self.entries.len() as u64;
        counters.live_bytes = self.live_bytes;
        CacheMutation {
            removed,
            retained: true,
        }
    }

    /// Empties the cache for a normal teardown, reporting everything held.
    ///
    /// The caller destroys what comes back: these values are backend objects
    /// this layer created, and dropping the table without destroying them would
    /// leak them in the driver for as long as the context lives.
    pub(crate) fn drain(&mut self, counters: &mut CacheCounters) -> Vec<(K, V)> {
        let mut doomed: Vec<(u64, K)> = self
            .entries
            .iter()
            .map(|(key, entry)| (entry.sequence, key.clone()))
            .collect();
        doomed.sort_by_key(|(sequence, _)| *sequence);

        let mut removed = Vec::with_capacity(doomed.len());
        for (_, key) in doomed {
            if let Some(entry) = self.entries.remove(&key) {
                removed.push((key, entry.value));
            }
        }
        counters.live_entries = 0;
        counters.live_bytes = 0;
        self.live_bytes = 0;
        removed
    }

    /// Empties the cache after a context loss, destroying nothing.
    ///
    /// The backend values in these entries are no longer callable, so nothing
    /// may be destroyed through them, and the only correct action is to forget
    /// them.  The purge is counted separately from an eviction because it is
    /// not a policy decision.
    pub(crate) fn purge(&mut self, counters: &mut CacheCounters) {
        counters.purged += self.entries.len() as u64;
        counters.live_entries = 0;
        counters.live_bytes = 0;
        self.entries.clear();
        self.live_bytes = 0;
    }

    /// The dependencies of one entry, for a report or a test.
    pub(crate) fn dependencies(&self, key: &K) -> Option<&DependencySet> {
        self.entries.get(key).map(|entry| &entry.dependencies)
    }
}

/// The budget the derived caches are constructed with.
///
/// The numbers are an initial choice, not a measurement: the performance funnel
/// in the plan is what decides whether they are right, and the counters report
/// live entries, live bytes, peaks and eviction counts so that decision has
/// evidence instead of a guess.  A budget that is never reached costs nothing
/// but the entries it allows, and a budget that is reached shows up as eviction
/// traffic on a workload whose keys repeat.
pub(crate) const DEFAULT_BUDGET: CacheBudget = CacheBudget::new(512, 8 * 1024 * 1024);

#[cfg(test)]
mod tests {
    use super::{CacheBudget, CacheMode, DependencySet, StructuralCache};
    use crate::webgl2::state::counters::CacheCounters;

    // `invalidate_resource` and `drain` are exercised here only through their
    // bookkeeping, because every `ResourceRef` variant holds an
    // `ObjectIdentity` whose constructor is `pub(super)` inside `api` -- this
    // module deliberately cannot mint one.  Their dependency-filtering and
    // removal-order behaviour is covered by the fixture-backed cache tests,
    // which obtain real identities from a provider.

    fn cache() -> StructuralCache<u32, u32> {
        StructuralCache::new(CacheBudget::new(2, 32))
    }

    #[test]
    fn eviction_is_deterministic_and_prefers_the_least_recently_used() {
        let mut cache = cache();
        let mut counters = CacheCounters::default();
        assert!(
            cache
                .insert(1, 10, 8, DependencySet::new(), &mut counters)
                .retained
        );
        assert!(
            cache
                .insert(2, 20, 8, DependencySet::new(), &mut counters)
                .retained
        );
        // Touch 1 so 2 is the least recently used.
        assert_eq!(cache.get(&1, &mut counters), Some(&10));
        let mutation = cache.insert(3, 30, 8, DependencySet::new(), &mut counters);

        assert_eq!(mutation.removed, vec![(2, 20)]);
        assert!(cache.peek(&1).is_some());
        assert!(cache.peek(&2).is_none());
        assert!(cache.peek(&3).is_some());
        assert_eq!(counters.evicted, 1);
        assert_eq!(counters.hits, 1);
    }

    #[test]
    fn a_value_that_cannot_fit_the_budget_is_refused_and_costs_the_cache_nothing() {
        // 32 bytes of room, and a value of 64: no amount of eviction makes room,
        // so the insert is refused rather than accepted and immediately evicted.
        let mut cache = StructuralCache::new(CacheBudget::new(2, 32));
        let mut counters = CacheCounters::default();
        assert!(
            cache
                .insert(1, 10, 8, DependencySet::new(), &mut counters)
                .retained
        );

        let mutation = cache.insert(2, 20, 64, DependencySet::new(), &mut counters);
        assert!(!mutation.retained);
        // Nothing was evicted and nothing is handed back: the caller still owns
        // the value it passed, and the resident entry kept its own accounting.
        assert!(mutation.removed.is_empty());
        assert!(cache.peek(&1).is_some());
        assert!(cache.peek(&2).is_none());
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.live_bytes(), 8);
        assert_eq!(counters.evicted, 0);
    }

    #[test]
    fn a_budget_of_no_entries_refuses_every_insert() {
        // The degenerate budget, kept honest: an entry bound of zero is a cache
        // that retains nothing, not a cache that silently holds one entry.
        let mut cache = StructuralCache::new(CacheBudget::new(0, 32));
        let mut counters = CacheCounters::default();
        let mutation = cache.insert(1, 10, 8, DependencySet::new(), &mut counters);

        assert!(!mutation.retained);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.live_bytes(), 0);
    }

    #[test]
    fn a_byte_budget_bounds_a_cache_that_is_under_its_entry_budget() {
        let mut cache = StructuralCache::new(CacheBudget::new(64, 24));
        let mut counters = CacheCounters::default();
        assert!(
            cache
                .insert(1, 1, 16, DependencySet::new(), &mut counters)
                .retained
        );
        // The second entry would take the cache to 32 bytes against a 24-byte
        // budget, so the first has to go even though the entry budget is 64.
        assert!(
            cache
                .insert(2, 2, 16, DependencySet::new(), &mut counters)
                .retained
        );
        assert!(cache.peek(&1).is_none());
        assert_eq!(cache.live_bytes(), 16);
        assert_eq!(counters.peak_bytes, 16);
    }

    #[test]
    fn a_purge_is_not_an_eviction_and_destroys_nothing() {
        let mut cache = cache();
        let mut counters = CacheCounters::default();
        assert!(
            cache
                .insert(1, 10, 8, DependencySet::new(), &mut counters)
                .retained
        );
        // `purge` returns nothing at all, which is the point: after a context
        // loss there is no callable value left to hand back.
        cache.purge(&mut counters);

        assert_eq!(cache.len(), 0);
        assert_eq!(counters.purged, 1);
        assert_eq!(counters.evicted, 0);
        assert_eq!(counters.live_bytes, 0);
    }

    #[test]
    fn a_normal_drain_hands_back_everything_for_the_caller_to_destroy() {
        let mut cache = cache();
        let mut counters = CacheCounters::default();
        assert!(
            cache
                .insert(1, 10, 8, DependencySet::new(), &mut counters)
                .retained
        );
        assert!(
            cache
                .insert(2, 20, 8, DependencySet::new(), &mut counters)
                .retained
        );

        let drained = cache.drain(&mut counters);
        // Creation order, not table order.
        assert_eq!(drained, vec![(1, 10), (2, 20)]);
        assert_eq!(cache.len(), 0);
        // A drain is neither an eviction nor a loss purge.
        assert_eq!(counters.evicted, 0);
        assert_eq!(counters.purged, 0);
    }

    #[test]
    fn replacing_a_key_reports_the_previous_value_and_does_not_double_count_bytes() {
        let mut cache = StructuralCache::new(CacheBudget::new(8, 100));
        let mut counters = CacheCounters::default();
        assert!(
            cache
                .insert(1, 10, 40, DependencySet::new(), &mut counters)
                .retained
        );
        let mutation = cache.insert(1, 11, 24, DependencySet::new(), &mut counters);

        assert_eq!(mutation.removed, vec![(1, 10)]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.live_bytes(), 24);
        assert_eq!(cache.peek(&1), Some(&11));
        assert_eq!(counters.evicted, 0);
    }

    #[test]
    fn cache_mode_states_the_reuse_rule_once() {
        assert!(CacheMode::Enabled.may_reuse());
        assert!(!CacheMode::Disabled.may_reuse());
    }
}
