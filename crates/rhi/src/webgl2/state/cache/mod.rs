//! Derived-state caches: budgets, deterministic eviction, and reverse
//! dependency invalidation.
//!
//! A derived cache holds a backend object Layer 2 created so a later frame can
//! reuse it -- a vertex-array object, a framebuffer, a linked program -- keyed
//! by the complete structural value it was derived from.  Three properties make
//! one safe to keep:
//!
//! 1. **Structural keys.**  A key carries every input that changes the value,
//!    including the context stamp.  A hash may accelerate a lookup; it may
//!    never replace equality, because the reference implementation this series
//!    was written against used hash-only geometry identity and the plan records
//!    that as a weakness not to reproduce.
//! 2. **Reverse dependencies.**  Every entry records the resources it was
//!    derived from, so deleting a buffer can invalidate the geometry records
//!    built from it *before* the backend deletes the name and a new object
//!    reuses it.
//! 3. **Deterministic eviction under a lease.**  Eviction is a function of the
//!    access trace alone -- never wall-clock age, and never weak-reference
//!    death, neither of which proves the GPU is done with an object.
//!
//! # Leases
//!
//! An entry referenced by accepted work is leased.  A leased entry is never
//! evicted and never destroyed; a cache that cannot make room without breaking a
//! lease refuses the insert and says so, and the caller then uses the object it
//! just created without retaining it.  That is a slower frame, not an incorrect
//! one.
//!
//! # Where the typed caches live
//!
//! This module holds the machinery and nothing else: budgets, the ordered
//! entry table, eviction, leasing, and reverse-dependency invalidation.  Each
//! domain's cache -- its key type, its value type, and what one entry costs --
//! is declared beside the domain that owns those types, because a key that
//! omitted one of its domain's inputs is a defect only that domain can notice.

use std::collections::BTreeMap;

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
    last_used: u64,
    /// Leases held by accepted work.  A leased entry is pinned.
    leases: u32,
    /// Resources this value was derived from.
    dependencies: DependencySet,
}

/// A structurally keyed cache with deterministic eviction.
///
/// The key is ordered rather than hashed so that eviction ties break by key and
/// a report of live entries is stable across runs of the same trace.
#[derive(Debug)]
pub(crate) struct StructuralCache<K, V> {
    budget: CacheBudget,
    entries: BTreeMap<K, CacheEntry<V>>,
    live_bytes: u64,
    peak_bytes: u64,
    tick: u64,
}

impl<K, V> StructuralCache<K, V>
where
    K: Ord + Clone,
{
    /// An empty cache with this budget.
    pub(crate) fn new(budget: CacheBudget) -> Self {
        Self {
            budget,
            entries: BTreeMap::new(),
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

    /// Inserts an entry, evicting unleased entries until it fits.
    ///
    /// Returns `false` when the budget could not be met without evicting a
    /// leased entry.  Nothing was inserted in that case and the caller must use
    /// its value uncached; see the module note on leases.
    pub(crate) fn insert(
        &mut self,
        key: K,
        value: V,
        bytes: u64,
        dependencies: DependencySet,
        counters: &mut CacheCounters,
    ) -> bool {
        if let Some(previous) = self.entries.remove(&key) {
            self.live_bytes -= previous.bytes;
        }
        while !self.fits_after(bytes) {
            if !self.evict_one(counters) {
                // Re-insert nothing: the caller keeps its value uncached.
                return false;
            }
        }
        let tick = self.advance();
        self.entries.insert(
            key,
            CacheEntry {
                value,
                bytes,
                last_used: tick,
                leases: 0,
                dependencies,
            },
        );
        self.live_bytes += bytes;
        self.peak_bytes = self.peak_bytes.max(self.live_bytes);
        counters.created += 1;
        counters.live_entries = self.entries.len() as u64;
        counters.live_bytes = self.live_bytes;
        counters.peak_bytes = self.peak_bytes;
        true
    }

    /// Whether one more entry of `bytes` would stay within both bounds.
    fn fits_after(&self, bytes: u64) -> bool {
        // The entry about to be inserted counts as one more; the loop above has
        // already removed the entry this key may be replacing.
        let entries = self.entries.len() as u64 + 1;
        entries <= self.budget.max_entries as u64
            && self.live_bytes + bytes <= self.budget.max_bytes
    }

    /// Evicts the least recently used unleased entry, if there is one.
    fn evict_one(&mut self, counters: &mut CacheCounters) -> bool {
        let victim = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.leases == 0)
            .min_by(|(left_key, left), (right_key, right)| {
                left.last_used
                    .cmp(&right.last_used)
                    .then_with(|| left_key.cmp(right_key))
            })
            .map(|(key, _)| key.clone());
        match victim {
            Some(key) => {
                if let Some(entry) = self.entries.remove(&key) {
                    self.live_bytes -= entry.bytes;
                    counters.evicted += 1;
                    counters.live_entries = self.entries.len() as u64;
                    counters.live_bytes = self.live_bytes;
                }
                true
            }
            None => false,
        }
    }

    /// Takes a lease on an entry so it cannot be evicted.
    ///
    /// Returns whether the entry exists.  A caller that accepted work naming
    /// this entry must hold a lease until the work's completion is terminal;
    /// releasing it early re-exposes the object to eviction while the GPU may
    /// still read it.
    pub(crate) fn lease(&mut self, key: &K) -> bool {
        match self.entries.get_mut(key) {
            Some(entry) => {
                entry.leases += 1;
                true
            }
            None => false,
        }
    }

    /// Releases one lease.
    pub(crate) fn release(&mut self, key: &K) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.leases = entry.leases.saturating_sub(1);
        }
    }

    /// Whether an entry is currently leased.
    pub(crate) fn is_leased(&self, key: &K) -> bool {
        self.entries.get(key).is_some_and(|entry| entry.leases > 0)
    }

    /// Drops every entry that names `resource`, returning how many were dropped.
    pub(crate) fn invalidate_resource(
        &mut self,
        resource: ResourceRef,
        counters: &mut CacheCounters,
    ) -> usize {
        let doomed: Vec<K> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.dependencies.contains(&resource))
            .map(|(key, _)| key.clone())
            .collect();
        self.drop_keys(doomed, counters)
    }

    /// Drops every entry, without calling anything on the backend.
    ///
    /// This is the context-loss path: the backend values in these entries are
    /// no longer callable, so nothing may be destroyed through them, and the
    /// only correct action is to forget them and count the purge separately
    /// from an eviction.
    pub(crate) fn purge(&mut self, counters: &mut CacheCounters) {
        counters.purged_on_loss += self.entries.len() as u64;
        counters.live_entries = 0;
        counters.live_bytes = 0;
        self.entries.clear();
        self.live_bytes = 0;
    }

    fn drop_keys(&mut self, keys: Vec<K>, counters: &mut CacheCounters) -> usize {
        let mut dropped = 0;
        for key in keys {
            if let Some(entry) = self.entries.remove(&key) {
                self.live_bytes -= entry.bytes;
                dropped += 1;
                counters.invalidated += 1;
            }
        }
        if dropped > 0 {
            counters.live_entries = self.entries.len() as u64;
            counters.live_bytes = self.live_bytes;
        }
        dropped
    }

    /// The live keys, in key order, for a deterministic report.
    pub(crate) fn keys(&self) -> impl Iterator<Item = &K> {
        self.entries.keys()
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

    fn cache() -> StructuralCache<u32, u32> {
        StructuralCache::new(CacheBudget::new(2, 32))
    }

    #[test]
    fn eviction_is_deterministic_and_prefers_the_least_recently_used() {
        let mut cache = cache();
        let mut counters = CacheCounters::default();
        assert!(cache.insert(1, 10, 8, DependencySet::new(), &mut counters));
        assert!(cache.insert(2, 20, 8, DependencySet::new(), &mut counters));
        // Touch 1 so 2 is the least recently used.
        assert_eq!(cache.get(&1, &mut counters), Some(&10));
        assert!(cache.insert(3, 30, 8, DependencySet::new(), &mut counters));

        assert!(cache.peek(&1).is_some());
        assert!(cache.peek(&2).is_none());
        assert!(cache.peek(&3).is_some());
        assert_eq!(counters.evicted, 1);
        assert_eq!(counters.hits, 1);
    }

    #[test]
    fn a_leased_entry_is_never_evicted_and_the_insert_is_refused_instead() {
        // One entry of room, so the second insert has to evict or refuse.
        let mut cache = StructuralCache::new(CacheBudget::new(1, 32));
        let mut counters = CacheCounters::default();
        assert!(cache.insert(1, 10, 8, DependencySet::new(), &mut counters));
        assert!(cache.lease(&1));

        assert!(!cache.insert(2, 20, 8, DependencySet::new(), &mut counters));
        assert!(cache.peek(&1).is_some());
        assert!(cache.peek(&2).is_none());
        // A refused insert is not an eviction: nothing left the cache.
        assert_eq!(counters.evicted, 0);
        assert_eq!(cache.live_bytes(), 8);

        cache.release(&1);
        assert!(cache.insert(2, 20, 8, DependencySet::new(), &mut counters));
        assert!(cache.peek(&2).is_some());
        assert!(cache.peek(&1).is_none());
        assert_eq!(counters.evicted, 1);
    }

    #[test]
    fn leasing_a_key_that_is_not_present_reports_instead_of_succeeding() {
        // A domain that accepts work naming a derived object has to be able to
        // tell that the object is gone rather than hold a lease on nothing.
        let mut cache = cache();
        assert!(!cache.lease(&7));
        assert!(!cache.is_leased(&7));

        let mut counters = CacheCounters::default();
        assert!(cache.insert(7, 70, 8, DependencySet::new(), &mut counters));
        assert!(cache.lease(&7));
        assert!(cache.is_leased(&7));
    }

    #[test]
    fn a_byte_budget_bounds_a_cache_that_is_under_its_entry_budget() {
        let mut cache = StructuralCache::new(CacheBudget::new(64, 24));
        let mut counters = CacheCounters::default();
        assert!(cache.insert(1, 1, 16, DependencySet::new(), &mut counters));
        // The second entry would take the cache to 32 bytes against a 24-byte
        // budget, so the first has to go even though the entry budget is 64.
        assert!(cache.insert(2, 2, 16, DependencySet::new(), &mut counters));
        assert!(cache.peek(&1).is_none());
        assert_eq!(cache.live_bytes(), 16);
        assert_eq!(counters.peak_bytes, 16);
    }

    #[test]
    fn a_loss_purge_is_not_an_eviction() {
        let mut cache = cache();
        let mut counters = CacheCounters::default();
        assert!(cache.insert(1, 10, 8, DependencySet::new(), &mut counters));
        cache.purge(&mut counters);

        assert_eq!(cache.len(), 0);
        assert_eq!(counters.purged_on_loss, 1);
        assert_eq!(counters.evicted, 0);
        assert_eq!(counters.live_bytes, 0);
    }

    #[test]
    fn replacing_a_key_does_not_double_count_its_bytes() {
        let mut cache = StructuralCache::new(CacheBudget::new(8, 100));
        let mut counters = CacheCounters::default();
        assert!(cache.insert(1, 10, 40, DependencySet::new(), &mut counters));
        assert!(cache.insert(1, 11, 24, DependencySet::new(), &mut counters));

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
