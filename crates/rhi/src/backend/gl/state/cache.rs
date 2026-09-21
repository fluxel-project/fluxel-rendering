//! Deterministic derived-object cache.  Cache values are never public handles.
use std::collections::{BTreeSet, HashMap};

use crate::backend::gl::api::{
    BufferId, FramebufferId, ProgramId, QueryId, RenderbufferId, SamplerId, ShaderId, SyncId,
    TextureId, VertexArrayId,
};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum ResourceRef {
    Buffer(BufferId),
    Texture(TextureId),
    Renderbuffer(RenderbufferId),
    Sampler(SamplerId),
    Shader(ShaderId),
    Program(ProgramId),
    VertexArray(VertexArrayId),
    Framebuffer(FramebufferId),
    Query(QueryId),
    Sync(SyncId),
}
pub(crate) type DependencySet = BTreeSet<ResourceRef>;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CacheBudget {
    pub max_entries: usize,
    pub max_bytes: u64,
}
impl CacheBudget {
    pub(crate) const fn new(max_entries: usize, max_bytes: u64) -> Self {
        Self {
            max_entries,
            max_bytes,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CacheCounters {
    pub hits: u64,
    pub misses: u64,
    pub created: u64,
    pub evicted: u64,
    pub invalidated: u64,
    pub purged: u64,
    pub live_entries: u64,
    pub live_bytes: u64,
    pub peak_bytes: u64,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CacheMutation<K, V> {
    pub removed: Vec<(K, V)>,
    pub retained: bool,
}
#[derive(Debug)]
struct Entry<V> {
    value: V,
    bytes: u64,
    used: u64,
    sequence: u64,
    deps: DependencySet,
}
#[derive(Debug)]
pub(crate) struct StructuralCache<K, V> {
    budget: CacheBudget,
    entries: HashMap<K, Entry<V>>,
    bytes: u64,
    peak: u64,
    tick: u64,
}
impl<K: Clone + Eq + std::hash::Hash, V> StructuralCache<K, V> {
    pub(crate) fn new(budget: CacheBudget) -> Self {
        Self {
            budget,
            entries: HashMap::new(),
            bytes: 0,
            peak: 0,
            tick: 0,
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
    pub(crate) const fn live_bytes(&self) -> u64 {
        self.bytes
    }
    fn tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }
    fn report(&self, c: &mut CacheCounters) {
        c.live_entries = self.entries.len() as u64;
        c.live_bytes = self.bytes;
        c.peak_bytes = self.peak;
    }
    pub(crate) fn get(&mut self, key: &K, c: &mut CacheCounters) -> Option<&V> {
        let t = self.tick();
        match self.entries.get_mut(key) {
            Some(e) => {
                e.used = t;
                c.hits += 1;
                Some(&e.value)
            }
            None => {
                c.misses += 1;
                None
            }
        }
    }
    pub(crate) fn insert(
        &mut self,
        key: K,
        value: V,
        bytes: u64,
        deps: DependencySet,
        c: &mut CacheCounters,
    ) -> CacheMutation<K, V> {
        let prior = self.entries.remove(&key);
        let prior_bytes = prior.as_ref().map_or(0, |e| e.bytes);
        let mut projected = self.bytes - prior_bytes;
        if bytes > self.budget.max_bytes || self.budget.max_entries == 0 {
            if let Some(e) = prior {
                self.entries.insert(key, e);
            }
            return CacheMutation {
                removed: vec![],
                retained: false,
            };
        }
        let mut victims: Vec<(u64, K)> = self
            .entries
            .iter()
            .map(|(k, e)| (e.used, k.clone()))
            .collect();
        victims.sort_by_key(|x| x.0);
        let mut remove = Vec::new();
        while self.entries.len() - remove.len() + 1 > self.budget.max_entries
            || projected + bytes > self.budget.max_bytes
        {
            let Some((_, k)) = victims.get(remove.len()) else {
                if let Some(e) = prior {
                    self.entries.insert(key, e);
                }
                return CacheMutation {
                    removed: vec![],
                    retained: false,
                };
            };
            projected -= self.entries.get(k).unwrap().bytes;
            remove.push(k.clone());
        }
        let mut removed = Vec::new();
        if let Some(e) = prior {
            self.bytes -= e.bytes;
            removed.push((key.clone(), e.value));
        }
        for k in remove {
            let e = self.entries.remove(&k).unwrap();
            self.bytes -= e.bytes;
            c.evicted += 1;
            removed.push((k, e.value));
        }
        let t = self.tick();
        self.entries.insert(
            key,
            Entry {
                value,
                bytes,
                used: t,
                sequence: t,
                deps,
            },
        );
        self.bytes += bytes;
        self.peak = self.peak.max(self.bytes);
        c.created += 1;
        self.report(c);
        CacheMutation {
            removed,
            retained: true,
        }
    }
    pub(crate) fn invalidate_resource(
        &mut self,
        resource: ResourceRef,
        c: &mut CacheCounters,
    ) -> Vec<(K, V)> {
        let mut keys: Vec<(u64, K)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.deps.contains(&resource))
            .map(|(k, e)| (e.sequence, k.clone()))
            .collect();
        keys.sort_by_key(|x| x.0);
        let mut result = Vec::new();
        for (_, k) in keys {
            let e = self.entries.remove(&k).unwrap();
            self.bytes -= e.bytes;
            c.invalidated += 1;
            result.push((k, e.value));
        }
        self.report(c);
        result
    }
    pub(crate) fn drain(&mut self, c: &mut CacheCounters) -> Vec<(K, V)> {
        let mut keys: Vec<(u64, K)> = self
            .entries
            .iter()
            .map(|(k, e)| (e.sequence, k.clone()))
            .collect();
        keys.sort_by_key(|x| x.0);
        let r = keys
            .into_iter()
            .map(|(_, k)| {
                let e = self.entries.remove(&k).unwrap();
                (k, e.value)
            })
            .collect();
        self.bytes = 0;
        self.report(c);
        r
    }
    pub(crate) fn purge(&mut self, c: &mut CacheCounters) {
        c.purged += self.entries.len() as u64;
        self.entries.clear();
        self.bytes = 0;
        self.report(c);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn texture() -> ResourceRef {
        use crate::backend::gl::api::{ContextEpoch, ContextStamp, DeviceIdentity, TextureId};
        ResourceRef::Texture(TextureId::new(
            ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL),
            1,
            1,
        ))
    }
    #[test]
    fn lru_and_dependency_invalidation_are_deterministic() {
        let mut c = StructuralCache::new(CacheBudget::new(2, 20));
        let mut n = CacheCounters::default();
        c.insert(1, 1, 8, DependencySet::new(), &mut n);
        c.insert(2, 2, 8, DependencySet::new(), &mut n);
        c.get(&1, &mut n);
        assert_eq!(
            c.insert(3, 3, 8, DependencySet::new(), &mut n).removed,
            vec![(2, 2)]
        );
        let mut d = DependencySet::new();
        d.insert(texture());
        c.insert(4, 4, 8, d, &mut n);
        assert_eq!(c.invalidate_resource(texture(), &mut n), vec![(4, 4)]);
    }
}
