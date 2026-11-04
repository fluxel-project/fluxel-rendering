//! Consumer-local opaque ordering. Compatibility policy belongs to this Feature.
use fluxel_renderer::RenderId;
use std::collections::BTreeMap;

/// Groups CPU draw candidates by the consumer's complete compatibility facts.
/// Each record remains a separate draw; this does not imply instancing.
pub(crate) fn group<K: Ord>(
    items: impl IntoIterator<Item = (K, RenderId)>,
) -> Vec<(K, Vec<RenderId>)> {
    let mut batches = BTreeMap::<K, Vec<RenderId>>::new();
    for (key, id) in items {
        batches.entry(key).or_default().push(id);
    }
    for ids in batches.values_mut() {
        ids.sort_unstable_by_key(|id| (id.slot(), id.generation()));
    }
    batches.into_iter().collect()
}
