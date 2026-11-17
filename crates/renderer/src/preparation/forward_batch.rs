//! Adjacent Forward draw batching and source-range coalescing.
use super::*;

pub(crate) fn batch_forward_draws(draws: Vec<ForwardDrawInput>) -> ForwardPreparedBatches {
    let runs = coalesce_adjacent(draws, ForwardCompatibilityKey::from_draw);
    let mut ordered = Vec::new();
    let mut batches = Vec::with_capacity(runs.len());
    for (key, members) in runs {
        let start = ordered.len();
        ordered.extend(members);
        batches.push(ForwardDrawBatch {
            key,
            draws: start..ordered.len(),
        });
    }
    ForwardPreparedBatches {
        draws: ordered,
        batches,
    }
}

fn coalesce_adjacent<T, K>(items: Vec<T>, key_for: impl Fn(&T) -> K) -> Vec<(K, Vec<T>)>
where
    K: Eq,
{
    let mut runs: Vec<(K, Vec<T>)> = Vec::new();
    for item in items {
        let key = key_for(&item);
        match runs.last_mut() {
            Some((last_key, members)) if *last_key == key => members.push(item),
            _ => runs.push((key, vec![item])),
        }
    }
    runs
}

#[cfg(test)]
#[path = "../tests/preparation/forward_batches.rs"]
mod forward_batches;

#[cfg(test)]
#[path = "../tests/preparation/feature_draws.rs"]
mod feature_draws;

#[cfg(test)]
#[path = "../tests/preparation/forward_pbr.rs"]
mod forward_pbr_tests;

#[cfg(test)]
#[path = "../tests/preparation/color_uv0.rs"]
mod color_uv0_tests;

#[cfg(test)]
#[path = "../tests/preparation/forward_texture_validation.rs"]
mod forward_texture_validation_tests;

#[cfg(test)]
#[path = "../tests/preparation/forward_sampler.rs"]
mod forward_sampler_tests;
