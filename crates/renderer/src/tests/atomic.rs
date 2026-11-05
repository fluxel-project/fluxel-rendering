use std::{
    collections::BTreeSet,
    sync::{Arc, atomic::AtomicU64},
    thread,
};

use super::claim_next;

#[test]
fn claims_every_identity_once_across_concurrent_callers() {
    let counter = Arc::new(AtomicU64::new(1));
    let mut workers = Vec::new();
    for _ in 0..8 {
        let counter = Arc::clone(&counter);
        workers.push(thread::spawn(move || {
            (0..32)
                .map(|_| claim_next(&counter).expect("identity space remains available"))
                .collect::<Vec<_>>()
        }));
    }

    let claimed = workers
        .into_iter()
        .flat_map(|worker| worker.join().expect("worker must not panic"))
        .collect::<BTreeSet<_>>();
    assert_eq!(claimed.len(), 256);
    assert_eq!(claimed.first(), Some(&1));
    assert_eq!(claimed.last(), Some(&256));
    assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 257);
}

#[test]
fn preserves_an_exhausted_counter() {
    let counter = AtomicU64::new(u64::MAX);
    assert_eq!(claim_next(&counter), None);
    assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), u64::MAX);
}
