//! MSRV-compatible atomic identity allocation.

use std::sync::atomic::{AtomicU64, Ordering};

/// Claims the current identity and advances the counter, or returns `None` on exhaustion.
///
/// This is equivalent to `fetch_update(Relaxed, Relaxed, checked_add(1))`,
/// without requiring the post-MSRV replacement API.
pub(crate) fn claim_next(counter: &AtomicU64) -> Option<u64> {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        let next = current.checked_add(1)?;
        match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return Some(current),
            Err(observed) => current = observed,
        }
    }
}

#[cfg(test)]
#[path = "tests/atomic.rs"]
mod tests;
