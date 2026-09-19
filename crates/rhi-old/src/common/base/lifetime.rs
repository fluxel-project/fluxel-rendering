//! When a resource may be released: the one rule ADR-0004 turns on.
//!
//! Fluxel's lifetime rule is short and absolute: **no lease and no physical
//! resource may be released before terminal completion, and "the backend cannot
//! presently tell" is not terminal.** `fluxel_rendergraph::CompletionStatus`
//! already documents that sentence, and the native path and the GL family each
//! implement it in their own retention type. Encoding it once means a backend
//! cannot get it wrong by re-deriving it, and a reader has one place to check.
//!
//! # Why the predicate is total over a `#[non_exhaustive]` foreign enum
//!
//! `CompletionStatus` is `#[non_exhaustive]`, so a crate outside `rendergraph`
//! cannot match it exhaustively. The rule is therefore written fail-closed: only
//! the two terminal variants release, and **every other variant -- including any
//! variant added upstream in a later release -- retains.** A future status is
//! treated as non-terminal until someone decides otherwise, which is the safe
//! direction for a lifetime rule and the same direction every capability fact in
//! this crate already fails.
//!
//! # What is deliberately not here
//!
//! No lease carrier and no `Drop` implementation. A generic carrier cannot own the
//! drop rule safely, because dropping a non-terminal lease must hand it to a
//! non-blocking retirement path -- and what that path is, and what it must keep
//! alive, is a backend's own property (the native path retires native objects
//! after a fence; the GL family retires browser objects after a sync object
//! settles). The base fixes the rule that decides *whether* release is allowed;
//! each backend owns what release costs.

use fluxel_rendergraph::CompletionStatus;

/// Whether resources retained for a submission may be released.
///
/// This is also the terminality test for a submission: a status that permits
/// release is exactly a status after which nothing more will be observed, so the
/// two questions have one answer and one function.
pub(crate) const fn may_release(status: CompletionStatus) -> bool {
    match status {
        // Completed work has finished reading its resources, and terminally
        // failed work will never read them again. Both are release points.
        CompletionStatus::Complete | CompletionStatus::Failed(_) => true,
        // `Pending` is work still in flight, and `Unknown` is acceptance or
        // completion that cannot currently be established. Neither is terminal,
        // and a status added to `CompletionStatus` later lands here by default.
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxel_rendergraph::CompletionFailure;

    #[test]
    fn completed_work_releases() {
        assert!(may_release(CompletionStatus::Complete));
    }

    #[test]
    fn a_terminal_failure_releases() {
        for failure in [
            CompletionFailure::DeviceLost,
            CompletionFailure::ExecutionFailed,
        ] {
            assert!(
                may_release(CompletionStatus::Failed(failure)),
                "{failure:?} is terminal"
            );
        }
    }

    #[test]
    fn pending_work_never_releases() {
        assert!(!may_release(CompletionStatus::Pending));
    }

    #[test]
    fn unknown_is_not_terminal_and_never_releases() {
        // The whole point of ADR-0004: a backend that cannot establish
        // completion must retain, not guess.
        assert!(!may_release(CompletionStatus::Unknown));
    }
}
