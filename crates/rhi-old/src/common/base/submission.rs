//! What happens to a submission's leases while its completion is not terminal.
//!
//! ADR-0004 is a rule about ownership, and [`super::lifetime::may_release`] states
//! the half of it that reads a status. This module states the other half: **who
//! holds the leases while the status is not terminal, and what happens when the
//! owner goes away.**
//!
//! ```text
//! Observed  --observe(Complete | Failed)-->  Terminal  (leases may be released)
//!    |  \--observe(Pending | Unknown)-->  Observed
//!    |
//!    `--abandon()-->  Abandoned  --observe(terminal)-->  Terminal
//! ```
//!
//! # Why `Abandoned` is a distinct state and not a boolean
//!
//! A dropped submission and a submission whose owner is still polling are both
//! non-terminal, but they differ in the one thing that matters to a retirement
//! path: whether anyone will poll again. Dropping a non-terminal submission must
//! **transfer** its completion and leases to a non-blocking retirement path rather
//! than release them, and that path needs to know it is now the owner. A boolean
//! "is terminal" cannot carry that; the three states can.
//!
//! # Why re-observing a terminal submission is an error
//!
//! Once a disposition is terminal, whatever held the leases has been released. A
//! later observation would be an observation of a submission that no longer exists
//! -- the use-after-release shape -- so it is refused as a value rather than
//! silently accepted. The alternative, treating it as idempotent, would let a
//! polling loop that outlived its own retirement keep running with no evidence
//! that anything was wrong.
//!
//! # What is deliberately not here
//!
//! No queue, no poll driver, no native or browser object. This is the decision a
//! backend's retirement path makes, not the mechanism it uses to make it.

use fluxel_rendergraph::CompletionStatus;

use crate::common::base::lifetime::may_release;

/// Where a submission's leases are held, as of the last observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Disposition {
    /// An owner is still observing this submission and holds its leases.
    Observed,
    /// The owner is gone; a retirement path holds the leases until terminal.
    ///
    /// Reached by dropping a non-terminal submission, which is an ownership
    /// transfer and not a release.
    Abandoned,
    /// Completion is terminal, so the leases have been released.
    Terminal,
}

/// A submission that was observed after its leases were already released.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AlreadyTerminal;

impl Disposition {
    /// The disposition of a submission that was just accepted.
    pub(crate) const fn accepted() -> Self {
        Self::Observed
    }

    /// Whether the leases may be released in this disposition.
    ///
    /// Only [`Self::Terminal`] permits release: `Abandoned` still holds them, and
    /// it is exactly the state that must not be mistaken for release.
    pub(crate) const fn may_release(self) -> bool {
        matches!(self, Self::Terminal)
    }

    /// Returns the disposition after observing `status`, or refuses to observe a
    /// submission whose leases are already gone.
    ///
    /// Terminality is decided by [`may_release`] rather than restated here, so the
    /// status rule and the ownership rule cannot disagree.
    pub(crate) fn observe(self, status: CompletionStatus) -> Result<Self, AlreadyTerminal> {
        if self == Self::Terminal {
            return Err(AlreadyTerminal);
        }
        Ok(if may_release(status) {
            Self::Terminal
        } else {
            self
        })
    }

    /// Transfers ownership to a retirement path.
    ///
    /// Total rather than fallible, because abandoning an already-terminal
    /// submission is not a mistake a caller can make in a useful way: the leases
    /// are gone, so there is nothing to transfer and nothing to report. The state
    /// stays `Terminal` instead of the transfer inventing a second terminal state.
    pub(crate) const fn abandon(self) -> Self {
        match self {
            Self::Observed => Self::Abandoned,
            Self::Abandoned | Self::Terminal => self,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxel_rendergraph::CompletionFailure;

    #[test]
    fn an_accepted_submission_holds_its_leases() {
        let disposition = Disposition::accepted();
        assert_eq!(disposition, Disposition::Observed);
        assert!(!disposition.may_release());
    }

    #[test]
    fn non_terminal_statuses_keep_the_disposition() {
        for status in [CompletionStatus::Pending, CompletionStatus::Unknown] {
            let disposition = Disposition::accepted()
                .observe(status)
                .expect("not terminal yet");
            assert_eq!(disposition, Disposition::Observed, "{status:?}");
            assert!(!disposition.may_release());
        }
    }

    #[test]
    fn an_abandoned_submission_still_keeps_non_terminal_statuses() {
        // A retirement path polls; a non-terminal answer must not release.
        let abandoned = Disposition::accepted().abandon();
        assert_eq!(abandoned, Disposition::Abandoned);
        let still = abandoned
            .observe(CompletionStatus::Unknown)
            .expect("not terminal yet");
        assert_eq!(still, Disposition::Abandoned);
        assert!(!still.may_release());
    }

    #[test]
    fn terminal_statuses_release_from_either_non_terminal_state() {
        for start in [Disposition::Observed, Disposition::Abandoned] {
            for status in [
                CompletionStatus::Complete,
                CompletionStatus::Failed(CompletionFailure::DeviceLost),
            ] {
                let terminal = start.observe(status).expect("not terminal yet");
                assert_eq!(terminal, Disposition::Terminal, "{start:?} {status:?}");
                assert!(terminal.may_release());
            }
        }
    }

    #[test]
    fn observing_a_released_submission_is_refused() {
        let terminal = Disposition::accepted()
            .observe(CompletionStatus::Complete)
            .expect("first observation succeeds");
        assert_eq!(
            terminal.observe(CompletionStatus::Complete),
            Err(AlreadyTerminal)
        );
        // Even a non-terminal answer is refused: the question is whether the
        // submission still exists, not what the answer would have been.
        assert_eq!(
            terminal.observe(CompletionStatus::Pending),
            Err(AlreadyTerminal)
        );
    }

    #[test]
    fn abandoning_a_terminal_submission_stays_terminal() {
        let terminal = Disposition::accepted()
            .observe(CompletionStatus::Complete)
            .expect("first observation succeeds");
        assert_eq!(terminal.abandon(), Disposition::Terminal);
        assert!(terminal.abandon().may_release());
    }

    #[test]
    fn abandoning_twice_changes_nothing() {
        let abandoned = Disposition::accepted().abandon().abandon();
        assert_eq!(abandoned, Disposition::Abandoned);
        assert!(!abandoned.may_release());
    }
}
