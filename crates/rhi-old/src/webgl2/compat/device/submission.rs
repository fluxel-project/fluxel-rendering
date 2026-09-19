//! The bounded ledger that makes a completion query total without polling.
//!
//! The common contract asks two different questions about a submission, and the
//! GL family can only answer one of them from the fence.  `retire` and
//! `collect_retired` are `&mut self` and may poll; `completion_status` is
//! `&self` and may not -- Layer 1's `poll_fence` takes `&mut self`, because
//! polling a fence is an owning-thread driver call like every other verb.  So a
//! backend that answered `completion_status` by polling would not compile, and
//! one that answered it some other way would be inventing an answer.
//!
//! What it does instead is *record* the answers it already obtained.  Every
//! outcome this adapter ever learns is learned at a poll, and is kept here until
//! someone asks for it.  `completion_status` then reads the record, and where
//! there is no record it answers [`CompletionStatus::Unknown`] -- the
//! contract's own fail-closed answer, documented as "deliberately distinct from
//! a terminal failure" with the instruction that callers "must retain every
//! lease and must not recycle any physical resource while the status is
//! unknown".  Not knowing is a thing this contract can say; guessing is not.
//!
//! # Why the key is the whole lease
//!
//! The record is keyed on the `GlFenceLease` and not on its `SyncId`, because a
//! `SyncId` is a slot in Layer 1's allocation table and a slot is reused: a
//! destroyed fence's slot can be handed out again for a new fence, and a record
//! keyed on the slot would report the old outcome for the new work.  The lease
//! carries a private monotonically increasing serial that is never reused
//! (`api/sync.rs`), and its `PartialEq` compares it, so two leases compare equal
//! exactly when they are the same issuance.
//!
//! # The two lists, and the bound on each
//!
//! `live` holds submissions with no terminal outcome yet.  It is bounded by
//! construction rather than by a window: only a submission that reached
//! `create_fence` gets an entry, and a context that cannot accept work cannot
//! add one.
//!
//! `retired` holds outcomes that *were* terminal, and it is the list that needs
//! a bound, because it grows on every completed submission whether or not anyone
//! asks about it.  When it is full the oldest outcome is dropped, which
//! eventually makes some very old completion answer `Unknown` again.  That is
//! the fail-closed direction and it is why the window can be a plain count: an
//! evicted outcome costs its holder a quarantine, never a premature release.
//!
//! `live` is deliberately not trimmed the same way.  A submission that never
//! becomes terminal -- because the context was lost, or because nothing ever
//! retires it -- keeps its leases, and dropping the entry to stay under a count
//! would release objects that submitted GPU work may still reference.

use std::collections::VecDeque;

use fluxel_rendergraph::{CompletionFailure, CompletionStatus, PresentationSubmission, QueueId};

use crate::webgl2::api::{GlContextLifecycle, GlError, GlFenceLease, GlFenceStatus};
use crate::webgl2::state::GlStateBackend;

use super::compute::ComputeDomain;
use super::encoder::GlCommandBuffer;
use super::retention::GlRetentionLease;
use super::{GlCompatibilityDevice, GlSurfaceToken};

/// How many terminal outcomes are remembered past the submission that made them.
///
/// Sixty-four is a window and not a promise.  It is chosen against the one
/// consumer that polls late: the executor's transient pool asks about a
/// submission on the frame after it completes, and the pool itself is bounded,
/// so a window an order of magnitude above the pool's own depth cannot evict an
/// outcome before its slot is reclaimed.
pub(super) const RETIRED_OUTCOME_WINDOW: usize = 64;

/// One submission this adapter made and has not seen reach a terminal outcome.
struct Submission {
    fence: GlFenceLease,
    status: CompletionStatus,
    /// Leases that were handed over for retirement and are waiting with it.
    leases: Vec<GlRetentionLease>,
}

/// What a retirement found, and so what its caller must do with the leases.
///
/// The leases are given back rather than dropped here so that the release lands
/// at a call site that can say why it is happening.  Dropping a retention lease
/// is not silent: it records the object in the release queue, and the adapter
/// destroys it at its next entry point.
#[derive(Debug)]
pub(super) enum Retirement {
    /// The submission has not reached a terminal outcome.  The leases are held
    /// here and released by whichever poll settles it.
    Held,
    /// Nothing here will ever release these leases, so the caller must.  Either
    /// the submission already reached a terminal outcome -- the common case, a
    /// caller that polls its own submission and then drops it never retires it
    /// at all -- or this adapter never issued the completion, in which case the
    /// leases name no work of ours to wait for.  Holding them forever in that
    /// second case would turn a caller's mistake into an unbounded leak, and
    /// releasing an object no submission of ours references cannot be unsafe.
    Release(Vec<GlRetentionLease>),
}

/// Every submission this adapter has made, and the outcomes it observed.
#[derive(Default)]
pub(super) struct SubmissionLedger {
    live: Vec<Submission>,
    retired: VecDeque<(GlFenceLease, CompletionStatus)>,
}

/// Whether an outcome ends a submission's life.
///
/// `Pending` and `Unknown` do not.  `Unknown` is the one worth naming: the
/// contract calls it out as "deliberately distinct from a terminal failure", so
/// a record that happens to read `Unknown` keeps its leases exactly like one
/// that reads `Pending`.
pub(super) const fn is_terminal(status: CompletionStatus) -> bool {
    matches!(
        status,
        CompletionStatus::Complete | CompletionStatus::Failed(_)
    )
}

impl SubmissionLedger {
    /// Records one freshly submitted fence.
    pub(super) fn record(&mut self, fence: GlFenceLease) {
        self.live.push(Submission {
            fence,
            status: CompletionStatus::Pending,
            leases: Vec::new(),
        });
    }

    /// Makes one live submission terminal without waiting to learn anything.
    ///
    /// The one caller is a submission whose presentation failed after the point
    /// where the contract stops allowing an `Err`: there is nothing left to poll
    /// *for*, because the fence reports the commands and not the present, and
    /// leaving the entry pending would make a frame that never reached the
    /// drawable look exactly like a frame still in flight.
    pub(super) fn fail(&mut self, fence: GlFenceLease, failure: CompletionFailure) {
        if let Some(pending) = self.live.iter_mut().find(|pending| pending.fence == fence) {
            pending.status = CompletionStatus::Failed(failure);
        }
    }

    /// The outcome recorded for `fence`, or `Unknown` where none is recorded.
    pub(super) fn outcome(&self, fence: &GlFenceLease) -> CompletionStatus {
        if let Some(submission) = self.live.iter().find(|pending| pending.fence == *fence) {
            return submission.status;
        }
        if let Some((_, status)) = self.retired.iter().find(|(known, _)| known == fence) {
            return *status;
        }
        CompletionStatus::Unknown
    }

    /// Every submission that has not reached a terminal outcome, with its fence.
    ///
    /// Returned as pairs rather than as indices because the caller polls these
    /// while holding a borrow of the backend, and the answers come back by fence
    /// -- an index would be invalidated by the very call that settles one.
    pub(super) fn unsettled(&self) -> impl Iterator<Item = GlFenceLease> + '_ {
        self.live
            .iter()
            .filter(|pending| !is_terminal(pending.status))
            .map(|pending| pending.fence)
    }

    /// Applies the outcomes one poll observed and hands back what they settle.
    ///
    /// A submission whose outcome is terminal is moved out of `live` into the
    /// bounded outcome window.  Two things come back to the caller, because it is
    /// the only party that can act on either: the leases the submission was
    /// holding, whose objects it must destroy, and the fence itself, which is a
    /// driver object that nothing will ask about again -- the outcome window is
    /// compared against, never polled, so the object behind a key may go as soon
    /// as the key stops being the only answer.
    ///
    /// Submissions absent from `observed` keep the outcome they already had,
    /// which is how a poll that failed leaves a submission quarantined.  A
    /// submission that is *already* terminal keeps its outcome for the same
    /// reason and one more: [`Self::fail`] records a failure the fence cannot
    /// report, and a poll that overwrote it would replace a known-missing frame
    /// with a fence's own good news about the commands.
    pub(super) fn settle(
        &mut self,
        observed: &[(GlFenceLease, CompletionStatus)],
    ) -> (Vec<GlRetentionLease>, Vec<GlFenceLease>) {
        let live = core::mem::take(&mut self.live);
        let mut kept = Vec::with_capacity(live.len());
        let mut released = Vec::new();
        let mut finished = Vec::new();
        for mut submission in live {
            if !is_terminal(submission.status) {
                if let Some((_, status)) = observed
                    .iter()
                    .find(|(fence, _)| *fence == submission.fence)
                {
                    submission.status = *status;
                }
            }
            if is_terminal(submission.status) {
                released.append(&mut submission.leases);
                finished.push((submission.fence, submission.status));
            } else {
                kept.push(submission);
            }
        }
        self.live = kept;
        let mut settled = Vec::with_capacity(finished.len());
        for (fence, status) in finished {
            self.remember(fence, status);
            settled.push(fence);
        }
        (released, settled)
    }

    /// Hands `leases` to the submission `fence` names, if it is still live.
    pub(super) fn retire(
        &mut self,
        fence: GlFenceLease,
        leases: Vec<GlRetentionLease>,
    ) -> Retirement {
        match self.live.iter_mut().find(|pending| pending.fence == fence) {
            Some(pending) => {
                pending.leases.extend(leases);
                Retirement::Held
            }
            None => Retirement::Release(leases),
        }
    }

    /// Drops every record without acting on it.
    ///
    /// What a context-generation change calls.  Every fence of the previous
    /// epoch was invalidated by the provider before it reported `Active` again,
    /// so there is no outcome left to learn and no lease left worth holding --
    /// and the leases go back to the caller to release, which is safe for the
    /// same reason [`ReleaseQueue::forget`](super::retention::ReleaseQueue::forget)
    /// is: the objects are already gone.
    pub(super) fn purge(&mut self) -> Vec<GlRetentionLease> {
        self.retired.clear();
        let mut released = Vec::new();
        for mut pending in core::mem::take(&mut self.live) {
            released.append(&mut pending.leases);
        }
        released
    }

    /// Adds one outcome to the window, evicting the oldest when it is full.
    fn remember(&mut self, fence: GlFenceLease, status: CompletionStatus) {
        if self.retired.len() == RETIRED_OUTCOME_WINDOW {
            self.retired.pop_front();
        }
        self.retired.push_back((fence, status));
    }
}

/// The common contract's failure for a fence the GL family reports as failed.
///
/// Layer 1's `GlFenceStatus::Failed` says a fence failed and nothing more, while
/// the contract asks which of two failures it was; that is a fact about the
/// context, and the lifecycle is where the context keeps it.  A context still
/// reporting `Active` while one of its own fences fails is a submission that
/// failed, and any other lifecycle is a device the caller can no longer use --
/// which is the distinction `DeviceLost` exists to carry.  The two states that
/// are neither (`Inactive`, `Suspended`) cannot have a fence reported against
/// them at all: submitting requires `Active`, and `GlSyncApi`'s preflight
/// refuses every sync verb outside it.
pub(super) fn failure(lifecycle: GlContextLifecycle) -> CompletionFailure {
    match lifecycle {
        GlContextLifecycle::Active => CompletionFailure::ExecutionFailed,
        _ => CompletionFailure::DeviceLost,
    }
}

/// The common completion state one GL fence report stands for.
///
/// Exhaustive rather than wildcarded: `GlFenceStatus` is this crate's own type
/// and is not `#[non_exhaustive]`, so a new variant is a change to the
/// GL-family contract and should stop this lowering rather than fall into a
/// default.  `Failed` is the one that needs a second fact: the GL family says a
/// fence failed and nothing more, while the contract asks which of two failures
/// it was, and the lifecycle is where the context keeps that.
fn completion_status(status: GlFenceStatus, lifecycle: GlContextLifecycle) -> CompletionStatus {
    match status {
        GlFenceStatus::Pending => CompletionStatus::Pending,
        // The report says the signal did not happen.  Reporting a failure would
        // be inventing one, and the contract already has a name for not knowing.
        GlFenceStatus::Unknown => CompletionStatus::Unknown,
        GlFenceStatus::Complete => CompletionStatus::Complete,
        GlFenceStatus::Failed => CompletionStatus::Failed(failure(lifecycle)),
    }
}

/// The three verbs of [`super::backend`] that read or advance this ledger, and
/// the submission that feeds it.
///
/// They live here for the reason the module documentation gives: this file is
/// where the ledger's keying, its two lists and the bound on each are argued,
/// and a verb that drives it is that argument's executable half.  What stays in
/// [`super::backend`] is the trait's own declaration order.
impl<B: GlStateBackend, C: ComputeDomain<B>> GlCompatibilityDevice<B, C> {
    /// The body of [`super::backend`]'s `submit`.
    ///
    /// The contract's verb carries, with each token, the presentation root that
    /// token satisfies -- and this adapter never reads it.  It does not need to:
    /// the pairing it acts on is already inside the token, which was minted for
    /// exactly one acquisition and consumed by exactly one publish, so the two
    /// could not disagree even if the list named a different root.  The tokens are
    /// therefore taken out here, and the whole of the submission is
    /// [`Self::submit_tokens`] -- which is also what makes the presenting half
    /// reachable from a test, since a `PresentTarget` can only be minted by the
    /// graph that declared the root.
    pub(super) fn submit_commands(
        &mut self,
        queue: QueueId,
        command_buffer: GlCommandBuffer,
        presentations: Vec<PresentationSubmission<GlSurfaceToken>>,
    ) -> Result<GlFenceLease, GlError> {
        let tokens = presentations
            .into_iter()
            .map(|presentation| presentation.token)
            .collect();
        self.submit_tokens(queue, command_buffer, tokens)
    }

    /// One submission and the acquisitions it presents.
    ///
    /// # Where a present sits between the contract's two guarantees
    ///
    /// The contract gives `submit` two rules that pull in opposite directions
    /// (`rendergraph/src/backend/contract.rs`): an `Err` guarantees that no
    /// presentation request reached a native queue, and after command acceptance
    /// the answer must be `Ok` even if presentation fails.  For a backend that
    /// presents by issuing commands on the same ordered path, the publishes *are*
    /// part of the submission, so the boundary is drawn where the contract draws
    /// it: everything that can refuse without touching the queue is answered
    /// first, and from the first publish attempt onward the answer is the
    /// completion -- with the failure recorded on it, which is the mechanism the
    /// second rule names.  A publish that refuses before its own driver call is
    /// answered the same way as one that fails inside it, because the two are one
    /// `GlError` by the time they arrive here and reading which it was would mean
    /// asking the provider a question its contract does not answer.
    ///
    /// The publishes come before `create_fence` so that the fence covers them --
    /// a fence reports the commands issued before it, and a present issued after
    /// one would be a present the completion says nothing about.  That is also
    /// the weaker of the two orderings for the *error* path: a publish the
    /// backend accepted hands its token's retention to this ledger, so the source
    /// texture outlives a fence that may have signalled before the blit read it,
    /// while a publish that refuses drops its retention where it stands, as do
    /// the tokens left unconsumed behind it.  That is one handle of a lifetime and
    /// not the last one: the frame's own bound texture holds another, the object
    /// dies when the last handle goes, and a refused publish issued no command
    /// that reads the source -- which is the only thing this ledger exists to
    /// outlive.
    pub(super) fn submit_tokens(
        &mut self,
        queue: QueueId,
        command_buffer: GlCommandBuffer,
        tokens: Vec<GlSurfaceToken>,
    ) -> Result<GlFenceLease, GlError> {
        if queue != QueueId::new(0) {
            return Err(GlError::Unsupported {
                operation: "submit",
                reason: "this backend has one ordered command path, which the common contract names as queue zero",
            });
        }
        self.refresh();
        self.release_pending()?;
        let backend = self.machine.backend();
        backend.validate_object_context("submit", command_buffer.context)?;
        // Every token is presented here, and a refusal stops the loop with the
        // rest still unconsumed -- dropping a token is the cancellation the
        // contract asks for, and a token that was never presented is a frame that
        // never reached the drawable.
        let mut presented = Vec::with_capacity(tokens.len());
        let mut refused = false;
        for token in tokens {
            let GlSurfaceToken {
                lease,
                texture,
                retention,
            } = token;
            match backend.publish_surface_image(lease, texture) {
                Ok(()) => presented.push(retention),
                Err(_) => {
                    refused = true;
                    break;
                }
            }
        }
        // `flush` makes the commands issued before it visible to the device, and
        // `create_fence` then inserts a fence they are ordered before.  Swapping
        // the two would make the fence report the previous submission, which is
        // the one mistake this pair can make.
        backend.flush()?;
        let fence = backend.create_fence()?;
        self.submissions.record(fence);
        if !presented.is_empty() {
            // Held until this submission settles rather than dropped here: the
            // blit that reads the source is issued before the fence, and a fence
            // is not a promise about commands issued after it.
            let retirement = self.submissions.retire(fence, presented);
            debug_assert!(
                matches!(retirement, Retirement::Held),
                "the submission was just recorded, so it is live"
            );
        }
        // A present that failed after an earlier one succeeded cannot be an `Err`:
        // a presentation request did reach a native queue, and the contract's
        // answer for that is a completion the executor can retire against.  The
        // submission is recorded as failed rather than left pending, so a frame
        // that did not reach the drawable is not reported as one that arrives a
        // poll later and looks fine.  The driver's own reason has nowhere to go --
        // `CompletionFailure` is a two-case fact and the contract has no channel
        // for a diagnostic -- and what the executor acts on is the terminal
        // outcome, which is what this records.
        if refused {
            let context = self.machine.backend().lifecycle();
            self.submissions.fail(fence, failure(context));
        }
        Ok(fence)
    }

    /// The body of [`super::backend`]'s `retire`.
    pub(super) fn retire_completion(
        &mut self,
        completion: GlFenceLease,
        leases: Vec<GlRetentionLease>,
    ) {
        match self.submissions.retire(completion, leases) {
            // The submission is still in flight: it holds them, and whichever
            // poll settles it releases them.
            Retirement::Held => {}
            // This is where they are released, and it is a real act: dropping a
            // retention lease records its object in the release queue, and the
            // frame's own `collect_retired` destroys it -- the call the executor
            // makes immediately after this one.
            Retirement::Release(leases) => drop(leases),
        }
    }

    /// The body of [`super::backend`]'s `collect_retired`.
    pub(super) fn collect_terminal_outcomes(&mut self) -> Result<usize, GlError> {
        self.refresh();
        // Every unsettled fence is polled under one borrow of the backend, and
        // the answers are applied outside it.  A fence the backend refuses to
        // describe keeps whatever outcome it already had, which is how a failed
        // poll leaves a submission quarantined instead of releasing work that
        // may still be running; the first refusal is what this call reports.
        let fences: Vec<GlFenceLease> = self.submissions.unsettled().collect();
        let mut observed = Vec::with_capacity(fences.len());
        let mut first_error = None;
        for fence in fences {
            match self.machine.backend().poll_fence(fence) {
                Ok(status) => observed.push((
                    fence,
                    completion_status(status, self.machine.backend().lifecycle()),
                )),
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        let (released, finished) = self.submissions.settle(&observed);
        // The count is the settlements and not the leases.  Every other backend
        // in this workspace reports how many retirement entries the poll
        // released, and one public number with two meanings is worse than a
        // slightly loose name.  It is a superset of theirs by construction: this
        // ledger tracks a submission from the moment it is issued, because
        // `completion_status` has to be answerable before anyone retires
        // anything, while a queue of retirements can only hold what was handed
        // over.
        let count = finished.len();
        // Released here rather than at the next entry point, so the frame that
        // learned the work is done is the frame that frees it.
        drop(released);
        for fence in finished {
            // A fence is a driver object and not a token.  Everything this
            // adapter can still be asked about a settled submission comes from
            // the recorded outcome, and the record is only ever compared against
            // -- never polled -- so the object goes as soon as it can be asked
            // nothing, and the key it leaves behind stays valid for exactly as
            // long as the record does.
            if let Err(error) = self.machine.backend().destroy_fence(fence) {
                first_error.get_or_insert(error);
            }
        }
        // Run whatever the release queue collected even when a poll or a destroy
        // failed: the objects in it are already unreachable, and deferring them
        // would only postpone the same call to a frame that may not come.
        let release_error = self.release_pending().err();
        match first_error.or(release_error) {
            Some(error) => Err(error),
            None => Ok(count),
        }
    }
}
