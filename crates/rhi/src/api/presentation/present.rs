//! Present planning identity and present outcome (specification section 45).
//!
//! What a plan's presentation is called, and what became of it: the plan-local
//! present identity the builder hands back, the failure and state vocabulary a
//! present terminates in, and the receipt `Device::submit` returns alongside the
//! completion token. It does not own the frame (`frame.rs`), the plan it belongs
//! to (section 39), or the closure rules that decide which work may touch a frame
//! (section 45.3, which is the builder's).
//!
//! Invariant: a present outcome and a GPU completion are **independent**. Section
//! 45.5 fixes the four ways they can disagree — GPU complete with the present
//! outdated or target-lost, both lost — and forbids the one inference that would
//! collapse them:
//!
//! ```text
//! a present failure is NOT "the work was not submitted"
//! Accepted is NOT "the screen has displayed it"
//! ```
//!
//! `Accepted` here means the presentation system or host lifecycle took ownership
//! of the frame. Metal's `present(_:)` schedules drawable presentation during
//! command-buffer scheduling rather than reporting scan-out, which is exactly why
//! the portable vocabulary stops at ownership.

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::DeviceIdentity;
use crate::api::platform::{Device, DeviceLossInfo};
use crate::api::submission::SubmissionPlanId;

/// Identity of one frame's presentation inside one plan.
///
/// Section 45.1 gives a plan a local present identity per presented frame, so a
/// receipt can name which presentation it is about without naming the frame again
/// — the frame token is consumed by `present_after`, and a receipt that carried an
/// `AcquiredFrameId` would keep a live-looking handle to a frame whose ownership
/// has already moved.
///
/// There is no public constructor, for the reason section 39.1 gives about
/// `PlanPoint`: a present identity is assigned by the builder that owns the plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PresentPlanId {
    plan: SubmissionPlanId,
    local: u32,
}

impl PresentPlanId {
    /// Names the presentation of one frame inside a plan.
    ///
    /// Crate-private: the plan identity and the local counter are both the
    /// builder's, so only `present_after` may mint one.
    ///
    /// No dead-code annotation is needed: `SubmissionPlanBuilder::present_after`
    /// mints one in this crate.
    pub(crate) fn new(plan: SubmissionPlanId, local: u32) -> Self {
        Self { plan, local }
    }

    /// The plan this presentation belongs to.
    ///
    /// Crate-private: it is what lets the builder check that a present plan
    /// belongs to the plan being built, and what the backend keys its present
    /// bookkeeping by. Section 45.1 declares no accessor, and a caller has no
    /// question this would answer — it holds the plan it built.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "read by the plan builder and the present path when section 40 and the \
                      backend port land"
        )
    )]
    pub(crate) fn plan(self) -> SubmissionPlanId {
        self.plan
    }

    /// This presentation's index within its plan.
    ///
    /// Crate-private, and deliberately not a public ordinal: it is how the plan
    /// orders and counts its presentations, not a number a caller addresses one by.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "read by the present path when the backend port lands"
        )
    )]
    pub(crate) fn local(self) -> u32 {
        self.local
    }
}

/// Identity of one present receipt.
///
/// Device-scoped and unique within the device's life, so a receipt is answerable
/// long after the submission that produced it — section 45.5's outcomes are
/// observed later than acceptance, and a caller that polls once and comes back must
/// still get an answer.
///
/// There is no public constructor: a receipt exists because a plan with a present
/// in it was accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PresentReceiptId {
    device: DeviceIdentity,
    serial: u64,
}

impl PresentReceiptId {
    /// Mints the identity of an accepted presentation.
    ///
    /// Crate-private: acceptance is observed by `Device::submit`.
    pub(crate) fn new(device: DeviceIdentity, serial: u64) -> Self {
        Self { device, serial }
    }

    /// The device whose present system this receipt observes.
    ///
    /// `Device::present_state` validates it before answering, exactly as section
    /// 41.1 requires for a completion token: a receipt from another device is a
    /// cross-device mistake rather than a presentation that is still pending.
    pub(crate) fn device_identity(self) -> DeviceIdentity {
        self.device
    }
}

/// Why a presentation failed.
///
/// The structured reason section 45.4 adds so that a failed present is not just a
/// state with no explanation. Deliberately narrow: it carries text and no kind,
/// because the state it appears inside — [`PresentState::Failed`] — is already the
/// classification, and a second one here would give a caller two things to branch
/// on that can disagree.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PresentFailure {
    message: String,
}

impl PresentFailure {
    /// Describes a failure.
    ///
    /// Crate-private: only the code that observed the failure may describe it.
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The diagnostic message. The text is not stable.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// What became of a presentation.
///
/// Five terminal answers and one that is not terminal, and the difference between
/// them is the point:
///
/// ```text
/// Pending      not terminal; keep polling
/// Accepted     ownership moved; NOT scan-out completion
/// Outdated     the surface changed; reconfigure and present again
/// TargetLost   the target is gone; terminal for this receipt
/// DeviceLost   the device is gone; terminal, and everything else is too
/// Failed       the present system refused for another reason
/// ```
///
/// `Accepted` is not `Presented` on purpose. A frame can be accepted and never
/// reach a display — the compositor may drop it, the window may be occluded — and
/// section 45.5 states that GPU completion and present outcome are independent
/// events that a caller must not infer from one another.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum PresentState {
    /// Submitted, and the presentation system has not answered yet.
    Pending,
    /// The presentation system or host lifecycle accepted and consumed frame
    /// ownership.
    ///
    /// This does not mean scan-out or display completion.
    Accepted,
    /// The surface changed underneath the presentation; the frame's configuration
    /// no longer describes it.
    Outdated,
    /// The presentation target is gone.
    TargetLost,
    /// The device is gone, with the reason.
    DeviceLost(DeviceLossInfo),
    /// The present system failed for a reason none of the above describes.
    Failed(PresentFailure),
}

/// The receipt for one presentation in an accepted plan.
///
/// Section 45.5 makes this the second of the two independent outcomes a submit
/// produces: the plan's `CompletionPoint` reports the GPU work, and this reports
/// ownership of the frame. A caller that only waits for one of them has not
/// observed the other.
///
/// Not `Clone`: it is returned by reference from `SubmissionReceipt::presents()`,
/// and whether a caller should be able to hold one independently of that receipt
/// is an open question in the ledger rather than a settled one. Until it is
/// settled, the borrow is the only way to hold one.
pub struct PresentReceipt {
    id: PresentReceiptId,
    plan_id: PresentPlanId,
    /// The shared observation backing. Terminal outcome lives here, so a live
    /// receipt answers `present_state` without the backend retaining an unbounded
    /// present-history map; when the last `PresentReceipt` clone drops, this Arc
    /// is dropped and the outcome is reclaimed.
    backing: std::sync::Arc<PresentOutcome>,
}

/// Shared outcome behind one observed presentation.
///
/// Holds the terminal answer a live `PresentReceipt` can be asked about
/// repeatedly with the same result (section 45.5). Instances are referenced
/// strongly by the `PresentReceipt`(s) that observe them and weakly by the
/// process registry that lets a bare `PresentReceiptId` query reach a live
/// outcome. When the last `PresentReceipt` drops, the strong reference is gone,
/// the registry's `Weak` upgrade fails, and the outcome is reclaimed — so
/// presentation bookkeeping scales with *live* receipts rather than with total
/// presents since device creation.
pub(crate) struct PresentOutcome {
    device: DeviceIdentity,
    state: std::sync::Mutex<PresentState>,
}

impl PresentOutcome {
    pub(crate) fn new(device: DeviceIdentity) -> Self {
        Self {
            device,
            state: std::sync::Mutex::new(PresentState::Pending),
        }
    }

    pub(crate) fn device(&self) -> DeviceIdentity {
        self.device
    }

    /// Publishes a terminal state into this outcome. Called by a backend while
    /// lowering an accepted present for this receipt's id.
    pub(crate) fn set(&self, state: PresentState) {
        *self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = state;
    }

    /// Reads the current published state.
    pub(crate) fn get(&self) -> PresentState {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

/// Process-wide registry from a bare present id to the live outcome it names.
///
/// Weak handles: the outcome is owned by its `PresentReceipt`(s); this table
/// only lets a poll-by-id reach an outcome that is still alive. An entry whose
/// strong refs have all dropped fails to upgrade and is dead weight, pruned on
/// a later insertion or, at worst, harmless until collected. Touched only on
/// submit/present/query, never on a per-frame hot path that runs without a
/// presentation.
static PRESENT_OUTCOMES: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<PresentReceiptId, std::sync::Weak<PresentOutcome>>>,
> = std::sync::LazyLock::new(|| {
    std::sync::Mutex::new(std::collections::HashMap::new())
});

/// Pastes a live outcome into the process registry so a bare id can reach it.
pub(crate) fn register_outcome(
    id: PresentReceiptId,
    outcome: &std::sync::Arc<PresentOutcome>,
) {
    let mut table = PRESENT_OUTCOMES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // Opportunistic reaper: throw out entries whose outcome is already gone so
    // a long-running app that drops many receipts does not accumulate tombstones.
    table.retain(|_, weak| weak.strong_count() > 0);
    table.insert(id, std::sync::Arc::downgrade(outcome));
}

/// Resolves a live outcome for a bare present id, or `None` when the receipt
/// that named it has all been dropped.
pub(crate) fn live_outcome(id: PresentReceiptId) -> Option<std::sync::Arc<PresentOutcome>> {
    PRESENT_OUTCOMES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&id)
        .and_then(std::sync::Weak::upgrade)
}

/// How many entries the present registry currently holds.
///
/// Test-only: the measurable half of the boundedness guarantee. A process that
/// presents and drops many receipts must not accumulate a tombstone per present —
/// and the regression it guards against (a strong per-present history map) keeps
/// every entry forever and reads as a count proportional to total presents. The
/// pruner runs on insertion, so after presenting and dropping a large batch, the
/// next registration folds the whole batch back down to a small working set.
#[cfg(test)]
pub(crate) fn present_registry_entry_count() -> usize {
    PRESENT_OUTCOMES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .len()
}

impl PresentReceipt {
    /// Assembles the receipt for an accepted presentation.
    ///
    /// Crate-private: a receipt exists because a plan was accepted.
    pub(crate) fn new(
        id: PresentReceiptId,
        plan_id: PresentPlanId,
        outcome: std::sync::Arc<PresentOutcome>,
    ) -> Self {
        Self {
            id,
            plan_id,
            backing: outcome,
        }
    }

    /// This receipt's identity, for identity checks and diagnostics.
    pub fn id(&self) -> PresentReceiptId {
        self.id
    }

    /// Which presentation in which plan this receipt is about.
    ///
    /// The link back to the plan's structure: a receipt says *which* frame's
    /// presentation it reports, so a caller presenting several frames in one plan
    /// can tell their outcomes apart.
    pub fn plan_id(&self) -> PresentPlanId {
        self.plan_id
    }

    /// The device this receipt observes, kept here so `present_state` validates
    /// identity without a backend round-trip.
    pub(crate) fn device_identity(&self) -> DeviceIdentity {
        self.backing.device()
    }

    /// The shared outcome this receipt observes.
    pub(crate) fn outcome(&self) -> &std::sync::Arc<PresentOutcome> {
        &self.backing
    }
}

impl core::fmt::Debug for PresentReceipt {
    /// Prints the two portable identities it carries.
    ///
    /// Hand-written rather than derived for the same reason
    /// [`crate::api::presentation::PresentationTarget`] is: this receipt is a
    /// handle onto state the backend owns, and the backend port is expected to add
    /// to it. The fields printed here are the two the specification writes out.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("PresentReceipt")
            .field("id", &self.id)
            .field("plan_id", &self.plan_id)
            .finish_non_exhaustive()
    }
}

impl Device {
    /// What became of one presentation.
    ///
    /// Non-blocking, like [`Device::completion_state`]: a presentation asked about
    /// twice gets the same answer once it is terminal, and progress comes from
    /// host/runtime progress, `Device::poll`, and backend callbacks rather than
    /// from waiting here.
    ///
    /// Cross-device receipts are refused rather than answered: a receipt names work
    /// accepted on one device, and another device has no such presentation.
    ///
    /// The backing outcome is what answers, not the backend: a present that a
    /// backend published into the live outcome keeps answering identically for
    /// as long as any receipt that names it is alive, and stops being answerable
    /// once the last such receipt is dropped (the outcome is reclaimed then).
    pub fn present_state(&self, receipt: PresentReceiptId) -> RhiResult<PresentState> {
        if receipt.device_identity() != self.identity() {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "this present receipt belongs to another device",
            )
            .at("Device::present_state"));
        }
        crate::api::presentation::present::live_outcome(receipt)
            .map(|outcome| outcome.get())
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "this present receipt is no longer live; every holder has been dropped",
                )
                .at("Device::present_state")
            })
    }

    /// Waits for presentation ownership to reach a terminal outcome.
    pub async fn wait_present(&self, receipt: PresentReceiptId) -> RhiResult<PresentState> {
        std::future::poll_fn(|_| {
            if receipt.device_identity() != self.identity() {
                return std::task::Poll::Ready(Err(RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "this present receipt belongs to another device",
                )
                .at("Device::wait_present")));
            }
            match crate::api::presentation::present::live_outcome(receipt) {
                Some(outcome) => match outcome.get() {
                    PresentState::Pending => std::task::Poll::Pending,
                    terminal => std::task::Poll::Ready(Ok(terminal)),
                },
                None => std::task::Poll::Ready(Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "this present receipt is no longer live; every holder has been dropped",
                )
                .at("Device::wait_present"))),
            }
        })
        .await
    }
}
