//! Completion of submitted work (specification section 41).
//!
//! The other half of acceptance: what a caller can learn after a plan was
//! submitted — the two completion levels a plan reports, the terminal states those
//! can reach, and the receipt that carries them. It does not own the plan
//! (section 39), the frames a plan presents ([`crate::api::presentation`] owns the
//! outcome of those), or the retirement bookkeeping that consumes this state
//! (section 41.6, which is an internal RHI property and has no verb here).
//!
//! Invariant, and section 41.3 calls it a critical frozen semantic:
//!
//! ```text
//! Device::submit returning Ok     means the RHI entered "accepted submission"
//! Device::submit returning Err    means no GPU work in the plan was accepted
//! ```
//!
//! There is no third outcome. Once any native work has been accepted, a later
//! failure — a lane that cannot continue, a lost device — is reported *through*
//! the receipt as `CompletionState::Failed` or `DeviceLost`, never as `Err`. The
//! case this forbids is the one where a batch runs on the GPU while the caller is
//! told the whole plan did not execute.
//!
//! ```text
//! accepted  != complete                                       (41.7)
//! no blocking wait in the frame loop                          (41.10)
//! device loss reaches every pending point, and never stays Pending   (41.8)
//! ```

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::DeviceIdentity;
use crate::api::platform::{Device, DeviceLossInfo, DeviceStatus};
use crate::api::presentation::present::PresentReceipt;
use crate::api::submission::plan::{
    CompletionPoint, PlanPoint, SubmissionPlan, SubmissionPlanId, SubmissionPoint,
};

/// Why GPU work terminated without completing.
///
/// The structured reason section 41.1 adds to the completion vocabulary: a state
/// that only said `Failed` would leave asynchronous failure unexplainable, and the
/// reason arrives long after the call that started the work, so it cannot be an
/// error return.
///
/// Carries text and no kind. The state it appears inside is already the
/// classification, and a second one here would give a caller two things to branch
/// on that can disagree.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct CompletionFailure {
    message: String,
}

impl CompletionFailure {
    /// Describes a failure.
    ///
    /// Crate-private: only the code that observed the failure may describe it.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "raised by the completion path when the backend port lands"
        )
    )]
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

/// The state of one completion point.
///
/// Two of these are terminal and two are not, and the difference is what a
/// caller's polling loop branches on:
///
/// ```text
/// Pending      not terminal
/// Complete     terminal: this point's work is done
/// DeviceLost   terminal, with the reason
/// Failed       terminal, with the reason
/// ```
///
/// Section 41.8 requires `Pending` to be temporary: after a device loss every
/// pending point for that device must reach [`Self::DeviceLost`] through bounded
/// polling progress, while points that were already [`Self::Complete`] stay
/// `Complete`.
///
/// Section 41.10 forbids a blocking wait on this state in a frame loop, which is
/// why the accessor is a query rather than a `wait()`. Progress comes from the
/// host's own progress, `Device::poll`, and backend callbacks.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum CompletionState {
    /// Accepted, and not yet terminal.
    Pending,
    /// Every unit of work this point covers is terminal.
    ///
    /// Section 41.4 defines what a point covers: the work it depends on, plus its
    /// own work. On one lane that makes earlier batches' completion happen-before
    /// later ones'; across lanes it guarantees nothing, because different lanes
    /// have no implicit order.
    Complete,
    /// The device was lost. Terminal, with the reason.
    DeviceLost(DeviceLossInfo),
    /// The work terminated without completing. Terminal, with the reason.
    Failed(CompletionFailure),
}

/// What one accepted plan reports about itself.
///
/// Section 41.2 splits completion into two levels, and the split is not a
/// convenience: a readback ticket, a transient allocator's reuse, and a resource
/// retirement must not be forced to await the slowest unrelated batch in the plan.
///
/// ```text
/// submitted()              acceptance, not completion          (41.7)
/// completion()             every batch in the plan
/// completion_for(point)    one batch, as coarse as the backend can be
/// presents()               the presentations this plan carries (45.5)
/// ```
///
/// The two outcomes section 45.5 keeps apart both travel through here:
/// [`Self::completion`] reports GPU work, and [`Self::presents`] reports frame
/// ownership. A caller that observes only one of them has not observed the other.
///
/// Held briefly rather than stored: the receipt is the RHI's statement about a
/// plan it accepted, and section 41.4 requires the RHI — not the caller — to keep
/// enough accepted-work use history to validate the next submission against
/// in-flight work.
pub struct SubmissionReceipt {
    plan: SubmissionPlanId,
    device: DeviceIdentity,
    submitted: SubmissionPoint,
    completion: CompletionPoint,
    points: Vec<(PlanPoint, CompletionPoint)>,
    presents: Vec<PresentReceipt>,
}

impl SubmissionReceipt {
    /// Assembles the receipt for an accepted plan.
    ///
    /// Crate-private: acceptance is something only `Device::submit` observes, and a
    /// caller-built receipt would be a claim about work nobody submitted.
    ///
    /// `points` carries the per-batch completion tokens this plan could provide.
    /// A backend with no finer primitive than "everything submitted so far" passes
    /// an empty list, which is legal and coarse rather than wrong: section 41.2
    /// explicitly permits several plan points to share one completion token, and
    /// [`Self::completion_for`] falls back to the overall token for a point it has
    /// no entry for.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "created by Device::submit when the backend port lands"
        )
    )]
    pub(crate) fn new(
        plan: SubmissionPlanId,
        device: DeviceIdentity,
        submitted: SubmissionPoint,
        completion: CompletionPoint,
        points: Vec<(PlanPoint, CompletionPoint)>,
        presents: Vec<PresentReceipt>,
    ) -> Self {
        Self {
            plan,
            device,
            submitted,
            completion,
            points,
            presents,
        }
    }

    /// The device that accepted the plan.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The logical serial at which the plan was accepted.
    ///
    /// Acceptance only. Section 41.7 forbids reading this as completion, and
    /// forbids treating the serial as a native fence value.
    pub fn submitted(&self) -> SubmissionPoint {
        self.submitted
    }

    /// Terminal completion of all GPU work in the plan.
    ///
    /// Does not include display scan-out completion: a frame's arrival on a display
    /// is [`PresentState`](crate::api::presentation::PresentState)'s business, and
    /// section 45.5 makes the two independent.
    pub fn completion(&self) -> CompletionPoint {
        self.completion
    }

    /// Terminal completion of the work one point covers.
    ///
    /// ```text
    /// point belongs to this plan, with its own token    -> that token
    /// point belongs to this plan, no token recorded    -> the overall token
    /// point belongs to another plan                    -> InvalidUsage
    /// ```
    ///
    /// The second row is section 41.2's "a backend unable to provide finer
    /// completion may return the same token as overall completion". It is coarse
    /// rather than wrong: on a backend with one completion primitive, "everything
    /// submitted as of this call" is a *later* bound than one batch, so waiting on
    /// it is conservative in the safe direction.
    ///
    /// The third row is the same rule as section 39.1's for a foreign `PlanPoint`:
    /// a point the plan never handed out cannot be answered for, and answering with
    /// the overall token would silently report another plan's work as this one's.
    pub fn completion_for(&self, point: PlanPoint) -> RhiResult<CompletionPoint> {
        if point.plan() != self.plan {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "this plan point belongs to a different plan",
            )
            .at("SubmissionReceipt::completion_for"));
        }
        Ok(self
            .points
            .iter()
            .find(|(candidate, _)| *candidate == point)
            .map_or(self.completion, |(_, token)| *token))
    }

    /// The presentations this plan carries, in the order they were added.
    ///
    /// Empty for a plan that presents nothing. Section 45.5 makes each entry the
    /// independent other half of the plan's fate: the work can be complete while a
    /// presentation is outdated, and both can be lost together.
    pub fn presents(&self) -> &[PresentReceipt] {
        &self.presents
    }
}

impl core::fmt::Debug for SubmissionReceipt {
    /// Prints portable identity, not the per-batch token table.
    ///
    /// Hand-written rather than derived, for the reason recorded as adjudication
    /// A16 in the 0.16 plan: the backend port will hold native completion
    /// bookkeeping here, and the interesting part of a receipt in a log is which
    /// plan it is about.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SubmissionReceipt")
            .field("plan", &self.plan)
            .field("device", &self.device)
            .field("submitted", &self.submitted)
            .field("completion", &self.completion)
            .field("points", &self.points.len())
            .field("presents", &self.presents.len())
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Submits a validated plan.
    ///
    /// Section 41.3 splits this into two phases, and the split is the contract:
    ///
    /// ```text
    /// Phase A  preflight, before any native submit
    ///          plan/device identity, lane and work-domain legality, the dependency
    ///          DAG, resource identity, the present relation, recorded-work
    ///          validity, backend lowering prerequisites
    ///          failure -> Err, and no plan work was submitted
    ///
    /// Phase B  native acceptance
    ///          once anything is accepted this may not return an Err that says
    ///          "nothing happened"; later failure becomes a terminal state on the
    ///          corresponding CompletionPoint, PlanPoint, and PresentReceipt
    /// ```
    ///
    /// What is implemented here is the portable part of Phase A: the plan's device
    /// identity, and the device's own liveness. Both are decidable without a
    /// backend, and section 3.1 requires them to be decided before one is touched —
    /// a plan built for another device may not be handed down for a driver to
    /// discover.
    ///
    /// Section 41.4's cross-plan hazard check — this plan against prior submissions
    /// that are not yet terminal — belongs to Phase A as well and is *not* here: it
    /// needs the accepted-work use history the backend keeps, and there is nothing
    /// to check it against until submissions exist.
    ///
    /// If this returns `Err`, the plan is dropped, which is the path section 41.9
    /// requires: every recorded work is released from plan ownership, an unsubmitted
    /// readback ticket goes to `Abandoned`, and a consumed frame takes the no-submit
    /// terminal path — the frame's own `Drop` performs the no-throw abandonment
    /// bookkeeping, so a caller that ignores the error still does not leak a
    /// drawable. No submission happens on that path, by construction.
    pub fn submit(&self, plan: SubmissionPlan) -> RhiResult<SubmissionReceipt> {
        if let DeviceStatus::Lost = self.status() {
            return Err(RhiError::new(
                RhiErrorKind::DeviceLost,
                "this device was lost; nothing can be submitted to it",
            )
            .at("Device::submit"));
        }
        if plan.device_identity() != self.identity() {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "this plan was built for another device",
            )
            .at("Device::submit"));
        }
        unimplemented!(
            "the rest of Phase A and all of Phase B need the backend: lane and \
             work-domain legality, the dependency DAG, the in-flight hazard check, and \
             the native submit; the contract is fixed, none of them is built"
        )
    }

    /// The state of one completion point.
    ///
    /// A non-blocking query, and the only way to observe completion: section 41.10
    /// provides no `wait()`, so a frame loop polls this alongside `Device::poll`,
    /// and `wait_idle` stays reserved for shutdown, recovery, and diagnostics.
    ///
    /// The point's device is validated first, as section 41.1 requires — and the
    /// device's own loss before it, because loss is terminal for the whole identity
    /// (section 3.1) and a foreign token on a lost device would otherwise send a
    /// caller looking for the wrong problem.
    ///
    /// On a lost device this returns [`RhiErrorKind::DeviceLost`] rather than
    /// `Ok(CompletionState::DeviceLost(..))`. Section 41.8 defines the per-point
    /// outcome — pending points reach `DeviceLost`, already-complete points stay
    /// `Complete` — and *that* answer needs to know which point it is about. The
    /// portable layer holds no per-point state, so it declines to guess one: it
    /// reports the loss, and the point's own state is what the backend's
    /// bookkeeping answers with once the port lands.
    ///
    /// Panics until that port exists.
    pub fn completion_state(&self, point: CompletionPoint) -> RhiResult<CompletionState> {
        if let DeviceStatus::Lost = self.status() {
            return Err(RhiError::new(
                RhiErrorKind::DeviceLost,
                "this device was lost; its completion points cannot be queried",
            )
            .at("Device::completion_state"));
        }
        if point.device_identity() != self.identity() {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "this completion point belongs to another device",
            )
            .at("Device::completion_state"));
        }
        unimplemented!(
            "completion state is tracked by the backend as it observes native work; the \
             contract is fixed, the bookkeeping is not built"
        )
    }
}
