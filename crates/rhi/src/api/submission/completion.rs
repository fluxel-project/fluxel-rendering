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

use crate::api::command::record::RecordedPayload;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::DeviceIdentity;
use crate::api::platform::{Device, DeviceLossInfo, DeviceStatus};
use crate::api::presentation::present::PresentReceipt;
use crate::api::resource::transfer::ReadbackStatus;
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
    ///
    /// Every backend raises these — the mock answers a serial it never reported
    /// with one, and a native backend answers post-commit trouble with one — but
    /// *only* a backend does, so a build with no backend compiled has no caller
    /// for this at all. That is a real configuration rather than a hypothetical
    /// one: `--no-default-features` compiles no native backend and no test mock.
    ///
    /// The feature list below is every backend feature the crate declares, not
    /// just the one that is implemented, so that adding a real Vulkan or WebGPU
    /// backend does not silently leave this expectation unfulfilled. A new backend
    /// feature goes on the list (rule 4.6: a matrix that is missing a row gets the
    /// row, not a patch to the code that needed it).
    #[cfg_attr(
        not(any(
            test,
            // The backend features that actually compile a lowering. A feature
            // that selects nothing must not appear here: it would remove this
            // expectation in a configuration where the item really is dead, and
            // the gate would then be silent about it. When Vulkan lands and starts
            // calling this, its feature joins the list — which is rule 4.6's
            // "the matrix gets the row" applied to the attribute itself.
            feature = "dx12"
        )),
        expect(
            dead_code,
            reason = "raised by whichever backend observed the failure; a build with no \
                      backend compiled has nothing that could observe one"
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
    pub async fn submit(&self, plan: SubmissionPlan) -> RhiResult<SubmissionReceipt> {
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

        // The rest of Phase A. Everything below is decided *before* the backend is
        // asked to commit, because section 41.3's invariant is that an `Err` from
        // this function proves no native work was accepted — and a check that ran
        // after the commit could not make that promise.
        //
        // Lane and work-domain legality, the dependency DAG, and the self-plan
        // hazard analysis are already done: `SubmissionPlanBuilder::build` runs
        // section 40.5's `validate_plan_graph` and refuses to produce a plan that
        // fails it. Re-running it here would be a second authority for one rule
        // (section 65.3), and a plan that exists is a plan that passed.
        //
        // What is left is the part `build` could not decide because it is about
        // *this* moment rather than about the plan: the present relation, and the
        // external dependencies that point at work already in flight.
        if !plan.presents().is_empty() {
            // Refused rather than dropped, and before the backend is reached.
            // Section 45.5 makes a presentation the independent other half of a
            // plan's fate, so a plan carrying one but lowered without it would
            // hand back a receipt whose `presents()` claimed an outcome for a
            // frame nothing ever touched — discipline 3, in the one direction
            // where the substitute is a *missing* effect rather than a different
            // one.
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this plan carries a presentation, and no backend lowering for \
                 presentation is built: submitting it would execute the work while \
                 silently dropping the frame. Retry with a plan that presents nothing",
            )
            .at("Device::submit"));
        }

        // Section 40.3's external dependencies are *not* re-checked here, and the
        // absence is deliberate rather than an omission. Every rule about one —
        // that the point belongs to this plan, and that the completion token is
        // this device's — already ran in
        // `SubmissionPlanBuilder::add_external_dependency`, and a plan only ever
        // comes from `SubmissionPlanBuilder::build`. Re-deciding them here would
        // be a second authority for one rule (section 65.3) *and* would describe a
        // reachable failure the code cannot actually reach, which is worse than
        // saying nothing. What is genuinely left to this moment is routability:
        // whether this backend can make the edge happen, which only the backend's
        // own completion bookkeeping can answer, and which it answers by
        // refusing the lowering rather than by failing this call.

        // Phase B. Everything from here on is the backend's, and section 41.3's
        // other half starts applying: once native work is accepted this function
        // may not return an `Err` that tells the caller nothing happened.
        //
        // The borrow of `plan` ends with this call, which is what lets the plan be
        // dropped on the way out — section 41.9's release path, and the reason
        // nothing below reads it again.
        let outcome = {
            let request = crate::base::command::SubmissionRequest {
                plan: plan.id(),
                batches: plan.batches(),
                dependencies: plan.dependencies(),
                external_dependencies: plan.external_dependencies(),
            };
            self.native().submit(&request)?
        };

        // Infallible from here. The tokens are wrapped, not derived: the serials
        // are the backend's numbers and the device half is this layer's, which is
        // what keeps section 3.1's identity rule on this side of the seam.
        let identity = self.identity();
        let overall = CompletionPoint::new(identity, outcome.completion);
        // Annotated rather than inferred: the walk below reads this before the
        // receipt takes it, so there is no longer a single consumer for the
        // element type to be inferred from.
        let points: Vec<(PlanPoint, CompletionPoint)> = outcome
            .points
            .into_iter()
            .map(|(point, serial)| (point, CompletionPoint::new(identity, serial)))
            .collect();
        let submitted = SubmissionPoint::new(identity, self.serials().next_submission());

        // Section 41.5, and the second half of section 41.3's Phase B. Every
        // readback ticket this plan carried is now bound to the point of *this*
        // submission that covers it, and moved out of `NotSubmitted` — a ticket
        // whose work has been accepted may not report that it never was.
        //
        // The walk lives here rather than in each backend, and the reason is
        // section 3.1: what a backend hands back is a *serial*, and turning a
        // serial into the token a caller holds is exactly the identity work this
        // layer owns. A backend that did it would need the plan's batches, the
        // point mapping, and the device identity — three portable facts — to
        // reproduce one rule, which is the second authority section 65.3 forbids.
        // It would also mean the mock, whose device has no work to do, had to
        // repeat the walk to keep the two backends answering alike.
        //
        // A batch the backend reported no point for falls back to the overall
        // token, which section 41.2 permits a coarse backend to make the same
        // value as every batch's. Nothing is *skipped*: the fallback is still a
        // point of this submission, and a ticket bound to a coarser answer is
        // bound correctly.
        for batch in plan.batches() {
            let point = points
                .iter()
                .find(|(planned, _)| *planned == batch.point)
                .map(|(_, completion)| *completion)
                .unwrap_or(overall);
            for work in &batch.work {
                for command in work.commands() {
                    let RecordedPayload::Readback(ticket) = &command.payload else {
                        continue;
                    };
                    // The order is `set_completion`'s own contract: the point
                    // first, so a caller that observes `Pending` also observes
                    // the point that covers it.
                    ticket.set_completion(point);
                    ticket.set_status(ReadbackStatus::Pending);
                }
            }
        }

        Ok(SubmissionReceipt::new(
            plan.id(),
            identity,
            submitted,
            overall,
            points,
            // Empty by construction: a plan carrying a presentation was refused
            // above, so there is no receipt to build for one here.
            Vec::new(),
        ))
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
    /// The port landed with the Direct3D 12 command spine, which answers per
    /// serial out of its fence. The lost-device answer above is therefore still
    /// the portable layer's — a *status* check ahead of the port — and that
    /// ordering is a known gap rather than a settled rule: section 41.8 splits the
    /// two cases (pending points reach `DeviceLost`, already-complete points stay
    /// `Complete`), and telling them apart needs the per-serial bookkeeping a
    /// loss-and-recovery unit will own. Until it exists, a device whose status is
    /// still `Active` but whose fence has stopped advancing answers from the
    /// backend only.
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

        // The backend answers by its own serial. Section 41.1 makes this
        // non-blocking, so nothing below waits: a backend advances its
        // bookkeeping from `Device::poll` and reports what it has observed.
        Ok(self.native().completion(point.serial()))
    }

    /// Waits until one completion point reaches a terminal state.
    ///
    /// This is intentionally distinct from [`Self::completion_state`]: the latter
    /// is the synchronous, non-blocking observation; this verb owns the potentially
    /// suspending completion wait.
    pub async fn wait_completion(&self, point: CompletionPoint) -> RhiResult<CompletionState> {
        // The portable identity and loss checks are shared with the non-blocking
        // query. A backend-specific waiter will replace this single observation;
        // keeping the public boundary async now prevents a later API split.
        let state = self.completion_state(point)?;
        match state {
            CompletionState::Pending => unimplemented!(
                "waiting for GPU completion requires backend async completion plumbing"
            ),
            terminal => Ok(terminal),
        }
    }
}
