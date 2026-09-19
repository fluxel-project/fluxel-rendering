//! Submission plan identity, and the validated plan itself (specification
//! sections 39 and 40).
//!
//! What this file owns: the identity of a plan, the identity of a batch inside it,
//! the point that names a batch, the two tokens a submitted plan hands back —
//! acceptance ([`SubmissionPoint`]) and terminal GPU completion
//! ([`CompletionPoint`]) — and the validated body of a plan, which is what
//! `Device::submit` lowers. What it does not own: *building* a plan
//! ([`crate::api::submission::SubmissionPlanBuilder`] and section 40's rules live
//! in `builder.rs`), recorded work (module 04's `RecordedWork`, which a batch
//! carries), the receipt and completion state of a submitted plan (section 41, in
//! `completion.rs`), and any native submission machinery — section 39's opening
//! list forbids a queue, fence, semaphore, event, or timeline value from appearing
//! in this surface at all.
//!
//! Invariant: a plan identity is minted by the builder that owns it and names the
//! device it will be submitted to, so a `PlanPoint` from one builder can never be
//! accepted by another (section 39.1) and no caller can forge one — the fields are
//! private, there is no public constructor, and batch numbering is not something a
//! caller can start at zero to slip past the check.

use crate::api::command::RecordedWork;
use crate::api::identity::DeviceIdentity;
use crate::api::presentation::{AcquiredFrame, AcquiredFrameId, PresentPlanId};
use crate::api::submission::SubmissionLaneId;

/// Identity of one submission plan.
///
/// Device-scoped: the serial is unique within the life of the device that minted
/// it, and the pair is what a `PlanPoint`, a present plan, and a receipt all agree
/// on. Section 39.1 gives every builder one at creation, which is why an empty
/// plan still has an identity.
///
/// There is no public constructor and no public accessor for the serial: this is
/// evidence that two points came from the same plan, not an ordinal. A caller that
/// logs a plan logs the whole token, and [`crate::api::presentation::PresentPlanId`]
/// is the only thing in the chapter that reads its fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubmissionPlanId {
    device: DeviceIdentity,
    serial: u64,
}

impl SubmissionPlanId {
    /// Mints the identity of a plan being built.
    ///
    /// Crate-private: the serial is unique per device, so only the code that hands
    /// out plan numbers may mint one. That code is the device's, not the builder's
    /// — a plan serial has to be unique across every builder on one device for
    /// section 39.1's "a `PlanPoint` from another plan is `InvalidUsage`" rule to
    /// hold — so [`crate::api::submission::SubmissionPlanBuilder::new`] receives
    /// one rather than computing one.
    pub(crate) fn new(device: DeviceIdentity, serial: u64) -> Self {
        Self { device, serial }
    }

    /// The device this plan will be submitted to.
    ///
    /// Read by the cross-plan checks of sections 40.3 and 41.4 rather than by a
    /// caller: a plan is bound to one device for its whole life, and a point from
    /// a plan on another device is a cross-device mistake even when the serials
    /// coincide.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "read by the cross-plan checks of 40.3 and 41.4 when the submit path lands"
        )
    )]
    pub(crate) fn device_identity(self) -> DeviceIdentity {
        self.device
    }
}

/// Identity of one batch inside one plan.
///
/// Section 39.1 makes this a plain `u32` counter behind a private field, and the
/// privacy is the point: the builder assigns batch numbers in insertion order, and
/// a caller that could write `SubmissionBatchId(0)` could address a batch it never
/// added — the invariant section 39.1 says batch numbering must not be bypassable
/// by forging. It is a distinct type from [`crate::api::presentation::PresentPlanId`]'s
/// `local` for the same reason it is distinct from a bare `u32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubmissionBatchId(u32);

impl SubmissionBatchId {
    /// Names a batch during plan building.
    ///
    /// Crate-private: batch numbers belong to the builder that assigned them.
    pub(crate) fn new(value: u32) -> Self {
        Self(value)
    }
}

/// One batch's position in one plan.
///
/// The pair is what every logical point in this chapter reduces to: a dependency
/// edge, a present point, and a per-batch completion query all name a batch inside
/// a plan, and the plan identity in the pair is what stops a point from one builder
/// being used in another — section 39.1 requires
/// [`crate::api::error::RhiErrorKind::InvalidUsage`] for exactly that mistake, and
/// the private plan field is what makes the check possible at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlanPoint {
    plan: SubmissionPlanId,
    batch: SubmissionBatchId,
}

impl PlanPoint {
    /// Names a batch inside a plan.
    ///
    /// Crate-private: only the builder that assigned the batch number and owns the
    /// plan identity may produce a point for it.
    pub(crate) fn new(plan: SubmissionPlanId, batch: SubmissionBatchId) -> Self {
        Self { plan, batch }
    }

    /// Which batch this point names.
    ///
    /// The batch, and not the plan: a caller that already holds a point uses it to
    /// talk about *that batch's* work, and the plan half is what the RHI compares.
    pub fn batch(self) -> SubmissionBatchId {
        self.batch
    }

    /// The plan this point belongs to.
    ///
    /// Crate-private and not part of section 39.1's public surface: it exists for
    /// the "point belongs to this plan" checks of sections 40.5 and 45.3, which
    /// the RHI performs rather than the caller.
    pub(crate) fn plan(self) -> SubmissionPlanId {
        self.plan
    }
}

/// The logical serial at which the RHI accepted a plan.
///
/// Section 41.7 separates this from [`CompletionPoint`] and forbids the two
/// equivalences that would collapse them:
///
/// ```text
/// submit() returning           does NOT mean the GPU is complete
/// a SubmissionPoint serial     is NOT a native fence value
/// ```
///
/// So this token answers "when was this accepted", never "has it finished". It is
/// the anchor of section 41.4's cross-plan hazard check, where a prior plan that is
/// still in flight is exactly a prior submission whose completion is not terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubmissionPoint {
    device: DeviceIdentity,
    serial: u64,
}

impl SubmissionPoint {
    /// Mints the acceptance serial of a plan.
    ///
    /// Crate-private: acceptance is something `Device::submit` observes, and the
    /// serial is the order in which that happened.
    pub(crate) fn new(device: DeviceIdentity, serial: u64) -> Self {
        Self { device, serial }
    }

    /// The device that accepted the plan.
    ///
    /// A completion query is answered by the device, so a token from another device
    /// is a cross-device mistake and not an empty query.
    pub fn device_identity(self) -> DeviceIdentity {
        self.device
    }
}

/// Fluxel's terminal-completion token for GPU work.
///
/// The logical observation point that section 41.1's
/// `Device::completion_state` answers about, and that section 41.5 binds a
/// readback ticket to. Two properties are load-bearing:
///
/// ```text
/// it is not a native fence or timeline value
/// distinct logical points may map to one native primitive
/// ```
///
/// The second is what lets a backend with a single completion primitive — a
/// WebGPU queue's `onSubmittedWorkDone`, say — hand several plan points the same
/// token. That is coarse but correct, and section 41.2 explicitly permits it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompletionPoint {
    device: DeviceIdentity,
    serial: u64,
}

impl CompletionPoint {
    /// Mints a completion token for a submitted plan or batch.
    ///
    /// Crate-private: a completion token exists because work was accepted, and the
    /// RHI is the only entity that knows when that happened.
    pub(crate) fn new(device: DeviceIdentity, serial: u64) -> Self {
        Self { device, serial }
    }

    /// The device whose work this token observes.
    ///
    /// Section 41.1 requires `Device::completion_state` to validate this before
    /// answering: a token from another device's plan is a cross-device mistake, and
    /// the alternative — answering `Pending` for work that does not exist — would
    /// wait forever.
    pub fn device_identity(self) -> DeviceIdentity {
        self.device
    }

    /// The backend-local serial this token names.
    ///
    /// Crate-private, and it is the one reader that makes the token useful: a
    /// backend reports completion by *its* serial
    /// ([`crate::base::command::SubmissionOutcome`]), the portable layer wraps
    /// that serial into this token, and asking the backend about the work again
    /// means handing the serial back. Nothing on the public surface exposes it,
    /// because section 41.7 forbids reading a completion token as a native fence
    /// value and a public accessor would invite exactly that.
    pub(crate) fn serial(self) -> u64 {
        self.serial
    }
}

/// One batch of a plan.
///
/// The unit section 40.1 orders: work added to one lane, in the order the caller
/// put it in the `Vec`, which is that batch's logical work order. Held by the
/// built plan rather than by the builder, because lowering it is what
/// `Device::submit` does.
pub(crate) struct PlanBatch {
    /// The point that names this batch, and the only way a caller refers to it.
    pub(crate) point: PlanPoint,
    /// The lane whose ordered execution domain it was added to.
    pub(crate) lane: SubmissionLaneId,
    /// The recorded work, in its logical order. Never empty: section 40.1 refuses
    /// an empty batch at insertion.
    pub(crate) work: Vec<RecordedWork>,
}

/// One presentation a plan carries.
///
/// The frame itself is owned by the plan (see `PlanBody::frames`), so this record
/// keeps only what the closure rules of section 45.3 and the receipt's
/// present list need: which frame, which plan-local identity names it, and which
/// batch it is presented after.
pub(crate) struct PlanPresent {
    /// The plan-local present identity returned to the caller.
    ///
    /// Written when the present is planned and read when the receipt is assembled, so
    /// the backend port is what reads it; the test suite reads it to check that the
    /// identity a caller was handed is the identity the plan carries.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "read by the receipt path when the backend port lands"
        )
    )]
    pub(crate) id: PresentPlanId,
    /// The frame being presented, which the plan owns.
    pub(crate) frame: AcquiredFrameId,
    /// The point the presentation is ordered after.
    pub(crate) after: PlanPoint,
}

/// Everything a validated plan carries into `Device::submit`.
///
/// Grouped into one value so that [`SubmissionPlan::new`] stays a three-argument
/// constructor rather than a seven-argument one, and so that the plan's contents
/// have one name to document them.
pub(crate) struct PlanBody {
    /// The batches, in insertion order.
    pub(crate) batches: Vec<PlanBatch>,
    /// Explicit happens-before edges between batches of this plan.
    pub(crate) dependencies: Vec<(PlanPoint, PlanPoint)>,
    /// Happens-before edges from earlier submitted work into this plan.
    pub(crate) external_dependencies: Vec<(CompletionPoint, PlanPoint)>,
    /// The presentations this plan carries.
    pub(crate) presents: Vec<PlanPresent>,
    /// The frames `present_after` consumed, owned until `Device::submit` accepts
    /// the plan (section 41.9).
    pub(crate) frames: Vec<AcquiredFrame>,
}

/// A validated submission plan, ready for `Device::submit`.
///
/// Section 40 keeps this opaque, and opaque is the whole design: everything a
/// caller may know about a plan it built is [`Self::id`] and
/// [`Self::device_identity`], because the validation that produced it is the RHI's
/// and the recorded work is no longer the caller's. A plan that exists is a plan
/// that passed section 40.5's checklist, including the hazard analysis of section
/// 40.4 — which is why there is no `validate()` a caller could forget to call, and
/// why dropping one is legal and submits nothing (section 41.9): the frames it
/// owns are dropped with it, and each frame's own `Drop` performs the no-submit
/// abandonment bookkeeping.
pub struct SubmissionPlan {
    id: SubmissionPlanId,
    device: DeviceIdentity,
    body: PlanBody,
}

impl SubmissionPlan {
    /// Takes ownership of a validated plan body.
    ///
    /// Crate-private: only `SubmissionPlanBuilder::build` may produce a plan,
    /// because a plan is by definition the output of section 40.5's validation.
    ///
    /// No dead-code annotation is needed: `build` calls it in this crate.
    pub(crate) fn new(id: SubmissionPlanId, device: DeviceIdentity, body: PlanBody) -> Self {
        Self { id, device, body }
    }

    /// This plan's identity.
    pub fn id(&self) -> SubmissionPlanId {
        self.id
    }

    /// The device this plan will be submitted to.
    ///
    /// `Device::submit` compares it against its own identity before anything else,
    /// which is section 40.5's "all object / work DeviceIdentity and liveness
    /// valid" at the plan level.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The batches this plan will submit.
    ///
    /// Crate-private: section 40 describes a plan as opaque, and these are the
    /// facts the backend lowering reads rather than anything a caller asks about.
    pub(crate) fn batches(&self) -> &[PlanBatch] {
        &self.body.batches
    }

    /// The explicit batch-to-batch dependencies this plan carries.
    pub(crate) fn dependencies(&self) -> &[(PlanPoint, PlanPoint)] {
        &self.body.dependencies
    }

    /// The dependencies from earlier submitted work into this plan.
    ///
    /// Read by `Device::submit` only to hand the edges to the backend, which is
    /// the only side that can resolve a serial into something it can wait on. Each
    /// token's device half is *not* checked there: the builder already refused a
    /// foreign one, so a plan that exists cannot carry one.
    pub(crate) fn external_dependencies(&self) -> &[(CompletionPoint, PlanPoint)] {
        &self.body.external_dependencies
    }

    /// The presentations this plan carries.
    ///
    /// Read by `Device::submit` to refuse a plan that carries one: no backend
    /// lowers presentation yet, and executing the work while dropping the frame
    /// would be the silent substitution discipline 3 forbids.
    pub(crate) fn presents(&self) -> &[PlanPresent] {
        &self.body.presents
    }

    /// The frames this plan owns, which are presented by it.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "read by the backend lowering and the present path when the port lands"
        )
    )]
    pub(crate) fn frames(&self) -> &[AcquiredFrame] {
        &self.body.frames
    }
}

impl core::fmt::Debug for SubmissionPlan {
    /// Prints portable identity and the size of the plan, not the recorded work.
    ///
    /// Hand-written rather than derived, for the reason recorded as adjudication
    /// A16 in the 0.16 plan: the body this plan carries is a backend lowering with
    /// native queues and synchronization in it, and a log should show which plan
    /// it was rather than what the driver was handed.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SubmissionPlan")
            .field("id", &self.id)
            .field("device", &self.device)
            .field("batches", &self.body.batches.len())
            .field("dependencies", &self.body.dependencies.len())
            .field(
                "external_dependencies",
                &self.body.external_dependencies.len(),
            )
            .field("presents", &self.body.presents.len())
            .field("frames", &self.body.frames.len())
            .finish_non_exhaustive()
    }
}
