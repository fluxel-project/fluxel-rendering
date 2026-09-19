//! Specification section 57: the submission and presentation tooling IR.
//!
//! One responsibility: **the complete logical relation a submission established,
//! stated as evidence rather than as counts.** Section 57 exists because an
//! earlier draft kept only `work_count`, `dependency_count`, and `present_count`,
//! and three integers cannot be replayed: they say how much happened and nothing
//! about what depended on what, which lane a batch ran on, or which work items a
//! batch contained.
//!
//! Not owned here: the live plan and its builder (`api::submission`, module 05),
//! the live receipt (`api::submission::SubmissionReceipt`), and the present
//! system (`api::presentation`). Every type below is a *record of* those, keyed
//! by the tokens they mint rather than by an [`ObjectId`], because a plan point
//! and a receipt are evidence of an event rather than objects in the device's
//! inventory — see the module-07 audit note on section 52.1 in [`super`].
//!
//! # Why the record can be complete
//!
//! The captured types hold [`PlanPoint`], [`SubmissionPlanId`], [`CompletionPoint`],
//! [`PresentPlanId`], [`PresentReceiptId`], and [`AcquiredFrameId`] — all of them
//! device-scoped tokens that are `Copy` and carry their own
//! [`DeviceIdentity`]. So a record built while the device was alive can be read
//! after the device is gone without resolving anything: the identity is inside
//! the token. That is what makes the ReplayRuntime's job (section 58.6)
//! mechanical rather than archaeological, and it is why this file adds no
//! capture-local ID of its own — section 57 assigns the runtime-to-capture-local
//! ID mapping to the Artifact Layer, deliberately not to RHI.
//!
//! # What a record of a plan must not become
//!
//! These are records, not handles. Nothing here can be submitted, awaited, or
//! asked for state: [`CapturedSubmissionReceipt`] names the receipt a live
//! `SubmissionReceipt` was minted for, and the live receipt is what
//! `Device::completion_state` answers about. A record that could be polled would
//! be a second submission system.

use crate::api::identity::{DeviceIdentity, ObjectId};
use crate::api::presentation::{AcquiredFrameId, PresentPlanId, PresentReceiptId};
use crate::api::submission::{
    CompletionPoint, PlanPoint, SubmissionLaneId, SubmissionPlanId, SubmissionPoint,
};

/// One batch of one captured plan.
///
/// The three fields are exactly the three things a count cannot state: which
/// point in the plan it is, which lane it runs on, and which recorded work items
/// it carries. `work` is a `Vec<ObjectId>` in the order the batch holds them,
/// which for a lane is the order the work is submitted in.
///
/// `lane` is a [`SubmissionLaneId`] and not a
/// [`crate::api::submission::SubmissionLaneClass`]: the class says what kind of
/// work a lane takes, and the id says which lane. A record that kept only the
/// class would not distinguish two raster lanes, which is what a multi-queue
/// backend's replay has to do.
#[derive(Clone)]
pub struct CapturedSubmissionBatch {
    /// This batch's position in the plan.
    pub point: PlanPoint,

    /// The lane this batch runs on.
    pub lane: SubmissionLaneId,

    /// The recorded work this batch carries, in submission order.
    pub work: Vec<ObjectId>,
}

/// Where a captured dependency edge starts.
///
/// Two sources, and the split is section 40's: an edge either orders one batch
/// of the same plan after another, or orders it after completion of work that a
/// *previous* plan already submitted. Collapsing the two into "before: PlanPoint |
/// CompletionPoint" would be the same two arms; keeping them named is what makes
/// a reader see that a cross-plan edge is a different claim about the GPU than an
/// intra-plan one, and it is the claim section 41.4 makes the RHI track in-flight
/// work to validate.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum CapturedDependencySource {
    /// Another batch of the same plan.
    PlanPoint(PlanPoint),

    /// Completion of a point from an earlier submission.
    PriorCompletion(CompletionPoint),
}

/// One captured dependency edge.
///
/// The type is a pair rather than a graph: `after` depends on `before`, and the
/// closure is the reader's to compute. RHI does not compute it — section 58.4
/// assigns capture dependency closure to RenderGraph and the Capture Coordinator
/// — so a record that shipped a precomputed closure would be answering a question
/// this chapter says RHI does not answer.
///
/// `Copy` because a pair of tokens is, and section 57 declares it so.
#[derive(Clone, Copy, Debug)]
pub struct CapturedPlanDependency {
    /// The point that must complete first.
    pub before: CapturedDependencySource,

    /// The point that waits for it.
    pub after: PlanPoint,
}

/// One captured presentation.
///
/// `frame` is the acquired frame being presented and `after` is the batch whose
/// completion the presentation was scheduled after, which is the `PresentPlan`
/// shape section 52.6 names: a present is a point in the plan, not a queue of its
/// own.
///
/// `id` is the plan-local [`PresentPlanId`] the live builder minted, kept so that
/// [`CapturedSubmissionReceipt`]'s `presents` pairs can be read against the
/// presents recorded here. Without it the receipt would name presentations the
/// plan record does not contain.
#[derive(Clone, Copy, Debug)]
pub struct CapturedPresentPlan {
    /// The presentation's plan-local identity.
    pub id: PresentPlanId,

    /// The acquired frame being presented.
    pub frame: AcquiredFrameId,

    /// The batch whose completion this presentation follows.
    pub after: PlanPoint,
}

/// A captured submission plan: the whole logical relation.
///
/// The five fields are section 57's replacement for the three counts, and each
/// one is load-bearing for a different replay decision:
///
/// ```text
/// batches        what ran, on which lane, in which order
/// dependencies   what had to finish before what
/// presents       which frame went to which target, after which batch
/// ```
///
/// `plan` is the [`SubmissionPlanId`] the live builder minted, so a reader can
/// tie the record to the [`CapturedSubmissionReceipt`] that followed it — the
/// receipt carries the same plan through its points, and the pair is what makes
/// "this plan was submitted and here is what happened" one statement rather than
/// two records a reader has to correlate by position.
#[derive(Clone)]
pub struct CapturedSubmissionPlan {
    /// The device the plan was built for.
    pub device: DeviceIdentity,

    /// The plan's identity.
    pub plan: SubmissionPlanId,

    /// The batches, in plan order.
    pub batches: Vec<CapturedSubmissionBatch>,

    /// The dependency edges.
    pub dependencies: Vec<CapturedPlanDependency>,

    /// The presentations the plan carries.
    pub presents: Vec<CapturedPresentPlan>,
}

/// What an accepted plan reported.
///
/// Section 41.2's two-level completion, recorded faithfully:
/// `overall_completion` is the point covering every batch in the plan, and
/// `point_completions` is one point per batch. Recording both is not redundancy —
/// section 41.2 exists precisely so that a readback or a retirement is not forced
/// to await the slowest unrelated batch, so a record that kept only the overall
/// point would describe an RHI with coarser completion than the one that ran.
///
/// `presents` pairs each plan-local [`PresentPlanId`] with the
/// [`PresentReceiptId`] the submission minted for it. The pair is what makes a
/// present's later outcome reachable: the receipt id is what
/// `Device::present_state` answers about, and the plan id is what ties it back to
/// the batch it followed.
///
/// This is a record of an accepted submission, so it is the *only* thing in this
/// file that could not have been built before acceptance: sections 56 and 57 both
/// carry it, and section 58.1 delivers it with the plan it belongs to.
#[derive(Clone)]
pub struct CapturedSubmissionReceipt {
    /// The serial at which the RHI accepted the plan.
    pub submitted: SubmissionPoint,

    /// The point covering every batch in the plan.
    pub overall_completion: CompletionPoint,

    /// One completion point per batch, in plan order.
    pub point_completions: Vec<(PlanPoint, CompletionPoint)>,

    /// One receipt per presentation, in plan order.
    pub presents: Vec<(PresentPlanId, PresentReceiptId)>,
}
