//! Captured work and the submission/presentation relation (rhi-design sections
//! 56 and 57).
//!
//! # What it is
//!
//! The complete logical relation of one recording and one submission:
//! `CapturedRecordedWork` is a recording's portable commands plus its merged
//! use summary, and `CapturedSubmissionPlan` names every batch, its lane, its
//! work, every dependency edge, and every present — followed by the receipt
//! that reports what actually happened.
//!
//! Counts are deliberately absent. `batch_count`, `dependency_count`, and
//! `present_count` cannot be replayed, so the complete relation is frozen
//! instead: an artifact layer canonicalizes these runtime ids into its own
//! capture-local ids, and it can only do that if it is given the relation.
//!
//! # What it deliberately does not own
//!
//! RHI does not decide which of these edges matter for a capture, which
//! resources must be snapshotted beforehand, or how many frames of a history
//! resource must be included (sections 58.4 and 58.5). The relation is
//! reported; the capture coordinator closes over it.

use super::super::format::{LaneWorkDomains, SubmissionLaneId};
use super::super::platform::{DeviceIdentity, ObjectId};
use super::super::presentation::{AcquiredFrameId, PresentPlanId, PresentReceiptId};
use super::super::submission::{CompletionPoint, PlanPoint, SubmissionPlanId, SubmissionPoint};
use super::commands::{CapturedCommand, CapturedResourceUse};

/// One captured recording.
#[derive(Clone)]
pub struct CapturedRecordedWork {
    /// The recording's object id.
    pub work: ObjectId,
    /// The device identity this recording belongs to.
    pub device: DeviceIdentity,
    /// The work domains the recording actually contains.
    pub domains: LaneWorkDomains,
    /// The recorded commands, in recording order.
    pub commands: Vec<CapturedCommand>,
    /// The recording's merged use of every resource it touched.
    pub merged_use_summary: Vec<CapturedResourceUse>,
}

/// One batch of one submission plan.
#[derive(Clone)]
pub struct CapturedSubmissionBatch {
    /// The batch's point in the plan.
    pub point: PlanPoint,
    /// The logical lane the batch runs on.
    pub lane: SubmissionLaneId,
    /// The work in the batch, in logical order.
    pub work: Vec<ObjectId>,
}

/// What a plan dependency orders a batch after.
///
/// Both sources are needed: an edge from a batch of this plan is interior, and
/// an edge from the completion of an earlier submission is cross-plan. A
/// relation that could express only the first could not describe a plan that
/// waits on previously submitted work.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum CapturedDependencySource {
    /// An earlier batch of the same plan.
    PlanPoint(PlanPoint),
    /// The completion of previously submitted work.
    PriorCompletion(CompletionPoint),
}

/// One happens-before edge of a captured plan.
#[derive(Clone, Copy, Debug)]
pub struct CapturedPlanDependency {
    /// What must happen first.
    pub before: CapturedDependencySource,
    /// The batch that waits.
    pub after: PlanPoint,
}

/// One captured present plan.
#[derive(Clone, Copy, Debug)]
pub struct CapturedPresentPlan {
    /// This present's identity inside the plan.
    pub id: PresentPlanId,
    /// The frame being presented.
    pub frame: AcquiredFrameId,
    /// The point the present follows.
    pub after: PlanPoint,
}

/// The complete logical relation of one submission plan.
///
/// It is captured at acceptance, before any of it has run, because the relation
/// is what a replay must rebuild; a relation recovered from a receipt could
/// only describe work that already finished.
#[derive(Clone)]
pub struct CapturedSubmissionPlan {
    /// The device identity this plan belongs to.
    pub device: DeviceIdentity,
    /// The plan's identity.
    pub plan: SubmissionPlanId,
    /// The batches.
    pub batches: Vec<CapturedSubmissionBatch>,
    /// The plan's happens-before edges, interior and cross-plan.
    pub dependencies: Vec<CapturedPlanDependency>,
    /// The presents the plan performs.
    pub presents: Vec<CapturedPresentPlan>,
}

/// What actually happened after acceptance.
#[derive(Clone)]
pub struct CapturedSubmissionReceipt {
    /// The logical serial at which RHI accepted the plan.
    pub submitted: SubmissionPoint,

    /// Terminal completion of all GPU work in the plan.
    ///
    /// It does not include display scan-out completion.
    pub overall_completion: CompletionPoint,

    /// Per-batch completion, for the batches that have one.
    ///
    /// A backend that cannot provide finer completion repeats the overall token
    /// rather than inventing precision it does not have.
    pub point_completions: Vec<(PlanPoint, CompletionPoint)>,

    /// The present receipt produced for each present plan.
    pub presents: Vec<(PresentPlanId, PresentReceiptId)>,
}
