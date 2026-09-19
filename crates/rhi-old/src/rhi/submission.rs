//! Submission identity, acceptance, and completion.
//!
//! This module owns rhi-design sections 39 and 41.
//!
//! # What it is
//!
//! Two tokens with two different meanings, deliberately not interchangeable:
//!
//! ```text
//! SubmissionPoint  = the logical serial at which RHI accepted the plan
//! CompletionPoint  = the token that observes GPU work becoming terminal
//! ```
//!
//! "`submit()` returned" never means "the GPU is done", and a submission serial
//! is never a native fence value. Distinct logical completion points may map to
//! the same native primitive; one backend with a single queue-completion
//! primitive conservatively shares a completion across several plan points,
//! which is correct but less precise.
//!
//! # What it deliberately does not own
//!
//! P0 exposes no fence, semaphore, event, timeline, queue-family index, native
//! command queue, or host-wait lane dependency. A host wait would turn a GPU
//! execution plan into CPU orchestration policy and would require blocking,
//! a runtime, or background workers; if a real caller needs "GPU work A, then a
//! CPU callback, then GPU work B", that belongs to a higher-level continuation,
//! not to a pretend GPU lane dependency.
//!
//! # Acceptance contract
//!
//! [`super::Device::submit`] may return an error only when it can guarantee that
//! no GPU work in the plan was accepted by a native backend. Once any native
//! queue work is accepted it must return a receipt, and move the affected
//! completion, plan point, and present receipt to a terminal `Failed` or
//! `DeviceLost` state instead. Otherwise a caller whose first batch is already
//! running would be told that the entire plan did not execute.

use std::sync::Arc;

mod plan;

pub use plan::{PlannedBatch, PlannedPresent, SubmissionPlan, SubmissionPlanBuilder};

use super::platform::{DeviceIdentity, DeviceLossInfo, ObjectId, RhiResult};
use super::presentation::PresentReceipt;

/// The identity of one submission plan.
///
/// It is device-scoped and has no public constructor, so a plan point from one
/// builder can never be used to forge an invariant in another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmissionPlanId {
    device: DeviceIdentity,
    serial: u64,
}

impl SubmissionPlanId {
    pub(crate) fn new(device: DeviceIdentity, serial: u64) -> Self {
        Self { device, serial }
    }

    /// The device identity this plan belongs to.
    pub fn device_identity(self) -> DeviceIdentity {
        self.device
    }

    /// The plan serial within that identity.
    pub fn serial(self) -> u64 {
        self.serial
    }
}

/// The identity of one batch inside a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmissionBatchId(u32);

impl SubmissionBatchId {
    pub(crate) fn new(value: u32) -> Self {
        Self(value)
    }

    /// The underlying value.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// A reference to one batch of one plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlanPoint {
    plan: SubmissionPlanId,
    batch: SubmissionBatchId,
}

impl PlanPoint {
    pub(crate) fn new(plan: SubmissionPlanId, batch: SubmissionBatchId) -> Self {
        Self { plan, batch }
    }

    /// The batch this point names.
    pub fn batch(self) -> SubmissionBatchId {
        self.batch
    }

    /// The plan this point belongs to.
    pub fn plan(self) -> SubmissionPlanId {
        self.plan
    }
}

/// The logical serial at which RHI accepted a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmissionPoint {
    device: DeviceIdentity,
    serial: u64,
}

impl SubmissionPoint {
    pub(crate) fn new(device: DeviceIdentity, serial: u64) -> Self {
        Self { device, serial }
    }

    /// The device identity this acceptance belongs to.
    pub fn device_identity(self) -> DeviceIdentity {
        self.device
    }

    /// The acceptance serial within that identity.
    pub fn serial(self) -> u64 {
        self.serial
    }
}

/// The terminal-observation token for a piece of GPU work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompletionPoint {
    device: DeviceIdentity,
    serial: u64,
}

impl CompletionPoint {
    pub(crate) fn new(device: DeviceIdentity, serial: u64) -> Self {
        Self { device, serial }
    }

    /// The device identity this completion belongs to.
    pub fn device_identity(self) -> DeviceIdentity {
        self.device
    }

    /// The completion serial within that identity.
    pub fn serial(self) -> u64 {
        self.serial
    }
}

/// A structured asynchronous failure.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionFailure {
    message: String,
}

impl CompletionFailure {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The human-readable detail.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// The state of one completion point.
///
/// After device loss, every pending completion for that identity must reach
/// [`CompletionState::DeviceLost`] through bounded host and device polling; it
/// may not stay pending forever. Tokens that already completed remain complete.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompletionState {
    /// The work is still running or queued.
    Pending,

    /// The work reached terminal completion.
    Complete,

    /// The device identity was lost.
    DeviceLost(DeviceLossInfo),

    /// The backend reported a terminal failure.
    Failed(CompletionFailure),
}

impl CompletionState {
    /// Whether this state is terminal.
    pub fn is_terminal(&self) -> bool {
        !matches!(self, Self::Pending)
    }
}

/// The receipt for one accepted plan.
///
/// It exposes two completion levels on purpose: overall plan completion, and
/// per-[`PlanPoint`] completion. Forcing readback, transient retirement, or
/// resource reuse to await an unrelated slowest batch in the plan would erase
/// the reason the plan has batches at all.
#[derive(Clone)]
pub struct SubmissionReceipt {
    inner: Arc<dyn SubmissionReceiptBackend>,
}

impl core::fmt::Debug for SubmissionReceipt {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SubmissionReceipt")
            .field("submitted", &self.submitted())
            .field("completion", &self.completion())
            .finish_non_exhaustive()
    }
}

impl SubmissionReceipt {
    pub(crate) fn new(inner: Arc<dyn SubmissionReceiptBackend>) -> Self {
        Self { inner }
    }

    /// The device identity this receipt belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device_identity()
    }

    /// The logical serial at which RHI accepted this plan.
    pub fn submitted(&self) -> SubmissionPoint {
        self.inner.submitted()
    }

    /// Terminal completion of all GPU work in this plan.
    ///
    /// It does not include display scan-out completion.
    pub fn completion(&self) -> CompletionPoint {
        self.inner.completion()
    }

    /// Terminal completion of the work corresponding to `point`.
    ///
    /// A backend that cannot provide finer completion returns the same token as
    /// overall completion rather than inventing precision it does not have.
    pub fn completion_for(&self, point: PlanPoint) -> RhiResult<CompletionPoint> {
        self.inner.completion_for(point)
    }

    /// The presentation receipts this plan produced.
    pub fn presents(&self) -> &[PresentReceipt] {
        self.inner.presents()
    }
}

/// The backend half of a [`SubmissionReceipt`].
pub(crate) trait SubmissionReceiptBackend: Send + Sync + 'static {
    /// The device identity this receipt belongs to.
    fn device_identity(&self) -> DeviceIdentity;

    /// The logical serial at which RHI accepted this plan.
    fn submitted(&self) -> SubmissionPoint;

    /// Terminal completion of all GPU work in this plan.
    fn completion(&self) -> CompletionPoint;

    /// Terminal completion of the work corresponding to `point`.
    fn completion_for(&self, point: PlanPoint) -> RhiResult<CompletionPoint>;

    /// The presentation receipts this plan produced.
    fn presents(&self) -> &[PresentReceipt];
}

/// The object id reserved for a plan's opaque handle.
///
/// Plans themselves are validated values rather than long-lived device objects,
/// but tooling still needs to name one, so the id is minted from the same
/// process-wide domain as every other object.
pub(crate) fn next_plan_object_id() -> ObjectId {
    super::platform::next_object_id()
}

#[cfg(test)]
mod tests;
