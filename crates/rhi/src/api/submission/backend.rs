//! Crate-private recording and submission backend contract (modules 04 and
//! 05, sections 29 through 41).
//!
//! Portable validation runs in the public API before this contract is reached;
//! a backend lowers the validated plan and does not redefine its legality.
//! This module adds one thing to them, and it is the thing that makes this seam
//! different from the platform and resource ones:
//!
//! # What crosses this seam is a *plan*, not a command list
//!
//! `CommandRecorder` is a portable command builder and never a native command
//! list (section 29): recording runs no backend code at all, and `finish()`
//! produces a [`crate::api::command::RecordedWork`] whose command tree is private
//! to this crate. The backend is first reached at
//! [`crate::api::platform::Device::submit`], and what it is handed there is a
//! whole validated plan — batches, in-lane order, and the happens-before edges
//! between them — rather than the individual calls the caller made.
//!
//! That shape is not a simplification. Section 40.1 requires the *plan* to carry
//! the ordering ("implicit same-lane order and explicit dependencies jointly
//! participate in cycle detection"), and section 41.4 requires the RHI to
//! validate a new plan against work already in flight — both are whole-plan
//! questions that a command-at-a-time seam could not answer. The spec never
//! assigns per-verb backend duties at all; ADR-0012 declares native
//! lowering explicitly unfrozen, so the shape below is this crate's own design
//! and not a transcription.
//!
//! # Why the completion tokens here are bare `u64`
//!
//! [`SubmissionOutcome`] answers with the backend's own serials, and the portable
//! layer wraps each into a [`CompletionPoint`] whose device half only it can
//! mint. The alternative — letting the backend hand back a finished
//! `CompletionPoint` — would give a backend the power to mint public identity,
//! which is the one thing section 3.1 reserves for the layer that enforces
//! generation rules.
//!
//! The serials are opaque to the portable layer in exactly the way section 39.2
//! requires: "it is not a native fence/timeline value; distinct logical points may
//! map to the same native completion primitive". A backend is therefore free to
//! hand several plan points the same serial — section 41.2 permits a coarse
//! backend explicitly — and is also free to leave
//! [`SubmissionOutcome::points`] empty, which
//! [`crate::api::submission::SubmissionReceipt::completion_for`] already reads as
//! "the overall token is the answer".
//!
//! # There is no `wait` here
//!
//! Section 41.10 forbids a blocking completion wait in the frame loop and
//! section 6.7 confines `wait_idle` to shutdown, recovery and diagnostics. So the
//! query below is a *state* rather than a wait, and a backend is expected to
//! advance that state from [`crate::api::platform::backend::DeviceBackend::poll`]. A
//! backend whose platform gives it no way to observe completion without blocking
//! has no honest implementation of
//! [`crate::api::platform::backend::DeviceBackend::completion`] and should say so where it
//! constructs its device rather than spin inside the query.

use crate::api::presentation::{FrameAttachment, PresentReceiptId};
use crate::api::submission::plan::{CompletionPoint, PlanBatch, PlanPoint, SubmissionPlanId};

pub(crate) struct BackendPresent {
    pub(crate) after: PlanPoint,
    pub(crate) receipt: PresentReceiptId,
    pub(crate) attachment: FrameAttachment,
}

/// One validated plan, as the backend receives it.
///
/// Borrowed rather than owned: the plan is dropped when `Device::submit` returns,
/// and section 38.2's strong-ownership rule is already discharged by
/// [`crate::api::command::RecordedWork`] holding its own logical objects, so the
/// backend may keep whatever it needs from the work without keeping the plan.
///
/// Every field here has already passed section 40.5's checklist — the batches are
/// non-empty, each lane is this device's, every work item's domains are contained
/// by its lane, the dependency graph is acyclic, and the in-flight hazard check
/// of section 41.4 has run. A backend therefore does not re-decide any of it;
/// what is left to it is lowering.
pub(crate) struct SubmissionRequest<'a> {
    /// The plan's portable identity.
    ///
    /// Carried for diagnostics and for the backend's own bookkeeping, never as
    /// something the backend validates: a plan that reached here was already
    /// compared against the device's identity in `Device::submit`.
    // The expectation here and on the two fields below is *unconditional*, unlike
    // the one on `batches`. The difference was measured rather than reasoned
    // about: these three are read by no backend in any build, so the lint fires
    // in a test build too and a `not(test)`-gated expectation would leave the
    // test build warning. `batches` is read by the mock and by the DX12 lowering,
    // and the mock exists only in a test build — so its expectation is gated on
    // the whole backend feature list, which is the only gate that is fulfilled in
    // every configuration (rule 4.6: when a configuration turns up that the gate
    // had no row for, the row is added rather than the code patched).
    #[expect(
        dead_code,
        reason = "read by a lowering that attributes a submission to the plan it came from; \
                  DX12 identifies a plan by the serials it hands back in the receipt, and the \
                  mock models a device with no work to do and needs none of it"
    )]
    pub(crate) plan: SubmissionPlanId,
    /// The batches, in insertion order.
    #[cfg_attr(
        not(any(
            test,
            // The backend features that actually compile a lowering. A feature
            // that selects nothing must not appear here: it would remove this
            // expectation in a configuration where the item really is dead, and
            // the gate would then be silent about it. When Vulkan lands and starts
            // calling this, its feature joins the list — which is rule 4.6's
            // "the matrix gets the row" applied to the attribute itself.
            feature = "dx12",
            feature = "vulkan"
        )),
        expect(
            dead_code,
            reason = "read by a backend's command lowering, which is what walks the batches; \
                      a build with no backend compiled has nothing that could lower one"
        )
    )]
    pub(crate) batches: &'a [PlanBatch],
    /// Explicit happens-before edges between batches of this plan.
    ///
    /// A backend that lowers several batches onto one queue gets these for free
    /// from the queue's own order. One that does not — several native queues, or
    /// a platform whose submissions are independent — must establish each edge
    /// itself, and section 40.2 permits it to decline only when it can prove the
    /// edge is unreachable, never merely because it has no separate primitive.
    #[expect(
        dead_code,
        reason = "read by a lowering that puts batches on more than one native queue; \
                  DX12's single queue supplies every one of these edges by its own order"
    )]
    pub(crate) dependencies: &'a [(PlanPoint, PlanPoint)],
    /// Happens-before edges from earlier submitted work into this plan.
    ///
    /// The `CompletionPoint` half is a *device-local alias*: it is the token
    /// `Device::submit` handed a caller earlier, whose serial is the value this
    /// backend reported in a previous [`SubmissionOutcome`]. A backend resolves it
    /// against its own completion bookkeeping — which is why
    /// [`CompletionPoint`]'s serial has a crate-private reader rather than none.
    #[expect(
        dead_code,
        reason = "read by a lowering that must wait on prior work; DX12's single queue already \
                  orders a submission after everything before it on that queue"
    )]
    pub(crate) external_dependencies: &'a [(CompletionPoint, PlanPoint)],
    pub(crate) presents: &'a [BackendPresent],
}

/// What a backend reports once it has accepted a plan.
///
/// Returned only on the Phase B path of section 41.3: the portable preflight has
/// already passed, so a backend reaching this call is committing native work and
/// an `Err` from here would be the one outcome section 41.3 forbids — a caller
/// told "nothing happened" while a queue has already been fed. A backend that
/// accepts work and then discovers a problem reports it through
/// [`crate::api::platform::backend::DeviceBackend::completion`] as a terminal `Failed`
/// state, not by failing this call.
pub(crate) struct SubmissionOutcome {
    /// The serial naming completion of every batch in the plan.
    ///
    /// The token [`crate::api::submission::SubmissionReceipt::completion`] will
    /// report, and the fallback for every plan point the backend has no finer
    /// answer for.
    pub(crate) completion: u64,
    /// Per-batch completion serials, where the backend can be finer.
    ///
    /// Empty is legal and coarse rather than wrong: section 41.2 says a backend
    /// unable to provide finer completion may return the same token as overall
    /// completion, and `completion_for` already falls back for a point with no
    /// entry. A backend that *can* be finer should be, because a readback ticket
    /// and a transient allocator's reuse both hang off the per-batch token.
    pub(crate) points: Vec<(PlanPoint, u64)>,
}

// The two operations that consume the types above — `submit` and `completion` —
// are declared on [`crate::api::platform::backend::DeviceBackend`] rather than on a trait
// of this module, following the rule `api/resource/backend.rs` sets out: a contract
// carries the *object*, and operations belong to the device backend. A recording
// is not an object with a native life of its own — it is a value the portable
// layer owns outright and hands down — so there is nothing here for a trait to
// carry, and a second trait would be a second shape for the same layer.
//
// What is left in this module is therefore vocabulary: the request a plan becomes
// on its way down, and the outcome a backend reports on its way back up.
