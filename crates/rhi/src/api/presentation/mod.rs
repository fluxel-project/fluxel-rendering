//! Presentation targets, configuration leases, acquired frames, and present
//! outcome (specification sections 42 through 45).
//!
//! This module owns the half of the frame loop that runs *before* a plan exists:
//! which surface facts a target reports, which configuration is leased on it,
//! which frame is outstanding, and what became of a presentation after it was
//! submitted. The other half lives in [`crate::api::submission`], and the two
//! meet at exactly three points — each of which is a rule rather than a
//! convenience:
//!
//! ```text
//! SubmissionPlanBuilder::present_after   consumes an AcquiredFrame   (45.1)
//! SubmissionReceipt::presents            the resulting PresentReceipts (41.2)
//! AcquiredFrameState / PresentState      two independent outcomes     (45.5)
//! ```
//!
//! # What this module does not own
//!
//! - **The plan.** Section 45.1 puts `present_after` on the plan builder, which
//!   is the submission module's type. Presentation is *planned before submit*
//!   (root section 3.7), so a frame belongs to the plan from the moment
//!   `present_after` consumes it, and this module owns it only up to that point.
//! - **The frame use as a recorded resource use.** Section 45.2 makes a frame a
//!   first-class `ResourceUse` variant owned by the recording chapter. This
//!   module supplies the frame identity and the state rule every such use must
//!   satisfy (`validate_frame_attachment_use`); it does not record commands.
//! - **The native surface.** Section 42.1 lists what portable RHI does *not*
//!   expose: an `HWND`, a `CAMetalLayer`, a `VkSurfaceKHR`, a canvas, or any
//!   context. A target is opaque and identified by [`crate::api::identity::ObjectId`].
//! - **Acquire refusal counters and present outcome counters.** The statistics
//!   chapter distinguishes an acquire refusal from a submitted present outcome;
//!   the counters belong there, and the states below are what they count.
//!
//! # Invariants this module enforces
//!
//! ```text
//! one active configuration lease per target                 (42.1)
//! one outstanding acquired frame per lease                  (43.4)
//! FrameAttachment is not a Texture and cannot become one    (44.3)
//! acquire refusal, GPU completion, and present outcome are three outcomes (45.5)
//! ```
//!
//! # Validation and lowering
//!
//! Every verb runs its portable rules before reaching a presentation backend,
//! in two forms:
//!
//! - the checks decidable from the arguments alone run first, inside the verb,
//!   before lowering — a wrong-device receipt or a second outstanding acquire is
//!   refused with its own kind instead of reaching a driver as an impossible
//!   request (root section 4);
//! - the checks that need a fact the caller holds run in a `pub(crate)`
//!   `validate_*` function next to the type they check, taking that fact as a
//!   parameter rather than reading a device. That is what
//!   `crate::api::resource::buffer::validate_buffer_descriptor` does with a
//!   capability answer, and it is what makes those rules testable without a GPU.

pub(crate) mod backend;
pub mod configure;
pub mod frame;
pub mod present;
pub mod target;

use crate::api::identity::ObjectId;

pub use configure::{ConfiguredPresentation, PresentationConfiguration, PresentationExtent};
pub use frame::{
    AcquireError, AcquireErrorKind, AcquiredFrame, AcquiredFrameId, AcquiredFrameState,
    FrameAttachment,
};
pub use present::{PresentFailure, PresentPlanId, PresentReceipt, PresentReceiptId, PresentState};
pub use target::{
    CompositeAlphaMode, DisplayHdrInfo, Extent2d, FrameLatencyRange, PresentMode,
    PresentationColorSpace, PresentationExtentControl, PresentationFormat,
    PresentationTargetCapabilities, PresentationTimestamp, PresentationTimingCapabilities,
};

/// A host surface the RHI can be asked to present to.
///
/// Opaque, and deliberately so. Section 42.1 lists what the portable RHI does
/// **not** expose about a target: an `HWND`, a `CAMetalLayer`, a `VkSurfaceKHR`,
/// a canvas, or any context. A caller holds a target it obtained from its host
/// integration and hands it to the RHI; it never reads a handle back out.
///
/// A target is *not* owned by a device, and section 42.1 makes that load-bearing:
/// several providers and devices may perform capability preflight against the
/// same target, but at most one active configured lease may exist at a time. That
/// rule is why the target carries an [`ObjectId`] of its own rather than being
/// identified relative to whichever device happens to be asking.
#[derive(Clone)]
pub struct PresentationTarget {
    id: ObjectId,
}

impl PresentationTarget {
    /// Wraps a target the host has already created.
    ///
    /// Crate-private: a target is a host object, so only the host-integration
    /// entry point may mint one. There is no portable constructor, because a
    /// target with no platform object behind it would be a surface the RHI could
    /// never actually present to.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "minted by host integration when api::platform is declared"
        )
    )]
    pub(crate) fn new(id: ObjectId) -> Self {
        Self { id }
    }

    /// This target's process-local identity.
    ///
    /// The only thing a caller may read from a target. It is what a diagnostic or
    /// capture tool names when it reports which surface a frame went to, and what
    /// a lease conflict is reported against.
    pub fn id(&self) -> ObjectId {
        self.id
    }
}

impl core::fmt::Debug for PresentationTarget {
    /// Prints portable identity rather than the platform object.
    ///
    /// Hand-written rather than derived, for the reason recorded as adjudication
    /// A16 in the 0.16 plan: the host-integration port will add a platform field
    /// that has no reason to be `Debug`, and printing a native surface handle into
    /// a log is exactly the leak section 42.1 keeps out of the portable surface.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PresentationTarget")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}
