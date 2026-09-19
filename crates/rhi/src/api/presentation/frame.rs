//! Acquiring a frame and using it as a render attachment (specification
//! section 44).
//!
//! The frame lifecycle: the refusal vocabulary an acquire reports, the frame's
//! identity and state, the attachment a raster scope draws into, and the
//! non-Clone token that owns the acquired drawable until it is presented or
//! abandoned. It does not own the present outcome (section 45), the lease the
//! frame came from (section 43), or the plan the frame enters (section 39).
//!
//! Invariant: an acquired drawable is owned by exactly one [`AcquiredFrame`], and
//! that ownership ends in exactly one of three ways — the frame enters an accepted
//! present plan, it is explicitly abandoned, or the target or device is lost.
//! Every other exit, including a plain `Drop`, is bookkeeping for the first of
//! those being skipped, not a fourth outcome.
//!
//! ```text
//! Acquired -> PlannedForPresent      by present_after          (45.1)
//! Acquired -> Abandoned              by abandon, or by Drop    (44.4, 44.5)
//! PlannedForPresent -> Abandoned     by Drop, when no plan was submitted  (41.9)
//! any      -> Outdated/TargetLost/DeviceLost                   (44.2, 45.5)
//! ```
//!
//! A `FrameAttachment` is not an ownership token and never becomes one: it may be
//! cloned freely, and `validate_frame_attachment_use` is the rule that keeps a
//! stale one out of a recording.
//!
//! The frame and the lease it came from meet at one shared record — the lease's record
//! of the frame it has out — and it is the only thing a frame holds that points back at
//! its lease. Section 44.5 requires a dropped frame to mark its
//! [`ConfiguredPresentation`]; a frame that ended unaccounted for would leave the lease
//! answering section 43.4's refusal forever, which is the state section 46.3 forbids.
//! This token has no lease reference to do that with, and is not given one: it is moved
//! into the plan builder and outlives arbitrary lease scopes. Reporting its ending to
//! the shared record is the whole of the link, and it is a report rather than ownership
//! — the frame still does not own the lease, and the lease is still not a thing a frame
//! may reach into.

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::TextureFormat;
use crate::api::identity::DeviceIdentity;
use crate::api::presentation::configure::{
    ConfiguredPresentation, FrameEnding, OutstandingFrame, validate_acquire_allowed,
};
use crate::api::resource::texture::Extent3d;
use std::sync::Arc;

/// Why an acquire did not return a frame.
///
/// A separate vocabulary from [`crate::api::error::RhiErrorKind`], and not a
/// subset of it: `FrameOutstanding` and `ZeroSizeOrSuspended` are normal
/// frame-loop outcomes rather than refusals of an operation, so folding them into
/// the general error kind would force every caller that branches on a render error
/// to also know about frame pacing. Section 44.1 fixes the eight names.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcquireErrorKind {
    /// The presentation system has no frame ready right now.
    ///
    /// Not a failure: this is what a caller retries after its next
    /// [`Device::poll`](crate::api::platform::Device::poll).
    NotReady,
    /// The acquire waited as long as it was willing to and gave up.
    Timeout,
    /// This configuration already has a frame outstanding (section 43.4).
    FrameOutstanding,
    /// The drawable is zero-sized or the surface is suspended — a minimized
    /// window or a hidden canvas.
    ZeroSizeOrSuspended,
    /// The surface changed underneath its configuration; reconfigure and retry.
    Outdated,
    /// The presentation target is gone.
    TargetLost,
    /// The device is gone, and permanently so.
    DeviceLost,
    /// The presentation system could not allocate what the acquire needs.
    OutOfMemory,
}

/// A refused acquire.
///
/// Carries the portable [`AcquireErrorKind`] a caller branches on, plus
/// diagnostic text that is not stable. Deliberately a value type rather than an
/// [`crate::api::error::RhiError`]: an acquire refusal is not an operation error,
/// and section 44.1 keeps the two channels apart.
#[derive(Debug)]
pub struct AcquireError {
    kind: AcquireErrorKind,
    message: String,
}

impl AcquireError {
    /// Builds a refusal.
    ///
    /// Crate-private: only the code that observed the refusal may describe it.
    ///
    /// No dead-code annotation is needed:
    /// [`crate::api::presentation::configure::validate_acquire_allowed`] builds one
    /// in this crate, and a test can build one directly.
    pub(crate) fn new(kind: AcquireErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// The portable classification.
    pub fn kind(&self) -> AcquireErrorKind {
        self.kind
    }

    /// The diagnostic message. The text is not stable.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// `AcquireError` is printable and is a `std::error::Error`.
///
/// Section 44.1 declares only `Debug`, so both of these are additions. They are
/// added because the two things a frame loop does with this value — log it, or
/// forward it into a caller's own boxed error — need them, and neither adds
/// vocabulary: the kind is still the part a caller branches on, and the message
/// is still unstable. `Debug` alone would make the type unusable in the second
/// case without hand-written boilerplate at every call site.
impl core::fmt::Display for AcquireError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{}: {}", self.kind.as_str(), self.message)
    }
}

impl std::error::Error for AcquireError {}

impl AcquireErrorKind {
    /// Returns a short stable name for logs and test assertions.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotReady => "NotReady",
            Self::Timeout => "Timeout",
            Self::FrameOutstanding => "FrameOutstanding",
            Self::ZeroSizeOrSuspended => "ZeroSizeOrSuspended",
            Self::Outdated => "Outdated",
            Self::TargetLost => "TargetLost",
            Self::DeviceLost => "DeviceLost",
            Self::OutOfMemory => "OutOfMemory",
            // `#[non_exhaustive]` is for downstream crates; inside this crate every
            // kind is named, so a new variant is a compile error here until it is
            // given a name. That is the intended pressure.
        }
    }
}

/// Identity of one acquired frame.
///
/// Device-scoped, like every other token in this chapter: the serial is unique
/// within one device's life, and the pair is what makes a frame from one device
/// distinguishable from a frame from another that happens to carry the same serial.
///
/// There is no public constructor, so a caller cannot forge the identity of a
/// frame it does not hold — the same rule section 39.1 states for `PlanPoint`, and
/// for the same reason: the identity is what a frame-use validation, an abandon,
/// and a diagnostic all agree on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AcquiredFrameId {
    device: DeviceIdentity,
    serial: u64,
}

impl AcquiredFrameId {
    /// Mints the identity of the frame a backend just acquired.
    ///
    /// Crate-private: a frame identity is evidence that an acquire happened.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "minted by the acquire path when the backend port lands"
        )
    )]
    pub(crate) fn new(device: DeviceIdentity, serial: u64) -> Self {
        Self { device, serial }
    }

    /// The device the frame was acquired on.
    ///
    /// Added rather than transcribed: section 44.2 declares no accessor, but this
    /// token takes part in the cross-device checks of section 44.6, and a
    /// `FrameAttachmentUse` carrying a frame has to be able to answer which device
    /// it belongs to. [`crate::api::submission::SubmissionPoint`] and
    /// `CompletionPoint` have exactly this accessor for exactly this reason.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }
}

/// Where a frame is in its lifecycle.
///
/// Seven states, and the two that are missing are as deliberate as the ones that
/// are present: there is no `Presented` that would claim the frame reached the
/// screen (section 45.5 makes `Accepted` mean ownership transfer, not scan-out),
/// and no `Copied` or `ReadBack`, because an acquired drawable is a color render
/// attachment and nothing else (section 44.3).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcquiredFrameState {
    /// Acquired, and not yet committed to a plan. Work may be recorded against it.
    Acquired,
    /// Consumed by `present_after` and owned by a plan that is being built.
    PlannedForPresent,
    /// The presentation system has accepted and consumed frame ownership.
    PresentAccepted,
    /// Explicitly abandoned, or dropped without being presented or abandoned.
    Abandoned,
    /// The surface changed underneath the frame; it can no longer be used.
    Outdated,
    /// The presentation target is gone.
    TargetLost,
    /// The device is gone, and permanently so.
    DeviceLost,
}

/// A drawable a raster scope may use as its final color attachment.
///
/// Section 44.3 draws a hard boundary and this type is where it lives. An
/// attachment guarantees exactly one thing — that it can be the color render
/// target of a raster scope — and it is **not** a texture:
///
/// ```text
/// a TextureView it is not            and it may not become one
/// BindGroup                          refused
/// copy_buffer_to_texture             refused
/// copy_texture                       refused
/// readback                           refused
/// storage                            refused
/// ```
///
/// The reason is the same one section 42.4 gives for not exposing a drawable
/// `TextureView`: a GL/WebGL2 default framebuffer is not a texture at all, and
/// manufacturing one for a few backends would put something into texture identity,
/// lifetime, and inventory that does not obey texture rules. Cloneable, and
/// cloning still confers no ownership — see `validate_frame_attachment_use`.
#[derive(Clone)]
pub struct FrameAttachment {
    frame_id: AcquiredFrameId,
    device: DeviceIdentity,
    format: TextureFormat,
    extent: Extent3d,
}

impl FrameAttachment {
    /// Describes the drawable of an acquired frame.
    ///
    /// Crate-private: an attachment is a view onto a frame the RHI acquired, and a
    /// caller-built one would name a drawable nobody owns.
    pub(crate) fn new(
        frame_id: AcquiredFrameId,
        device: DeviceIdentity,
        format: TextureFormat,
        extent: Extent3d,
    ) -> Self {
        Self {
            frame_id,
            device,
            format,
            extent,
        }
    }

    /// The frame this attachment describes.
    ///
    /// What a validation looks the frame's state up by, and what a diagnostic
    /// reports when it says which drawable a raster scope drew into.
    pub fn frame_id(&self) -> AcquiredFrameId {
        self.frame_id
    }

    /// The device the frame was acquired on.
    ///
    /// An attachment used by another device is refused with
    /// [`RhiErrorKind::WrongDevice`] rather than being translated: section 3.3
    /// gives P0 no implicit peer copy or staging bridge to fall back on.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The format the frame is in.
    ///
    /// The format the configuration was validated against, so a raster pipeline
    /// built for it is legal without a further query.
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// The drawable's current texel extent.
    ///
    /// "Current" is load-bearing: with a host-managed extent the drawable can
    /// change size between frames, so this is a property of this frame rather than
    /// of the configuration, and a pipeline that depends on a size must be
    /// re-checked when it changes.
    pub fn extent(&self) -> Extent3d {
        self.extent
    }

    /// The sample count of the drawable, which P0 fixes at 1.
    ///
    /// Not stored, because section 44.3 makes it a constant: a multisampled
    /// presentation frame does not exist in P0, so there is no state in which this
    /// could answer anything else. Multisampling reaches a frame only through
    /// section 46.1's resolve route — a multisampled color texture resolved into
    /// this single-sample attachment — and never by making the attachment itself
    /// multisampled.
    pub fn sample_count(&self) -> u32 {
        1
    }
}

impl core::fmt::Debug for FrameAttachment {
    /// Prints portable identity, not the native drawable.
    ///
    /// Hand-written rather than derived, for the reason recorded as adjudication
    /// A16 in the 0.16 plan: the backend port adds the swapchain image or drawable
    /// this describes, and printing a native handle into a log is the leak section
    /// 42.1 keeps out of the portable surface.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("FrameAttachment")
            .field("frame_id", &self.frame_id)
            .field("device", &self.device)
            .field("format", &self.format)
            .field("extent", &self.extent)
            .finish_non_exhaustive()
    }
}

/// The owner of one acquired drawable.
///
/// Non-Clone on purpose. The drawable behind a frame is a single native resource
/// that the presentation system has leased to this RHI, so two owners would mean
/// two ways for it to end, and section 43.4's one-outstanding-frame rule would be
/// unenforceable. Movement is the only way to hand a frame on, and
/// `present_after` is where it goes.
///
/// Dropping a frame in the [`AcquiredFrameState::Acquired`] state is legal and
/// does not leak, but it is not the intended path: see [`Self::abandon`] and the
/// `Drop` impl. A frame the builder already consumed
/// ([`AcquiredFrameState::PlannedForPresent`]) is in the same position, and its
/// `Drop` performs the same transition — that path is what makes section 41.9's
/// "a plan that is never submitted leaves no frame owned" true.
pub struct AcquiredFrame {
    id: AcquiredFrameId,
    device: DeviceIdentity,
    state: AcquiredFrameState,
    attachment: FrameAttachment,
    /// The lease's record of this frame, when a lease is waiting on it.
    ///
    /// `None` for a frame no lease recorded — the shape the contract tests build — and
    /// `Some` for every frame [`ConfiguredPresentation::acquire`] produces. Section
    /// 44.5's drop path writes the ending here, and the lease reads it for section
    /// 43.4's refusal, which is what keeps the two drop paths of this chapter from
    /// disagreeing about whether a frame is still out.
    lease_record: Option<Arc<OutstandingFrame>>,
}

impl AcquiredFrame {
    /// Opens a frame in the [`AcquiredFrameState::Acquired`] state.
    ///
    /// Crate-private: a frame exists because an acquire succeeded, and only the
    /// acquire path can know that a drawable is now owned. The lease that acquired
    /// it records the identity with
    /// [`ConfiguredPresentation::set_outstanding_frame`], which is what makes the
    /// frame visible to section 43.4's rule, and hands back the record
    /// [`Self::reporting_to`] links this token to.
    ///
    /// Annotated dead code: a frame is minted by the half of the acquire that reads
    /// the surface, and that half is
    /// [`ConfiguredPresentation::acquire_from_surface`], which has no backend to call
    /// yet — so today the only callers this constructor has are the contract tests,
    /// exactly like [`AcquiredFrameId::new`] above.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "minted by the acquire path's surface port when it lands"
        )
    )]
    pub(crate) fn new(
        id: AcquiredFrameId,
        device: DeviceIdentity,
        format: TextureFormat,
        extent: Extent3d,
    ) -> Self {
        Self {
            id,
            device,
            state: AcquiredFrameState::Acquired,
            attachment: FrameAttachment::new(id, device, format, extent),
            lease_record: None,
        }
    }

    /// Links this frame to the lease record its ending is reported to.
    ///
    /// Crate-private, and called by [`ConfiguredPresentation::acquire`] with the
    /// record it just installed: section 44.5's report is the only way the lease learns
    /// that the frame it has out is finished, so a token that never received one is a
    /// token whose ending the lease cannot see. Putting the link in the acquire path —
    /// rather than in the lease's later reads — is what makes that impossible to
    /// forget: a frame exists only because that path built it, and this is the same
    /// expression.
    ///
    /// Consuming and returning `self` keeps the construction and the link one
    /// expression. The parameter is an `Option` because that is what the lease's
    /// recorder returns rather than because a frame without a record is expected: it is
    /// `None` only where no lease recorded the frame at all.
    pub(crate) fn reporting_to(mut self, record: Option<Arc<OutstandingFrame>>) -> Self {
        self.lease_record = record;
        self
    }

    /// Reports this frame's ending to its lease, when it has one.
    ///
    /// Section 44.5's "when necessary, mark `ConfiguredPresentation`" is this call: the
    /// lease answers section 43.4's refusal, and the ending is the fact that decides
    /// whether the frame is still outstanding. Nothing else may report to the record —
    /// the ending is observed in exactly one place, this token's `Drop`.
    fn report(&self, ending: FrameEnding) {
        if let Some(record) = &self.lease_record {
            record.report(ending);
        }
    }

    /// This frame's identity.
    pub fn id(&self) -> AcquiredFrameId {
        self.id
    }

    /// The device the frame was acquired on.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// Where the frame is in its lifecycle.
    pub fn state(&self) -> AcquiredFrameState {
        self.state
    }

    /// The attachment a raster scope draws into.
    ///
    /// Returns a fresh attachment each call and confers no ownership: the frame
    /// still owns the drawable, and every use of the returned value must pass
    /// `validate_frame_attachment_use` before it is recorded.
    pub fn attachment(&self) -> FrameAttachment {
        self.attachment.clone()
    }

    /// Transfers this frame into a plan that will present it.
    ///
    /// Section 45.1's state change, and the only writer of
    /// [`AcquiredFrameState::PlannedForPresent`]: `present_after` consumes the
    /// frame and calls this before the plan exists, because section 44.6 decides
    /// whether *other* work may still touch the attachment by looking at this
    /// state. Until the plan is accepted, the frame belongs to the plan being built
    /// (section 41.9).
    ///
    /// ```text
    /// Acquired            -> PlannedForPresent, and the plan now owns the frame
    /// PlannedForPresent   -> InvalidUsage; a frame enters one plan
    /// every other state   -> InvalidUsage for Abandoned/PresentAccepted, and the
    ///                        state's own kind for Outdated/TargetLost/DeviceLost
    /// ```
    ///
    /// The refused states keep the kinds [`Self::abandon`] gives them, because the
    /// question a caller has is the same one — why can this frame not be used —
    /// and answering it with `InvalidUsage` for a lost device would blame the
    /// caller for the device going away.
    ///
    /// Crate-private: a frame changes hands because a builder consumed it, and the
    /// builder is the only caller.
    pub(crate) fn mark_planned_for_present(&mut self) -> RhiResult<()> {
        match self.state {
            AcquiredFrameState::Acquired => {
                self.state = AcquiredFrameState::PlannedForPresent;
                Ok(())
            }
            AcquiredFrameState::PlannedForPresent => Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "frame {:?} already belongs to a plan being built; a frame enters one \
                     present plan",
                    self.id
                ),
            )
            .at("SubmissionPlanBuilder::present_after")),
            AcquiredFrameState::Abandoned => Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("frame {:?} was abandoned", self.id),
            )
            .at("SubmissionPlanBuilder::present_after")),
            AcquiredFrameState::PresentAccepted => Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "frame {:?} has already been accepted for presentation",
                    self.id
                ),
            )
            .at("SubmissionPlanBuilder::present_after")),
            AcquiredFrameState::Outdated => Err(RhiError::new(
                RhiErrorKind::TargetOutdated,
                format!("frame {:?} no longer matches its surface", self.id),
            )
            .at("SubmissionPlanBuilder::present_after")),
            AcquiredFrameState::TargetLost => Err(RhiError::new(
                RhiErrorKind::TargetLost,
                format!("frame {:?}'s presentation target was lost", self.id),
            )
            .at("SubmissionPlanBuilder::present_after")),
            AcquiredFrameState::DeviceLost => Err(RhiError::new(
                RhiErrorKind::DeviceLost,
                format!("frame {:?}'s device was lost", self.id),
            )
            .at("SubmissionPlanBuilder::present_after")),
        }
    }

    /// Explicitly declines to present this frame.
    ///
    /// The portable promise is only *"do not present; this RHI is responsible for
    /// safely terminating frame ownership"*, and section 44.4 is emphatic that it
    /// does not promise to be cheap: a backend may release the acquired image,
    /// retire and recreate the swapchain, or drop the drawable outright, because
    /// the alternative is retaining a frame forever.
    ///
    /// The name is `abandon` rather than `discard` for that reason. "Discard"
    /// suggests every backend has a cheap release-acquired-image primitive, and
    /// they do not — Vulkan's `vkReleaseSwapchainImagesKHR` is a maintenance1
    /// capability, and in the standard lifecycle the image is released by present.
    ///
    /// Consuming `self` is what makes double abandonment impossible: after this
    /// call there is no token left to abandon again.
    pub fn abandon(mut self) -> RhiResult<()> {
        match self.state {
            AcquiredFrameState::Acquired => self.state = AcquiredFrameState::Abandoned,
            AcquiredFrameState::PlannedForPresent => {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "this frame is owned by a plan being built; a plan being built is \
                     abandoned as a whole, not frame by frame",
                )
                .at("AcquiredFrame::abandon"));
            }
            AcquiredFrameState::Abandoned => {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "this frame was already abandoned",
                )
                .at("AcquiredFrame::abandon"));
            }
            AcquiredFrameState::PresentAccepted => {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "this frame has already been accepted for presentation",
                )
                .at("AcquiredFrame::abandon"));
            }
            AcquiredFrameState::Outdated => {
                return Err(RhiError::new(
                    RhiErrorKind::TargetOutdated,
                    "this frame's drawable no longer matches its surface",
                )
                .at("AcquiredFrame::abandon"));
            }
            AcquiredFrameState::TargetLost => {
                return Err(RhiError::new(
                    RhiErrorKind::TargetLost,
                    "this frame's presentation target was lost",
                )
                .at("AcquiredFrame::abandon"));
            }
            AcquiredFrameState::DeviceLost => {
                return Err(RhiError::new(
                    RhiErrorKind::DeviceLost,
                    "this frame's device was lost",
                )
                .at("AcquiredFrame::abandon"));
            }
        }
        unimplemented!(
            "releasing the acquired drawable needs the presentation backend; the \
             contract is fixed, the release is not built"
        )
    }
}

impl Drop for AcquiredFrame {
    /// Performs no-throw abandonment when a frame is dropped unpresented and reports
    /// the ending to the lease that acquired it (section 44.5).
    ///
    /// Section 44.5 requires this path to exist because Rust's `Drop` cannot
    /// return an error: the normal path is an explicit `present_after` or
    /// `abandon`, but a caller that does neither must not leak the acquired image
    /// or drawable. So a drop in the [`AcquiredFrameState::Acquired`] state
    /// transitions to [`AcquiredFrameState::Abandoned`] — which is the entire
    /// portable content of "abandonment bookkeeping" — and the next acquire may
    /// then report `Outdated` or `NotReady` until cleanup or reconfiguration
    /// finishes.
    ///
    /// [`AcquiredFrameState::PlannedForPresent`] takes the same transition, and
    /// that is section 41.9's other half rather than a widening of this one: a plan
    /// owns the frames it consumed until it is accepted, so a builder that fails
    /// its validation, a plan dropped without being submitted, and a `submit` that
    /// returned an `Err` all end here, per frame, with no submission performed. If
    /// this arm did not exist, the frames on those paths would be retained for the
    /// life of the process, because no other code holds them.
    ///
    /// Section 44.5's three obligations, and where each one is carried out:
    ///
    /// ```text
    /// no-throw abandonment bookkeeping        the state transition below, here
    /// emit a DiagnosticEvent                  carried here, not emitted: the queue is
    ///                                         the device's, and a token holds a
    ///                                         DeviceIdentity rather than a sink to
    ///                                         write to. The record keeps the ending an
    ///                                         event would have named, so the port that
    ///                                         owns the queue has the fact it needs
    /// mark the lease Outdated/NeedsRecovery   reported here, through the record
    ///                                         acquire linked this token to; an
    ///                                         abandonment is the ending that owes a
    ///                                         release recovery, and the record keeps
    ///                                         that apart from a present or a loss
    /// ```
    ///
    /// The third is what makes the two drop paths of this chapter agree. This token
    /// cannot reach its lease's outstanding-frame record by clearing it — it holds an
    /// identity rather than a lease, and deliberately so — and the lease cannot observe
    /// an ending, because the token is what observes it. Reporting into the record they
    /// share is the only place both can see, and it is why a lease cannot be left
    /// answering section 43.4's "acquired frame already exists" after the frame it
    /// named is gone, which is the state section 46.3 forbids.
    ///
    /// Nothing here can fail and nothing here panics. Section 44.5's obligations are
    /// bookkeeping precisely because they run where an error cannot be returned: this
    /// `Drop` may run during unwinding, where a panic aborts the process instead of
    /// propagating, so the parts whose machinery does not exist yet are reported as
    /// missing rather than reached for — the same rule `ToolingSubscription`'s empty
    /// `Drop` and `RasterScope`'s poisoning `Drop` are written to.
    fn drop(&mut self) {
        match self.state {
            AcquiredFrameState::Acquired | AcquiredFrameState::PlannedForPresent => {
                self.state = AcquiredFrameState::Abandoned;
                self.report(FrameEnding::Abandoned);
            }
            // An explicit `abandon` leaves the token here and drops it immediately —
            // the call consumes `self`, so this arm is always the one that follows it
            // — and the ending is the same abandonment the arm above reports.
            AcquiredFrameState::Abandoned => self.report(FrameEnding::Abandoned),
            // Section 45.1's other outcome: the presentation system took the drawable,
            // so the lease owes the surface no recovery.
            AcquiredFrameState::PresentAccepted => self.report(FrameEnding::Presented),
            // Terminal states are already where this path would put them, and the
            // two loss states must not be overwritten: section 45.5 keeps a lost
            // frame lost, and rewriting it as `Abandoned` would tell a diagnostic
            // the caller abandoned a frame the device took away. They still end the
            // lease's claim — section 43.4 lists loss as one of the three ways an
            // outstanding frame ends — but they report the ending they are, not an
            // abandonment.
            AcquiredFrameState::Outdated
            | AcquiredFrameState::TargetLost
            | AcquiredFrameState::DeviceLost => self.report(FrameEnding::Lost),
        }
    }
}

impl core::fmt::Debug for AcquiredFrame {
    /// Prints portable identity and state, not the native drawable.
    ///
    /// Hand-written rather than derived, for the reason recorded as adjudication
    /// A16 in the 0.16 plan.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("AcquiredFrame")
            .field("id", &self.id)
            .field("device", &self.device)
            .field("state", &self.state)
            .field("attachment", &self.attachment)
            .finish_non_exhaustive()
    }
}

impl ConfiguredPresentation {
    /// Acquires the next frame to draw into.
    ///
    /// The single-outstanding-frame rule of section 43.4 makes this the one place
    /// where the portable state of a lease is checked: a second acquire while a
    /// frame is out is refused with [`AcquireErrorKind::FrameOutstanding`] before
    /// the backend is reached, and the next acquire is allowed only after that
    /// frame enters an accepted present plan, is abandoned, or is terminated by
    /// target or device loss.
    ///
    /// The other refusals section 44.1 lists come from the surface itself, so they
    /// are answered by `Self::acquire_from_surface`, which is the backend that can
    /// observe them. The outstanding-frame refusal above it is real, and so is
    /// everything below it: the frame the surface returns is recorded in this lease
    /// and linked to the record its own `Drop` reports its ending to, because section
    /// 44.5's report is what ends the refusal above.
    pub fn acquire(&mut self) -> Result<AcquiredFrame, AcquireError> {
        validate_acquire_allowed(self.outstanding_frame())?;
        let frame = self.acquire_from_surface()?;
        // Recording and linking are one step, and the record is what section 44.5's
        // drop path writes: a frame handed to a caller without it would be a frame
        // whose abandonment the lease could not see, and the lease would refuse the
        // next acquire forever — the state section 46.3 forbids.
        let record = self.set_outstanding_frame(Some(frame.id()));
        Ok(frame.reporting_to(record))
    }

    /// The native half of an acquire: the drawable the surface hands over, as a frame.
    ///
    /// Panics until the presentation backend exists. The drawable, the format and
    /// extent a [`FrameAttachment`] describes, and the identity
    /// [`AcquiredFrameId::new`] mints for it are all facts only a surface can answer,
    /// and the refusals section 44.1 lists beyond [`AcquireErrorKind::FrameOutstanding`]
    /// — `NotReady`, `Timeout`, `ZeroSizeOrSuspended`, `Outdated`, `TargetLost`,
    /// `DeviceLost`, `OutOfMemory` — are observations of it.
    ///
    /// It is a separate function rather than an `unimplemented!()` inside
    /// [`Self::acquire`] because the portable wiring around it is not a placeholder:
    /// the record that frame reports its ending to has to be installed by the acquire
    /// that produced it, and a verb that panicked before installing it would leave
    /// section 44.5's report with nothing to write to for as long as the port takes to
    /// arrive.
    fn acquire_from_surface(&self) -> Result<AcquiredFrame, AcquireError> {
        unimplemented!(
            "the acquired drawable comes from the presentation backend; the contract \
             is fixed, the acquire is not built"
        )
    }
}

/// Checks whether a [`FrameAttachment`] may be used by the work being recorded.
///
/// Section 44.6 requires every command that uses an attachment to check it, and
/// this is that check. Cloning an attachment confers no ownership, so the state it
/// names can change behind a copy that is still held; without this rule a stale
/// native drawable or swapchain image would be sent into the backend.
///
/// `frame_state` must be looked up by [`FrameAttachment::frame_id`] — it is the
/// state of the frame this attachment describes, not of some other frame — and
/// `planned_in_current_closure` says whether that frame has already been committed
/// to the present plan currently being built. The rules:
///
/// ```text
/// another device                            WrongDevice
/// Acquired                                  legal
/// PlannedForPresent, this plan's closure     legal
/// PlannedForPresent, another closure         InvalidUsage
/// Abandoned                                  InvalidUsage
/// PresentAccepted                            InvalidUsage
/// Outdated                                   TargetOutdated
/// TargetLost                                 TargetLost
/// DeviceLost                                 DeviceLost
/// ```
///
/// The `Outdated` row is the one kind not named in section 44.6's list: that
/// section enumerates the four cases "recording after present accepted", "after
/// abandon", "after loss", and "by another Device", each with its kind, while
/// `Outdated` is a fifth state the frame model has. Its kind comes from section
/// 4's mapping instead — a surface whose configuration no longer matches it is
/// `TargetOutdated`, not `TargetLost`, and the distinction matters because
/// reconfiguring recovers from one and not the other.
///
/// The `PlannedForPresent` pair is what section 44.6 means by the same "present-plan
/// closure that is being built and has not yet been accepted": while that plan is
/// still open, its work may keep using the attachment, and every other recording
/// may not.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "called by the recorder's frame-attachment path and by the plan \
                  builder once either checks a frame use; neither does yet"
    )
)]
pub(crate) fn validate_frame_attachment_use(
    attachment: &FrameAttachment,
    frame_state: AcquiredFrameState,
    using_device: DeviceIdentity,
    planned_in_current_closure: bool,
) -> RhiResult<()> {
    if attachment.device_identity() != using_device {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            format!(
                "frame {:?} belongs to another device; section 3.3 gives P0 no implicit \
                 copy or staging bridge between devices",
                attachment.frame_id()
            ),
        ));
    }
    match frame_state {
        AcquiredFrameState::Acquired => Ok(()),
        AcquiredFrameState::PlannedForPresent if planned_in_current_closure => Ok(()),
        AcquiredFrameState::PlannedForPresent => Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "frame {:?} is planned for presentation by a different plan; it may only \
                 be used by the closure that will present it",
                attachment.frame_id()
            ),
        )),
        AcquiredFrameState::Abandoned => Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "frame {:?} was abandoned and cannot be recorded against",
                attachment.frame_id()
            ),
        )),
        AcquiredFrameState::PresentAccepted => Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "frame {:?} has already been accepted for presentation",
                attachment.frame_id()
            ),
        )),
        AcquiredFrameState::Outdated => Err(RhiError::new(
            RhiErrorKind::TargetOutdated,
            format!(
                "frame {:?} no longer matches its surface; reconfigure and acquire again",
                attachment.frame_id()
            ),
        )),
        AcquiredFrameState::TargetLost => Err(RhiError::new(
            RhiErrorKind::TargetLost,
            format!(
                "frame {:?}'s presentation target was lost",
                attachment.frame_id()
            ),
        )),
        AcquiredFrameState::DeviceLost => Err(RhiError::new(
            RhiErrorKind::DeviceLost,
            format!("frame {:?}'s device was lost", attachment.frame_id()),
        )),
    }
}
