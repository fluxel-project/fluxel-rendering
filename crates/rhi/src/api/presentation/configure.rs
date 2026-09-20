//! Presentation configuration and the configuration lease (specification
//! section 43).
//!
//! What a caller asks a target to be, and the lease that makes the answer
//! exclusive: the format, the present mode, and who owns the drawable extent,
//! plus [`ConfiguredPresentation`], which is the only thing an
//! [`AcquiredFrame`](crate::api::presentation::AcquiredFrame) can be acquired
//! from. It does not own the target (a host object), the surface facts (section
//! 42), or the frame lifecycle (section 44).
//!
//! Invariant: at most one active lease per target, and at most one outstanding
//! frame per lease. Both are *preconditions* rather than reported state, so the
//! validators here take the facts as parameters and the verbs that need a
//! backend refuse before they panic.
//!
//! ```text
//! one active ConfiguredPresentation per target                       (42.1)
//! one outstanding AcquiredFrame per ConfiguredPresentation           (43.4)
//! capability query does not guarantee configure                        (42.5)
//! ```

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::TextureFormat;
use crate::api::identity::{DeviceIdentity, ObjectId};
use crate::api::platform::{Device, DeviceStatus};
use crate::api::presentation::PresentationTarget;
use crate::api::presentation::frame::{AcquireError, AcquireErrorKind, AcquiredFrameId};
use crate::api::presentation::target::{
    Extent2d, PresentMode, PresentationExtentControl, PresentationTargetCapabilities,
};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

/// The drawable extent a configuration asks for.
///
/// Section 43.1's one rule is the reason this is not just an `Option<Extent2d>`:
/// [`Self::Exact`] is legal only against a target whose capability is
/// [`crate::api::presentation::PresentationExtentControl::Configurable`], and the
/// refusal belongs to the request rather than to the number inside it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentationExtent {
    /// Let the target keep owning its drawable size.
    HostManaged,
    /// Ask for exactly this drawable extent, in texels.
    Exact(Extent2d),
}

/// What a target is being configured to be.
///
/// Built with [`Self::new`] and the two `with_` methods, and validated by
/// `validate_presentation_configuration` against the target's current facts.
///
/// Section 43.2 fixes exactly one guarantee for a configured target: a
/// [`FrameAttachment`](crate::api::presentation::FrameAttachment) may be used as
/// the **final color render target**. It does *not* promise that an acquired
/// drawable may be sampled, copied from or to, used as storage, or read back —
/// which is the same boundary section 42.4 draws when it refuses to hand out a
/// drawable `TextureView`.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PresentationConfiguration {
    format: TextureFormat,
    present_mode: PresentMode,
    extent: PresentationExtent,
}

impl PresentationConfiguration {
    /// A configuration using `format`, with the portable defaults.
    ///
    /// The defaults are the two choices that ask the target for nothing it might
    /// not be able to give: [`PresentMode::Automatic`], the only mode every
    /// backend supports (section 42.2), and [`PresentationExtent::HostManaged`],
    /// which leaves the drawable size with the host. A caller that needs a
    /// specific mode or extent adds it with the `with_` methods and accepts that
    /// the configuration can now be refused.
    ///
    /// Section 43.2 declares no default values, so these are a choice rather than
    /// a transcription; they are the only pair that cannot narrow what a target
    /// already does.
    pub fn new(format: TextureFormat) -> Self {
        Self {
            format,
            present_mode: PresentMode::Automatic,
            extent: PresentationExtent::HostManaged,
        }
    }

    /// Sets the present mode.
    pub fn with_present_mode(mut self, mode: PresentMode) -> Self {
        self.present_mode = mode;
        self
    }

    /// Sets the drawable-extent request.
    pub fn with_extent(mut self, extent: PresentationExtent) -> Self {
        self.extent = extent;
        self
    }

    /// The format a frame acquired from this configuration will be in.
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// The requested present mode.
    pub fn present_mode(&self) -> PresentMode {
        self.present_mode
    }

    /// The requested drawable extent.
    pub fn extent(&self) -> PresentationExtent {
        self.extent
    }
}

/// How a frame's token reported that its ownership ended (section 44.5).
///
/// The state of a frame that has not ended, plus section 43.4's three ways an
/// outstanding frame ends: an accepted present, an explicit or no-throw
/// abandonment, and target or device loss. The three stay apart because section
/// 44.5 asks for different things after each — an abandonment left the acquired
/// drawable unreleased, so it is the ending that asks a release for recovery
/// ("when necessary, mark `ConfiguredPresentation` as `Outdated`/`NeedsRecovery`"),
/// while an accepted present transferred the drawable to the presentation system
/// and a loss took the target away with it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameEnding {
    /// The token still owns the drawable, so section 43.4 refuses the next acquire.
    Outstanding,
    /// Ended without a present — explicitly, or by `Drop` (sections 44.4, 44.5).
    Abandoned,
    /// Ended because the presentation system accepted it (section 45.1).
    Presented,
    /// Ended with its target or device (sections 44.2, 45.5).
    Lost,
}

impl FrameEnding {
    /// The wire value the shared record stores.
    ///
    /// A match rather than `as u8`, so that the bytes in the record are this
    /// enum's to define: a value whose meaning changed because a variant moved
    /// would corrupt every record already open, and section 44.5's obligation to
    /// mark the lease rests on reading one of them correctly.
    fn to_wire(self) -> u8 {
        match self {
            Self::Outstanding => 0,
            Self::Abandoned => 1,
            Self::Presented => 2,
            Self::Lost => 3,
        }
    }

    /// The inverse of [`Self::to_wire`].
    ///
    /// The catch-all is unreachable for every value this crate writes, and it
    /// answers [`Self::Abandoned`] for the same reason `ReadbackStatus::from_raw`
    /// answers `Failed`: both of the other candidate answers are the direction that
    /// loses something. [`Self::Outstanding`] would leave the lost byte reading as
    /// "this lease still has a frame out", which is the state section 46.3 forbids a
    /// target to be left in, and [`Self::Lost`] would tell the release that the
    /// presentation system owns the drawable, which is what section 44.5 forbids an
    /// unaccounted-for frame to leave behind. [`Self::Abandoned`] asks for the
    /// release instead, and an unneeded release is legal where a leaked drawable is
    /// not.
    fn from_wire(raw: u8) -> Self {
        match raw {
            0 => Self::Outstanding,
            1 => Self::Abandoned,
            2 => Self::Presented,
            3 => Self::Lost,
            // Same answer as `1`, and stated separately rather than folded into that arm
            // so the encoding above reads as the four endings it is.
            _ => Self::Abandoned,
        }
    }
}

/// The lease's record of the one frame it has out, shared with that frame's token.
///
/// A record rather than a bare [`AcquiredFrameId`], because the ending of that frame
/// is a fact two owners need and neither can compute alone: the lease answers section
/// 43.4's "already has a frame outstanding" from it, and the token is the only thing
/// that observes the ending — section 44.5's `Drop` — while holding no reference to
/// the lease. It cannot be given one: a frame token is moved into the plan builder
/// and outlives arbitrary lease scopes, so a lease reference would either put a
/// lifetime on a public token or put a shared cell where section 44.3's one-owner rule
/// lives. Sharing this record instead is where the chapter's two drop paths meet, and
/// is what keeps them from contradicting each other: the token ends the claim, and the
/// lease's own release ends the lease.
///
/// `AtomicU8` rather than a plain field, for the reason
/// `ReadbackTicket`'s shared state is atomic: the token is dropped on whatever thread
/// owns it while the lease is read from the thread that holds it, and neither side is
/// allowed to impose a lock on the other.
pub(crate) struct OutstandingFrame {
    /// The frame every refusal names, and the identity a diagnostic reports.
    id: AcquiredFrameId,
    /// The ending the token reported, as [`FrameEnding`]'s wire value.
    ending: AtomicU8,
}

/// The one shared ownership domain of a configured native lease and any frame
/// it has handed out.  A frame has to keep the configured native object alive
/// after its `ConfiguredPresentation` wrapper was moved or dropped; keeping
/// that relationship in one inner avoids independently reference-counting the
/// native lease and its outstanding-frame bookkeeping.
pub(crate) struct ConfiguredPresentationInner {
    native: Box<dyn crate::api::presentation::backend::ConfiguredPresentationBackend>,
    outstanding: Mutex<Option<OutstandingFrame>>,
}

impl ConfiguredPresentationInner {
    pub(crate) fn new(
        native: Box<dyn crate::api::presentation::backend::ConfiguredPresentationBackend>,
    ) -> Self {
        Self {
            native,
            outstanding: Mutex::new(None),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_backed() -> Arc<Self> {
        Arc::new(Self::new(Box::new(TestConfiguredPresentationBackend)))
    }

    pub(crate) fn native(
        &self,
    ) -> &dyn crate::api::presentation::backend::ConfiguredPresentationBackend {
        self.native.as_ref()
    }

    fn outstanding_frame(&self) -> Option<AcquiredFrameId> {
        let outstanding = self
            .outstanding
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match outstanding.as_ref() {
            Some(frame) if frame.ending() == FrameEnding::Outstanding => Some(frame.id()),
            _ => None,
        }
    }

    fn set_outstanding_frame(&self, frame: Option<AcquiredFrameId>) {
        let mut outstanding = self
            .outstanding
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *outstanding = frame.map(OutstandingFrame::new);
    }

    pub(crate) fn report(&self, id: AcquiredFrameId, ending: FrameEnding) {
        let outstanding = self
            .outstanding
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(record) = outstanding.as_ref().filter(|record| record.id() == id) {
            record.report(ending);
        }
    }
}

impl Drop for ConfiguredPresentationInner {
    fn drop(&mut self) {
        self.native.release();
    }
}

impl OutstandingFrame {
    /// Opens a record for a frame a lease is about to hand out.
    fn new(id: AcquiredFrameId) -> Self {
        Self {
            id,
            ending: AtomicU8::new(FrameEnding::Outstanding.to_wire()),
        }
    }

    /// The frame this record is about.
    fn id(&self) -> AcquiredFrameId {
        self.id
    }

    /// The ending the token last reported.
    fn ending(&self) -> FrameEnding {
        FrameEnding::from_wire(self.ending.load(Ordering::Acquire))
    }

    /// Records how the frame ended (section 44.5).
    ///
    /// Called from the frame token's `Drop`, which is the only place the ending is
    /// observed, and by nothing else: a write from the lease's side would be the lease
    /// inventing the one fact it does not hold.
    pub(crate) fn report(&self, ending: FrameEnding) {
        self.ending.store(ending.to_wire(), Ordering::Release);
    }
}

impl core::fmt::Debug for OutstandingFrame {
    /// Prints the frame and the ending it reported, not the wire byte.
    ///
    /// Hand-written rather than derived for the reason the type's other diagnostics
    /// are: a derived `Debug` would print the record's private encoding, and a reader
    /// of the log could not tell an abandoned frame from a lost one — which is the
    /// distinction a frame loop debugging "why did my next acquire report
    /// `Outdated`" needs.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let ending = match self.ending() {
            FrameEnding::Outstanding => "outstanding",
            FrameEnding::Abandoned => "abandoned",
            FrameEnding::Presented => "presented",
            FrameEnding::Lost => "lost",
        };
        formatter
            .debug_struct("OutstandingFrame")
            .field("id", &self.id)
            .field("ending", &ending)
            .finish()
    }
}

/// An exclusive configuration of one presentation target on one device.
///
/// Section 42.1 makes this the *only* way to hold a target configured: the same
/// target may be preflighted by several providers and devices, but there may be
/// only one active lease at a time, so a second `configure` of the same target —
/// from the same device or another — is refused. Changing backend or device
/// therefore means dropping the old lease first.
///
/// This is also the only source of [`crate::api::presentation::AcquiredFrame`]s,
/// and the reason section 43.4 can state the one-outstanding-frame rule at all: a
/// lease knows whether it has a frame out.
///
/// Releasing a lease with a frame still out is section 46.3's case. The shared
/// inner remains alive until that frame ends, then releases the native lease exactly
/// once from its own `Drop`.
pub struct ConfiguredPresentation {
    id: ObjectId,
    pub(crate) device: DeviceIdentity,
    target_id: ObjectId,
    pub(crate) configuration: PresentationConfiguration,
    /// The exact shared domain held by this lease and every frame it acquires.
    pub(crate) inner: Arc<ConfiguredPresentationInner>,
}

impl ConfiguredPresentation {
    /// Opens a lease over `target_id` with an already validated configuration.
    ///
    /// Crate-private: a lease exists because a device configured a surface, so
    /// only `Device::configure_presentation` may create one. The configuration is
    /// expected to have passed
    /// [`validate_presentation_configuration`] against facts queried at
    /// configuration time — section 42.5 makes that a fresh check rather than a
    /// reuse of whatever the caller queried earlier.
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        target_id: ObjectId,
        configuration: PresentationConfiguration,
        native: Box<dyn crate::api::presentation::backend::ConfiguredPresentationBackend>,
    ) -> Self {
        Self {
            id,
            device,
            target_id,
            configuration,
            inner: Arc::new(ConfiguredPresentationInner::new(native)),
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(
        id: ObjectId,
        device: DeviceIdentity,
        target_id: ObjectId,
        configuration: PresentationConfiguration,
    ) -> Self {
        Self::new(
            id,
            device,
            target_id,
            configuration,
            Box::new(TestConfiguredPresentationBackend),
        )
    }

    /// This lease's process-local identity.
    ///
    /// Distinct from the target's: one target can carry several leases over its
    /// life (each after the previous was dropped), and a diagnostic that reported
    /// only the target could not tell a reconfiguration from a second lease.
    pub fn id(&self) -> ObjectId {
        self.id
    }

    /// The device this lease belongs to.
    ///
    /// Section 42.1's exclusion is per lease, not per target alone, so this is
    /// what a caller compares when it wonders why a second device was refused.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The target this lease configures.
    pub fn target_id(&self) -> ObjectId {
        self.target_id
    }

    /// The configuration this lease is currently under.
    pub fn configuration(&self) -> &PresentationConfiguration {
        &self.configuration
    }

    /// Reconfigures this lease on the same device and target.
    ///
    /// Valid in two steps, because section 43.3 lists both:
    ///
    /// ```text
    /// there is no outstanding frame        checked here, portably
    /// format/mode/extent still valid       §42.5 requires a fresh check
    /// target/device loss                   answered by the backend that owns them
    /// ```
    ///
    /// The second and third steps need facts this lease does not hold: the fresh
    /// surface facts, and whether the device is still alive. Both come from the
    /// backend call below the portable refusal rather than being assumed here. A
    /// caller that wants the extent and format answer before that point can call
    /// `validate_presentation_configuration` against a fresh
    /// `presentation_capabilities` snapshot.
    ///
    /// The lease keeps its identity across a reconfigure: this is the same lease,
    /// with the same [`Self::id`], which is what distinguishes it from dropping
    /// the lease and configuring again.
    pub async fn reconfigure(&mut self, config: &PresentationConfiguration) -> RhiResult<()> {
        validate_reconfigure_allowed(self.outstanding_frame())?;
        let capabilities = self.inner.native().capabilities()?;
        validate_presentation_configuration(config, &capabilities)?;
        std::future::poll_fn(|context| {
            self.inner
                .native()
                .reconfigure_or_register_waker(config, context.waker())
        })
        .await?;
        self.configuration = config.clone();
        Ok(())
    }

    /// The frame this lease has outstanding, if any.
    ///
    /// The fact behind section 43.4's refusal, exposed crate-internally for
    /// [`validate_acquire_allowed`] and [`validate_reconfigure_allowed`].
    ///
    /// "Outstanding" is read from the frame's own record rather than from its
    /// presence, because section 43.4 lists three ways a frame stops being
    /// outstanding — it enters an accepted present plan, it is explicitly abandoned,
    /// or target/device loss terminates it — and only the frame token observes any of
    /// them (section 44.5). A lease that answered from the record's existence would
    /// refuse the next acquire forever after a caller dropped a frame unpresented,
    /// which is exactly the "acquired frame already exists" state section 46.3
    /// forbids the target to be left in.
    ///
    /// No dead-code annotation: this crate's own `acquire` and `reconfigure` read
    /// it already, so the item is live in every build rather than pending a port.
    pub(crate) fn outstanding_frame(&self) -> Option<AcquiredFrameId> {
        self.inner.outstanding_frame()
    }

    /// Records which frame this lease has outstanding, and opens the record its
    /// token reports its ending to.
    ///
    /// Crate-private and the only installer of the record: `acquire` sets it, and the
    /// token's own `Drop` is the only thing that ever ends it. Clearing the record
    /// before the frame is really terminal would violate section 43.4 in the direction
    /// that matters, because it would let a second drawable be acquired while the first
    /// is still owned — so nothing on this side clears it early, and the ending comes
    /// from the one place that observes it.
    ///
    /// Returns the record, because [`crate::api::presentation::AcquiredFrame`] must
    /// hold the same one: section 44.5's drop path is this token's, and a frame whose
    /// record the lease never handed over is a frame whose ending the lease cannot
    /// see.
    ///
    /// No dead-code annotation: `ConfiguredPresentation::acquire` drives it in every
    /// build, which is also why this returns the record rather than only installing it.
    pub(crate) fn set_outstanding_frame(&self, frame: Option<AcquiredFrameId>) {
        self.inner.set_outstanding_frame(frame);
    }
}

#[cfg(test)]
struct TestConfiguredPresentationBackend;

#[cfg(test)]
impl crate::api::presentation::backend::ConfiguredPresentationBackend
    for TestConfiguredPresentationBackend
{
    fn capabilities(&self) -> RhiResult<PresentationTargetCapabilities> {
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "test presentation backing has no surface capabilities",
        ))
    }

    fn reconfigure_or_register_waker(
        &self,
        _: &PresentationConfiguration,
        _: &std::task::Waker,
    ) -> std::task::Poll<RhiResult<()>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn try_acquire(
        &self,
        _: DeviceIdentity,
    ) -> Result<Option<crate::api::presentation::backend::AcquiredSurfaceFrame>, AcquireError> {
        Err(AcquireError::new(
            AcquireErrorKind::TargetLost,
            "test presentation backing cannot acquire a host drawable",
        ))
    }

    fn acquire_or_register_waker(
        &self,
        _: DeviceIdentity,
        _: &std::task::Waker,
    ) -> std::task::Poll<
        Result<crate::api::presentation::backend::AcquiredSurfaceFrame, AcquireError>,
    > {
        std::task::Poll::Ready(Err(AcquireError::new(
            AcquireErrorKind::TargetLost,
            "test presentation backing cannot acquire a host drawable",
        )))
    }

    fn abandon(&self, _: AcquiredFrameId) -> RhiResult<()> {
        Ok(())
    }

    fn abandon_no_throw(&self, _: AcquiredFrameId) {}

    fn release(&self) {}
}
impl core::fmt::Debug for ConfiguredPresentation {
    /// Prints portable identity, not the platform lease.
    ///
    /// Hand-written rather than derived, for the reason recorded as adjudication
    /// A16 in the 0.16 plan: the backend port adds the native swapchain or
    /// surface it holds, and printing that into a log is the leak section 42.1
    /// keeps out of the portable surface.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ConfiguredPresentation")
            .field("id", &self.id)
            .field("device", &self.device)
            .field("target_id", &self.target_id)
            .field("configuration", &self.configuration)
            .field("outstanding", &self.outstanding_frame())
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Opens an exclusive configuration lease over `target`.
    ///
    /// Section 42.1's single-lease rule is checked here, and the configuration is
    /// validated against facts queried *now* rather than against whatever the
    /// caller queried earlier: section 42.5 makes the capability snapshot a
    /// query-time answer, so a successful query is not a guarantee that
    /// configuration succeeds.
    ///
    /// Panics until the presentation backend exists. The device check below is
    /// still performed first — a lost device is terminal (section 3.1), and a
    /// caller that gets [`RhiErrorKind::DeviceLost`] learns something true even
    /// though no surface was touched.
    pub async fn configure_presentation(
        &self,
        target: &PresentationTarget,
        config: &PresentationConfiguration,
    ) -> RhiResult<ConfiguredPresentation> {
        if let DeviceStatus::Lost = self.status() {
            return Err(RhiError::new(
                RhiErrorKind::DeviceLost,
                "this device was lost; configuration is terminal until a new device is created",
            )
            .at("Device::configure_presentation"));
        }
        let backend = self.native().presentation().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                "this backend does not implement presentation",
            )
            .at("Device::configure_presentation")
        })?;
        let capabilities = backend.capabilities(target.id())?;
        validate_presentation_configuration(config, &capabilities)?;
        let native = backend.configure(self.identity(), target.id(), config)?;
        Ok(ConfiguredPresentation::new(
            ObjectId::next(),
            self.identity(),
            target.id(),
            config.clone(),
            native,
        ))
    }
}

/// Checks a configuration against one target's facts.
///
/// Section 43.3 lists what `configure` and `reconfigure` must validate, and
/// section 42.5 makes that check necessary even straight after a successful
/// query. The rules, with the kind each one produces under section 4's mapping:
///
/// ```text
/// format is one the target offers                 else Unsupported
/// mode is Automatic, or one the target offers     else Unsupported
/// Exact extent is legal only when Configurable    else InvalidUsage
/// Exact extent is non-zero and within [min, max]  else InvalidUsage
/// HostManaged over a currently zero-sized surface else TargetOutdated
/// ```
///
/// The three kinds are not interchangeable. A format or mode the surface never
/// reported is a capability gap, which is `Unsupported`. An exact extent past a
/// bound, or a zero-sized one, is a range error on a value the caller chose,
/// which is `InvalidUsage` — and it must not be `TargetOutdated`, which would
/// blame the surface for a request it never had to satisfy. The last line is the
/// opposite case: nothing about the request is wrong, and the surface has gone to
/// zero size (a minimized window, a hidden canvas), which section 4 maps to
/// `TargetOutdated` and section 44.1 also reports as `ZeroSizeOrSuspended` on the
/// acquire path.
///
/// Crate-private but not hidden: it takes the facts as a parameter rather than
/// reading a device, which is what makes it exercisable without a GPU.
pub(crate) fn validate_presentation_configuration(
    config: &PresentationConfiguration,
    capabilities: &PresentationTargetCapabilities,
) -> RhiResult<()> {
    if !capabilities.formats().contains(&config.format) {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            format!("this target cannot be configured with {:?}", config.format),
        ));
    }
    if config.present_mode != PresentMode::Automatic
        && !capabilities.present_modes().contains(&config.present_mode)
    {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            format!(
                "this target does not offer {:?}; Automatic is the only mode every \
                 presentation backend supports",
                config.present_mode
            ),
        ));
    }
    match (config.extent, capabilities.extent_control()) {
        (PresentationExtent::Exact(_), PresentationExtentControl::HostManaged { .. }) => {
            Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "an exact extent may only be requested from a target whose extent \
                 control is Configurable; this target's drawable size belongs to the host",
            ))
        }
        (
            PresentationExtent::Exact(extent),
            PresentationExtentControl::Configurable { min, max },
        ) => {
            if extent.width == 0 || extent.height == 0 {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "an exact extent must have a non-zero width and height",
                ));
            }
            if extent.width < min.width
                || extent.height < min.height
                || extent.width > max.width
                || extent.height > max.height
            {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "the requested extent {}x{} is outside this target's {}x{}..{}x{} range",
                        extent.width, extent.height, min.width, min.height, max.width, max.height
                    ),
                ));
            }
            Ok(())
        }
        (PresentationExtent::HostManaged, PresentationExtentControl::HostManaged { current }) => {
            match current {
                Some(current) if current.width == 0 || current.height == 0 => Err(RhiError::new(
                    RhiErrorKind::TargetOutdated,
                    "this target currently reports a zero-sized drawable, which usually means \
                 it is suspended or minimized; configure it again once it has a size",
                )),
                _ => Ok(()),
            }
        }
        // A host-managed request against a configurable target asks for nothing the
        // target has to agree to: the RHI simply does not use the exact extent it
        // could have asked for.
        (PresentationExtent::HostManaged, PresentationExtentControl::Configurable { .. }) => Ok(()),
    }
}

/// Checks that a lease may acquire its next frame.
///
/// Section 43.4's rule, and only the part of it a caller's own state decides:
///
/// ```text
/// a frame outstanding    -> AcquireErrorKind::FrameOutstanding
/// otherwise              -> acquire may proceed
/// ```
///
/// The other four acquire refusals of section 44.1 — `ZeroSizeOrSuspended`,
/// `TargetLost`, `DeviceLost`, `OutOfMemory` — are facts about the surface and the
/// device at the instant of the acquire call, which only the backend can observe.
/// They are deliberately not synthesized here: a lease that guessed "probably
/// active" would be inventing the one fact it does not hold. The order they are
/// reported in therefore belongs to the acquire path, with device loss first
/// because section 3.1 makes it terminal and section 43.4 lists loss as one of the
/// three ways an outstanding frame ends.
///
/// Like [`validate_presentation_configuration`], this takes its fact as a
/// parameter so it can be exercised without a GPU — and unlike it, no dead-code
/// annotation is needed: the crate's own `acquire` calls it in every build.
pub(crate) fn validate_acquire_allowed(
    outstanding: Option<AcquiredFrameId>,
) -> Result<(), AcquireError> {
    if let Some(frame) = outstanding {
        return Err(AcquireError::new(
            AcquireErrorKind::FrameOutstanding,
            format!(
                "this configuration already has a frame outstanding ({frame:?}); present \
                 it, abandon it, or wait for target or device loss before acquiring again"
            ),
        ));
    }
    Ok(())
}

/// Checks that a lease may be reconfigured.
///
/// Section 43.3 requires "there must be no outstanding frame" for a
/// reconfigure, and section 43.4 says why: the frame that is out was acquired
/// against the *old* configuration, so changing format, mode, or extent under it
/// would leave the acquired drawable described by facts that no longer hold.
///
/// ```text
/// a frame outstanding    -> InvalidUsage
/// otherwise              -> reconfigure may proceed
/// ```
///
/// The refusal is [`RhiErrorKind::InvalidUsage`] rather than a target state: the
/// rule is a precondition on the *call*, and a caller satisfies it by abandoning
/// or presenting the frame — the surface is not at fault. Device and target loss
/// are not decided here for the same reason [`validate_acquire_allowed`] does not
/// decide them: this lease does not hold the device, so the backend answers that
/// part.
///
/// No dead-code annotation: the crate's own `reconfigure` calls it in every build.
pub(crate) fn validate_reconfigure_allowed(outstanding: Option<AcquiredFrameId>) -> RhiResult<()> {
    if let Some(frame) = outstanding {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "this configuration still has a frame outstanding ({frame:?}); present it \
                 or abandon it before reconfiguring"
            ),
        ));
    }
    Ok(())
}
