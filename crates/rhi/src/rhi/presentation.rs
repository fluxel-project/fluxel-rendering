//! Presentation targets, configuration leases, acquired frames, and present
//! outcomes.
//!
//! This module owns rhi-design sections 42 to 46.
//!
//! # What it is
//!
//! Surface facts are a property of *(Device, PresentationTarget)*, never of a
//! Device alone, because one device may drive several targets with different
//! formats and extent ownership. The same target may be preflighted by several
//! providers, but only one [`ConfiguredPresentation`] lease may be active at a
//! time.
//!
//! P0 permits at most one outstanding [`AcquiredFrame`] per configured
//! presentation. That is not a limitation inherited from one backend: it keeps
//! upper layers from quietly depending on a two- or three-image swapchain when
//! P0 explicitly does not freeze an image count, and every backend can implement
//! it.
//!
//! # What it deliberately does not own
//!
//! A presentation frame is *not* a texture. P0 exposes no drawable
//! [`TextureView`], no `min_image_count`/`max_image_count`/
//! `desired_image_count`, and no copy or blit to a frame, because a GL or WebGL2
//! default framebuffer is not a texture at all and inventing a "surface texture"
//! for a few backends would pollute resource identity, lifetime, and inventory.
//! The portable final-output route is a raster scope whose color attachment is a
//! [`FrameAttachment`]; a direct MSAA resolve into a frame is legal only when the
//! active presentation facts and route facts prove that exact target, format,
//! sample count, and resolve route.
//!
//! GPU completion and presentation outcome are independent. `Accepted` means the
//! presentation system consumed frame ownership; it never means the screen has
//! displayed the frame, and a present failure is never written back as "the work
//! was not submitted".

use std::sync::Arc;

use super::format::TextureFormat;
use super::platform::{DeviceIdentity, DeviceLossInfo, Label, ObjectId, RhiResult};
use super::resource::Extent3d;

/// An opaque host or platform presentation target.
///
/// It does not belong to the device execution domain. The portable core exposes
/// no `HWND`, `CAMetalLayer`, `VkSurfaceKHR`, or canvas/context constructor; the
/// host or platform integration creates one and the backend that owns it
/// recovers its own surface from the opaque payload.
#[derive(Clone)]
pub struct PresentationTarget {
    id: ObjectId,
    payload: Arc<dyn core::any::Any + Send + Sync>,
}

impl core::fmt::Debug for PresentationTarget {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PresentationTarget")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl PresentationTarget {
    /// This target's object id, stable across clones.
    pub fn id(&self) -> ObjectId {
        self.id
    }

    /// Wraps a host-owned payload. Only the backend that produced the payload
    /// can recover it, by downcasting to its own private target type.
    pub(crate) fn new(payload: Arc<dyn core::any::Any + Send + Sync>) -> Self {
        Self {
            id: super::platform::next_object_id(),
            payload,
        }
    }

    /// The opaque payload, for the owning backend.
    pub(crate) fn payload(&self) -> &(dyn core::any::Any + Send + Sync) {
        self.payload.as_ref()
    }
}

/// A presentation policy.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PresentMode {
    /// The only portable mode every presentation backend must support.
    ///
    /// It promises no particular native mode mapping; a host or compositor
    /// policy is never disguised as [`Self::Fifo`].
    Automatic,
    /// First in, first out.
    Fifo,
    /// A mailbox queue.
    Mailbox,
    /// Immediate presentation.
    Immediate,
}

/// A two-component extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Extent2d {
    /// The width in texels.
    pub width: u32,
    /// The height in texels.
    pub height: u32,
}

impl Extent2d {
    /// A 2D extent.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Whether both components are non-zero. A zero extent means the target is
    /// suspended or minimized, which is not an acquire-able state.
    pub fn is_drawable(self) -> bool {
        self.width > 0 && self.height > 0
    }

    /// This extent as a 3D extent with depth one.
    pub fn to_extent3d(self) -> Extent3d {
        Extent3d::d2(self.width, self.height)
    }
}

/// Who owns the drawable extent.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PresentationExtentControl {
    /// The host owns the extent: a browser canvas, an adopted GL context, or
    /// some surface and window paths.
    HostManaged {
        /// The current host extent, absent while the target is suspended.
        current: Option<Extent2d>,
    },

    /// The RHI may request an exact drawable extent.
    Configurable {
        /// The smallest permitted extent.
        min: Extent2d,
        /// The largest permitted extent.
        max: Extent2d,
    },
}

/// Surface facts for one target under one device.
///
/// This is a snapshot. Resize, display move, host context recreation,
/// compositor change, or surface loss may all make it stale, so a capability
/// query is never a permanent guarantee that configure will succeed.
#[derive(Clone, Debug)]
pub struct PresentationTargetCapabilities {
    formats: Vec<TextureFormat>,
    present_modes: Vec<PresentMode>,
    extent_control: PresentationExtentControl,
}

impl PresentationTargetCapabilities {
    /// Builds a snapshot.
    ///
    /// [`PresentMode::Automatic`] is inserted when absent, because it is the one
    /// mode every presentation backend must support.
    pub(crate) fn new(
        mut formats: Vec<TextureFormat>,
        mut present_modes: Vec<PresentMode>,
        extent_control: PresentationExtentControl,
    ) -> Self {
        formats.sort_unstable();
        formats.dedup();
        if !present_modes.contains(&PresentMode::Automatic) {
            present_modes.push(PresentMode::Automatic);
        }
        present_modes.sort_unstable();
        present_modes.dedup();
        Self {
            formats,
            present_modes,
            extent_control,
        }
    }

    /// The formats the target accepts, sorted and deduplicated.
    pub fn formats(&self) -> &[TextureFormat] {
        &self.formats
    }

    /// The present modes the target accepts, always including
    /// [`PresentMode::Automatic`].
    pub fn present_modes(&self) -> &[PresentMode] {
        &self.present_modes
    }

    /// Who owns the drawable extent.
    pub fn extent_control(&self) -> PresentationExtentControl {
        self.extent_control
    }
}

/// The extent a caller requests at configuration time.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PresentationExtent {
    /// The host owns the extent.
    HostManaged,
    /// An exact drawable extent. Legal only when the capability is
    /// [`PresentationExtentControl::Configurable`].
    Exact(Extent2d),
}

/// A requested presentation configuration.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PresentationConfiguration {
    format: TextureFormat,
    present_mode: PresentMode,
    extent: PresentationExtent,
}

impl PresentationConfiguration {
    /// A configuration for `format`, using the portable defaults.
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

    /// Sets the extent request.
    pub fn with_extent(mut self, extent: PresentationExtent) -> Self {
        self.extent = extent;
        self
    }

    /// The requested format.
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// The requested present mode.
    pub fn present_mode(&self) -> PresentMode {
        self.present_mode
    }

    /// The requested extent.
    pub fn extent(&self) -> PresentationExtent {
        self.extent
    }
}

/// Why an acquire did not produce a frame.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AcquireErrorKind {
    /// No frame is available yet.
    NotReady,
    /// The wait budget elapsed.
    Timeout,
    /// P0 allows at most one outstanding frame per configured presentation.
    FrameOutstanding,
    /// The target has zero extent, for example because it is minimized.
    ZeroSizeOrSuspended,
    /// The target needs reconfiguration.
    Outdated,
    /// The target is lost.
    TargetLost,
    /// The device identity is lost.
    DeviceLost,
    /// Allocation failed.
    OutOfMemory,
}

/// A structured acquire failure.
#[derive(Debug)]
pub struct AcquireError {
    kind: AcquireErrorKind,
    message: String,
}

impl AcquireError {
    /// The classification of this failure.
    pub fn kind(&self) -> AcquireErrorKind {
        self.kind
    }

    /// The human-readable detail.
    pub fn message(&self) -> &str {
        &self.message
    }

    pub(crate) fn new(kind: AcquireErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl core::fmt::Display for AcquireError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for AcquireError {}

/// The identity of one acquired frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AcquiredFrameId {
    device: DeviceIdentity,
    serial: u64,
}

impl AcquiredFrameId {
    pub(crate) fn new(device: DeviceIdentity, serial: u64) -> Self {
        Self { device, serial }
    }

    /// The device identity this frame belongs to.
    pub fn device_identity(self) -> DeviceIdentity {
        self.device
    }

    /// The acquisition serial within that identity.
    pub fn serial(self) -> u64 {
        self.serial
    }
}

/// Where an acquired frame is in its lifecycle.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AcquiredFrameState {
    /// Acquired and usable as a color attachment.
    Acquired,
    /// Consumed by a present plan that has not been accepted yet.
    PlannedForPresent,
    /// The presentation system accepted the frame.
    PresentAccepted,
    /// Explicitly abandoned.
    Abandoned,
    /// The target needs reconfiguration.
    Outdated,
    /// The target is lost.
    TargetLost,
    /// The device identity is lost.
    DeviceLost,
}

/// A reference to an acquired frame's color attachment.
///
/// It guarantees only that it may be used as a color render attachment. It is
/// not a texture and cannot enter a bind group, a copy, a readback, or a
/// storage binding, so no backend can accidentally treat a default framebuffer
/// as an ordinary resource.
#[derive(Clone)]
pub struct FrameAttachment {
    inner: Arc<dyn FrameAttachmentBackend>,
}

impl core::fmt::Debug for FrameAttachment {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FrameAttachment")
            .field("frame", &self.frame_id())
            .finish_non_exhaustive()
    }
}

impl FrameAttachment {
    pub(crate) fn new(inner: Arc<dyn FrameAttachmentBackend>) -> Self {
        Self { inner }
    }

    /// The frame this attachment refers to.
    pub fn frame_id(&self) -> AcquiredFrameId {
        self.inner.frame_id()
    }

    /// The device identity this frame belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device_identity()
    }

    /// The frame's format.
    pub fn format(&self) -> TextureFormat {
        self.inner.format()
    }

    /// The current drawable texel extent.
    pub fn extent(&self) -> Extent3d {
        self.inner.extent()
    }

    /// The sample count, fixed at one for P0 presentation frames.
    pub fn sample_count(&self) -> u32 {
        self.inner.sample_count()
    }

    /// The frame's current lifecycle state.
    ///
    /// Every command that uses a frame attachment checks this, so recording the
    /// same attachment after present acceptance, after an abandon, or after
    /// target or device loss all fail rather than sending a stale native
    /// drawable into a backend.
    pub fn state(&self) -> AcquiredFrameState {
        self.inner.state()
    }
}

/// The backend half of a [`FrameAttachment`].
pub(crate) trait FrameAttachmentBackend: Send + Sync + 'static {
    /// The frame this attachment refers to.
    fn frame_id(&self) -> AcquiredFrameId;

    /// The device identity this frame belongs to.
    fn device_identity(&self) -> DeviceIdentity;

    /// The frame's format.
    fn format(&self) -> TextureFormat;

    /// The current drawable texel extent.
    fn extent(&self) -> Extent3d;

    /// The sample count.
    fn sample_count(&self) -> u32;

    /// The frame's current lifecycle state.
    fn state(&self) -> AcquiredFrameState;
}

/// A non-cloneable owner token for one acquired frame.
///
/// It is deliberately not `Clone`: the frame's attachment has exactly one owner
/// at a time, and duplicating the token would let two recorders believe they own
/// the same image.
pub struct AcquiredFrame {
    inner: Option<Arc<dyn AcquiredFrameBackend>>,
}

impl core::fmt::Debug for AcquiredFrame {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.inner.as_ref() {
            None => f.debug_struct("AcquiredFrame").field("consumed", &true).finish(),
            Some(backend) => f
                .debug_struct("AcquiredFrame")
                .field("id", &backend.frame_id())
                .field("state", &backend.state())
                .finish_non_exhaustive(),
        }
    }
}

impl AcquiredFrame {
    pub(crate) fn new(inner: Arc<dyn AcquiredFrameBackend>) -> Self {
        Self { inner: Some(inner) }
    }

    fn backend(&self) -> &Arc<dyn AcquiredFrameBackend> {
        // The option is only taken in `abandon` and `drop`, which consume self.
        self.inner
            .as_ref()
            .expect("acquired frame backend is present until consumed")
    }

    /// This frame's identity.
    pub fn id(&self) -> AcquiredFrameId {
        self.backend().frame_id()
    }

    /// The device identity this frame belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.backend().device_identity()
    }

    /// This frame's current lifecycle state.
    pub fn state(&self) -> AcquiredFrameState {
        self.backend().state()
    }

    /// The frame's color attachment.
    pub fn attachment(&self) -> FrameAttachment {
        self.backend().attachment()
    }

    /// Moves the frame to `PlannedForPresent`.
    ///
    /// A submission plan calls this when it takes ownership of the frame. It is
    /// crate-visible rather than public because the transition is not a caller's
    /// decision: a caller expresses the same intent by handing the frame to
    /// `SubmissionPlanBuilder::present_after`.
    pub(crate) fn plan_for_present(&self) -> RhiResult<()> {
        self.backend().plan_for_present()
    }

    /// Declares that this frame must not be presented.
    ///
    /// This is a lifecycle escape, not a performance promise: a backend may have
    /// to recreate the swapchain to make sure the presentation system does not
    /// retain the frame permanently.
    pub fn abandon(mut self) -> RhiResult<()> {
        let backend = self
            .inner
            .take()
            .expect("acquired frame backend is present until consumed");
        backend.abandon()
    }
}

impl Drop for AcquiredFrame {
    fn drop(&mut self) {
        if let Some(backend) = self.inner.take() {
            // Drop cannot return an error, so the backend performs no-throw
            // abandonment bookkeeping and emits a diagnostic. The alternative
            // would be leaking an acquired image or drawable permanently.
            backend.abandon_on_drop();
        }
    }
}

/// The backend half of an [`AcquiredFrame`].
pub(crate) trait AcquiredFrameBackend: Send + Sync + 'static {
    /// This frame's identity.
    fn frame_id(&self) -> AcquiredFrameId;

    /// The device identity this frame belongs to.
    fn device_identity(&self) -> DeviceIdentity;

    /// This frame's current lifecycle state.
    fn state(&self) -> AcquiredFrameState;

    /// The frame's color attachment.
    fn attachment(&self) -> FrameAttachment;

    /// The explicit abandon path.
    fn abandon(&self) -> RhiResult<()>;

    /// The no-throw drop path.
    fn abandon_on_drop(&self);

    /// Moves the frame from `Acquired` to `PlannedForPresent`.
    ///
    /// A submission plan calls this when it takes ownership of the frame, so a
    /// frame that is already promised to a present cannot also be abandoned or
    /// presented a second time.
    fn plan_for_present(&self) -> RhiResult<()>;
}

/// An active presentation configuration lease.
///
/// Dropping it releases the target's configuration lease; if a frame is still
/// outstanding, the no-throw abandonment path runs first.
#[derive(Clone)]
pub struct ConfiguredPresentation {
    inner: Arc<dyn ConfiguredPresentationBackend>,
}

impl core::fmt::Debug for ConfiguredPresentation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ConfiguredPresentation")
            .field("id", &self.id())
            .field("target", &self.target_id())
            .finish_non_exhaustive()
    }
}

impl ConfiguredPresentation {
    pub(crate) fn new(inner: Arc<dyn ConfiguredPresentationBackend>) -> Self {
        Self { inner }
    }

    /// This lease's object id.
    pub fn id(&self) -> ObjectId {
        self.inner.id()
    }

    /// The device identity this lease belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device_identity()
    }

    /// The presentation target this lease configures.
    pub fn target_id(&self) -> ObjectId {
        self.inner.target_id()
    }

    /// The active configuration.
    pub fn configuration(&self) -> &PresentationConfiguration {
        self.inner.configuration()
    }

    /// Reconfigures the same device and target.
    ///
    /// # Panics
    ///
    /// Never. The backend refuses when a frame is outstanding.
    pub fn reconfigure(&mut self, config: &PresentationConfiguration) -> RhiResult<()> {
        self.inner.reconfigure(config)
    }

    /// Acquires the next frame.
    ///
    /// Returns [`AcquireErrorKind::FrameOutstanding`] while the previous frame
    /// has neither entered an accepted present plan nor been abandoned.
    pub fn acquire(&mut self) -> Result<AcquiredFrame, AcquireError> {
        self.inner.acquire()
    }
}

/// The backend half of a [`ConfiguredPresentation`].
pub(crate) trait ConfiguredPresentationBackend: Send + Sync + 'static {
    /// This lease's object id.
    fn id(&self) -> ObjectId;

    /// The device identity this lease belongs to.
    fn device_identity(&self) -> DeviceIdentity;

    /// The presentation target this lease configures.
    fn target_id(&self) -> ObjectId;

    /// The active configuration.
    fn configuration(&self) -> &PresentationConfiguration;

    /// Reconfigures the same device and target.
    fn reconfigure(&self, config: &PresentationConfiguration) -> RhiResult<()>;

    /// Acquires the next frame.
    fn acquire(&self) -> Result<AcquiredFrame, AcquireError>;
}

/// The identity of one present plan inside a submission plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PresentPlanId {
    pub(crate) plan: super::submission::SubmissionPlanId,
    pub(crate) local: u32,
}

impl PresentPlanId {
    /// The submission plan this present belongs to.
    pub fn submission_plan(self) -> super::submission::SubmissionPlanId {
        self.plan
    }
}

/// The identity of one present receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PresentReceiptId {
    device: DeviceIdentity,
    serial: u64,
}

impl PresentReceiptId {
    pub(crate) fn new(device: DeviceIdentity, serial: u64) -> Self {
        Self { device, serial }
    }

    /// The device identity this receipt belongs to.
    pub fn device_identity(self) -> DeviceIdentity {
        self.device
    }

    /// The receipt serial within that identity.
    pub fn serial(self) -> u64 {
        self.serial
    }
}

/// A structured present failure.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PresentFailure {
    message: String,
}

impl PresentFailure {
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

/// The outcome of a planned presentation.
///
/// This is independent of GPU work completion: GPU work may be `Complete` while
/// the present is `Outdated`, and `Accepted` never means the screen has already
/// displayed the frame.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum PresentState {
    /// The presentation is planned and not yet resolved.
    Pending,

    /// The presentation system or host lifecycle accepted and consumed frame
    /// ownership. This does not mean scan-out or display completion.
    Accepted,

    /// The target needs reconfiguration.
    Outdated,
    /// The target is lost.
    TargetLost,
    /// The device identity is lost.
    DeviceLost(DeviceLossInfo),
    /// The backend reported a terminal present failure.
    Failed(PresentFailure),
}

/// The outcome handle for one planned presentation.
#[derive(Clone, Debug)]
pub struct PresentReceipt {
    id: PresentReceiptId,
    plan_id: PresentPlanId,
}

impl PresentReceipt {
    pub(crate) fn new(id: PresentReceiptId, plan_id: PresentPlanId) -> Self {
        Self { id, plan_id }
    }

    /// This receipt's identity, used to query [`Device::present_state`].
    ///
    /// [`Device::present_state`]: super::Device::present_state
    pub fn id(&self) -> PresentReceiptId {
        self.id
    }

    /// The present plan this receipt resolves.
    pub fn plan_id(&self) -> PresentPlanId {
        self.plan_id
    }
}

/// The label a presentation configuration uses when none was supplied.
pub(crate) fn default_configuration_label() -> Label {
    Label::none()
}
