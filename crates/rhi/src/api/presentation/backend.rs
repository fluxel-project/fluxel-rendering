//! Crate-private presentation backend seam.
//!
//! Portable lifecycle and validation stay in this chapter. Native targets,
//! swapchains, drawables and completion mechanisms stay behind these traits.

use std::any::Any;
use std::task::{Poll, Waker};

use crate::api::error::RhiResult;
use crate::api::identity::{DeviceIdentity, ObjectId};
use crate::api::presentation::{
    AcquireError, AcquiredFrameId, Extent2d, PresentReceiptId, PresentState,
    PresentationConfiguration, PresentationTargetCapabilities, PresentationTimestamp,
};

/// Native facts returned for one acquired drawable.
pub(crate) struct AcquiredSurfaceFrame {
    pub(crate) serial: u64,
    pub(crate) extent: Extent2d,
    pub(crate) suboptimal: bool,
    /// Backend-private drawable backing. It is carried by FrameAttachment, never
    /// exposed through the public presentation vocabulary.
    pub(crate) attachment: Box<dyn FrameAttachmentBackend>,
}

/// Native drawable behind one portable frame attachment.
pub(crate) trait FrameAttachmentBackend: Send + Sync + 'static {
    fn as_any(&self) -> &dyn Any;
    /// Invoked only after the batch named by a validated present relation was
    /// executed. Failure is recorded in backend presentation state: Phase B must
    /// not return an Err after GPU work has been accepted.
    fn present(&self, _receipt: PresentReceiptId) {}
    /// Publishes a terminal receipt without issuing a native present. This is
    /// used only after Phase B has made the execution domain terminal before a
    /// later present relation could be reached. It prevents a valid receipt
    /// from remaining unknown/Pending after device loss.
    fn terminate_present(&self, _receipt: PresentReceiptId, _state: PresentState) {}
}

/// Per-lease native state. A lease is shared with acquired frame tokens so their
/// explicit and no-throw abandonment paths reach the same native owner.
pub(crate) trait ConfiguredPresentationBackend: Send + Sync + 'static {
    fn capabilities(&self) -> RhiResult<PresentationTargetCapabilities>;
    /// Attempts a reconfiguration without blocking an executor thread.
    ///
    /// A native surface may have backend-private retirement work before it can
    /// resize (DXGI `ResizeBuffers` is the important case).  Returning
    /// `Pending` is valid only after retaining `waker` for every event that can
    /// make the resize legal, including device loss.  The public operation is
    /// already async; this seam keeps that fact real without exposing native
    /// fences, swapchain generations, or retirement tokens.
    ///
    /// Every backend implements this one async seam. Backends whose native
    /// operation is immediate return `Poll::Ready`; a second synchronous trait
    /// verb would create two authorities for the same lease transition.
    fn reconfigure_or_register_waker(
        &self,
        config: &PresentationConfiguration,
        waker: &Waker,
    ) -> Poll<RhiResult<()>>;
    /// Samples acquisition without waiting. `Ok(None)` is the only answer for a
    /// drawable that is not ready yet; a backend must not turn that normal race
    /// into a busy wait or a synthetic error.
    fn try_acquire(
        &self,
        device: DeviceIdentity,
    ) -> Result<Option<AcquiredSurfaceFrame>, AcquireError>;
    /// Samples acquisition and, when it returns `Pending`, registers `waker` for
    /// every future event that can change the answer (including device loss).
    ///
    /// This is deliberately a backend-private seam: browser callbacks, DXGI
    /// waitable objects and native event loops differ, while the public RHI
    /// contract is simply an async acquire that makes progress without polling.
    fn acquire_or_register_waker(
        &self,
        device: DeviceIdentity,
        waker: &Waker,
    ) -> Poll<Result<AcquiredSurfaceFrame, AcquireError>>;
    fn abandon(&self, frame: AcquiredFrameId) -> RhiResult<()>;
    fn abandon_no_throw(&self, frame: AcquiredFrameId);
    fn release(&self);
}

/// The presentation facet of one native device.
pub(crate) trait PresentationBackend: Send + Sync + 'static {
    fn capabilities(&self, target: ObjectId) -> RhiResult<PresentationTargetCapabilities>;
    fn configure(
        &self,
        device: DeviceIdentity,
        target: ObjectId,
        config: &PresentationConfiguration,
    ) -> RhiResult<Box<dyn ConfiguredPresentationBackend>>;
    fn present_state(&self, receipt: PresentReceiptId) -> RhiResult<PresentState>;
    /// Samples one receipt and registers for its next state transition when it
    /// is pending. Returning `Pending` without retaining the waker is invalid:
    /// `Device::wait_present` must be a real future, not a scheduler yield.
    fn present_state_or_register_waker(
        &self,
        receipt: PresentReceiptId,
        waker: &Waker,
    ) -> RhiResult<PresentState>;
    fn presentation_timestamp(&self, _target: ObjectId) -> RhiResult<PresentationTimestamp> {
        Err(crate::api::error::RhiError::new(
            crate::api::error::RhiErrorKind::Unsupported,
            "this presentation backend has no presentation-clock lowering",
        ))
    }
}
