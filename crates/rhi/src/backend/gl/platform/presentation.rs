//! GL default-framebuffer presentation adapter.
//!
//! A lease names a configured canvas/window/default framebuffer.  Its acquired
//! attachment remains a `FrameAttachmentBackend`; it is deliberately not a
//! texture view, because the WebGL2/GLES default framebuffer has no texture
//! identity or sampling/copy/storage rights.

use std::any::Any;
use std::sync::Arc;
use std::task::{Poll, Waker};

use crate::api::error::RhiResult;
use crate::api::identity::DeviceIdentity;
use crate::api::presentation::backend::{
    AcquiredSurfaceFrame, ConfiguredPresentationBackend, FrameAttachmentBackend,
};
use crate::api::presentation::{
    AcquireError, AcquiredFrameId, Extent2d, PresentReceiptId, PresentState,
    PresentationConfiguration, PresentationTargetCapabilities,
};

use super::{GlExecutionDriver, GlLossSink};

/// Backend-private configured default-framebuffer lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct GlPresentationLease(pub(crate) u64);

/// One acquired default framebuffer. `serial` is device-local and becomes the
/// portable acquired-frame identity; `framebuffer` is driver-private.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlAcquiredFramebuffer {
    pub(crate) serial: u64,
    pub(crate) extent: Extent2d,
    pub(crate) suboptimal: bool,
    pub(crate) lease: GlPresentationLease,
    pub(crate) framebuffer: u32,
}

pub(super) struct GlConfiguredPresentation {
    driver: Arc<dyn GlExecutionDriver>,
    loss_sink: Arc<dyn GlLossSink>,
    lease: GlPresentationLease,
}

impl GlConfiguredPresentation {
    /// Presentation futures bypass `DeviceBackend` after configuration, so
    /// their first observation of a dead native context must still enter the
    /// same terminal DeviceIdentity authority as submit/resource operations.
    fn observe<T>(&self, result: RhiResult<T>) -> RhiResult<T> {
        if let Err(error) = &result
            && error.kind() == crate::api::error::RhiErrorKind::DeviceLost
        {
            let info = crate::api::platform::DeviceLossInfo::new(error.to_string());
            self.driver.device_lost(&info);
            self.loss_sink.report_context_loss(info);
        }
        result
    }

    fn observe_acquire<T>(&self, result: Result<T, AcquireError>) -> Result<T, AcquireError> {
        if let Err(error) = &result
            && matches!(
                error.kind(),
                crate::api::presentation::AcquireErrorKind::DeviceLost
            )
        {
            let info = crate::api::platform::DeviceLossInfo::new(error.to_string());
            self.driver.device_lost(&info);
            self.loss_sink.report_context_loss(info);
        }
        result
    }
}

impl ConfiguredPresentationBackend for GlConfiguredPresentation {
    fn capabilities(&self) -> RhiResult<PresentationTargetCapabilities> {
        self.observe(self.driver.lease_capabilities(self.lease))
    }
    fn reconfigure_or_register_waker(
        &self,
        config: &PresentationConfiguration,
        waker: &Waker,
    ) -> Poll<RhiResult<()>> {
        match self
            .driver
            .reconfigure_or_register_waker(self.lease, config, waker)
        {
            Poll::Ready(result) => Poll::Ready(self.observe(result)),
            Poll::Pending => Poll::Pending,
        }
    }
    fn try_acquire(
        &self,
        device: DeviceIdentity,
    ) -> Result<Option<AcquiredSurfaceFrame>, AcquireError> {
        self.observe_acquire(self.driver.try_acquire(self.lease))
            .map(|frame| frame.map(|frame| surface_frame(device, Arc::clone(&self.driver), frame)))
    }
    fn acquire_or_register_waker(
        &self,
        device: DeviceIdentity,
        waker: &Waker,
    ) -> Poll<Result<AcquiredSurfaceFrame, AcquireError>> {
        match self.driver.acquire_or_register_waker(self.lease, waker) {
            Poll::Ready(result) => Poll::Ready(
                self.observe_acquire(result)
                    .map(|frame| surface_frame(device, Arc::clone(&self.driver), frame)),
            ),
            Poll::Pending => Poll::Pending,
        }
    }
    fn abandon(&self, frame: AcquiredFrameId) -> RhiResult<()> {
        self.observe(self.driver.abandon(self.lease, frame))
    }
    fn abandon_no_throw(&self, frame: AcquiredFrameId) {
        self.driver.abandon_no_throw(self.lease, frame)
    }
    fn release(&self) {
        self.driver.release_presentation(self.lease)
    }
}

fn surface_frame(
    device: DeviceIdentity,
    driver: Arc<dyn GlExecutionDriver>,
    frame: GlAcquiredFramebuffer,
) -> AcquiredSurfaceFrame {
    let _ = device;
    AcquiredSurfaceFrame {
        serial: frame.serial,
        extent: frame.extent,
        suboptimal: frame.suboptimal,
        attachment: Box::new(GlFrameAttachment { driver, frame }),
    }
}

pub(super) struct GlFrameAttachment {
    driver: Arc<dyn GlExecutionDriver>,
    frame: GlAcquiredFramebuffer,
}
impl FrameAttachmentBackend for GlFrameAttachment {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn present(&self, receipt: PresentReceiptId) {
        self.driver.present(self.frame, receipt)
    }
    fn terminate_present(&self, receipt: PresentReceiptId, state: PresentState) {
        self.driver.terminate_present(self.frame, receipt, state)
    }
}

pub(super) fn configured(
    driver: Arc<dyn GlExecutionDriver>,
    loss_sink: Arc<dyn GlLossSink>,
    lease: GlPresentationLease,
) -> Box<dyn ConfiguredPresentationBackend> {
    Box::new(GlConfiguredPresentation {
        driver,
        loss_sink,
        lease,
    })
}

/// Extracts the driver-private default framebuffer from an attachment after the
/// portable recording layer has validated device ownership and attachment use.
pub(crate) fn framebuffer_ref(
    frame: &crate::api::presentation::FrameAttachment,
) -> RhiResult<GlAcquiredFramebuffer> {
    frame
        .native()
        .as_any()
        .downcast_ref::<GlFrameAttachment>()
        .map(|native| native.frame)
        .ok_or_else(|| {
            crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::Unsupported,
                "frame attachment has no GL default-framebuffer backing",
            )
        })
}
