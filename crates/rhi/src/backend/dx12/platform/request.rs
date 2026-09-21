//! One-shot Direct3D 12 device-request handover.
//!
//! D3D12 device creation is synchronous, but the portable provider contract is
//! asynchronous for backends that genuinely need it. This request owns the
//! completed device directly and transfers it exactly once when polled.

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::platform::backend::{DeviceRequestBackend, RequestProgress};

use super::device::Dx12Device;

pub(super) struct Dx12Request {
    native: Option<Dx12Device>,
}

impl Dx12Request {
    pub(super) fn new(native: Dx12Device) -> Self {
        Self {
            native: Some(native),
        }
    }
}

impl DeviceRequestBackend for Dx12Request {
    fn poll_or_register_waker(&mut self, _waker: &std::task::Waker) -> RhiResult<RequestProgress> {
        let native = self.native.take().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a completed DX12 device request was polled more than once",
            )
            .at("Dx12Request::poll_or_register_waker")
        })?;
        Ok(RequestProgress::Ready(Box::new(native)))
    }
}
