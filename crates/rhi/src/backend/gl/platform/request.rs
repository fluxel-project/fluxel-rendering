use std::sync::Arc;

use crate::api::capability::CapabilityFacts;
use crate::api::error::RhiResult;
use crate::api::platform::backend::{DeviceRequestBackend, RequestProgress};
use crate::api::platform::provider::{AdapterInfo, BackendKind};

use super::device::{GlDevice, GlExecutionDriver};

/// A one-poll handover: context adoption does not wait for a native device.
pub(super) struct GlRequest {
    device: Option<GlDevice>,
}

impl GlRequest {
    pub(super) fn new(
        backend: BackendKind,
        adapter: AdapterInfo,
        facts: CapabilityFacts,
        driver: Arc<dyn GlExecutionDriver>,
    ) -> Self {
        Self {
            device: Some(GlDevice::new(backend, adapter, facts, driver)),
        }
    }
}

impl DeviceRequestBackend for GlRequest {
    fn poll(&mut self) -> RhiResult<RequestProgress> {
        // The portable request is still async.  It becomes ready on its first
        // poll, exactly like an adopted WebGL context already made current by
        // the host; a second poll is prohibited by the portable single-shot rule.
        let device = self.device.take().ok_or_else(|| {
            crate::api::error::RhiError::new(
                crate::api::error::RhiErrorKind::InvalidUsage,
                "the adopted GL device request was polled after it became ready",
            )
        })?;
        Ok(RequestProgress::Ready(Box::new(device)))
    }
}
