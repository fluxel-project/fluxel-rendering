//! One-shot handover for a synchronously created Vulkan device.

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::platform::backend::{DeviceRequestBackend, RequestProgress};

use super::device::VulkanDevice;

pub(super) struct VulkanRequest {
    device: Option<VulkanDevice>,
}

impl VulkanRequest {
    pub(super) fn new(device: VulkanDevice) -> Self {
        Self {
            device: Some(device),
        }
    }
}

impl DeviceRequestBackend for VulkanRequest {
    fn poll(&mut self) -> RhiResult<RequestProgress> {
        let device = self.device.take().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a completed Vulkan device request was polled more than once",
            )
            .at("VulkanRequest::poll")
        })?;
        Ok(RequestProgress::Ready(Box::new(device)))
    }
}
