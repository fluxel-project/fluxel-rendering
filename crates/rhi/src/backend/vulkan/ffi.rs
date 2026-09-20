//! Vulkan's narrow result boundary.
//!
//! No Vulkan handle crosses this chapter.  In particular, `VkResult` is turned
//! into the portable error vocabulary here, before a caller can observe it.

use ash::vk;

use crate::api::error::{RhiError, RhiErrorKind};

/// What a native Vulkan result means for the producing device identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeFailure {
    /// `VK_ERROR_DEVICE_LOST`: the `VkDevice` cannot be recovered.
    Terminal,
    /// Allocation failed without proving that the device ended.
    OutOfMemory,
    /// The driver refused an operation while the device may remain usable.
    Refused,
}

impl NativeFailure {
    pub(super) fn classify(result: vk::Result) -> Self {
        match result {
            vk::Result::ERROR_DEVICE_LOST => Self::Terminal,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                Self::OutOfMemory
            }
            _ => Self::Refused,
        }
    }

    pub(super) fn is_terminal(self) -> bool {
        matches!(self, Self::Terminal)
    }

    fn kind(self) -> RhiErrorKind {
        match self {
            Self::Terminal => RhiErrorKind::DeviceLost,
            Self::OutOfMemory => RhiErrorKind::OutOfMemory,
            Self::Refused => RhiErrorKind::BackendFailure,
        }
    }
}

/// A classified `VkResult`, retained until the owning device can publish loss.
pub(super) struct NativeError {
    result: vk::Result,
    failure: NativeFailure,
    operation: &'static str,
}

impl NativeError {
    pub(super) fn new(result: vk::Result, operation: &'static str) -> Self {
        Self {
            result,
            failure: NativeFailure::classify(result),
            operation,
        }
    }

    pub(super) fn failure(&self) -> NativeFailure {
        self.failure
    }

    pub(super) fn into_rhi(self) -> RhiError {
        RhiError::new(
            self.failure.kind(),
            format!("Vulkan call failed: {:?}", self.result),
        )
        .at(self.operation)
    }
}

pub(super) fn to_rhi(result: vk::Result, operation: &'static str) -> RhiError {
    NativeError::new(result, operation).into_rhi()
}
