//! Backend-private Vulkan failure vocabulary.

use crate::api::error::{RhiError, RhiErrorKind};
use crate::backend::vulkan::ffi;

pub(super) enum VulkanFailure {
    Unsupported {
        what: &'static str,
        why: &'static str,
    },
    Native(ffi::NativeError),
}

impl VulkanFailure {
    pub(super) fn is_terminal(&self) -> bool {
        matches!(self, Self::Native(native) if native.failure().is_terminal())
    }

    pub(super) fn message(&self) -> String {
        match self {
            Self::Unsupported { what, why } => format!("{what}: {why}"),
            Self::Native(native) => {
                // Keep the native result in the message without converting it
                // twice. The loss authority owns the final public error.
                format!("Vulkan native failure ({:?})", native.failure())
            }
        }
    }

    pub(super) fn into_rhi(self, operation: &'static str) -> RhiError {
        match self {
            Self::Unsupported { what, why } => {
                RhiError::new(RhiErrorKind::Unsupported, format!("{what}: {why}")).at(operation)
            }
            Self::Native(native) => native.into_rhi(),
        }
    }
}
