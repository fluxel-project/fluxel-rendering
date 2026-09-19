//! Structured browser-session error presentation.

use core::fmt;

use super::WebGpuSessionError;

impl fmt::Display for WebGpuSessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for WebGpuSessionError {}

impl WebGpuSessionError {
    /// Stable code used by both synchronous errors and rejected async operations.
    pub const fn code(&self) -> &'static str {
        match self {
            Self::CanvasUnavailable => "canvas-unavailable",
            Self::Unavailable => "webgpu-unavailable",
            Self::UnsupportedFormat(_) => "unsupported-canvas-format",
            Self::State(_) => "invalid-state",
            Self::Contract(code) => code,
            Self::Browser { code, .. } => code,
        }
    }
    /// Closed operation category for this error.
    pub const fn operation(&self) -> &'static str {
        match self {
            Self::CanvasUnavailable => "open-canvas",
            Self::Unavailable => "request-adapter",
            Self::UnsupportedFormat(_) => "preferred-format",
            Self::State(_) => "lifecycle",
            Self::Contract(_) => "validate-contract",
            Self::Browser { operation, .. } => operation,
        }
    }
    /// Device generation affected by the error, zero before a device exists.
    pub const fn generation(&self) -> u64 {
        match self {
            Self::Browser { generation, .. } => *generation,
            _ => 0,
        }
    }
    /// Stable human-readable error message.
    pub fn message(&self) -> String {
        self.to_string()
    }
}
