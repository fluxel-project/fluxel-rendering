//! Closed canvas-format parsing and browser spelling.

use super::WebGpuCanvasFormat;

impl WebGpuCanvasFormat {
    pub(super) fn parse(value: &str) -> Option<Self> {
        match value {
            "rgba8unorm" => Some(Self::Rgba8Unorm),
            "bgra8unorm" => Some(Self::Bgra8Unorm),
            _ => None,
        }
    }
    /// Browser spelling, kept closed to the two portable renderer profiles.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rgba8Unorm => "rgba8unorm",
            Self::Bgra8Unorm => "bgra8unorm",
        }
    }
}
