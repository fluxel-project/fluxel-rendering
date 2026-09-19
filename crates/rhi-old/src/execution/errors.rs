//! Execution error contracts.

use super::*;

/// Failure while lowering or executing the current native fixed command slice.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NativeExecutionError {
    /// An owned transient could not be created.
    Resource(ResourceCreateError),
    /// The plan selected a command family outside this backend's supported slice.
    UnsupportedCommandFamily,
    /// A logical queue other than the backend's only queue was requested.
    UnknownQueue(QueueId),
    /// A command referenced a resource owned by another device.
    ForeignResource,
    /// An encoder was used through a backend for another device.
    ForeignEncoder,
    /// A finished command buffer was submitted through another device.
    ForeignCommandBuffer,
    /// A copy range or texture region was invalid at the native boundary.
    InvalidTransfer(&'static str),
    /// A Copy command or transition was issued in the wrong pass scope.
    CopyStateMismatch,
    /// A native recording operation failed before submission.
    Recording(String),
    /// A finished command buffer was rejected without being accepted.
    SubmitRejected(String),
    /// A completion query failed after submission was accepted.
    Completion(String),
    /// A compute dispatch was zero-sized or exceeded device limits.
    InvalidDispatch,
    /// The active pipeline and bound fixed binding recipe do not match.
    ComputeBindingMismatch,
    /// The active raster state does not satisfy a fixed draw recipe.
    RasterStateMismatch,
    /// A fixed multi-stream raster recipe addressed a vertex slot outside its ABI.
    RasterVertexSlotOutOfRange {
        /// The requested vertex-buffer slot.
        slot: u32,
    },
    /// A vertex buffer does not match the role assigned to its fixed ABI slot.
    RasterVertexRoleMismatch {
        /// The requested vertex-buffer slot.
        slot: u32,
    },
    /// A fixed multi-stream ABI vertex slot was already successfully bound in this epoch.
    RasterVertexSlotAlreadyBound {
        /// The requested vertex-buffer slot.
        slot: u32,
    },
    /// A required fixed multi-stream ABI vertex slot was not bound in this epoch.
    RasterVertexSlotMissing {
        /// The required vertex-buffer slot.
        slot: u32,
    },
    /// A fixed multi-stream ABI vertex buffer range is not the required full stream.
    RasterVertexRangeMismatch {
        /// The requested vertex-buffer slot.
        slot: u32,
    },
    /// A fixed multi-stream ABI vertex buffer lacks vertex usage.
    RasterVertexUsageMismatch {
        /// The requested vertex-buffer slot.
        slot: u32,
    },
    /// Fixed multi-stream bindings were already successfully set in this pipeline epoch.
    RasterBindingsAlreadySet,
    /// A fixed multi-stream recipe object belongs to a previous pipeline epoch.
    RasterEpochMismatch,
}

impl fmt::Display for NativeExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resource(error) => write!(formatter, "resource creation failed: {error}"),
            Self::UnsupportedCommandFamily => {
                formatter.write_str("command family is not supported by this native backend")
            }
            Self::UnknownQueue(queue) => write!(formatter, "unknown logical queue {queue:?}"),
            Self::ForeignResource => {
                formatter.write_str("resource belongs to another native device")
            }
            Self::ForeignEncoder => formatter.write_str("encoder belongs to another native device"),
            Self::ForeignCommandBuffer => {
                formatter.write_str("command buffer belongs to another native device")
            }
            Self::InvalidTransfer(reason) => write!(formatter, "invalid native transfer: {reason}"),
            Self::CopyStateMismatch => {
                formatter.write_str("copy command or transition is outside its required scope")
            }
            Self::Recording(reason) => write!(formatter, "native recording failed: {reason}"),
            Self::SubmitRejected(reason) => {
                write!(formatter, "native submission was rejected: {reason}")
            }
            Self::Completion(reason) => {
                write!(formatter, "native completion query failed: {reason}")
            }
            Self::InvalidDispatch => formatter.write_str("invalid compute dispatch dimensions"),
            Self::ComputeBindingMismatch => {
                formatter.write_str("compute bindings do not match the active pipeline")
            }
            Self::RasterStateMismatch => {
                formatter.write_str("raster command does not satisfy the active fixed recipe")
            }
            Self::RasterVertexSlotOutOfRange { slot } => write!(
                formatter,
                "fixed raster vertex slot {slot} is outside the closed ABI"
            ),
            Self::RasterVertexRoleMismatch { slot } => write!(
                formatter,
                "fixed raster vertex slot {slot} has the wrong buffer role"
            ),
            Self::RasterVertexSlotAlreadyBound { slot } => {
                write!(
                    formatter,
                    "fixed raster vertex slot {slot} is already bound"
                )
            }
            Self::RasterVertexSlotMissing { slot } => {
                write!(formatter, "fixed raster vertex slot {slot} is missing")
            }
            Self::RasterVertexRangeMismatch { slot } => write!(
                formatter,
                "fixed raster vertex slot {slot} has an invalid range"
            ),
            Self::RasterVertexUsageMismatch { slot } => {
                write!(
                    formatter,
                    "fixed raster vertex slot {slot} lacks vertex usage"
                )
            }
            Self::RasterBindingsAlreadySet => {
                formatter.write_str("fixed raster bindings are already set for this epoch")
            }
            Self::RasterEpochMismatch => formatter
                .write_str("fixed raster object does not belong to the active pipeline epoch"),
        }
    }
}

impl std::error::Error for NativeExecutionError {}

impl From<ResourceCreateError> for NativeExecutionError {
    fn from(value: ResourceCreateError) -> Self {
        Self::Resource(value)
    }
}

/// Outcome of a bounded CPU wait used outside graph execution.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WaitError {
    /// The accepted submission remained pending at the deadline.
    Timeout,
    /// The accepted submission reached a terminal failure.
    Failed(CompletionFailure),
    /// The native completion query itself failed.
    Backend(NativeExecutionError),
}

impl fmt::Display for WaitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for WaitError {}
