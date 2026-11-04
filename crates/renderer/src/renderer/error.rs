use std::fmt;

use fluxel_rhi::api::{Error, ExecutionStatus};

use crate::graph::BindingError;

/// Failures reported by the narrow S0/S1 renderer facade.
#[derive(Debug)]
pub enum RendererError {
    /// The renderer was shut down or its device became unavailable.
    RendererClosed,
    /// The submitted immutable byte value was empty.
    EmptyAsset,
    /// All three completion-owned slots are occupied.
    Backpressure,
    /// The submission is owned by another renderer or RHI device.
    ForeignSubmission,
    /// The submission has retired or its slot has been reused.
    StaleSubmission,
    /// A local serial or global plan ordinal could not advance.
    IdentityExhausted,
    /// Accepted execution supplied no checkpoint covering the readback step.
    MissingCompletion,
    /// The fixed Raster fixture input cannot be represented safely.
    InvalidRasterInput,
    /// This RHI device has no compute route for the fixed S1 fixture.
    UnsupportedCompute,
    /// The fixed `u32` compute fixture input cannot be represented safely.
    InvalidComputeInput,
    /// A frame's concrete resources do not satisfy its compiled graph bindings.
    InvalidBinding,
    /// Native facts do not prove that the renderer may retire its owners.
    RetainedExecution {
        /// The native execution classification that requires retention.
        status: ExecutionStatus,
        /// Recording failure retained with the native outcome, when present.
        recording_error: Option<Error>,
        /// Failure reported by the finalized execution outcome, when present.
        execution_error: Option<Error>,
    },
    /// A typed RHI failure, preserved without string flattening.
    Rhi(Error),
    /// Logical Shader requirements could not be admitted for this device and pipeline.
    ShaderAdmission(super::ShaderAdmissionError),
    /// CPU Scene selection failed before Forward device preparation.
    ForwardPipeline(crate::PipelineError),
    /// Logical Forward assets did not satisfy the selected consumer.
    ForwardPreparation(crate::ForwardPreparationError),
    /// A Forward device operation failed; the original RHI error is preserved.
    ForwardOperation {
        /// The device operation being performed.
        operation: &'static str,
        /// The unmodified RHI failure.
        error: Error,
    },
}

impl fmt::Display for RendererError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RendererClosed => formatter.write_str("renderer is closed"),
            Self::EmptyAsset => formatter.write_str("renderer byte asset is empty"),
            Self::Backpressure => formatter.write_str("renderer submission slots are full"),
            Self::ForeignSubmission => {
                formatter.write_str("submission belongs to another renderer")
            }
            Self::StaleSubmission => formatter.write_str("submission is no longer pending"),
            Self::IdentityExhausted => formatter.write_str("renderer identity space is exhausted"),
            Self::MissingCompletion => formatter.write_str("execution has no covering completion"),
            Self::InvalidRasterInput => formatter.write_str("invalid fixed raster fixture input"),
            Self::UnsupportedCompute => {
                formatter.write_str("compute is unavailable on this device")
            }
            Self::InvalidComputeInput => formatter.write_str("invalid fixed compute fixture input"),
            Self::InvalidBinding => {
                formatter.write_str("frame resources do not satisfy the compiled graph bindings")
            }
            Self::RetainedExecution {
                status,
                recording_error,
                execution_error,
            } => {
                write!(formatter, "execution must remain retained: {status:?}")?;
                if let Some(error) = recording_error {
                    write!(formatter, " ({error})")?;
                }
                if let Some(error) = execution_error {
                    write!(formatter, " ({error})")?;
                }
                Ok(())
            }
            Self::Rhi(error) => error.fmt(formatter),
            Self::ShaderAdmission(error) => error.fmt(formatter),
            Self::ForwardPipeline(error) => error.fmt(formatter),
            Self::ForwardPreparation(error) => error.fmt(formatter),
            Self::ForwardOperation { operation, error } => {
                write!(formatter, "Forward {operation}: {error}")
            }
        }
    }
}
impl std::error::Error for RendererError {}
impl From<Error> for RendererError {
    fn from(error: Error) -> Self {
        Self::Rhi(error)
    }
}
impl From<BindingError> for RendererError {
    fn from(error: BindingError) -> Self {
        match error {
            BindingError::IdentityExhausted => Self::IdentityExhausted,
            BindingError::WrongDevice => Self::Rhi(Error::WrongDevice),
            _ => Self::InvalidBinding,
        }
    }
}
