//! Immutable upload contracts and pending upload values.
use super::*;
/// The stage at which an immutable buffer upload reached the native boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum BufferUploadStage {
    /// The private staging allocation or host write failed.
    Staging,
    /// Recording the fixed copy operation failed.
    Recording,
    /// The queue rejected the submission before it was accepted.
    SubmitRejected,
    /// Querying or waiting for an accepted completion failed.
    Completion,
}

/// The stage at which an immutable texture upload reached the native boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum TextureUploadStage {
    /// Creating or writing private row-padded staging storage failed.
    Staging,
    /// Recording the fixed buffer-to-texture copy failed.
    Recording,
    /// The queue rejected the submission before accepting it.
    SubmitRejected,
    /// Querying or waiting for an accepted completion failed.
    Completion,
}

/// A stable reason why an immutable buffer upload request was rejected before native work.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum InvalidBufferUploadReason {
    /// The source byte slice was empty.
    EmptyData,
    /// The source byte length was not a multiple of the portable copy alignment.
    DataLengthNotCopyAligned,
    /// The buffer descriptor size was not exactly the source byte length.
    DescriptorSizeMismatch,
    /// The upload slice accepts only device-local destination buffers.
    MemoryPolicyUnsupported,
    /// The destination declaration did not request copy-destination usage.
    CopyDestinationUsageRequired,
    /// Test-only observation requires the finalized buffer to allow copy-source use.
    CopySourceUsageRequired,
}

/// A stable reason why an immutable RGBA8 texture request was rejected.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum InvalidTextureUploadReason {
    /// Only a two-dimensional image is accepted.
    DimensionUnsupported,
    /// Only `Rgba8Unorm` or `Rgba8UnormSrgb` is accepted.
    FormatUnsupported,
    /// The closed upload has exactly one mip level.
    MipLevelsUnsupported,
    /// The closed upload has exactly one array layer.
    ArrayLayersUnsupported,
    /// The closed upload is not multisampled.
    SampleCountUnsupported,
    /// The closed upload only owns device-local storage.
    MemoryPolicyUnsupported,
    /// Required copy-destination usage is missing.
    CopyDestinationUsageRequired,
    /// Required sampled usage is missing.
    SampledUsageRequired,
    /// The closed upload declaration requested an operation outside its
    /// production contract. Test-only readback may additionally request
    /// `CopySource`; no other widening is accepted.
    UnexpectedUsage,
    /// The supplied bytes are not exactly tight RGBA8 image bytes.
    DataLengthMismatch,
    /// The declared extent cannot be represented as a tight byte length.
    DataLengthOverflow,
}

/// Why an immutable buffer upload could not be started or observed.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BufferUploadError {
    /// The portable upload request was malformed.
    InvalidRequest(InvalidBufferUploadReason),
    /// Creating the destination buffer failed.
    Resource(ResourceCreateError),
    /// An observation request used a buffer from a different native device.
    ForeignDevice,
    /// A private native boundary operation failed.
    Native {
        /// Backend selected by the owning device.
        backend: Backend,
        /// Upload stage that failed.
        stage: BufferUploadStage,
        /// Native diagnostic retained without exposing HAL types.
        reason: String,
    },
}

impl fmt::Display for BufferUploadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(reason) => {
                write!(f, "invalid immutable buffer upload: {reason:?}")
            }
            Self::Resource(error) => write!(f, "immutable buffer upload resource error: {error}"),
            Self::ForeignDevice => f.write_str("immutable buffer upload used a foreign device"),
            Self::Native {
                backend,
                stage,
                reason,
            } => write!(
                f,
                "{backend:?} immutable buffer upload {stage:?} failed: {reason}"
            ),
        }
    }
}

impl std::error::Error for BufferUploadError {}

/// Why an immutable texture upload could not be started or observed.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TextureUploadError {
    /// The portable upload request was malformed.
    InvalidRequest(InvalidTextureUploadReason),
    /// Creating the destination texture failed.
    Resource(ResourceCreateError),
    /// An observation request used a texture from a different native device.
    ForeignDevice,
    /// A private native boundary operation failed.
    Native {
        /// Backend selected by the owning device.
        backend: Backend,
        /// Upload stage that failed.
        stage: TextureUploadStage,
        /// Native diagnostic retained without exposing HAL types.
        reason: String,
    },
}

impl fmt::Display for TextureUploadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(reason) => {
                write!(f, "invalid immutable texture upload: {reason:?}")
            }
            Self::Resource(error) => write!(f, "immutable texture upload resource error: {error}"),
            Self::ForeignDevice => f.write_str("immutable texture upload used a foreign device"),
            Self::Native {
                backend,
                stage,
                reason,
            } => write!(
                f,
                "{backend:?} immutable texture upload {stage:?} failed: {reason}"
            ),
        }
    }
}

impl std::error::Error for TextureUploadError {}

/// An immutable buffer upload accepted by the native queue but not yet finalized.
///
/// This value retains the destination and the accepted submission. Dropping it
/// never releases potentially in-flight native storage early: the private
/// completion owns the staging allocation and applies the accepted-unknown
/// quarantine contract when completion cannot be proven.
pub struct PendingBufferUpload {
    pub(in crate::resource) buffer: Option<Buffer>,
    pub(in crate::resource) completion: NativeCompletion,
    pub(in crate::resource) backend: Backend,
}

/// A non-complete immutable upload returned by [`PendingBufferUpload::finalize`].
pub struct IncompleteBufferUpload {
    pub(in crate::resource) upload: PendingBufferUpload,
    pub(in crate::resource) status: CompletionStatus,
}

/// An immutable buffer whose initial contents are available to later GPU work.
///
/// Its only published incoming/outgoing state is [`ResourceAccessState::CopyDestination`].
#[derive(Clone)]
pub struct UploadedBuffer {
    pub(in crate::resource) buffer: Buffer,
}

/// An immutable texture upload accepted by the native queue but not yet finalized.
pub struct PendingTextureUpload {
    pub(in crate::resource) texture: Option<Texture>,
    pub(in crate::resource) completion: NativeCompletion,
    pub(in crate::resource) backend: Backend,
}

/// A non-complete texture upload returned by [`PendingTextureUpload::finalize`].
pub struct IncompleteTextureUpload {
    pub(in crate::resource) upload: PendingTextureUpload,
    pub(in crate::resource) status: CompletionStatus,
}

/// An immutable RGBA8 texture whose contents are available to later GPU work.
#[derive(Clone)]
pub struct UploadedTexture {
    pub(in crate::resource) texture: Texture,
}
