//! Render scene validation failures.

use std::fmt;

/// A structured scene validation refusal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SceneError {
    /// A transform contains a non-finite component.
    InvalidTransform,
    /// Bounds are non-finite or inverted.
    InvalidBounds,
    /// A view has an empty extent.
    InvalidView,
    /// Mesh visibility is outside the finite range from zero to one.
    InvalidMeshVisibility,
    /// Directional light inputs cannot be represented as shader data.
    InvalidDirectionalLight,
    /// An ID came from another scene.
    ForeignId {
        /// Index of the rejected operation in the submitted batch.
        operation: usize,
    },
    /// An ID was removed, reused, or was never issued.
    StaleId {
        /// Index of the rejected operation in the submitted batch.
        operation: usize,
    },
    /// A later operation writes an ID already written by the batch.
    ConflictingChange {
        /// Index of the later conflicting operation in the submitted batch.
        operation: usize,
    },
    /// A batch exceeds record capacity.
    RecordCapacityExceeded {
        /// Index of the operation that would exceed capacity.
        operation: usize,
    },
    /// A batch exceeds view capacity.
    ViewCapacityExceeded {
        /// Index of the operation that would exceed capacity.
        operation: usize,
    },
    /// A generation cannot advance without wrapping.
    IdentityExhausted {
        /// Index of the operation that would require generation wraparound.
        operation: usize,
    },
    /// The process-local scene identity source is exhausted.
    SceneIdentityExhausted,
    /// The scene cannot issue another feature issuer token.
    FeatureIssuerExhausted,
    /// A nonempty batch cannot be assigned a later snapshot version.
    SnapshotVersionExhausted,
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransform => f.write_str("render transform contains a non-finite value"),
            Self::InvalidBounds => f.write_str("render bounds are non-finite or inverted"),
            Self::InvalidView => f.write_str("render view has an empty extent"),
            Self::InvalidMeshVisibility => {
                f.write_str("mesh visibility must be between zero and one")
            }
            Self::InvalidDirectionalLight => f.write_str("invalid directional light inputs"),
            Self::ForeignId { operation } => {
                write!(f, "change {operation} names an ID from another scene")
            }
            Self::StaleId { operation } => write!(f, "change {operation} names a stale ID"),
            Self::ConflictingChange { operation } => {
                write!(f, "change {operation} conflicts with an earlier change")
            }
            Self::RecordCapacityExceeded { operation } => {
                write!(f, "change {operation} exceeds record capacity")
            }
            Self::ViewCapacityExceeded { operation } => {
                write!(f, "change {operation} exceeds view capacity")
            }
            Self::IdentityExhausted { operation } => {
                write!(f, "change {operation} exhausts an identity generation")
            }
            Self::SceneIdentityExhausted => {
                f.write_str("render scene identity source is exhausted")
            }
            Self::FeatureIssuerExhausted => {
                f.write_str("render scene feature issuer source is exhausted")
            }
            Self::SnapshotVersionExhausted => {
                f.write_str("render scene snapshot versions are exhausted")
            }
        }
    }
}

impl std::error::Error for SceneError {}
