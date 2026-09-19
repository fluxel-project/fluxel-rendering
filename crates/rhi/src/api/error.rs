//! The portable error model (specification section 4).
//!
//! Section 4 fixes both the vocabulary and the mapping from a discovered problem
//! to a kind:
//!
//! ```text
//! There is no Adapter/Device that meets the requirements -> NoSuitableAdapter
//! unsupported device/format/route                        -> Unsupported
//! Cross Device                                           -> WrongDevice
//! binding/pipeline interface mismatch                    -> IncompatibleInterface
//! Missing GPU happens-before dependency                  -> MissingDependency
//! range/alignment/usage error                            -> InvalidUsage
//! OOM                                                    -> OutOfMemory
//! device lost                                            -> DeviceLost
//! ```
//!
//! The closing rule of that section is the one that shapes this crate's
//! layering: *"Problems that have been discovered by portable validation are not
//! allowed to be deliberately sent to the backend and then relied on driver
//! validation."* A backend that returns [`RhiErrorKind::InvalidUsage`] has
//! therefore found a gap in the portable layer above it, not a normal outcome —
//! see the device façade for where that decision is required to be made.
//!
//! `RhiErrorKind` is `#[non_exhaustive]`: new kinds are added when a new
//! capability family is frozen, and callers must keep a wildcard arm. The
//! variants that exist are the eleven section 4 names and nothing else.

use core::fmt;

use super::identity::ObjectId;

/// The portable classification of a refused RHI operation.
///
/// The kind is the part of an error a caller may branch on; the message is
/// diagnostic text and is explicitly not stable. See the module documentation
/// for the mapping rules that decide which kind a given problem receives.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RhiErrorKind {
    /// A range, alignment, usage, or descriptor constraint was violated.
    ///
    /// This is the kind portable validation is expected to produce. A backend
    /// that produces it has discovered a constraint the portable layer should
    /// already have refused.
    InvalidUsage,
    /// The current provider cannot find an adapter or device satisfying the
    /// request.
    NoSuitableAdapter,
    /// The device, format, route, or feature is not supported.
    ///
    /// An unsupported route is always this kind; section 9.4 forbids a backend
    /// from silently inserting a shader or CPU fallback instead.
    Unsupported,
    /// A binding, pipeline-interface, or shader-stage interface mismatch.
    IncompatibleInterface,
    /// A required GPU happens-before dependency is absent.
    ///
    /// Section 40.4 raises this for an unordered overlapping-write hazard inside
    /// one plan, and section 41.4 for one across pending plans.
    MissingDependency,
    /// An object was used with a device other than the one that created it.
    ///
    /// Section 3.3 makes this the *only* answer for cross-device use in P0:
    /// there is no implicit copy, binding, handle unwrap, staging bridge, or
    /// peer transfer to fall back on.
    WrongDevice,
    /// Allocation failed.
    OutOfMemory,
    /// A presentation target's configuration no longer matches its surface.
    TargetOutdated,
    /// The presentation target was lost.
    TargetLost,
    /// The device was lost. Loss is terminal (section 3.1).
    DeviceLost,
    /// The backend failed for a reason with no more precise portable kind.
    BackendFailure,
}

impl RhiErrorKind {
    /// Returns a short stable name for logs and test assertions.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidUsage => "InvalidUsage",
            Self::NoSuitableAdapter => "NoSuitableAdapter",
            Self::Unsupported => "Unsupported",
            Self::IncompatibleInterface => "IncompatibleInterface",
            Self::MissingDependency => "MissingDependency",
            Self::WrongDevice => "WrongDevice",
            Self::OutOfMemory => "OutOfMemory",
            Self::TargetOutdated => "TargetOutdated",
            Self::TargetLost => "TargetLost",
            Self::DeviceLost => "DeviceLost",
            Self::BackendFailure => "BackendFailure",
            // `#[non_exhaustive]` is for downstream crates; inside this crate
            // every kind is named, so a new variant is a compile error here
            // until it is given a name. That is the intended pressure.
        }
    }
}

impl fmt::Display for RhiErrorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A refused RHI operation.
///
/// Carries the portable [`RhiErrorKind`] a caller may branch on, diagnostic text
/// that is not stable, the [`ObjectId`] of the object the refusal concerns when
/// there is exactly one, and the name of the operation that refused.
///
/// The operation name is `&'static str` rather than a `String` on purpose: it
/// names a compile-time call site such as `"Device::create_buffer"`, so it costs
/// no allocation on an error path that may run per command, and it cannot drift
/// into carrying caller data.
#[derive(Debug)]
pub struct RhiError {
    kind: RhiErrorKind,
    message: String,
    object: Option<ObjectId>,
    operation: Option<&'static str>,
}

impl RhiError {
    /// Returns the portable classification.
    pub fn kind(&self) -> RhiErrorKind {
        self.kind
    }

    /// Returns the diagnostic message. The text is not stable.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the object this refusal concerns, when there is exactly one.
    pub fn object(&self) -> Option<ObjectId> {
        self.object
    }

    /// Returns the name of the operation that refused.
    pub fn operation(&self) -> Option<&'static str> {
        self.operation
    }

    /// Builds an error with no object and no operation attached.
    ///
    /// Crate-private: a caller may not mint an error, for the same reason it may
    /// not mint an identity token. The façade and the backends attach the
    /// operation with [`RhiError::at`] as the error crosses each layer, so the
    /// outermost name a caller sees is the one they called.
    pub(crate) fn new(kind: RhiErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            object: None,
            operation: None,
        }
    }

    /// Attaches the object this refusal concerns.
    pub(crate) fn with_object(mut self, object: ObjectId) -> Self {
        self.object = Some(object);
        self
    }

    /// Attaches the name of the operation that refused, unless one is present.
    ///
    /// The first name attached wins so that an error raised deep in a backend
    /// keeps its most specific call site; layers above add context only when the
    /// error crossed them without one.
    pub(crate) fn at(mut self, operation: &'static str) -> Self {
        if self.operation.is_none() {
            self.operation = Some(operation);
        }
        self
    }
}

impl fmt::Display for RhiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.kind, self.message)?;
        if let Some(operation) = self.operation {
            write!(formatter, " (at {operation})")?;
        }
        if let Some(object) = self.object {
            write!(formatter, " [object {}]", object.as_u64())?;
        }
        Ok(())
    }
}

/// `RhiError` is a `std::error::Error`.
///
/// Section 4 does not list this impl, but it adds no vocabulary: it is what
/// makes the error usable from `?` in a caller whose own error type is boxed,
/// which every consumer of a fallible device verb needs. It carries no `source`,
/// because section 4 defines the kind as the portable classification rather than
/// a wrapper around a backend error.
impl std::error::Error for RhiError {}

/// The result of an RHI operation.
pub type RhiResult<T> = Result<T, RhiError>;
