//! Structured, platform-neutral GL-family errors and lifecycle states.

use super::object::{ContextStamp, OwnerThreadIdentity};

/// Lifecycle of one owned GL-family context.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GlContextLifecycle {
    /// Context exists but has not accepted work.
    Inactive,
    /// Context may accept API calls.
    Active,
    /// Surface is unavailable but the context is not lost.
    Suspended,
    /// The driver or browser invalidated the context.
    Lost,
    /// Context recreation is in progress.
    Restoring,
    /// A failed transition makes further work unsafe.
    Poisoned,
    /// Context ownership has ended permanently.
    Disposed,
}

impl GlContextLifecycle {
    /// Returns whether commands may be issued in this lifecycle state.
    pub const fn accepts_commands(self) -> bool {
        matches!(self, Self::Active)
    }
}

/// A failure which is meaningful to Layer 2 and Layer 3 without GL bindings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GlError {
    /// A profile lacks the required command domain or proven feature.
    Unsupported {
        operation: &'static str,
        reason: &'static str,
    },
    /// Logical input failed validation before a side effect.
    Validation {
        operation: &'static str,
        message: String,
    },
    /// A command was submitted for an object from another context generation.
    StaleObject {
        operation: &'static str,
        object: ContextStamp,
        current: ContextStamp,
    },
    /// A command was submitted for an object owned by another device/context.
    WrongContext {
        /// Operation which rejected the foreign object.
        operation: &'static str,
        /// Context which created the object.
        object: ContextStamp,
        /// Context which received the object.
        current: ContextStamp,
    },
    /// An owner-thread constraint was violated.
    WrongThread {
        operation: &'static str,
        expected: OwnerThreadIdentity,
        actual: OwnerThreadIdentity,
    },
    /// The context exists, but its current lifecycle cannot execute this call.
    InvalidLifecycle {
        operation: &'static str,
        lifecycle: GlContextLifecycle,
    },
    /// The context was lost.
    ContextLost { operation: &'static str },
    /// The context has been disposed.
    Disposed { operation: &'static str },
    /// A poisoned context cannot safely continue.
    Poisoned { operation: &'static str },
    /// Allocation failed in the driver or browser.
    OutOfMemory { operation: &'static str },
    /// Shader compilation failed with a complete driver log.
    Shader { stage: &'static str, log: String },
    /// Program linking or validation failed with a complete driver log.
    Program {
        operation: &'static str,
        log: String,
    },
    /// An output target was incomplete.
    IncompleteFramebuffer {
        operation: &'static str,
        status: u32,
    },
    /// A synchronization operation reached its timeout.
    Timeout { operation: &'static str },
    /// A native driver or browser returned a non-specialized error.
    Driver {
        operation: &'static str,
        message: String,
    },
}
