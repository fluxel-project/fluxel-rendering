//! The Layer 2 transition error: what failed, in which domain, and how far the
//! group had already been applied.
//!
//! Layer 2 does not invent a second taxonomy for driver failures.  A backend
//! refusal is carried as the backend's own [`GlError`], because a caller that
//! needs to distinguish "this profile lacks the feature" from "the driver
//! rejected the call" must be able to do so without matching on prose.  What
//! Layer 2 adds is the two facts the backend cannot know: which state domain
//! the failure left unknown, and how much of that domain's group had already
//! been emitted.
//!
//! # Why a partial application is recorded rather than described
//!
//! A failed group is marked entirely unknown and is re-applied in full on the
//! next reconcile, so the count is not needed to *repair* the group.  It is
//! recorded because it is the one fact that distinguishes "the driver rejected
//! the first call of this group" from "the driver rejected the seventeenth",
//! and those two look identical in every other part of the error.

use super::knowledge::StateDomain;
use crate::webgl2::api::GlError;

/// How much of one state group had been emitted when a transition failed.
///
/// A caller must not read `emitted` as a count of successful calls: a domain
/// applies its group in a documented order and stops at the first failure, so
/// every call before the failing one succeeded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PartialApplication {
    /// Calls emitted before the failure was reported.
    pub emitted: u32,
    /// Calls this domain intended to emit for the whole group.
    pub expected: u32,
}

impl PartialApplication {
    /// No call was emitted, so no driver state changed.
    pub(crate) const NONE: Self = Self {
        emitted: 0,
        expected: 0,
    };

    /// A group of `expected` calls of which `emitted` were issued.
    pub(crate) const fn new(emitted: u32, expected: u32) -> Self {
        Self { emitted, expected }
    }
}

/// A refused or failed Layer 2 transition.
///
/// The three variants have disjoint producers, which is what makes the error
/// actionable: `Rejected` and `Unsupported` are decided by Layer 2 before any
/// side effect, and `Backend` is reported by the backend after one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum StateError {
    /// Layer 2 refused the request: its own precondition or the caller's input
    /// is invalid.  No backend call was made, so no state changed.
    Rejected {
        /// Domain whose group the request would have changed.
        domain: StateDomain,
        /// The logical operation the caller asked for.
        operation: &'static str,
        /// What was wrong, in terms a caller can act on.
        message: String,
    },
    /// The resolved capability facts do not support the request.
    ///
    /// Raised before the request creates an object, acquires an extension, or
    /// emits a command, so an unsupported optional domain costs the caller
    /// nothing but the error.
    Unsupported {
        /// Domain whose group the request would have changed.
        domain: StateDomain,
        /// The logical operation the caller asked for.
        operation: &'static str,
        /// The capability or extension fact that is missing.
        reason: &'static str,
    },
    /// A backend call failed.  The whole group is unknown afterwards.
    Backend {
        /// Domain whose group is now unknown.
        domain: StateDomain,
        /// The logical operation the caller asked for.
        operation: &'static str,
        /// How far the group had been applied.
        applied: PartialApplication,
        /// The backend's own structured failure.
        source: GlError,
    },
}

impl StateError {
    /// The domain whose group this failure governs.
    pub(crate) const fn domain(&self) -> StateDomain {
        match self {
            Self::Rejected { domain, .. }
            | Self::Unsupported { domain, .. }
            | Self::Backend { domain, .. } => *domain,
        }
    }

    /// The logical operation the caller asked for.
    pub(crate) const fn operation(&self) -> &'static str {
        match self {
            Self::Rejected { operation, .. }
            | Self::Unsupported { operation, .. }
            | Self::Backend { operation, .. } => operation,
        }
    }

    /// Whether this failure happened before any backend call.
    ///
    /// This is the property a differential test keys on: a pre-side-effect
    /// rejection must leave the backend's call trace untouched, and only this
    /// predicate can tell the two rejection paths from the failure path without
    /// inspecting the backend.
    pub(crate) const fn is_pre_side_effect(&self) -> bool {
        !matches!(self, Self::Backend { .. })
    }

    /// Layer 2 refused the caller's request.
    pub(crate) fn rejected(
        domain: StateDomain,
        operation: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self::Rejected {
            domain,
            operation,
            message: message.into(),
        }
    }

    /// The capability facts do not support the request.
    pub(crate) const fn unsupported(
        domain: StateDomain,
        operation: &'static str,
        reason: &'static str,
    ) -> Self {
        Self::Unsupported {
            domain,
            operation,
            reason,
        }
    }

    /// A backend call failed after `applied` of the group's `expected` calls.
    pub(crate) const fn backend(
        domain: StateDomain,
        operation: &'static str,
        applied: PartialApplication,
        source: GlError,
    ) -> Self {
        Self::Backend {
            domain,
            operation,
            applied,
            source,
        }
    }
}
