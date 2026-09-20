//! Why this backend could not lower a request.
//!
//! Variants preserve whether a refusal is about lowering, bounded waiting, an
//! already-observed device loss, or a native call. This distinction is what lets
//! the device layer terminate the execution domain without treating ordinary
//! unsupported work or timeout as loss.
//!
//! # Why this sits at the chapter root, beside `ffi`
//!
//! Every lowering in this backend returns it: `command`'s four step modules, the
//! descriptor writes in [`crate::backend::dx12::binding`], and the root-signature
//! and state-object builds in [`crate::backend::dx12::pipeline`]. It was written
//! inside `command` when the command spine was the only lowering there was, and it
//! moved here when the second and third chapters arrived — a vocabulary three
//! sibling chapters and the platform chapter all agree on is a contract, and
//! `CLAUDE.md` section 8 keeps contracts out of whichever module happens to use
//! them most. The move is what keeps the dependency one-directional: `command`
//! lowers dispatches *through* `binding`, so `binding` importing a type from
//! `command` would have pointed the arrow backwards.
//!
//! [`ffi`] sits beside it for the same reason and owns the other half: this type
//! says *which kind* of failure it is, and `ffi` is what decides that from a
//! `HRESULT`.

use crate::api::error::{RhiError, RhiErrorKind};
use crate::backend::dx12::ffi;

/// Why the backend could not lower or observe a request.
///
/// The variants keep backend support, bounded waiting, an already-published loss,
/// and a newly returned native failure distinct. Only the latter two can end the
/// device identity.
pub(crate) enum Dx12Failure {
    /// The request names something this backend has no lowering for.
    ///
    /// Reported as [`RhiErrorKind::Unsupported`] rather than as a silent skip:
    /// section 9.4 forbids substituting a path for one that does not exist, and
    /// a batch whose raster work was quietly dropped would execute as a
    /// copy-only plan while the caller believed it had drawn something.
    Unsupported {
        /// What the request asked for, for the refusal's first clause.
        what: &'static str,
        /// Why this backend does not lower it.
        why: &'static str,
    },
    /// The GPU did not reach the last submitted serial inside the bound.
    ///
    /// Only [`crate::backend::dx12::command::Dx12CommandSpine::wait_idle`]
    /// produces this. It is deliberately not terminal: a GPU that is merely slow
    /// and a GPU that has hung are indistinguishable from here until a native
    /// terminal HRESULT is observed.
    Stalled {
        /// The bound that expired, so the message states what was waited for.
        bound_ms: u32,
    },
    /// A helper below a native boundary already observed and published loss.
    DeviceLost {
        /// Stable diagnostic captured by the first observer.
        reason: String,
    },
    /// A Direct3D 12 call failed.
    Native(ffi::NativeError),
}

impl Dx12Failure {
    /// Whether this failure ended the device.
    ///
    /// Neither `Unsupported` nor `Stalled` does. This backend not having built a
    /// lowering says nothing about the driver, and a slow frame is not a dead
    /// device; marking a healthy device lost on either would retire a usable
    /// device on a transient fact.
    pub(crate) fn is_terminal(&self) -> bool {
        match self {
            Self::Unsupported { .. } | Self::Stalled { .. } => false,
            Self::DeviceLost { .. } => true,
            Self::Native(native) => native.failure().is_terminal(),
        }
    }

    /// The sentence this failure reports, without its operation tag.
    ///
    /// Read by [`crate::backend::dx12::platform::provider`], which builds a
    /// device-loss summary from it and renders it into the
    /// [`CompletionFailure`](crate::api::submission::CompletionFailure) a
    /// permanently unobservable serial answers with.
    pub(crate) fn message(&self) -> String {
        match self {
            Self::Unsupported { what, why } => format!("{what}: {why}"),
            Self::Stalled { bound_ms } => {
                format!("the GPU did not reach the last submitted serial within {bound_ms} ms")
            }
            Self::DeviceLost { reason } => reason.clone(),
            Self::Native(native) => native.as_error().to_string(),
        }
    }

    /// Converts into the portable error a caller sees, tagged with `operation`.
    ///
    /// The tag is a parameter rather than a constant because the same vocabulary
    /// now reaches a caller through four different device verbs — `submit`,
    /// `create_bind_group`, `create_compute_pipeline` and `wait_idle` — and a
    /// failure tagged `Dx12Device::submit` when it came from binding would send
    /// whoever reads it to the wrong place. Every caller passes its own name.
    pub(crate) fn into_rhi(self, operation: &'static str) -> RhiError {
        match self {
            Self::Unsupported { what, why } => {
                RhiError::new(RhiErrorKind::Unsupported, format!("{what}: {why}"))
            }
            Self::Stalled { bound_ms } => RhiError::new(
                RhiErrorKind::BackendFailure,
                format!("the GPU did not reach the last submitted serial within {bound_ms} ms"),
            ),
            Self::DeviceLost { reason } => RhiError::new(RhiErrorKind::DeviceLost, reason),
            Self::Native(native) => return native.into_rhi(),
        }
        .at(operation)
    }
}

/// Builds a [`ffi::NativeError`] naming the submission path.
///
/// A free function rather than a closure at each `map_err`, because the operation
/// tag must be the same string at every one of them and a closure would have to
/// be re-typed to stay identical. It is the *command* path's tag specifically —
/// the other chapters name themselves at their own call sites — and that is the
/// whole of why it is `pub(super)`, which in a file at the chapter root means
/// "throughout `crate::backend::dx12`", rather than `pub(crate)`.
pub(super) fn ref_native(error: &windows::core::Error) -> Dx12Failure {
    Dx12Failure::Native(ffi::NativeError::new(error, "Dx12Device::submit"))
}
