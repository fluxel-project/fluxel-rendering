//! Why the Direct3D 12 spine could not lower or observe a plan.
//!
//! Three variants, mirroring [`ffi::NativeFailure`]'s split: one is a fact about
//! what this backend has built, one is a fact about how long the GPU took, and
//! one is a fact about the driver. Only the third can end the device, which is
//! why the distinction survives to this type rather than being flattened into an
//! [`RhiError`] where the failure is produced.
//!
//! # Why this is not a type inside `spine`
//!
//! Every lowering module in the chapter returns it — `copy` and `transfer` as
//! well as `spine` — and
//! [`crate::backend::dx12::platform::provider`] consumes it to build a
//! device-loss summary. A failure vocabulary that the whole chapter and the
//! platform chapter both agree on is a contract, and `CLAUDE.md` section 8 keeps
//! contracts out of whichever module happens to use them most.

use crate::api::error::{RhiError, RhiErrorKind};
use crate::backend::dx12::ffi;

/// Why the spine could not lower or observe a plan.
///
/// Three variants, mirroring [`ffi::NativeFailure`]'s split: one is a fact about
/// what this backend has built, one is a fact about how long the GPU took, and
/// one is a fact about the driver. Only the third can end the device, which is
/// why the distinction survives to this type rather than being flattened into an
/// [`RhiError`] here.
pub(crate) enum SpineFailure {
    /// The recording names something this spine has no lowering for.
    ///
    /// Reported as [`RhiErrorKind::Unsupported`] rather than as a silent skip:
    /// section 9.4 forbids substituting a path for one that does not exist, and
    /// a batch whose raster work was quietly dropped would execute as a
    /// copy-only plan while the caller believed it had drawn something.
    Unsupported {
        /// What the recording asked for, for the refusal's first clause.
        what: &'static str,
        /// Why this spine does not lower it.
        why: &'static str,
    },
    /// The GPU did not reach the last submitted serial inside the bound.
    ///
    /// Only [`crate::backend::dx12::command::Dx12CommandSpine::wait_idle`]
    /// produces this. It is deliberately not terminal: a GPU that is merely slow
    /// and a GPU that has hung are indistinguishable from here, and
    /// [`ffi::NativeFailure`] already records why treating a hung device as alive
    /// is the cheaper of the two mistakes.
    Stalled {
        /// The bound that expired, so the message states what was waited for.
        bound_ms: u32,
    },
    /// A Direct3D 12 call failed.
    Native(ffi::NativeError),
}

impl SpineFailure {
    /// Whether this failure ended the device.
    ///
    /// Neither `Unsupported` nor `Stalled` does. This backend not having built a
    /// lowering says nothing about the driver, and a slow frame is not a dead
    /// device; marking a healthy device lost on either would retire a usable
    /// device on a transient fact.
    pub(crate) fn is_terminal(&self) -> bool {
        match self {
            Self::Unsupported { .. } | Self::Stalled { .. } => false,
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
            Self::Native(native) => native.as_error().to_string(),
        }
    }

    /// Converts into the portable error a caller sees.
    pub(crate) fn into_rhi(self) -> RhiError {
        match self {
            Self::Unsupported { what, why } => {
                RhiError::new(RhiErrorKind::Unsupported, format!("{what}: {why}"))
            }
            Self::Stalled { bound_ms } => RhiError::new(
                RhiErrorKind::BackendFailure,
                format!("the GPU did not reach the last submitted serial within {bound_ms} ms"),
            ),
            Self::Native(native) => return native.into_rhi(),
        }
        .at("Dx12Device::submit")
    }
}

/// Builds a [`ffi::NativeError`] naming the submission path.
///
/// A free function rather than a closure at each `map_err`, because the operation
/// tag must be the same string at every one of them and a closure would have to
/// be re-typed to stay identical.
pub(super) fn ref_native(error: &windows::core::Error) -> SpineFailure {
    SpineFailure::Native(ffi::NativeError::new(error, "Dx12Device::submit"))
}
