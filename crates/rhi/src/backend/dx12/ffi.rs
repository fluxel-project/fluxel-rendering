//! The narrow Direct3D 12 boundary: return-code classification and the two
//! native readings that have no portable spelling.
//!
//! This is the only module in the DX12 backend where a `HRESULT` becomes an API
//! v1 error, and for that reason it is also the only module that decides whether
//! a failure ends the device's identity. Two callers want those two answers
//! separately — the device layer must set a terminal status *and* return an
//! error, while a resource allocation must return an error and leave the device
//! running — so classification and error construction are two functions rather
//! than one.
//!
//! # Why Direct3D 12 learns about loss late
//!
//! There is no device-loss callback and no `VK_ERROR_DEVICE_LOST`. A removed
//! device reports `DXGI_ERROR_DEVICE_REMOVED` from the next call that touches
//! it, and the call that first reports it is arbitrary. That is why nothing here
//! caches or pre-checks a liveness flag: a cached "the device is fine" is a
//! second source of truth that is wrong for exactly the window in which it
//! matters.
//!
//! # Safety
//!
//! The `windows` crate supplies these declarations and its COM wrappers, so no
//! vtable is hand-built here and no pointer arithmetic appears in this module.
//! The `unsafe` that a Direct3D 12 backend cannot avoid — `D3D12CreateDevice`'s
//! out-parameter, mapping a resource, executing a command list — lives at the
//! call sites that own the object being passed, each with its own written
//! rationale. This module holds none of it.
//!
//! # Test reach, stated rather than assumed
//!
//! The tests below are pure logic — a return code maps to a classification, a
//! UTF-16 buffer becomes a `String` — and nothing in them needs a GPU. They still
//! only run on Windows, because they construct `windows::core::Error` and read
//! `DXGI_ERROR_*` out of the binding crate, which does not exist elsewhere. That
//! is a real gap against the cross-platform conformance suite `version-plan.md`
//! section 4 asks for, recorded here instead of left to be discovered: a Linux CI
//! row would silently run none of it. Closing it means classifying a raw `i32`
//! against locally declared codes, which trades the binding crate's authority for
//! reach, and that trade has not been made yet.

use windows::Win32::Foundation::E_OUTOFMEMORY;
use windows::Win32::Graphics::Dxgi::{
    DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET, DXGI_ERROR_DRIVER_INTERNAL_ERROR,
};
use windows::core::Error as WinError;

use crate::api::error::{RhiError, RhiErrorKind};

/// What one native failure means for the device that produced it.
///
/// The distinction is not cosmetic: [`NativeFailure::Terminal`] ends a device
/// identity and every later call on that identity must be refused, while
/// [`NativeFailure::Refused`] leaves the device running and the caller free to
/// try something else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeFailure {
    /// The device is gone. Its identity is terminal; a working device needs a
    /// new request and a new identity.
    Terminal,

    /// The device is alive, but this allocation could not be satisfied.
    OutOfMemory,

    /// The device is alive and declined the request on its merits.
    Refused,
}

impl NativeFailure {
    /// Classifies one Direct3D 12 return code.
    ///
    /// `DXGI_ERROR_DEVICE_HUNG` is deliberately not terminal. A hung device
    /// that the driver recovers from continues to serve calls, and reporting it
    /// as terminal would retire a device that is still usable — the expensive
    /// direction of the mistake. It is reported as [`NativeFailure::Refused`]
    /// with its own code preserved in the message, so a caller that does know
    /// the device is finished can still see what happened.
    pub(super) fn classify(error: &WinError) -> Self {
        match error.code() {
            DXGI_ERROR_DEVICE_REMOVED
            | DXGI_ERROR_DEVICE_RESET
            | DXGI_ERROR_DRIVER_INTERNAL_ERROR => Self::Terminal,
            E_OUTOFMEMORY => Self::OutOfMemory,
            _ => Self::Refused,
        }
    }

    /// Whether this failure ends the producing device's identity.
    ///
    /// The device layer is its caller — the one that must both set a terminal
    /// status and return the error — and it is the half of this module's split
    /// that [`to_rhi`] deliberately does not use. This used to carry an
    /// `expect(dead_code)` because that layer was not written; the buffer
    /// allocation path is now such a caller, and an expectation that is no longer
    /// fulfilled is an error, so the attribute is gone rather than left to
    /// outlive its reason.
    pub(super) fn is_terminal(self) -> bool {
        matches!(self, Self::Terminal)
    }

    /// The API v1 error kind this failure reports as.
    pub(super) fn kind(self) -> RhiErrorKind {
        match self {
            Self::Terminal => RhiErrorKind::DeviceLost,
            Self::OutOfMemory => RhiErrorKind::OutOfMemory,
            Self::Refused => RhiErrorKind::BackendFailure,
        }
    }
}

/// One native failure, classified once and not yet reported.
///
/// `to_rhi` is enough for a call site that only has to return an error, and most
/// have only that to do. The allocation path is the exception: it must *also*
/// ask whether the failure ended the device, because `DXGI_ERROR_DEVICE_REMOVED`
/// arrives from an arbitrary call rather than from a dedicated notification, so
/// a failure that reads as an ordinary refusal may be the device ending. This
/// type is how that call site reads the classification without classifying the
/// same `HRESULT` twice — or worse, recovering it from the formatted message,
/// which would make the message part of the API rather than part of the
/// diagnosis.
pub(super) struct NativeError {
    /// The API v1 error, already built so that the message is formatted in one
    /// place regardless of which of the two callers asks.
    error: RhiError,
    /// What the `HRESULT` meant for the device that produced it.
    failure: NativeFailure,
}

impl NativeError {
    /// Classifies `error` and builds the API v1 error for `operation`.
    pub(super) fn new(error: &WinError, operation: &'static str) -> Self {
        let failure = NativeFailure::classify(error);
        Self {
            // The numeric code is kept in the message: a DX12 diagnosis that has
            // thrown away the `HRESULT` has thrown away the only thing that
            // distinguishes "the driver refused this root signature" from "the
            // driver refused this heap".
            error: RhiError::new(
                failure.kind(),
                format!(
                    "Direct3D 12 call failed: {error} (HRESULT {:#010x})",
                    error.code().0
                ),
            )
            .at(operation),
            failure,
        }
    }

    /// A call that claimed success and produced nothing.
    ///
    /// `S_OK` with a null out-parameter is a driver contract violation, and there
    /// is no `HRESULT` to classify precisely because the call reported that it
    /// worked. It is therefore [`NativeFailure::Refused`] and not `Terminal`: a
    /// driver that lies about one allocation has not said the device is gone, and
    /// retiring a usable device on it would be the expensive direction of the
    /// mistake.
    pub(super) fn driver_contract_violation(what: &str, operation: &'static str) -> Self {
        Self {
            error: RhiError::new(RhiErrorKind::BackendFailure, what.to_string()).at(operation),
            failure: NativeFailure::Refused,
        }
    }

    /// What this failure meant for the device.
    pub(super) fn failure(&self) -> NativeFailure {
        self.failure
    }

    /// The API v1 error, borrowed so a caller can describe the failure before
    /// reporting it.
    pub(super) fn as_error(&self) -> &RhiError {
        &self.error
    }

    /// The API v1 error.
    pub(super) fn into_rhi(self) -> RhiError {
        self.error
    }
}

/// Turns a native failure into the API v1 error for `operation`.
///
/// A thin wrapper over [`NativeError::new`], kept because the call sites that
/// only return an error read better without naming a type they never look at.
pub(super) fn to_rhi(error: &WinError, operation: &'static str) -> RhiError {
    NativeError::new(error, operation).into_rhi()
}

/// A Direct3D 12 adapter's human-readable name.
///
/// `DXGI_ADAPTER_DESC1::Description` is a fixed 128-element UTF-16 buffer that
/// the driver fills and NUL-terminates when it fits. It is not guaranteed to be
/// terminated when the name fills the buffer exactly, so the split is taken at
/// the first NUL and the whole buffer is used when there is none — the
/// alternative, trusting the terminator, reads past the driver's text on exactly
/// the adapters with the longest names.
pub(super) fn adapter_name(description: &[u16]) -> String {
    let end = description
        .iter()
        .position(|unit| *unit == 0)
        .unwrap_or(description.len());
    String::from_utf16_lossy(&description[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::DXGI_ERROR_DEVICE_HUNG;
    use windows::core::HRESULT;

    fn error(code: i32) -> WinError {
        WinError::from_hresult(HRESULT(code))
    }

    #[test]
    fn a_removed_device_is_terminal() {
        let failure = NativeFailure::classify(&error(DXGI_ERROR_DEVICE_REMOVED.0));

        assert!(failure.is_terminal());
        assert_eq!(failure.kind(), RhiErrorKind::DeviceLost);
    }

    #[test]
    fn a_reset_or_internal_driver_failure_is_terminal() {
        assert!(NativeFailure::classify(&error(DXGI_ERROR_DEVICE_RESET.0)).is_terminal());
        assert!(NativeFailure::classify(&error(DXGI_ERROR_DRIVER_INTERNAL_ERROR.0)).is_terminal());
    }

    #[test]
    fn a_hung_device_is_not_terminal_because_the_driver_may_recover() {
        let failure = NativeFailure::classify(&error(DXGI_ERROR_DEVICE_HUNG.0));

        assert!(!failure.is_terminal());
        assert_eq!(failure.kind(), RhiErrorKind::BackendFailure);
    }

    #[test]
    fn a_failed_allocation_is_not_terminal() {
        let failure = NativeFailure::classify(&error(E_OUTOFMEMORY.0));

        assert!(!failure.is_terminal());
        assert_eq!(failure.kind(), RhiErrorKind::OutOfMemory);
    }

    #[test]
    fn an_unclassified_code_is_a_backend_failure_and_keeps_its_number() {
        let failure = NativeFailure::classify(&error(0x8000_4005u32 as i32));

        assert!(!failure.is_terminal());
        assert_eq!(failure.kind(), RhiErrorKind::BackendFailure);

        let rendered = to_rhi(&error(0x8000_4005u32 as i32), "Device::create_buffer").to_string();

        assert!(
            rendered.contains("0x80004005"),
            "the HRESULT must survive into the message: {rendered}"
        );
    }

    #[test]
    fn a_nul_terminated_adapter_name_stops_at_the_nul() {
        let mut raw = [0u16; 128];
        for (slot, unit) in raw.iter_mut().zip("Radeon 780M".encode_utf16()) {
            *slot = unit;
        }

        assert_eq!(adapter_name(&raw), "Radeon 780M");
    }

    #[test]
    fn an_adapter_name_filling_the_buffer_is_not_read_past_its_end() {
        let raw = [b'X' as u16; 128];

        let name = adapter_name(&raw);

        assert_eq!(name.len(), 128);
        assert!(name.chars().all(|character| character == 'X'));
    }
}
