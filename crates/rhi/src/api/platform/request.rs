//! Asking for a device, and waiting for the answer (specification section 5.8
//! and 5.9).
//!
//! Requesting a device is asynchronous on every platform this crate serves:
//! WebGPU resolves an adapter and then a device over several turns, and the
//! native backends may create or wrap a logical device behind a driver call that
//! is not instant. This module owns that shape — the descriptor that states what
//! is wanted, and the handle that reports when it is ready.
//!
//! It deliberately does not bind to an async runtime. Section 5.9 rules out
//! Tokio, async-std, `async_trait`, and the JS Promise ABI alike: the host owns
//! its event loop, and [`DeviceRequest::poll`] only advances bookkeeping the RHI
//! itself controls.

use std::sync::Arc;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::platform::device::Device;
use crate::api::platform::provider::{AdapterSelection, PlatformProvider};
use crate::api::platform::requirements::DeviceRequirements;
use crate::api::presentation::PresentationTarget;
use crate::base::platform::{DeviceRequestBackend, RequestProgress};

/// A request for a device, before any adapter has been chosen.
///
/// Presentation belongs here rather than in a separate preflight. Section 5.8
/// makes the point directly: asking whether an adapter *can* present to a surface
/// is not enough, because device creation is what selects the queue family, the
/// execution and presentation route, and the backend-specific presentation
/// support. A target that must be served has to be visible to creation, so it
/// travels in this descriptor.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DeviceRequestDescriptor {
    selection: AdapterSelection,
    requirements: DeviceRequirements,
    presentation_targets: Vec<PresentationTarget>,
}

impl DeviceRequestDescriptor {
    /// Requests a device chosen by `selection` that satisfies `requirements`.
    ///
    /// The result is headless until a presentation target is added: an empty
    /// target list is the compute-only, offscreen-only case, not an error.
    pub fn new(selection: AdapterSelection, requirements: DeviceRequirements) -> Self {
        Self {
            selection,
            requirements,
            presentation_targets: Vec::new(),
        }
    }

    /// Requires the final device to have a portable presentation route to
    /// `target`.
    ///
    /// The concrete surface facts — format, present mode, extent — are still
    /// queried through the presentation module; this states only that the device
    /// must be able to serve the target at all.
    pub fn require_presentation_target(mut self, target: PresentationTarget) -> Self {
        self.presentation_targets.push(target);
        self
    }

    /// How the adapter is to be chosen.
    pub fn selection(&self) -> AdapterSelection {
        self.selection
    }

    /// What the resulting device must satisfy.
    pub fn requirements(&self) -> &DeviceRequirements {
        &self.requirements
    }

    /// The targets the resulting device must be able to present to.
    pub fn presentation_targets(&self) -> &[PresentationTarget] {
        &self.presentation_targets
    }
}

/// The progress of a [`DeviceRequest`].
///
/// Two states and no error variant: a request that fails reports it through the
/// [`RhiResult`] of the `poll` call that discovered the failure, so the failure
/// carries the structured [`RhiError`] rather than being folded into the status.
#[derive(Debug)]
pub enum RequestStatus<T> {
    /// The request is still in progress; poll again later.
    Pending,
    /// The request completed, yielding its value.
    Ready(T),
}

/// An in-flight request for a device.
///
/// Single-shot by contract (section 5.9):
///
/// ```text
/// Pending -> Ready(Device)
/// Pending -> Err(..)
/// ```
///
/// Once the first `Ready` or terminal error has been returned the request is
/// complete, and a further [`Self::poll`] is [`RhiErrorKind::InvalidUsage`].
/// Dropping the request means the caller has abandoned the result: a backend may
/// then cancel or complete the underlying work, but must never hand a caller a
/// half-initialized device.
pub struct DeviceRequest {
    /// Set once `Ready` or a terminal error has been reported.
    complete: bool,
    /// The provider that opened this request.
    ///
    /// Held so that the request can mint the identity of the device it produces.
    /// Section 6.1 ties identity minting to a completed request, and the provider
    /// is where the generation counter for this instance lives — a backend that
    /// composed its own identity could hand two domains the same one, or revive
    /// an old one by choosing a generation, and section 3.1 lists both under
    /// "P0 None".
    ///
    /// It also keeps the native instance alive for as long as the request is, so
    /// a request cannot outlive the provider that would have produced its device.
    provider: PlatformProvider,
    /// The request's native side.
    ///
    /// Boxed rather than shared because section 5.9 makes a request single-shot:
    /// one owner is the shape that enforces it, and the backend never needs to be
    /// reached from anywhere else.
    native: Box<dyn DeviceRequestBackend>,
}

impl DeviceRequest {
    /// Starts an in-flight request.
    ///
    /// Crate-private: a request is produced by
    /// [`crate::api::platform::PlatformProvider::request_device`], which is what
    /// gives it a provider to advance against.
    pub(crate) fn new(provider: PlatformProvider, native: Box<dyn DeviceRequestBackend>) -> Self {
        Self {
            complete: false,
            provider,
            native,
        }
    }

    /// Non-blockingly observes or advances the request.
    ///
    /// This does not pump the browser or operating-system event loop. Section
    /// 5.9 requires the host to keep running its own loop normally; a host that
    /// stopped pumping in order to poll would deadlock the very request it is
    /// waiting on.
    pub fn poll(&mut self) -> RhiResult<RequestStatus<Device>> {
        if self.complete {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "device request has already completed",
            )
            .at("DeviceRequest::poll"));
        }

        match self.native.poll() {
            Ok(RequestProgress::Pending) => Ok(RequestStatus::Pending),
            Ok(RequestProgress::Ready(native)) => {
                // Retire the request before composing the device: the request's
                // own rule is that the first `Ready` ends it, so a panic or an
                // early return between here and the answer must not leave it
                // looking in-flight.
                self.complete = true;
                let identity = self.provider.mint_identity();
                Ok(RequestStatus::Ready(Device::new(
                    identity,
                    Arc::from(native),
                )))
            }
            Err(error) => {
                // Section 5.9's diagram has exactly two terminal outcomes, so an
                // error ends the request as surely as a device does. Leaving it
                // pending on error would invite a caller to poll a request whose
                // backend has already given up.
                self.complete = true;
                Err(error)
            }
        }
    }

    /// Marks the request finished.
    ///
    /// Crate-private: only the code that reports `Ready` or a terminal error may
    /// retire the request, which is what keeps a second `poll` from appearing to
    /// still be in flight. [`Self::poll`] is now that code; this remains part of
    /// the frozen surface because it is how a contract test reaches the
    /// already-completed state without a backend that can produce one.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "called by the contract tests; `poll` retires the request itself once a backend reports its outcome"
        )
    )]
    pub(crate) fn mark_complete(&mut self) {
        self.complete = true;
    }
}
