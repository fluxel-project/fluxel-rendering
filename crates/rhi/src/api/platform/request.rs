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

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::platform::device::Device;
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::requirements::DeviceRequirements;
use crate::api::presentation::PresentationTarget;

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
}

impl DeviceRequest {
    /// Starts an in-flight request.
    ///
    /// Crate-private: a request is produced by
    /// [`crate::api::platform::PlatformProvider::request_device`], which is what
    /// gives it a provider to advance against.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "called by the contract tests; the platform layer that starts a request is not written"
        )
    )]
    pub(crate) fn new() -> Self {
        Self { complete: false }
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
        unimplemented!(
            "advancing the request arrives with the backend port; the contract \
             is fixed, the stages are not built"
        )
    }

    /// Marks the request finished.
    ///
    /// Crate-private: only the code that reports `Ready` or a terminal error may
    /// retire the request, which is what keeps a second `poll` from appearing to
    /// still be in flight.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "called by the contract tests; the backend port that reports the request's                       outcome is not written"
        )
    )]
    pub(crate) fn mark_complete(&mut self) {
        self.complete = true;
    }
}
