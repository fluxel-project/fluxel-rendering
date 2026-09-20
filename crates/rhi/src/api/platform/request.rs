//! Device-request descriptor (specification section 5.8).
//!
//! Requesting a device is asynchronous on every platform this crate serves:
//! The public request operation is [`super::PlatformProvider::request_device`],
//! an `async fn`. The former public polling protocol is intentionally absent:
//! callers await the result rather than
//! driving a backend-specific state machine.

use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::requirements::DeviceRequirements;
use crate::api::presentation::PresentationTarget;
use crate::api::resource::MemoryPolicy;

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
    memory_policy: MemoryPolicy,
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
            memory_policy: MemoryPolicy::Automatic,
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

    /// Supplies an allocator strategy hint for the requested device.
    pub fn with_memory_policy(mut self, policy: MemoryPolicy) -> Self {
        self.memory_policy = policy;
        self
    }

    /// The requested allocator strategy hint.
    pub fn memory_policy(&self) -> MemoryPolicy {
        self.memory_policy
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
