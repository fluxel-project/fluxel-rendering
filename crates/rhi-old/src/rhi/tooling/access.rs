//! `ToolingAccess`, its subscription, and the contract they promise
//! (rhi-design sections 53.2 and 58.2).
//!
//! # The linearization contract
//!
//! `subscribe()` has exactly one start linearization point: the successful
//! insertion of the observer into the device's observer set. It returns only
//! after that point. The returned subscription then receives every event whose
//! `SemanticEventId` was assigned after the point and before its `Drop`
//! unregistration point, exactly once and in increasing `SemanticEventId`
//! order, and it never receives an event assigned before the point.
//!
//! The mechanism that implements this — the delivery gate, the observer set,
//! and the event-identity mint, with their lock order and its proof — is the
//! `registry` submodule. This module is the surface a consumer codes against.
//!
//! # Prohibited cases
//!
//! Three cases are *prohibited* by the callback contract and are refused rather
//! than deadlocked, because a callback runs while its thread holds the delivery
//! gate:
//!
//! ```text
//! an observer dropping its own subscription from on_event
//! a callback subscribing, or emitting, re-entrantly on its own device
//! a callback panicking
//! ```
//!
//! Each is counted — see [`ToolingAccess::violations`] — because a refusal
//! nobody can observe is indistinguishable from silence.
//!
//! The rest of section 58.2 is a rule on the observer, not an invariant this
//! module can check: a callback must not re-enter a *mutating* RHI operation on
//! the same device, must not wait for GPU completion, and must not block. Only
//! the entry points this module owns can be refused, and they are. A caller that
//! blocks inside `on_event` blocks the device's event delivery on that thread,
//! which is exactly the cost the rule exists to avoid.
//!
//! A callback that panics is contained: it is counted as a violation and the
//! remaining observers of that event still receive it. An observer is
//! diagnostics, and a broken one must not abort a device operation or silence
//! the other observers.

use std::sync::Arc;

use super::super::platform::{DeviceIdentity, ObjectId, RhiError, RhiErrorKind, RhiResult};
use super::events::SemanticEvent;
use super::objects::CapturedObjectDefinition;
use super::registry::{ToolingBackend, ToolingRegistry, ToolingViolation};
use super::spi::{TOOLING_SPI_VERSION, ToolingSpiVersion};
use super::work::CapturedRecordedWork;

/// The synchronous observation callback.
///
/// The callback is a synchronous observation callback. If the observer needs to
/// retain data, it must copy or own it before returning.
///
/// The callback must not re-enter a mutating RHI API on the same `Device` or
/// `DeviceIdentity`, wait for GPU completion, or block. In particular, it must
/// not drop its own [`ToolingSubscription`] while this callback is running: the
/// registry refuses that rather than deadlocking, unregisters the subscription
/// at that instant, and counts the violation.
pub trait SemanticObserver: Send + Sync + 'static {
    /// Observes one event.
    ///
    /// Every reference in `event` is valid only until this call returns.
    fn on_event(&self, event: SemanticEvent<'_>);
}

/// What an observer did that the callback contract forbids.
///
/// The counts exist because the prohibited cases are refused rather than
/// deadlocked, and a refused case that nobody can observe is indistinguishable
/// from a silent one. They are diagnostic facts: a non-zero count means some
/// observer in this device is violating section 58.2, and the events it thinks
/// it observed are not a complete record.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToolingViolationCounts {
    /// How many times an observer dropped its own subscription from `on_event`.
    ///
    /// The drop was honoured without waiting and the observer stopped receiving
    /// events at that instant; only the waiting half of the contract was
    /// refused, because waiting for the callback that is running is exactly
    /// what the contract prohibits.
    pub self_drop_attempts: u64,

    /// How many times a callback re-entered the tooling SPI on its own device.
    ///
    /// The entry point returned [`RhiErrorKind::InvalidUsage`]; a nested event
    /// emission was dropped.
    pub reentrant_attempts: u64,

    /// How many callbacks panicked.
    ///
    /// The panic was contained so the device operation and the other observers
    /// were unaffected.
    pub callback_panics: u64,
}

/// The device-scoped tooling surface.
///
/// It is a device-scoped handle: cloning it yields another handle to the same
/// device's observer set and description service, exactly as cloning the
/// `Device` does. It is obtained from `Device::tooling`.
///
/// The surface exists so that RenderGraph diagnostics, a capture coordinator,
/// and replay tooling can observe semantics without the RHI execution API
/// growing a debug product (section 53). It is deliberately separate from the
/// ordinary public surface and versioned by [`ToolingSpiVersion`], not by the
/// crate's semver.
#[derive(Clone)]
pub struct ToolingAccess {
    identity: DeviceIdentity,
    backend: Option<Arc<dyn ToolingBackend>>,
}

impl ToolingAccess {
    /// An access over `backend`, for the device named by `identity`.
    pub(crate) fn new(identity: DeviceIdentity, backend: Option<Arc<dyn ToolingBackend>>) -> Self {
        Self { identity, backend }
    }

    /// The version of the tooling SPI this access speaks.
    ///
    /// A consumer checks this before interpreting anything, because the SPI may
    /// move inside a compatible crate release. It is not a capture artifact
    /// schema version.
    pub fn spi_version(&self) -> ToolingSpiVersion {
        TOOLING_SPI_VERSION
    }

    /// The device identity this access is scoped to.
    ///
    /// A semantic event does not repeat its device, because a subscription
    /// belongs to exactly one; this is how a consumer names that device.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.identity
    }

    /// Registers `observer` and returns its subscription.
    ///
    /// The returned subscription is the only way to unregister: dropping it is
    /// the unregistration point. It must not be dropped from inside the
    /// observer's own `on_event`; the registry refuses that rather than
    /// deadlocking.
    ///
    /// A device whose backend does not implement the tooling seam answers
    /// [`RhiErrorKind::Unsupported`] instead of accepting an observer that would
    /// never be called.
    pub fn subscribe(&self, observer: Arc<dyn SemanticObserver>) -> RhiResult<ToolingSubscription> {
        let registry = self.registry()?;
        if registry.is_emitting_on_current_thread() {
            registry.note_violation(ToolingViolation::ReentrantEntry);
            return Err(RhiError::invalid_usage(
                "a tooling observer callback must not subscribe to the tooling SPI on its own device",
            )
            .at("ToolingAccess::subscribe"));
        }
        let slot = registry.allocate(observer);
        Ok(ToolingSubscription::new(registry, slot))
    }

    /// Queries the definition of an object that is currently still live.
    ///
    /// This is the seam that makes a capture scope possible at all: an object
    /// created before the scope began never produced an `ObjectCreated` event,
    /// and a consumer that meets its `ObjectId` can pull its definition here
    /// instead of requiring the object to have been observed from creation.
    ///
    /// It is a read. It may be called from inside an observer callback, and it
    /// is not required to be called on an active identity: a definition belongs
    /// to the object, and a capture coordinator must still be able to describe
    /// what it saw after the identity is lost.
    ///
    /// An id that names no live object of this device, or an object of another
    /// device, is a structured error; the backend decides which one and returns
    /// [`RhiErrorKind::WrongDevice`] for the latter.
    pub fn describe_object(&self, id: ObjectId) -> RhiResult<CapturedObjectDefinition> {
        self.backend()?
            .describe_object(id)
            .map_err(|error| error.at("ToolingAccess::describe_object"))
    }

    /// Obtains complete portable semantics for live `RecordedWork`.
    ///
    /// This is the seam for work recorded before the capture scope began but
    /// still retained by plan or submission ownership.
    pub fn describe_work(&self, work: ObjectId) -> RhiResult<CapturedRecordedWork> {
        self.backend()?
            .describe_work(work)
            .map_err(|error| error.at("ToolingAccess::describe_work"))
    }

    /// The callback-contract violations this device's registry has observed.
    ///
    /// The counts are cumulative for the identity and are never reset.
    pub fn violations(&self) -> ToolingViolationCounts {
        match &self.backend {
            Some(backend) => match backend.observers() {
                Some(registry) => registry.violations(),
                None => ToolingViolationCounts::default(),
            },
            None => ToolingViolationCounts::default(),
        }
    }

    /// The backend, or the portable "this device has no tooling seam" answer.
    fn backend(&self) -> RhiResult<&Arc<dyn ToolingBackend>> {
        self.backend.as_ref().ok_or_else(no_tooling_backend)
    }

    /// The observer registry, or the portable "no tooling seam" answer.
    fn registry(&self) -> RhiResult<Arc<ToolingRegistry>> {
        self.backend()?
            .observers()
            .ok_or_else(|| no_tooling_registry(self.identity))
    }
}

impl core::fmt::Debug for ToolingAccess {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ToolingAccess")
            .field("device", &self.identity)
            .field("available", &self.backend.is_some())
            .finish_non_exhaustive()
    }
}

/// The portable answer for a device whose backend exposes no tooling seam.
fn no_tooling_backend() -> RhiError {
    RhiError::new(
        RhiErrorKind::Unsupported,
        "this device backend implements no tooling SPI",
    )
    .at("ToolingAccess")
}

/// The portable answer for a backend that describes objects but observes none.
fn no_tooling_registry(identity: DeviceIdentity) -> RhiError {
    RhiError::new(
        RhiErrorKind::Unsupported,
        format!(
            "device instance {} generation {} exposes no tooling observer registry",
            identity.instance().as_u64(),
            identity.generation().as_u64()
        ),
    )
    .at("ToolingAccess::subscribe")
}

/// An observer registration, unregistered on drop.
///
/// Dropping it unregisters the observer and waits for every callback that began
/// before unregistration to return. Once `Drop` returns, no callback for this
/// subscription can begin or remain active.
pub struct ToolingSubscription {
    registry: Arc<ToolingRegistry>,
    slot: usize,
    /// Whether the slot is still registered. A refused self-drop clears it too,
    /// so a second `Drop` can never run.
    active: bool,
}

impl ToolingSubscription {
    fn new(registry: Arc<ToolingRegistry>, slot: usize) -> Self {
        Self {
            registry,
            slot,
            active: true,
        }
    }
}

impl Drop for ToolingSubscription {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        if self.registry.is_emitting_on_current_thread() {
            // Self-drop: waiting here would wait for this very callback. Honour
            // the unregistration without the wait, and record the violation.
            self.registry.unregister(self.slot);
            self.registry.note_violation(ToolingViolation::SelfDrop);
            return;
        }
        // Take the gate: it is held across callbacks, so acquiring it means
        // every callback that began before this point has returned, and holding
        // it while unregistering means no later emit can snapshot this observer.
        let gate = self.registry.lock_gate();
        self.registry.unregister(self.slot);
        drop(gate);
    }
}

impl core::fmt::Debug for ToolingSubscription {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ToolingSubscription")
            .field("slot", &self.slot)
            .field("active", &self.active)
            .finish_non_exhaustive()
    }
}
