//! The CPU/mock backend: the seam's conformance vehicle.
//!
//! `version-plan.md` section 4 requires a "CPU/mock conformance suite shared by
//! all three backends" as proof for 0.16, and this module is the backend half of
//! it. It exists for a specific reason rather than for convenience: most of the
//! portable contract is decidable without hardware, and a contract rule that can
//! only be checked on a machine with a discrete GPU is a rule that will be broken
//! by the next change. A backend that answers from memory makes those rules
//! testable everywhere, on every platform, in the same run as everything else.
//!
//! It is a **mock and not a reference implementation**, and the difference
//! matters for what it may be used to prove. It performs no lowering, holds no
//! native object, and has no opinion about GPUs; it cannot show that anything is
//! correct on hardware, and `CLAUDE.md` section 4.8 forbids the reverse reading —
//! a mock or a `TestRhi` must never stand in for real-GPU correctness. What it
//! can show, and what it is here to show, is that the portable layer's decisions
//! are made in the portable layer.
//!
//! # Scope limits, recorded rather than implied
//!
//! Two things this backend deliberately does not yet do, so that neither is
//! mistaken for an omission:
//!
//! - **It does not supply a capability table.** [`DeviceBackend`] here has no
//!   `capabilities` method because building one is a separate design question:
//!   `EnabledCapabilities` panics on a query it has no recorded answer for
//!   ("enumeration must record an answer for every query a caller can ask"), so a
//!   conforming backend has to record the *complete* table, and deciding how that
//!   table is enumerated and interned is not part of building the seam. Until it
//!   is decided, `Device::capabilities` keeps its documented `unimplemented!()`
//!   and no creation verb's validation becomes reachable — which is the honest
//!   state, not a regression.
//! - **It does not lower anything that creates a resource.** Buffer, texture,
//!   view, sampler, shader, binding, pipeline, recorder, submission, and
//!   presentation seams do not exist yet; when they do, this backend grows the
//!   same way the native ones will.
//!
//! # Why it is `cfg(test)` and not behind `test-support`
//!
//! The `test-support` feature exists in `Cargo.toml`, and the mock will move
//! behind it as soon as something outside this crate's own test build needs it —
//! a sibling crate's integration test, an example, or a conformance binary. It is
//! not there yet because the promise would be premature: moving it now would mean
//! this backend's use in a `--features test-support` (non-`test`) build
//! un-fulfilling the `#[expect(dead_code)]` attributes that guard the
//! constructors it calls, and trading a real diagnostic for a feature name.

use std::sync::{Arc, Mutex, MutexGuard, atomic::AtomicU64, atomic::Ordering};

use crate::api::capability::{AvailableCapabilities, CapabilityFacts};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, DeviceInstanceId, ObjectId};
use crate::api::platform::{
    AdapterId, AdapterInfo, BackendKind, Device, DeviceLossInfo, DeviceRequestDescriptor,
    DeviceStatus,
};
use crate::api::presentation::PresentationTarget;
use crate::base::platform::{
    DeviceBackend, DeviceRequestBackend, ProviderBackend, RequestProgress,
};

/// Process-local object IDs for everything this backend mints.
///
/// A process-local counter is what section 3 asks [`ObjectId`] to be: opaque,
/// distinct from any native handle, and never stable across processes.
static NEXT_OBJECT: AtomicU64 = AtomicU64::new(1);

/// Mints the next process-local object ID.
fn next_object() -> ObjectId {
    ObjectId::new(NEXT_OBJECT.fetch_add(1, Ordering::Relaxed))
}

/// What a mock device request eventually reports.
///
/// A failure carries a message rather than a built [`RhiError`] because the
/// request is single-shot and the error is produced at the moment it is
/// reported, which is the only moment at which "the backend failed" is still
/// true of the request.
enum MockOutcome {
    /// A device is available.
    Succeeds,
    /// The request fails with [`RhiErrorKind::Unsupported`] and this message.
    Fails(String),
}

/// How a mock provider answers `enumerate_adapters`.
///
/// The three shapes are the three the contract distinguishes, and they are
/// distinct on purpose: see [`ProviderBackend::enumerate_adapters`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MockEnumeration {
    /// The provider exposes no portable enumeration at all (`Ok(None)`).
    NotExposed,
    /// The provider can enumerate and currently has no candidate.
    NoCandidate,
    /// The provider can enumerate its one adapter.
    Available,
}

/// A provider that answers from memory.
pub(crate) struct MockProvider {
    backend: BackendKind,
    instance: DeviceInstanceId,
    enumeration: MockEnumeration,
    presents: bool,
    /// How many `poll` calls report `Pending` before the request resolves.
    pending_steps: u32,
    outcome: MockOutcome,
}

impl MockProvider {
    /// A provider for `backend` under `instance` that enumerates one adapter and
    /// resolves a device request on the first poll.
    ///
    /// The default is the boring one deliberately: a test that wants the
    /// interesting shapes — no enumeration, a multi-step request, a failure —
    /// asks for them by name, so a reader of the test learns which contract shape
    /// is under examination.
    pub(crate) fn new(backend: BackendKind, instance: DeviceInstanceId) -> Self {
        Self {
            backend,
            instance,
            enumeration: MockEnumeration::Available,
            presents: true,
            pending_steps: 0,
            outcome: MockOutcome::Succeeds,
        }
    }

    /// The adapter this provider reports, under this provider's identity.
    ///
    /// Visible so a test can hand the same snapshot to a [`MockDevice`] directly
    /// instead of going through a device request, which is what the tests that
    /// need to observe a loss have to do.
    pub(crate) fn adapter(&self) -> AdapterInfo {
        AdapterInfo::new(
            AdapterId::new(self.instance.as_u64(), 0),
            format!("mock {:?} adapter", self.backend),
            self.backend,
            None,
            None,
            AvailableCapabilities::from_facts(CapabilityFacts::empty()),
        )
    }

    /// Changes what `enumerate_adapters` reports.
    pub(crate) fn enumerating(mut self, enumeration: MockEnumeration) -> Self {
        self.enumeration = enumeration;
        self
    }

    /// Changes whether `supports_presentation` answers yes.
    pub(crate) fn presenting(mut self, presents: bool) -> Self {
        self.presents = presents;
        self
    }

    /// Makes a device request stay `Pending` for `steps` polls before resolving.
    pub(crate) fn pending_steps(mut self, steps: u32) -> Self {
        self.pending_steps = steps;
        self
    }

    /// Makes a device request fail instead of producing a device.
    pub(crate) fn failing(mut self, message: &str) -> Self {
        self.outcome = MockOutcome::Fails(message.to_string());
        self
    }

    /// Wraps this provider for a [`crate::api::platform::PlatformProvider`].
    pub(crate) fn shared(self) -> Arc<dyn ProviderBackend> {
        Arc::new(self)
    }
}

impl ProviderBackend for MockProvider {
    fn enumerate_adapters(&self) -> RhiResult<Option<Vec<AdapterInfo>>> {
        match self.enumeration {
            MockEnumeration::NotExposed => Ok(None),
            MockEnumeration::NoCandidate => Ok(Some(Vec::new())),
            MockEnumeration::Available => Ok(Some(vec![self.adapter()])),
        }
    }

    fn supports_presentation(
        &self,
        _adapter: AdapterId,
        _target: &PresentationTarget,
    ) -> RhiResult<bool> {
        Ok(self.presents)
    }

    fn request_device(
        &self,
        _descriptor: &DeviceRequestDescriptor,
    ) -> RhiResult<Box<dyn DeviceRequestBackend>> {
        Ok(Box::new(MockRequest {
            backend: self.backend,
            adapter: self.adapter(),
            remaining: self.pending_steps,
            outcome: match &self.outcome {
                MockOutcome::Succeeds => MockOutcome::Succeeds,
                MockOutcome::Fails(message) => MockOutcome::Fails(message.clone()),
            },
        }))
    }
}

/// A device request that resolves after a fixed number of polls.
struct MockRequest {
    backend: BackendKind,
    adapter: AdapterInfo,
    remaining: u32,
    outcome: MockOutcome,
}

impl DeviceRequestBackend for MockRequest {
    fn poll(&mut self) -> RhiResult<RequestProgress> {
        if self.remaining > 0 {
            self.remaining -= 1;
            return Ok(RequestProgress::Pending);
        }
        match &self.outcome {
            MockOutcome::Succeeds => Ok(RequestProgress::Ready(Box::new(MockDevice {
                backend: self.backend,
                adapter: self.adapter.clone(),
                object: next_object(),
                liveness: Mutex::new(Liveness {
                    status: DeviceStatus::Active,
                    loss: None,
                }),
            }))),
            MockOutcome::Fails(message) => {
                Err(RhiError::new(RhiErrorKind::Unsupported, message.clone()))
            }
        }
    }
}

/// A device's liveness, as the backend observes it.
struct Liveness {
    status: DeviceStatus,
    loss: Option<DeviceLossInfo>,
}

/// A device that answers from memory.
pub(crate) struct MockDevice {
    backend: BackendKind,
    adapter: AdapterInfo,
    object: ObjectId,
    liveness: Mutex<Liveness>,
}

impl MockDevice {
    /// A live device under the given backend, reporting `adapter`.
    ///
    /// Returned as an `Arc` rather than by value because a test that wants to
    /// observe a loss has to keep a handle of its own: the portable
    /// [`crate::api::platform::Device`] owns the backend, and the backend — not
    /// the handle — is what observes a native loss.
    pub(crate) fn new(backend: BackendKind, adapter: AdapterInfo) -> Arc<Self> {
        Arc::new(Self {
            backend,
            adapter,
            object: next_object(),
            liveness: Mutex::new(Liveness {
                status: DeviceStatus::Active,
                loss: None,
            }),
        })
    }

    /// Records that this device is gone, with the reason.
    ///
    /// One-way, like the loss it records: section 6.5 makes device loss terminal
    /// for the whole identity, so there is no matching `mark_active`.
    pub(crate) fn mark_lost(&self, info: DeviceLossInfo) {
        let mut liveness = self.liveness();
        liveness.status = DeviceStatus::Lost;
        liveness.loss = Some(info);
    }

    /// Borrows the liveness cell, surviving a poisoned lock.
    ///
    /// Recovering from poisoning rather than propagating it is correct here and
    /// only here: the guarded value is two plain fields with no invariant that a
    /// panicking holder could have left half-written, so a panic elsewhere must
    /// not turn a later `status()` into a second panic and hide the first one.
    fn liveness(&self) -> MutexGuard<'_, Liveness> {
        self.liveness
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl DeviceBackend for MockDevice {
    fn backend_kind(&self) -> BackendKind {
        self.backend
    }

    fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter
    }

    fn object_id(&self) -> ObjectId {
        self.object
    }

    fn status(&self) -> DeviceStatus {
        self.liveness().status
    }

    fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.liveness().loss.clone()
    }

    fn poll(&self) -> RhiResult<()> {
        Ok(())
    }

    fn wait_idle(&self) -> RhiResult<()> {
        Ok(())
    }
}

/// A portable device handle over a fresh mock backend.
///
/// For a test that needs a device and has no use for the backend behind it. A
/// test that needs to observe a loss wants [`paired_device_for_test`] instead,
/// because the handle owns the backend and cannot hand it back.
pub(crate) fn device_for_test(identity: DeviceIdentity) -> Device {
    Device::new(identity, mock_native(BackendKind::Dx12))
}

/// A portable device handle paired with the backend that owns its liveness.
pub(crate) fn paired_device_for_test(identity: DeviceIdentity) -> (Device, Arc<MockDevice>) {
    let native = mock_native(BackendKind::Dx12);
    (Device::new(identity, native.clone()), native)
}

/// A mock backend for a device, under the DX12 family and one adapter.
fn mock_native(backend: BackendKind) -> Arc<MockDevice> {
    MockDevice::new(
        backend,
        MockProvider::new(backend, DeviceInstanceId::new(1)).adapter(),
    )
}
