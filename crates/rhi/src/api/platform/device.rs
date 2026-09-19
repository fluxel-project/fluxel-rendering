//! The device: a shared logical execution domain (specification section 6).
//!
//! Everything a caller creates, records, submits, or presents is owned by a
//! device, and every one of those objects must be traceable to the same
//! [`DeviceIdentity`]. This module owns the device's own surface — who it is,
//! what it came from, what it can do, and whether it is still alive.
//!
//! The verbs that *create* things are not here. `Device::create_buffer` is
//! written in the resource module, `Device::create_pipeline` in the pipeline
//! module, and so on, each next to the types it produces. Rust allows the
//! inherent impl to live in another module of the same crate, and doing so keeps
//! one rule in one place instead of collecting every creation verb into a file
//! that would have to know about every resource in the crate.
//!
//! This module deliberately does not own the capability vocabulary
//! ([`crate::api::capability`]) or presentation
//! ([`crate::api::presentation`]); it only hands out handles to them.

use crate::api::capability::EnabledCapabilities;
use crate::api::error::RhiResult;
use crate::api::identity::{DeviceIdentity, ObjectId};
use crate::api::platform::provider::{AdapterInfo, BackendKind};

/// Whether a device is still usable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceStatus {
    /// The device is usable.
    Active,
    /// The device is gone, and permanently so.
    Lost,
}

/// A stable summary of why a device was lost.
///
/// Section 6.5 requires the summary to be stable rather than a one-shot
/// notification: a caller that asks twice, or asks long after the loss, must get
/// the same answer.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DeviceLossInfo {
    message: String,
}

impl DeviceLossInfo {
    /// Describes a loss.
    ///
    /// Crate-private: only the code that observed the loss may summarize it.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "called by the contract tests; the code that observes a native loss is not written"
        )
    )]
    pub(crate) fn new(message: String) -> Self {
        Self { message }
    }

    /// The human-readable loss summary.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// The backend-private execution domain behind a [`Device`].
///
/// Reserved. A device's contract is fixed by this module; the native instance,
/// queue, allocator, and retirement bookkeeping that satisfy it arrive with the
/// backend port. Keeping the seam as a named private type means the port adds
/// fields here rather than reshaping `Device`.
#[derive(Clone)]
struct DeviceDomain;

/// A shared logical execution domain.
///
/// Cloneable, and cloning is not a new device:
///
/// ```text
/// Device::clone()        the same domain, the same DeviceIdentity
/// request_device() again a new domain, a new identity and generation
/// ```
///
/// Section 6.1 makes that a portable contract rather than an implementation
/// detail: even where a backend reuses one native device internally, two
/// successful requests still produce two isolated logical domains that must not
/// accept each other's objects.
///
/// Loss is terminal for the whole identity. There is no path that increments a
/// generation on an existing public device or revives an old handle — a retry is
/// a new [`crate::api::platform::DeviceRequest`] and therefore a new identity.
#[derive(Clone)]
pub struct Device {
    identity: DeviceIdentity,
    status: DeviceStatus,
    loss: Option<DeviceLossInfo>,
    #[expect(
        dead_code,
        reason = "read once the backend port lowers device operations through it"
    )]
    domain: DeviceDomain,
}

impl Device {
    /// Opens a logical execution domain under a freshly minted identity.
    ///
    /// Crate-private: section 6.1 ties identity minting to a completed device
    /// request, so nothing else may produce the pair.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "called by the contract tests; a completed device request opens one"
        )
    )]
    pub(crate) fn new(identity: DeviceIdentity) -> Self {
        Self {
            identity,
            status: DeviceStatus::Active,
            loss: None,
            domain: DeviceDomain,
        }
    }

    /// This device's identity.
    ///
    /// Every object the device owns carries the same identity, and section 3.1
    /// makes the comparison against it the first thing any public operation does
    /// — in O(1), before a backend is touched.
    pub fn identity(&self) -> DeviceIdentity {
        self.identity
    }

    /// The backend family this device came from.
    ///
    /// For diagnostics, UI, capture provenance, and benchmark reports only. It
    /// is not a capability oracle: the same family exposes different
    /// capabilities on different drivers, so asking the backend what it is
    /// instead of asking the device what it can do is the mistake section 6.3
    /// names.
    pub fn backend(&self) -> BackendKind {
        unimplemented!(
            "backend provenance arrives with the backend port; the contract is \
             fixed, the device state is not built"
        )
    }

    /// A snapshot of the adapter that was actually selected.
    ///
    /// Available even when the provider does not support adapter enumeration:
    /// section 6.3 asks a device to report what it actually got, which is a
    /// weaker and always-answerable question than listing the candidates.
    pub fn adapter_info(&self) -> &AdapterInfo {
        unimplemented!(
            "adapter provenance arrives with the backend port; the contract is \
             fixed, the device state is not built"
        )
    }

    /// What this device can actually do.
    ///
    /// The only correct source of capability answers. The relationship to the
    /// adapter's snapshot is one-way:
    ///
    /// ```text
    /// EnabledOnDevice subset-of AvailableOnAdapter
    /// ```
    ///
    /// so a feature the adapter reported as available may still be absent here,
    /// and a caller that planned against the adapter would be wrong.
    pub fn capabilities(&self) -> &EnabledCapabilities {
        unimplemented!(
            "enabled capabilities arrive with the backend port; the contract is \
             fixed, the device state is not built"
        )
    }

    /// Whether the device is still usable.
    pub fn status(&self) -> DeviceStatus {
        self.status
    }

    /// Why the device was lost, or `None` while it is active.
    pub fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.loss.clone()
    }

    /// Non-blockingly advances completion, loss, and callback bookkeeping.
    ///
    /// This is RHI-owned bookkeeping, not the host's event loop. Section 6.6
    /// requires the host to keep pumping its own loop; a host that polled this
    /// instead would starve the browser or window messages the RHI depends on.
    ///
    /// Pending work must not stay pending forever (section 6.5), which is what
    /// makes polling the device a way to make progress rather than only to
    /// observe it.
    pub fn poll(&self) -> RhiResult<()> {
        unimplemented!(
            "polling arrives with the backend port; the contract is fixed, the \
             bookkeeping is not built"
        )
    }

    /// Blocks until the device is idle.
    ///
    /// For shutdown and diagnostics only. Section 6.7 forbids it as a per-frame
    /// retirement mechanism and as the correctness mechanism of a render loop:
    /// a caller that needs to know when work finished is asking about a
    /// completion point, and a caller that needs resources back is asking about
    /// retirement. On a restricted host or backend this returns
    /// [`crate::api::error::RhiErrorKind::Unsupported`] rather than pretending to
    /// have waited.
    pub fn wait_idle(&self) -> RhiResult<()> {
        unimplemented!(
            "waiting for idle arrives with the backend port; the contract is \
             fixed, the device state is not built"
        )
    }

    /// This device's process-local object ID.
    ///
    /// Section 3 gives every RHI object a process-local [`ObjectId`] distinct
    /// from any native handle, and section 7.1 requires tooling to be able to
    /// *describe* what it observes by that ID rather than by a pointer.
    ///
    /// Sections 3 through 7 declare no accessor that yields an `ObjectId`, so
    /// this verb is an addition rather than a transcription. It is added because
    /// the alternative is worse: `RhiError::object` returns an `ObjectId` and
    /// section 7.1 requires tooling to describe objects by one, which is
    /// unreachable if no object can name its own ID.
    pub fn object_id(&self) -> ObjectId {
        unimplemented!(
            "object identity arrives with the backend port; the contract is \
             fixed, the registry is not built"
        )
    }

    /// Records that this device is gone, with the reason.
    ///
    /// Crate-private, and one-way: section 6.5 makes loss terminal for the whole
    /// identity, so there is no matching `mark_active`.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "reached by the backend port that observes the loss"
        )
    )]
    pub(crate) fn mark_lost(&mut self, info: DeviceLossInfo) {
        self.status = DeviceStatus::Lost;
        self.loss = Some(info);
    }
}

impl core::fmt::Debug for Device {
    /// Prints portable identity, status, and loss summary.
    ///
    /// Hand-written rather than derived, for the reason recorded as adjudication
    /// A16 in the 0.16 plan. The trait is required rather than optional: section
    /// 5.9's `DeviceRequest::poll` returns `RhiResult<RequestStatus<Device>>`, and
    /// a caller that unwraps or logs a failed request needs the payload to be
    /// printable. Printing the execution domain instead would be wrong on two
    /// counts — the backend port will add a native field that has no reason to be
    /// `Debug`, and a device's native state is not something a log should
    /// describe.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Device")
            .field("identity", &self.identity)
            .field("status", &self.status)
            .field("loss", &self.loss)
            .finish_non_exhaustive()
    }
}
