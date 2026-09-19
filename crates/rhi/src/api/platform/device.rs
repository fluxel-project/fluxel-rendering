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
//!
//! # What is decided here and what is asked elsewhere
//!
//! The device holds an [`DeviceIdentity`] and a backend, and nothing else. Every
//! verb that needs a native fact — provenance, identity, liveness, progress —
//! asks the backend through the crate-private lowering seam, which section 59 of
//! `08-governance-freeze-checklist.md` keeps off the public surface and which
//! this documentation therefore cannot link to. What stays here is the part a
//! backend must not decide: the identity comparison section 3.1 puts first, the
//! order in which ownership and liveness are judged, and the structured error a
//! caller sees.
//!
//! Liveness in particular has one home and it is not this one. The backend
//! observes the loss and holds the fact; this module holds the rules about it —
//! that it is terminal, that the summary is stable (section 6.5), and that a lost
//! device answers `DeviceLost` only once ownership has been settled. Caching the
//! fact here as well would create a second authority for it, which is the thing
//! section 65.3 rules out.

use std::sync::Arc;

use crate::api::capability::EnabledCapabilities;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, ObjectId};
use crate::api::platform::provider::{AdapterInfo, BackendKind};
use crate::api::platform::requirements::OptionalFeature;
use crate::base::platform::DeviceBackend;

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
    ///
    /// The DX12 backend's allocation path is now such an observer, so the
    /// expectation below is gated on the backend feature as well as on the test
    /// build. A module-scope expectation in that backend makes references *out*
    /// of it count as live for the items they point at, so with `dx12` on this
    /// constructor is reached in a non-test build too and a `not(test)`-only
    /// expectation would sit unfulfilled — the same trap the provider's module
    /// documentation records for its own callees.
    #[cfg_attr(
        all(not(test), not(feature = "dx12")),
        expect(
            dead_code,
            reason = "called by the contract tests and by the DX12 allocation path; with that backend compiled out, the code that observes a native loss is not written"
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
    /// The native execution domain this handle lowers through.
    ///
    /// Shared rather than owned, and that is what makes `Clone` mean what
    /// section 6.1 says it means: a clone is the same domain under the same
    /// identity, so it must reach the same native device rather than open a
    /// second one. It is never replaced after construction — loss does not
    /// re-point it, because section 6.5 makes loss terminal and recovery a new
    /// request with a new identity.
    ///
    /// Liveness lives on the backend rather than beside this field, because the
    /// backend is what observes a native loss. Keeping one copy is section 65.3's
    /// rule; keeping it on the side that can see the event is what makes the copy
    /// authoritative.
    native: Arc<dyn DeviceBackend>,
    /// What this device can actually do, built once from the backend's
    /// enumeration and never rebuilt.
    ///
    /// Interned here and nowhere else. Section 7.1 makes the compatibility id the
    /// interning of the contract's semantics, so the id has to be minted at the one
    /// moment the whole contract is in hand — which is construction. A device whose
    /// id were computed per call could answer two different ids for one contract.
    ///
    /// Shared rather than owned, because [`Device`] is cloned liberally and a clone
    /// is the same domain under the same identity (section 6.1): cloning the maps
    /// would copy a few hundred entries per clone to re-derive an answer that
    /// cannot have changed. `Arc` makes a clone cheap and keeps the guarantee that
    /// every clone reports the *same* id, which is what section 7.1's reuse rule
    /// depends on.
    ///
    /// Not a second authority for anything: it is immutable by contract (section
    /// 7.2), the backend owns no copy of it, and it is never replaced — loss does
    /// not re-point it, because section 6.5 makes recovery a new request with a new
    /// identity.
    capabilities: Arc<EnabledCapabilities>,
}

impl Device {
    /// Opens a logical execution domain under a freshly minted identity.
    ///
    /// Crate-private: section 6.1 ties identity minting to a completed device
    /// request, and section 6.1's other half is that a backend does not mint its
    /// own — so only the request path that did both may call this.
    ///
    /// # Why this can fail
    ///
    /// It reads the backend's enumeration, interns it, and checks section 7.2's
    /// base guarantee in one place. That check is a portable rule about a device
    /// fact, and putting it here rather than in the request path means it holds for
    /// *every* way a device comes into existence — including the mock backend's
    /// direct construction, where a test device would otherwise be exempt from the
    /// contract the tests exist to check.
    ///
    /// A failure here is [`RhiErrorKind::BackendFailure`] rather than
    /// [`RhiErrorKind::Unsupported`]: every device is required to have a lane
    /// accepting `RASTER | COPY`, so a snapshot without one is a defect in the
    /// enumeration that produced it, not a capability the caller may not use. The
    /// device is not published, which is the point — section 6.9 forbids handing a
    /// portable defect down for a driver or a validation layer to discover later.
    ///
    /// # Errors
    ///
    /// [`RhiErrorKind::BackendFailure`] when the enumeration violates the base
    /// guarantee, naming which half of it was violated.
    pub(crate) fn new(identity: DeviceIdentity, native: Arc<dyn DeviceBackend>) -> RhiResult<Self> {
        let capabilities = EnabledCapabilities::from_facts(
            native.capability_facts(),
            native.submission_capabilities(),
        );
        // The `Compute` feature is read back out of the contract that was just
        // interned rather than asked of the backend separately: section 7.2's rule
        // is about the *enabled* feature set, and reading it from anywhere else
        // would let the two answers disagree at exactly the moment the rule is
        // being checked.
        capabilities
            .submission()
            .validate_base_guarantee(capabilities.supports_feature(OptionalFeature::Compute))?;
        Ok(Self {
            identity,
            native,
            capabilities: Arc::new(capabilities),
        })
    }

    /// This device's identity.
    ///
    /// Every object the device owns carries the same identity, and section 3.1
    /// makes the comparison against it the first thing any public operation does
    /// — in O(1), before a backend is touched.
    pub fn identity(&self) -> DeviceIdentity {
        self.identity
    }

    /// The native domain this handle lowers through.
    ///
    /// Crate-private because section 59 keeps native lowering out of the exported
    /// surface and because no caller outside this crate may name a backend — the
    /// traits it returns are `pub(crate)` for the same reason, so this is not a
    /// leak with a narrow door but the seam's ordinary inside face.
    ///
    /// The callers are the creation verbs of the later chapters, which live
    /// beside the types they produce (adjudication A28) rather than here, and
    /// therefore need to reach the backend through the handle they were given.
    /// Reaching it *through* the handle rather than storing a copy is what keeps
    /// one device from having two authoritative backends.
    pub(crate) fn native(&self) -> &Arc<dyn DeviceBackend> {
        &self.native
    }

    /// The backend family this device came from.
    ///
    /// For diagnostics, UI, capture provenance, and benchmark reports only. It
    /// is not a capability oracle: the same family exposes different
    /// capabilities on different drivers, so asking the backend what it is
    /// instead of asking the device what it can do is the mistake section 6.3
    /// names.
    pub fn backend(&self) -> BackendKind {
        self.native.backend_kind()
    }

    /// A snapshot of the adapter that was actually selected.
    ///
    /// Available even when the provider does not support adapter enumeration:
    /// section 6.3 asks a device to report what it actually got, which is a
    /// weaker and always-answerable question than listing the candidates.
    pub fn adapter_info(&self) -> &AdapterInfo {
        self.native.adapter_info()
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
    ///
    /// The contract was interned once, at construction, so this is a borrow of
    /// storage the device already owns rather than a query: two calls, and two
    /// calls on two clones of one device, return the same
    /// [`crate::api::capability::CapabilityCompatibilityId`] — which is what
    /// section 7.1's `CompiledGraph` reuse rule reads.
    pub fn capabilities(&self) -> &EnabledCapabilities {
        &self.capabilities
    }

    /// Whether the device is still usable.
    ///
    /// The answer comes from the backend, which is what observes a loss, and it
    /// is asked every time rather than cached here: caching it would give the
    /// crate two places that know whether this device is alive, and section 65.3
    /// allows exactly one authority per concern.
    pub fn status(&self) -> DeviceStatus {
        self.native.status()
    }

    /// Why the device was lost, or `None` while it is active.
    pub fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.native.loss_info()
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
        self.native.poll()
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
        self.native.wait_idle()
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
        self.native.object_id()
    }

    /// Refuses an operation that would use this device while it is lost.
    ///
    /// Section 6.5 states the rule for exactly this case, so it is quoted rather
    /// than paraphrased: after a loss the handles listed there "must return
    /// `WrongDevice` when passed to that new Device, and return **`DeviceLost`
    /// when used through their lost original Device**". Every creation verb is
    /// such a use, which is why each one calls this.
    ///
    /// It is a portable verdict and not something the backend is left to notice.
    /// Section 6.9 names this case beside the wrong-device one and draws the line
    /// in the same place for both:
    ///
    /// ```text
    /// wrong device / device lost
    ///     -> Fluxel RHI structured validation -> Err(WrongDevice | DeviceLost)
    /// ```
    ///
    /// rather than passing a stale handle down and letting a driver, a validation
    /// layer, or a browser "handle it unpredictably". The reason section 6.9
    /// gives is the reason this belongs here: native validation may not even be
    /// enabled in a release environment.
    ///
    /// # Why this runs *after* the ownership comparison
    ///
    /// Section 3.1 puts the O(1) identity comparison first, and section 6.5 gives
    /// the two questions different answers. An object handed to a device that is
    /// not its own is `WrongDevice` even when that device is also lost; if
    /// liveness were checked first, such a caller would be told `DeviceLost` when
    /// the actual mistake was the object it passed. So this check sits after
    /// every portable ownership verdict and before the first device-fact read.
    ///
    /// # Errors
    ///
    /// [`RhiErrorKind::DeviceLost`], carrying section 6.5's stable loss summary
    /// when one was recorded. The summary is in the message rather than beside it
    /// because [`DeviceLossInfo`]'s own accessor is on the device, and an error
    /// that says only "lost" would send every caller back to ask the device a
    /// question this call site already had the answer to.
    pub(crate) fn require_active(&self) -> RhiResult<()> {
        match self.native.status() {
            DeviceStatus::Active => Ok(()),
            DeviceStatus::Lost => {
                let message = match &self.native.loss_info() {
                    Some(loss) => format!(
                        "this device is lost and section 6.5 makes loss terminal, so this \
                         operation cannot be performed through it: {}",
                        loss.message()
                    ),
                    None => "this device is lost and section 6.5 makes loss terminal, so this \
                             operation cannot be performed through it"
                        .to_string(),
                };
                Err(RhiError::new(RhiErrorKind::DeviceLost, message))
            }
        }
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
            .field("status", &self.native.status())
            .field("loss", &self.native.loss_info())
            .finish_non_exhaustive()
    }
}
