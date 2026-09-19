//! The Direct3D 12 provider: adapter selection and logical device creation.
//!
//! This is the DX12 half of [`crate::base::platform::ProviderBackend`]. It owns
//! the DXGI factory and the one native step that turns an adapter into an
//! `ID3D12Device`, and it decides nothing about legality — every descriptor here
//! has already passed the portable layer.
//!
//! # Why adapter enumeration refuses instead of answering
//!
//! `enumerate_adapters` returns a structured `Unsupported` rather than a list,
//! and that is not a gap left for later. [`AdapterInfo`] carries an
//! [`AvailableCapabilities`] snapshot, and [`crate::api::capability`] states the
//! completeness rule in the strongest terms it has: "an absent entry is not a
//! fact at all: it means enumeration never asked. ... A backend that leaves a hole
//! in its snapshot has produced a bug". Every accessor on that snapshot panics on
//! a hole rather than guessing, because both `Supported` and `Unsupported` would
//! be lies.
//!
//! DX12 answers capability questions through a *device* — `CheckFeatureSupport`,
//! format support, resource-binding tiers — so a provider that enumerated before
//! it could populate a table would have to publish a snapshot full of holes, and
//! the first caller to ask one of those questions would panic in release code.
//! Refusing says the true thing: this provider cannot yet answer what an adapter
//! list promises. `Ok(None)` would say something false instead — that DX12 has no
//! portable enumeration at all, which is exactly what section 5's `Ok(None)` case
//! is reserved for.
//!
//! The ordering this implies is the ordering in `super`: device creation, then
//! capability enumeration, then adapter enumeration. It is also why
//! [`ProviderBackend::request_device`] can be complete while enumeration is not —
//! selecting an adapter needs DXGI only, and the `AdapterInfo` a *device* reports
//! is the snapshot of the adapter it actually got, which is a different and
//! narrower promise than listing candidates.
//!
//! # Reachability, and the shape the expectation takes
//!
//! Nothing outside this module's tests reaches anything here: section 59 keeps
//! the provider off the public surface, and section 5.1 puts the host integration
//! that would open one in another crate. The module therefore opens with a single
//! `#![cfg_attr(not(test), expect(dead_code, ..))]`.
//!
//! That one expectation is not shorthand for a pile of per-item ones — it changes
//! what the lint reports *elsewhere*, which is invisible from inside this file and
//! was measured rather than reasoned about. A module-scope expectation makes
//! references out of that module count as live for the modules they point at. So
//! the portable items this provider calls — `AdapterId::new`, `AdapterId::serial`,
//! `AdapterInfo::new`, `AvailableCapabilities::from_facts`, `ObjectId::new`,
//! `DeviceRequirements::is_empty` — are *not* dead whenever this module is
//! compiled, and *are* dead whenever it is not.
//!
//! Their expectations are therefore gated on the backend feature, not on
//! `not(test)`. The per-item `not(test)` form was written first and does not hold:
//! with `dx12` on, the provider's reference makes each callee live, so a `not(test)`
//! expectation on the callee sits unfulfilled and fails the gate.
//!
//! The same propagation is why [`ffi`] carries no expectation on the two helpers
//! this module calls. Its one remaining expectation is on an item with no caller
//! in any configuration, which is a different case and is fulfilled everywhere.
//!
//! The alternative repair — conditioning those expectations on `feature = "dx12"`
//! — was rejected for putting a backend's name into the portable layer six times
//! over, and for needing six edits per backend from here on. Per-item
//! expectations here keep the direction of the dependency intact and stay true:
//! this module is unreached, so its callers in the portable layer are unreached
//! too, in every feature combination.
//!
//! The `ID3D12Device` field is the one item that is unread in *both* builds —
//! the tests construct the device but nothing reads the handle — so its
//! expectation is unconditional rather than test-conditional.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "unreached outside this module's tests: section 59 keeps the provider off the public surface, and the host integration that would open one is not written"
    )
)]
use std::sync::{Arc, Mutex};

use windows::Win32::Foundation::LUID;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::{D3D12CreateDevice, ID3D12Device};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS,
    DXGI_ERROR_NOT_FOUND, IDXGIAdapter1, IDXGIFactory1,
};

use super::ffi;
use crate::api::capability::{AvailableCapabilities, CapabilityFacts};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceInstanceId, ObjectId};
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::{AdapterId, AdapterInfo, BackendKind, DeviceLossInfo, DeviceStatus};
use crate::api::presentation::PresentationTarget;
use crate::api::submission::{
    LaneWorkDomains, SubmissionCapabilities, SubmissionLaneClass, SubmissionLaneId,
    SubmissionLaneInfo,
};
use crate::base::platform::{
    DeviceBackend, DeviceRequestBackend, ProviderBackend, RequestProgress,
};

/// Turns a DXGI adapter LUID into the serial half of an [`AdapterId`].
///
/// A LUID rather than the enumeration index, because the index is a position in
/// a list that reorders when a driver updates or a device is plugged in, and
/// [`crate::api::platform::AdapterSelection::Explicit`] hands an id back to this
/// provider later. A stale index would silently select a *different* adapter —
/// the failure section 3.1 spends the whole identity model preventing. A LUID is
/// what DXGI itself uses to name an adapter across calls.
///
/// Both halves are kept rather than hashing: the mapping stays injective, so two
/// adapters cannot collide onto one id even in principle.
fn luid_serial(luid: LUID) -> u64 {
    ((luid.HighPart as u32 as u64) << 32) | luid.LowPart as u64
}

/// What one enumerated adapter is, before any capability question is asked.
struct Candidate {
    /// The DXGI adapter itself, kept so `request_device` does not re-enumerate.
    adapter: IDXGIAdapter1,
    /// The serial this adapter is named by within its provider.
    serial: u64,
    /// Its driver-reported name.
    name: String,
    /// `VendorId` / `DeviceId` from the adapter description.
    vendor: u32,
    device: u32,
    /// Whether DXGI flags this as the software (WARP) adapter.
    software: bool,
    /// `DedicatedVideoMemory`, which is what the two preference selections rank
    /// by.
    dedicated_video_memory: usize,
}

/// A Direct3D 12 provider: one DXGI factory and the adapters reachable from it.
pub(crate) struct Dx12Provider {
    /// This provider's instance identity.
    ///
    /// Held so that every [`AdapterId`] it mints names the provider it came from,
    /// which is what makes the portable ownership check an O(1) comparison
    /// (section 3.1) instead of a search.
    instance: DeviceInstanceId,
    /// The factory every enumeration goes through.
    factory: IDXGIFactory1,
}

impl Dx12Provider {
    /// Opens a provider under `instance`.
    ///
    /// # Errors
    ///
    /// Whatever DXGI reports, classified by [`ffi`]. In practice this is
    /// [`RhiErrorKind::BackendFailure`] — DXGI is absent or unregisterable in some
    /// server and container configurations, and a host that asked for DX12 there
    /// should hear that rather than watch device creation fail later with a code
    /// that describes the wrong problem. It is deliberately *not* reported as
    /// `Unsupported`: section 4 reserves that kind for a request the platform
    /// cannot serve, and a machine with no working DXGI has not declined anything
    /// — its backend is broken.
    pub(crate) fn new(instance: DeviceInstanceId) -> RhiResult<Self> {
        // SAFETY: `CreateDXGIFactory2` writes one interface pointer into the
        // out-parameter the binding owns, and the binding converts it to
        // `IDXGIFactory1` only on success. No argument outlives the call, and
        // nothing here dereferences a raw pointer of its own.
        //
        // Flags are zero, not `DXGI_CREATE_FACTORY_DEBUG`: the debug layer is a
        // developer opt-in with a real cost, and turning it on here would make
        // every host pay for a diagnostic it did not ask for.
        let factory = unsafe { CreateDXGIFactory2::<IDXGIFactory1>(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .map_err(|error| ffi::to_rhi(&error, "Dx12Provider::new"))?;
        Ok(Self { instance, factory })
    }

    /// This provider's instance identity.
    pub(crate) fn instance(&self) -> DeviceInstanceId {
        self.instance
    }

    /// Every adapter this factory exposes, in DXGI's own order.
    ///
    /// The WARP adapter is included and flagged rather than skipped, because
    /// `Default` selection has to be able to tell the two apart and a caller that
    /// asked for software rendering should get it.
    fn candidates(&self) -> RhiResult<Vec<Candidate>> {
        let mut candidates = Vec::new();
        let mut index = 0u32;
        loop {
            // SAFETY: `EnumAdapters1` either writes one interface pointer into the
            // out-parameter the binding owns or returns an error; the binding
            // converts only on success. `index` is a plain ordinal.
            let adapter = match unsafe { self.factory.EnumAdapters1(index) } {
                Ok(adapter) => adapter,
                Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(error) => return Err(ffi::to_rhi(&error, "Dx12Provider::enumerate_adapters")),
            };
            index += 1;

            // SAFETY: `GetDesc1` fills a by-value struct the binding owns and
            // returns it by value. It takes no pointer from this code and the
            // adapter outlives the call in `candidates`.
            let description = unsafe { adapter.GetDesc1() }
                .map_err(|error| ffi::to_rhi(&error, "Dx12Provider::enumerate_adapters"))?;

            candidates.push(Candidate {
                serial: luid_serial(description.AdapterLuid),
                name: ffi::adapter_name(&description.Description),
                vendor: description.VendorId,
                device: description.DeviceId,
                software: description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
                dedicated_video_memory: description.DedicatedVideoMemory,
                adapter,
            });
        }
        Ok(candidates)
    }

    /// Picks the adapter `selection` asks for.
    ///
    /// # Errors
    ///
    /// `Unsupported` when the request cannot be satisfied at all: no adapter
    /// exists, no hardware adapter exists where one was required, or an
    /// `Explicit` id names an adapter this provider does not have.
    fn select(&self, selection: AdapterSelection) -> RhiResult<Candidate> {
        let candidates = self.candidates()?;

        // An `Explicit` id is looked up by the LUID-derived serial, so an id whose
        // adapter has since been removed misses rather than landing on whatever
        // now occupies that enumeration position. That the id belongs to *this*
        // provider was already decided portably, in
        // `PlatformProvider::request_device`, before this call — the serial alone
        // cannot tell, and a backend is not the layer that may rule on identity.
        if let AdapterSelection::Explicit(id) = selection {
            return candidates
                .into_iter()
                .find(|candidate| candidate.serial == id.serial())
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::Unsupported,
                        "the explicitly selected adapter is not present on this provider; \
                         enumerate again and select one of the ids it reports",
                    )
                    .at("Dx12Provider::request_device")
                });
        }

        // Preferences rank hardware adapters only. Selecting WARP under
        // `PreferHighPerformance` would be the opposite of what was asked, and
        // WARP is reachable on purpose through `Explicit`.
        let mut hardware: Vec<Candidate> = candidates
            .into_iter()
            .filter(|candidate| !candidate.software)
            .collect();

        match selection {
            AdapterSelection::PreferHighPerformance => {
                hardware
                    .sort_by_key(|candidate| std::cmp::Reverse(candidate.dedicated_video_memory));
            }
            AdapterSelection::PreferLowPower => {
                hardware.sort_by_key(|candidate| candidate.dedicated_video_memory);
            }
            AdapterSelection::Default | AdapterSelection::Explicit(_) => {}
        }

        hardware.into_iter().next().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                "no hardware Direct3D 12 adapter is present on this provider",
            )
            .at("Dx12Provider::request_device")
        })
    }

    /// Selects an adapter and creates the logical device on it.
    ///
    /// Split out from `request_device` because "make the native device" and "wrap
    /// it in the request shape the caller polls" are two responsibilities, and
    /// only the second one is about asynchrony. Keeping them in one function would
    /// force the tests to go through a `Box<dyn DeviceRequestBackend>` to reach a
    /// device they then need to observe losing its liveness — which is exactly the
    /// situation that invites a test-only downcast escape hatch on the seam trait.
    /// Reaching the device directly is the smaller design.
    fn create_native(&self, selection: AdapterSelection) -> RhiResult<Arc<Dx12Device>> {
        let candidate = self.select(selection)?;

        // `D3D_FEATURE_LEVEL_11_0` is the floor Direct3D 12 itself requires, so
        // asking for less is not possible and asking for more would refuse
        // adapters that can run the contract. What was actually achieved is a
        // capability fact, and capability enumeration is where it is reported.
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: `D3D12CreateDevice` writes one interface pointer into
        // `device` and returns an error otherwise; the binding converts only on
        // success. `candidate.adapter` outlives the call and is the adapter the
        // created device is bound to.
        unsafe {
            D3D12CreateDevice(&candidate.adapter, D3D_FEATURE_LEVEL_11_0, &mut device)
                .map_err(|error| ffi::to_rhi(&error, "Dx12Provider::request_device"))?;
        }
        let device = device.ok_or_else(|| {
            // `S_OK` with a null out-parameter is a contract violation by the
            // driver. Reported as a backend failure rather than unwrapped, so a
            // diagnosis says what actually happened.
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "D3D12CreateDevice reported success without producing a device",
            )
            .at("Dx12Provider::request_device")
        })?;

        // One lane, because Direct3D 12 gives a device exactly one direct command
        // queue that the portable layer can name before the fact enumeration
        // exists: several logical lanes do not promise hardware overlap (section
        // 10.3), and reporting more than one would be claiming a scheduling
        // structure this backend has not established.
        //
        // `COMPUTE` is deliberately absent. A direct queue does accept dispatches,
        // but the fact table beside this cannot say whether the `Compute` feature
        // is enabled — it records nothing yet — and a lane accepting compute work
        // on a device whose own contract denies the feature is the exact
        // half-consistency section 7.2's base guarantee is written against.
        // Under-reporting a domain costs a refusal a caller can restructure around;
        // the bit goes in when the enumeration that justifies it lands.
        let submission = SubmissionCapabilities::new(vec![SubmissionLaneInfo::new(
            SubmissionLaneId::new(0),
            SubmissionLaneClass::General,
            LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY),
        )]);

        Ok(Arc::new(Dx12Device {
            adapter: deferred_adapter_info(&candidate, self.instance),
            object: next_object(),
            _device: device,
            facts: CapabilityFacts::empty(),
            submission,
            liveness: Mutex::new(Liveness {
                status: DeviceStatus::Active,
                loss: None,
            }),
        }))
    }
}

/// The adapter snapshot DXGI can answer before any capability question.
///
/// The capability half is deliberately not filled: see the module note. Nothing
/// hands this out today — it is what the capability-enumeration work will need to
/// complete before `enumerate_adapters` can return — so it exists to record what
/// is already known and what is still missing, not to be published.
fn deferred_adapter_info(candidate: &Candidate, instance: DeviceInstanceId) -> AdapterInfo {
    AdapterInfo::new(
        AdapterId::new(instance.as_u64(), candidate.serial),
        candidate.name.clone(),
        BackendKind::Dx12,
        Some(candidate.vendor),
        Some(candidate.device),
        AvailableCapabilities::from_facts(CapabilityFacts::empty()),
    )
}

impl ProviderBackend for Dx12Provider {
    fn enumerate_adapters(&self) -> RhiResult<Option<Vec<AdapterInfo>>> {
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "this provider cannot enumerate adapters yet: an AdapterInfo carries a capability \
             snapshot, DX12 answers capability questions through a device, and the capability \
             enumeration that would fill that snapshot is not built. Returning a list with an \
             empty snapshot would panic the first time a caller asked it anything, and \
             returning Ok(None) would instead claim DX12 has no portable enumeration at all",
        )
        .at("Dx12Provider::enumerate_adapters"))
    }

    fn supports_presentation(
        &self,
        _adapter: AdapterId,
        _target: &PresentationTarget,
    ) -> RhiResult<bool> {
        // Not `Ok(false)`: `PresentationTarget` carries only an `ObjectId`, so
        // there is no channel from it to an `HWND` or a DXGI output and this
        // provider genuinely cannot answer. Reporting "no" would turn "there is no
        // way to ask" into "the hardware says no", which is the silent
        // substitution the seam's third discipline forbids. The missing piece is a
        // host-side `ObjectId -> surface` registry (section 5.1).
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "presentation support cannot be answered yet: a PresentationTarget carries only its \
             object id, and resolving one to a native surface needs the host-side registry that \
             host/platform integration owns. This is not a statement that the adapter cannot \
             present",
        )
        .at("Dx12Provider::supports_presentation"))
    }

    fn request_device(
        &self,
        descriptor: &DeviceRequestDescriptor,
    ) -> RhiResult<Box<dyn DeviceRequestBackend>> {
        // A presentation target in the request is refused here rather than
        // ignored. Section 5.8 puts it in the descriptor precisely so that
        // creation can select the queue family and presentation route; accepting
        // the request and dropping the requirement would produce a device that
        // cannot present while the caller believes it can.
        if !descriptor.presentation_targets().is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this provider cannot yet create a device that is required to present: \
                 presenting needs the same host-side surface registry that \
                 supports_presentation needs. Retry without a presentation target for a \
                 headless device",
            )
            .at("Dx12Provider::request_device"));
        }

        // Requirements are refused rather than dropped, and the distinction is the
        // whole of this check: an *empty* requirement set asks for nothing and can
        // be satisfied honestly, while a non-empty one asks for facts this
        // provider cannot compare yet — exactly the capability enumeration the
        // module note describes. Creating a device and ignoring a
        // `require_feature(Compute)` would hand back a device whose documented
        // contract the caller believes was checked. Section 9.4 forbids that
        // substitution in the shader case and the reasoning does not depend on
        // which requirement was dropped.
        if !descriptor.requirements().is_empty() {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this provider cannot yet honour device requirements: checking them needs the \
                 capability enumeration that reads facts off a live device, and it is not built. \
                 A device requested with no requirements is created without consulting it",
            )
            .at("Dx12Provider::request_device"));
        }

        Ok(Box::new(Dx12Request {
            native: self.create_native(descriptor.selection())?,
        }))
    }
}

/// Process-local object IDs for the devices this backend mints.
///
/// Section 3 asks an [`ObjectId`] to be opaque, distinct from any native handle,
/// and never stable across processes; a process-local counter is all three.
static NEXT_OBJECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Mints the next process-local object ID.
fn next_object() -> ObjectId {
    ObjectId::new(NEXT_OBJECT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

/// A device's liveness, as this backend observes it.
struct Liveness {
    status: DeviceStatus,
    loss: Option<DeviceLossInfo>,
}

/// The native device behind a portable [`crate::api::platform::Device`].
pub(crate) struct Dx12Device {
    /// The adapter snapshot, taken from the adapter that was actually selected.
    adapter: AdapterInfo,
    /// This device's process-local id.
    object: ObjectId,
    /// The Direct3D 12 device.
    ///
    /// Held rather than dropped because every later chapter lowers through it,
    /// and because dropping it is what releases the adapter. Kept rather than
    /// removed even while nothing reads it: removing the field would make
    /// `request_device` create a device and destroy it in the same call, which
    /// would pass every test here while allocating nothing.
    ///
    /// The leading underscore is the honest name for that, not a way to quiet a
    /// lint. An `#[expect(dead_code)]` here was tried and is wrong in a way the
    /// gate catches: rustc does not report an unread field while nothing
    /// constructs the struct at all, so the expectation sits unfulfilled in a
    /// non-test build while being fulfilled in a test one — no single attribute
    /// satisfies both. An underscore-prefixed field is exempt by construction, and
    /// the reason the exemption is correct here is the paragraph above.
    _device: ID3D12Device,
    liveness: Mutex<Liveness>,
    /// The contract this device reports.
    ///
    /// # The gap, stated where it is created
    ///
    /// Empty, and that is not a placeholder for "no capabilities" — it is "not
    /// enumerated yet". D3D12 answers every one of these questions through
    /// `CheckFeatureSupport` and the format-support tables, and reading them off a
    /// live device is the next block of this series; until it lands, the only
    /// honest thing this backend can say is that it has asked nothing.
    ///
    /// What that costs, precisely, so it is not discovered later: the four
    /// accessors that answer by exact key lookup —
    /// [`crate::api::capability::EnabledCapabilities::buffer_support`],
    /// `texture_support`, `binding_support`, and `route` — panic on a query this
    /// table holds no entry for, because
    /// [`crate::api::capability::CapabilityFacts::recorded`] refuses to guess
    /// between `Supported` and `Unsupported`. The feature, limit, format, and
    /// submission accessors answer correctly from this table; they simply answer
    /// "no" and "none", which for an unenumerated device is under-reporting rather
    /// than a false claim. Nothing in the tree calls the four yet, so this is a gap
    /// waiting for its first caller rather than a live defect.
    facts: CapabilityFacts,
    /// The lanes this device offers.
    submission: SubmissionCapabilities,
}

impl Dx12Device {
    /// Borrows the liveness cell, surviving a poisoned lock.
    ///
    /// Recovering rather than propagating is correct here and only here: the
    /// guarded value is two plain fields with no invariant a panicking holder
    /// could have left half-written, so letting one panic hide the first is
    /// strictly worse than reading the fields.
    fn liveness(&self) -> std::sync::MutexGuard<'_, Liveness> {
        self.liveness
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Records that this device is gone, with the reason.
    ///
    /// One-way, like the loss it records: section 6.5 makes device loss terminal
    /// for the whole identity, so there is no matching `mark_active`.
    ///
    /// Crate-private and reached from the failure path that observes a terminal
    /// `HRESULT`; that path is the resource and command lowering, which is not
    /// written, so today only the tests call it.
    pub(crate) fn mark_lost(&self, info: DeviceLossInfo) {
        let mut liveness = self.liveness();
        liveness.status = DeviceStatus::Lost;
        liveness.loss = Some(info);
    }
}

impl DeviceBackend for Dx12Device {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Dx12
    }

    fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter
    }

    fn capability_facts(&self) -> CapabilityFacts {
        self.facts.clone()
    }

    fn submission_capabilities(&self) -> SubmissionCapabilities {
        self.submission.clone()
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
        // Direct3D 12 reports a removed device from the next call that touches it
        // rather than through a callback, so there is no pending state for a poll
        // to advance: the return code of a real call is the only signal there is.
        // See the module note in `super` for why nothing caches a liveness flag.
        Ok(())
    }

    fn wait_idle(&self) -> RhiResult<()> {
        // Correct rather than unimplemented: this device has no queue and no
        // submitted work, so it is idle by construction. Once submission lands,
        // this becomes a fence wait and this comment is the thing that must be
        // deleted — not quietly outlived.
        Ok(())
    }
}

/// A device request that has already produced its device.
///
/// Creation is the single native step and it happened in `request_device`, so
/// there is nothing left to be pending about. `Pending` exists in the contract
/// for providers with a genuinely asynchronous path — WebGPU resolves an adapter
/// and then a device over several turns — and a backend that reported it here
/// would be inventing a delay rather than reporting one.
struct Dx12Request {
    native: Arc<Dx12Device>,
}

impl DeviceRequestBackend for Dx12Request {
    fn poll(&mut self) -> RhiResult<RequestProgress> {
        // The one place this backend's device pointer is handed to the portable
        // layer. `Arc` rather than a fresh box so that whatever the portable side
        // keeps reaches the same native device.
        Ok(RequestProgress::Ready(Box::new(ArcDevice(Arc::clone(
            &self.native,
        )))))
    }
}

/// Adapts a shared [`Dx12Device`] into the owned box the seam hands over.
///
/// The seam transfers an owned `Box<dyn DeviceBackend>` and the portable
/// `Device` puts it behind an `Arc`, but this backend must keep its own handle to
/// observe a loss. This wrapper keeps the two from being two different devices.
struct ArcDevice(Arc<Dx12Device>);

impl DeviceBackend for ArcDevice {
    fn backend_kind(&self) -> BackendKind {
        self.0.backend_kind()
    }

    fn adapter_info(&self) -> &AdapterInfo {
        self.0.adapter_info()
    }

    fn capability_facts(&self) -> CapabilityFacts {
        self.0.capability_facts()
    }

    fn submission_capabilities(&self) -> SubmissionCapabilities {
        self.0.submission_capabilities()
    }

    fn object_id(&self) -> ObjectId {
        self.0.object_id()
    }

    fn status(&self) -> DeviceStatus {
        self.0.status()
    }

    fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.0.loss_info()
    }

    fn poll(&self) -> RhiResult<()> {
        self.0.poll()
    }

    fn wait_idle(&self) -> RhiResult<()> {
        self.0.wait_idle()
    }
}

// The module's tests live beside it in `provider/tests/`, per the workspace
// convention that a module's test set is its own `tests` directory.
#[cfg(test)]
mod tests;
