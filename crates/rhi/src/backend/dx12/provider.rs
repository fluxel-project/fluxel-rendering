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

use super::command::{Dx12CommandSpine, SpineFailure};
use super::facts;
use super::ffi;
use super::resource;
use crate::api::capability::{AvailableCapabilities, CapabilityFacts};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceInstanceId, ObjectId};
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::{AdapterId, AdapterInfo, BackendKind, DeviceLossInfo, DeviceStatus};
use crate::api::presentation::PresentationTarget;
use crate::api::resource::buffer::BufferDescriptor;
use crate::api::submission::{
    LaneWorkDomains, SubmissionCapabilities, SubmissionLaneClass, SubmissionLaneId,
    SubmissionLaneInfo,
};
use crate::base::platform::{
    DeviceBackend, DeviceRequestBackend, ProviderBackend, RequestProgress,
};
use crate::base::resource::BufferBackend;

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

        // One lane. Direct3D 12 does expose more than one queue type — a compute
        // queue and up to three copy queues exist beside the direct queue — but
        // several *logical* lanes do not promise hardware overlap (section 10.3),
        // so reporting the extra queues as lanes would claim a scheduling
        // structure this backend has not established. They arrive when a caller
        // can ask for one by name, which is a question the submission chapter
        // owns rather than this one.
        //
        // `COMPUTE` is present because the fact table beside it now says so.
        // It was absent while the table recorded nothing, because a lane
        // accepting compute work on a device whose own contract denied the
        // `Compute` feature is the half-consistency section 7.2's base guarantee
        // is written against; `facts::probe` records that feature as a structural
        // property of Direct3D 12, so the under-report is no longer the only
        // consistent answer and keeping it would refuse dispatches the device can
        // run.
        let facts = facts::probe(&device)?;

        // The spine is created beside the facts rather than lazily on the first
        // submission, because both are native objects a device either has or does
        // not: `CreateCommandQueue` and `CreateFence` are two calls that can fail,
        // and a failure here is better reported as a refused device request than
        // as a surprising first-submission error. It is also what lets
        // `submission_capabilities` below be a statement about a queue that
        // exists. The three command allocators this device will end up making are
        // *not* created here: those are the ring `command` grows on demand, so a
        // device that never submits never pays for one.
        let spine = Dx12CommandSpine::new(&device).map_err(|native| native.into_rhi())?;

        let submission = SubmissionCapabilities::new(vec![SubmissionLaneInfo::new(
            SubmissionLaneId::new(0),
            SubmissionLaneClass::General,
            LaneWorkDomains::RASTER
                .union(LaneWorkDomains::COMPUTE)
                .union(LaneWorkDomains::COPY),
        )]);

        Ok(Arc::new(Dx12Device {
            adapter: deferred_adapter_info(&candidate, self.instance),
            object: ObjectId::next(),
            device,
            spine,
            facts,
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
    /// Held rather than dropped because every chapter of this backend lowers
    /// through it, and because dropping it is what releases the adapter. It was
    /// named `_device` for as long as nothing read it — the underscore was the
    /// honest name for an allocation that had to outlive the call that made it,
    /// not a way to quiet a lint — and the name lost its underscore in the round
    /// that gave it its first reader, [`Self::create_buffer`]. An
    /// `#[expect(dead_code)]` was tried in that earlier state and is wrong in a
    /// way the gate catches: rustc does not report an unread field while nothing
    /// constructs the struct at all, so the expectation sits unfulfilled in a
    /// non-test build while being fulfilled in a test one, and no single
    /// attribute satisfies both.
    device: ID3D12Device,
    /// The queue, fence and command-list ring every submission goes through.
    ///
    /// Held by value and never cloned: it owns the one queue this device has, and
    /// a second handle to the same queue would be a second path to `Signal` on
    /// the same fence, which is the only place a serial is minted.
    spine: Dx12CommandSpine,
    liveness: Mutex<Liveness>,
    /// The contract this device reports.
    ///
    /// # What is in the table, and what is still missing from it
    ///
    /// Filled by `facts::probe` from the live `ID3D12Device` beside it, which is
    /// where the per-table detail lives — read that module's doc for what each
    /// table is derived from and which questions this backend still cannot answer.
    /// Two things belong here rather than there, because they are about the shape
    /// of this struct and not about Direct3D 12.
    ///
    /// The first is why the probe is called *here*. It runs once, in
    /// `create_device`, next to `CreateCommandQueue` and `CreateFence` and for the
    /// same reason: these are the native questions a device either answers or does
    /// not, and a device that cannot be asked is better refused at creation than
    /// discovered halfway through a frame. A table filled lazily would put the
    /// first capability answer on whichever call happened to arrive first.
    ///
    /// The second is what an absent entry still means, so it is not discovered
    /// later. The table is a snapshot of *this device*, and where it is incomplete
    /// the incompleteness is `CapabilityFacts`'s own documented behaviour rather
    /// than something this backend invents: `route`, `texture_support`,
    /// `view_compatibility` and `binding_limit` answer conservatively when they
    /// have no entry, which costs throughput and not correctness. `buffer_support`
    /// is the exception that panics, and the probe fills its key space completely
    /// for exactly that reason. `binding_support` is now in the same position
    /// despite answering rather than panicking, because a binding answer of
    /// `Unsupported` refuses a legal layout rather than merely declining to
    /// advertise one — so its key space, too, is filled rather than sampled.
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
    /// `HRESULT`. That path is now written for allocation — see
    /// [`Self::create_buffer`] — and the command lowering will be the next
    /// caller; until it lands, allocation failures are the only ones that can
    /// end a device's identity here.
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
        // rather than through a callback, so there is no *liveness* state for a
        // poll to advance: the return code of a real call is the only signal
        // there is, and nothing caches a liveness flag (see the module note in
        // `super`).
        //
        // What a poll does advance is the submission side. A readback's bytes
        // become readable when the GPU finishes writing them, and the fence is
        // where that becomes known; this is the portable layer's only progress
        // verb (section 6.7 keeps a blocking wait out of the frame loop), so it is
        // the only place those bytes can be published. `advance` reads the fence
        // and copies out whatever it reports finished — it never waits.
        self.spine.advance();
        Ok(())
    }

    fn wait_idle(&self) -> RhiResult<()> {
        // Section 6.7: shutdown, recovery, and diagnostics only. The wait is a
        // bounded one on a fence-signalled event rather than an `INFINITE` block,
        // because a removed device leaves fence values that will never be written
        // and a library must not turn that into a hung host.
        self.spine.wait_idle().map_err(SpineFailure::into_rhi)
    }

    /// Allocates one buffer, and is the only place in this backend that acts on a
    /// terminal native failure.
    ///
    /// The `HRESULT` alone cannot answer the question that matters here: a
    /// removed device reports that from an arbitrary call, and this one is as
    /// likely as any other, so a failure that looks like a plain allocation
    /// refusal may be the device ending. [`ffi::NativeError`] carries the
    /// classification beside the error so this reads it once rather than
    /// re-deriving it, and `Terminal` is the only case that is recorded: marking
    /// a device lost because one allocation ran out of memory would retire a
    /// usable device on a transient failure, which is the expensive direction of
    /// the mistake and the same reasoning [`ffi::NativeFailure`] gives for
    /// treating a hung device as alive.
    fn create_buffer(&self, descriptor: &BufferDescriptor) -> RhiResult<Box<dyn BufferBackend>> {
        match resource::create_buffer(&self.device, descriptor) {
            Ok(buffer) => Ok(Box::new(buffer) as Box<dyn BufferBackend>),
            Err(native) => {
                if native.failure().is_terminal() {
                    // Built before the error is consumed, because the summary is
                    // about the same failure the error reports and the port read
                    // that follows must answer the same thing.
                    let summary = format!(
                        "Direct3D 12 reported a terminal failure while allocating a buffer, \
                         and section 6.5 makes loss terminal for the identity: {}",
                        native.as_error()
                    );
                    self.mark_lost(DeviceLossInfo::new(summary));
                }
                Err(native.into_rhi())
            }
        }
    }

    /// Lowers a plan onto the spine's queue, and is the second place in this
    /// backend that acts on a terminal native failure.
    ///
    /// The two directions of section 41.3 meet here. Phase A — everything
    /// recorded, nothing committed — is [`Dx12CommandSpine::submit`]'s, and its
    /// `Err` genuinely proves no native work was accepted. Phase B — once
    /// anything is accepted, no `Err` may claim otherwise — is also the spine's,
    /// which is why a post-commit `Signal` failure comes back as `Ok` and is
    /// reported through [`Self::completion`] instead.
    ///
    /// What is left for this layer is the one question only the device can
    /// answer: whether a failure ended it. `Terminal` marks the device lost, for
    /// the same reason and in the same shape as [`Self::create_buffer`] — a
    /// failure that looks like a plain refusal may be the device ending, and
    /// `ffi::NativeFailure` already carries the classification so this reads it
    /// once rather than re-deriving it from a message.
    fn submit(
        &self,
        request: &crate::base::command::SubmissionRequest<'_>,
    ) -> RhiResult<crate::base::command::SubmissionOutcome> {
        match self.spine.submit(request) {
            Ok(outcome) => Ok(outcome),
            Err(failure) => {
                if failure.is_terminal() {
                    // Built before the failure is consumed, because the summary is
                    // about the same failure the error reports and the port read
                    // that follows must answer the same thing.
                    let summary = format!(
                        "Direct3D 12 reported a terminal failure while lowering a plan, and \
                         section 6.5 makes loss terminal for the identity: {}",
                        failure.message()
                    );
                    self.mark_lost(DeviceLossInfo::new(summary));
                }
                Err(failure.into_rhi())
            }
        }
    }

    /// Reports one serial's state, asking the spine first.
    ///
    /// The order is section 41.8's, and it is the whole reason this is not a
    /// two-line delegate. Points already reported `Complete` must stay `Complete`
    /// after a loss — the work really did finish, and a caller that saw it finish
    /// must not be told it did not. So the spine's answer wins when it is
    /// `Complete`, and only a spine answer of `Pending` or `Failed` can be
    /// upgraded to `DeviceLost` by the liveness cell.
    ///
    /// The upgrade is what closes the loop the spine opens: a serial past the
    /// first unobservable one is answered `Failed` by [`Dx12CommandSpine`], which
    /// the portable layer would surface as a backend failure even though the
    /// device is gone. With the liveness cell consulted last, the same serial
    /// answers `DeviceLost` — the terminal state section 41.8 requires and the one
    /// a caller branches on to recover.
    fn completion(&self, serial: u64) -> crate::api::submission::CompletionState {
        let spine = self.spine.completion(serial);
        if matches!(spine, crate::api::submission::CompletionState::Complete) {
            return spine;
        }
        match self.loss_info() {
            Some(info) => crate::api::submission::CompletionState::DeviceLost(info),
            None => spine,
        }
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

    /// Forwarded rather than reimplemented, and that is the point of this
    /// wrapper: a device that reached the portable layer as `ArcDevice` and one
    /// the tests hold as `Arc<Dx12Device>` must allocate through the same code,
    /// or the terminal-failure path above would be exercised by nobody.
    fn create_buffer(&self, descriptor: &BufferDescriptor) -> RhiResult<Box<dyn BufferBackend>> {
        self.0.create_buffer(descriptor)
    }

    /// Forwarded for the same reason `create_buffer` is: a device that reached the
    /// portable layer as `ArcDevice` and one the tests hold as `Arc<Dx12Device>`
    /// must submit through the same code, or the terminal-failure path would be
    /// exercised by nobody.
    fn submit(
        &self,
        request: &crate::base::command::SubmissionRequest<'_>,
    ) -> RhiResult<crate::base::command::SubmissionOutcome> {
        self.0.submit(request)
    }

    fn completion(&self, serial: u64) -> crate::api::submission::CompletionState {
        self.0.completion(serial)
    }
}

// The module's tests live beside it in `provider/tests/`, per the workspace
// convention that a module's test set is its own `tests` directory.
#[cfg(test)]
mod tests;
