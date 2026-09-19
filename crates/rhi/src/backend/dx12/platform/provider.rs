//! The Direct3D 12 provider: adapter selection and logical device creation.
//!
//! This is the DX12 half of [`crate::base::platform::ProviderBackend`]. It owns
//! the DXGI factory and the one native step that turns an adapter into an
//! `ID3D12Device`, and it decides nothing about legality — every descriptor here
//! has already passed the portable layer.
//!
//! The device this module creates is [`super::device::Dx12Device`], and the
//! request that carries it back to the portable layer is
//! [`super::request::Dx12Request`]. Both are separate files for the reason
//! [`crate::base`] splits its own seam: a provider is asked about the *domain*,
//! a device carries one native object, and a request is the single-shot
//! handover between them.
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
//! The ordering this implies is the ordering in `super::super`: device creation,
//! then capability enumeration, then adapter enumeration. It is also why
//! [`ProviderBackend::request_device`] can be complete while enumeration is not —
//! selecting an adapter needs DXGI only, and the `AdapterInfo` a *device* reports
//! is the snapshot of the adapter it actually got, which is a different and
//! narrower promise than listing candidates.
//!
//! # Reachability, and the shape the expectation takes
//!
//! This file opens with the one `#![cfg_attr(not(test), expect(dead_code, ..))]`
//! that carries the whole chapter's unreachability; [`super`] explains why it
//! opens here rather than at the chapter root, and what that expectation does to
//! the lint in the files beside this one. The short version: the provider is the
//! only entry point into the chapter, nothing outside the chapter names it, and
//! an expectation placed at the chapter root would make every file beneath it a
//! live root instead of recording that none of them is.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "unreached outside this chapter's tests: section 59 keeps the provider off the public surface, and the host integration that would open one is not written"
    )
)]

use std::sync::Arc;

use windows::Win32::Foundation::LUID;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::{D3D12CreateDevice, ID3D12Device};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS,
    DXGI_ERROR_NOT_FOUND, IDXGIAdapter1, IDXGIFactory1,
};

use crate::api::capability::{AvailableCapabilities, CapabilityFacts};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::DeviceInstanceId;
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::{AdapterId, AdapterInfo, BackendKind};
use crate::api::presentation::PresentationTarget;
use crate::api::submission::{
    LaneWorkDomains, SubmissionCapabilities, SubmissionLaneClass, SubmissionLaneId,
    SubmissionLaneInfo,
};
use crate::base::platform::ProviderBackend;

use crate::backend::dx12::command::Dx12CommandSpine;
use crate::backend::dx12::ffi;

use super::device::Dx12Device;
use super::facts;
use super::request::Dx12Request;

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
///
/// `pub(super)` on the type and on every field, which is the visibility this
/// chapter's own test set needs and no wider: the tests assert *which* adapter a
/// selection landed on and compare its vendor, device and memory against the
/// DXGI description they read themselves, and re-deriving those through a
/// created device would test the device rather than the selection. The
/// alternative — accessors for six fields — would be six functions whose only
/// caller is a test, which is the shape the file set deletes rather than keeps.
///
/// Nothing outside `platform` names this type, and there is no reason to: a
/// candidate is an intermediate between `select` and `create_native`, and the
/// portable layer's counterpart is `AdapterId`, which `select` returns.
pub(super) struct Candidate {
    /// The DXGI adapter itself, kept so `request_device` does not re-enumerate.
    pub(super) adapter: IDXGIAdapter1,
    /// The serial this adapter is named by within its provider.
    pub(super) serial: u64,
    /// Its driver-reported name.
    pub(super) name: String,
    /// `VendorId` / `DeviceId` from the adapter description.
    pub(super) vendor: u32,
    pub(super) device: u32,
    /// Whether DXGI flags this as the software (WARP) adapter.
    pub(super) software: bool,
    /// `DedicatedVideoMemory`, which is what the two preference selections rank
    /// by.
    pub(super) dedicated_video_memory: usize,
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
    /// `pub(super)` for the same reason [`Candidate`]'s fields are: this
    /// chapter's test set reads the adapter list directly, to check that the
    /// candidates DXGI reports are the ones it sees, which a created device
    /// cannot answer.
    pub(super) fn candidates(&self) -> RhiResult<Vec<Candidate>> {
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
    /// `pub(super)` because the chapter's test set calls it by itself — the
    /// selection rules are worth testing without creating a device, and creating
    /// one per selection rule would make the test set slower than the thing it
    /// checks.
    pub(super) fn select(&self, selection: AdapterSelection) -> RhiResult<Candidate> {
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
    pub(super) fn create_native(&self, selection: AdapterSelection) -> RhiResult<Arc<Dx12Device>> {
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
        // is written against; `platform::facts::probe` records that feature as a
        // structural property of Direct3D 12, so the under-report is no longer the
        // only consistent answer and keeping it would refuse dispatches the device
        // can run.
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

        Ok(std::sync::Arc::new(Dx12Device::new(
            deferred_adapter_info(&candidate, self.instance),
            device,
            spine,
            facts,
            submission,
        )))
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
    ) -> RhiResult<Box<dyn crate::base::platform::DeviceRequestBackend>> {
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

        Ok(Box::new(Dx12Request::new(
            self.create_native(descriptor.selection())?,
        )))
    }
}
