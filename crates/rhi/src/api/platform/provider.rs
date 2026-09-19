//! Adapter discovery and the provider that owns it (specification section 5.2
//! through 5.6).
//!
//! This module owns what a caller may learn about the hardware *before* a device
//! exists: which backend family a provider speaks, what adapters it can name, and
//! what those adapters can do. It deliberately does not own device creation
//! ([`super::request`]) or the capability vocabulary itself
//! ([`crate::api::capability`]) — a discovery snapshot answers questions in that
//! vocabulary without defining it.

use crate::api::capability::AvailableCapabilities;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::platform::request::{DeviceRequest, DeviceRequestDescriptor};
use crate::api::presentation::PresentationTarget;

/// The backend family a provider or device speaks.
///
/// Section 5.2 restricts this to four uses — diagnostics, selection and capture
/// provenance, backend-specific shader acceptance, and tooling UI — and forbids
/// the one use it most invites:
///
/// ```text
/// if device.backend() == BackendKind::Vulkan {
///     // assume feature X exists
/// }
/// ```
///
/// That is wrong even when it happens to be true today, because the same backend
/// family exposes different capabilities on different drivers, and because a
/// portable caller must keep working when a backend is added. Capability
/// questions are asked of the device:
///
/// ```text
/// if device.capabilities().supports_feature(feature) { ... }
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// Direct3D 12.
    Dx12,
    /// Vulkan.
    Vulkan,
    /// Metal.
    Metal,
    /// WebGPU.
    WebGpu,
    /// A desktop OpenGL context (WGL or EGL).
    OpenGl,
    /// A WebGL2 context.
    WebGl2,
}

/// Provider-scoped adapter identity.
///
/// Opaque: a caller may compare, hash, and print it, but cannot construct one
/// (section 3, `design-rhi.md:L140`). Section 5.4 fixes what it is *not* — not an
/// enumeration index, not a native pointer, LUID, or `VkPhysicalDevice`, and not
/// a persistent cross-process hardware ID — and what it is guaranteed for: being
/// passed back to the same provider that produced it.
///
/// The two private fields exist because that guarantee has to be decidable
/// rather than assumed. Section 5.4 makes passing one provider's `AdapterId` to
/// another `InvalidUsage`, and section 3.1 requires that decision in O(1) before
/// any backend call, so the producing provider's identity travels inside the
/// token. Keeping it there is what lets [`PlatformProvider::supports_presentation`]
/// refuse a foreign adapter without asking a driver whether the number means
/// anything.
///
/// A future pipeline-cache or persistent adapter preference needs a hardware
/// fingerprint; section 5.4 says that is frozen separately and does not reuse
/// this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AdapterId {
    /// Identity of the provider that minted this adapter.
    provider: u64,
    /// Provider-chosen serial, unique within that provider.
    serial: u64,
}

impl AdapterId {
    /// Mints the identity of one adapter discovered by one provider.
    ///
    /// Crate-private: section 3 forbids a caller constructing a token, and only
    /// the provider that owns the adapter can know either half.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "called by the contract tests; adapter enumeration is what mints one"
        )
    )]
    pub(crate) fn new(provider: u64, serial: u64) -> Self {
        Self { provider, serial }
    }
}

/// A discovery snapshot of one adapter.
///
/// Every field is private and read through an accessor, because section 5.6
/// warns against exactly the use the raw fields invite: an enumeration index, a
/// name, or a vendor/device pair is not a stable cross-run key. They are
/// diagnostics, shown to a user choosing an adapter and recorded in capture
/// provenance.
#[derive(Clone, Debug)]
pub struct AdapterInfo {
    id: AdapterId,
    name: String,
    backend: BackendKind,
    /// Present only when the provider can safely provide it.
    vendor_id: Option<u32>,
    /// Present only when the provider can safely provide it.
    device_id: Option<u32>,
    /// What the adapter can do, not what a device ultimately enables.
    available: AvailableCapabilities,
}

impl AdapterInfo {
    /// Assembles a discovery snapshot.
    ///
    /// Crate-private: snapshots come from a provider's enumeration, and a
    /// caller-built one would describe hardware that was never probed.
    #[expect(
        dead_code,
        reason = "nothing calls it: adapter enumeration, the only producer of a snapshot, is not written"
    )]
    pub(crate) fn new(
        id: AdapterId,
        name: String,
        backend: BackendKind,
        vendor_id: Option<u32>,
        device_id: Option<u32>,
        available: AvailableCapabilities,
    ) -> Self {
        Self {
            id,
            name,
            backend,
            vendor_id,
            device_id,
            available,
        }
    }

    /// This adapter's provider-scoped identity.
    pub fn id(&self) -> AdapterId {
        self.id
    }

    /// The adapter's human-readable name, for diagnostics and UI.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The backend family this adapter belongs to.
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    /// The PCI vendor ID, when the provider can supply it safely.
    pub fn vendor_id(&self) -> Option<u32> {
        self.vendor_id
    }

    /// The PCI device ID, when the provider can supply it safely.
    pub fn device_id(&self) -> Option<u32> {
        self.device_id
    }

    /// What this adapter can do, as facts rather than promises.
    ///
    /// These are `AvailableOnAdapter` facts, not what a device ends up enabling.
    /// Section 7.2 requires that distinction be visible: on a platform where a
    /// format needs explicit feature enablement, an adapter may report a format
    /// as available while the device that enables fewer features reports it as
    /// unavailable. Correctness therefore always reads
    /// [`crate::api::platform::Device::capabilities`], never this snapshot.
    pub fn available_capabilities(&self) -> &AvailableCapabilities {
        &self.available
    }
}

/// How a device request should choose among the provider's adapters.
///
/// The two preference variants are preferences, not guarantees: a system may have
/// no discrete GPU to prefer, or no integrated one to fall back to.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterSelection {
    /// The provider, operating system, or browser selects the candidate.
    Default,
    /// Ask for the higher-performance candidate. Not a guarantee of a discrete
    /// GPU.
    PreferHighPerformance,
    /// Ask for the lower-power candidate. Not a guarantee of an integrated GPU.
    PreferLowPower,
    /// A specific adapter, named by the provider's optional enumeration.
    Explicit(AdapterId),
}

/// Backend-private state behind a [`PlatformProvider`].
///
/// Only the facts the portable rules must decide *without* a backend are kept
/// here: which family this provider speaks, and which provider an [`AdapterId`]
/// must belong to in order for a call to be legal. Everything a real provider
/// holds — the native instance, the enumerated adapters, the device factory —
/// arrives with the backend port and stays behind this type.
#[derive(Clone)]
struct ProviderState {
    backend: BackendKind,
    identity: u64,
}

/// One backend family's entry point for adapter discovery and device creation.
///
/// A provider corresponds to exactly one family (section 5). It may create more
/// than one device: section 3.1 permits several independent logical devices of
/// the same backend, and each request that succeeds gets its own
/// [`crate::api::DeviceIdentity`]. The provider itself is therefore a factory
/// held by shared handle rather than a singleton.
///
/// The portable core does not expose a constructor (section 5.1): a provider is
/// created by Fluxel's host or platform integration, because that is where the
/// native instance it wraps becomes available. Nothing here names `HWND`,
/// `IDXGIAdapter`, `VkInstance`, `CAMetalLayer`, a `GPU` object, or a
/// `WebGLRenderingContext` — section 5.1 keeps those on the integration side of
/// the seam.
#[derive(Clone)]
pub struct PlatformProvider {
    state: ProviderState,
}

impl PlatformProvider {
    /// Wraps one backend family's native instance.
    ///
    /// Crate-private: only the host/provider integration that owns the native
    /// instance may call this (section 5.1).
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "called by the contract tests; the host integration that owns a native instance is not written"
        )
    )]
    pub(crate) fn new(backend: BackendKind, identity: u64) -> Self {
        Self {
            state: ProviderState { backend, identity },
        }
    }

    /// The backend family this provider speaks.
    pub fn backend(&self) -> BackendKind {
        self.state.backend
    }

    /// Attempts to enumerate the adapters this provider can expose explicitly.
    ///
    /// Three outcomes, and the first two are deliberately distinct:
    ///
    /// ```text
    /// Ok(Some(list))  the provider supports portable enumeration
    /// Ok(None)        the provider does not expose enumeration at all;
    ///                 WebGPU and adopted-context providers may legitimately
    ///                 be in this state
    /// Err(..)         enumeration itself failed
    /// ```
    ///
    /// `Some(vec![])` is a fourth state and is not the same as `None`: the
    /// provider can enumerate, and currently has no candidate adapter.
    ///
    /// This is not a prerequisite for [`Self::request_device`]. A caller that
    /// only wants a device — the common case — never calls it, which is what
    /// lets a provider that cannot enumerate stay fully usable.
    pub fn enumerate_adapters(&self) -> RhiResult<Option<Vec<AdapterInfo>>> {
        unimplemented!(
            "adapter enumeration arrives with the backend port; the contract is \
             fixed, the probing is not built"
        )
    }

    /// Performs presentation preflight for one of this provider's adapters.
    ///
    /// For adapter pickers and diagnostics only. A device request that needs
    /// presentation must still carry the target in its
    /// [`DeviceRequestDescriptor`] — a preflight that said yes is not a
    /// substitute, because the device that is finally created is what has to
    /// present.
    pub fn supports_presentation(
        &self,
        adapter: AdapterId,
        target: &PresentationTarget,
    ) -> RhiResult<bool> {
        // Section 5.4: an `AdapterId` is guaranteed only to be passed back to
        // the provider that produced it, and section 3.1 requires the portable
        // checks to run in O(1) before any backend call. This one is portable,
        // so it is decided here rather than left for a driver to notice.
        if adapter.provider != self.state.identity {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "adapter belongs to a different provider",
            )
            .at("PlatformProvider::supports_presentation"));
        }
        let _ = target;
        unimplemented!(
            "presentation preflight arrives with the backend port; the contract \
             is fixed, the probing is not built"
        )
    }

    /// The canonical path to a device.
    ///
    /// Returns a [`DeviceRequest`] rather than a device, because creation may be
    /// genuinely asynchronous on the platforms this crate serves.
    pub fn request_device(&self, desc: DeviceRequestDescriptor) -> RhiResult<DeviceRequest> {
        let _ = desc;
        unimplemented!(
            "device creation arrives with the backend port; the contract is \
             fixed, the lowering is not built"
        )
    }
}
