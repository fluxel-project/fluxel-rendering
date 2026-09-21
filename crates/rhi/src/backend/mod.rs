//! The platform backends: one module per native API.
//!
//! `01:126-132` fixes these names — `crate::backend::{dx12, vulkan, metal,
//! webgpu, gl}` — and states the rule that shapes everything in them: "Backend
//! private objects must not be returned from the portable API." A backend
//! implements the crate-private contracts beside the corresponding
//! [`crate::api`] domains and decides nothing about legality;
//! read that module's four disciplines before adding one.
//!
//! # Why this module is crate-private
//!
//! The same reading of
//! `08-governance-freeze-checklist.md` section 59: the rule bans *exporting* the
//! capability traits and native types, and a `pub(crate)` module exports
//! nothing. What it protects is that a caller never names a backend type, never
//! picks a trait to bound on, and never learns the platform in order to write
//! correct code.
//!
//! Section 5.1 says a provider is created by Fluxel host/platform integration.
//! Native APIs with process-owned discovery have narrow crate-root composition
//! functions such as [`crate::create_dx12_provider`] and
//! [`crate::create_vulkan_provider`]. They return only the portable
//! [`crate::api::platform::PlatformProvider`]. Context-adopting APIs (GL and
//! browser canvas/context integration) remain behind their host bridge until a
//! portable host-composition boundary can be stated without exporting a native
//! context, session, or token.
//!
//! # Dependency direction
//!
//! ```text
//! crate::api      the frozen portable contract; names no backend
//! crate::api/*/backend.rs   private implementation contracts; name no native handle
//! crate::backend  lowering; the only place a native handle exists
//! ```
//!
//! Lowering depends on the seam and on the contract, never the reverse. A
//! `crate::api` item that mentions a backend name is a defect however convenient
//! it is: it would make the next platform's architecture a copy of this one's.

/// Test-only building blocks shared by native hardware conformance fixtures.
///
/// A fixture still owns its backend-native provider creation and shader
/// artifact: those are deliberately private and code-form specific. The
/// observed behaviour after a portable device exists, however, is shared.
#[cfg(test)]
#[path = "../../tests/common/mod.rs"]
pub(crate) mod conformance;

/// Platform-neutral conformance execution and result classification.
///
/// This remains test-only because it composes crate-private provider fixtures;
/// its workload vocabulary is public-RHI-only and contains no native handle.
#[cfg(test)]
#[path = "../../tests/harness/mod.rs"]
pub(crate) mod test_harness;

// Each backend is behind both its feature and its target. The target half is not
// redundant: `dx12` is in `default`, so a Linux or wasm build enables the feature
// without having the binding crate at all.
#[cfg(all(feature = "dx12", windows))]
pub(crate) mod dx12;

#[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
pub(crate) mod vulkan;

#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub(crate) mod metal;

// GL-family lowering is split by profile below one backend-private state
// machine.  `gl-family` contains only portable state/probe logic; WGL, EGL and
// browser context ownership are enabled by their narrower provider features.
#[cfg(feature = "gl-family")]
pub(crate) mod gl;

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
pub(crate) mod webgpu;
