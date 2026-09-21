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
//! Section 5.1 says a provider "is created by Fluxel host/platform
//! integration", so a host crate does need *some* way in. That entry point is
//! not written yet, and deliberately so: this repository has no consumer that
//! could call it — the harness under `examples/windows-dx12` is written against
//! the pre-rewrite API and does not compile against API v1 — and a public
//! constructor designed with no caller is a guess about its shape. Until a real
//! consumer exists, the way in is `pub(crate)`, and the real-GPU evidence runs
//! as in-crate tests that reach it from inside.
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

// Each backend is behind both its feature and its target. The target half is not
// redundant: `dx12` is in `default`, so a Linux or wasm build enables the feature
// without having the binding crate at all.
#[cfg(all(feature = "dx12", windows))]
pub(crate) mod dx12;

#[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
pub(crate) mod vulkan;

// GL-family lowering is split by profile below one backend-private state
// machine.  `gl-family` contains only portable state/probe logic; WGL, EGL and
// browser context ownership are enabled by their narrower provider features.
#[cfg(feature = "gl-family")]
pub(crate) mod gl;
