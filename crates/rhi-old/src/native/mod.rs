//! The native modern family: Direct3D 12, Vulkan and Metal.
//!
//! ```text
//!   native/mod.rs          this statement
//!   native/dx12/           windows crate + gpu-allocator
//!   native/vulkan/         ash + gpu-allocator
//!   native/metal/          extern "C" to Metal / QuartzCore / the objc runtime
//! ```
//!
//! # These three share a layer, and nothing else shares it
//!
//! The native family is a *grouping*, not a shared implementation. Each backend
//! implements API v1's contract directly — [`crate::rhi::platform::DeviceBackend`]
//! and the per-object `*Backend` traits its handles wrap — and owns its own
//! mechanism: barriers, descriptors, encoder model, memory heaps and shader
//! compilation all differ. Which facts are shared (identity, lifetime,
//! submission, completion and quarantine, presentation, capability lowering,
//! structured errors) and which are not is listed in the lead 3F plan.
//!
//! Grouping them here is therefore documentation and module hygiene, not a
//! promise that they have a common parent type. A reader looking for what DX12 and
//! Vulkan share should find it in `crate::rhi`; a reader looking for what they do
//! not share should find it in their own module.
//!
//! # Where this tree stands, stated plainly
//!
//! `vulkan/` was written against the previous `crate::common` contract and is
//! being brought onto API v1; until that lands it is not an API v1 backend, and
//! nothing here should be read as claiming one exists. `dx12/` and `metal/` do
//! not exist yet.
//!
//! # Metal has no Apple dependency, on purpose
//!
//! `metal/` declares the Metal, QuartzCore and objc-runtime symbols it calls with
//! `extern "C"` instead of depending on `objc2-metal` or `metal`. Cargo resolves
//! every target's dependencies into the lockfile even when the local build never
//! compiles them, so adding such a crate would put Apple dependencies into this
//! workspace's dependency graph outright. The whole module is gated on
//! `target_os = "macos"`, so it does not participate in a non-Apple build at all.
//!
//! The cost is `unsafe` at every call site. Per the workspace rule that `unsafe`
//! lives only in owner-private native boundaries with a written safety rationale,
//! each declaration and each `objc_msgSend` carries its own. Because this module
//! is never compiled here, its reviewer is the only thing standing between it and
//! a silent mistake: every signature must be checkable against Apple's published
//! documentation, and every selector must be spelled as the header spells it.
//!
//! Metal is implemented and **unverified**. No Apple compile and no macOS run
//! happens in this workspace, so no claim of Metal support exists — see the
//! `0.16-review.md` entry that registers this as an open gate rather than a
//! closed one.
//!
//! # The GL family is not here
//!
//! It converges onto the same contract, but its implementation idiom is a
//! different one: a stateful API needs desired/applied/unknown reconciliation, and
//! that machinery stays with it. See ADR-0011 and `crate::webgl2` (renamed to
//! `gl_family` in package W6, together with the boundary-check constant that
//! currently keys on the old path).
//!
//! # What dispatch looks like
//!
//! API v1 hides each backend behind the trait object its handle owns: a
//! `Device` holds an `Arc<dyn DeviceBackend>`, a `Buffer` an
//! `Arc<dyn BufferBackend>`. Which backend can be reached is still decided by
//! Cargo features at compile time, so the selection is static even though the
//! object behind a handle is not. There is no `Backend` enum and no dispatch
//! table, and a backend never branches on another backend's identity.

#![allow(
    dead_code,
    reason = "Each backend's pure decision layer lands before its FFI caller, so the caller's \
              existence is what removes this allow. Every item is exercised by tests in the \
              meantime; nothing here is speculative vocabulary."
)]

/// The Direct3D 12 backend, present only when its feature is selected.
#[cfg(all(windows, feature = "dx12"))]
pub(crate) mod dx12;

/// The Vulkan backend, present only when its feature is selected.
#[cfg(all(windows, feature = "vulkan"))]
pub(crate) mod vulkan;

/// The Metal backend.
///
/// Gated on the target, not on a feature: it is built only where Metal exists,
/// and this workspace never builds for such a target.
#[cfg(target_os = "macos")]
pub(crate) mod metal;
