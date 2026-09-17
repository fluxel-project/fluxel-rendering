//! The native modern family: Direct3D 12, Vulkan and Metal.
//!
//! ```text
//!   native/mod.rs          this statement
//!   native/dx12/           windows + gpu-allocator / range-alloc
//!   native/vulkan/         ash + gpu-allocator
//!   native/metal/          objc2-metal + block2 + objc2-quartz-core
//! ```
//!
//! # These three share a layer, and nothing else shares it
//!
//! The native family is a *grouping*, not a shared implementation. Each backend
//! implements [`crate::common`]'s contract directly and owns its own mechanism:
//! barriers, descriptors, encoder model, memory heaps and shader compilation all
//! differ, and section 3 of the lead 3F plan lists exactly which facts are shared
//! (identity, lifetime, submission, completion and quarantine, presentation,
//! capability lowering, structured errors) and which are not.
//!
//! Grouping them here is therefore documentation and module hygiene, not a
//! promise that they have a common parent type. A reader looking for what DX12 and
//! Vulkan share should find it in `common`; a reader looking for what they do not
//! share should find it in their own module.
//!
//! # The GL family is not here
//!
//! It converges onto the same contract, but its implementation idiom is a
//! different one: a stateful API needs desired/applied/unknown reconciliation, and
//! that machinery stays with it. See ADR-0011 and `crate::webgl2` (renamed to
//! `gl_family` in package W6, together with the boundary-check constant that
//! currently keys on the old path).
//!
//! # What is deliberately absent
//!
//! No `Backend` enum, no dispatch table, no trait object. Dispatch is static, and
//! which backend exists is decided by Cargo features at compile time, exactly as
//! `fluxel-rhi` decides it today.

#![allow(
    dead_code,
    reason = "Each backend's pure decision layer lands before its FFI caller, so the caller's \
              existence is what removes this allow. Every item is exercised by tests in the \
              meantime; nothing here is speculative vocabulary."
)]

/// The Vulkan backend, present only when its feature is selected.
#[cfg(all(windows, feature = "vulkan"))]
pub(crate) mod vulkan;
