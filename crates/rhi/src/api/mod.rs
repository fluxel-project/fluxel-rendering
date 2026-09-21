//! Fluxel RHI API freeze v13 — the portable interface.
//!
//! This tree follows the focused RHI ADRs and the compact crate design guide.
//! Where the rest of this crate holds an implementation, this module holds
//! the *contract*: the public types, the refusal paths, and the invariants that
//! a backend must lower without being consulted about legality.
//!
//! # Layout
//!
//! The public module tree is intentionally small; this crate spells the RHI
//! root `api`, and the submodules below are its stable vocabulary:
//!
//! ```text
//! platform      capability    format        resource      shader
//! binding       pipeline      command       submission    presentation
//! statistics    diagnostics   tooling (doc-hidden)
//! ```
//!
//! and the backend tree it names is `crate::backend::{dx12, vulkan, metal,
//! webgpu, gl}`.
//!
//! Identity and error sit at the root because every other module refers to
//! them; they live in [`identity`] and [`error`] and are re-exported here.
//!
//! # What this module owns
//!
//! The recorder derives command-ordered actual resource uses from portable
//! commands, and the backend owns lowering only. No upper-layer scheduling
//! contract enters this API. Capability is instance data for an adapter, device,
//! format, surface, or route, and is never inferred from the presence of a Rust
//! trait.
//!
//! Two rules decide most of the shape below:
//!
//! 1. *Any public operation must first perform O(1) identity validation before
//!    touching the backend*, and a problem portable validation can find may not
//!    be handed to a backend for a driver to discover.
//!    This is why the public verbs here are validating façades rather than thin
//!    wrappers over a native call.
//! 2. *Capability is instance data*. This is why there is no `trait ComputeApi`
//!    to bound a generic on.
//!
//! # Status
//!
//! Public verbs are executable implementations rather than interface-only
//! placeholders. Portable validation runs before a backend call, and a backend
//! that cannot lower an otherwise legal request returns a structured error.
//! Plain data — snapshots, descriptors, builders, and opaque tokens — likewise
//! exposes total accessors over its retained state.

#![deny(missing_docs)]

pub mod binding;
pub mod capability;
pub mod command;
pub mod diagnostics;
pub mod error;
pub mod external;
pub mod format;
pub mod identity;
pub mod pipeline;
pub mod platform;
pub mod presentation;
pub mod query;
pub mod resource;
pub mod shader;
pub mod statistics;
pub mod submission;

// Portable implementation details shared by more than one public API domain.
// These are deliberately not part of the exported contract.
pub(crate) mod internal;

// The capture and diagnostic tooling SPI. Doc-hidden because it is an
// audience statement rather than a stability one: this is what a capture tool
// consumes, not what a rendering caller learns. The semver rule for its types is
// fixed by its own semver policy.
//
// Written as a plain comment rather than an outer doc comment on purpose. Rustdoc
// merges an outer doc on a module declaration with the module's own `//!` doc and
// then resolves every link in the merged block against *this* module — so a bare
// link to SemanticEventId written in `tooling.rs` is looked up in `api` and fails,
// with no source location in the warning. The text that used to live here is in
// `tooling.rs`'s module note.
#[doc(hidden)]
pub mod tooling;

// Each public family keeps positive, refusal, and boundary conformance tests
// adjacent to this contract.

pub use error::{RhiError, RhiErrorKind, RhiResult};
pub use identity::{DeviceIdentity, DeviceInstanceId, Label, ObjectId};

#[cfg(test)]
pub(crate) mod tests;
