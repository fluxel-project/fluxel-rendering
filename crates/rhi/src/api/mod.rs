//! Fluxel RHI API v1 — the frozen portable interface.
//!
//! This tree is written from `documents/rhi-design/01`–`08` and from nothing
//! else. Where the rest of this crate holds an implementation, this module holds
//! the *contract*: the public types, the refusal paths, and the invariants that
//! a backend must lower without being consulted about legality.
//!
//! # Layout
//!
//! Specification section 2 fixes the submodule list. The specification spells
//! the module `rhi`; this crate spells it `api`, and the submodules below are
//! that list verbatim:
//!
//! ```text
//! platform      capability    format        resource      shader
//! binding       pipeline      command       submission    presentation
//! statistics    diagnostics   graph_bridge  tooling (doc-hidden)
//! ```
//!
//! and the backend tree it names is `crate::backend::{dx12, vulkan, metal,
//! webgpu, gl}`.
//!
//! Identity and error are the two exceptions to that list. Sections 3 and 4 sit
//! at the root of module 01, before the platform chapter, because every other
//! module refers to them; they live in [`identity`] and [`error`] here and are
//! re-exported at this module's root so that the path a caller writes matches the
//! path the specification's own examples use.
//!
//! # What this module owns
//!
//! Section 5 of the root specification gives the layering: RenderGraph owns
//! declarations, versions, dependencies, culling, scheduling, logical lifetime,
//! and presentation intent; the recorder owns command-ordered actual uses; the
//! backend owns lowering only. Capability is instance data for an adapter,
//! device, format, surface, or route, and is never inferred from the presence of
//! a Rust trait (root section 3.1).
//!
//! Two rules from the root specification decide most of the shape below:
//!
//! 1. *Any public operation must first perform O(1) identity validation before
//!    touching the backend* (section 3.1), and a problem portable validation can
//!    find may not be handed to a backend for a driver to discover (section 4).
//!    This is why the public verbs here are validating façades rather than thin
//!    wrappers over a native call.
//! 2. *Capability is instance data* (root section 3.1). This is why there is no
//!    `trait ComputeApi` to bound a generic on, and why section 59 lists nine
//!    capability traits that are explicitly absent.
//!
//! # Status
//!
//! The behavioural verbs are being written ahead of their backends: the
//! signatures and the refusal paths land first, so that the call sites in
//! `tests` can be used to review whether the interface is usable before any
//! lowering exists.
//!
//! Two tiers, and the difference is deliberate:
//!
//! - A verb that must reach a backend panics with `unimplemented!()` and a
//!   message naming what is missing. Its contract is fixed; its lowering is not
//!   built.
//! - Plain data — snapshots, descriptors, builders, opaque tokens — is
//!   implemented, because its accessors are forced by its own field list and
//!   writing `unimplemented!()` over `&self.name` would hide the shape the tests
//!   exist to check.
//!
//! The portable *rules* are implemented wherever they are decidable without a
//! backend, even inside an otherwise unbuilt verb. Section 3.1 requires identity
//! validation before any backend call; a verb that panics has nothing to
//! validate, but one that can refuse a wrong-device argument does so before it
//! panics.

#![deny(missing_docs)]

pub mod binding;
pub mod capability;
pub mod command;
pub mod diagnostics;
pub mod error;
pub mod format;
pub mod graph_bridge;
pub mod identity;
pub mod pipeline;
pub mod platform;
pub mod presentation;
pub mod resource;
pub mod shader;
pub mod statistics;
pub mod submission;

// The capture and diagnostic tooling SPI (module 07). Doc-hidden because it is an
// audience statement rather than a stability one: this is what a capture tool
// consumes, not what a rendering caller learns. The semver rule for its types is
// fixed separately by section 47.20.1.
//
// Written as a plain comment rather than an outer doc comment on purpose. Rustdoc
// merges an outer doc on a module declaration with the module's own `//!` doc and
// then resolves every link in the merged block against *this* module — so a bare
// link to SemanticEventId written in `tooling.rs` is looked up in `api` and fails,
// with no source location in the warning. The text that used to live here is in
// `tooling.rs`'s module note.
#[doc(hidden)]
pub mod tooling;

// All seven chapters are written as of 2026-09-20: 01 (platform, capability,
// identity, error), 02 (format, resource), 03 (shader, binding, pipeline), 04
// (command), 05 (submission, presentation), 06 (statistics, diagnostics,
// graph_bridge), and 07 (tooling). Each carries a module note recording how far
// it got and what it could not close; the open items are collected as numbered
// adjudications in `documents/draft/0.16-plan.md`.

pub use error::{RhiError, RhiErrorKind, RhiResult};
pub use identity::{DeviceGeneration, DeviceIdentity, DeviceInstanceId, Label, ObjectId};

#[cfg(test)]
mod tests;
