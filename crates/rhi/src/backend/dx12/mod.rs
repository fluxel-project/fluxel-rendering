//! The Direct3D 12 backend: API v1 lowering onto `ID3D12Device` and DXGI.
//!
//! This module owns exactly one thing: turning an already-validated API v1
//! request into Direct3D 12 objects and calls. It decides nothing about
//! legality. Every descriptor reaching this module has already passed the
//! `Device` façade's identity check, canonicalization, portable validation and
//! capability query, so a refusal produced here must be one only DX12 can know —
//! an allocation that the driver refused, a shader the device rejected, a
//! synchronization object that could not be created. An
//! [`crate::api::RhiErrorKind::InvalidUsage`] returned from this module means the
//! façade missed a portable check, and that is a bug in the façade rather than a
//! fact about the request.
//!
//! # What is private here, and why it must stay private
//!
//! `ID3D12Device`, `ID3D12Resource`, descriptor heaps, root signatures, command
//! allocators, fences, `D3D12_RESOURCE_STATES`, and every memory offset live
//! behind this boundary. None of them is reachable from a public type, a public
//! error, a diagnostic, or a tooling event: API v1's public model is
//! device-identity plus generation, and a native handle in it would make a
//! second, DX12-shaped architecture for the next platform to copy.
//!
//! # Lifetime
//!
//! Direct3D 12 has no device-loss callback and no `VK_ERROR_DEVICE_LOST`
//! equivalent. A removed device reports `DXGI_ERROR_DEVICE_REMOVED` (or
//! `DXGI_ERROR_DEVICE_RESET`) from the next call that touches it, so this
//! backend learns about loss the way it learns about every other failure: from a
//! return code. The mapping from those codes to a terminal device status lives in
//! [`ffi`] so that the one place a `HRESULT` becomes an API v1 error is also the
//! one place that decides whether the device has ended.
//!
//! # What this backend does not own
//!
//! Window creation, the message pump, and native handle lifetime belong to
//! `fluxel-host`; this module consumes a
//! [`crate::api::presentation::PresentationTarget`] and never creates a window or
//! a swapchain-before-request. Format/route facts are *not* invented here — they
//! are read from the device and recorded as instance data, because a capability
//! that is asserted from a trait's presence rather than queried is the exact
//! failure API v1 exists to prevent.
//!
//! # The shape of this tree
//!
//! This backend's directories mirror `crate::api`'s: a chapter per subject, each
//! with a `mod.rs` that declares its files and re-exports the chapter's inside
//! face, and each file named for the one thing it owns. The correspondence is not
//! decoration — it is how a reader moves between a portable rule and the lowering
//! that serves it — and it is not exact either, because Direct3D 12 does not split
//! its work the way API v1 splits its model:
//!
//! - [`platform`] answers API v1's `platform` — the DXGI provider, the logical
//!   device, the device request, and the capability table that device reports.
//! - [`resource`] answers `resource`, and today holds only [`resource`]'s buffer
//!   half.
//! - [`shader`] answers `shader`, and is the smallest chapter because D3D12 has
//!   no shader-module object at all.
//! - [`command`] answers `command` and `submission`: recording, committing and
//!   observing work.
//! - [`binding`] answers `binding`, and owns the descriptor heaps and the bind
//!   groups written into them.
//! - [`pipeline`] answers `pipeline`, and owns root signatures and state objects
//!   — the two portable pipeline verbs that D3D12 has no separate object for.
//! - [`ffi`] answers nothing in API v1: it is the one place a `HRESULT` becomes
//!   an API v1 error, which is why it sits outside the chapter directories.
//!
//! # Where this tree stands, stated plainly
//!
//! **The native boundary ([`ffi`]), the platform chapter ([`platform`]), buffer
//! allocation ([`resource`]), the command spine ([`command`]), and shader entry
//! points ([`shader`]) are written. [`binding`] and [`pipeline`] are declared and
//! not written** — the device chapter's two verbs refuse with a list of what is
//! missing rather than fabricating an object. There is no texture or presentation
//! lowering yet, and no claim of DX12 support exists until the shared contract
//! suite and a real Windows run close on one revision.
//!
//! [`shader`] is the smallest of the written chapters and the one whose size is
//! easiest to misread: Direct3D 12 has no shader-module object, so preparing an
//! entry point is keeping bytes alive for `CreateComputePipelineState` and nothing
//! more. A module that was created has **not** been compiled, and [`shader`] says
//! so at length because the opposite reading is the natural one.
//!
//! The order the rest arrives in is fixed by a dependency rather than by
//! preference. DX12 answers every capability question (`CheckFeatureSupport`,
//! format support, sampler feedback, resource-binding tiers) through a live
//! device, and an `AdapterInfo` carries a capability snapshot that must have no
//! holes in it, so **device creation comes before capability enumeration, and
//! capability enumeration comes before adapter enumeration** — which is why
//! `ProviderBackend::enumerate_adapters` refuses today while `request_device`
//! works (see the note in [`platform::provider`]).
//!
//! Capability enumeration has started: [`platform::facts`] probes a created device
//! and is what `request_device` now records. It is not finished — view
//! compatibility and the last of the binding facts are still absent, and seven of
//! the twenty-seven portable limits have no Direct3D 12 ceiling to cite — and that
//! module's documentation states which of the three kinds each missing entry is.
//!
//! Allocation arrived for buffers only, and it is the smallest lowering this
//! backend will have: one `CreateCommittedResource` per descriptor, with the
//! heap, the initial state and the one creation-time flag argued in [`resource`].
//! The command spine is what turns an allocation into an observable result: one
//! direct queue, one fence, a ring of command-list slots claimed on demand, and
//! the lowering of buffer copies, buffer uploads and buffer readbacks. It lowers
//! nothing else, and refuses a plan naming anything else rather than executing a
//! silently shortened version of it — see [`command`] for why that refusal is the
//! point rather than a gap. Every list it records leaves every buffer it touched
//! in `D3D12_RESOURCE_STATE_COMMON`, which is why there is no persistent resource
//! state tracker to get out of step with the driver.

mod binding;
mod command;
mod ffi;
mod pipeline;
mod platform;
mod resource;
mod shader;
