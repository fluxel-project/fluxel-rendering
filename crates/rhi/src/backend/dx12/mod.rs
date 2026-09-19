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
//! # Where this tree stands, stated plainly
//!
//! **The native boundary ([`ffi`]) and the provider ([`provider`]) are written.**
//! There is no resource, shader, pipeline, command or presentation lowering yet,
//! and no claim of DX12 support exists until the shared contract suite and a real
//! Windows run close on one revision.
//!
//! The order the rest arrives in is fixed by a dependency rather than by
//! preference. DX12 answers every capability question (`CheckFeatureSupport`,
//! format support, sampler feedback, resource-binding tiers) through a live
//! device, and an `AdapterInfo` carries a capability snapshot that must have no
//! holes in it, so **device creation comes before capability enumeration, and
//! capability enumeration comes before adapter enumeration** — which is why
//! `ProviderBackend::enumerate_adapters` refuses today while `request_device`
//! works (see the note in [`provider`]).

mod ffi;
mod provider;
