//! The common layer: one RHI-semantics contract that every backend implements.
//!
//! # What this layer is
//!
//! `common` is the only place in this crate where a platform-agnostic statement
//! about GPU execution is written down. Five backends implement it: Direct3D 12,
//! Vulkan and Metal in [`crate::native`], the browser WebGPU adapter, and the GL
//! family. Lead 3F's boundary is that `common` names none of them: it may not
//! mention `ash`, `windows`, `objc2`, `web_sys`, `glow`, or any handle, pointer
//! or extension object those crates define.
//!
//! # What it may contain, and what it may not
//!
//! The contract is stated at the level RenderGraph and the RHI already require:
//! resources with usage and access state, passes with declared attachments, draw
//! / dispatch / copy at recipe granularity, completion, presentation, and
//! capability facts. It deliberately contains **no** descriptor set, no pipeline
//! barrier, no root signature and no encoder-hazard type. Those are a backend's
//! own business:
//!
//! - DX12 records with `ResourceBarrier`, root signatures and descriptor heaps;
//! - Vulkan records with `vkCmdPipelineBarrier`, descriptor sets and command
//!   buffers;
//! - Metal records with encoder boundaries and its own hazard tracking;
//! - the GL family keeps its desired/applied/unknown state machine, because that
//!   is what a stateful API needs.
//!
//! This is the difference between converging on the RHI's semantics and building
//! a lowest-common-denominator Vulkan. A change that would make one backend
//! simulate another's mechanism has overshot.
//!
//! # The vocabulary this layer owns, and the vocabulary it consumes
//!
//! `fluxel-rendergraph` already owns the *portable* vocabulary, and this layer
//! consumes it rather than restating it: `TextureFormat`, `IndexFormat`,
//! `Extent3d`, `TextureDimension`, `TextureDesc`, `BufferDesc`,
//! `ResourceAccessState`, `TextureFormatCapabilities`, `DeviceCapabilities`,
//! `DeviceLimits`, `LoadOp` / `StoreOp`, `Viewport`, `ScissorRect`, and the
//! read/write use enums in `rendergraph/src/access.rs`.
//!
//! What is left for this layer is the vocabulary the portable contract
//! deliberately does not model — the facts a backend needs and a graph must never
//! be able to depend on: vertex formats and step modes, sampler descriptors,
//! bind-group layout entries, pipeline state, attachment operations, native
//! per-format capability bits, copy footprints, surface configuration, and the
//! adapter limits and features.
//!
//! The rule that separates them: **if a value would let a graph depend on a
//! backend fact, it belongs in `rendergraph`, not here.**
//!
//! # Capability domains are traits, not optional methods
//!
//! The contract is not one wide interface with methods a backend may decline.
//! It is a small required floor plus one trait per optional domain, and the
//! availability of a domain is a property of the backend *type*:
//!
//! - the floor trait is what every backend implements, and it carries only what
//!   all five can do;
//! - each optional domain — compute, storage buffers, storage images, indirect
//!   draw, indirect dispatch, multi-draw, multiview, occlusion queries, elapsed
//!   and timestamp queries, base vertex, first instance, anisotropic filtering —
//!   is its own trait, implemented by the backends that have it;
//! - a caller that needs a domain is bounded on that domain's trait, so a
//!   backend without it is *structurally* unable to be asked rather than
//!   answering a refusal at run time.
//!
//! [`caps::Capability`] has exactly one row per domain trait, so the render graph
//! asks one question about a device and gets one answer that names a batch of API.
//! The GL family already proved this shape: `GlStateBackend` is its floor,
//! `GlOptionalComputeBackend` and `GlOptionalIndirectBackend` are domains, and
//! `NoCompute` / `WithCompute` witnesses make the choice compile-time. Lead 3F
//! promotes that structure rather than replacing it with a feature-flag struct.
//!
//! Two refusals stay distinct, and the sentence a caller reads depends on which
//! one happened:
//!
//! 1. **The backend type has no such domain.** The vocabulary is absent, so no
//!    request could have been made and no call site exists.
//! 2. **The backend has the domain and this context did not prove it.** The
//!    vocabulary exists, the row is unproved, and the refusal happens before any
//!    object, extension or command side effect.
//!
//! Collapsing these into one "unsupported" answer is what forces a lowest
//! common denominator: a caller can no longer tell a machine that will never
//! support a domain from one that has not been shown to support it yet.

//! # The one rule every fact obeys
//!
//! A capability is reported only where discovery proved it, and every unproved or
//! absent fact keeps the value that rejects work. The GL family already encodes
//! this as a ledger of evidence plus a probe outcome rather than as a set of
//! booleans copied from the driver ([`caps::CapabilityLedger`]); that shape is
//! promoted here so one lowering can read all five backends.

#![allow(
    dead_code,
    reason = "W2/W3 consume this vocabulary; lead 3F lands it before its consumers by the \
              extraction rule that shared code is never designed before two implementations \
              need it. Every item here is removed from this allow as its consumer arrives."
)]

pub(crate) mod api;
pub(crate) mod base;
pub(crate) mod binding;
pub(crate) mod caps;
pub(crate) mod formats;
pub(crate) mod pipeline;
pub(crate) mod sampler;
pub(crate) mod vertex;
