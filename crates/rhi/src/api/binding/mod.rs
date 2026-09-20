//! Bind groups: logical group/slot identity, the binding vocabulary, layout
//! compatibility, and the immutable resource packet (specification sections 20
//! through 22).
//!
//! The core definition this module implements is one sentence of section 20:
//!
//! > `BindGroup` is an immutable resource packet validated against a logical layout.
//!
//! Not a `VkDescriptorSet`, not a D3D12 descriptor table, not a Metal argument
//! buffer, not a GL texture-unit packet. Those are lowerings, and section 20 says
//! so in the same breath as it defines the type.
//!
//! # What this module owns
//!
//! - The logical indexing vocabulary ([`BindGroupIndex`], [`BindingSlotId`]) and
//!   the count vocabulary ([`BindingCount`]). Section 20.1 is explicit that these
//!   are Fluxel logical indices and are *not* native registers, sets, or indices.
//! - The binding kinds ([`BindingKind`]) and their component enums
//!   ([`TextureSampleType`], [`StorageAccess`], [`SamplerKind`],
//!   [`BufferBindingAccess`]).
//! - The capability *query* for bindings ([`BindingSupportQuery`],
//!   [`BindingSupport`], [`BindingLimitClass`]). The answers live in
//!   [`crate::api::capability`]; this module owns only the question, which is what
//!   section 20.4 means by refusing to recreate infinitely many
//!   `supports_storage_texture_cube` booleans.
//! - Layout vocabulary: [`BindingSlot`], [`BindGroupLayoutDescriptor`],
//!   [`BindGroupLayout`], and the two distinct tokens of section 21.1 —
//!   [`BindGroupLayoutCompatibilityId`] (exact, same-Device, unforgeable) and
//!   [`LayoutFingerprint`] (a cache/tooling hint that must never replace
//!   correctness validation).
//! - The packet: [`BindingResource`], [`BindGroupEntry`],
//!   [`BindGroupDescriptor`], [`BindGroup`].
//!
//! # What this module deliberately does not own
//!
//! - Whether the *device* can express a binding. That is
//!   `EnabledCapabilities::binding_support`, and the layout validator takes the
//!   answer as a parameter rather than reading a device.
//! - Aggregate limits such as per-stage resource count and
//!   dynamic-buffers-per-pipeline-layout. Section 20.5 says they "cannot be
//!   determined at an individual BindGroupLayout stage" because they span
//!   multiple groups; they are validated uniformly when a
//!   [`crate::api::pipeline::PipelineInterface`] is created.
//! - Whether a sampler and a sampled texture are legal as a *paired use*. Section
//!   22.3 says that is finally decided by the shader interface and pipeline
//!   validation, so a filtering sampler does not make every texture it is bound
//!   with filterable.
//! - The recorder's dynamic-offset state. Section 22.4 says a dynamic offset does
//!   not modify the [`BindGroup`]; it is part of current command binding state.
//!   This module freezes only the *order* those offsets are consumed in, through
//!   [`BindGroupLayout::dynamic_offset_count`].
//! - [`LayoutFingerprint`]'s algorithm. Section 21.1 fixes the type and the rule
//!   that it is a hint, not a hash function; the value comes from interning.
//!
//! # The invariant this module enforces
//!
//! A layout and a packet are both canonical before they exist. Section 21.2
//! canonicalizes layout entries into ascending [`BindingSlotId`] order and
//! rejects duplicate slots; section 22.2 canonicalizes bind-group entries the
//! same way and makes a duplicate slot `InvalidUsage`. Nothing here sorts,
//! merges, or chooses on a caller's behalf — a duplicate is a refusal, not
//! something to be repaired.
//!
//! Identity comes first, in the sense of section 3.1: every resource check is
//! preceded by a [`DeviceIdentity`](crate::api::identity::DeviceIdentity)
//! comparison, because a resource from another device is a refusal and never a
//! migration.
//!
//! # Files
//!
//! One section per file, with this file holding only the declarations and the
//! re-exports of the *public* types:
//!
//! ```text
//! mod.rs          declarations and re-exports, no rule of its own
//! vocabulary.rs   section 20, the binding vocabulary and its queries
//! layout.rs       section 21, the layout and its validator
//! group.rs        section 22, the packet and its validator
//! ```
//!
//! The re-export list is the module's public contract with the rest of the
//! crate: `crate::api::binding::X` names every public type this chapter defines.
//!
//! The validators are crate-private and are *not* re-exported. Each submodule
//! stays crate-visible rather than private because the validators it owns are
//! crate-private entry points of their own: a `pub(crate) use` of one would be an
//! unused import in a non-test build, since the device verbs that call it are not
//! written yet, and this module does not carry lint attributes as a substitute for
//! a caller — so a crate-internal caller names the file that defines the item, e.g.
//! `crate::api::binding::group::validate_bind_group_descriptor`.

pub(crate) mod backend;
pub(crate) mod group;
pub(crate) mod layout;
pub(crate) mod vocabulary;

pub use group::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource};
pub use layout::{
    BindGroupLayout, BindGroupLayoutCompatibilityId, BindGroupLayoutDescriptor, BindingSlot,
    LayoutFingerprint,
};
pub use vocabulary::{
    BindGroupIndex, BindingCount, BindingKind, BindingLimitClass, BindingSlotId, BindingSupport,
    BindingSupportQuery, BufferBindingAccess, SamplerKind, StorageAccess, TextureSampleType,
};
