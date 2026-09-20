//! Bindings on Direct3D 12: the shader-visible descriptor heaps, and the bind
//! groups that write into them.
//!
//! # Two portable verbs, one native chapter
//!
//! Section 22's `BindGroupLayout` and `BindGroup` have no native counterpart of
//! their own here. Direct3D 12's descriptor *tables* are declared by the root
//! signature — a pipeline property, built in [`crate::backend::dx12::pipeline`]
//! from the whole ordered group sequence — and a bind group is the act of writing
//! views into a range of a descriptor heap. So this chapter owns the second half:
//! the heaps, the slot allocation, and the per-entry view writes that make a heap
//! range mean what a layout said it means.
//!
//! # The mapping, decided once
//!
//! `space = group index`, `register number = slot id`, `register class =
//! BindingKind`: read-write storage buffers and storage textures as UAVs,
//! read-only storage buffers and sampled textures as SRVs, uniform buffers as
//! CBVs, and samplers on their own heap because D3D12 keeps sampler descriptors
//! in a separate heap type that no CBV/SRV/UAV copy can mix with.
//!
//! That mapping is what makes [`crate::api::shader`]'s rule true in both
//! directions: section 19.3 states that a logical `group`/`slot` is *not* a
//! Vulkan descriptor set, an HLSL `register`/`space`, a Metal index or a GL
//! binding point, and here it is deliberately lowered onto the HLSL pair rather
//! than being made identical to it. The artifact's `ShaderAbiVersion` is what
//! pins the pair together across the crate boundary.
//!
//! # Ownership
//!
//! A native descriptor holds an *address*, not a reference. Section 22.2 makes
//! the bind group the owner of everything it binds, which on this backend is
//! literal: the object a bind group returns keeps an `Arc` to every resource it
//! wrote an address for, and returns its heap slots only when the last handle to
//! it is gone.
//!
//! Buffer, sampled-texture, read-only storage-texture, and sampler descriptors
//! are copied into the device-wide shader-visible heaps. Read-write texture UAV
//! lowering remains deliberately refused until its per-view descriptor rules are
//! implemented.

mod group;
mod heap;
pub(crate) mod layout;
pub(crate) mod vocabulary;

pub(crate) use group::{Dx12BindGroup, create_bind_group};
pub(crate) use heap::DescriptorHeap;
