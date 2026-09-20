//! What a portable binding kind becomes on Direct3D 12: one register class, and
//! one view.
//!
//! This is the file where section 19.3 stops being a rule about *not* conflating
//! two vocabularies and becomes the lowering that deliberately bridges them. A
//! logical `group`/`slot` is not an HLSL `register`/`space`; the artifact's
//! [`ShaderAbiVersion`](crate::api::shader::ShaderAbiVersion) is what pins the two
//! together, and ABI 1.0 says:
//!
//! ```text
//! space           = group index
//! register number = slot id
//! register class  = BindingKind, by the table below
//! ```
//!
//! | portable kind | register class | view |
//! |---|---|---|
//! | `UniformBuffer` | CBV (`b`) | `D3D12_CONSTANT_BUFFER_VIEW_DESC` |
//! | `StorageBuffer { ReadOnly }` | SRV (`t`) | raw buffer SRV |
//! | `StorageBuffer { ReadWrite }` | UAV (`u`) | raw buffer UAV |
//! | `SampledTexture` | SRV (`t`) | typed texture SRV |
//! | `StorageTexture { ReadOnly }` | SRV (`t`) | typed texture SRV |
//! | `StorageTexture { ReadWrite }` | UAV (`u`) | typed texture UAV |
//! | `Sampler` | `s` | `D3D12_SAMPLER_DESC` on the sampler heap |
//!
//! # Why a storage buffer is a *raw* view
//!
//! `D3D12_BUFFER_SRV`/`UAV` can address a buffer either as a byte range
//! (`Flags = ..._FLAG_RAW`, `Format = R32_TYPELESS`, the HLSL `ByteAddressBuffer`
//! and `RWByteAddressBuffer` pair) or as an array of fixed-size elements
//! (`StructureByteStride != 0`, the HLSL `StructuredBuffer<T>` family). The
//! structured form needs the element stride, and nothing in v13 carries one:
//! section 18.1 makes a buffer byte-addressed with no element type, and
//! [`BindingKind`](crate::api::binding::BindingKind) has no stride field. So the
//! raw form is the only one this ABI can express, and it is the only one this
//! backend builds. A `RWStructuredBuffer` in a shader is therefore *not* ABI 1.0
//! — the shape it needs cannot be described — and the mismatch is a driver
//! validation error rather than a silent misread, which is the direction this
//! crate prefers.
//!
//! # Why one kind maps to one class and not to a choice
//!
//! `BindingKind::StorageBuffer` carries `access`, and the class follows from it
//! rather than from anything the caller passes at bind time. A read-only storage
//! buffer *is* an SRV: presenting it as a UAV would let the shader write through
//! a binding section 22.2 made read-only, which is exactly the class of mistake
//! the access field exists to prevent.

use windows::Win32::Graphics::Direct3D12::{
    D3D12_DESCRIPTOR_RANGE_TYPE, D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
    D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
    D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
};

use crate::api::binding::{BindingKind, BufferBindingAccess, StorageAccess};

/// The register class a portable binding kind occupies.
///
/// A four-valued enum rather than Direct3D 12's `D3D12_DESCRIPTOR_RANGE_TYPE`
/// directly, because three of the four have a second question attached — which
/// `Create*View` call writes them and which heap they live on — and answering it
/// from a raw `i32` would mean matching on a numeric newtype.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RegisterClass {
    /// A `b` register: a constant buffer view.
    ConstantBuffer,
    /// A `t` register: a shader resource view.
    ShaderResource,
    /// A `u` register: an unordered access view.
    UnorderedAccess,
    /// An `s` register. Samplers live on their own heap type.
    Sampler,
}

/// The register class one binding kind occupies.
pub(crate) fn class_of(kind: &BindingKind) -> RegisterClass {
    match kind {
        BindingKind::UniformBuffer { .. } => RegisterClass::ConstantBuffer,
        BindingKind::StorageBuffer { access, .. } => match access {
            BufferBindingAccess::ReadOnly => RegisterClass::ShaderResource,
            BufferBindingAccess::ReadWrite => RegisterClass::UnorderedAccess,
        },
        // A sampled texture is read-only by construction; there is no writable
        // sampled image in v13.
        BindingKind::SampledTexture { .. } => RegisterClass::ShaderResource,
        // The one place the two access vocabularies differ in *shape*: a storage
        // buffer can only be read-only or read-write, while a storage texture
        // also has write-only. Both writable forms are a UAV, which is the class
        // that carries a write at all.
        BindingKind::StorageTexture { access, .. } => match access {
            StorageAccess::ReadOnly => RegisterClass::ShaderResource,
            StorageAccess::WriteOnly | StorageAccess::ReadWrite => RegisterClass::UnorderedAccess,
        },
        BindingKind::Sampler { .. } => RegisterClass::Sampler,
    }
}

impl RegisterClass {
    /// The `D3D12_DESCRIPTOR_RANGE_TYPE` a root-signature range declares.
    pub(crate) fn range_type(self) -> D3D12_DESCRIPTOR_RANGE_TYPE {
        match self {
            Self::ConstantBuffer => D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
            Self::ShaderResource => D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            Self::UnorderedAccess => D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            Self::Sampler => D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        }
    }

    /// The letter a caller sees in a refusal, so the message names the register
    /// class rather than a number.
    pub(crate) fn letter(self) -> char {
        match self {
            Self::ConstantBuffer => 'b',
            Self::ShaderResource => 't',
            Self::UnorderedAccess => 'u',
            Self::Sampler => 's',
        }
    }
}
