//! Lowering a compiled resource requirement onto a Layer 1 creation descriptor.
//!
//! A transient is the one resource the adapter creates rather than receives, so
//! this is the only place the two contracts' resource vocabularies have to
//! agree, and they agree in neither direction by accident.  Three of the
//! differences are decisions rather than renames, and each is made here once:
//!
//! - **`array_layers` is a count in one contract and a target in the other.**
//!   The common contract folds array-ness into a layer count on a `D2`
//!   descriptor, while the GL family makes an array its own target with its own
//!   sampling rules.  The layer count therefore decides the target, and the
//!   third extent component is whatever that target makes it -- texels for a
//!   volume, slices for an array.  Layer 1 keeps both in one
//!   `GlExtent3d::depth_or_layers` for the same reason.
//! - **The GL usage set is coarser than the common operation set.**  One GL bit
//!   covers shader storage in both directions, and one covers both sides of an
//!   attachment.  Lowering the request is a union; deriving what the physical
//!   object permits is therefore *not* its inverse, and the returned set is
//!   wider than the request in exactly those two places.  That is what "must
//!   describe the physical object, not merely echo the compiled requirement"
//!   (`rendergraph/src/backend/resource.rs`) asks for.
//! - **One operation has no GL counterpart at all.**  `Present` says what
//!   happens to an image *after* execution; it is not a creation fact of any GL
//!   texture and no usage bit records it.  A request that is nothing but
//!   presentation has no GL descriptor to lower to, and this module says so
//!   instead of creating a texture with an empty usage set that Layer 1 would
//!   then reject with a message about the wrong layer.
//!
//! Format correspondence is not duplicated here.  `common_format` already owns
//! the one statement of which GL formats the common contract can name, and the
//! reverse is read off it rather than written beside it: two tables that must
//! agree are two tables that can disagree.

use fluxel_rendergraph::{
    BufferDesc, BufferUsage, BufferUsageKind, PhysicalResourceIdentity, TextureDesc,
    TextureDimension, TextureFormat, TextureUsage, TextureUsageKind,
};

use crate::webgl2::api::{
    GlBufferDesc, GlBufferUsage, GlError, GlExtent3d, GlFormat, GlTextureDesc, GlTextureDimension,
    GlTextureUsage,
};

use super::super::capabilities::{common_format, is_depth_stencil};

/// The formats whose correspondence this adapter reads in the reverse direction.
///
/// Not a second table: it names the GL side of the mapping `common_format`
/// already states, and [`gl_format`] reads the mapping itself out of that
/// function.  A GL format absent from this list is one `common_format` answers
/// `None` for, and adding one here without adding it there changes nothing.
const GL_FORMATS: [GlFormat; 4] = [
    GlFormat::Rgba8Unorm,
    GlFormat::Rgba8Srgb,
    GlFormat::Rgba16Float,
    GlFormat::Depth32Float,
];

/// The GL format a common format lowers to, where the correspondence has one.
pub(super) fn gl_format(format: TextureFormat) -> Option<GlFormat> {
    GL_FORMATS
        .into_iter()
        .find(|candidate| common_format(*candidate) == Some(format))
}

/// The physical identity the common contract reports for one Layer 1 object.
///
/// Layer 1's allocation table owns both halves: a slot is stable for the life of
/// an allocation and its generation advances every time the slot is reused
/// (`api/object.rs`).  Together they name one physical generation, which is what
/// the contract asks this value to be.
///
/// Uniqueness across a context recreation is deliberately not this value's job.
/// The table resets with the epoch, so a slot and a generation can repeat in a
/// new context -- and that is harmless, because a context recreation also
/// changes the `DeviceIdentity` the resource is bound to, and the contract
/// rejects a bound resource by device before it ever compares resource
/// identities.
pub(super) fn resource_identity(slot: u32, generation: u32) -> PhysicalResourceIdentity {
    PhysicalResourceIdentity::new(((generation as u64) << 32) | slot as u64)
}

/// Lowers one compiled texture requirement onto a Layer 1 creation descriptor.
pub(super) fn texture_descriptor(
    descriptor: TextureDesc,
    usage: TextureUsage,
) -> Result<GlTextureDesc, GlError> {
    let format = gl_format(descriptor.format).ok_or(GlError::Unsupported {
        operation: "create-transient-texture",
        reason: "the GL family has no texture format for the requested pixel format",
    })?;
    let bits = texture_usage_bits(usage);
    if bits.is_empty() {
        return Err(GlError::Unsupported {
            operation: "create-transient-texture",
            reason: "every requested operation is a statement about the image after execution, so no GL texture usage is left to create it with",
        });
    }
    let (dimension, depth_or_layers) = target(&descriptor)?;
    Ok(GlTextureDesc {
        dimension,
        extent: GlExtent3d {
            width: descriptor.extent.width,
            height: descriptor.extent.height,
            depth_or_layers,
        },
        mip_level_count: descriptor.mip_levels,
        sample_count: descriptor.sample_count,
        format,
        usage: bits,
    })
}

/// Lowers one compiled buffer requirement onto a Layer 1 creation descriptor.
///
/// Total on the operation set: every common buffer operation has a GL role, so
/// the only way to an empty descriptor is a request with no operations in it at
/// all, and that is refused here rather than left for Layer 1's validator to
/// describe as an empty usage set.
pub(super) fn buffer_descriptor(
    descriptor: BufferDesc,
    usage: BufferUsage,
) -> Result<GlBufferDesc, GlError> {
    let bits = buffer_usage_bits(usage);
    if bits.is_empty() {
        return Err(GlError::Unsupported {
            operation: "create-transient-buffer",
            reason: "a physical buffer is created for some operation, and the request named none",
        });
    }
    Ok(GlBufferDesc {
        size: descriptor.size,
        usage: bits,
    })
}

/// The operations a physical texture created this way actually permits.
///
/// Wider than the request in the two places the GL usage set is coarser, and
/// narrower in none: a GL texture created with a usage bit really does permit
/// every common operation that bit stands for.  Which side of an attachment the
/// physical object is comes from its format rather than from the request,
/// because the GL bit does not distinguish the sides -- the same reason the
/// capability lowering reads the side off the format when the table's
/// `renderable` fact cannot say.
pub(super) fn texture_usage(usage: GlTextureUsage, format: GlFormat) -> TextureUsage {
    let mut derived = TextureUsage::empty();
    if usage.contains(GlTextureUsage::SAMPLED) {
        derived = derived.with(TextureUsageKind::Sampled);
    }
    if usage.contains(GlTextureUsage::STORAGE_BINDING) {
        derived = derived
            .with(TextureUsageKind::StorageRead)
            .with(TextureUsageKind::StorageWrite);
    }
    if usage.contains(GlTextureUsage::RENDER_ATTACHMENT) {
        derived = derived.with(if is_depth_stencil(format) {
            TextureUsageKind::DepthStencilAttachment
        } else {
            TextureUsageKind::ColorAttachment
        });
    }
    if usage.contains(GlTextureUsage::COPY_SOURCE) {
        derived = derived.with(TextureUsageKind::CopySource);
    }
    if usage.contains(GlTextureUsage::COPY_DESTINATION) {
        derived = derived.with(TextureUsageKind::CopyDestination);
    }
    derived
}

/// The operations a physical buffer created this way actually permits.
///
/// The one widening is shader storage, reported in both directions because GL's
/// single storage bit does not distinguish them -- the same normalization the
/// native `Buffer::allowed_usage` documents ("buffer storage write is reported
/// as storage read/write").  The mapping roles have no common counterpart: a
/// transient is device-local, and the common contract has no operation for a CPU
/// mapping.
pub(super) fn buffer_usage(usage: GlBufferUsage) -> BufferUsage {
    let mut derived = BufferUsage::empty();
    if usage.contains(GlBufferUsage::UNIFORM) {
        derived = derived.with(BufferUsageKind::Uniform);
    }
    if usage.contains(GlBufferUsage::STORAGE) {
        derived = derived
            .with(BufferUsageKind::StorageRead)
            .with(BufferUsageKind::StorageWrite);
    }
    if usage.contains(GlBufferUsage::VERTEX) {
        derived = derived.with(BufferUsageKind::Vertex);
    }
    if usage.contains(GlBufferUsage::INDEX) {
        derived = derived.with(BufferUsageKind::Index);
    }
    if usage.contains(GlBufferUsage::INDIRECT) {
        derived = derived.with(BufferUsageKind::Indirect);
    }
    if usage.contains(GlBufferUsage::COPY_SOURCE) {
        derived = derived.with(BufferUsageKind::CopySource);
    }
    if usage.contains(GlBufferUsage::COPY_DESTINATION) {
        derived = derived.with(BufferUsageKind::CopyDestination);
    }
    derived
}

/// The GL dimension and the third extent component, which are one decision.
///
/// See the module documentation for why they cannot be chosen apart.  The
/// wildcard is the whole answer for the combinations the GL family has no target
/// for: a layered one-dimensional texture, a volume with layers, and a layer
/// count of zero, which no dimension can carry.
fn target(descriptor: &TextureDesc) -> Result<(GlTextureDimension, u32), GlError> {
    let layers = descriptor.array_layers;
    match descriptor.dimension {
        TextureDimension::D1 if layers == 1 => Ok((GlTextureDimension::D1, 1)),
        TextureDimension::D2 if layers == 1 => Ok((GlTextureDimension::D2, 1)),
        TextureDimension::D2 if layers > 1 => Ok((GlTextureDimension::D2Array, layers)),
        TextureDimension::D3 if layers == 1 => {
            Ok((GlTextureDimension::D3, descriptor.extent.depth))
        }
        _ => Err(GlError::Unsupported {
            operation: "create-transient-texture",
            reason: "the requested dimension and array-layer count describe a texture the GL family has no target for",
        }),
    }
}

/// The GL usage bits the requested operations need.
fn texture_usage_bits(usage: TextureUsage) -> GlTextureUsage {
    let mut bits = GlTextureUsage::EMPTY;
    for kind in [
        TextureUsageKind::Sampled,
        TextureUsageKind::StorageRead,
        TextureUsageKind::StorageWrite,
        TextureUsageKind::ColorAttachment,
        TextureUsageKind::DepthStencilAttachment,
        TextureUsageKind::CopySource,
        TextureUsageKind::CopyDestination,
        TextureUsageKind::Present,
    ] {
        if usage.contains(kind) {
            if let Some(bit) = gl_texture_bit(kind) {
                bits = bits | bit;
            }
        }
    }
    bits
}

/// The GL usage bit one common texture operation needs, where it needs one.
///
/// `None` is the presentation case the module documentation describes.  The
/// wildcard is for an operation this contract version does not name: an
/// unnameable operation is not evidence for a usage bit, and leaving it out is
/// how a request keeps the fail-closed direction -- the descriptor that results
/// carries only what was recognized, and a request with nothing recognizable
/// left is refused.
fn gl_texture_bit(kind: TextureUsageKind) -> Option<GlTextureUsage> {
    match kind {
        TextureUsageKind::Sampled => Some(GlTextureUsage::SAMPLED),
        TextureUsageKind::StorageRead | TextureUsageKind::StorageWrite => {
            Some(GlTextureUsage::STORAGE_BINDING)
        }
        TextureUsageKind::ColorAttachment | TextureUsageKind::DepthStencilAttachment => {
            Some(GlTextureUsage::RENDER_ATTACHMENT)
        }
        TextureUsageKind::CopySource => Some(GlTextureUsage::COPY_SOURCE),
        TextureUsageKind::CopyDestination => Some(GlTextureUsage::COPY_DESTINATION),
        _ => None,
    }
}

/// The GL usage bits the requested operations need.
fn buffer_usage_bits(usage: BufferUsage) -> GlBufferUsage {
    let mut bits = GlBufferUsage::EMPTY;
    for kind in [
        BufferUsageKind::Uniform,
        BufferUsageKind::StorageRead,
        BufferUsageKind::StorageWrite,
        BufferUsageKind::Vertex,
        BufferUsageKind::Index,
        BufferUsageKind::Indirect,
        BufferUsageKind::CopySource,
        BufferUsageKind::CopyDestination,
    ] {
        if usage.contains(kind) {
            if let Some(bit) = gl_buffer_bit(kind) {
                bits = bits | bit;
            }
        }
    }
    bits
}

/// The GL usage bit one common buffer operation needs, where it needs one.
fn gl_buffer_bit(kind: BufferUsageKind) -> Option<GlBufferUsage> {
    match kind {
        BufferUsageKind::Uniform => Some(GlBufferUsage::UNIFORM),
        BufferUsageKind::StorageRead | BufferUsageKind::StorageWrite => {
            Some(GlBufferUsage::STORAGE)
        }
        BufferUsageKind::Vertex => Some(GlBufferUsage::VERTEX),
        BufferUsageKind::Index => Some(GlBufferUsage::INDEX),
        BufferUsageKind::Indirect => Some(GlBufferUsage::INDIRECT),
        BufferUsageKind::CopySource => Some(GlBufferUsage::COPY_SOURCE),
        BufferUsageKind::CopyDestination => Some(GlBufferUsage::COPY_DESTINATION),
        _ => None,
    }
}
