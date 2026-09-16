//! The common contract's copy addressing, lowered onto Layer 1's.
//!
//! This module owns one thing: turning the addressing half of a
//! `TextureCopyRegion` or a `BufferCopyRegion` into the Layer 1 values the copy
//! verbs take.  It does not validate, and it cannot: Layer 1's providers check a
//! region against the real descriptor, which is the only place a descriptor
//! exists -- the contract's copy verbs hand over an identity, while the
//! `BoundTexture` carrying `descriptor` stays with the executor.
//!
//! # The aspect, and why it is `All`
//!
//! The contract's `TextureCopyRegion` names no aspect, and this crate has no way
//! to find one out for a given identity: Layer 1 exposes no verb that reads a
//! texture's descriptor back, and the descriptor the adapter itself was handed at
//! creation is not retained anywhere.  So the aspect is chosen from what is
//! knowable, and the only value that is not a guess about the resource is
//! [`GlTextureAspect::All`] -- "all aspects supported by the format".  A color
//! texture's all is its color plane; a depth-stencil texture's all is both of its
//! planes.  Claiming `Color` would assert a fact about a resource this module has
//! never seen, and would be wrong for every depth-format copy.
//!
//! Nothing downstream is misled by it, which is worth stating because `All` would
//! otherwise read as a claim about driver behaviour: `.aspect` is read in exactly
//! one place in the whole of Layer 1, the equality check in
//! `validate_texture_copy` (`api/resource.rs:522`), and no provider consults it
//! when choosing a command.  Its whole job is to keep one caller from declaring
//! the source and destination of a copy in different planes.  Both sides here are
//! built by [`texture_region`], so they agree by construction.
//!
//! # One extent, and one layer
//!
//! Both sides are built from the *same* `extent`, because the contract's region
//! carries one extent for both.  That is not a convenience: it is why
//! `validate_texture_copy`'s "the two extents agree" rule cannot be what rejects
//! a copy this module lowered.
//!
//! The layer range is the contract's missing half.  A `TextureCopyRegion` names
//! two mips and no layer, while a Layer 1 subresource names a base layer and a
//! count, so the lowering selects one layer -- `base_layer: 0, layer_count: 1` --
//! which is the whole of what the contract's region can express.  For a 3D texture
//! that pairs with Layer 1's own reading of the third extent as depth
//! (`GlTextureRegion::validate_for`), so a 3D copy passes through unchanged; for an
//! array texture it pins the copy to layer zero, and a region whose third extent
//! is anything but one is then rejected by Layer 1 as out of bounds.  That
//! rejection is the correct outcome and not a limitation to work around: an
//! array copy spanning layers is not something the contract's copy region can ask
//! for, so a backend that accepted it would be accepting a request nobody made.

use crate::webgl2::api::{
    BufferId, GlBufferRange, GlExtent3d, GlTextureAspect, GlTextureRegion, GlTextureSubresource,
    TextureId,
};

/// One side of a texture copy, as Layer 1 addresses it.
pub(super) fn texture_region(
    texture: TextureId,
    mip_level: u32,
    origin: [u32; 3],
    extent: [u32; 3],
) -> GlTextureRegion {
    GlTextureRegion {
        subresource: GlTextureSubresource {
            texture,
            aspect: GlTextureAspect::All,
            mip_level,
            base_layer: 0,
            layer_count: 1,
        },
        origin,
        extent: GlExtent3d {
            width: extent[0],
            height: extent[1],
            depth_or_layers: extent[2],
        },
    }
}

/// One side of a buffer copy, as Layer 1 addresses it.
///
/// The mapping is the identity, and it is written out rather than derived because
/// the two types are the same three numbers in two spellings -- the contract's
/// pair of offsets and one size, Layer 1's pair of offset-and-size ranges.  The
/// size is shared between the two sides for the same reason the extent is.
pub(super) fn buffer_range(buffer: BufferId, offset: u64, size: u64) -> GlBufferRange {
    GlBufferRange {
        buffer,
        offset,
        size,
    }
}
