//! Upload jobs and readback tickets (specification sections 17.1 through 18.8).
//!
//! The two host-boundary workflows of P0, and the module where the resource
//! chapter's lifetime rules become observable.
//!
//! ```text
//! upload     caller bytes -> Device::create_*_upload -> UploadJob
//!            -> CommandRecorder::encode_upload -> submitted work
//! readback   ReadbackRequest -> CommandRecorder::encode_readback -> ReadbackTicket
//!            -> terminal completion -> ReadbackData (scoped CPU read)
//! ```
//!
//! # Upload is a workflow, not a copy command
//!
//! Section 17 says so to close one loophole: a `UploadJob` may repack caller
//! bytes into a private staging buffer, because section 14.5's
//! [`HostTexelLayout`](crate::api::resource::HostTexelLayout) describes CPU
//! bytes rather than the 256-byte row pitch a native copy footprint wants.
//! What it may *not* do is apply the same trick on
//! the readback side of a command — the repack is host-to-GPU staging that
//! upload itself promises, not a backend silently turning a copy into a
//! round-trip (section 9.4's rule).
//!
//! # Readback is not map-immediately
//!
//! Section 18's chain is the reason [`ReadbackTicket`] is a token rather than a
//! future: the caller encodes a request, submits the work through a recorder,
//! advances the device (`Device::poll`) until the GPU work is terminal, and only
//! then may read. The ticket never spins or waits on the GPU itself, and its
//! state machine cannot rest permanently in `NotSubmitted` or `Pending` —
//! section 18.2 makes those two transient, which is what stops a lost device
//! from leaving a caller waiting forever (section 18.7).
//!
//! # What this module does not own
//!
//! - The two encode verbs. Section 18.5 declares them on `CommandRecorder`,
//!   which is the command chapter's type; see this crate's `0.16` series audit
//!   (A7) for the adjudication that puts them there.
//! - Completion points. The type belongs to `submission`
//!   ([`crate::api::submission::CompletionPoint`]) and the value belongs to the
//!   device: [`ReadbackTicket::completion`] reports the point a successful
//!   submission recorded, and `None` before one does, because a ticket must not
//!   mint a point for work the device has not accepted.
//! - Retirement bookkeeping. Section 18.6 states the rule — native backing must
//!   outlive the last logical owner *and* all terminal GPU work — and it belongs
//!   to the device, not to a token in this module: a ticket records a completion
//!   point, and only the device can know when the last logical owner and the last
//!   terminal GPU work have both passed. The bookkeeping that enforces it arrives
//!   with the backend port; nothing in this tree enforces it today.
//!
//! The two workflows live in their own files — upload in `upload` and
//! readback in `readback` — because section 17 and section 18 are
//! separately specified, separately encoded, and separately consumed. This
//! file is the composition entry point: it re-exports both halves at the module
//! path callers already use, and owns the one rule they share.
//!
//! Both halves stay crate-visible rather than private because the validators they
//! own are crate-private entry points of their own: a `pub(crate) use` of one
//! would be an unused import in a non-test build, since the device façade that
//! calls it is not written yet, and this module does not carry lint attributes as
//! a substitute for a caller.

pub(crate) mod readback;
pub(crate) mod upload;

pub use readback::{
    ReadbackData, ReadbackRequest, ReadbackStatus, ReadbackTexelLayout, ReadbackTicket,
};
pub use upload::{BufferUploadDescriptor, TextureUploadDescriptor, UploadDescriptor, UploadJob};

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::format_aspects;
use crate::api::resource::subresource::{
    Origin3d, TextureSubresourceLayers, aspect_bits, validate_origin_extent,
    validate_subresource_layers,
};
use crate::api::resource::texture::{Extent3d, TextureDescriptor, mip_extent};

/// Checks a copy region against the texture it addresses.
///
/// Shared by upload and readback, because a region that is legal to write is
/// legal to read: the two differ in usage bit and in which route carries them,
/// never in geometry.
///
/// Beyond section 17.3's "subresource/origin/extent valid", this checks that the
/// region is *inside* the texture — the mip level exists, the layer range exists,
/// and `origin + extent` stays within that level's extent. Those are the reasons
/// the word "valid" has content, and root section 4 requires them to be refused
/// portably rather than by a driver that may answer differently on each backend.
pub(crate) fn validate_texture_region(
    base: &TextureDescriptor,
    subresource: TextureSubresourceLayers,
    origin: Origin3d,
    extent: Extent3d,
) -> RhiResult<()> {
    validate_subresource_layers(subresource, base.dimension)?;
    validate_origin_extent(origin, extent, base.dimension)?;

    if subresource.mip_level >= base.mip_levels {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "the region addresses mip level {}, but the texture has {}",
                subresource.mip_level, base.mip_levels
            ),
        ));
    }
    let last_layer = subresource
        .base_layer
        .checked_add(subresource.layer_count)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the region's array layer range overflows",
            )
        })?;
    if last_layer > base.array_layers {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "the region covers array layers {}..{}, but the texture has {}",
                subresource.base_layer, last_layer, base.array_layers
            ),
        ));
    }

    if !format_aspects(base.format).contains(aspect_bits(subresource.aspect)) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "the {:?} aspect does not exist in a {:?} texture",
                subresource.aspect, base.format
            ),
        ));
    }

    let level = mip_extent(base.extent, base.dimension, subresource.mip_level);
    let end_x = origin.x.checked_add(extent.width);
    let end_y = origin.y.checked_add(extent.height);
    let end_z = origin.z.checked_add(extent.depth);
    let (Some(end_x), Some(end_y), Some(end_z)) = (end_x, end_y, end_z) else {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the region's origin plus extent overflows",
        ));
    };
    if end_x > level.width || end_y > level.height || end_z > level.depth {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "the region {}x{}x{} at {}x{}x{} extends past the {}x{}x{} of mip level {}",
                extent.width,
                extent.height,
                extent.depth,
                origin.x,
                origin.y,
                origin.z,
                level.width,
                level.height,
                level.depth,
                subresource.mip_level
            ),
        ));
    }
    Ok(())
}
