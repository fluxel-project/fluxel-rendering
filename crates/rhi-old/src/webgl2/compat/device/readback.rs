//! Reading host bytes back out of one object this device created.
//!
//! Responsibility: turn an identity this adapter is holding a creation record for
//! into the one Layer 1 verb that reads bytes there, and refuse the cases Layer 1
//! cannot recognise.
//!
//! Not owned here: how an object comes to exist ([`super::backend`]'s transient
//! verb), when it is destroyed ([`super::retention`]), which rectangle and which
//! client encoding a transfer of it implies ([`super::transfer`]), and the
//! *orientation* of the bytes that come back.  The family's copy verbs follow the
//! GL bottom-left row convention (`api/native/exec_copy.rs` states that split), so
//! a caller that wants a top-down image flips the rows itself; this verb reports
//! the order it produced rather than silently choosing one.
//!
//! # Why this is the mirror of `upload`, and why that is the whole design
//!
//! [`super::upload`] writes host bytes into an object; this reads them back out.
//! Both are inherent verbs on the adapter for the reason that module gives -- the
//! frame contract has no verb that moves host bytes, because a graph-created
//! resource's contents come from the passes that write it -- and both answer the
//! same three questions the same way: *which* object (the adapter's own creation
//! record, so an identity this device did not create, or one of its own already
//! destroyed, is refused before any driver call), *what shape* (one whole level of
//! a two-dimensional attachment, derived from that record rather than stated),
//! and *in what encoding* (the one client encoding this family states for RGBA8).
//!
//! The consumer is what makes the pair worth having: a frame renders into a
//! transient attachment, and the caller that wants to know what the frame actually
//! drew has no other way to ask.  Layer 1 owns no path from a graph-created
//! texture back to the host, and the contract deliberately has none, so without
//! this verb the only evidence a rendering integration can produce is what the
//! counters say.

use crate::webgl2::api::{GlError, TextureId};

use crate::webgl2::state::GlStateBackend;

use super::GlCompatibilityDevice;
use super::compute::ComputeDomain;
use super::failure::malformed;
use super::transfer;

/// One whole level zero of a one-layer two-dimensional texture, as host bytes.
///
/// The bytes are the level's packed rows in the order the family produced them,
/// which is why the order is a field on the report rather than a sentence in a
/// doc comment: a consumer that has to flip them needs to be told, and a consumer
/// that assumes is a consumer that can silently produce a mirrored image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct TexturePixels {
    /// The pixel extent the read covered.
    pub extent: [u32; 2],
    /// Four eight-bit channels per pixel, `extent[0] * extent[1] * 4` bytes, rows
    /// bottom-up: the first row is the one at the region's lowest `y`.
    pub bytes: Vec<u8>,
}

impl<B: GlStateBackend, C: ComputeDomain<B>> GlCompatibilityDevice<B, C> {
    /// Reads the whole of `texture`'s level zero back as host bytes.
    ///
    /// `texture` is resolved against the adapter's own creation record, which is
    /// also what supplies the extent, the shape and the format the read covers:
    /// there is nothing for the caller to state, and therefore nothing for it to
    /// state wrongly.  A texture this device did not create -- or one of its own
    /// that has already been destroyed -- is refused there, before any driver
    /// call.
    ///
    /// # Why the operation string is Layer 1's too
    ///
    /// Every refusal below and every refusal Layer 1 makes while serving this call
    /// reach the caller under one name, `read-texture`, because a caller is
    /// answering one question -- what did the read I asked for do -- and a name
    /// that changed with the layer that refused it would make the kind of refusal
    /// the only thing the string told them.  [`Self::upload_texture`] and its
    /// provider agree on a name for the same reason rather than by coincidence.
    pub(super) fn read_texture(&mut self, texture: TextureId) -> Result<TexturePixels, GlError> {
        const OP: &str = "read-texture";
        // The lifecycle is adopted first, exactly as on the upload arm and for the
        // reason stated there: a context generation change must be observed before
        // an identity is resolved against a record, since `refresh` is what drops
        // the records belonging to the superseded generation.
        self.refresh();
        self.release_pending()?;
        let Some(facts) = self.attachments.get(&texture).copied() else {
            return Err(malformed(
                OP,
                "a readback names a texture this device did not create",
            ));
        };
        let (region, layout) = transfer::whole_level(texture, facts, OP)?;
        let readback = self.machine.backend().read_texture(region, layout)?;
        // Layer 1's own statement that what it returned is exactly the region's
        // size.  Checked rather than assumed, because the alternative is a
        // truncated image that every later check would read as a different
        // picture instead of as a failed read.
        readback
            .validate_for(region)
            .map_err(|_| malformed(OP, "the provider returned a readback of the wrong length"))?;
        let (width, height) = facts.extent();
        Ok(TexturePixels {
            extent: [width, height],
            bytes: readback.bytes,
        })
    }
}
