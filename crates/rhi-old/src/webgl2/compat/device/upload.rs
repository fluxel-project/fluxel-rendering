//! Writing host bytes into one object this device created.
//!
//! Responsibility: turn a caller's identity -- a `BufferId` with an offset, or a
//! `TextureId` and nothing else -- into the one Layer 1 verb that writes bytes
//! there, and refuse the cases Layer 1 cannot recognise.
//!
//! Not owned here: how an object comes to exist ([`super::backend`]'s two
//! transient verbs), and when it is destroyed ([`super::retention`]).
//!
//! # Why this is not a contract verb
//!
//! [`ExecutionBackend`](fluxel_rendergraph::ExecutionBackend) has no verb that
//! writes host bytes, and the omission is not an oversight to close here.  Every
//! resource verb in that contract is about *existence* -- create, transition,
//! bind, copy device-to-device -- because a graph-created resource's contents are
//! produced by the passes that write it, and the executor has no bytes of its
//! own to put anywhere.  Upload is what a *caller* does to a resource it owns,
//! which is exactly the population an import is drawn from: the contract's
//! import half says a caller-owned object arrives already backed, and says
//! nothing about how it got that way.
//!
//! So these are inherent verbs on the adapter, and the composability is the
//! point: a caller calls `create_transient_buffer` for the object, calls one of
//! these to fill it, and hands the identity back through a
//! [`FrameResourceProvider`](fluxel_rendergraph::FrameResourceProvider).  None of
//! the three steps needs a frame, an encoder or an executor, which is what makes
//! the imported half of the frame contract reachable at all.
//!
//! # Where the extent comes from, and why the two arms answer it differently
//!
//! A buffer has no shape, so the caller's slice is the only thing that can say
//! how much of it one upload covers: [`Self::upload_buffer`] *derives* the
//! range's size from `bytes.len()` instead of accepting one, and a length that
//! disagrees with a size the caller also had in mind is not a case it detects but
//! a state it cannot construct.
//!
//! A texture is the other way round, and this is the whole of what the second
//! verb adds.  Its extent is a fact about the object -- recorded at creation and
//! readable back from [`super::pass::Attachment`] -- and no slice can stand in
//! for it, because a slice of the wrong length is exactly the mistake worth
//! catching.  So [`Self::upload_texture`] derives the region from the record and
//! takes the bytes as given, and it is the caller's slice having to *agree* with
//! a derived layout that makes this verb the first one here where a length
//! mismatch is constructible.  What it does with that is the section below.
//!
//! # What each check here is, and what is left to Layer 1
//!
//! The adapter checks one thing per arm: that the object is one *this device*
//! created, which is also what gives it the facts the verb needs -- a buffer's
//! allocation size, a texture's extent, shape and format.  Those records are
//! adapter knowledge -- Layer 1 exposes no verb that reads a descriptor back, and
//! an identity names an object rather than describing one -- and they are the same
//! records [`Self::storage_range`] and the pass lowering read for the identical
//! reason, kept and dropped beside the creation that wrote them so an identity
//! reused after a deletion cannot resolve to its predecessor's facts.
//!
//! Everything else is Layer 1's, and deliberately not restated.  Whether a buffer
//! range is non-empty and inside the allocation is a rule about a `GlBufferDesc`;
//! whether a texture layout covers exactly the bytes handed in is a rule about a
//! `GlTextureDesc` and a `GlPixelLayout` together.  Both are stated once where the
//! descriptors live, and a second spelling here would be a second thing to keep
//! true and would drift.  The refusals reach the caller under the same operation
//! strings these verbs use, so nothing about them misnames the call, and by then
//! no driver command has been issued.
//!
//! The texture arm's *own* two refusals -- a dimension and a format this family
//! cannot transfer a rectangle for -- are the exception, and they are not stated
//! here either: they are [`super::transfer`]'s, because
//! [`Self::read_texture`](super::GlCompatibilityDevice::read_texture) reads the
//! same rectangle back out and must refuse the same shapes as this verb fills.
//! This arm passes the record and its own operation name and does nothing else
//! with them.
//!
//! Nothing is batched and nothing is staged: one call is one upload.  §1 of the
//! plan forbids building the framework ahead of the consumer, and the consumer
//! this exists for is one imported vertex stream and one imported sampled image.

use crate::webgl2::api::{BufferId, GlError, TextureId};

use super::GlCompatibilityDevice;
use super::compute::ComputeDomain;
use super::failure::malformed;
use super::region;
use super::transfer;
use crate::webgl2::state::GlStateBackend;

impl<B: GlStateBackend, C: ComputeDomain<B>> GlCompatibilityDevice<B, C> {
    /// Writes `bytes` at `offset` in one buffer this device created.
    ///
    /// The lifecycle is adopted and pending retirements are dispatched first,
    /// like every other verb on this adapter and for the reason stated there: a
    /// context generation change must be observed before an identity is resolved
    /// against a record, since [`Self::refresh`] is what drops records belonging
    /// to the superseded generation.
    ///
    /// A buffer this device did not create is refused rather than attempted.
    /// There are two ways to hold such an identity -- one minted by another
    /// device, or one of this device's that has already been destroyed -- and the
    /// sentence covers both because the caller's mistake is the same one: the
    /// upload and the allocation did not come from the same place.
    pub(super) fn upload_buffer(
        &mut self,
        buffer: BufferId,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), GlError> {
        const OP: &str = "upload-buffer";
        self.refresh();
        self.release_pending()?;
        if !self.buffers.contains_key(&buffer) {
            return Err(malformed(
                OP,
                "an upload names a buffer this device did not create",
            ));
        }
        let range = region::buffer_range(buffer, offset, bytes.len() as u64);
        self.machine.backend().upload_buffer(range, bytes)
    }

    /// Writes `bytes` into the whole of mip zero of one texture this device made.
    ///
    /// The bytes are the attachment's pixels, tightly packed in the one client
    /// encoding this family transfers, and the caller states nothing else: the
    /// extent is the attachment's, and the row pitch, the image height and the
    /// layer all follow from it.  [`transfer::whole_level`] argues why that is the
    /// only shape of transfer this family offers and what it refuses instead.
    ///
    /// As on the buffer arm, an identity this device did not create -- or one of
    /// its own that has already been destroyed -- is refused against the adapter's
    /// own record, before any driver call.
    pub(super) fn upload_texture(
        &mut self,
        texture: TextureId,
        bytes: &[u8],
    ) -> Result<(), GlError> {
        const OP: &str = "upload-texture";
        self.refresh();
        self.release_pending()?;
        let Some(facts) = self.attachments.get(&texture).copied() else {
            return Err(malformed(
                OP,
                "an upload names a texture this device did not create",
            ));
        };
        let (region, layout) = transfer::whole_level(texture, facts, OP)?;
        self.machine.backend().upload_texture(region, layout, bytes)
    }
}
