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
//! allocation size, a texture's extent and format.  Those records are adapter
//! knowledge -- Layer 1 exposes no verb that reads a descriptor back, and an
//! identity names an object rather than describing one -- and they are the same
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
//! cannot transfer a rectangle for -- are the exception, and only because they
//! are facts the adapter is the first to know rather than rules Layer 1 applies.
//! Both are conditions of the family, not of the request, so they refuse as
//! [`unsupported`] rather than as a validation failure.
//!
//! Nothing is batched and nothing is staged: one call is one upload.  §1 of the
//! plan forbids building the framework ahead of the consumer, and the consumer
//! this exists for is one imported vertex stream and one imported sampled image.

use crate::webgl2::api::{
    BufferId, GlError, GlFormat, GlPixelFormat, GlPixelLayout, GlRepackPolicy, GlTextureDimension,
    GlTextureRegion, TextureId,
};

use super::GlCompatibilityDevice;
use super::compute::ComputeDomain;
use super::failure::{malformed, unsupported};
use super::pass;
use super::region;
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
    /// layer all follow from it.  [`addressing`] argues why that is the only
    /// shape of upload this verb offers and what it refuses instead.
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
        let (region, layout) = addressing(texture, facts)?;
        self.machine.backend().upload_texture(region, layout, bytes)
    }
}

/// The region and client layout a whole-level upload of one recorded texture
/// implies.
///
/// The two are derived together because neither is a free choice: the region's
/// extent is the attachment's own, and the layout is that extent's tight packing
/// in the one client encoding this family transfers.  Returning them as a pair
/// keeps the caller from pairing a region with a layout derived from a different
/// extent, which is the mistake the derivation exists to prevent.
///
/// # Why only a two-dimensional attachment
///
/// Because a rectangle is the only thing a whole-level upload can be *said* to
/// cover here.  For a two-dimensional attachment that rectangle is its whole
/// extent, and the question never arises.  For a `D3` or an arrayed one it is one
/// slice of several, and the subresource addressing that would have to name
/// *which* slices is exactly what [`region::texture_region`] pins at one layer
/// for the copy verbs' reason -- so a verb that accepted one would be claiming
/// the whole of a level it was about to fill a fraction of.  Refusing keeps that
/// a named gap rather than a silent partial upload, and closing it is a question
/// about layer ranges and not about this verb.
///
/// A one-dimensional attachment is refused by the same test, and it is the one
/// case where the rectangle genuinely is the whole level: what is missing there
/// is not the addressing but the binding, since no point in this family samples a
/// `D1` texture at all, so there is no consumer a verb for it would serve.
///
/// # Why the refusal is unsupported and the overflow is not
///
/// The dimension and the format are facts about what this family can transfer, so
/// the same sentence is true of every caller and a caller cannot fix either one.
/// A width whose row does not fit in the layout's own 32-bit pitch field is not
/// that -- it is a fact about this attachment -- and it is stated as a validation
/// failure for the same reason.  It is also unreachable through the public path,
/// since a texture this wide cannot be created in the first place; it is checked
/// rather than asserted because the arithmetic is Layer 1's own spelling too, and
/// a saturating or wrapping one here would be a quiet way to upload the wrong
/// number of bytes.
///
/// The format rule is the one decision here that Layer 1 also makes, and the
/// overlap is worth stating rather than hiding.  Both executable providers accept
/// a CPU pixel upload for exactly the two RGBA8 formats this match names, so the
/// pair is stated twice.  What makes that safe is which way the two can diverge:
/// this match is the *narrower* place by construction, since it can only ever
/// refuse a call the provider would have accepted, and a verb that refuses is
/// wrong in a way a caller sees and reports.  A provider that widened its pair
/// without widening this one would cost an upload, never a mis-transfer.
fn addressing(
    texture: TextureId,
    facts: pass::Attachment,
) -> Result<(GlTextureRegion, GlPixelLayout), GlError> {
    const OP: &str = "upload-texture";
    if !matches!(facts.dimension(), GlTextureDimension::D2) {
        return Err(unsupported(
            OP,
            "this verb fills a whole two-dimensional level, which is not the shape this attachment was created with",
        ));
    }
    // The two formats this verb can state a client encoding for.  Both are four
    // eight-bit channels on the client side, which is why one encoding covers the
    // pair, and the family's other two creatable formats -- the half-float one,
    // which has no client encoding in this vocabulary at all, and the depth one,
    // whose encoding no provider transfers -- are refused below.
    let format = match facts.format() {
        GlFormat::Rgba8Unorm | GlFormat::Rgba8Srgb => GlPixelFormat::Rgba8,
        _ => {
            return Err(unsupported(
                OP,
                "this verb states a client encoding for the two RGBA8 formats this family transfers, and this attachment is not one of them",
            ));
        }
    };
    let (width, height) = facts.extent();
    let Some(bytes_per_row) = width.checked_mul(format.bytes_per_pixel()) else {
        return Err(malformed(
            OP,
            "this attachment's rows are too wide to state as a client row pitch",
        ));
    };
    // The pitch is the packed width and the alignment divides it, so the pair is
    // always expressible by GL pixel-store and a repack can never be asked for.
    // `Disallow` says that rather than leaving a bound nobody chose.
    let layout = GlPixelLayout {
        format,
        bytes_per_row,
        rows_per_image: height,
        offset: 0,
        alignment: 4,
        repack: GlRepackPolicy::Disallow,
    };
    Ok((
        region::texture_region(texture, 0, [0, 0, 0], [width, height, 1]),
        layout,
    ))
}
