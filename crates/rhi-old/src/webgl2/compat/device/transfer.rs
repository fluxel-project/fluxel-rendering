//! The region and client layout one whole-level host-byte transfer implies.
//!
//! Responsibility: turn the creation record of one texture this device made into
//! the pair of values Layer 1's transfer verbs take, and refuse the shapes and
//! formats this family has no rectangle for.
//!
//! Not owned here: which verb asks ([`super::upload`] writes host bytes in,
//! [`super::readback`] reads them out), what either does with the pair, and the
//! lifecycle either adopts before it calls.
//!
//! # Why both directions share one statement
//!
//! A whole-level transfer is the same rectangle and the same packing whichever
//! way the bytes go: level zero, from the origin, tight rows of four eight-bit
//! channels.  So the two verbs differ in what they do with the pair and in
//! nothing about how it is derived -- and a second derivation would be a second
//! thing to keep true, one that could drift into reading a padded pitch or a
//! transposed rectangle the writing side never produced.
//!
//! The *operation string* stays the caller's rather than becoming this module's,
//! because a refusal reaches the caller under the name of the verb it called, and
//! two verbs reaching one shared decision must not report it as each other's.
//!
//! # Why the record, and not a descriptor the caller states
//!
//! The extent, the format and the shape are facts the adapter already holds, kept
//! beside the creation that wrote them and dropped when the object dies
//! ([`super::pass::Attachment`]).  A caller who stated them instead would make a
//! wrong statement *constructible*, and the only thing left to catch it would be
//! Layer 1 validating a region against a descriptor it holds -- a refusal that
//! exists, but that a verb invites whenever it asks a question it can answer
//! itself.
//!
//! # What is checked here, and what is left to Layer 1
//!
//! The two conditions checked here are facts about the *family* rather than about
//! the request -- the shape of a rectangle this layer transfers at all, and the
//! client encodings it can state -- so both refuse as [`unsupported`] and a caller
//! cannot fix either by asking differently.  A width whose row does not fit the
//! layout's own 32-bit pitch field is not that; it is a fact about this
//! attachment, and it is stated as a validation failure for the same reason.  It
//! is also unreachable through the public path, since a texture this wide cannot
//! be created; it is checked rather than asserted because the arithmetic is Layer
//! 1's own spelling too, and a saturating or wrapping one here would be a quiet
//! way to transfer the wrong number of bytes.
//!
//! The format rule is the one decision here that Layer 1 also makes, and the
//! overlap is worth stating rather than hiding.  Both executable providers accept
//! a CPU pixel transfer for exactly the two RGBA8 formats this match names, so the
//! pair is stated twice.  What makes that safe is which way the two can diverge:
//! this match is the *narrower* place by construction, since it can only ever
//! refuse a call the provider would have accepted, and a verb that refuses is
//! wrong in a way a caller sees and reports.  A provider that widened its pair
//! without widening this one would cost a transfer, never a mis-transfer.
//!
//! A multisampled attachment needs no check here for the same reason: the copy
//! domain refuses a transfer whose either end is multisampled
//! (`api/native/exec_copy.rs` names both), so the refusal already exists at the
//! layer that owns the rectangle and restating it would be the second spelling
//! this module exists to avoid.

use crate::webgl2::api::{
    GlError, GlFormat, GlPixelFormat, GlPixelLayout, GlRepackPolicy, GlTextureDimension,
    GlTextureRegion, TextureId,
};

use super::failure::{malformed, unsupported};
use super::pass;
use super::region;

/// The region and client layout a whole-level transfer of one recorded texture
/// implies, refused under `op` when this family has no such transfer.
///
/// The two are derived together because neither is a free choice: the region's
/// extent is the attachment's own, and the layout is that extent's tight packing
/// in the one client encoding this family transfers.  Returning them as a pair
/// keeps a caller from pairing a region with a layout derived from a different
/// extent, which is the mistake the derivation exists to prevent.
///
/// # Why only a two-dimensional attachment
///
/// Because a rectangle is the only thing a whole-level transfer can be *said* to
/// cover here.  For a two-dimensional attachment that rectangle is its whole
/// extent, and the question never arises.  For a `D3` or an arrayed one it is one
/// slice of several, and the subresource addressing that would have to name
/// *which* slices is exactly what [`region::texture_region`] pins at one layer
/// for the copy verbs' reason -- so a caller that accepted one would be receiving
/// a claim about the whole of a level it was handed a fraction of.  Refusing keeps
/// that a named gap rather than a silent partial transfer, and closing it is a
/// question about layer ranges and not about the verb that asked.
///
/// A one-dimensional attachment is refused by the same test, and it is the one
/// case where the rectangle genuinely is the whole level: what is missing there
/// is not the addressing but the binding, since no point in this family samples a
/// `D1` texture at all, so there is no consumer a verb for it would serve.
pub(super) fn whole_level(
    texture: TextureId,
    facts: pass::Attachment,
    op: &'static str,
) -> Result<(GlTextureRegion, GlPixelLayout), GlError> {
    if !matches!(facts.dimension(), GlTextureDimension::D2) {
        return Err(unsupported(
            op,
            "this verb transfers a whole two-dimensional level, which is not the shape this attachment was created with",
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
                op,
                "this verb states a client encoding for the two RGBA8 formats this family transfers, and this attachment is not one of them",
            ));
        }
    };
    let (width, height) = facts.extent();
    let Some(bytes_per_row) = width.checked_mul(format.bytes_per_pixel()) else {
        return Err(malformed(
            op,
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
