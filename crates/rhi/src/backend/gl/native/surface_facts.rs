//! Records the default framebuffer's observed format, or says explicitly why it
//! could not be observed.
//!
//! One responsibility: read the drawable's own component widths and sample state
//! out of a context that is already current, and render both the acting value and
//! the reporting strings from that one set of queries. FBO 0 is not a
//! `GlFormatTable` row, so the surface the platform flips is the one piece of
//! format evidence no other record carries, and a presenter needs it to know what
//! it is presenting.
//!
//! Not responsible for: which profile is accepted, what a provider acquires, or
//! any format that is not the default framebuffer's -- a texture or renderbuffer
//! row belongs to `discovery`'s format tables. It binds nothing and creates
//! nothing; the context it reads is the caller's.
//!
//! The component widths are read from the drawable's own colour attachment, which
//! every accepted profile answers about the *bound* draw framebuffer -- hence the
//! binding check first: with an application framebuffer bound, those same queries
//! describe that framebuffer, and recording them as the surface format would be
//! recording a different object's format under the surface's name. Every path that
//! cannot observe the drawable records a reason instead of a value, so a missing
//! surface format is never mistaken for an observed one.
//!
//! **Required and reported are two different sets.** A fact is required when a
//! consumer acts on it -- the colour widths and the sample count, which are what a
//! presentation path sizes what it presents from -- and reported when only a
//! reader acts on it. Depth and stencil are the second kind: nothing derives from
//! them, and a window visual with no depth buffer is legal, so requiring them
//! would refuse a conformant context for a fact nobody reads, while recording a
//! width for one would claim an attachment the drawable does not have. Both of
//! those are recorded as what they are instead.
//!
//! The color encoding of the drawable is deliberately not recorded: no accepted
//! profile exposes a portable query for the default framebuffer's encoding, and a
//! guess between linear and sRGB is a double-gamma error rather than a missing
//! fact. It is recorded as unavailable instead.
//!
//! Both renderings come out of this one module and one set of queries. The strings
//! are the reporting channel and the [`GlSurfaceFacts`] value is the acting one;
//! deriving either from the other would put a formatted marker back on the path a
//! consumer acts on, which is what typing the value is for. The narrowing to the
//! value's own widths is part of that: a negative answer cannot be a width, so it
//! fails the observation the same way a missing component does, and neither
//! rendering is written for it.

use std::collections::BTreeSet;

use super::discovery::{NativeGlQuery, glow_const};
use crate::backend::gl::api::GlSurfaceFacts;

/// The colour buffers a drawable can have, in the order the observation asks for
/// one.
///
/// A double-buffered window draws into its back buffer, and that is the buffer a
/// drawable that can be presented has; a single-buffered one has only the front.
/// Asking in this order asks in the order the visual itself has, so the
/// single-buffered case is a fallback with no rule of its own rather than a
/// second path with its own semantics. Which entry answered is recorded, so a
/// reader can see which buffer the widths describe -- and, on the one machine
/// that can exercise the fallback, that it ran. This repository's fixture
/// exercises the first entry only; the second rests on the fake-query tests.
const COLOR_ATTACHMENTS: [(&str, u32); 2] = [
    ("GL_BACK_LEFT", glow_const::BACK_LEFT),
    ("GL_FRONT_LEFT", glow_const::FRONT_LEFT),
];

/// The four colour component widths, in the order the typed value holds them.
///
/// The names are what a failure marker reports, and they are the *current*
/// spelling of the question: the `GL_RED_BITS` family this observation used to
/// ask is what made it fail on every desktop core context there is, since the
/// core profile answers that family with `GL_INVALID_ENUM`. One table so the name
/// and the token cannot drift -- a name retyped beside a token is a second
/// spelling of the same fact that can rot without anything failing.
const COLOR_WIDTHS: [(&str, u32); 4] = [
    (
        "GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE",
        glow_const::FRAMEBUFFER_ATTACHMENT_RED_SIZE,
    ),
    (
        "GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE",
        glow_const::FRAMEBUFFER_ATTACHMENT_GREEN_SIZE,
    ),
    (
        "GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE",
        glow_const::FRAMEBUFFER_ATTACHMENT_BLUE_SIZE,
    ),
    (
        "GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE",
        glow_const::FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE,
    ),
];

/// The two required facts that are not component widths, with the profile's own
/// query for each. The core profile kept both, which the hardware this row was
/// found on shows directly: the six bit queries that used to sit beside them were
/// refused there and these two answered.
const SAMPLE_FACTS: [(&str, u32); 2] = [
    ("GL_SAMPLE_BUFFERS", glow_const::SAMPLE_BUFFERS),
    ("GL_SAMPLES", glow_const::SAMPLES),
];

/// The two attachments the observation reports without requiring, as
/// `(key, attachment, component-size pname)`.
///
/// These are the facts only a reader acts on, so neither an absent attachment nor
/// an unreadable one may fail the observation, and neither may be reported as a
/// width: see [`record_optional_attachment`]. Their outcomes are carried under
/// their own keys rather than under `gl.surface-facts-unavailable=`, which stays
/// the marker for "no surface was observed at all" -- a drawable whose depth
/// attachment could not be read is still a drawable whose colour format was
/// observed, and one marker cannot say both.
const OPTIONAL_ATTACHMENTS: [(&str, u32, u32); 2] = [
    (
        "depth",
        glow_const::DEPTH,
        glow_const::FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE,
    ),
    (
        "stencil",
        glow_const::STENCIL,
        glow_const::FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE,
    ),
];

/// Picks the drawable's colour buffer, or records why there is none.
fn color_buffer(
    query: &impl NativeGlQuery,
    facts: &mut BTreeSet<String>,
) -> Option<(&'static str, u32)> {
    let mut refused: Vec<String> = Vec::new();
    for (name, attachment) in COLOR_ATTACHMENTS {
        match query.drawable_attachment(attachment, glow_const::FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE)
        {
            // An object type of `GL_NONE` is the driver saying this buffer is not
            // an attachment of the drawable, and a refused query says the same
            // thing for a driver that reports an absent attachment as an error
            // instead. Both mean "not this one", so the next candidate is tried
            // rather than the observation failing; `refused` keeps the second of
            // those apart from the first in case no candidate is present. Every
            // refusal is kept rather than the first, since a reader cannot tell
            // from one of them whether the other candidate was asked at all.
            Some(0) => {}
            Some(_) => return Some((name, attachment)),
            None => {
                refused.push(format!(
                    "gl.surface-facts-unavailable=query-failed:{name}:\
                     GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE"
                ));
            }
        }
    }
    // "This drawable has no colour buffer" and "the driver would not say" are
    // different facts about it, and the record carries whichever it was.
    if refused.is_empty() {
        facts.insert("gl.surface-facts-unavailable=no-color-attachment".into());
    } else {
        facts.extend(refused);
    }
    None
}

/// Records one attachment's width, or records which of the three ways there is
/// no width to record.
///
/// Depth and stencil are the observation's reported-but-not-required half. An
/// absent attachment is legal -- a window visual is free to have no depth buffer
/// -- so failing the observation for one would refuse a conformant context for a
/// fact nothing reads, which is the shape of the inverted requirement this crate
/// already had to remove once. Recording it as a width would be the other error,
/// and worse: zero bits is a claim about an attachment, and there may be none.
/// So the value is a word when there is no width, one word per fact: `none` for
/// an attachment the drawable does not have, `unqueried` for one the driver would
/// not answer for, `invalid` for an answer that cannot be a width. A reader that
/// conflates those three would be reading a missing attachment, a missing answer
/// and a nonsensical one as the same thing.
fn record_optional_attachment(
    query: &impl NativeGlQuery,
    facts: &mut BTreeSet<String>,
    (key, attachment, pname): (&str, u32, u32),
) {
    let object_type =
        query.drawable_attachment(attachment, glow_const::FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE);
    if object_type == Some(0) {
        facts.insert(format!("gl.surface-{key}-attachment=none"));
        return;
    }
    let width = object_type.and_then(|_| query.drawable_attachment(attachment, pname));
    match width.map(u32::try_from) {
        Some(Ok(width)) => {
            facts.insert(format!("gl.surface-{key}-bits={width}"));
        }
        Some(Err(_)) => {
            facts.insert(format!("gl.surface-{key}-attachment=invalid"));
        }
        None => {
            facts.insert(format!("gl.surface-{key}-attachment=unqueried"));
        }
    }
}

/// The drawable's typed facts, with the strings they were observed alongside.
pub(super) fn observe(query: &impl NativeGlQuery) -> (GlSurfaceFacts, BTreeSet<String>) {
    let mut facts = BTreeSet::new();
    let unavailable = |facts: BTreeSet<String>| (GlSurfaceFacts::Unavailable, facts);
    let Some(binding) = query.integer(glow_const::DRAW_FRAMEBUFFER_BINDING) else {
        facts.insert("gl.surface-facts-unavailable=unqueried".into());
        return unavailable(facts);
    };
    if binding != 0 {
        facts.insert("gl.surface-facts-unavailable=draw-framebuffer-bound".into());
        return unavailable(facts);
    }
    let Some((buffer, attachment)) = color_buffer(query, &mut facts) else {
        return unavailable(facts);
    };
    // The required set is read together and reported together: a partial surface
    // format cannot decide anything a presenter would ask it, so the record says
    // "not observed" rather than half a format -- and it names every component
    // that failed, since a failure a reader outside this crate cannot attribute is
    // a failure that reader has to reproduce by hand.
    let mut color = [0_i64; 4];
    let mut sample_facts = [0_i64; 2];
    let mut failed: Vec<String> = Vec::new();
    for (slot, (name, pname)) in color.iter_mut().zip(COLOR_WIDTHS) {
        match query.drawable_attachment(attachment, pname) {
            Some(value) => *slot = value,
            None => failed.push(format!("{buffer}:{name}")),
        }
    }
    for (slot, (name, pname)) in sample_facts.iter_mut().zip(SAMPLE_FACTS) {
        match query.integer(pname) {
            Some(value) => *slot = value,
            None => failed.push((*name).to_owned()),
        }
    }
    if !failed.is_empty() {
        for marker in failed {
            facts.insert(format!(
                "gl.surface-facts-unavailable=query-failed:{marker}"
            ));
        }
        return unavailable(facts);
    }
    // The typed value is unsigned by construction: a component width and a sample
    // count are never negative, so a negative answer is the driver answering a
    // different question than the one asked. It fails the whole observation for the
    // same reason a missing component does -- narrowing it unchecked would wrap into
    // a huge width, and a huge width is a format claim rather than a missing fact.
    let mut widths = [0_u32; 4];
    for ((slot, value), (name, _)) in widths.iter_mut().zip(color).zip(COLOR_WIDTHS) {
        match u32::try_from(value) {
            Ok(width) => *slot = width,
            Err(_) => {
                facts.insert(format!(
                    "gl.surface-facts-unavailable=query-failed:{buffer}:{name}"
                ));
                return unavailable(facts);
            }
        }
    }
    let [red, green, blue, alpha] = widths;
    let mut counts = [0_u32; 2];
    for ((slot, value), (name, _)) in counts.iter_mut().zip(sample_facts).zip(SAMPLE_FACTS) {
        match u32::try_from(value) {
            Ok(count) => *slot = count,
            Err(_) => {
                facts.insert(format!("gl.surface-facts-unavailable=query-failed:{name}"));
                return unavailable(facts);
            }
        }
    }
    let [sample_buffers, samples] = counts;
    // Reported, never required: see `OPTIONAL_ATTACHMENTS`.
    for optional in OPTIONAL_ATTACHMENTS {
        record_optional_attachment(query, &mut facts, optional);
    }
    // Written last, and only here: a path that could not observe the drawable
    // records reasons instead of values, and which buffer the widths describe is
    // a value. It is written at all because without it a reader cannot tell that
    // the single-buffered fallback ran, and that fallback is the one path in this
    // module no test in this repository can reach on real hardware.
    facts.insert(format!("gl.surface-color-buffer={buffer}"));
    facts.insert(format!(
        "gl.surface-color-bits={red},{green},{blue},{alpha}"
    ));
    facts.insert(format!("gl.surface-sample-buffers={sample_buffers}"));
    facts.insert(format!("gl.surface-samples={samples}"));
    facts.insert("gl.surface-srgb=unavailable".into());
    (
        GlSurfaceFacts::Observed {
            color_bits: [red, green, blue, alpha],
        },
        facts,
    )
}
