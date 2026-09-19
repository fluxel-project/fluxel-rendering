//! The drawable's recorded format: what the observation answers, what it says
//! when it cannot answer, and the agreement between the typed value and the
//! keys it was observed alongside.
//!
//! Built on `samples`' fixture so that the drawable's overrides are the only
//! difference between a context that answers about a surface and one that does
//! not, which is what makes a wrong answer here attributable to the drawable.

use super::samples::SampleFactQuery;
use super::*;

/// `GL_FRAMEBUFFER_DEFAULT`, the object type a default framebuffer's attachment
/// reports when it has one.
///
/// Named here rather than in the crate's token table because nothing on the
/// observation path needs it: that path asks whether the object type is
/// `GL_NONE`, not whether it is any particular present value, so a present value
/// it does not recognise is still a present attachment.
const FRAMEBUFFER_DEFAULT: i64 = 0x8218;

/// The drawable fixture: every answer `SampleFactQuery` gives, with the default
/// framebuffer's answers overridden.
///
/// The surface record is the one place where a withheld answer and an observed
/// one have to be distinguishable, so this fixture models every state the
/// observation has to tell apart: a binding query withheld entirely, an
/// application framebuffer bound -- the state in which the drawable queries
/// answer about a different object -- a drawable whose colour attachment exists
/// but whose sizes are refused, a drawable with no colour attachment at all, and
/// a driver that refuses the attachment query rather than reporting the absence.
/// It also models the two states the reported half has to survive: a single
/// buffered visual, whose only colour attachment is the front buffer, and a
/// visual with no depth or stencil attachment, which is legal and common.
struct SurfaceFactQuery {
    inner: SampleFactQuery,
    binding: Option<i64>,
    /// The colour buffer this visual has and the widths it answers, or `None`
    /// when the attachment query is refused instead of answered.
    color: Option<(u32, Option<[i64; 4]>)>,
    /// Depth and stencil, as `(object type, component size)`. An object type of
    /// zero is a visual without that attachment.
    depth: (i64, i64),
    stencil: (i64, i64),
    /// The two required facts that are not widths, withheld together.
    samples: Option<[i64; 2]>,
}

impl SurfaceFactQuery {
    /// A complete, drawable-bound context: every surface query answers.
    fn observed() -> Self {
        Self {
            inner: SampleFactQuery::new("4.6 test"),
            binding: Some(0),
            color: Some((glow_const::BACK_LEFT, Some([8, 8, 8, 8]))),
            depth: (FRAMEBUFFER_DEFAULT, 24),
            stencil: (FRAMEBUFFER_DEFAULT, 8),
            samples: Some([1, 4]),
        }
    }
}

impl NativeGlQuery for SurfaceFactQuery {
    fn take_error(&self) -> bool {
        self.inner.take_error()
    }
    fn string(&self, name: u32) -> Option<String> {
        self.inner.string(name)
    }
    fn integer(&self, name: u32) -> Option<i64> {
        match name {
            glow_const::DRAW_FRAMEBUFFER_BINDING => self.binding,
            glow_const::SAMPLE_BUFFERS => self.samples.map(|facts| facts[0]),
            glow_const::SAMPLES => self.samples.map(|facts| facts[1]),
            _ => self.inner.integer(name),
        }
    }
    fn integer_pair(&self, name: u32) -> Option<[i64; 2]> {
        self.inner.integer_pair(name)
    }
    fn indexed_integer(&self, name: u32, index: u32) -> Option<i64> {
        self.inner.indexed_integer(name, index)
    }
    fn float(&self, name: u32) -> Option<f32> {
        self.inner.float(name)
    }
    fn indexed_string(&self, name: u32, index: u32) -> Option<String> {
        self.inner.indexed_string(name, index)
    }
    fn drawable_attachment(&self, attachment: u32, name: u32) -> Option<i64> {
        // Depth and stencil answer their own object type and their own size. A
        // size query against an attachment the visual does not have is the one
        // question this fixture does not answer, because the real one is not
        // asked either: the observation reads the object type first and stops
        // there.
        let optional = |(object_type, size): (i64, i64)| match name {
            glow_const::FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE => Some(object_type),
            glow_const::FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE
            | glow_const::FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE => (object_type != 0).then_some(size),
            _ => None,
        };
        match attachment {
            glow_const::DEPTH => optional(self.depth),
            glow_const::STENCIL => optional(self.stencil),
            glow_const::BACK_LEFT | glow_const::FRONT_LEFT => {
                let (present, widths) = self.color?;
                if present != attachment {
                    // `GL_NONE`, answered rather than raised: a driver is free to
                    // report an absent attachment this way, and the observation
                    // has to read it as the absence it is.
                    return Some(0);
                }
                match name {
                    glow_const::FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE => Some(FRAMEBUFFER_DEFAULT),
                    glow_const::FRAMEBUFFER_ATTACHMENT_RED_SIZE => widths.map(|w| w[0]),
                    glow_const::FRAMEBUFFER_ATTACHMENT_GREEN_SIZE => widths.map(|w| w[1]),
                    glow_const::FRAMEBUFFER_ATTACHMENT_BLUE_SIZE => widths.map(|w| w[2]),
                    glow_const::FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE => widths.map(|w| w[3]),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

/// The drawable's format is recorded from the drawable, or the record says why
/// it could not be.
///
/// The property that matters is that "no surface format was observed" can never
/// be read as an observed one: with an application framebuffer bound the same
/// queries answer about that framebuffer instead of the surface, and a driver
/// that refuses the query answers nothing at all. Both of those states must
/// leave a reason, not a plausible-looking format.
#[test]
fn surface_format_is_recorded_from_the_drawable_or_marked_unavailable() {
    let flags_of = |query: &SurfaceFactQuery| {
        discover_with(query, &all_pass_plan())
            .expect("complete mock discovery")
            .context()
            .flags()
            .other
            .clone()
    };
    let observed = flags_of(&SurfaceFactQuery::observed());
    for marker in [
        "gl.surface-color-buffer=GL_BACK_LEFT",
        "gl.surface-color-bits=8,8,8,8",
        "gl.surface-depth-bits=24",
        "gl.surface-stencil-bits=8",
        "gl.surface-sample-buffers=1",
        "gl.surface-samples=4",
    ] {
        assert!(observed.contains(marker), "{marker} in {observed:?}");
    }
    // No accepted profile exposes the drawable's color encoding, so the record
    // says so rather than guessing between linear or sRGB.
    assert!(observed.contains("gl.surface-srgb=unavailable"));
    assert!(
        !observed
            .iter()
            .any(|marker| marker.contains("facts-unavailable")),
        "{observed:?}"
    );
    // A single-buffered visual has one colour attachment and it is the front
    // buffer. The widths are the same fact whichever buffer answered, which is
    // why the fallback carries no rule of its own -- and which buffer it was is
    // recorded, so the one path here the fixture cannot reach on hardware is
    // still visible in the record of a context that does reach it.
    let single = flags_of(&SurfaceFactQuery {
        color: Some((glow_const::FRONT_LEFT, Some([5, 6, 5, 0]))),
        ..SurfaceFactQuery::observed()
    });
    assert!(
        single.contains("gl.surface-color-buffer=GL_FRONT_LEFT"),
        "{single:?}"
    );
    assert!(
        single.contains("gl.surface-color-bits=5,6,5,0"),
        "{single:?}"
    );
    assert!(
        !single
            .iter()
            .any(|marker| marker.contains("facts-unavailable")),
        "{single:?}"
    );
    for (query, reason) in [
        (
            SurfaceFactQuery {
                binding: Some(2),
                ..SurfaceFactQuery::observed()
            },
            "gl.surface-facts-unavailable=draw-framebuffer-bound",
        ),
        (
            SurfaceFactQuery {
                binding: None,
                ..SurfaceFactQuery::observed()
            },
            "gl.surface-facts-unavailable=unqueried",
        ),
    ] {
        let flags = flags_of(&query);
        assert!(flags.contains(reason), "{reason} in {flags:?}");
        // A partially observed surface format would be worse than none: a
        // presenter cannot act on half a format, and a recorded value would
        // hide which half was missing.
        assert!(
            !flags
                .iter()
                .any(|marker| marker.starts_with("gl.surface-color-bits")),
            "{flags:?}"
        );
    }
    // A failed component names itself, and names the attachment it was asked
    // about.  The record has to say which query failed: a bare "query-failed"
    // cannot be adjudicated by a reader outside the crate, who has neither the
    // context nor the driver that produced it -- and now that the widths come
    // from an attachment, the component alone would not say which colour buffer
    // was asked either.  This asserts every one of the four is named, so a
    // component added to the observation without a name is a failure and not a
    // silently broader claim.
    let flags = flags_of(&SurfaceFactQuery {
        color: Some((glow_const::BACK_LEFT, None)),
        ..SurfaceFactQuery::observed()
    });
    let mut named: Vec<&str> = flags
        .iter()
        .filter_map(|marker| marker.strip_prefix("gl.surface-facts-unavailable=query-failed:"))
        .collect();
    named.sort_unstable();
    assert_eq!(
        named,
        [
            "GL_BACK_LEFT:GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE",
            "GL_BACK_LEFT:GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE",
            "GL_BACK_LEFT:GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE",
            "GL_BACK_LEFT:GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE",
        ],
        "{flags:?}"
    );
    assert!(
        !flags.contains("gl.surface-facts-unavailable=query-failed"),
        "an unattributable failure was recorded beside the attributed ones: {flags:?}"
    );
    assert!(
        !flags
            .iter()
            .any(|marker| marker.starts_with("gl.surface-color-bits")),
        "{flags:?}"
    );
    // The two required facts that are not widths are attributed the same way, and
    // they are required for a reason of their own: a presenter sizes what it
    // presents from the sample count. They are asked as parameters rather than
    // as attachments, so their names carry no buffer -- there is only one of each.
    let flags = flags_of(&SurfaceFactQuery {
        samples: None,
        ..SurfaceFactQuery::observed()
    });
    let mut named: Vec<&str> = flags
        .iter()
        .filter_map(|marker| marker.strip_prefix("gl.surface-facts-unavailable=query-failed:"))
        .collect();
    named.sort_unstable();
    assert_eq!(named, ["GL_SAMPLES", "GL_SAMPLE_BUFFERS"], "{flags:?}");
    // A drawable with no colour attachment at all and a driver that refuses the
    // attachment query are different facts, and the record carries which one it
    // was rather than one marker for both.
    let flags = flags_of(&SurfaceFactQuery {
        color: None,
        ..SurfaceFactQuery::observed()
    });
    assert_eq!(
        flags
            .iter()
            .filter(|marker| marker.contains("facts-unavailable"))
            .cloned()
            .collect::<Vec<String>>(),
        [
            "gl.surface-facts-unavailable=query-failed:GL_BACK_LEFT:\
             GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE"
                .to_owned(),
            "gl.surface-facts-unavailable=query-failed:GL_FRONT_LEFT:\
             GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE"
                .to_owned(),
        ],
        "{flags:?}"
    );
}

/// A visual without a depth or stencil attachment is reported, not refused.
///
/// The typed value carries the colour widths and nothing else, and no field of
/// the surface row is derived from depth or stencil, so requiring them would
/// refuse a conformant context for a fact nothing reads -- the shape of the
/// inverted requirement this crate already had to remove once, one layer down.
/// Recording them as widths would be the other error: zero bits is a claim about
/// an attachment, and there is none.  The three outcomes are therefore three
/// different words, and this asserts the two that are not a width.
#[test]
fn an_absent_attachment_is_reported_rather_than_refused_or_counted_as_zero() {
    let snapshot_of = |query: &SurfaceFactQuery| {
        discover_with(query, &all_pass_plan()).expect("complete mock discovery")
    };
    let absent = snapshot_of(&SurfaceFactQuery {
        depth: (0, 0),
        stencil: (0, 0),
        ..SurfaceFactQuery::observed()
    });
    assert_eq!(
        absent.surface_facts(),
        GlSurfaceFacts::Observed {
            color_bits: [8, 8, 8, 8],
        },
        "a visual with no depth attachment still has an observed colour format"
    );
    let flags = absent.context().flags().other.clone();
    assert!(
        flags.contains("gl.surface-depth-attachment=none"),
        "{flags:?}"
    );
    assert!(
        flags.contains("gl.surface-stencil-attachment=none"),
        "{flags:?}"
    );
    assert!(
        !flags
            .iter()
            .any(|marker| marker.starts_with("gl.surface-depth-bits")
                || marker.starts_with("gl.surface-stencil-bits")),
        "an absent attachment was recorded as a width: {flags:?}"
    );
    assert!(
        !flags
            .iter()
            .any(|marker| marker.contains("facts-unavailable")),
        "{flags:?}"
    );
    // Present but unreadable is the third outcome, and it must not read as
    // either of the other two.
    let refused = snapshot_of(&SurfaceFactQuery {
        depth: (FRAMEBUFFER_DEFAULT, 24),
        ..SurfaceFactQuery::observed()
    });
    let flags = refused.context().flags().other.clone();
    assert!(flags.contains("gl.surface-depth-bits=24"), "{flags:?}");
    assert!(
        !flags.contains("gl.surface-depth-attachment=none"),
        "{flags:?}"
    );
}

/// The typed surface facts answer the same question the recorded keys do.
///
/// One observation produces both renderings, and this asserts the property that
/// makes the duplicate worth having: the value and the keys agree about what was
/// observed, and a width the value cannot hold is a width the observation did not
/// produce. The last case is the one that separates "narrow the number and carry
/// on" from "the observation failed", and it has to be the second: a wrapped width
/// is a format claim, and the record would then say a format was observed that the
/// drawable never had.
#[test]
fn the_typed_surface_facts_agree_with_the_recorded_keys() {
    let facts_of = |query: &SurfaceFactQuery| {
        discover_with(query, &all_pass_plan())
            .expect("complete mock discovery")
            .surface_facts()
    };
    assert_eq!(
        facts_of(&SurfaceFactQuery::observed()),
        GlSurfaceFacts::Observed {
            color_bits: [8, 8, 8, 8],
        }
    );
    for (query, case) in [
        (
            SurfaceFactQuery {
                binding: Some(2),
                ..SurfaceFactQuery::observed()
            },
            "draw framebuffer bound",
        ),
        (
            SurfaceFactQuery {
                binding: None,
                ..SurfaceFactQuery::observed()
            },
            "binding unqueried",
        ),
        (
            SurfaceFactQuery {
                color: Some((glow_const::BACK_LEFT, None)),
                ..SurfaceFactQuery::observed()
            },
            "colour widths unqueried",
        ),
    ] {
        assert_eq!(
            facts_of(&query),
            GlSurfaceFacts::Unavailable,
            "{case} must claim no format"
        );
    }
    let negative = SurfaceFactQuery {
        color: Some((glow_const::BACK_LEFT, Some([8, 8, 8, -8]))),
        ..SurfaceFactQuery::observed()
    };
    let snapshot = discover_with(&negative, &all_pass_plan()).expect("complete mock discovery");
    assert_eq!(snapshot.surface_facts(), GlSurfaceFacts::Unavailable);
    let flags = snapshot.context().flags().other.clone();
    assert!(
        flags.contains(
            "gl.surface-facts-unavailable=query-failed:GL_BACK_LEFT:\
             GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE"
        ),
        "{flags:?}"
    );
    assert!(
        !flags
            .iter()
            .any(|marker| marker.starts_with("gl.surface-") && !marker.contains("unavailable")),
        "{flags:?}"
    );
}
