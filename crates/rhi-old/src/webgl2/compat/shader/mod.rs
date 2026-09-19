//! The fixed recipes of this family, lowered onto Layer 1's shader contract.
//!
//! Responsibility: own the two things every lowering here needs and neither
//! family decides for itself -- the dialect rule that maps a context profile to
//! the GLSL version it requires, and the composition of one stage's source from
//! a version header, an optional uniform block and a body.  A family is one
//! module, because a new raster recipe and a new compute kernel are changes that
//! do not touch one another:
//!
//! - [`raster`] lowers one
//!   [`RasterKernel`](crate::resource::RasterKernel) and a profile into the
//!   program descriptor the provider accepts, together with the vertex input
//!   shape that descriptor is paired with.
//! - [`compute`] lowers one [`ComputeKernel`](crate::resource::ComputeKernel)
//!   and a profile into the same descriptor shape with a single compute stage.
//!   It narrows the dialect rule on the way, because two of the profiles that
//!   rule admits have no compute stage to lower onto.
//!
//! Nothing here links, creates, or records: a descriptor is data, and Layer 2's
//! program cache is what makes a program out of it.
//!
//! # Both families are re-exported here now
//!
//! A family's entry points are reached by `compat` through a re-export, because
//! its own module is private to this one.  Only [`raster`]'s were, while the
//! compute lowering stood alone: a re-export no non-test code uses is an
//! `unused_imports` warning, which this crate's `-D warnings` gate turns into a
//! failure, so the compute re-export waited for the compute object that
//! consumes it.  That object is `compat::device::object::ComputePipeline`, and
//! the re-export below landed with it rather than ahead of it.
//!
//! # Why the text is authored per family rather than translated
//!
//! [`GlShaderSource`] takes text that has already been lowered: it carries a
//! [`GlShaderDialect`] and a provider rejects a dialect that does not match its
//! profile.  The private preparation seam
//! (`crate::shader_contract::ShaderModuleSource::route_for`) answers the cross
//! from the retained path's source language to this family's as
//! `ShaderRoute::Translatable` -- naga's `wgsl-in` with `glsl-out`.  That route
//! does not exist in this crate for the target that needs it: `naga` is a
//! Windows-only dependency with the `wgsl-in` feature alone
//! (`crates/rhi/Cargo.toml`), while the contexts that need GLSL are wasm32
//! (behind the `webgl2` feature, which pulls no naga at all) and the native GL
//! providers, where the `glsl-out` backend is not compiled.  So this family
//! lowers its own fixed recipes, and the argument for doing that rather than
//! adding naga to a browser build is the set itself: fifteen closed recipes
//! (ten raster, five compute), fixed before any of this exists, whose GLSL is a
//! table and not a compiler.
//!
//! The consequence is a limit on what the tests here can prove, and it is
//! stated rather than implied.  The mock provider does not compile GLSL --
//! `MockGlFamilyApi::create_program` checks the descriptor against the profile
//! and never reads the text past `is_empty()` -- so what is measured here is
//! *structural*: which dialect each profile requires, that a program with these
//! declarations and this layout is one Layer 1 accepts, that the block and
//! uniform names the layout claims are the ones the text declares, and that the
//! same kernel lowers to the same bytes twice.  That the shaders are valid GLSL,
//! and that they compute what their WGSL originals compute, is hardware
//! evidence and belongs to the release gate's real-context runs.  A green suite
//! here is not that evidence and must not be read as it.

mod compute;
mod raster;
mod text;

#[cfg(test)]
mod tests;

// Each family's entry points, in the order `compat`'s two objects call them.
// The compute layout is re-exported beside the lowering that produces it
// because a binding set is validated against the *arrangement* and not against
// the descriptor: the object that validates it has no profile to lower for, and
// a second table there would be a third statement about the same kernel beside
// the body and the layout.
pub(super) use compute::{compute_program, layout as compute_layout};
pub(super) use raster::{program, vertex_layout};

use crate::webgl2::api::{
    GlFamilyProfile, GlShaderDialect, GlShaderSource, GlShaderStage, ShaderSourceHash,
};

/// The GLSL ES preamble, without its version line.
///
/// `precision highp float` is mandatory for a fragment shader in ES, and is
/// declared for every stage so that one body serves either.  The `int` and
/// `sampler2D` statements are there for the texel-fetch pair, which computes
/// `ivec2` texel coordinates and fetches from a sampler whose ES default
/// precision is `lowp`: a default would be legal and would quietly truncate the
/// coordinate on a large texture.
const ES_PRECISION: &str = "\
precision highp float;
precision highp int;
precision highp sampler2D;
";

/// The entry point every GLSL stage of every fixed recipe has.
///
/// GLSL has no entry-point selection: the linked program's entry is `main` in
/// every stage, and the recipes' own entry-point names (which the WGSL sources
/// do carry, and which
/// [`RasterKernel::portable_identity`](crate::resource::RasterKernel::portable_identity)
/// and
/// [`ComputeKernel::portable_identity`](crate::resource::ComputeKernel::portable_identity)
/// record) are not names this family can use.
const ENTRY_POINT: &str = "main";

/// The dialect a profile requires, and the preamble that goes with it.
///
/// This mirrors Layer 1's own rule (`GlShaderSource::validate_for`) rather than
/// extending it: the two must agree, and the set admitted here is exactly the
/// set that rule admits, so a profile this function answers for cannot be one
/// the provider then refuses for its dialect.
///
/// It answers for a profile's *dialect* and nothing more.  Whether a family can
/// be expressed in that dialect is the family's question: GLSL ES 300 has no
/// compute stage, so the compute family will narrow this answer rather than
/// replace it, and desktop versions below 4.30 are admitted here and will be
/// refused there.
fn dialect_for(profile: GlFamilyProfile) -> Option<(GlShaderDialect, String)> {
    let embedded = |version: u16| {
        (
            GlShaderDialect::Embedded { version },
            format!("#version {version} es\n{ES_PRECISION}"),
        )
    };
    match profile {
        GlFamilyProfile::WebGl2 => Some(embedded(300)),
        GlFamilyProfile::Embedded { major: 3, minor } => match minor {
            0 => Some(embedded(300)),
            1 => Some(embedded(310)),
            2 => Some(embedded(320)),
            _ => None,
        },
        GlFamilyProfile::Desktop { major: 4, minor } => {
            let version = 400 + u16::from(minor) * 10;
            Some((
                GlShaderDialect::Desktop { version },
                format!("#version {version}\n"),
            ))
        }
        _ => None,
    }
}

/// One stage's lowered source.
fn source(stage: GlShaderStage, dialect: GlShaderDialect, text: String) -> GlShaderSource {
    GlShaderSource {
        stage,
        dialect,
        entry_point: ENTRY_POINT.to_owned(),
        source_hash: content_hash(&text),
        text,
        debug_name: None,
    }
}

/// A content identity for one lowered stage.
///
/// The contract says the digest algorithm is the producer's and that the
/// producer and the artifact cache must agree on it; nothing in this crate
/// reads the value back, and Layer 2's program cache keys on the descriptor's
/// own contents rather than on this.  So the digest is what it needs to be --
/// a deterministic function of the text, equal for equal text -- and not a
/// security primitive.  It is one 64-bit FNV-1a fold, written across the
/// transport's 32 bytes by folding the previous lane into the next; the entropy
/// is 64 bits and the width is the type's.
fn content_hash(text: &str) -> ShaderSourceHash {
    let mut digest = [0u8; 32];
    let mut carry = crate::resource::fnv1a64(text.as_bytes());
    for lane in digest.chunks_exact_mut(8) {
        lane.copy_from_slice(&carry.to_le_bytes());
        carry = crate::resource::fnv1a64(&carry.to_le_bytes());
    }
    ShaderSourceHash(digest)
}

/// The raster lowering was asked for a profile it has no dialect for.
///
/// A unit error rather than a [`crate::webgl2::api::GlError`], because it is a
/// fact about the request and not about a context: the caller names the
/// operation it was serving when it reports this.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct UnsupportedProfile;

impl UnsupportedProfile {
    /// The reason an adapter reports for this refusal.
    pub(super) const REASON: &'static str = "the fixed raster artifacts are lowered only for the profiles this family's dialect rule admits";
}

/// The compute lowering was asked for a profile with no compute stage.
///
/// A unit error for [`UnsupportedProfile`]'s reason: it is a fact about the
/// request and not about a context, and the caller names the operation it was
/// serving when it reports this.
///
/// It is a second type rather than a second variant of the first because the
/// two refusals are different facts -- a profile the dialect rule does not admit
/// at all, and a profile it admits whose language has no compute stage -- and a
/// caller that wants to say which one it hit should not have to inspect a
/// payload to find out.  Both live here rather than in the family that raises
/// them because the parent owns the dialect rule they are both about.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct UnsupportedComputeProfile;

impl UnsupportedComputeProfile {
    /// The reason an adapter reports for this refusal.
    ///
    /// It names the stage rather than a version list, because the list is
    /// `compute::supports_compute`'s and a second copy of it here would be the
    /// thing that goes stale.
    pub(super) const REASON: &'static str = "the fixed compute artifacts are lowered only for profiles whose shading language has a compute stage";
}
