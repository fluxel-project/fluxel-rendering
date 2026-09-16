//! The ten closed raster artifacts, lowered onto Layer 1's shader contract.
//!
//! Responsibility: turn one [`RasterKernel`] and the profile of the context it
//! will run on into the [`GlProgramDescriptor`] and [`GlVertexLayout`] a
//! provider accepts -- the program's two stages, its logical binding layout,
//! and the vertex input shape.  Nothing here links, creates, or records: the
//! descriptor is data, and Layer 2's program cache is what makes a program out
//! of it.
//!
//! Not owned here: the *native* resources a binding names (the adapter's
//! binding factories do that), the rasterization state a pipeline carries
//! (the pipeline slice's), and the pass a draw happens in (the session's).
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
//! adding naga to a browser build is the set itself: ten closed recipes, fixed
//! before any of this exists, whose GLSL is a table and not a compiler.
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
//!
//! # Two of the ten are translations of one recipe, not one recipe each
//!
//! The sRGB kernel's WGSL differs from the linear-clamp kernel's only in the
//! texture it samples, and on this family the sRGB decode is a property of the
//! texture's internal format rather than of the shader.  So the two share a
//! fragment body and differ in their binding, which is the binding factory's
//! business rather than this module's -- and the shared body is recorded in
//! [`text`] so that a reader comparing the two kernels is not left wondering
//! which of them is missing.

use crate::resource::{RasterKernel, RasterVertexLayout};
use crate::webgl2::api::{
    GlBindingLocation, GlFamilyProfile, GlLogicalBinding, GlPipelineLayout, GlProgramDescriptor,
    GlProgramKind, GlShaderDialect, GlShaderResourceKind, GlShaderSource, GlShaderStage,
    GlVertexAttribute, GlVertexBufferLayout, GlVertexFormat, GlVertexLayout, GlVertexStepMode,
    ShaderSourceHash,
};

mod text;

#[cfg(test)]
mod tests;

/// The GLSL ES preamble, without its version line.
///
/// `precision highp float` is mandatory for a fragment shader in ES and is
/// declared here for both stages so that one body serves either.  The `int` and
/// `sampler2D` statements are there for the texel-fetch pair, which computes
/// `ivec2` texel coordinates and fetches from a sampler whose ES default
/// precision is `lowp`: a default would be legal and would quietly truncate the
/// coordinate on a large texture.
const ES_PRECISION: &str = "\
precision highp float;
precision highp int;
precision highp sampler2D;
";

/// The two fixed entry points a GLSL program has, whatever its recipe is.
///
/// GLSL has no entry-point selection: the linked program's entry is `main` in
/// both stages, and the recipe's own entry-point names (which the WGSL sources
/// do carry, and which [`RasterKernel::portable_identity`] records) are not
/// names this family can use.
const ENTRY_POINT: &str = "main";

/// Lowers one fixed raster artifact for one context profile.
///
/// The error case is a profile this family has no dialect for, and it is the
/// same set Layer 1's own dialect rule admits: a profile [`GlShaderSource`]
/// would refuse is one this lowering refuses first, with a reason of its own
/// rather than a link-time surprise.
pub(super) fn program(
    kernel: RasterKernel,
    profile: GlFamilyProfile,
) -> Result<GlProgramDescriptor, UnsupportedProfile> {
    let (dialect, header) = dialect_for(profile).ok_or(UnsupportedProfile)?;
    let block = if kernel.portable_identity().uniform_binding.is_some() {
        text::FRAME_UNIFORMS
    } else {
        ""
    };
    let (vertex_body, fragment_body) = bodies(kernel);
    let vertex = source(
        GlShaderStage::Vertex,
        dialect,
        format!("{header}{block}{vertex_body}"),
    );
    let fragment = source(
        GlShaderStage::Fragment,
        dialect,
        format!("{header}{block}{fragment_body}"),
    );
    Ok(GlProgramDescriptor {
        kind: GlProgramKind::Raster { vertex, fragment },
        layout: layout(kernel),
        debug_name: Some(format!("fixed-raster:{}", kernel.vertex_entry_point())),
    })
}

/// The vertex input shape one fixed artifact declares.
///
/// The strides come from the artifact's own identity rather than from numbers
/// written here, so a recipe whose layout changed would change this with it.
/// Slot one is the only place a second stream appears.  It holds exactly one
/// attribute in each of the three recipes that have one, and the recipe calls
/// it tightly packed, so its stride is *zero* -- which is what Layer 1's own
/// validation reads a zero stride as, and what the provider then passes to the
/// GL entry point that reads zero the same way.  The identity's recorded
/// slot-one stride is used where it exists, because a recipe that recorded one
/// would have recorded why.
pub(super) fn vertex_layout(kernel: RasterKernel) -> GlVertexLayout {
    let identity = kernel.portable_identity();
    let stride = identity.vertex_stride;
    let slot = |slot: u32, stride: u32| GlVertexBufferLayout {
        slot,
        stride,
        step_mode: GlVertexStepMode::Vertex,
    };
    let attribute =
        |location: u32, buffer_slot: u32, format: GlVertexFormat, offset: u32| GlVertexAttribute {
            location,
            buffer_slot,
            format,
            offset,
        };
    match identity.vertex_layout {
        RasterVertexLayout::None => GlVertexLayout {
            buffers: Vec::new(),
            attributes: Vec::new(),
        },
        RasterVertexLayout::PositionFloat32x2ColorUnorm8x4 => GlVertexLayout {
            buffers: vec![slot(0, stride)],
            attributes: vec![
                attribute(0, 0, GlVertexFormat::Float32x2, 0),
                attribute(1, 0, GlVertexFormat::Unorm8x4, 8),
            ],
        },
        RasterVertexLayout::PositionFloat32x3 => GlVertexLayout {
            buffers: vec![slot(0, stride)],
            attributes: vec![attribute(0, 0, GlVertexFormat::Float32x3, 0)],
        },
        RasterVertexLayout::PositionFloat32x3AndTextureCoordinateFloat32x2 => GlVertexLayout {
            buffers: vec![slot(0, stride), slot(1, second_stream_stride(&identity))],
            attributes: vec![
                attribute(0, 0, GlVertexFormat::Float32x3, 0),
                attribute(1, 1, GlVertexFormat::Float32x2, 0),
            ],
        },
        RasterVertexLayout::PositionFloat32x3AndNormalFloat32x3 => GlVertexLayout {
            buffers: vec![slot(0, stride), slot(1, second_stream_stride(&identity))],
            attributes: vec![
                attribute(0, 0, GlVertexFormat::Float32x3, 0),
                attribute(1, 1, GlVertexFormat::Float32x3, 0),
            ],
        },
        RasterVertexLayout::PositionFloat32x3AndColorUnorm8x4 => GlVertexLayout {
            buffers: vec![slot(0, stride), slot(1, second_stream_stride(&identity))],
            attributes: vec![
                attribute(0, 0, GlVertexFormat::Float32x3, 0),
                attribute(1, 1, GlVertexFormat::Unorm8x4, 0),
            ],
        },
    }
}

/// The stride of a recipe's second vertex stream.
fn second_stream_stride(identity: &crate::resource::RasterArtifactIdentity) -> u32 {
    identity.texture_coordinate_vertex_stride.unwrap_or(0)
}

/// The logical bindings one fixed artifact declares, in program order.
///
/// A bounded number of entries, and deliberately not one per WGSL binding: a
/// GLSL `sampler2D` *is* a texture and its sampler, so the linear-clamp pair's
/// two WGSL bindings -- a texture and a sampler -- are one logical binding
/// here.  Layer 1's own vocabulary names that case
/// ([`GlShaderResourceKind::CombinedTextureSampler`]), and its reflection rule
/// requires distinct executable assignments, which two logical bindings sharing
/// one texture unit could not satisfy.  Which WGSL binding the sampler arrived
/// on is therefore a fact the binding factory needs and this layout cannot
/// carry; it stays in the identity, where `sampler_binding` records it.
pub(super) fn layout(kernel: RasterKernel) -> GlPipelineLayout {
    let identity = kernel.portable_identity();
    let mut bindings = Vec::new();
    if let Some(binding) = identity.uniform_binding {
        bindings.push(GlLogicalBinding {
            name: text::FRAME_UNIFORMS_NAME.to_owned(),
            location: GlBindingLocation { group: 0, binding },
            kind: GlShaderResourceKind::UniformBuffer,
            array_count: 1,
        });
    }
    if let Some(binding) = identity.texture_binding {
        bindings.push(GlLogicalBinding {
            name: text::TEXTURE_NAME.to_owned(),
            location: GlBindingLocation { group: 0, binding },
            kind: GlShaderResourceKind::CombinedTextureSampler,
            array_count: 1,
        });
    }
    GlPipelineLayout { bindings }
}

/// The two stage bodies one fixed artifact is built from.
fn bodies(kernel: RasterKernel) -> (&'static str, &'static str) {
    match kernel {
        RasterKernel::Triangle => (text::TRIANGLE_VERTEX, text::COLOR_FRAGMENT),
        RasterKernel::IndexedPositionColor => (text::POSITION_COLOR_VERTEX, text::COLOR_FRAGMENT),
        RasterKernel::IndexedPositionFloat32x3 => {
            (text::POSITION_FLOAT32X3_VERTEX, text::FIXED_COLOR_FRAGMENT)
        }
        RasterKernel::IndexedPositionFloat32x3CameraMaterial => {
            (text::CAMERA_MATERIAL_VERTEX, text::CAMERA_MATERIAL_FRAGMENT)
        }
        RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture => (
            text::CAMERA_MATERIAL_TEXTURE_VERTEX,
            text::TEXTURE_COLOR_FRAGMENT,
        ),
        RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv => (
            text::CAMERA_MATERIAL_TEXTURE_UV_VERTEX,
            text::TEXTURE_COLOR_FRAGMENT,
        ),
        RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
        | RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb => (
            text::CAMERA_MATERIAL_TEXTURE_UV_VERTEX,
            text::LINEAR_CLAMP_TEXTURE_COLOR_FRAGMENT,
        ),
        RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert => (
            text::CAMERA_MATERIAL_NORMAL_LAMBERT_VERTEX,
            text::NORMAL_LAMBERT_FRAGMENT,
        ),
        RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor => (
            text::CAMERA_MATERIAL_VERTEX_COLOR_VERTEX,
            text::VERTEX_COLOR_FRAGMENT,
        ),
    }
}

/// The dialect a profile requires, and the preamble that goes with it.
///
/// This mirrors Layer 1's own rule (`GlShaderSource::validate_for`) rather than
/// extending it: the two must agree, and the set admitted here is exactly the
/// set that rule admits, so a profile this function answers for cannot be one
/// the provider then refuses for its dialect.
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

/// The lowering was asked for a profile it has no dialect for.
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
