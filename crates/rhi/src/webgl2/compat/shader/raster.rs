//! The ten closed raster artifacts, lowered onto Layer 1's shader contract.
//!
//! Responsibility: turn one [`RasterKernel`] and the profile of the context it
//! will run on into the [`GlProgramDescriptor`] and [`GlVertexLayout`] a
//! provider accepts -- the program's two stages, its logical binding layout,
//! and the vertex input shape.  The dialect rule and the source composition are
//! [`super`]'s, because the compute lowering needs the same two and neither
//! family decides them.
//!
//! Not owned here: the *native* resources a binding names (the adapter's
//! binding factories do that), the rasterization state a pipeline carries
//! (the pipeline slice's), and the pass a draw happens in (the session's).
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
    GlProgramKind, GlShaderResourceKind, GlShaderStage, GlVertexAttribute, GlVertexBufferLayout,
    GlVertexFormat, GlVertexLayout, GlVertexStepMode,
};

use super::text;
use super::{UnsupportedProfile, dialect_for, source};

/// Lowers one fixed raster artifact for one context profile.
///
/// The error case is a profile this family has no dialect for, and it is the
/// same set Layer 1's own dialect rule admits: a profile
/// [`GlShaderSource`](crate::webgl2::api::GlShaderSource) would refuse is one
/// this lowering refuses first, with a reason of its own rather than a
/// link-time surprise.
///
/// The three entry points here are `pub(in crate::webgl2::compat)` rather than
/// `pub(super)`: the module that *composes* them is [`super`], and the layer
/// that consumes them is `compat` -- which is where they were visible when they
/// were written in [`super`] itself, so this is the same surface and not a wider
/// one.
pub(in crate::webgl2::compat) fn program(
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
pub(in crate::webgl2::compat) fn vertex_layout(kernel: RasterKernel) -> GlVertexLayout {
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
pub(in crate::webgl2::compat) fn layout(kernel: RasterKernel) -> GlPipelineLayout {
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
