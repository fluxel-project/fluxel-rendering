//! The lowered GLSL bodies of the ten closed raster artifacts.
//!
//! One body per stage.  A body carries the declarations that stage alone makes
//! and nothing that belongs to the context or to the program: no `#version`
//! line, no precision statements, and no frame block.  Those come from
//! [`super::program`], which composes `header + block + body`, and the frame
//! block is a single constant here rather than seven copies below, because a
//! uniform block is declared once per *program* and restated in each stage that
//! reads it -- a repetition the language requires and a mistake a reader should
//! only have to check once.
//!
//! Everything below the header is written once and works for every profile this
//! family lowers for, which is a claim about the two dialects rather than a
//! hope: every construct used here -- `layout(location = ...) in`,
//! `layout(std140) uniform` blocks, `gl_VertexID`,
//! `textureSize`/`texelFetch`/`texture`, `inversesqrt` -- is core in GLSL ES
//! 3.00 and in desktop GLSL 1.40 or later, and Layer 1 admits no profile below
//! either.
//!
//! # These are translations, and they are meant to be read beside the originals
//!
//! Each body computes what [`RasterKernel::wgsl_source`] computes, statement
//! for statement, because the two are two lowerings of one fixed recipe and a
//! difference between them is a difference between two backends' images.  Three
//! of the ten need a decision that is not a syntax swap:
//!
//! - the **texel-fetch pair** (`...Texture` and `...TextureUv`) reproduces the
//!   WGSL's manual addressing -- clamp, scale by the queried extent, floor,
//!   clamp to the last texel -- rather than calling `texture`, because that is
//!   what the original does and a nearest-neighbour fetch at an explicitly
//!   computed texel is not the same image as a filtered lookup.
//! - the **two linear-clamp kernels share one body**, and the sRGB one is not a
//!   separate program.  The WGSL pair differ only in the texture they sample,
//!   and the decode its name refers to happens in the *sampler object and the
//!   texture's internal format* on this family, not in shader arithmetic: a
//!   texture whose internal format is sRGB is decoded by the hardware before
//!   the texel reaches the shader.  Writing a manual decode here would decode
//!   twice.
//! - the **Lambert fragment** writes the zero-length case out rather than
//!   leaving it to `normalize`, which divides by a zero length and would
//!   produce a NaN where the recipe specifies exactly zero contribution.
//!
//! [`RasterKernel::wgsl_source`]: crate::resource::RasterKernel::wgsl_source

/// The frame block, declared by every camera/material kernel in both of its
/// stages.
///
/// A GLSL uniform block is visible only to the stage that declares it, so a
/// fragment shader reading `base_color` while the vertex shader declared the
/// block would not link.  The block is one block -- a program has one, and a
/// provider reflects it once -- but its declaration has to appear in each stage
/// that reads it, which is why this is composed into both rather than written
/// into one of them.
pub(super) const FRAME_UNIFORMS: &str = "\
layout(std140) uniform FrameUniforms {
    mat4 view_projection;
    vec4 base_color;
};
";

/// The uniform block's name, which is also the logical binding's name.
///
/// Kept beside the declaration because the two must agree: a provider reflects
/// the linked program by looking the block up *by this string*, so a layout
/// that spelled it differently from the shader would fail at link time on a
/// real context and nowhere else.
pub(super) const FRAME_UNIFORMS_NAME: &str = "FrameUniforms";

/// The texture's name, for the same reason.
pub(super) const TEXTURE_NAME: &str = "tex";

/// A vertex shader that derives its three positions from the vertex index.
pub(super) const TRIANGLE_VERTEX: &str = "\
out vec4 v_color;
void main() {
    if (gl_VertexID == 0) {
        gl_Position = vec4(-0.70, -0.60, 0.0, 1.0);
    } else if (gl_VertexID == 1) {
        gl_Position = vec4(0.70, -0.60, 0.0, 1.0);
    } else {
        gl_Position = vec4(0.0, 0.70, 0.0, 1.0);
    }
    v_color = vec4(64.0 / 255.0, 160.0 / 255.0, 1.0, 1.0);
}
";

/// The fixed color, passed through from whichever stage produced it.
pub(super) const COLOR_FRAGMENT: &str = "\
in vec4 v_color;
out vec4 out_color;
void main() {
    out_color = v_color;
}
";

/// `float32x2` position with a `unorm8x4` color, both pre-normalized by the
/// vertex-array format rather than in the shader.
pub(super) const POSITION_COLOR_VERTEX: &str = "\
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec4 a_color;
out vec4 v_color;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_color = a_color;
}
";

/// `float32x3` position with no attribute-derived varying.
pub(super) const POSITION_FLOAT32X3_VERTEX: &str = "\
layout(location = 0) in vec3 a_position;
void main() {
    gl_Position = vec4(a_position, 1.0);
}
";

/// The one fragment body with neither a varying nor a binding to read.
pub(super) const FIXED_COLOR_FRAGMENT: &str = "\
out vec4 out_color;
void main() {
    out_color = vec4(48.0 / 255.0, 176.0 / 255.0, 112.0 / 255.0, 1.0);
}
";

/// The camera/material vertex stage: transform, and nothing else.
pub(super) const CAMERA_MATERIAL_VERTEX: &str = "\
layout(location = 0) in vec3 a_position;
void main() {
    gl_Position = view_projection * vec4(a_position, 1.0);
}
";

/// The camera/material fragment stage: the material color alone.
pub(super) const CAMERA_MATERIAL_FRAGMENT: &str = "\
out vec4 out_color;
void main() {
    out_color = base_color;
}
";

/// The camera/material vertex stage that derives texture coordinates from the
/// position, as the fixed recipe does rather than from a second stream.
pub(super) const CAMERA_MATERIAL_TEXTURE_VERTEX: &str = "\
layout(location = 0) in vec3 a_position;
out vec2 v_uv;
void main() {
    gl_Position = view_projection * vec4(a_position, 1.0);
    v_uv = a_position.xy * vec2(0.5, -0.5) + vec2(0.5);
}
";

/// The texel-fetch fragment stage, over an explicitly computed texel.
pub(super) const TEXTURE_COLOR_FRAGMENT: &str = "\
uniform sampler2D tex;
in vec2 v_uv;
out vec4 out_color;
void main() {
    ivec2 dimensions = textureSize(tex, 0);
    ivec2 texel = min(
        ivec2(floor(clamp(v_uv, vec2(0.0), vec2(1.0)) * vec2(dimensions))),
        dimensions - ivec2(1)
    );
    out_color = base_color * texelFetch(tex, texel, 0);
}
";

/// The same vertex stage with its texture coordinates taken from a second
/// stream instead of from the position.
pub(super) const CAMERA_MATERIAL_TEXTURE_UV_VERTEX: &str = "\
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_uv;
out vec2 v_uv;
void main() {
    gl_Position = view_projection * vec4(a_position, 1.0);
    v_uv = a_uv;
}
";

/// The linear-clamp fragment stage.  Level zero and the clamp addressing are
/// properties of the sampler bound to `tex`, and the sRGB kernel shares this
/// body because its decode is a property of the texture's internal format.
pub(super) const LINEAR_CLAMP_TEXTURE_COLOR_FRAGMENT: &str = "\
uniform sampler2D tex;
in vec2 v_uv;
out vec4 out_color;
void main() {
    out_color = base_color * texture(tex, v_uv);
}
";

/// The Lambert vertex stage: position and unit normal, both object space.
pub(super) const CAMERA_MATERIAL_NORMAL_LAMBERT_VERTEX: &str = "\
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
out vec3 v_normal;
void main() {
    gl_Position = view_projection * vec4(a_position, 1.0);
    v_normal = a_normal;
}
";

/// The Lambert fragment stage.
pub(super) const NORMAL_LAMBERT_FRAGMENT: &str = "\
in vec3 v_normal;
out vec4 out_color;
void main() {
    float length_squared = dot(v_normal, v_normal);
    if (length_squared > 0.0) {
        float lambert = max(
            dot(v_normal * inversesqrt(length_squared), vec3(0.0, 0.0, 1.0)),
            0.0
        );
        out_color = vec4(base_color.rgb * lambert, base_color.a);
    } else {
        out_color = vec4(vec3(0.0), base_color.a);
    }
}
";

/// The vertex-color vertex stage.
pub(super) const CAMERA_MATERIAL_VERTEX_COLOR_VERTEX: &str = "\
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4 a_color;
out vec4 v_color;
void main() {
    gl_Position = view_projection * vec4(a_position, 1.0);
    v_color = a_color;
}
";

/// The vertex-color fragment stage.
pub(super) const VERTEX_COLOR_FRAGMENT: &str = "\
in vec4 v_color;
out vec4 out_color;
void main() {
    out_color = v_color * base_color;
}
";
