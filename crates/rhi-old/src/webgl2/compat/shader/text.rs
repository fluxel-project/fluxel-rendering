//! The lowered GLSL bodies of the fifteen closed artifacts.
//!
//! One body per stage.  A body carries the declarations that stage alone makes
//! and nothing that belongs to the context or to the program: no `#version`
//! line, no precision statements, and no frame block.  Those come from
//! [`super::raster`] and [`super::compute`], which compose
//! `header + block + body`, and the frame block is a single constant here
//! rather than seven copies below, because a uniform block is declared once per
//! *program* and restated in each stage that reads it -- a repetition the
//! language requires and a mistake a reader should only have to check once.
//!
//! The ten raster bodies and the five compute ones share this module because
//! the two families' texts have the same property and are read the same way --
//! a table, not a compiler -- and because both are translations of a WGSL
//! original that lives in the artifact module.  What separates them is the
//! profile floor rather than the kind of text: everything in the raster section
//! is core in GLSL ES 3.00 and desktop GLSL 1.40, which is where Layer 1's
//! dialect rule starts, while the compute section uses a stage and a storage
//! qualifier neither of those has, and the lowering that composes it refuses
//! every profile below embedded 3.10 and desktop 4.30.
//!
//! Everything in the raster section below the header is written once and works
//! for every profile that family lowers for, which is a claim about the two
//! dialects rather than a hope: every construct used there --
//! `layout(location = ...) in`, `layout(std140) uniform` blocks,
//! `gl_VertexID`, `textureSize`/`texelFetch`/`texture`, `inversesqrt` -- is core
//! in GLSL ES 3.00 and in desktop GLSL 1.40 or later, and Layer 1 admits no
//! profile below either.
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

// ---------------------------------------------------------------------------
// The five compute artifacts.
// ---------------------------------------------------------------------------

/// The storage block the arithmetic pair reads and writes.
///
/// A GLSL storage block's *name* is what a provider reflects by -- the
/// declarations below carry no instance name -- so the block takes the
/// PascalCase form the frame uniform block established while its member keeps
/// the WGSL resource's own spelling.
pub(super) const COMPUTE_VALUES_BLOCK: &str = "Values";

/// The storage block the two packing kernels write.
pub(super) const COMPUTE_DESTINATION_BLOCK: &str = "Destination";

/// The sampled texture the pack kernel fetches from.
pub(super) const COMPUTE_SOURCE_TEXTURE: &str = "source";

/// The storage image the store kernel writes.
pub(super) const COMPUTE_OUTPUT_IMAGE: &str = "output_image";

/// The storage image the load kernel reads.
///
/// The same spelling as [`COMPUTE_SOURCE_TEXTURE`] because it is the same WGSL
/// name in a different kernel: the two never share a program, and giving them
/// different names here would invent a distinction the originals do not make.
pub(super) const COMPUTE_SOURCE_IMAGE: &str = "source";

/// Adds one to every addressed element, with wrapping arithmetic.
///
/// GLSL has no unsigned wrapping builtin, and none is needed: the language
/// defines `+` on `uint` as modulo 2^32, which is what the WGSL original's
/// `+ 1u` means.
pub(super) const COMPUTE_WRAPPING_ADD: &str = "\
layout(std430) buffer Values {
    uint values[];
};
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < values.length()) {
        values[index] = values[index] + 1u;
    }
}
";

/// Multiplies every addressed element by three, with wrapping arithmetic.
///
/// It differs from [`COMPUTE_WRAPPING_ADD`] in one expression and is a full
/// body rather than a parameterized one, because a shader body composed through
/// `format!` would need every one of its braces doubled and would stop being
/// readable beside the WGSL it translates.  What keeps the two from drifting is
/// a test that holds the difference to exactly that expression.
pub(super) const COMPUTE_WRAPPING_MULTIPLY: &str = "\
layout(std430) buffer Values {
    uint values[];
};
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < values.length()) {
        values[index] = values[index] * 3u;
    }
}
";

/// Packs every texel of one texture into row-major `u32`s.
///
/// `texelFetch` rather than `texture`: the WGSL loads an unnormalized integer
/// coordinate with no sampler, so a filtered lookup would be a different image.
pub(super) const COMPUTE_TEXTURE_PACK_RGBA8: &str = "\
layout(std430) buffer Destination {
    uint destination[];
};
uniform sampler2D source;
void main() {
    ivec2 dimensions = textureSize(source, 0);
    ivec2 coordinate = ivec2(gl_GlobalInvocationID.xy);
    if (coordinate.x >= dimensions.x || coordinate.y >= dimensions.y) { return; }
    vec4 pixel = texelFetch(source, coordinate, 0);
    uint r = uint(round(pixel.r * 255.0));
    uint g = uint(round(pixel.g * 255.0));
    uint b = uint(round(pixel.b * 255.0));
    uint a = uint(round(pixel.a * 255.0));
    destination[coordinate.y * dimensions.x + coordinate.x] = r | (g << 8u) | (b << 16u) | (a << 24u);
}
";

/// Stores a fixed value into every texel of a storage image.
///
/// The format qualifier is not decoration: a storage image declaration has to
/// carry one, and `rgba8` is the format the artifact's name and its WGSL
/// declaration both state.
pub(super) const COMPUTE_TEXTURE_STORE_RGBA8: &str = "\
layout(rgba8) writeonly uniform image2D output_image;
void main() {
    ivec2 dimensions = imageSize(output_image);
    ivec2 coordinate = ivec2(gl_GlobalInvocationID.xy);
    if (coordinate.x < dimensions.x && coordinate.y < dimensions.y) {
        imageStore(output_image, coordinate, vec4(0.25, 0.5, 0.75, 1.0));
    }
}
";

/// Loads storage-image texels and packs them into a storage buffer.
///
/// The same packing as [`COMPUTE_TEXTURE_PACK_RGBA8`] over a different source,
/// which is what the two WGSL originals differ by: one fetches a sampled
/// texture, this one reads a storage image, and the arithmetic is identical.
pub(super) const COMPUTE_TEXTURE_LOAD_RGBA8: &str = "\
layout(rgba8) readonly uniform image2D source;
layout(std430) buffer Destination {
    uint destination[];
};
void main() {
    ivec2 dimensions = imageSize(source);
    ivec2 coordinate = ivec2(gl_GlobalInvocationID.xy);
    if (coordinate.x >= dimensions.x || coordinate.y >= dimensions.y) { return; }
    vec4 pixel = imageLoad(source, coordinate);
    uint r = uint(round(pixel.r * 255.0));
    uint g = uint(round(pixel.g * 255.0));
    uint b = uint(round(pixel.b * 255.0));
    uint a = uint(round(pixel.a * 255.0));
    destination[coordinate.y * dimensions.x + coordinate.x] = r | (g << 8u) | (b << 16u) | (a << 24u);
}
";
