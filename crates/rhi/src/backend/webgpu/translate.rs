//! Pure, fail-closed translation from the RHI vocabulary to WebGPU wire names.
//!
//! This module deliberately has no `web_sys` dependency.  The browser driver
//! owns JS object construction; keeping the vocabulary conversion here makes it
//! testable on the host and prevents an unsupported RHI variant from becoming a
//! convenient JavaScript fallback.

use crate::api::command::{IndexFormat, LoadOp, StoreOp};
use crate::api::format::TextureFormat;
use crate::api::pipeline::{
    BlendFactor, BlendOperation, ColorWriteMask, CullMode, FrontFace, PrimitiveTopology,
    VertexFormat, VertexStepMode,
};
use crate::api::query::QueryType;
use crate::api::resource::subresource::TextureAspects;
use crate::api::resource::{
    AddressMode, BufferUsage, CompareFunction, FilterMode, TextureDimension, TextureUsage,
    TextureViewDimension,
};

/// A portable value which has no WebGPU spelling.
///
/// This is intentionally more specific than `Option`: callers must propagate a
/// refusal to capability/creation validation rather than silently selecting a
/// nearby WebGPU feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Unsupported {
    /// The RHI vocabulary family that could not be represented.
    pub(crate) what: &'static str,
}

pub(crate) type TranslateResult<T> = Result<T, Unsupported>;

const fn unsupported(what: &'static str) -> Unsupported {
    Unsupported { what }
}

/// WebGPU's `GPUTextureFormat` string, if the format has one.
pub(crate) fn texture_format(value: TextureFormat) -> TranslateResult<&'static str> {
    use TextureFormat::*;
    Ok(match value {
        R8Unorm => "r8unorm",
        R8Snorm => "r8snorm",
        R8Uint => "r8uint",
        R8Sint => "r8sint",
        Rg8Unorm => "rg8unorm",
        Rg8Snorm => "rg8snorm",
        Rg8Uint => "rg8uint",
        Rg8Sint => "rg8sint",
        Rgba8Unorm => "rgba8unorm",
        Rgba8UnormSrgb => "rgba8unorm-srgb",
        Rgba8Snorm => "rgba8snorm",
        Rgba8Uint => "rgba8uint",
        Rgba8Sint => "rgba8sint",
        Bgra8Unorm => "bgra8unorm",
        Bgra8UnormSrgb => "bgra8unorm-srgb",
        Bc1RgbaUnorm => "bc1-rgba-unorm",
        Bc1RgbaUnormSrgb => "bc1-rgba-unorm-srgb",
        Bc2RgbaUnorm => "bc2-rgba-unorm",
        Bc2RgbaUnormSrgb => "bc2-rgba-unorm-srgb",
        Bc3RgbaUnorm => "bc3-rgba-unorm",
        Bc3RgbaUnormSrgb => "bc3-rgba-unorm-srgb",
        Bc4RUnorm => "bc4-r-unorm",
        Bc4RSnorm => "bc4-r-snorm",
        Bc5RgUnorm => "bc5-rg-unorm",
        Bc5RgSnorm => "bc5-rg-snorm",
        Bc6hRgbUfloat => "bc6h-rgb-ufloat",
        Bc6hRgbFloat => "bc6h-rgb-float",
        Bc7RgbaUnorm => "bc7-rgba-unorm",
        Bc7RgbaUnormSrgb => "bc7-rgba-unorm-srgb",
        Etc2Rgb8Unorm => "etc2-rgb8unorm",
        Etc2Rgb8UnormSrgb => "etc2-rgb8unorm-srgb",
        Etc2Rgb8A1Unorm => "etc2-rgb8a1unorm",
        Etc2Rgb8A1UnormSrgb => "etc2-rgb8a1unorm-srgb",
        Etc2Rgba8Unorm => "etc2-rgba8unorm",
        Etc2Rgba8UnormSrgb => "etc2-rgba8unorm-srgb",
        EacR11Unorm => "eac-r11unorm",
        EacR11Snorm => "eac-r11snorm",
        EacRg11Unorm => "eac-rg11unorm",
        EacRg11Snorm => "eac-rg11snorm",
        Astc4x4Unorm => "astc-4x4-unorm",
        Astc4x4UnormSrgb => "astc-4x4-unorm-srgb",
        Astc5x4Unorm => "astc-5x4-unorm",
        Astc5x4UnormSrgb => "astc-5x4-unorm-srgb",
        Astc5x5Unorm => "astc-5x5-unorm",
        Astc5x5UnormSrgb => "astc-5x5-unorm-srgb",
        Astc6x5Unorm => "astc-6x5-unorm",
        Astc6x5UnormSrgb => "astc-6x5-unorm-srgb",
        Astc6x6Unorm => "astc-6x6-unorm",
        Astc6x6UnormSrgb => "astc-6x6-unorm-srgb",
        Astc8x5Unorm => "astc-8x5-unorm",
        Astc8x5UnormSrgb => "astc-8x5-unorm-srgb",
        Astc8x6Unorm => "astc-8x6-unorm",
        Astc8x6UnormSrgb => "astc-8x6-unorm-srgb",
        Astc8x8Unorm => "astc-8x8-unorm",
        Astc8x8UnormSrgb => "astc-8x8-unorm-srgb",
        Astc10x5Unorm => "astc-10x5-unorm",
        Astc10x5UnormSrgb => "astc-10x5-unorm-srgb",
        Astc10x6Unorm => "astc-10x6-unorm",
        Astc10x6UnormSrgb => "astc-10x6-unorm-srgb",
        Astc10x8Unorm => "astc-10x8-unorm",
        Astc10x8UnormSrgb => "astc-10x8-unorm-srgb",
        Astc10x10Unorm => "astc-10x10-unorm",
        Astc10x10UnormSrgb => "astc-10x10-unorm-srgb",
        Astc12x10Unorm => "astc-12x10-unorm",
        Astc12x10UnormSrgb => "astc-12x10-unorm-srgb",
        Astc12x12Unorm => "astc-12x12-unorm",
        Astc12x12UnormSrgb => "astc-12x12-unorm-srgb",
        R16Uint => "r16uint",
        R16Sint => "r16sint",
        R16Float => "r16float",
        R16Unorm => "r16unorm",
        R16Snorm => "r16snorm",
        Rg16Uint => "rg16uint",
        Rg16Sint => "rg16sint",
        Rg16Float => "rg16float",
        Rg16Unorm => "rg16unorm",
        Rg16Snorm => "rg16snorm",
        Rgba16Uint => "rgba16uint",
        Rgba16Sint => "rgba16sint",
        Rgba16Float => "rgba16float",
        Rgba16Unorm => "rgba16unorm",
        Rgba16Snorm => "rgba16snorm",
        Rgb9e5Ufloat => "rgb9e5ufloat",
        Rgb10a2Uint => "rgb10a2uint",
        Rgb10a2Unorm => "rgb10a2unorm",
        Rg11b10Ufloat => "rg11b10ufloat",
        R32Uint => "r32uint",
        R32Sint => "r32sint",
        R32Float => "r32float",
        Rg32Uint => "rg32uint",
        Rg32Sint => "rg32sint",
        Rg32Float => "rg32float",
        Rgba32Uint => "rgba32uint",
        Rgba32Sint => "rgba32sint",
        Rgba32Float => "rgba32float",
        Depth16Unorm => "depth16unorm",
        Depth24Plus => "depth24plus",
        Depth24PlusStencil8 => "depth24plus-stencil8",
        Depth32Float => "depth32float",
        Depth32FloatStencil8 => "depth32float-stencil8",
        Stencil8 => "stencil8",
        // ASTC HDR, 64-bit integer, and multi-planar textures are deliberately
        // absent from GPUTextureFormat. External video import is not texture creation.
        Astc4x4Hdr | Astc5x4Hdr | Astc5x5Hdr | Astc6x5Hdr | Astc6x6Hdr | Astc8x5Hdr
        | Astc8x6Hdr | Astc8x8Hdr | Astc10x5Hdr | Astc10x6Hdr | Astc10x8Hdr | Astc10x10Hdr
        | Astc12x10Hdr | Astc12x12Hdr | R64Uint | Nv12 | P010 => {
            return Err(unsupported("TextureFormat"));
        }
    })
}

pub(crate) fn texture_dimension(value: TextureDimension) -> TranslateResult<&'static str> {
    Ok(match value {
        TextureDimension::D1 => "1d",
        TextureDimension::D2 => "2d",
        TextureDimension::D3 => "3d",
    })
}
pub(crate) fn texture_view_dimension(value: TextureViewDimension) -> TranslateResult<&'static str> {
    Ok(match value {
        TextureViewDimension::D1 => "1d",
        TextureViewDimension::D2 => "2d",
        TextureViewDimension::D2Array => "2d-array",
        TextureViewDimension::Cube => "cube",
        TextureViewDimension::CubeArray => "cube-array",
        TextureViewDimension::D3 => "3d",
    })
}

/// Converts a validated aspect selection to `GPUTextureViewDescriptor.aspect`.
pub(crate) fn texture_aspect(value: TextureAspects) -> TranslateResult<&'static str> {
    if value == TextureAspects::COLOR
        || value == TextureAspects::DEPTH.union(TextureAspects::STENCIL)
    {
        return Ok("all");
    }
    if value == TextureAspects::DEPTH {
        return Ok("depth-only");
    }
    if value == TextureAspects::STENCIL {
        return Ok("stencil-only");
    }
    if value == TextureAspects::PLANE0 {
        return Ok("plane0-only");
    }
    if value == TextureAspects::PLANE1 {
        return Ok("plane1-only");
    }
    Err(unsupported("TextureAspects"))
}

pub(crate) fn buffer_usage(value: BufferUsage) -> TranslateResult<u32> {
    let pairs = [
        (BufferUsage::MAP_READ, 1),
        (BufferUsage::MAP_WRITE, 2),
        (BufferUsage::COPY_SRC, 4),
        (BufferUsage::COPY_DST, 8),
        (BufferUsage::INDEX, 16),
        (BufferUsage::VERTEX, 32),
        (BufferUsage::UNIFORM, 64),
        (BufferUsage::STORAGE, 128),
        (BufferUsage::INDIRECT, 256),
        (BufferUsage::QUERY_RESOLVE, 512),
    ];
    let mut flags = 0;
    for (portable, wire) in pairs {
        if value.contains(portable) {
            flags |= wire;
        }
    }
    if value.contains(BufferUsage::BLAS_INPUT)
        || value.contains(BufferUsage::TLAS_INPUT)
        || value.contains(BufferUsage::ACCELERATION_STRUCTURE_SCRATCH)
    {
        return Err(unsupported("BufferUsage acceleration structure"));
    }
    Ok(flags)
}
pub(crate) fn texture_usage(value: TextureUsage) -> u32 {
    let mut flags = 0;
    if value.contains(TextureUsage::COPY_SRC) {
        flags |= 1;
    }
    if value.contains(TextureUsage::COPY_DST) {
        flags |= 2;
    }
    if value.contains(TextureUsage::SAMPLED) {
        flags |= 4;
    }
    if value.contains(TextureUsage::STORAGE) {
        flags |= 8;
    }
    if value.contains(TextureUsage::COLOR_ATTACHMENT)
        || value.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT)
    {
        flags |= 16;
    }
    flags
}

pub(crate) fn address_mode(value: AddressMode) -> TranslateResult<&'static str> {
    Ok(match value {
        AddressMode::ClampToEdge => "clamp-to-edge",
        AddressMode::Repeat => "repeat",
        AddressMode::MirrorRepeat => "mirror-repeat",
        AddressMode::ClampToBorder => return Err(unsupported("AddressMode::ClampToBorder")),
    })
}
pub(crate) fn filter_mode(value: FilterMode) -> &'static str {
    match value {
        FilterMode::Nearest => "nearest",
        FilterMode::Linear => "linear",
    }
}
pub(crate) fn compare_function(value: CompareFunction) -> TranslateResult<&'static str> {
    Ok(match value {
        CompareFunction::Never => "never",
        CompareFunction::Less => "less",
        CompareFunction::Equal => "equal",
        CompareFunction::LessEqual => "less-equal",
        CompareFunction::Greater => "greater",
        CompareFunction::NotEqual => "not-equal",
        CompareFunction::GreaterEqual => "greater-equal",
        CompareFunction::Always => "always",
    })
}

pub(crate) fn vertex_format(value: VertexFormat) -> TranslateResult<&'static str> {
    use VertexFormat::*;
    Ok(match value {
        Uint8 => "uint8",
        Uint8x2 => "uint8x2",
        Uint8x4 => "uint8x4",
        Sint8 => "sint8",
        Sint8x2 => "sint8x2",
        Sint8x4 => "sint8x4",
        Unorm8 => "unorm8",
        Unorm8x2 => "unorm8x2",
        Unorm8x4 => "unorm8x4",
        Unorm8x4Bgra => "unorm8x4-bgra",
        Snorm8 => "snorm8",
        Snorm8x2 => "snorm8x2",
        Snorm8x4 => "snorm8x4",
        Uint16 => "uint16",
        Uint16x2 => "uint16x2",
        Uint16x4 => "uint16x4",
        Sint16 => "sint16",
        Sint16x2 => "sint16x2",
        Sint16x4 => "sint16x4",
        Unorm16 => "unorm16",
        Unorm16x2 => "unorm16x2",
        Unorm16x4 => "unorm16x4",
        Snorm16 => "snorm16",
        Snorm16x2 => "snorm16x2",
        Snorm16x4 => "snorm16x4",
        Float16 => "float16",
        Float16x2 => "float16x2",
        Float16x4 => "float16x4",
        Float32 => "float32",
        Float32x2 => "float32x2",
        Float32x3 => "float32x3",
        Float32x4 => "float32x4",
        Uint32 => "uint32",
        Uint32x2 => "uint32x2",
        Uint32x3 => "uint32x3",
        Uint32x4 => "uint32x4",
        Sint32 => "sint32",
        Sint32x2 => "sint32x2",
        Sint32x3 => "sint32x3",
        Sint32x4 => "sint32x4",
        Unorm10_10_10_2 => "unorm10-10-10-2",
        Float64 | Float64x2 | Float64x3 | Float64x4 => return Err(unsupported("VertexFormat f64")),
    })
}
pub(crate) fn vertex_step_mode(value: VertexStepMode) -> TranslateResult<&'static str> {
    Ok(match value {
        VertexStepMode::Vertex => "vertex",
        VertexStepMode::Instance => "instance",
    })
}
pub(crate) fn primitive_topology(value: PrimitiveTopology) -> TranslateResult<&'static str> {
    Ok(match value {
        PrimitiveTopology::PointList => "point-list",
        PrimitiveTopology::LineList => "line-list",
        PrimitiveTopology::LineStrip => "line-strip",
        PrimitiveTopology::TriangleList => "triangle-list",
        PrimitiveTopology::TriangleStrip => "triangle-strip",
    })
}
pub(crate) fn front_face(value: FrontFace) -> TranslateResult<&'static str> {
    Ok(match value {
        FrontFace::Ccw => "ccw",
        FrontFace::Cw => "cw",
    })
}
pub(crate) fn cull_mode(value: CullMode) -> TranslateResult<Option<&'static str>> {
    Ok(match value {
        CullMode::None => None,
        CullMode::Front => Some("front"),
        CullMode::Back => Some("back"),
    })
}
pub(crate) fn index_format(value: IndexFormat) -> &'static str {
    match value {
        IndexFormat::Uint16 => "uint16",
        IndexFormat::Uint32 => "uint32",
    }
}
pub(crate) fn blend_factor(value: BlendFactor) -> TranslateResult<&'static str> {
    use BlendFactor::*;
    Ok(match value {
        Zero => "zero",
        One => "one",
        Src => "src",
        OneMinusSrc => "one-minus-src",
        SrcAlpha => "src-alpha",
        OneMinusSrcAlpha => "one-minus-src-alpha",
        Src1 => "src1",
        OneMinusSrc1 => "one-minus-src1",
        Src1Alpha => "src1-alpha",
        OneMinusSrc1Alpha => "one-minus-src1-alpha",
        Dst => "dst",
        OneMinusDst => "one-minus-dst",
        DstAlpha => "dst-alpha",
        OneMinusDstAlpha => "one-minus-dst-alpha",
        SrcAlphaSaturated => "src-alpha-saturated",
        Constant => "constant",
        OneMinusConstant => "one-minus-constant",
    })
}
pub(crate) fn blend_operation(value: BlendOperation) -> TranslateResult<&'static str> {
    Ok(match value {
        BlendOperation::Add => "add",
        BlendOperation::Subtract => "subtract",
        BlendOperation::ReverseSubtract => "reverse-subtract",
        BlendOperation::Min => "min",
        BlendOperation::Max => "max",
    })
}
pub(crate) fn color_write_mask(value: ColorWriteMask) -> u32 {
    let mut result = 0;
    if value.contains(ColorWriteMask::RED) {
        result |= 1;
    }
    if value.contains(ColorWriteMask::GREEN) {
        result |= 2;
    }
    if value.contains(ColorWriteMask::BLUE) {
        result |= 4;
    }
    if value.contains(ColorWriteMask::ALPHA) {
        result |= 8;
    }
    result
}
pub(crate) fn load_op<T>(value: LoadOp<T>) -> &'static str {
    match value {
        LoadOp::Load => "load",
        LoadOp::Clear(_) => "clear",
    }
}
pub(crate) fn store_op(value: StoreOp) -> &'static str {
    match value {
        StoreOp::Store => "store",
        StoreOp::Discard => "discard",
    }
}
pub(crate) fn query_type(value: QueryType) -> TranslateResult<&'static str> {
    match value {
        QueryType::Occlusion => Ok("occlusion"),
        QueryType::Timestamp => Ok("timestamp"),
        QueryType::PipelineStatistics(_) => Err(unsupported("QueryType::PipelineStatistics")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compressed_and_depth_formats_keep_their_exact_webgpu_spelling() {
        assert_eq!(
            texture_format(TextureFormat::Bc7RgbaUnormSrgb),
            Ok("bc7-rgba-unorm-srgb")
        );
        assert_eq!(
            texture_format(TextureFormat::Astc12x12Unorm),
            Ok("astc-12x12-unorm")
        );
        assert_eq!(
            texture_format(TextureFormat::Depth24PlusStencil8),
            Ok("depth24plus-stencil8")
        );
    }
    #[test]
    fn webgpu_absent_texture_formats_are_refused_not_substituted() {
        assert_eq!(
            texture_format(TextureFormat::Astc4x4Hdr),
            Err(Unsupported {
                what: "TextureFormat"
            })
        );
        assert_eq!(
            texture_format(TextureFormat::Nv12),
            Err(Unsupported {
                what: "TextureFormat"
            })
        );
    }
    #[test]
    fn usage_flags_use_webgpu_wire_bits_and_refuse_ray_tracing_bits() {
        assert_eq!(
            buffer_usage(BufferUsage::COPY_DST.union(BufferUsage::VERTEX)),
            Ok(8 | 32)
        );
        assert_eq!(
            buffer_usage(BufferUsage::BLAS_INPUT),
            Err(Unsupported {
                what: "BufferUsage acceleration structure"
            })
        );
        assert_eq!(
            texture_usage(TextureUsage::SAMPLED.union(TextureUsage::COLOR_ATTACHMENT)),
            4 | 16
        );
    }
    #[test]
    fn sampler_and_pipeline_edge_cases_are_fail_closed() {
        assert_eq!(
            address_mode(AddressMode::ClampToBorder),
            Err(Unsupported {
                what: "AddressMode::ClampToBorder"
            })
        );
        assert_eq!(
            vertex_format(VertexFormat::Float64x4),
            Err(Unsupported {
                what: "VertexFormat f64"
            })
        );
        assert_eq!(
            query_type(QueryType::PipelineStatistics(
                crate::api::query::PipelineStatistics::NONE
            )),
            Err(Unsupported {
                what: "QueryType::PipelineStatistics"
            })
        );
    }
    #[test]
    fn view_aspects_do_not_turn_an_invalid_mixture_into_all() {
        assert_eq!(texture_aspect(TextureAspects::DEPTH), Ok("depth-only"));
        assert_eq!(
            texture_aspect(TextureAspects::DEPTH.union(TextureAspects::COLOR)),
            Err(Unsupported {
                what: "TextureAspects"
            })
        );
        assert_eq!(
            color_write_mask(ColorWriteMask::RED.union(ColorWriteMask::ALPHA)),
            9
        );
    }
}
