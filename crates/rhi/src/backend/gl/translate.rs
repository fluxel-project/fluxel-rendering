//! Lossless translation from the frozen RHI vocabulary to GL-family values.
//!
//! This module is intentionally below `api` and above the native/browser
//! executors.  It is the only place where a public descriptor acquires a
//! GL-family spelling, which prevents WGL/EGL and WebGL2 from drifting into
//! subtly different acceptance rules.  Translation never creates a driver
//! object: an error here is therefore an `Unsupported` refusal before native
//! work has been accepted.

use crate::api::command::IndexFormat;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::{TextureFormat, format_aspects};
use crate::api::pipeline::PipelineInterface;
use crate::api::pipeline::vertex_input::VertexFormat;
use crate::api::resource::buffer::{BufferDescriptor, BufferUsage};
use crate::api::resource::sampler::{AddressMode, CompareFunction, FilterMode, SamplerDescriptor};
use crate::api::resource::texture::{TextureDescriptor, TextureDimension, TextureUsage};
use crate::api::resource::transfer::TextureUploadDescriptor;
use crate::api::resource::view::{TextureViewDescriptor, TextureViewDimension};
use crate::api::shader::vocabulary::IMPLEMENTED_ABI;
use crate::api::shader::{ShaderArtifact, ShaderCode, ShaderStage};

use super::api::*;

fn unsupported(operation: &'static str, detail: impl Into<String>) -> RhiError {
    RhiError::new(RhiErrorKind::Unsupported, detail).at(operation)
}

/// Maps a public format into the exact GL-family format key.
///
/// A format which the common GL vocabulary cannot represent is refused here,
/// rather than being approximated by a near neighbour.  Per-context format
/// facts still decide whether a represented format is actually available.
pub(crate) fn texture_format(format: TextureFormat) -> RhiResult<GlFormat> {
    use GlAstcBlock::*;
    use GlCompressedColorSpace::*;
    use TextureFormat::*;
    let value = match format {
        R8Unorm => GlFormat::R8Unorm,
        R8Snorm => GlFormat::R8Snorm,
        R8Uint => GlFormat::R8Uint,
        R8Sint => GlFormat::R8Sint,
        Rg8Unorm => GlFormat::Rg8Unorm,
        Rg8Snorm => GlFormat::Rg8Snorm,
        Rg8Uint => GlFormat::Rg8Uint,
        Rg8Sint => GlFormat::Rg8Sint,
        Rgba8Unorm => GlFormat::Rgba8Unorm,
        Rgba8UnormSrgb => GlFormat::Rgba8Srgb,
        Rgba8Snorm => GlFormat::Rgba8Snorm,
        Rgba8Uint => GlFormat::Rgba8Uint,
        Rgba8Sint => GlFormat::Rgba8Sint,
        Bgra8Unorm => GlFormat::Bgra8Unorm,
        Bgra8UnormSrgb => GlFormat::Bgra8Srgb,
        R16Uint => GlFormat::R16Uint,
        R16Sint => GlFormat::R16Sint,
        R16Float => GlFormat::R16Float,
        R16Unorm => GlFormat::R16Unorm,
        R16Snorm => GlFormat::R16Snorm,
        Rg16Uint => GlFormat::Rg16Uint,
        Rg16Sint => GlFormat::Rg16Sint,
        Rg16Float => GlFormat::Rg16Float,
        Rg16Unorm => GlFormat::Rg16Unorm,
        Rg16Snorm => GlFormat::Rg16Snorm,
        Rgba16Uint => GlFormat::Rgba16Uint,
        Rgba16Sint => GlFormat::Rgba16Sint,
        Rgba16Unorm => GlFormat::Rgba16Unorm,
        Rgba16Snorm => GlFormat::Rgba16Snorm,
        Rgba16Float => GlFormat::Rgba16Float,
        Rgba32Float => GlFormat::Rgba32Float,
        Rgb9e5Ufloat => GlFormat::Rgb9e5Ufloat,
        Rgb10a2Uint => GlFormat::Rgb10a2Uint,
        Rgb10a2Unorm => GlFormat::Rgb10a2Unorm,
        Rg11b10Ufloat => GlFormat::Rg11b10Ufloat,
        R32Uint => GlFormat::R32Uint,
        R32Sint => GlFormat::R32Sint,
        R32Float => GlFormat::R32Float,
        R64Uint => GlFormat::R64Uint,
        Rg32Uint => GlFormat::Rg32Uint,
        Rg32Sint => GlFormat::Rg32Sint,
        Rg32Float => GlFormat::Rg32Float,
        Rgba32Uint => GlFormat::Rgba32Uint,
        Rgba32Sint => GlFormat::Rgba32Sint,
        Depth16Unorm => GlFormat::Depth16Unorm,
        Depth24Plus => GlFormat::Depth24Unorm,
        Depth24PlusStencil8 => GlFormat::Depth24PlusStencil8,
        Depth32Float => GlFormat::Depth32Float,
        Depth32FloatStencil8 => GlFormat::Depth32FloatStencil8,
        Stencil8 => GlFormat::Stencil8,
        Bc1RgbaUnorm => GlFormat::Bc1RgbaUnorm,
        Bc1RgbaUnormSrgb => GlFormat::Bc1RgbaSrgb,
        Bc2RgbaUnorm => GlFormat::Bc2RgbaUnorm,
        Bc2RgbaUnormSrgb => GlFormat::Bc2RgbaSrgb,
        Bc3RgbaUnorm => GlFormat::Bc3RgbaUnorm,
        Bc3RgbaUnormSrgb => GlFormat::Bc3RgbaSrgb,
        Bc4RUnorm => GlFormat::Bc4RUnorm,
        Bc4RSnorm => GlFormat::Bc4RSnorm,
        Bc5RgUnorm => GlFormat::Bc5RgUnorm,
        Bc5RgSnorm => GlFormat::Bc5RgSnorm,
        Bc6hRgbUfloat => GlFormat::Bc6hRgbUfloat,
        Bc6hRgbFloat => GlFormat::Bc6hRgbSfloat,
        Bc7RgbaUnorm => GlFormat::Bc7RgbaUnorm,
        Bc7RgbaUnormSrgb => GlFormat::Bc7RgbaSrgb,
        Etc2Rgb8Unorm => GlFormat::Etc2Rgb8Unorm,
        Etc2Rgb8UnormSrgb => GlFormat::Etc2Rgb8Srgb,
        Etc2Rgb8A1Unorm => GlFormat::Etc2Rgb8A1Unorm,
        Etc2Rgb8A1UnormSrgb => GlFormat::Etc2Rgb8A1Srgb,
        Etc2Rgba8Unorm => GlFormat::Etc2Rgba8Unorm,
        Etc2Rgba8UnormSrgb => GlFormat::Etc2Rgba8Srgb,
        EacR11Unorm => GlFormat::EacR11Unorm,
        EacR11Snorm => GlFormat::EacR11Snorm,
        EacRg11Unorm => GlFormat::EacRg11Unorm,
        EacRg11Snorm => GlFormat::EacRg11Snorm,
        Astc4x4Unorm => GlFormat::Astc {
            block: B4x4,
            color_space: Linear,
        },
        Astc4x4UnormSrgb => GlFormat::Astc {
            block: B4x4,
            color_space: Srgb,
        },
        Astc4x4Hdr => GlFormat::Astc {
            block: B4x4,
            color_space: Hdr,
        },
        Astc5x4Unorm => GlFormat::Astc {
            block: B5x4,
            color_space: Linear,
        },
        Astc5x4UnormSrgb => GlFormat::Astc {
            block: B5x4,
            color_space: Srgb,
        },
        Astc5x4Hdr => GlFormat::Astc {
            block: B5x4,
            color_space: Hdr,
        },
        Astc5x5Unorm => GlFormat::Astc {
            block: B5x5,
            color_space: Linear,
        },
        Astc5x5UnormSrgb => GlFormat::Astc {
            block: B5x5,
            color_space: Srgb,
        },
        Astc5x5Hdr => GlFormat::Astc {
            block: B5x5,
            color_space: Hdr,
        },
        Astc6x5Unorm => GlFormat::Astc {
            block: B6x5,
            color_space: Linear,
        },
        Astc6x5UnormSrgb => GlFormat::Astc {
            block: B6x5,
            color_space: Srgb,
        },
        Astc6x5Hdr => GlFormat::Astc {
            block: B6x5,
            color_space: Hdr,
        },
        Astc6x6Unorm => GlFormat::Astc {
            block: B6x6,
            color_space: Linear,
        },
        Astc6x6UnormSrgb => GlFormat::Astc {
            block: B6x6,
            color_space: Srgb,
        },
        Astc6x6Hdr => GlFormat::Astc {
            block: B6x6,
            color_space: Hdr,
        },
        Astc8x5Unorm => GlFormat::Astc {
            block: B8x5,
            color_space: Linear,
        },
        Astc8x5UnormSrgb => GlFormat::Astc {
            block: B8x5,
            color_space: Srgb,
        },
        Astc8x5Hdr => GlFormat::Astc {
            block: B8x5,
            color_space: Hdr,
        },
        Astc8x6Unorm => GlFormat::Astc {
            block: B8x6,
            color_space: Linear,
        },
        Astc8x6UnormSrgb => GlFormat::Astc {
            block: B8x6,
            color_space: Srgb,
        },
        Astc8x6Hdr => GlFormat::Astc {
            block: B8x6,
            color_space: Hdr,
        },
        Astc8x8Unorm => GlFormat::Astc {
            block: B8x8,
            color_space: Linear,
        },
        Astc8x8UnormSrgb => GlFormat::Astc {
            block: B8x8,
            color_space: Srgb,
        },
        Astc8x8Hdr => GlFormat::Astc {
            block: B8x8,
            color_space: Hdr,
        },
        Astc10x5Unorm => GlFormat::Astc {
            block: B10x5,
            color_space: Linear,
        },
        Astc10x5UnormSrgb => GlFormat::Astc {
            block: B10x5,
            color_space: Srgb,
        },
        Astc10x5Hdr => GlFormat::Astc {
            block: B10x5,
            color_space: Hdr,
        },
        Astc10x6Unorm => GlFormat::Astc {
            block: B10x6,
            color_space: Linear,
        },
        Astc10x6UnormSrgb => GlFormat::Astc {
            block: B10x6,
            color_space: Srgb,
        },
        Astc10x6Hdr => GlFormat::Astc {
            block: B10x6,
            color_space: Hdr,
        },
        Astc10x8Unorm => GlFormat::Astc {
            block: B10x8,
            color_space: Linear,
        },
        Astc10x8UnormSrgb => GlFormat::Astc {
            block: B10x8,
            color_space: Srgb,
        },
        Astc10x8Hdr => GlFormat::Astc {
            block: B10x8,
            color_space: Hdr,
        },
        Astc10x10Unorm => GlFormat::Astc {
            block: B10x10,
            color_space: Linear,
        },
        Astc10x10UnormSrgb => GlFormat::Astc {
            block: B10x10,
            color_space: Srgb,
        },
        Astc10x10Hdr => GlFormat::Astc {
            block: B10x10,
            color_space: Hdr,
        },
        Astc12x10Unorm => GlFormat::Astc {
            block: B12x10,
            color_space: Linear,
        },
        Astc12x10UnormSrgb => GlFormat::Astc {
            block: B12x10,
            color_space: Srgb,
        },
        Astc12x10Hdr => GlFormat::Astc {
            block: B12x10,
            color_space: Hdr,
        },
        Astc12x12Unorm => GlFormat::Astc {
            block: B12x12,
            color_space: Linear,
        },
        Astc12x12UnormSrgb => GlFormat::Astc {
            block: B12x12,
            color_space: Srgb,
        },
        Astc12x12Hdr => GlFormat::Astc {
            block: B12x12,
            color_space: Hdr,
        },
        _ => {
            return Err(unsupported(
                "GL::texture_format",
                format!("GL-family lowering has no exact representation for {format:?}"),
            ));
        }
    };
    Ok(value)
}

/// Whether the portable format names encoded texel blocks rather than ordinary
/// one-texel elements. This is deliberately derived from the central format
/// table, so adding a compressed `TextureFormat` cannot bypass upload review.
pub(crate) fn is_compressed_texture_format(format: TextureFormat) -> bool {
    crate::api::format::block_extent(format) != (1, 1)
}

/// Validates the deliberately narrow GL-family compressed upload word.
///
/// `compressedTexImage2D` defines an entire mip from encoded blocks. It is not
/// `texSubImage2D`: accepting an arbitrary region, padded host rows, or an
/// array layer would either be silently ignored by the executor or require a
/// different native command contract. Keep this check in shared Phase-A
/// lowering so browser and native never drift.
pub(crate) fn validate_compressed_texture_upload(
    upload: &TextureUploadDescriptor,
    operation: &'static str,
) -> RhiResult<()> {
    use crate::api::resource::{TextureAspect, TextureDimension};
    if !is_compressed_texture_format(upload.dst.descriptor().format) {
        return Ok(());
    }
    let descriptor = upload.dst.descriptor();
    if descriptor.dimension != TextureDimension::D2
        || upload.subresource.aspect != TextureAspect::Color
        || upload.subresource.base_layer != 0
        || upload.subresource.layer_count != 1
        || upload.origin.x != 0
        || upload.origin.y != 0
        || upload.origin.z != 0
        || upload.extent.depth != 1
    {
        return Err(unsupported(
            operation,
            "compressed GL uploads must define one complete 2D color mip",
        ));
    }
    let expected_extent = crate::api::resource::texture::mip_extent(
        descriptor.extent,
        descriptor.dimension,
        upload.subresource.mip_level,
    );
    if upload.extent != expected_extent {
        return Err(unsupported(
            operation,
            "compressed GL uploads cannot define a sub-rectangle or partial mip",
        ));
    }
    let (block_width, block_height) = crate::api::format::block_extent(descriptor.format);
    let block_bytes =
        crate::api::format::logical_bytes_per_block(descriptor.format).ok_or_else(|| {
            unsupported(
                operation,
                "compressed format has no exact encoded block size",
            )
        })?;
    let blocks_wide = upload.extent.width.div_ceil(block_width);
    let blocks_high = upload.extent.height.div_ceil(block_height);
    let row_bytes = blocks_wide.checked_mul(block_bytes).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::OutOfMemory,
            "compressed upload row size overflows",
        )
        .at(operation)
    })?;
    let exact = u64::from(row_bytes)
        .checked_mul(u64::from(blocks_high))
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::OutOfMemory,
                "compressed upload size overflows",
            )
            .at(operation)
        })?;
    if upload.source_layout.bytes_per_row != row_bytes
        || upload.source_layout.rows_per_image != blocks_high
        || u64::try_from(upload.bytes.len()).ok() != Some(exact)
    {
        return Err(unsupported(
            operation,
            "compressed GL uploads require a tightly packed complete encoded mip",
        ));
    }
    Ok(())
}

pub(crate) fn buffer_descriptor(desc: &BufferDescriptor) -> RhiResult<GlBufferDesc> {
    let mut usage = GlBufferUsage::EMPTY;
    let map = [
        (BufferUsage::COPY_SRC, GlBufferUsage::COPY_SOURCE),
        (BufferUsage::COPY_DST, GlBufferUsage::COPY_DESTINATION),
        (BufferUsage::VERTEX, GlBufferUsage::VERTEX),
        (BufferUsage::INDEX, GlBufferUsage::INDEX),
        (BufferUsage::UNIFORM, GlBufferUsage::UNIFORM),
        (BufferUsage::STORAGE, GlBufferUsage::STORAGE),
        (BufferUsage::INDIRECT, GlBufferUsage::INDIRECT),
        (BufferUsage::MAP_READ, GlBufferUsage::MAP_READ),
        (BufferUsage::MAP_WRITE, GlBufferUsage::MAP_WRITE),
        (BufferUsage::QUERY_RESOLVE, GlBufferUsage::COPY_DESTINATION),
    ];
    for (portable, gl) in map {
        if desc.usage.contains(portable) {
            usage = usage | gl;
        }
    }
    if desc.usage.contains(BufferUsage::BLAS_INPUT)
        || desc.usage.contains(BufferUsage::TLAS_INPUT)
        || desc
            .usage
            .contains(BufferUsage::ACCELERATION_STRUCTURE_SCRATCH)
    {
        return Err(unsupported(
            "GL::create_buffer",
            "acceleration-structure buffer usage is not representable by GL-family lowering",
        ));
    }
    Ok(GlBufferDesc {
        size: desc.size,
        usage,
    })
}

pub(crate) fn texture_descriptor(desc: &TextureDescriptor) -> RhiResult<GlTextureDesc> {
    let dimension = match desc.dimension {
        TextureDimension::D1 => GlTextureDimension::D1,
        TextureDimension::D2 if desc.array_layers == 1 => GlTextureDimension::D2,
        TextureDimension::D2 => GlTextureDimension::D2Array,
        TextureDimension::D3 => GlTextureDimension::D3,
    };
    let mut usage = GlTextureUsage::EMPTY;
    for (portable, gl) in [
        (TextureUsage::COPY_SRC, GlTextureUsage::COPY_SOURCE),
        (TextureUsage::COPY_DST, GlTextureUsage::COPY_DESTINATION),
        (TextureUsage::SAMPLED, GlTextureUsage::SAMPLED),
        (TextureUsage::STORAGE, GlTextureUsage::STORAGE_BINDING),
    ] {
        if desc.usage.contains(portable) {
            usage = usage | gl;
        }
    }
    if desc.usage.contains(TextureUsage::COLOR_ATTACHMENT)
        || desc.usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT)
    {
        usage = usage | GlTextureUsage::RENDER_ATTACHMENT;
    }
    Ok(GlTextureDesc {
        dimension,
        extent: GlExtent3d {
            width: desc.extent.width,
            height: desc.extent.height,
            depth_or_layers: if matches!(desc.dimension, TextureDimension::D3) {
                desc.extent.depth
            } else {
                desc.array_layers
            },
        },
        mip_level_count: desc.mip_levels,
        sample_count: desc.sample_count,
        format: texture_format(desc.format)?,
        usage,
    })
}

pub(crate) fn view_dimension(value: TextureViewDimension) -> RhiResult<GlTextureDimension> {
    match value {
        TextureViewDimension::D1 => Ok(GlTextureDimension::D1),
        TextureViewDimension::D2 => Ok(GlTextureDimension::D2),
        TextureViewDimension::D2Array => Ok(GlTextureDimension::D2Array),
        TextureViewDimension::Cube => Ok(GlTextureDimension::Cube),
        TextureViewDimension::CubeArray => Err(unsupported(
            "GL::create_texture_view",
            "GL typed view vocabulary has no distinct cube-array target",
        )),
        TextureViewDimension::D3 => Ok(GlTextureDimension::D3),
    }
}

pub(crate) fn view_format(
    desc: &TextureViewDescriptor,
    base: TextureFormat,
) -> RhiResult<GlFormat> {
    texture_format(desc.format.unwrap_or(base))
}

/// Checks the only TextureView shape that a GL-family implementation without
/// `glTextureView` can represent without changing shared texture state.
///
/// In particular, this is intentionally *not* implemented with
/// `TEXTURE_BASE_LEVEL`/`TEXTURE_MAX_LEVEL`: those parameters belong to the
/// texture object, so two portable views of one texture would alias each
/// other's state.  WebGL2 and GLES 3.x have no native view object at all and
/// therefore use this boundary. Native GL must use a separately allocated
/// view once its `GL_ARB_texture_view` route is installed; until then it uses
/// the same conservative boundary rather than silently dropping a range.
pub(crate) fn whole_compatible_virtual_view(
    view: &TextureViewDescriptor,
    base_format: TextureFormat,
    base: GlTextureDesc,
) -> RhiResult<GlTextureDimension> {
    let dimension = view_dimension(view.dimension)?;
    let whole_aspects = format_aspects(base_format);
    if dimension != base.dimension
        || view.format.unwrap_or(base_format) != base_format
        || view.aspects != whole_aspects
        || view.base_mip != 0
        || view.mip_count != base.mip_level_count
        || view.base_layer != 0
        || view.layer_count != base.extent.depth_or_layers
    {
        return Err(unsupported(
            "GL::create_texture_view",
            "this GL profile has no independent texture-view object; only the base texture's whole, same-format, same-dimension view is representable",
        ));
    }
    Ok(dimension)
}

pub(crate) fn sampler_descriptor(desc: &SamplerDescriptor) -> RhiResult<GlSamplerDesc> {
    fn address(value: AddressMode) -> RhiResult<GlAddressMode> {
        match value {
            AddressMode::ClampToEdge => Ok(GlAddressMode::ClampToEdge),
            AddressMode::Repeat => Ok(GlAddressMode::Repeat),
            AddressMode::MirrorRepeat => Ok(GlAddressMode::MirroredRepeat),
            AddressMode::ClampToBorder => Err(unsupported(
                "GL::create_sampler",
                "clamp-to-border requires profile-specific GL extension admission",
            )),
        }
    }
    fn filter(value: FilterMode) -> GlFilterMode {
        match value {
            FilterMode::Nearest => GlFilterMode::Nearest,
            FilterMode::Linear => GlFilterMode::Linear,
        }
    }
    fn compare(value: CompareFunction) -> GlCompareFunction {
        match value {
            CompareFunction::Never => GlCompareFunction::Never,
            CompareFunction::Less => GlCompareFunction::Less,
            CompareFunction::Equal => GlCompareFunction::Equal,
            CompareFunction::LessEqual => GlCompareFunction::LessEqual,
            CompareFunction::Greater => GlCompareFunction::Greater,
            CompareFunction::NotEqual => GlCompareFunction::NotEqual,
            CompareFunction::GreaterEqual => GlCompareFunction::GreaterEqual,
            CompareFunction::Always => GlCompareFunction::Always,
        }
    }
    Ok(GlSamplerDesc {
        address_mode_u: address(desc.address_u)?,
        address_mode_v: address(desc.address_v)?,
        address_mode_w: address(desc.address_w)?,
        mag_filter: filter(desc.mag_filter),
        min_filter: filter(desc.min_filter),
        mipmap_filter: match desc.mip_filter {
            FilterMode::Nearest => GlMipmapFilterMode::Nearest,
            FilterMode::Linear => GlMipmapFilterMode::Linear,
        },
        lod_min_bits: desc.lod_min.to_bits(),
        lod_max_bits: desc.lod_max.to_bits(),
        compare: desc.compare.map(compare),
        max_anisotropy_bits: (desc.max_anisotropy > 1)
            .then(|| f32::from(desc.max_anisotropy).to_bits()),
    })
}

pub(crate) fn shader_stage(stage: ShaderStage) -> RhiResult<GlShaderStage> {
    match stage {
        ShaderStage::Vertex => Ok(GlShaderStage::Vertex),
        ShaderStage::Fragment => Ok(GlShaderStage::Fragment),
        ShaderStage::Compute => Ok(GlShaderStage::Compute),
        _ => Err(unsupported(
            "GL::create_shader",
            "mesh and ray shader stages are not representable by GL-family lowering",
        )),
    }
}

/// Selects only an already-GLSL artifact.  Source transpilation is deliberately
/// not hidden in a backend: callers must provide a capability-admitted artifact.
pub(crate) fn shader_source(artifact: &ShaderArtifact) -> RhiResult<GlShaderSource> {
    let stage = shader_stage(artifact.stage)?;
    let (dialect, text) = match &artifact.code {
        ShaderCode::Glsl {
            version, source, ..
        } => (
            GlShaderDialect::Desktop { version: *version },
            source.to_string(),
        ),
        ShaderCode::GlslEs { version, source } => (
            GlShaderDialect::Embedded { version: *version },
            source.to_string(),
        ),
        _ => {
            return Err(unsupported(
                "GL::create_shader",
                "GL-family lowering requires GLSL or GLSL ES source artifact",
            ));
        }
    };
    Ok(GlShaderSource {
        stage,
        dialect,
        entry_point: artifact.entry_point.clone(),
        source_hash: artifact.content_hash,
        text,
        debug_name: None,
    })
}

/// Lowers the portable resource portion of one pipeline interface to the GL
/// program-layout vocabulary.
///
/// `GlLogicalBinding::name` is deliberately a *native reflection name*, not a
/// public binding identity.  The GL providers use it to find uniform blocks and
/// uniforms after linking; substituting a guessed spelling would make a valid
/// portable `(group, slot)` silently bind whichever GLSL declaration happened to
/// have that spelling.  ABI 1.0 does not define such a spelling, and
/// `ShaderArtifact` carries no backend-private reflection-name table.  Therefore
/// this lowering is intentionally fail-closed for every resource-bearing
/// pipeline until a future GL ABI revision supplies that table.
///
/// Keeping this verdict here, before a program descriptor is handed to either
/// provider, is important: an empty `GlPipelineLayout` is valid only when the
/// artifacts genuinely declare no resources.  It must never be used as a
/// placeholder for bindings that the submission path later claims it can bind.
pub(crate) fn pipeline_layout_from_artifacts(
    interface: &PipelineInterface,
    artifacts: &[&ShaderArtifact],
) -> RhiResult<GlPipelineLayout> {
    for artifact in artifacts {
        if artifact.abi_version != IMPLEMENTED_ABI {
            return Err(unsupported(
                "GL::pipeline_layout_from_artifacts",
                format!(
                    "GL pipeline layout requires ABI {}.{}, but shader artifact uses ABI {}.{}",
                    IMPLEMENTED_ABI.major,
                    IMPLEMENTED_ABI.minor,
                    artifact.abi_version.major,
                    artifact.abi_version.minor,
                ),
            ));
        }
        for resource in artifact.interface.resources() {
            // Read the interface as part of the lowering boundary rather than
            // treating an artifact's logical location as a GL binding point.
            // The public pipeline validator has already checked that this
            // declaration is present and compatible; this check protects direct
            // shared-lowering callers too and makes the refusal diagnostic point
            // at the missing piece of the ABI rather than at GL reflection.
            let declared = interface
                .group(resource.group)
                .and_then(|group| group.slot(resource.slot));
            if declared.is_none() {
                return Err(unsupported(
                    "GL::pipeline_layout_from_artifacts",
                    format!(
                        "shader resource group {} slot {} has no pipeline-interface declaration",
                        resource.group.get(),
                        resource.slot.get(),
                    ),
                ));
            }
            return Err(unsupported(
                "GL::pipeline_layout_from_artifacts",
                format!(
                    "GL ABI {}.{} has no reflection-name mapping for logical group {} slot {}; \
                     ShaderArtifact records logical resources but no GLSL uniform/block name",
                    IMPLEMENTED_ABI.major,
                    IMPLEMENTED_ABI.minor,
                    resource.group.get(),
                    resource.slot.get(),
                ),
            ));
        }
    }
    Ok(GlPipelineLayout {
        bindings: Vec::new(),
    })
}

pub(crate) fn vertex_format(value: VertexFormat) -> RhiResult<GlVertexFormat> {
    use VertexFormat::*;
    let result = match value {
        Uint8x2 => GlVertexFormat::Uint8x2,
        Uint8x4 => GlVertexFormat::Uint8x4,
        Sint8x2 => GlVertexFormat::Sint8x2,
        Sint8x4 => GlVertexFormat::Sint8x4,
        Unorm8x2 => GlVertexFormat::Unorm8x2,
        Unorm8x4 | Unorm8x4Bgra => GlVertexFormat::Unorm8x4,
        Snorm8x2 => GlVertexFormat::Snorm8x2,
        Snorm8x4 => GlVertexFormat::Snorm8x4,
        Uint16x2 => GlVertexFormat::Uint16x2,
        Uint16x4 => GlVertexFormat::Uint16x4,
        Sint16x2 => GlVertexFormat::Sint16x2,
        Sint16x4 => GlVertexFormat::Sint16x4,
        Unorm16x2 => GlVertexFormat::Unorm16x2,
        Unorm16x4 => GlVertexFormat::Unorm16x4,
        Snorm16x2 => GlVertexFormat::Snorm16x2,
        Snorm16x4 => GlVertexFormat::Snorm16x4,
        Float16x2 => GlVertexFormat::Float16x2,
        Float16x4 => GlVertexFormat::Float16x4,
        Float32 => GlVertexFormat::Float32,
        Float32x2 => GlVertexFormat::Float32x2,
        Float32x3 => GlVertexFormat::Float32x3,
        Float32x4 => GlVertexFormat::Float32x4,
        Uint32 => GlVertexFormat::Uint32,
        Uint32x2 => GlVertexFormat::Uint32x2,
        Uint32x3 => GlVertexFormat::Uint32x3,
        Uint32x4 => GlVertexFormat::Uint32x4,
        Sint32 => GlVertexFormat::Sint32,
        Sint32x2 => GlVertexFormat::Sint32x2,
        Sint32x3 => GlVertexFormat::Sint32x3,
        Sint32x4 => GlVertexFormat::Sint32x4,
        _ => {
            return Err(unsupported(
                "GL::vertex_format",
                format!(
                    "GL-family typed vertex vocabulary has no exact representation for {value:?}"
                ),
            ));
        }
    };
    Ok(result)
}

/// Converts the index spelling used by recorded draw commands.  Kept beside
/// vertex conversion so an executor never has to reinterpret public command
/// values while a GL VAO is being assembled.
pub(crate) fn index_format(value: IndexFormat) -> GlIndexFormat {
    match value {
        IndexFormat::Uint16 => GlIndexFormat::Uint16,
        IndexFormat::Uint32 => GlIndexFormat::Uint32,
    }
}

/// Classifies a captured command before an executor resolves its object IDs.
///
/// This is deliberately a *preflight* classification, not a lowering: object
/// lookup, pass-state validation, and the exact copy region remain execution
/// responsibilities.  It makes unsupported mesh/ray commands transactional --
/// an executor can reject a complete submission before issuing any GL work.
pub(crate) fn preflight_portable_command(
    command: &crate::api::tooling::PortableCommand,
) -> RhiResult<()> {
    use crate::api::tooling::PortableCommand;
    match command {
        PortableCommand::SetMeshPipeline(_)
        | PortableCommand::DispatchMesh { .. }
        | PortableCommand::DispatchMeshIndirect { .. }
        | PortableCommand::SetRayTracingPipeline(_)
        | PortableCommand::BeginRayTracing { .. }
        | PortableCommand::EndRayTracing
        | PortableCommand::TraceRays { .. }
        | PortableCommand::BuildAccelerationStructure { .. }
        | PortableCommand::CopyAccelerationStructure { .. }
        | PortableCommand::WriteAccelerationStructureCompactedSize { .. } => Err(unsupported(
            "GL::submit",
            "mesh and ray-tracing command is not representable by GL-family lowering",
        )),
        // Everything else has a typed GL vocabulary. Context capability and
        // exact object/route validation are intentionally deferred until the
        // owner-context executor has its discovery snapshot and object table.
        _ => Ok(()),
    }
}

/// Equivalent transactional gate for the actual recorder payload consumed by
/// `GlExecutionDriver::submit`.  Capture tooling uses `PortableCommand`, while
/// submit owns `RecordedCommand`; keeping both gates here makes the distinction
/// explicit and prevents one path from accidentally accepting mesh/ray work.
pub(crate) fn preflight_recorded_command(
    command: &crate::api::command::record::RecordedCommand,
) -> RhiResult<()> {
    use crate::api::command::record::RecordedPayload;
    match &command.payload {
        RecordedPayload::MeshDispatch(_)
        | RecordedPayload::MeshIndirect(_)
        | RecordedPayload::RayTracingBegin(_)
        | RecordedPayload::RayTracingDispatch(_)
        | RecordedPayload::RayTracingEnd
        | RecordedPayload::AccelerationStructure(_) => Err(unsupported(
            "GL::submit",
            "mesh, ray-tracing, and acceleration-structure recording is not representable by GL-family lowering",
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::resource::subresource::TextureAspects;
    #[test]
    fn compressed_formats_keep_exact_family_and_srgb() {
        assert_eq!(
            texture_format(TextureFormat::Bc7RgbaUnormSrgb).unwrap(),
            GlFormat::Bc7RgbaSrgb
        );
        assert_eq!(
            texture_format(TextureFormat::Astc10x6UnormSrgb).unwrap(),
            GlFormat::Astc {
                block: GlAstcBlock::B10x6,
                color_space: GlCompressedColorSpace::Srgb
            }
        );
        assert_eq!(
            texture_format(TextureFormat::Etc2Rgba8Unorm).unwrap(),
            GlFormat::Etc2Rgba8Unorm
        );
        assert_eq!(
            texture_format(TextureFormat::Astc6x6Hdr).unwrap(),
            GlFormat::Astc {
                block: GlAstcBlock::B6x6,
                color_space: GlCompressedColorSpace::Hdr
            }
        );
    }
    #[test]
    fn ordinary_gl_sized_formats_are_not_rejected_by_a_stale_typed_enum() {
        for format in [
            TextureFormat::R8Unorm,
            TextureFormat::Rg16Float,
            TextureFormat::Rgba32Sint,
            TextureFormat::Rgb10a2Unorm,
            TextureFormat::Depth32FloatStencil8,
        ] {
            assert!(texture_format(format).is_ok(), "{format:?}");
        }
        assert_eq!(
            texture_format(TextureFormat::Nv12).unwrap_err().kind(),
            RhiErrorKind::Unsupported
        );
    }
    #[test]
    fn array_texture_is_not_mistaken_for_plain_2d() {
        let desc =
            TextureDescriptor::new_2d(4, 4, TextureFormat::Rgba8Unorm, TextureUsage::SAMPLED)
                .with_array_layers(2);
        assert_eq!(
            texture_descriptor(&desc).unwrap().dimension,
            GlTextureDimension::D2Array
        );
    }
    #[test]
    fn acceleration_structure_usage_is_not_silently_dropped() {
        let desc = BufferDescriptor::new(64, BufferUsage::BLAS_INPUT);
        assert_eq!(
            buffer_descriptor(&desc).unwrap_err().kind(),
            RhiErrorKind::Unsupported
        );
    }

    fn virtual_view_base() -> GlTextureDesc {
        GlTextureDesc {
            dimension: GlTextureDimension::D2,
            extent: GlExtent3d {
                width: 8,
                height: 8,
                depth_or_layers: 1,
            },
            mip_level_count: 3,
            sample_count: 1,
            format: texture_format(TextureFormat::Rgba8Unorm).unwrap(),
            usage: GlTextureUsage::SAMPLED,
        }
    }

    #[test]
    fn virtual_view_accepts_only_the_exact_whole_base_shape() {
        let view =
            TextureViewDescriptor::new(TextureViewDimension::D2, TextureAspects::COLOR, 0, 3, 0, 1);
        assert_eq!(
            whole_compatible_virtual_view(&view, TextureFormat::Rgba8Unorm, virtual_view_base())
                .unwrap(),
            GlTextureDimension::D2
        );
    }

    #[test]
    fn virtual_view_refuses_partial_mips_reinterpretation_and_dimension_changes() {
        let partial =
            TextureViewDescriptor::new(TextureViewDimension::D2, TextureAspects::COLOR, 1, 1, 0, 1);
        assert_eq!(
            whole_compatible_virtual_view(&partial, TextureFormat::Rgba8Unorm, virtual_view_base())
                .unwrap_err()
                .kind(),
            RhiErrorKind::Unsupported
        );
        let reinterpreted =
            TextureViewDescriptor::new(TextureViewDimension::D2, TextureAspects::COLOR, 0, 3, 0, 1)
                .with_format(TextureFormat::Rgba8UnormSrgb);
        assert_eq!(
            whole_compatible_virtual_view(
                &reinterpreted,
                TextureFormat::Rgba8Unorm,
                virtual_view_base()
            )
            .unwrap_err()
            .kind(),
            RhiErrorKind::Unsupported
        );
    }
}
