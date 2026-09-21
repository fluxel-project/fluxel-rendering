//! WebGL2 exact format, pixel-transfer, and command-constant mappings.
//!
//! Every mapping here is backed by the WebGL 2.0 specification tables or by a
//! registry constant that the `web-sys` bindings do not name. Constants that
//! the bindings already expose are used through those bindings instead.

use web_sys::WebGl2RenderingContext as Gl;

use super::super::api::GlFormatTableError;

use super::super::api::{
    GlAddressMode, GlAstcBlock as B, GlCompareFunction, GlCompressedColorSpace as C,
    GlExtensionSet, GlFilterMode, GlFormat, GlFormatCapabilities, GlFormatEvidence,
    GlFormatResourceKind, GlFormatTable, GlKnownExtension, GlMipmapFilterMode, GlPixelFormat,
    GlSamplerDesc,
};

/// `GL_SAMPLES_PASSED`. The web-sys bindings omit this WebGL2 constant, so it
/// is spelled from the ES 3.0 registry like the other unnamed constants below.
pub(super) const SAMPLES_PASSED: u32 = 0x8914;
/// `GL_TEXTURE_COMPARE_MODE` value `COMPARE_REF_TO_TEXTURE` is already bound;
/// only the mode pname itself is unnamed in the bindings.
pub(super) const TEXTURE_COMPARE_MODE: u32 = 0x884C;
/// `GL_TEXTURE_MAX_ANISOTROPY_EXT` from `EXT_texture_filter_anisotropic`.
pub(super) const TEXTURE_MAX_ANISOTROPY_EXT: u32 = 0x84FE;

/// The WebGL2 internal format a discovered `GlFormat` allocates as.
pub(super) fn internal_format(format: GlFormat) -> Option<u32> {
    match format {
        GlFormat::Rgba8Unorm => Some(Gl::RGBA8),
        GlFormat::Rgba8Srgb => Some(Gl::SRGB8_ALPHA8),
        GlFormat::Depth32Float => Some(Gl::DEPTH_COMPONENT32F),
        GlFormat::Bc1RgbUnorm => Some(0x83F0),
        GlFormat::Bc1RgbaUnorm => Some(0x83F1),
        GlFormat::Bc2RgbaUnorm => Some(0x83F2),
        GlFormat::Bc3RgbaUnorm => Some(0x83F3),
        GlFormat::Bc1RgbSrgb => Some(0x8C4C),
        GlFormat::Bc1RgbaSrgb => Some(0x8C4D),
        GlFormat::Bc2RgbaSrgb => Some(0x8C4E),
        GlFormat::Bc3RgbaSrgb => Some(0x8C4F),
        GlFormat::Bc4RUnorm => Some(0x8DBB),
        GlFormat::Bc4RSnorm => Some(0x8DBC),
        GlFormat::Bc5RgUnorm => Some(0x8DBD),
        GlFormat::Bc5RgSnorm => Some(0x8DBE),
        GlFormat::Bc6hRgbUfloat => Some(0x8E8F),
        GlFormat::Bc6hRgbSfloat => Some(0x8E8E),
        GlFormat::Bc7RgbaUnorm => Some(0x8E8C),
        GlFormat::Bc7RgbaSrgb => Some(0x8E8D),
        GlFormat::Etc2Rgb8Unorm => Some(0x9274),
        GlFormat::Etc2Rgb8Srgb => Some(0x9275),
        GlFormat::Etc2Rgb8A1Unorm => Some(0x9276),
        GlFormat::Etc2Rgb8A1Srgb => Some(0x9277),
        GlFormat::Etc2Rgba8Unorm => Some(0x9278),
        GlFormat::Etc2Rgba8Srgb => Some(0x9279),
        GlFormat::EacR11Unorm => Some(0x9270),
        GlFormat::EacR11Snorm => Some(0x9271),
        GlFormat::EacRg11Unorm => Some(0x9272),
        GlFormat::EacRg11Snorm => Some(0x9273),
        GlFormat::Astc { block, color_space } => {
            let index = match block {
                B::B4x4 => 0,
                B::B5x4 => 1,
                B::B5x5 => 2,
                B::B6x5 => 3,
                B::B6x6 => 4,
                B::B8x5 => 5,
                B::B8x6 => 6,
                B::B8x8 => 7,
                B::B10x5 => 8,
                B::B10x6 => 9,
                B::B10x8 => 10,
                B::B10x10 => 11,
                B::B12x10 => 12,
                B::B12x12 => 13,
            };
            Some(match color_space {
                // `KHR_texture_compression_astc_hdr` deliberately reuses the
                // linear ASTC internal-format token.  The decode mode is not
                // encoded in the GLenum: discovery/capability admission must
                // prove the HDR extension before this `Hdr` typed format is
                // published or allocated.  Mapping it to the sRGB token would
                // silently change the texture's transfer function.
                C::Linear | C::Hdr => 0x93B0 + index,
                C::Srgb => 0x93D0 + index,
            })
        }
        _ => None,
    }
}

/// The pixel-transfer encoding a texture format accepts for CPU uploads.
///
/// ES 3.0 accepts exactly one `(format, type)` pair per sized internal format,
/// so unsupported pairs return `None` instead of a driver-dependent guess.
pub(super) fn upload_encoding(format: GlFormat) -> Option<(u32, u32)> {
    match format {
        GlFormat::Rgba8Unorm | GlFormat::Rgba8Srgb => Some((Gl::RGBA, Gl::UNSIGNED_BYTE)),
        _ => None,
    }
}

/// The readback encoding this backend returns for a discovered texture format.
///
/// `RGBA`/`UNSIGNED_BYTE` is the only combination every WebGL2 implementation
/// must accept; reading an sRGB attachment through it yields the linearized
/// values mandated by the specification.
pub(super) fn readback_encoding(format: GlFormat) -> Option<(u32, u32)> {
    match format {
        GlFormat::Rgba8Unorm | GlFormat::Rgba8Srgb => Some((Gl::RGBA, Gl::UNSIGNED_BYTE)),
        _ => None,
    }
}

/// The client pixel format a transfer must use, if this backend accepts it.
pub(super) fn transfer_format(format: GlPixelFormat) -> Option<()> {
    match format {
        GlPixelFormat::Rgba8 => Some(()),
        // WebGL2 has no BGRA upload, no RGB/RED encoding for the discovered
        // RGBA8 textures, and no ES 3.0 core readback encoding for depth
        // planes; every one of those rejects before touching pixel state.
        _ => None,
    }
}

/// The immutable-format internal constant backing a framebuffer attachment.
///
/// Returns the attachment point for depth/stencil views together with the
/// attach call's attachment name.
pub(super) const fn depth_attachment_point(format: GlFormat) -> Option<u32> {
    match format {
        GlFormat::Depth16Unorm | GlFormat::Depth32Float => Some(Gl::DEPTH_ATTACHMENT),
        GlFormat::Depth24PlusStencil8 => Some(Gl::DEPTH_STENCIL_ATTACHMENT),
        _ => None,
    }
}

/// Returns whether a depth/stencil format carries a stencil plane.
pub(super) const fn has_stencil_plane(format: GlFormat) -> bool {
    matches!(format, GlFormat::Depth24PlusStencil8)
}

pub(super) const fn address_mode(mode: GlAddressMode) -> i32 {
    match mode {
        GlAddressMode::ClampToEdge => Gl::CLAMP_TO_EDGE as i32,
        GlAddressMode::Repeat => Gl::REPEAT as i32,
        GlAddressMode::MirroredRepeat => Gl::MIRRORED_REPEAT as i32,
    }
}

pub(super) const fn filter_mode(mode: GlFilterMode) -> i32 {
    match mode {
        GlFilterMode::Nearest => Gl::NEAREST as i32,
        GlFilterMode::Linear => Gl::LINEAR as i32,
    }
}

pub(super) const fn min_filter(desc: GlSamplerDesc) -> i32 {
    match (desc.min_filter, desc.mipmap_filter) {
        (GlFilterMode::Nearest, GlMipmapFilterMode::Nearest) => Gl::NEAREST_MIPMAP_NEAREST as i32,
        (GlFilterMode::Nearest, GlMipmapFilterMode::Linear) => Gl::NEAREST_MIPMAP_LINEAR as i32,
        (GlFilterMode::Linear, GlMipmapFilterMode::Nearest) => Gl::LINEAR_MIPMAP_NEAREST as i32,
        (GlFilterMode::Linear, GlMipmapFilterMode::Linear) => Gl::LINEAR_MIPMAP_LINEAR as i32,
    }
}

pub(super) const fn compare_function(compare: GlCompareFunction) -> i32 {
    match compare {
        GlCompareFunction::Never => Gl::NEVER as i32,
        GlCompareFunction::Less => Gl::LESS as i32,
        GlCompareFunction::Equal => Gl::EQUAL as i32,
        GlCompareFunction::LessEqual => Gl::LEQUAL as i32,
        GlCompareFunction::Greater => Gl::GREATER as i32,
        GlCompareFunction::NotEqual => Gl::NOTEQUAL as i32,
        GlCompareFunction::GreaterEqual => Gl::GEQUAL as i32,
        GlCompareFunction::Always => Gl::ALWAYS as i32,
    }
}

/// Applies every sampler parameter a descriptor carries onto a fresh object.
pub(super) fn configure_sampler(raw: &Gl, sampler: &web_sys::WebGlSampler, desc: GlSamplerDesc) {
    raw.sampler_parameteri(
        sampler,
        Gl::TEXTURE_WRAP_S,
        address_mode(desc.address_mode_u),
    );
    raw.sampler_parameteri(
        sampler,
        Gl::TEXTURE_WRAP_T,
        address_mode(desc.address_mode_v),
    );
    raw.sampler_parameteri(
        sampler,
        Gl::TEXTURE_WRAP_R,
        address_mode(desc.address_mode_w),
    );
    raw.sampler_parameteri(
        sampler,
        Gl::TEXTURE_MAG_FILTER,
        filter_mode(desc.mag_filter),
    );
    raw.sampler_parameteri(sampler, Gl::TEXTURE_MIN_FILTER, min_filter(desc));
    raw.sampler_parameterf(
        sampler,
        Gl::TEXTURE_MIN_LOD,
        f32::from_bits(desc.lod_min_bits),
    );
    raw.sampler_parameterf(
        sampler,
        Gl::TEXTURE_MAX_LOD,
        f32::from_bits(desc.lod_max_bits),
    );
    if let Some(compare) = desc.compare {
        raw.sampler_parameteri(
            sampler,
            TEXTURE_COMPARE_MODE,
            Gl::COMPARE_REF_TO_TEXTURE as i32,
        );
        raw.sampler_parameteri(sampler, Gl::TEXTURE_COMPARE_FUNC, compare_function(compare));
    }
    if let Some(anisotropy) = desc.max_anisotropy_bits {
        raw.sampler_parameterf(
            sampler,
            TEXTURE_MAX_ANISOTROPY_EXT,
            f32::from_bits(anisotropy),
        );
    }
}

/// Baseline RGBA8 facts every WebGL 2.0 core context guarantees, plus float
/// facts answered by real framebuffer-completeness probes (audit P1-6).
///
/// `DEPTH_COMPONENT32F`, `RGBA16F`, and `RGBA32F` rendering are **not**
/// core-guaranteed in WebGL2; `EXT_color_buffer_float` supplies them. The
/// facts below therefore come only from attachment probes on this exact
/// context: an incomplete FBO records `renderable: false`, and a probe that
/// could not run leaves the conservative core record instead of an optimistic
/// claim.
pub(super) fn webgl2_baseline_formats(
    raw: &Gl,
    extensions: &GlExtensionSet,
) -> Result<GlFormatTable, super::super::api::GlError> {
    const RGBA16F: u32 = 0x881A;
    const RGBA32F: u32 = 0x8814;
    const HALF_FLOAT: u32 = 0x140B;
    let mut formats = GlFormatTable::default();
    for (format, copy) in [(GlFormat::Rgba8Unorm, true), (GlFormat::Rgba8Srgb, true)] {
        formats
            .record(GlFormatCapabilities {
                format,
                resource_kind: GlFormatResourceKind::Texture,
                sample_count: 1,
                evidence: GlFormatEvidence::CoreGuaranteed,
                sampled: true,
                filterable: true,
                renderable: true,
                blendable: true,
                storage_read: false,
                storage_write: false,
                copy_source: copy,
                copy_destination: copy,
            })
            .map_err(|error| record_error("record WebGL2 baseline format", error))?;
    }
    // Float depth: renderability is decided by the depth-attachment probe.
    let depth = attachment_completes(
        raw,
        Gl::DEPTH_COMPONENT32F,
        Gl::DEPTH_COMPONENT,
        Gl::FLOAT,
        Gl::DEPTH_ATTACHMENT,
    );
    formats
        .record(GlFormatCapabilities {
            format: GlFormat::Depth32Float,
            resource_kind: GlFormatResourceKind::Texture,
            sample_count: 1,
            evidence: match depth {
                Some(_) => GlFormatEvidence::OperationProbed,
                None => GlFormatEvidence::CoreGuaranteed,
            },
            sampled: true,
            filterable: false,
            renderable: depth.unwrap_or(false),
            blendable: false,
            storage_read: false,
            storage_write: false,
            // Do not claim a concrete copy operation from a static baseline.
            copy_source: false,
            copy_destination: false,
        })
        .map_err(|error| record_error("record WebGL2 depth fact", error))?;
    // Float color: renderability is decided per format by the color-attachment
    // probe; filtering of 32F additionally requires `OES_texture_float_linear`,
    // and blending of 32F additionally requires `EXT_float_blend`.
    let float_linear = extensions.is_acquired(GlKnownExtension::OesTextureFloatLinear);
    let float_blend = extensions.is_acquired(GlKnownExtension::ExtFloatBlend);
    for (format, internal_format, upload_type, filterable, blend_ext) in [
        (GlFormat::Rgba16Float, RGBA16F, HALF_FLOAT, true, false),
        (
            GlFormat::Rgba32Float,
            RGBA32F,
            Gl::FLOAT,
            float_linear,
            true,
        ),
    ] {
        let renderable = attachment_completes(
            raw,
            internal_format,
            Gl::RGBA,
            upload_type,
            Gl::COLOR_ATTACHMENT0,
        );
        let renderable = renderable.unwrap_or(false);
        formats
            .record(GlFormatCapabilities {
                format,
                resource_kind: GlFormatResourceKind::Texture,
                sample_count: 1,
                evidence: GlFormatEvidence::OperationProbed,
                sampled: true,
                filterable,
                renderable,
                blendable: renderable && blendable_rule(blend_ext, float_blend),
                storage_read: false,
                storage_write: false,
                copy_source: renderable,
                copy_destination: renderable,
            })
            .map_err(|error| record_error("record WebGL2 float fact", error))?;
    }
    for (extension, exact_formats) in compressed_extension_formats() {
        if !extensions.is_acquired(extension) {
            continue;
        }
        for format in exact_formats {
            formats
                .record(GlFormatCapabilities {
                    format,
                    resource_kind: GlFormatResourceKind::Texture,
                    sample_count: 1,
                    evidence: GlFormatEvidence::ExtensionAcquired(extension),
                    sampled: true,
                    filterable: true,
                    renderable: false,
                    blendable: false,
                    storage_read: false,
                    storage_write: false,
                    copy_source: false,
                    copy_destination: false,
                })
                .map_err(|error| record_error("record acquired compressed WebGL2 format", error))?;
        }
    }
    Ok(formats)
}

/// Adds ASTC HDR rows only after the exact WebGL extension object reported its
/// `hdr` profile. The HDR payload still uses the linear ASTC GLenum, but it is
/// a distinct typed format and must not inherit LDR evidence.
pub(super) fn record_astc_hdr_formats(
    formats: &mut GlFormatTable,
) -> Result<(), super::super::api::GlError> {
    for block in [
        B::B4x4,
        B::B5x4,
        B::B5x5,
        B::B6x5,
        B::B6x6,
        B::B8x5,
        B::B8x6,
        B::B8x8,
        B::B10x5,
        B::B10x6,
        B::B10x8,
        B::B10x10,
        B::B12x10,
        B::B12x12,
    ] {
        formats
            .record(GlFormatCapabilities {
                format: GlFormat::Astc {
                    block,
                    color_space: C::Hdr,
                },
                resource_kind: GlFormatResourceKind::Texture,
                sample_count: 1,
                evidence: GlFormatEvidence::ExtensionAcquired(
                    GlKnownExtension::CompressedTextureAstc,
                ),
                sampled: true,
                filterable: true,
                renderable: false,
                blendable: false,
                storage_read: false,
                storage_write: false,
                copy_source: false,
                copy_destination: false,
            })
            .map_err(|error| record_error("record WebGL2 ASTC HDR format", error))?;
    }
    Ok(())
}

fn blendable_rule(needs_ext: bool, ext_acquired: bool) -> bool {
    !needs_ext || ext_acquired
}

fn record_error(operation: &'static str, error: GlFormatTableError) -> super::super::api::GlError {
    super::super::api::GlError::Driver {
        operation,
        message: format!("{error:?}"),
    }
}

/// Attaches one freshly stored 4x4 texture to a scratch framebuffer and
/// reports completeness. Scratch objects are unbound and deleted on every
/// path; `None` means the probe could not run (browser exception or driver
/// error), never that rendering is supported.
fn attachment_completes(
    raw: &Gl,
    internal_format: u32,
    upload_format: u32,
    upload_type: u32,
    attachment: u32,
) -> Option<bool> {
    let texture = raw.create_texture()?;
    raw.bind_texture(Gl::TEXTURE_2D, Some(&texture));
    let stored = raw.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
        Gl::TEXTURE_2D,
        0,
        internal_format as i32,
        4,
        4,
        0,
        upload_format,
        upload_type,
        None,
    );
    let answer = stored.ok().and_then(|()| {
        let framebuffer = raw.create_framebuffer()?;
        raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&framebuffer));
        raw.framebuffer_texture_2d(
            Gl::FRAMEBUFFER,
            attachment,
            Gl::TEXTURE_2D,
            Some(&texture),
            0,
        );
        let status = raw.check_framebuffer_status(Gl::FRAMEBUFFER);
        let errored = raw.get_error() != Gl::NO_ERROR;
        raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
        raw.delete_framebuffer(Some(&framebuffer));
        if errored {
            None
        } else {
            Some(status == Gl::FRAMEBUFFER_COMPLETE)
        }
    });
    raw.bind_texture(Gl::TEXTURE_2D, None);
    raw.delete_texture(Some(&texture));
    let _ = raw.get_error();
    answer
}

fn compressed_extension_formats() -> Vec<(GlKnownExtension, Vec<GlFormat>)> {
    vec![
        (
            GlKnownExtension::CompressedTextureS3tc,
            vec![
                GlFormat::Bc1RgbUnorm,
                GlFormat::Bc1RgbaUnorm,
                GlFormat::Bc2RgbaUnorm,
                GlFormat::Bc3RgbaUnorm,
            ],
        ),
        (
            GlKnownExtension::CompressedTextureS3tcSrgb,
            vec![
                GlFormat::Bc1RgbSrgb,
                GlFormat::Bc1RgbaSrgb,
                GlFormat::Bc2RgbaSrgb,
                GlFormat::Bc3RgbaSrgb,
            ],
        ),
        (
            GlKnownExtension::CompressedTextureRgtc,
            vec![
                GlFormat::Bc4RUnorm,
                GlFormat::Bc4RSnorm,
                GlFormat::Bc5RgUnorm,
                GlFormat::Bc5RgSnorm,
            ],
        ),
        (
            GlKnownExtension::CompressedTextureBptc,
            vec![
                GlFormat::Bc6hRgbUfloat,
                GlFormat::Bc6hRgbSfloat,
                GlFormat::Bc7RgbaUnorm,
                GlFormat::Bc7RgbaSrgb,
            ],
        ),
        (
            GlKnownExtension::CompressedTextureEtc,
            vec![
                GlFormat::Etc2Rgb8Unorm,
                GlFormat::Etc2Rgb8Srgb,
                GlFormat::Etc2Rgba8Unorm,
                GlFormat::Etc2Rgba8Srgb,
                GlFormat::Etc2Rgb8A1Unorm,
                GlFormat::Etc2Rgb8A1Srgb,
                GlFormat::EacR11Unorm,
                GlFormat::EacRg11Unorm,
                GlFormat::EacR11Snorm,
                GlFormat::EacRg11Snorm,
            ],
        ),
        (
            GlKnownExtension::CompressedTextureAstc,
            [
                B::B4x4,
                B::B5x4,
                B::B5x5,
                B::B6x5,
                B::B6x6,
                B::B8x5,
                B::B8x6,
                B::B8x8,
                B::B10x5,
                B::B10x6,
                B::B10x8,
                B::B10x10,
                B::B12x10,
                B::B12x12,
            ]
            .into_iter()
            .flat_map(|block| {
                [
                    GlFormat::Astc {
                        block,
                        color_space: C::Linear,
                    },
                    GlFormat::Astc {
                        block,
                        color_space: C::Srgb,
                    },
                ]
            })
            .collect(),
        ),
    ]
}
