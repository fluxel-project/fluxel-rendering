//! The single Metal pixel-format naming authority.
//!
//! A native spelling is deliberately not a capability claim. In particular,
//! macOS and Apple-family GPUs expose different compressed-format sets. The
//! capability probe must additionally gate `metal_format` through the device
//! family and through the lowering it has actually implemented.

use objc2_metal::MTLPixelFormat;

use crate::api::format::TextureFormat;

/// Returns Metal's exact spelling for a portable format.
///
/// `None` means there is no one-to-one Metal representation in the current
/// backend. It must never be replaced by emulation: e.g. `R64Uint` is not an
/// `RG32Uint` texture, and the portable depth-24 aliases have no stable Metal
/// representation across all supported devices.
pub(super) fn metal_format(format: TextureFormat) -> Option<MTLPixelFormat> {
    use MTLPixelFormat as Mtl;
    use TextureFormat as F;

    Some(match format {
        F::R8Unorm => Mtl::R8Unorm,
        F::R8Snorm => Mtl::R8Snorm,
        F::R8Uint => Mtl::R8Uint,
        F::R8Sint => Mtl::R8Sint,
        F::Rg8Unorm => Mtl::RG8Unorm,
        F::Rg8Snorm => Mtl::RG8Snorm,
        F::Rg8Uint => Mtl::RG8Uint,
        F::Rg8Sint => Mtl::RG8Sint,
        F::Rgba8Unorm => Mtl::RGBA8Unorm,
        F::Rgba8UnormSrgb => Mtl::RGBA8Unorm_sRGB,
        F::Rgba8Snorm => Mtl::RGBA8Snorm,
        F::Rgba8Uint => Mtl::RGBA8Uint,
        F::Rgba8Sint => Mtl::RGBA8Sint,
        F::Bgra8Unorm => Mtl::BGRA8Unorm,
        F::Bgra8UnormSrgb => Mtl::BGRA8Unorm_sRGB,

        F::R16Unorm => Mtl::R16Unorm,
        F::R16Snorm => Mtl::R16Snorm,
        F::R16Uint => Mtl::R16Uint,
        F::R16Sint => Mtl::R16Sint,
        F::R16Float => Mtl::R16Float,
        F::Rg16Unorm => Mtl::RG16Unorm,
        F::Rg16Snorm => Mtl::RG16Snorm,
        F::Rg16Uint => Mtl::RG16Uint,
        F::Rg16Sint => Mtl::RG16Sint,
        F::Rg16Float => Mtl::RG16Float,
        F::Rgba16Unorm => Mtl::RGBA16Unorm,
        F::Rgba16Snorm => Mtl::RGBA16Snorm,
        F::Rgba16Uint => Mtl::RGBA16Uint,
        F::Rgba16Sint => Mtl::RGBA16Sint,
        F::Rgba16Float => Mtl::RGBA16Float,

        F::R32Uint => Mtl::R32Uint,
        F::R32Sint => Mtl::R32Sint,
        F::R32Float => Mtl::R32Float,
        F::Rg32Uint => Mtl::RG32Uint,
        F::Rg32Sint => Mtl::RG32Sint,
        F::Rg32Float => Mtl::RG32Float,
        F::Rgba32Uint => Mtl::RGBA32Uint,
        F::Rgba32Sint => Mtl::RGBA32Sint,
        F::Rgba32Float => Mtl::RGBA32Float,
        F::Rgb9e5Ufloat => Mtl::RGB9E5Float,
        F::Rgb10a2Uint => Mtl::RGB10A2Uint,
        F::Rgb10a2Unorm => Mtl::RGB10A2Unorm,
        F::Rg11b10Ufloat => Mtl::RG11B10Float,

        F::Depth16Unorm => Mtl::Depth16Unorm,
        F::Depth32Float => Mtl::Depth32Float,
        F::Depth32FloatStencil8 => Mtl::Depth32Float_Stencil8,
        F::Stencil8 => Mtl::Stencil8,

        F::Bc1RgbaUnorm => Mtl::BC1_RGBA,
        F::Bc1RgbaUnormSrgb => Mtl::BC1_RGBA_sRGB,
        F::Bc2RgbaUnorm => Mtl::BC2_RGBA,
        F::Bc2RgbaUnormSrgb => Mtl::BC2_RGBA_sRGB,
        F::Bc3RgbaUnorm => Mtl::BC3_RGBA,
        F::Bc3RgbaUnormSrgb => Mtl::BC3_RGBA_sRGB,
        F::Bc4RUnorm => Mtl::BC4_RUnorm,
        F::Bc4RSnorm => Mtl::BC4_RSnorm,
        F::Bc5RgUnorm => Mtl::BC5_RGUnorm,
        F::Bc5RgSnorm => Mtl::BC5_RGSnorm,
        F::Bc6hRgbFloat => Mtl::BC6H_RGBFloat,
        F::Bc6hRgbUfloat => Mtl::BC6H_RGBUfloat,
        F::Bc7RgbaUnorm => Mtl::BC7_RGBAUnorm,
        F::Bc7RgbaUnormSrgb => Mtl::BC7_RGBAUnorm_sRGB,

        F::Etc2Rgb8Unorm => Mtl::ETC2_RGB8,
        F::Etc2Rgb8UnormSrgb => Mtl::ETC2_RGB8_sRGB,
        F::Etc2Rgb8A1Unorm => Mtl::ETC2_RGB8A1,
        F::Etc2Rgb8A1UnormSrgb => Mtl::ETC2_RGB8A1_sRGB,
        F::Etc2Rgba8Unorm => Mtl::EAC_RGBA8,
        F::Etc2Rgba8UnormSrgb => Mtl::EAC_RGBA8_sRGB,
        F::EacR11Unorm => Mtl::EAC_R11Unorm,
        F::EacR11Snorm => Mtl::EAC_R11Snorm,
        F::EacRg11Unorm => Mtl::EAC_RG11Unorm,
        F::EacRg11Snorm => Mtl::EAC_RG11Snorm,

        F::Astc4x4Unorm => Mtl::ASTC_4x4_LDR,
        F::Astc4x4UnormSrgb => Mtl::ASTC_4x4_sRGB,
        F::Astc4x4Hdr => Mtl::ASTC_4x4_HDR,
        F::Astc5x4Unorm => Mtl::ASTC_5x4_LDR,
        F::Astc5x4UnormSrgb => Mtl::ASTC_5x4_sRGB,
        F::Astc5x4Hdr => Mtl::ASTC_5x4_HDR,
        F::Astc5x5Unorm => Mtl::ASTC_5x5_LDR,
        F::Astc5x5UnormSrgb => Mtl::ASTC_5x5_sRGB,
        F::Astc5x5Hdr => Mtl::ASTC_5x5_HDR,
        F::Astc6x5Unorm => Mtl::ASTC_6x5_LDR,
        F::Astc6x5UnormSrgb => Mtl::ASTC_6x5_sRGB,
        F::Astc6x5Hdr => Mtl::ASTC_6x5_HDR,
        F::Astc6x6Unorm => Mtl::ASTC_6x6_LDR,
        F::Astc6x6UnormSrgb => Mtl::ASTC_6x6_sRGB,
        F::Astc6x6Hdr => Mtl::ASTC_6x6_HDR,
        F::Astc8x5Unorm => Mtl::ASTC_8x5_LDR,
        F::Astc8x5UnormSrgb => Mtl::ASTC_8x5_sRGB,
        F::Astc8x5Hdr => Mtl::ASTC_8x5_HDR,
        F::Astc8x6Unorm => Mtl::ASTC_8x6_LDR,
        F::Astc8x6UnormSrgb => Mtl::ASTC_8x6_sRGB,
        F::Astc8x6Hdr => Mtl::ASTC_8x6_HDR,
        F::Astc8x8Unorm => Mtl::ASTC_8x8_LDR,
        F::Astc8x8UnormSrgb => Mtl::ASTC_8x8_sRGB,
        F::Astc8x8Hdr => Mtl::ASTC_8x8_HDR,
        F::Astc10x5Unorm => Mtl::ASTC_10x5_LDR,
        F::Astc10x5UnormSrgb => Mtl::ASTC_10x5_sRGB,
        F::Astc10x5Hdr => Mtl::ASTC_10x5_HDR,
        F::Astc10x6Unorm => Mtl::ASTC_10x6_LDR,
        F::Astc10x6UnormSrgb => Mtl::ASTC_10x6_sRGB,
        F::Astc10x6Hdr => Mtl::ASTC_10x6_HDR,
        F::Astc10x8Unorm => Mtl::ASTC_10x8_LDR,
        F::Astc10x8UnormSrgb => Mtl::ASTC_10x8_sRGB,
        F::Astc10x8Hdr => Mtl::ASTC_10x8_HDR,
        F::Astc10x10Unorm => Mtl::ASTC_10x10_LDR,
        F::Astc10x10UnormSrgb => Mtl::ASTC_10x10_sRGB,
        F::Astc10x10Hdr => Mtl::ASTC_10x10_HDR,
        F::Astc12x10Unorm => Mtl::ASTC_12x10_LDR,
        F::Astc12x10UnormSrgb => Mtl::ASTC_12x10_sRGB,
        F::Astc12x10Hdr => Mtl::ASTC_12x10_HDR,
        F::Astc12x12Unorm => Mtl::ASTC_12x12_LDR,
        F::Astc12x12UnormSrgb => Mtl::ASTC_12x12_sRGB,
        F::Astc12x12Hdr => Mtl::ASTC_12x12_HDR,

        // These require plane-aware view/copy lowering or a device-specific
        // depth choice. The native spellings must stay inaccessible until then.
        F::R64Uint | F::Depth24Plus | F::Depth24PlusStencil8 | F::Nv12 | F::P010 => return None,
    })
}

/// Compressed format family. Native spelling alone is insufficient: this is
/// used by facts to apply a device-family gate before publishing the format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CompressedFamily {
    Bc,
    Etc2Eac,
    AstcLdr,
    AstcHdr,
}

pub(super) const fn compressed_family(format: TextureFormat) -> Option<CompressedFamily> {
    use CompressedFamily as C;
    use TextureFormat as F;
    Some(match format {
        F::Bc1RgbaUnorm
        | F::Bc1RgbaUnormSrgb
        | F::Bc2RgbaUnorm
        | F::Bc2RgbaUnormSrgb
        | F::Bc3RgbaUnorm
        | F::Bc3RgbaUnormSrgb
        | F::Bc4RUnorm
        | F::Bc4RSnorm
        | F::Bc5RgUnorm
        | F::Bc5RgSnorm
        | F::Bc6hRgbUfloat
        | F::Bc6hRgbFloat
        | F::Bc7RgbaUnorm
        | F::Bc7RgbaUnormSrgb => C::Bc,
        F::Etc2Rgb8Unorm
        | F::Etc2Rgb8UnormSrgb
        | F::Etc2Rgb8A1Unorm
        | F::Etc2Rgb8A1UnormSrgb
        | F::Etc2Rgba8Unorm
        | F::Etc2Rgba8UnormSrgb
        | F::EacR11Unorm
        | F::EacR11Snorm
        | F::EacRg11Unorm
        | F::EacRg11Snorm => C::Etc2Eac,
        F::Astc4x4Unorm
        | F::Astc4x4UnormSrgb
        | F::Astc5x4Unorm
        | F::Astc5x4UnormSrgb
        | F::Astc5x5Unorm
        | F::Astc5x5UnormSrgb
        | F::Astc6x5Unorm
        | F::Astc6x5UnormSrgb
        | F::Astc6x6Unorm
        | F::Astc6x6UnormSrgb
        | F::Astc8x5Unorm
        | F::Astc8x5UnormSrgb
        | F::Astc8x6Unorm
        | F::Astc8x6UnormSrgb
        | F::Astc8x8Unorm
        | F::Astc8x8UnormSrgb
        | F::Astc10x5Unorm
        | F::Astc10x5UnormSrgb
        | F::Astc10x6Unorm
        | F::Astc10x6UnormSrgb
        | F::Astc10x8Unorm
        | F::Astc10x8UnormSrgb
        | F::Astc10x10Unorm
        | F::Astc10x10UnormSrgb
        | F::Astc12x10Unorm
        | F::Astc12x10UnormSrgb
        | F::Astc12x12Unorm
        | F::Astc12x12UnormSrgb => C::AstcLdr,
        F::Astc4x4Hdr
        | F::Astc5x4Hdr
        | F::Astc5x5Hdr
        | F::Astc6x5Hdr
        | F::Astc6x6Hdr
        | F::Astc8x5Hdr
        | F::Astc8x6Hdr
        | F::Astc8x8Hdr
        | F::Astc10x5Hdr
        | F::Astc10x6Hdr
        | F::Astc10x8Hdr
        | F::Astc10x10Hdr
        | F::Astc12x10Hdr
        | F::Astc12x12Hdr => C::AstcHdr,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_and_planar_formats_do_not_get_an_unsafe_native_spelling() {
        assert!(metal_format(TextureFormat::R64Uint).is_none());
        assert!(metal_format(TextureFormat::Depth24Plus).is_none());
        assert!(metal_format(TextureFormat::Nv12).is_none());
    }

    #[test]
    fn astc_classes_distinguish_hdr_from_ldr() {
        assert_eq!(
            compressed_family(TextureFormat::Astc4x4Unorm),
            Some(CompressedFamily::AstcLdr)
        );
        assert_eq!(
            compressed_family(TextureFormat::Astc4x4Hdr),
            Some(CompressedFamily::AstcHdr)
        );
    }
}
