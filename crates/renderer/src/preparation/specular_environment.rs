//! Immutable CPU contract for Babylon-compatible split-sum specular IBL.
//!
//! Radiance is a linear, latitude-longitude `RGBA16Float` mip chain.  The BRDF
//! LUT is also linear `RGBA16Float`; its RG channels are supplied by the
//! height-correlated split-sum generator and are not rescaled here.

use fluxel_assets::AssetKind;
use half::f16;

/// Marker for an immutable prefiltered-radiance plus BRDF-LUT environment.
pub struct EnvironmentSpecularIblAsset;

impl AssetKind for EnvironmentSpecularIblAsset {}

/// One tightly packed, little-endian `RGBA16Float` radiance mip.
#[derive(Clone, Debug)]
pub struct SpecularIblMip {
    /// Exact `[width, height]` for this mip.
    pub extent: [u32; 2],
    /// Four IEEE-754 binary16 values per texel, in RGBA order.
    pub rgba16f_le: Vec<u8>,
}

/// A tightly packed, little-endian `RGBA16Float` two-dimensional image.
#[derive(Clone, Debug)]
pub struct SpecularIblImage {
    /// Exact nonzero `[width, height]` image extent.
    pub extent: [u32; 2],
    /// Four IEEE-754 binary16 values per texel, in RGBA order.
    pub rgba16f_le: Vec<u8>,
}

impl SpecularIblImage {
    /// Expands a Babylon RGBD BRDF LUT into filterable linear RGBA16Float.
    ///
    /// Supply tightly packed RGBA8 texels in lookup order. X corresponds to
    /// NdotV and Y to roughness. RGB is gamma-encoded and alpha is the linear
    /// divisor D. Choose the exact sRGB transfer for Babylon's exact conversion
    /// option. Otherwise use the default gamma 2.2 transfer.
    ///
    /// Returns `None` for an invalid extent, byte count, or zero divisor.
    pub fn from_babylon_rgbd8(extent: [u32; 2], rgba8: &[u8], exact_srgb: bool) -> Option<Self> {
        let source_len = image_byte_len(extent, 4)?;
        let expanded_len = image_byte_len(extent, 8)?;
        if rgba8.len() != source_len {
            return None;
        }
        let mut rgba16f_le = Vec::with_capacity(expanded_len);
        for texel in rgba8.chunks_exact(4) {
            let divisor = f32::from(texel[3]) / 255.0;
            if divisor == 0.0 {
                return None;
            }
            for &component in &texel[..3] {
                let gamma = f32::from(component) / 255.0;
                let linear = if !exact_srgb {
                    gamma.powf(2.2)
                } else if gamma <= 0.04045 {
                    gamma / 12.92
                } else {
                    ((gamma + 0.055) / 1.055).powf(2.4)
                };
                rgba16f_le.extend_from_slice(&f16::from_f32(linear / divisor).to_le_bytes());
            }
            rgba16f_le.extend_from_slice(&f16::ONE.to_le_bytes());
        }
        Some(Self { extent, rgba16f_le })
    }
}

/// Babylon's `LinearSpecularReflection` LOD metadata.
#[derive(Clone, Copy, Debug)]
pub struct LinearSpecularLod {
    /// Babylon's `lodGenerationScale`.
    pub scale: f32,
    /// Babylon's `lodGenerationOffset`.
    pub offset: f32,
}

impl LinearSpecularLod {
    /// Maps roughness for an equivalent power-of-two texture of this mip count.
    ///
    /// The sampling shader uses the actual base width for a bound texture and
    /// clamps the resulting LOD to the available mip range.
    pub fn map(self, perceptual_roughness: f32, mip_count: u32) -> Option<f32> {
        if !self.is_valid() || !perceptual_roughness.is_finite() || mip_count == 0 {
            return None;
        }
        let max_lod = (mip_count - 1) as f32;
        Some(
            (max_lod * perceptual_roughness.clamp(0.0, 1.0) * self.scale + self.offset)
                .clamp(0.0, max_lod),
        )
    }

    /// Returns whether both authored conversion parameters are finite.
    pub fn is_valid(self) -> bool {
        self.scale.is_finite() && self.offset.is_finite()
    }
}

/// Immutable CPU payload for split-sum specular image-based lighting.
#[derive(Clone, Debug)]
pub struct EnvironmentSpecularIblData {
    /// Complete base-to-1×1 lat-long radiance chain.
    pub radiance_mips: Vec<SpecularIblMip>,
    /// Height-correlated BRDF integration LUT. Its channel convention belongs
    /// to the BRDF generator/consumer contract.
    pub brdf_lut: SpecularIblImage,
    /// Authored Babylon linear-specular LOD conversion.
    pub lod: LinearSpecularLod,
}

impl EnvironmentSpecularIblData {
    /// Validates all CPU bytes before any GPU resource allocation.
    pub fn is_valid(&self) -> bool {
        if self.radiance_mips.is_empty() || !self.lod.is_valid() || !valid_image(&self.brdf_lut) {
            return false;
        }
        let mut expected = self.radiance_mips[0].extent;
        if expected.contains(&0) {
            return false;
        }
        for mip in &self.radiance_mips {
            if mip.extent != expected || !valid_bytes(expected, &mip.rgba16f_le) {
                return false;
            }
            if expected == [1, 1] {
                expected = [0, 0];
            } else {
                expected = [(expected[0] / 2).max(1), (expected[1] / 2).max(1)];
            }
        }
        expected == [0, 0]
    }
}

fn valid_image(image: &SpecularIblImage) -> bool {
    valid_bytes(image.extent, &image.rgba16f_le)
}

fn image_byte_len(extent: [u32; 2], bytes_per_texel: usize) -> Option<usize> {
    if extent.contains(&0) {
        return None;
    }
    usize::try_from(extent[0].checked_mul(extent[1])?)
        .ok()?
        .checked_mul(bytes_per_texel)
}

fn valid_bytes(extent: [u32; 2], bytes: &[u8]) -> bool {
    if image_byte_len(extent, 8) != Some(bytes.len()) {
        return false;
    }
    bytes.chunks_exact(2).enumerate().all(|(component, pair)| {
        let half = u16::from_le_bytes([pair[0], pair[1]]);
        let exponent = (half >> 10) & 0x1f;
        let negative_nonzero = (half & 0x8000) != 0 && (half & 0x7fff) != 0;
        exponent != 0x1f && (component % 4 >= 3 || !negative_nonzero)
    })
}

#[cfg(test)]
#[path = "../tests/preparation/specular_environment.rs"]
mod tests;
