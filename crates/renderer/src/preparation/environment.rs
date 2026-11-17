//! Immutable input contract for the first environment-lighting consumer.
//!
//! The initial environment subset is **diffuse irradiance only**.  Its pixels
//! are a linear, non-premultiplied RGBA8 lat-long map; RGB is irradiance
//! **already divided by pi** and A is ignored. This is the Lambert diffuse
//! term consumed directly as `E_over_pi * base_color * (1 - metallic)`.
//! It is deliberately not described as a prefiltered radiance
//! map or full specular IBL.  The Renderer maps a normalized world-space
//! direction `d` to `u = fract(atan2(d.z, d.x) / TAU + 0.5)` and
//! `v = acos(clamp(d.y, -1, 1)) / PI`.  Positive Y is the north pole.

use fluxel_assets::AssetKind;

/// Marker for immutable, scene-level diffuse irradiance assets.
///
/// This is separate from a material color texture: an environment is selected
/// once for a scene/view and is owned by the Renderer through the completion
/// of each consumer submission.
pub struct EnvironmentIrradianceAsset;

impl AssetKind for EnvironmentIrradianceAsset {}

/// CPU payload for a linear RGBA8 equirectangular diffuse-irradiance map.
///
/// `rgba8` is tightly packed in row-major order.  Values are already linear;
/// callers must decode file encodings before constructing this payload.
#[derive(Clone, Debug)]
pub struct EnvironmentIrradianceData {
    /// Nonzero `[width, height]` of the latitude-longitude map.
    pub extent: [u32; 2],
    /// Four bytes per texel, in linear RGBA order.
    pub rgba8: Vec<u8>,
}

impl EnvironmentIrradianceData {
    /// Checks the CPU payload before any texture allocation or upload.
    pub fn is_valid(&self) -> bool {
        let Some(texels) = self.extent[0].checked_mul(self.extent[1]) else {
            return false;
        };
        !self.extent.contains(&0)
            && usize::try_from(texels)
                .ok()
                .and_then(|texels| texels.checked_mul(4))
                == Some(self.rgba8.len())
    }
}

#[cfg(test)]
#[path = "../tests/preparation/environment.rs"]
mod tests;
