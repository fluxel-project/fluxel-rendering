//! Babylon-style diffuse-roughness authoring controls.

/// One linear texture contribution to diffuse roughness.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DiffuseRoughnessTexture {
    /// Declared linear `Texture2dF32` resource sampled by the evaluator.
    pub texture: String,
    /// Declared non-comparison sampler resource paired with [`Self::texture`].
    pub sampler: String,
    /// Optional `F32` material parameter naming Babylon's texture level.
    pub level_parameter: Option<String>,
}

/// Independent diffuse-lobe roughness controls for a PBR material.
///
/// This follows Babylon's PBR relation: a finite base value, zero by default,
/// is multiplied by the selected linear texture's red channel and texture
/// level. It deliberately has no association with specular or perceptual
/// roughness controls.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DiffuseRoughnessConfiguration {
    base_roughness_bits: u32,
    /// Optional linear red-channel texture contribution.
    pub texture: Option<DiffuseRoughnessTexture>,
}

impl DiffuseRoughnessConfiguration {
    /// Creates controls when the base roughness is finite.
    #[must_use]
    pub fn new(base_roughness: f32) -> Option<Self> {
        base_roughness.is_finite().then(|| Self {
            base_roughness_bits: base_roughness.to_bits(),
            texture: None,
        })
    }

    /// Returns the authored diffuse-lobe roughness before texture modulation.
    #[must_use]
    pub const fn base_roughness(&self) -> f32 {
        f32::from_bits(self.base_roughness_bits)
    }
}

impl Default for DiffuseRoughnessConfiguration {
    fn default() -> Self {
        Self::new(0.0).expect("zero diffuse roughness is finite")
    }
}
