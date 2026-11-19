//! Babylon-style PBR opacity authoring controls.

/// Texture channels used by a PBR opacity map.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PbrOpacityChannels {
    /// Multiplies the incoming alpha by sampled alpha.
    #[default]
    Alpha,
    /// Replaces the incoming alpha with saturated linear-RGB luminance.
    Luminance,
}

/// One PBR opacity texture binding.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PbrOpacityTexture {
    /// Declared `Texture2dF32` resource, linear for luminance sampling.
    pub texture: String,
    /// Declared non-comparison sampler resource.
    pub sampler: String,
    /// Selected texture alpha relation.
    pub channels: PbrOpacityChannels,
    /// Optional `F32` material parameter naming Babylon's opacity level.
    pub level_parameter: Option<String>,
}

/// Optional Babylon PBR opacity controls.
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PbrOpacityConfiguration {
    /// Optional `F32` parameter multiplied into authored base alpha first.
    pub alpha_parameter: Option<String>,
    /// Optional opacity map.
    pub texture: Option<PbrOpacityTexture>,
}
