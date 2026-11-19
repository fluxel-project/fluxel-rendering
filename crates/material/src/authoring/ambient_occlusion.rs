//! Babylon-style ambient-occlusion authoring controls.

/// Color channels consumed from one linear ambient-occlusion map.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AmbientOcclusionMapChannels {
    /// Replicates the sampled red channel across RGB.
    Red,
    /// Uses the sampled linear RGB value directly.
    #[default]
    Rgb,
}

/// One declared linear ambient-occlusion map binding.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AmbientOcclusionMap {
    /// Declared linear `Texture2dF32` resource sampled by the evaluator.
    pub texture: String,
    /// Declared non-comparison sampler resource paired with [`Self::texture`].
    pub sampler: String,
    /// Optional `F32` material parameter naming the map level.
    pub level_parameter: Option<String>,
    /// Selected map color channels.
    pub channels: AmbientOcclusionMapChannels,
}

/// Babylon-style colored ambient-occlusion controls.
///
/// The evaluator first mixes the existing authored scalar occlusion with this
/// required map by strength, then applies any packed PBR occlusion channel.
/// Direct-light weighting remains a separate scalar in the fourth component.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AmbientOcclusionConfiguration {
    /// Required linear ambient-occlusion map.
    pub map: AmbientOcclusionMap,
    /// Optional `F32` material parameter naming the map mix strength.
    pub strength_parameter: Option<String>,
    /// Optional `F32` material parameter naming the direct-light weight.
    pub direct_weight_parameter: Option<String>,
}
