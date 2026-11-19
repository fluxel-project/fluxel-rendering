//! Author-facing material descriptions.

mod ambient_occlusion;
mod ambient_occlusion_definition;
mod assets;
mod code_node;
mod definition;
mod diffuse_roughness;
mod diffuse_roughness_definition;
mod nodes;
mod normal_map;
mod surface_policy;
mod transmission;
mod types;

pub use ambient_occlusion::{
    AmbientOcclusionConfiguration, AmbientOcclusionMap, AmbientOcclusionMapChannels,
};
pub use assets::{MaterialAsset, MaterialAssetId, MaterialAssetRef};
pub use code_node::MaterialCodeNode;
pub(crate) use definition::VertexUv0;
pub use definition::{
    EmissiveConfiguration, EmissiveSrgbDecode, EmissiveTexture, MaterialDefinition,
    MaterialTextureUv, MaterialVertexVarying, PbrBaseTexture, PbrPackedChannel,
    PbrPackedMetallicRoughness, PbrSurfaceFactors, SurfaceExtensions, SurfaceOutput, TextureUvSet,
};
pub use diffuse_roughness::{DiffuseRoughnessConfiguration, DiffuseRoughnessTexture};
pub use nodes::*;
pub use normal_map::NormalMapSettings;
pub use surface_policy::{
    SurfaceAlphaMode, SurfaceBlendMode, SurfacePolicy, SurfaceSideOrientation,
};
pub use transmission::{RefractionEnvironment, TransmissionExtensions};
pub use types::*;
