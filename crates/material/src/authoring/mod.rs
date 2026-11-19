//! Author-facing material descriptions.

mod assets;
mod code_node;
mod definition;
mod nodes;
mod surface_policy;
mod transmission;
mod types;

pub use assets::{MaterialAsset, MaterialAssetId, MaterialAssetRef};
pub use code_node::MaterialCodeNode;
pub(crate) use definition::VertexUv0;
pub use definition::{
    EmissiveConfiguration, EmissiveSrgbDecode, EmissiveTexture, MaterialDefinition,
    MaterialTextureUv, MaterialVertexVarying, PbrBaseTexture, PbrPackedChannel,
    PbrPackedMetallicRoughness, PbrSurfaceFactors, SurfaceExtensions, SurfaceOutput, TextureUvSet,
};
pub use nodes::*;
pub use surface_policy::{
    SurfaceAlphaMode, SurfaceBlendMode, SurfacePolicy, SurfaceSideOrientation,
};
pub use transmission::{RefractionEnvironment, TransmissionExtensions};
pub use types::*;
