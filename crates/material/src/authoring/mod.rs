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
pub use definition::{
    EmissiveConfiguration, EmissiveSrgbDecode, EmissiveTexture, MaterialDefinition, PbrBaseTexture,
    PbrPackedChannel, PbrPackedMetallicRoughness, PbrSurfaceFactors, SurfaceExtensions,
    SurfaceOutput,
};
pub use nodes::*;
pub use surface_policy::{
    SurfaceAlphaMode, SurfaceBlendMode, SurfacePolicy, SurfaceSideOrientation,
};
pub use transmission::{RefractionEnvironment, TransmissionExtensions};
pub use types::*;
