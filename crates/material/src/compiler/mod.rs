//! Material normalization and generic shader-contribution compilation.

mod ambient_occlusion;
mod canonical;
mod compile;
mod contribution;
mod diffuse_roughness;
mod names;
mod node_graph;
mod pbr_factors;
mod surface;
mod texture_encoding;
mod texture_uv;
mod transmission;
mod validation;
mod validation_code_nodes;
mod vertex_uv0;

use crate::{
    AmbientOcclusionConfiguration, BindingLocation, CompiledMaterialVariant,
    DiffuseRoughnessConfiguration, EmissiveConfiguration, FragmentCall, FragmentInput,
    MaterialAsset, MaterialAssetRef, MaterialCodeNode, MaterialDefinition, MaterialError,
    MaterialFragment, MaterialParameter, MaterialResource, MaterialResourceKind,
    MaterialTextureEncodingRequirement, MaterialTextureUv, MaterialType, MaterialVariantKey,
    NormalMapSettings, PbrSurfaceFactors, RefractionEnvironment, StaticValue, SurfaceAlphaMode,
    SurfaceExtensions, SurfaceOutput, SurfacePolicy, TransmissionExtensions,
    TransmissionTextureConfiguration,
};
use fluxel_assets::AssetSnapshot;

pub use compile::MaterialCompiler;
pub use contribution::{
    MaterialBindingDeclaration, MaterialParameterBinding, MaterialShaderContribution,
};
