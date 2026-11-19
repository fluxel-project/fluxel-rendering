//! Material normalization and generic shader-contribution compilation.

mod canonical;
mod compile;
mod contribution;
mod names;
mod node_graph;
mod pbr_factors;
mod surface;
mod texture_encoding;
mod texture_uv;
mod transmission;
mod validation;
mod vertex_uv0;

use crate::{
    BindingLocation, CompiledMaterialVariant, EmissiveConfiguration, FragmentCall, FragmentInput,
    MaterialAsset, MaterialAssetRef, MaterialCodeNode, MaterialDefinition, MaterialError,
    MaterialFragment, MaterialParameter, MaterialResource, MaterialResourceKind,
    MaterialTextureEncodingRequirement, MaterialTextureUv, MaterialType, MaterialVariantKey,
    NormalMapSettings, PbrSurfaceFactors, RefractionEnvironment, StaticValue, SurfaceAlphaMode,
    SurfaceExtensions, SurfaceOutput, SurfacePolicy, TransmissionExtensions,
};
use fluxel_assets::AssetSnapshot;

pub use compile::MaterialCompiler;
pub use contribution::{
    MaterialBindingDeclaration, MaterialParameterBinding, MaterialShaderContribution,
};
