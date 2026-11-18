//! Material normalization and generic shader-contribution compilation.

mod canonical;
mod compile;
mod contribution;
mod names;
mod node_graph;
mod pbr_factors;
mod surface;
mod texture_encoding;
mod transmission;
mod validation;

use crate::{
    BindingLocation, CompiledMaterialVariant, EmissiveConfiguration, FragmentCall, FragmentInput,
    MaterialAsset, MaterialAssetRef, MaterialCodeNode, MaterialDefinition, MaterialError,
    MaterialFragment, MaterialParameter, MaterialResource, MaterialResourceKind,
    MaterialTextureEncodingRequirement, MaterialType, MaterialVariantKey, PbrSurfaceFactors,
    StaticValue, SurfaceAlphaMode, SurfaceExtensions, SurfaceOutput, SurfacePolicy,
    TransmissionExtensions,
};
use fluxel_assets::AssetSnapshot;

pub use compile::MaterialCompiler;
pub use contribution::{
    MaterialBindingDeclaration, MaterialParameterBinding, MaterialShaderContribution,
};
