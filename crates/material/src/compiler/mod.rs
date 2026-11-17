//! Material normalization and generic shader-contribution compilation.

mod canonical;
mod compile;
mod contribution;
mod names;
mod node_graph;
mod surface;
mod transmission;
mod validation;

use crate::{
    BindingLocation, CompiledMaterialVariant, FragmentCall, FragmentInput, MaterialAsset,
    MaterialAssetRef, MaterialCodeNode, MaterialDefinition, MaterialError, MaterialFragment,
    MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType, MaterialVariantKey,
    StaticValue, SurfaceAlphaMode, SurfaceExtensions, SurfaceOutput, SurfacePolicy,
    TransmissionExtensions,
};
use fluxel_assets::AssetSnapshot;

pub use compile::MaterialCompiler;
pub use contribution::{
    MaterialBindingDeclaration, MaterialParameterBinding, MaterialShaderContribution,
};
