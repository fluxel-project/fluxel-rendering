//! Material compilation entry points.

use super::validation::NormalizedMaterialIr;
use super::*;

/// Compiles a validated authoring definition into one static material variant.
pub struct MaterialCompiler;

impl MaterialCompiler {
    /// Validates a definition, normalizes its private graph, and emits a generic contribution.
    pub fn compile(
        definition: &MaterialDefinition,
    ) -> Result<CompiledMaterialVariant, MaterialError> {
        let ir = NormalizedMaterialIr::from_definition(definition)?;
        Ok(CompiledMaterialVariant::new(
            ir.variant_key(),
            ir.contribution(),
            ir.parameters,
            ir.resources,
            ir.texture_encoding_requirements,
            ir.output,
            ir.normal_map_settings,
            ir.surface_policy,
            ir.transmission_extensions,
        ))
    }

    /// Compiles the definition owned by one exact immutable asset snapshot.
    pub fn compile_for(
        snapshot: &AssetSnapshot<MaterialAsset, MaterialDefinition>,
    ) -> Result<CompiledMaterialVariant, MaterialError> {
        let material = MaterialAssetRef::from_snapshot(snapshot);
        let mut compiled = Self::compile(snapshot.value())?;
        compiled.bind_material(material);
        Ok(compiled)
    }
}
