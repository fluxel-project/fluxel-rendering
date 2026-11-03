//! Material compilation entry points.

use super::*;

impl MaterialCompiler {
    /// Validates a definition, normalizes its private graph, and emits a generic contribution.
    pub fn compile(
        definition: &MaterialDefinition,
    ) -> Result<CompiledMaterialVariant, MaterialError> {
        let ir = NormalizedMaterialIr::from_definition(definition)?;
        Ok(CompiledMaterialVariant {
            material: None,
            variant_key: ir.variant_key(),
            contribution: ir.contribution(),
            parameters: ir.parameters,
            resources: ir.resources,
        })
    }

    /// Compiles the definition owned by one exact immutable asset snapshot.
    pub fn compile_for(
        snapshot: &AssetSnapshot<MaterialAsset, MaterialDefinition>,
    ) -> Result<CompiledMaterialVariant, MaterialError> {
        let material = MaterialAssetRef::from_snapshot(snapshot);
        let mut compiled = Self::compile(snapshot.value())?;
        compiled.material = Some(material);
        compiled.variant_key.material = Some(material);
        Ok(compiled)
    }
}
