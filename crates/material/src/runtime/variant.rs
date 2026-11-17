//! Compiled static material variants.

use crate::{
    MaterialAssetRef, MaterialParameter, MaterialResource, MaterialShaderContribution,
    SurfaceOutput, SurfacePolicy, TransmissionExtensions,
};

/// A stable static-variant identity derived from canonical material input.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MaterialVariantKey {
    canonical: String,
    material: Option<MaterialAssetRef>,
}

impl MaterialVariantKey {
    pub(crate) fn from_canonical(canonical: String) -> Self {
        Self {
            canonical,
            material: None,
        }
    }
    /// Returns the canonical variant representation for diagnostics and caches.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.canonical
    }
}

/// A compiled static material variant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledMaterialVariant {
    material: Option<MaterialAssetRef>,
    variant_key: MaterialVariantKey,
    contribution: MaterialShaderContribution,
    parameters: Vec<MaterialParameter>,
    resources: Vec<MaterialResource>,
    output: SurfaceOutput,
    surface_policy: SurfacePolicy,
    transmission_extensions: Option<TransmissionExtensions>,
}

impl CompiledMaterialVariant {
    pub(crate) fn new(
        variant_key: MaterialVariantKey,
        contribution: MaterialShaderContribution,
        parameters: Vec<MaterialParameter>,
        resources: Vec<MaterialResource>,
        output: SurfaceOutput,
        surface_policy: SurfacePolicy,
        transmission_extensions: Option<TransmissionExtensions>,
    ) -> Self {
        Self {
            material: None,
            variant_key,
            contribution,
            parameters,
            resources,
            output,
            surface_policy,
            transmission_extensions,
        }
    }

    pub(crate) fn bind_material(&mut self, material: MaterialAssetRef) {
        self.material = Some(material);
        self.variant_key.material = Some(material);
    }

    /// Returns the asset generation this variant represents, when it was compiled from an asset.
    #[must_use]
    pub const fn material(&self) -> Option<MaterialAssetRef> {
        self.material
    }

    /// Returns the variant's static identity.
    #[must_use]
    pub fn key(&self) -> &MaterialVariantKey {
        &self.variant_key
    }

    /// Returns the generic contribution consumed by Shader.
    #[must_use]
    pub fn contribution(&self) -> &MaterialShaderContribution {
        &self.contribution
    }

    /// Returns the selected surface domain for this compiled variant.
    #[must_use]
    pub fn output(&self) -> &SurfaceOutput {
        &self.output
    }

    /// Returns the static alpha and raster facts for this compiled variant.
    #[must_use]
    pub const fn surface_policy(&self) -> SurfacePolicy {
        self.surface_policy
    }

    /// Returns requested transmission and volume-attenuation controls.
    #[must_use]
    pub fn transmission_extensions(&self) -> Option<&TransmissionExtensions> {
        self.transmission_extensions.as_ref()
    }

    pub(crate) fn parameters(&self) -> &[MaterialParameter] {
        &self.parameters
    }
    pub(crate) fn resources(&self) -> &[MaterialResource] {
        &self.resources
    }
}
