//! Per-instance material values.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    CompiledMaterialVariant, DynamicValue, MaterialAssetRef, MaterialError, MaterialVariantKey,
};

/// Per-instance dynamic material data over a compiled static variant.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialInstance {
    material: MaterialAssetRef,
    variant: MaterialVariantKey,
    parameters: BTreeMap<String, DynamicValue>,
    resources: BTreeMap<String, String>,
}

impl MaterialInstance {
    /// Validates dynamic values against a compiled variant and creates an instance.
    pub fn new(
        material: MaterialAssetRef,
        variant: &CompiledMaterialVariant,
        parameters: BTreeMap<String, DynamicValue>,
        resources: BTreeMap<String, String>,
    ) -> Result<Self, MaterialError> {
        if let Some(expected) = variant.material() {
            if expected != material {
                return Err(MaterialError::VariantMaterialMismatch {
                    expected,
                    actual: material,
                });
            }
        }
        let expected_parameters = variant
            .parameters()
            .iter()
            .map(|parameter| (parameter.name.as_str(), parameter.ty))
            .collect::<BTreeMap<_, _>>();
        for (name, value) in &parameters {
            let expected = expected_parameters
                .get(name.as_str())
                .ok_or_else(|| MaterialError::UnknownParameter { name: name.clone() })?;
            if *expected != value.ty() {
                return Err(MaterialError::ParameterTypeMismatch {
                    name: name.clone(),
                    expected: *expected,
                    actual: value.ty(),
                });
            }
        }
        for parameter in variant.parameters() {
            if !parameters.contains_key(&parameter.name) {
                return Err(MaterialError::MissingParameter {
                    name: parameter.name.clone(),
                });
            }
        }
        let expected_resources = variant
            .resources()
            .iter()
            .map(|resource| resource.name.as_str())
            .collect::<BTreeSet<_>>();
        let parameter_resources = variant
            .parameters()
            .iter()
            .map(|parameter| parameter.resource.as_str())
            .collect::<BTreeSet<_>>();
        for name in resources.keys() {
            if !expected_resources.contains(name.as_str()) {
                return Err(MaterialError::UnknownResource { name: name.clone() });
            }
        }
        for resource in variant.resources() {
            if !parameter_resources.contains(resource.name.as_str())
                && !resources.contains_key(&resource.name)
            {
                return Err(MaterialError::MissingResource {
                    name: resource.name.clone(),
                });
            }
        }
        Ok(Self {
            material,
            variant: variant.key().clone(),
            parameters,
            resources,
        })
    }

    /// Returns the asset identity and content generation this instance represents.
    #[must_use]
    pub const fn material(&self) -> MaterialAssetRef {
        self.material
    }
    /// Returns the selected static variant key.
    #[must_use]
    pub fn variant(&self) -> &MaterialVariantKey {
        &self.variant
    }
    /// Returns the validated dynamic parameter values for concrete uniform upload.
    #[must_use]
    pub fn parameters(&self) -> &BTreeMap<String, DynamicValue> {
        &self.parameters
    }
    /// Returns one validated dynamic parameter by name.
    #[must_use]
    pub fn parameter(&self, name: &str) -> Option<&DynamicValue> {
        self.parameters.get(name)
    }
    /// Returns the validated logical resource values for concrete binding resolution.
    #[must_use]
    pub fn resources(&self) -> &BTreeMap<String, String> {
        &self.resources
    }
    /// Returns one validated logical resource value by name.
    #[must_use]
    pub fn resource(&self, name: &str) -> Option<&str> {
        self.resources.get(name).map(String::as_str)
    }
}
