//! Diffuse-roughness selection on complete material definitions.

use super::{DiffuseRoughnessConfiguration, MaterialDefinition};

impl MaterialDefinition {
    /// Selects independent Babylon-style diffuse-lobe roughness controls.
    #[must_use]
    pub fn with_diffuse_roughness_configuration(
        mut self,
        configuration: DiffuseRoughnessConfiguration,
    ) -> Self {
        self.diffuse_roughness_configuration = Some(configuration);
        self
    }

    /// Returns independent diffuse-lobe roughness controls, when selected.
    #[must_use]
    pub fn diffuse_roughness_configuration(&self) -> Option<&DiffuseRoughnessConfiguration> {
        self.diffuse_roughness_configuration.as_ref()
    }
}
