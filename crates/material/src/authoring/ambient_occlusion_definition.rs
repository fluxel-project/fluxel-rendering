//! Ambient-occlusion selection on complete material definitions.

use super::{AmbientOcclusionConfiguration, MaterialDefinition};

impl MaterialDefinition {
    /// Selects Babylon-style colored ambient-occlusion controls.
    #[must_use]
    pub fn with_ambient_occlusion_configuration(
        mut self,
        configuration: AmbientOcclusionConfiguration,
    ) -> Self {
        self.ambient_occlusion_configuration = Some(configuration);
        self
    }

    /// Returns colored ambient-occlusion controls, when selected.
    #[must_use]
    pub fn ambient_occlusion_configuration(&self) -> Option<&AmbientOcclusionConfiguration> {
        self.ambient_occlusion_configuration.as_ref()
    }
}
