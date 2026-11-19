//! Transmission texture selection on complete material definitions.

use super::{MaterialDefinition, TransmissionTextureConfiguration};

impl MaterialDefinition {
    /// Selects optional linear texture sources for transmission controls.
    #[must_use]
    pub fn with_transmission_texture_configuration(
        mut self,
        configuration: TransmissionTextureConfiguration,
    ) -> Self {
        self.transmission_texture_configuration = Some(configuration);
        self
    }

    /// Returns selected transmission texture sources.
    #[must_use]
    pub fn transmission_texture_configuration(&self) -> Option<&TransmissionTextureConfiguration> {
        self.transmission_texture_configuration.as_ref()
    }
}
