//! PBR opacity selection on complete material definitions.
use super::{MaterialDefinition, PbrOpacityConfiguration};
impl MaterialDefinition {
    /// Selects optional Babylon PBR opacity controls.
    #[must_use]
    pub fn with_pbr_opacity_configuration(
        mut self,
        configuration: PbrOpacityConfiguration,
    ) -> Self {
        self.pbr_opacity_configuration = Some(configuration);
        self
    }
    /// Returns selected PBR opacity controls.
    #[must_use]
    pub fn pbr_opacity_configuration(&self) -> Option<&PbrOpacityConfiguration> {
        self.pbr_opacity_configuration.as_ref()
    }
}
