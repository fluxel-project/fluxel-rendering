//! Refraction-energy selection on complete material definitions.

use super::{MaterialDefinition, RefractionEnergyMode};

impl MaterialDefinition {
    /// Selects the renderer's transmission refraction-energy relation.
    #[must_use]
    pub fn with_refraction_energy_mode(mut self, mode: RefractionEnergyMode) -> Self {
        self.refraction_energy_mode = mode;
        self
    }

    /// Returns the selected transmission refraction-energy relation.
    #[must_use]
    pub const fn refraction_energy_mode(&self) -> RefractionEnergyMode {
        self.refraction_energy_mode
    }
}
