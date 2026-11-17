//! Static transmission and volume-attenuation facts for PBR surfaces.

/// Author-selected transmission controls for a UV0 normal PBR surface.
///
/// `intensity` and `thickness` name optional `vec2<f32> -> f32` callables.
/// Material only emits evaluation helpers; consumers own refraction-source
/// selection, pass ordering, and resource lifetime.
#[derive(Clone, Debug)]
pub struct TransmissionExtensions {
    /// Optional UV0 callable for the transmission amount.
    pub intensity: Option<String>,
    /// Optional UV0 callable for a thickness sample before interval remapping.
    ///
    /// Imported texture sources commonly return normalized values, but Material
    /// intentionally does not clamp an author Code Node's result.
    pub thickness: Option<String>,
    /// Surface index of refraction.
    pub index_of_refraction: f32,
    /// Volume index of refraction, or the surface IOR when absent.
    pub volume_index_of_refraction: Option<f32>,
    /// The transmitted color at [`Self::attenuation_distance`].
    pub attenuation_color: [f32; 3],
    /// Positive distance used to derive Beer-Lambert absorption.
    pub attenuation_distance: f32,
    /// Lower bound for normalized thickness samples.
    pub minimum_thickness: f32,
    /// Upper bound for normalized thickness samples.
    pub maximum_thickness: f32,
}

impl Default for TransmissionExtensions {
    fn default() -> Self {
        Self {
            intensity: None,
            thickness: None,
            index_of_refraction: 1.5,
            volume_index_of_refraction: None,
            attenuation_color: [1.0; 3],
            attenuation_distance: 1.0,
            minimum_thickness: 0.0,
            maximum_thickness: 1.0,
        }
    }
}

impl PartialEq for TransmissionExtensions {
    fn eq(&self, other: &Self) -> bool {
        self.intensity == other.intensity
            && self.thickness == other.thickness
            && self.index_of_refraction.to_bits() == other.index_of_refraction.to_bits()
            && self.volume_index_of_refraction.map(f32::to_bits)
                == other.volume_index_of_refraction.map(f32::to_bits)
            && self.attenuation_color.map(f32::to_bits) == other.attenuation_color.map(f32::to_bits)
            && self.attenuation_distance.to_bits() == other.attenuation_distance.to_bits()
            && self.minimum_thickness.to_bits() == other.minimum_thickness.to_bits()
            && self.maximum_thickness.to_bits() == other.maximum_thickness.to_bits()
    }
}

impl Eq for TransmissionExtensions {}
