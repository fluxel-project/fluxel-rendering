//! Static transmission and volume-attenuation facts for PBR surfaces.

/// Refraction energy relation selected for transmission lighting.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RefractionEnergyMode {
    /// Preserves the established transmission energy relation.
    #[default]
    Legacy,
    /// Applies base Fresnel energy partitioning in the renderer's refraction path.
    BaseFresnel,
}

/// Channel layout used by one transmission texture.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TransmissionTextureLayout {
    /// Babylon native packing: thickness red, intensity green.
    BabylonNative,
    /// glTF packing: thickness green, intensity red.
    #[default]
    Gltf,
}

/// One linear transmission texture binding.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TransmissionTexture {
    /// Declared linear `Texture2dF32` resource.
    pub texture: String,
    /// Declared non-comparison sampler resource.
    pub sampler: String,
}

/// Optional texture sources for transmission extensions.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TransmissionTextureConfiguration {
    /// Channel layout used by the selected textures.
    pub layout: TransmissionTextureLayout,
    /// Optional thickness source; its channel can also mask intensity.
    pub thickness: Option<TransmissionTexture>,
    /// Uses the thickness texture mask when a separate intensity map is absent.
    pub use_as_intensity_mask: bool,
    /// Optional separate transmission-intensity source.
    pub intensity: Option<TransmissionTexture>,
}

/// A material-local, prefiltered environment used for transmission refraction.
///
/// The texture and sampler name ordinary group-zero material resources. Its
/// RGB values must already be linear. `lod_scale` and `lod_offset` transform
/// the consumer's normalized roughness LOD before it is clamped to the bound
/// texture's available mip range. `direction_from_world` rotates a world-space
/// refraction direction into the texture's environment direction convention.
#[derive(Clone, Debug)]
pub struct RefractionEnvironment {
    /// Declared linear `Texture2dF32` material resource.
    pub texture: String,
    /// Declared non-comparison material sampler resource.
    pub sampler: String,
    /// Scale applied to the normalized roughness LOD.
    pub lod_scale: f32,
    /// Offset applied to the normalized roughness LOD.
    pub lod_offset: f32,
    /// Environment direction derived from the Fluxel world-space direction.
    pub direction_from_world: [[f32; 3]; 3],
}

impl PartialEq for RefractionEnvironment {
    fn eq(&self, other: &Self) -> bool {
        self.texture == other.texture
            && self.sampler == other.sampler
            && self.lod_scale.to_bits() == other.lod_scale.to_bits()
            && self.lod_offset.to_bits() == other.lod_offset.to_bits()
            && self
                .direction_from_world
                .iter()
                .flatten()
                .map(|value| value.to_bits())
                .eq(other
                    .direction_from_world
                    .iter()
                    .flatten()
                    .map(|value| value.to_bits()))
    }
}

impl Eq for RefractionEnvironment {}

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
