//! Tangent-space normal-map authoring controls.

/// Static tangent-space normal-map controls for a normal-bearing PBR surface.
///
/// The values describe the authored normal-map convention only. They do not
/// change Fluxel's RHI coordinate system, winding, or backend transforms.
/// `invert_x` and `invert_y` are consumed by the renderer's derivative TBN
/// fallback; an explicit mesh tangent basis already carries its handedness.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NormalMapSettings {
    strength_bits: u32,
    /// Inverts the tangent-space X channel for the derivative fallback.
    pub invert_x: bool,
    /// Inverts the tangent-space Y channel for the derivative fallback.
    pub invert_y: bool,
}

impl NormalMapSettings {
    /// Creates settings when strength is finite and non-negative.
    #[must_use]
    pub fn new(strength: f32, invert_x: bool, invert_y: bool) -> Option<Self> {
        (strength.is_finite() && strength >= 0.0).then(|| Self {
            strength_bits: strength.to_bits(),
            invert_x,
            invert_y,
        })
    }

    /// Returns the authored normal-map strength.
    #[must_use]
    pub const fn strength(self) -> f32 {
        f32::from_bits(self.strength_bits)
    }
}

impl Default for NormalMapSettings {
    fn default() -> Self {
        Self::new(1.0, false, false).expect("unit normal-map settings are valid")
    }
}
