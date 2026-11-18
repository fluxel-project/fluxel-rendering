//! Mesh opacity supplied independently of the material and render view.

use super::SceneError;

/// Validated mesh opacity multiplier for transparent composition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshVisibility(f32);

impl MeshVisibility {
    /// Constructs a visibility in the inclusive range from zero to one.
    pub fn new(visibility: f32) -> Result<Self, SceneError> {
        if (0.0..=1.0).contains(&visibility) {
            Ok(Self(visibility))
        } else {
            Err(SceneError::InvalidMeshVisibility)
        }
    }

    /// Returns the validated opacity multiplier.
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl Default for MeshVisibility {
    fn default() -> Self {
        Self(1.0)
    }
}
