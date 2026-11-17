//! Explicit world-space camera facts used by surface lighting consumers.

use super::SceneError;

/// Camera origin or parallel viewing direction for world-space surface lighting.
///
/// Projection remains in [`super::RenderView::transform`]. This value supplies
/// the physical eye information that cannot be recovered unambiguously from an
/// arbitrary combined clip transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadingCamera {
    encoded: [f32; 4],
}

impl ShadingCamera {
    /// Creates a perspective camera with a finite world-space eye position.
    pub fn perspective(position: [f32; 3]) -> Result<Self, SceneError> {
        if !position.iter().all(|x| x.is_finite()) {
            return Err(SceneError::InvalidView);
        }
        Ok(Self {
            encoded: [position[0], position[1], position[2], 1.0],
        })
    }

    /// Creates a parallel camera from a nonzero world-space direction toward the eye.
    /// The direction is normalized without overflowing on large finite inputs.
    pub fn orthographic(direction_to_eye: [f32; 3]) -> Result<Self, SceneError> {
        if !direction_to_eye.iter().all(|x| x.is_finite()) {
            return Err(SceneError::InvalidView);
        }
        let v = direction_to_eye.map(f64::from);
        let length = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if length == 0.0 {
            return Err(SceneError::InvalidView);
        }
        let v = v.map(|x| (x / length) as f32);
        Ok(Self {
            encoded: [v[0], v[1], v[2], 0.0],
        })
    }

    /// Returns the perspective world-space eye, if this is a perspective camera.
    pub fn position(self) -> Option<[f32; 3]> {
        (self.encoded[3] == 1.0).then(|| [self.encoded[0], self.encoded[1], self.encoded[2]])
    }

    /// Returns the normalized parallel direction toward the eye, if applicable.
    pub fn direction_to_eye(self) -> Option<[f32; 3]> {
        (self.encoded[3] == 0.0).then(|| [self.encoded[0], self.encoded[1], self.encoded[2]])
    }

    pub(crate) fn uniform(self) -> [f32; 4] {
        self.encoded
    }
}
