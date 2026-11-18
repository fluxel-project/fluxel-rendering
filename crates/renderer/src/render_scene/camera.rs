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
    sort_position: Option<[f32; 3]>,
    lighting_uses_eye: bool,
}

impl ShadingCamera {
    /// Creates a perspective camera with a finite world-space eye position.
    pub fn perspective(position: [f32; 3]) -> Result<Self, SceneError> {
        if !position.iter().all(|x| x.is_finite()) {
            return Err(SceneError::InvalidView);
        }
        Ok(Self {
            encoded: [position[0], position[1], position[2], 1.0],
            sort_position: Some(position),
            lighting_uses_eye: true,
        })
    }

    /// Creates a parallel camera from a nonzero world-space direction toward the eye.
    /// The direction is normalized without overflowing on large finite inputs.
    /// Sorted transparency also needs an eye position; use
    /// [`Self::orthographic_at`] when drawing transparent materials.
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
            sort_position: None,
            lighting_uses_eye: false,
        })
    }

    /// Creates a parallel lighting camera with an explicit world-space eye
    /// used for transparent-object distance sorting. The eye does not change
    /// the parallel lighting direction or the shader camera ABI.
    pub fn orthographic_at(
        position: [f32; 3],
        direction_to_eye: [f32; 3],
    ) -> Result<Self, SceneError> {
        if !position.iter().all(|value| value.is_finite()) {
            return Err(SceneError::InvalidView);
        }
        let mut camera = Self::orthographic(direction_to_eye)?;
        camera.sort_position = Some(position);
        Ok(camera)
    }

    /// Creates an orthographic camera with Babylon-style eye-position lighting.
    ///
    /// Surface lighting uses the direction from each world-space fragment to
    /// the eye, as Babylon PBR does for orthographic views. The camera retains
    /// its parallel direction, and the view still supplies its own projection.
    pub fn orthographic_with_eye_lighting(
        position: [f32; 3],
        direction_to_eye: [f32; 3],
    ) -> Result<Self, SceneError> {
        let mut camera = Self::orthographic_at(position, direction_to_eye)?;
        camera.lighting_uses_eye = true;
        Ok(camera)
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
        match self.sort_position.filter(|_| self.lighting_uses_eye) {
            Some([x, y, z]) => [x, y, z, 1.0],
            None => self.encoded,
        }
    }

    pub(crate) fn sorting_position(self) -> Option<[f32; 3]> {
        self.sort_position
    }
}
