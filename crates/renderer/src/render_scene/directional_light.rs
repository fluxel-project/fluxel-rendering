//! Directional direct lighting captured in a render view.

use super::SceneError;

/// One world-space directional light for the built-in PBR paths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionalLight {
    direction_and_roughness: [f32; 4],
    radiance: [f32; 4],
}

impl DirectionalLight {
    /// Creates a light from the emitted ray direction, linear RGB, intensity,
    /// and angular radius. With a unit direction the radius is in radians.
    pub fn new(
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        angular_radius: f32,
    ) -> Result<Self, SceneError> {
        if direction.iter().any(|value| !value.is_finite())
            || color.iter().any(|value| !value.is_finite() || *value < 0.0)
            || !intensity.is_finite()
            || intensity < 0.0
            || !angular_radius.is_finite()
            || angular_radius < 0.0
        {
            return Err(SceneError::InvalidDirectionalLight);
        }

        let scale = direction.into_iter().map(f32::abs).fold(0.0, f32::max);
        let radiance = color.map(|channel| channel * intensity);
        if scale == 0.0 || radiance.iter().any(|value| !value.is_finite()) {
            return Err(SceneError::InvalidDirectionalLight);
        }

        // Babylon adjusts direct specular roughness by radius / length(direction).
        // Keep the original magnitude for this calculation, then upload a scaled
        // vector so shader normalization also works for extreme finite inputs.
        let length = direction
            .into_iter()
            .map(|value| f64::from(value).powi(2))
            .sum::<f64>()
            .sqrt();
        let roughness = (f64::from(angular_radius) / length).min(1.0) as f32;
        let [x, y, z] = direction.map(|component| (f64::from(component) / f64::from(scale)) as f32);
        let [r, g, b] = radiance;

        Ok(Self {
            direction_and_roughness: [x, y, z, roughness],
            radiance: [r, g, b, 0.0],
        })
    }

    pub(crate) fn uniform(self) -> [f32; 8] {
        let [x, y, z, roughness] = self.direction_and_roughness;
        let [r, g, b, padding] = self.radiance;
        [x, y, z, roughness, r, g, b, padding]
    }
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction_and_roughness: [-0.6, 0.0, -0.8, 0.0],
            radiance: [3.0, 3.0, 3.0, 0.0],
        }
    }
}
