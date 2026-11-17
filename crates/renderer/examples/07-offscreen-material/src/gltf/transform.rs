//! Flat non-singular TRS transforms for the selected reference meshes.

use super::GltfError;
use serde_json::Value;

#[derive(Clone, Copy)]
pub(super) struct Transform {
    rotation: [f32; 4],
    scale: [f32; 3],
    translation: [f32; 3],
}

impl Transform {
    pub(super) fn position(self, value: [f32; 3]) -> [f32; 3] {
        let value = self.rotate([
            value[0] * self.scale[0],
            value[1] * self.scale[1],
            value[2] * self.scale[2],
        ]);
        [
            value[0] + self.translation[0],
            value[1] + self.translation[1],
            value[2] + self.translation[2],
        ]
    }
    pub(super) fn normal(self, value: [f32; 3]) -> [f32; 3] {
        // For T*R*S, the inverse-transpose normal transform is R*(N/S).
        // Keeping each signed scale is essential for the nonuniform tunnel
        // and for mirrored nodes; position scaling is not a normal transform.
        let value = self.rotate(core::array::from_fn(|axis| value[axis] / self.scale[axis]));
        let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
        if !length.is_finite() || length == 0.0 {
            return [0.0, 0.0, 0.0];
        }
        [value[0] / length, value[1] / length, value[2] / length]
    }
    fn rotate(self, value: [f32; 3]) -> [f32; 3] {
        let [x, y, z, w] = self.rotation;
        let first = [
            y * value[2] - z * value[1],
            z * value[0] - x * value[2],
            x * value[1] - y * value[0],
        ];
        let twice = [2.0 * first[0], 2.0 * first[1], 2.0 * first[2]];
        let second = [
            y * twice[2] - z * twice[1],
            z * twice[0] - x * twice[2],
            x * twice[1] - y * twice[0],
        ];
        [
            value[0] + w * twice[0] + second[0],
            value[1] + w * twice[1] + second[1],
            value[2] + w * twice[2] + second[2],
        ]
    }
}

pub(super) fn node_transform(node: &Value) -> Result<Transform, GltfError> {
    let rotation = node
        .get("rotation")
        .map(|value| f32_array::<4>(value, "node rotation"))
        .transpose()?
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let translation = node
        .get("translation")
        .map(|value| f32_array::<3>(value, "node translation"))
        .transpose()?
        .unwrap_or([0.0; 3]);
    let scales = node
        .get("scale")
        .map(|value| f32_array::<3>(value, "node scale"))
        .transpose()?
        .unwrap_or([1.0; 3]);
    if !rotation
        .into_iter()
        .chain(translation)
        .chain(scales)
        .all(f32::is_finite)
    {
        return Err(GltfError::new("node transform contains non-finite value"));
    }
    if scales.contains(&0.0) {
        return Err(GltfError::new("node scale must be nonzero on every axis"));
    }
    let length = rotation
        .into_iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if length == 0.0 {
        return Err(GltfError::new("node rotation is zero"));
    }
    Ok(Transform {
        rotation: rotation.map(|value| value / length),
        scale: scales,
        translation,
    })
}

fn f32_array<const N: usize>(value: &Value, name: &str) -> Result<[f32; N], GltfError> {
    let values = value
        .as_array()
        .ok_or_else(|| GltfError::new(format!("{name} must be an array")))?;
    if values.len() != N {
        return Err(GltfError::new(format!("{name} has wrong component count")));
    }
    let mut output = [0.0; N];
    for (destination, source) in output.iter_mut().zip(values) {
        *destination = source
            .as_f64()
            .ok_or_else(|| GltfError::new(format!("{name} component must be numeric")))?
            as f32;
    }
    Ok(output)
}
