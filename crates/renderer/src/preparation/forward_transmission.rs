//! Known CPU-side representability bounds for world-space transmission.

use fluxel_material::MaterialDefinition;

use crate::RenderTransform;

use super::{ForwardDrawSource, ForwardPreparationError};

pub(super) fn validate_transform(
    material: &MaterialDefinition,
    transform: RenderTransform,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    let Some(transmission) = material.transmission_extensions() else {
        return Ok(());
    };
    let invalid = || ForwardPreparationError::InvalidTransmissionTransform { source };
    let matrix = transform.columns().map(|column| column.map(f64::from));
    if matrix.iter().flatten().any(|value| !value.is_finite())
        || matrix[0][3] != 0.0
        || matrix[1][3] != 0.0
        || matrix[2][3] != 0.0
        || matrix[3][3] != 1.0
    {
        return Err(invalid());
    }
    // Babylon scales authored thickness by the largest absolute world scale.
    // f64 makes this check independent of an overflowing f32 dot product; the
    // shader computes the corresponding column lengths with scaled operands.
    let scale = matrix[..3]
        .iter()
        .map(|column| {
            column[..3]
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max);
    if scale == 0.0 || !scale.is_finite() || scale > f64::from(f32::MAX) {
        return Err(invalid());
    }
    if transmission.thickness.is_none() {
        // Only the absent-callable path has a statically known result. A Code
        // Node can return values outside the imported texture's usual interval;
        // do not treat its authored range as evidence of its actual output.
        let local_thickness = transmission.maximum_thickness - transmission.minimum_thickness;
        let thickness = f64::from(local_thickness) * scale;
        if !thickness.is_finite() || thickness > f64::from(f32::MAX) {
            return Err(invalid());
        }
    }
    Ok(())
}
