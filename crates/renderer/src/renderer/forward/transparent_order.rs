//! Renderer-owned final ordering of ordinary transparent material draws.

use fluxel_material::SurfaceAlphaMode;
use fluxel_rhi::api::{VertexFormat, VertexStepMode};

use crate::{ForwardDrawInput, PipelineError};

pub(super) fn is_transparent(input: &ForwardDrawInput) -> bool {
    matches!(
        input.prepared().material().surface_policy().alpha_mode,
        SurfaceAlphaMode::Blend | SurfaceAlphaMode::MaskBlend
    )
}

/// Orders all transparent sources together, irrespective of asset/Feature owner.
/// Squared distance has the same order as Babylon's Euclidean distance and
/// avoids rounding distinct far distances to the same square root. Equal keys
/// retain the received draw order, matching Babylon's stable array sort.
pub(super) fn sort(
    inputs: Vec<ForwardDrawInput>,
    mut alpha_index: impl FnMut(&ForwardDrawInput) -> i64,
) -> Result<Vec<ForwardDrawInput>, PipelineError> {
    let mut keyed = inputs
        .into_iter()
        .map(|input| Ok((alpha_index(&input), distance_squared(&input)?, input)))
        .collect::<Result<Vec<_>, PipelineError>>()?;
    keyed.sort_by(
        |(left_index, left_distance, _), (right_index, right_distance, _)| {
            left_index
                .cmp(right_index)
                .then_with(|| right_distance.total_cmp(left_distance))
        },
    );
    Ok(keyed.into_iter().map(|(_, _, input)| input).collect())
}

pub(super) fn validate(input: &ForwardDrawInput) -> Result<(), PipelineError> {
    if is_transparent(input) {
        distance_squared(input)?;
    }
    Ok(())
}

fn distance_squared(input: &ForwardDrawInput) -> Result<f64, PipelineError> {
    let source = input.source();
    let eye = input
        .view()
        .shading_camera
        .and_then(crate::ShadingCamera::sorting_position)
        .ok_or(PipelineError::MissingTransparencyCamera { source })?;
    let invalid = || PipelineError::InvalidTransparencyGeometry { source };
    let geometry = input.geometry().value();
    // Use the authored local bounds center, transformed into world space, as
    // Babylon does for its bounding sphere center. Conservative scene bounds
    // used for visibility are not a substitute for this geometry fact.
    let mut positions = geometry
        .layout
        .streams
        .iter()
        .zip(&geometry.streams)
        .flat_map(|(layout, bytes)| {
            layout
                .attributes
                .iter()
                .filter(|attribute| attribute.semantic == "POSITION")
                .map(move |attribute| (layout, bytes, attribute))
        });
    let (layout, bytes, attribute) = positions.next().ok_or_else(invalid)?;
    if positions.next().is_some()
        || !matches!(layout.step_mode, VertexStepMode::Vertex)
        || !matches!(attribute.format, VertexFormat::Float32(3))
        || geometry.vertices.is_empty()
    {
        return Err(invalid());
    }
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for vertex in geometry.vertices.clone() {
        let start = u64::from(vertex)
            .checked_mul(layout.stride)
            .and_then(|offset| offset.checked_add(attribute.offset))
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or_else(invalid)?;
        for component in 0..3 {
            let offset = start.checked_add(component * 4).ok_or_else(invalid)?;
            let end = offset.checked_add(4).ok_or_else(invalid)?;
            let value = bytes
                .get(offset..end)
                .and_then(|bytes| bytes.try_into().ok())
                .map(f32::from_le_bytes)
                .filter(|value| value.is_finite())
                .ok_or_else(invalid)?;
            minimum[component] = minimum[component].min(f64::from(value));
            maximum[component] = maximum[component].max(f64::from(value));
        }
    }
    let center: [f64; 3] = std::array::from_fn(|axis| (minimum[axis] + maximum[axis]) * 0.5);
    let world = input
        .transform()
        .columns()
        .map(|column| column.map(f64::from));
    // World-position shader consumers use affine xyz, without a homogeneous
    // divide. Do not invent a different sorting space for a projective matrix.
    if world.iter().flatten().any(|value| !value.is_finite())
        || world[0][3] != 0.0
        || world[1][3] != 0.0
        || world[2][3] != 0.0
        || world[3][3] != 1.0
    {
        return Err(invalid());
    }
    let distance = (0..3)
        .map(|axis| {
            let coordinate = world[0][axis] * center[0]
                + world[1][axis] * center[1]
                + world[2][axis] * center[2]
                + world[3][axis];
            (coordinate - f64::from(eye[axis])).powi(2)
        })
        .sum::<f64>();
    if !distance.is_finite() {
        return Err(invalid());
    }
    Ok(distance)
}
