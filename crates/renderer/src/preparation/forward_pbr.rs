//! CPU validation and normal-matrix construction for PBR Forward draws.

use fluxel_rhi::api::VertexFormat;

use crate::{PreparedGeometry, RenderTransform};

use super::{ForwardDrawSource, ForwardPreparationError, GeometryData};

pub(super) fn normal_from_local(
    geometry: &GeometryData,
    prepared: &PreparedGeometry,
    transform: RenderTransform,
    pbr: bool,
    source: ForwardDrawSource,
) -> Result<Option<[[f32; 4]; 4]>, ForwardPreparationError> {
    if !pbr {
        return Ok(None);
    }
    let normal = inverse_transpose(transform, source)?;
    validate_normals(geometry, prepared, normal, source)?;
    Ok(Some(normal))
}

fn validate_normals(
    geometry: &GeometryData,
    prepared: &PreparedGeometry,
    normal_from_local: [[f32; 4]; 4],
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    let mut found = false;
    for (stream_index, (layout, bytes)) in
        prepared.streams.iter().zip(&geometry.streams).enumerate()
    {
        for attribute in &layout.attributes {
            // The selected PBR request maps NORMAL to location 1. Use the
            // prepared physical layout, after semantic and payload validation.
            if attribute.location != 1 {
                continue;
            }
            if !matches!(attribute.format, VertexFormat::Float32(3)) {
                return Err(ForwardPreparationError::MissingNormal { source });
            }
            found = true;
            for vertex in geometry.vertices.clone() {
                let start = usize::try_from(u64::from(vertex) * layout.stride + attribute.offset)
                    .map_err(|_| ForwardPreparationError::InvalidNormal {
                    source,
                    stream: stream_index,
                })?;
                let mut values = [0.0; 3];
                for (component, value) in values.iter_mut().enumerate() {
                    let offset = start + component * 4;
                    *value = bytes
                        .get(offset..offset + 4)
                        .and_then(|value| value.try_into().ok())
                        .map(f32::from_le_bytes)
                        .ok_or(ForwardPreparationError::InvalidNormal {
                            source,
                            stream: stream_index,
                        })?;
                }
                if !values.iter().all(|value| value.is_finite())
                    || values.iter().all(|value| *value == 0.0)
                {
                    return Err(ForwardPreparationError::InvalidNormal {
                        source,
                        stream: stream_index,
                    });
                }
                let transformed: [f32; 3] = std::array::from_fn(|row| {
                    normal_from_local[0][row] * values[0]
                        + normal_from_local[1][row] * values[1]
                        + normal_from_local[2][row] * values[2]
                });
                if !transformed.iter().all(|value| value.is_finite())
                    || transformed.iter().all(|value| *value == 0.0)
                {
                    return Err(ForwardPreparationError::InvalidNormal {
                        source,
                        stream: stream_index,
                    });
                }
            }
        }
    }
    found
        .then_some(())
        .ok_or(ForwardPreparationError::MissingNormal { source })
}

fn inverse_transpose(
    transform: RenderTransform,
    source: ForwardDrawSource,
) -> Result<[[f32; 4]; 4], ForwardPreparationError> {
    normal_matrix(transform).ok_or(ForwardPreparationError::NonInvertibleNormalTransform { source })
}

pub(super) fn normal_matrix(transform: RenderTransform) -> Option<[[f32; 4]; 4]> {
    // Compute cofactors in f64: finite f32 scales can overflow an f32
    // determinant even when the resulting normal matrix is representable.
    let matrix = transform.columns().map(|column| column.map(f64::from));
    if matrix[0][3] != 0.0 || matrix[1][3] != 0.0 || matrix[2][3] != 0.0 || matrix[3][3] != 1.0 {
        return None;
    }
    let (a, b, c) = (matrix[0][0], matrix[1][0], matrix[2][0]);
    let (d, e, f) = (matrix[0][1], matrix[1][1], matrix[2][1]);
    let (g, h, i) = (matrix[0][2], matrix[1][2], matrix[2][2]);
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if !determinant.is_finite() || determinant == 0.0 {
        return None;
    }
    let inverse = [
        [
            (e * i - f * h) / determinant,
            (c * h - b * i) / determinant,
            (b * f - c * e) / determinant,
        ],
        [
            (f * g - d * i) / determinant,
            (a * i - c * g) / determinant,
            (c * d - a * f) / determinant,
        ],
        [
            (d * h - e * g) / determinant,
            (b * g - a * h) / determinant,
            (a * e - b * d) / determinant,
        ],
    ];
    let mut normal = [[0.0; 4]; 4];
    for column in 0..3 {
        for row in 0..3 {
            normal[column][row] = inverse[column][row] as f32;
        }
    }
    normal[3][3] = 1.0;
    if normal.iter().flatten().all(|value| value.is_finite()) {
        Some(normal)
    } else {
        None
    }
}
