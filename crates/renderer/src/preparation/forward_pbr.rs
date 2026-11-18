//! CPU validation and normal-matrix construction for PBR Forward draws.

use fluxel_rhi::api::VertexFormat;

use crate::{PreparedGeometry, RenderTransform};

use super::{ForwardDrawSource, ForwardPreparationError, GeometryData, PreparedMaterial};

pub(super) fn normal_from_local(
    geometry: &GeometryData,
    prepared: &PreparedMaterial,
    transform: RenderTransform,
    source: ForwardDrawSource,
) -> Result<Option<[[f32; 4]; 4]>, ForwardPreparationError> {
    let output = prepared.material().output();
    if !crate::pbr::is_pbr(output) {
        return Ok(None);
    }
    let normal_transform = if prepared.uses_transmission() && transform.collapses_to_point() {
        // Triangle positions keep the real collapsed transform. The neutral
        // basis lets us still validate the original mesh normals and tangents.
        RenderTransform::identity()
    } else {
        transform
    };
    let normal = inverse_transpose(normal_transform, source)?;
    validate_normals(
        geometry,
        prepared.geometry(),
        normal,
        normal_transform,
        matches!(
            output,
            fluxel_material::SurfaceOutput::MetallicRoughnessUv0Normal { .. }
        ),
        source,
    )?;
    Ok(Some(normal))
}

fn validate_normals(
    geometry: &GeometryData,
    prepared: &PreparedGeometry,
    normal_from_local: [[f32; 4]; 4],
    world_from_local: RenderTransform,
    normal_map: bool,
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
    if !found {
        return Err(ForwardPreparationError::MissingNormal { source });
    }
    if normal_map {
        validate_tangents(
            geometry,
            prepared,
            normal_from_local,
            world_from_local,
            source,
        )?;
    }
    Ok(())
}

fn validate_tangents(
    geometry: &GeometryData,
    prepared: &PreparedGeometry,
    normal_from_local: [[f32; 4]; 4],
    world_from_local: RenderTransform,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    let mut found = false;
    for (stream_index, (layout, bytes)) in
        prepared.streams.iter().zip(&geometry.streams).enumerate()
    {
        for attribute in &layout.attributes {
            if attribute.location != 3 {
                continue;
            }
            if !matches!(attribute.format, VertexFormat::Float32(4)) {
                return Err(ForwardPreparationError::MissingTangent { source });
            }
            found = true;
            for vertex in geometry.vertices.clone() {
                let start = usize::try_from(u64::from(vertex) * layout.stride + attribute.offset)
                    .map_err(|_| ForwardPreparationError::InvalidTangent {
                    source,
                    stream: stream_index,
                })?;
                let values: [f32; 4] = std::array::from_fn(|component| {
                    bytes
                        .get(start + component * 4..start + component * 4 + 4)
                        .and_then(|value| value.try_into().ok())
                        .map(f32::from_le_bytes)
                        .unwrap_or(f32::NAN)
                });
                let length_squared =
                    values[0] * values[0] + values[1] * values[1] + values[2] * values[2];
                if !values.iter().all(|value| value.is_finite())
                    || length_squared == 0.0
                    || !matches!(values[3], 1.0 | -1.0)
                {
                    return Err(ForwardPreparationError::InvalidTangent {
                        source,
                        stream: stream_index,
                    });
                }
                let normal = normal_for_vertex(geometry, prepared, vertex).ok_or(
                    ForwardPreparationError::InvalidTangent {
                        source,
                        stream: stream_index,
                    },
                )?;
                let world_normal = transform3(normal_from_local, normal);
                let world_tangent = transform3(
                    world_from_local.columns(),
                    [values[0], values[1], values[2]],
                );
                if !world_normal
                    .iter()
                    .chain(world_tangent.iter())
                    .all(|value| value.is_finite())
                {
                    return Err(ForwardPreparationError::InvalidTangent {
                        source,
                        stream: stream_index,
                    });
                }
                let normal_scale = max_component(world_normal);
                let tangent_scale = max_component(world_tangent);
                if normal_scale == 0.0 || tangent_scale == 0.0 {
                    return Err(ForwardPreparationError::InvalidTangent {
                        source,
                        stream: stream_index,
                    });
                }
                let normal = world_normal.map(|value| value / normal_scale);
                let tangent = world_tangent.map(|value| value / tangent_scale);
                let normal_length = normal.iter().map(|value| value * value).sum::<f32>();
                let tangent_length = tangent.iter().map(|value| value * value).sum::<f32>();
                let dot = tangent[0] * normal[0] + tangent[1] * normal[1] + tangent[2] * normal[2];
                // A cross product close to zero leaves no stable bitangent;
                // reject before interpolation rather than selecting a view-dependent fallback.
                if tangent_length == 0.0
                    || normal_length == 0.0
                    || dot * dot >= tangent_length * normal_length * 0.999_999
                {
                    return Err(ForwardPreparationError::InvalidTangent {
                        source,
                        stream: stream_index,
                    });
                }
            }
        }
    }
    found
        .then_some(())
        .ok_or(ForwardPreparationError::MissingTangent { source })
}

pub(super) fn max_component(value: [f32; 3]) -> f32 {
    value.into_iter().map(f32::abs).fold(0.0_f32, f32::max)
}

fn transform3(matrix: [[f32; 4]; 4], value: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|row| {
        matrix[0][row] * value[0] + matrix[1][row] * value[1] + matrix[2][row] * value[2]
    })
}

fn normal_for_vertex(
    geometry: &GeometryData,
    prepared: &PreparedGeometry,
    vertex: u32,
) -> Option<[f32; 3]> {
    prepared
        .streams
        .iter()
        .zip(&geometry.streams)
        .find_map(|(layout, bytes)| {
            layout
                .attributes
                .iter()
                .find(|attribute| attribute.location == 1)
                .and_then(|attribute| {
                    (matches!(attribute.format, VertexFormat::Float32(3))).then_some(())?;
                    let start =
                        usize::try_from(u64::from(vertex) * layout.stride + attribute.offset)
                            .ok()?;
                    let values: [Option<f32>; 3] = std::array::from_fn(|component| {
                        bytes
                            .get(start + component * 4..start + component * 4 + 4)
                            .and_then(|value| value.try_into().ok())
                            .map(f32::from_le_bytes)
                    });
                    Some([values[0]?, values[1]?, values[2]?])
                })
        })
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
