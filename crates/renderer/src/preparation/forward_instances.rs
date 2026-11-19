//! Immutable affine instance-transform preparation for Forward draws.

use std::borrow::Cow;

use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::{
    ScalarType, ShaderCompositionRequest, SourceUnit, VertexInputDeclaration, VertexType,
};

use crate::RenderTransform;

use super::{
    ForwardDrawSource, ForwardPreparationError, GeometryAttribute, GeometryData, GeometryStream,
};

const COLUMNS: [&str; 4] = [
    "INSTANCE_WORLD_0",
    "INSTANCE_WORLD_1",
    "INSTANCE_WORLD_2",
    "INSTANCE_WORLD_3",
];

#[derive(Clone, Copy)]
pub(super) struct InstanceTransforms {
    pub(super) count: u32,
    pub(super) reverses_winding: bool,
}

impl GeometryData {
    /// Appends immutable geometry-local affine instance transforms.
    ///
    /// The renderer applies the record transform after each instance transform.
    /// Supply a nonempty set of finite, non-singular affine matrices with a
    /// common orientation and bounds enclosing every instance in the record.
    #[must_use]
    pub fn with_instance_transforms(mut self, transforms: &[RenderTransform]) -> Self {
        self.layout.streams.push(GeometryStream {
            stride: 64,
            step_mode: VertexStepMode::Instance,
            attributes: COLUMNS
                .iter()
                .enumerate()
                .map(|(index, semantic)| GeometryAttribute {
                    semantic: (*semantic).into(),
                    offset: index as u64 * 16,
                    format: VertexFormat::Float32(4),
                })
                .collect(),
        });
        let mut bytes = Vec::with_capacity(transforms.len().saturating_mul(64));
        for transform in transforms {
            for value in transform.columns().into_iter().flatten() {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        self.streams.push(bytes);
        self
    }
}

pub(super) fn instance_transforms(
    geometry: &GeometryData,
    source: ForwardDrawSource,
) -> Result<Option<InstanceTransforms>, ForwardPreparationError> {
    let mut columns = [None; 4];
    let mut count = None;
    if geometry.layout.streams.len() != geometry.streams.len() {
        return Err(ForwardPreparationError::GeometryStreamCount {
            source,
            expected: geometry.layout.streams.len(),
            actual: geometry.streams.len(),
        });
    }
    for (stream_index, (stream, bytes)) in geometry
        .layout
        .streams
        .iter()
        .zip(&geometry.streams)
        .enumerate()
    {
        let unsupported = || ForwardPreparationError::InstanceStreamUnsupported {
            source,
            stream: stream_index,
        };
        if matches!(stream.step_mode, VertexStepMode::Vertex) {
            if stream
                .attributes
                .iter()
                .any(|attribute| COLUMNS.contains(&attribute.semantic.as_str()))
            {
                return Err(unsupported());
            }
            continue;
        }
        let stride = usize::try_from(stream.stride).map_err(|_| unsupported())?;
        if stride == 0
            || bytes.is_empty()
            || bytes.len() % stride != 0
            || stream.attributes.is_empty()
        {
            return Err(unsupported());
        }
        let stream_count = u32::try_from(bytes.len() / stride).map_err(|_| unsupported())?;
        if count
            .replace(stream_count)
            .is_some_and(|known| known != stream_count)
        {
            return Err(unsupported());
        }
        for attribute in &stream.attributes {
            let column = COLUMNS
                .iter()
                .position(|name| *name == attribute.semantic)
                .ok_or_else(unsupported)?;
            let offset = usize::try_from(attribute.offset).map_err(|_| unsupported())?;
            if columns[column].is_some()
                || !matches!(attribute.format, VertexFormat::Float32(4))
                || offset.checked_add(16).is_none_or(|end| end > stride)
            {
                return Err(unsupported());
            }
            columns[column] = Some((bytes.as_slice(), stride, offset));
        }
    }
    let Some(count) = count else { return Ok(None) };
    let [Some(c0), Some(c1), Some(c2), Some(c3)] = columns else {
        return Err(ForwardPreparationError::MissingInstanceTransformColumns { source });
    };
    let mut winding = None;
    for index in 0..count {
        let matrix = [c0, c1, c2, c3].map(|(bytes, stride, offset)| {
            let offset = index as usize * stride + offset;
            std::array::from_fn(|row| {
                let start = offset + 4 * row;
                f32::from_le_bytes(
                    bytes[start..start + 4]
                        .try_into()
                        .expect("validated instance column"),
                )
            })
        });
        let Some(reverse) = reversed_affine(matrix) else {
            return Err(ForwardPreparationError::InvalidInstanceTransform { source, index });
        };
        if winding
            .replace(reverse)
            .is_some_and(|known| known != reverse)
        {
            return Err(ForwardPreparationError::MixedInstanceOrientation { source });
        }
    }
    Ok(Some(InstanceTransforms {
        count,
        reverses_winding: winding.unwrap_or(false),
    }))
}

fn reversed_affine(matrix: [[f32; 4]; 4]) -> Option<bool> {
    if !matrix.iter().flatten().all(|value| value.is_finite())
        || matrix.map(|column| column[3]) != [0.0, 0.0, 0.0, 1.0]
    {
        return None;
    }
    let [x, y, z, _] = matrix;
    let cross = |a: [f32; 4], b: [f32; 4]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let cofactors = [cross(y, z), cross(z, x), cross(x, y)];
    let determinant = x[0] * cofactors[0][0] + x[1] * cofactors[0][1] + x[2] * cofactors[0][2];
    if !determinant.is_finite()
        || determinant == 0.0
        || !cofactors
            .iter()
            .flatten()
            .all(|value| (value / determinant).is_finite())
    {
        return None;
    }
    Some(determinant.is_sign_negative())
}

/// Mesh templates select their vertex-rate interface before the instance
/// columns are added to the request. Reflection still sees the full layout.
pub(crate) fn mesh_vertex_layout(layout: &super::GeometryLayout) -> Cow<'_, super::GeometryLayout> {
    if layout
        .streams
        .iter()
        .all(|stream| matches!(stream.step_mode, VertexStepMode::Vertex))
    {
        return Cow::Borrowed(layout);
    }
    Cow::Owned(super::GeometryLayout {
        streams: layout
            .streams
            .iter()
            .filter(|stream| matches!(stream.step_mode, VertexStepMode::Vertex))
            .cloned()
            .collect(),
    })
}

pub(crate) fn with_instance_transform(
    mut request: ShaderCompositionRequest,
    source: ForwardDrawSource,
) -> Result<ShaderCompositionRequest, ForwardPreparationError> {
    const PARAMETERS: &str = "/* instance_parameters */";
    const SETUP: &str = "/* instance_setup */";
    let mut vertex_units = 0;
    for unit in &mut request.source_units {
        vertex_units += unit.source.matches(PARAMETERS).count();
        unit.source = unit.source
            .replace(PARAMETERS, ", @location(4) instance_world_0: vec4<f32>, @location(5) instance_world_1: vec4<f32>, @location(6) instance_world_2: vec4<f32>, @location(7) instance_world_3: vec4<f32>")
            .replace(SETUP, "let instance_world = mat4x4<f32>(instance_world_0, instance_world_1, instance_world_2, instance_world_3);")
            .replace("/* instance_matrix */", "* instance_world")
            .replace("/* instance_normal_matrix */", "* renderer_instance_normal_matrix(instance_world)")
            .replace("/* instance_basis_argument */", ", instance_world")
            .replace("/* instance_basis_parameter */", ", instance_world: mat4x4<f32>")
            .replace("/* instance_basis_matrix */", "* mat3x3<f32>(instance_world[0].xyz, instance_world[1].xyz, instance_world[2].xyz)")
            .replace("/* instance_position_prefix */", "(instance_world * vec4<f32>(")
            .replace("/* instance_position_suffix */", ", 1.0)).xyz");
    }
    if vertex_units != 1 {
        return Err(ForwardPreparationError::InstanceMaterialUnsupported { source });
    }
    request.source_units.push(SourceUnit::new(
        "renderer-instance-normal",
        NORMAL_TRANSFORM,
    ));
    for (index, semantic) in COLUMNS.into_iter().enumerate() {
        request = request.with_vertex_input(VertexInputDeclaration::new(
            semantic,
            4 + index as u32,
            VertexType::Vec4(ScalarType::F32),
        ));
    }
    Ok(request.with_static_option("renderer.instance_transform", "affine-columns"))
}

const NORMAL_TRANSFORM: &str = r#"
fn renderer_instance_normal_matrix(world: mat4x4<f32>) -> mat4x4<f32> {
    let x = world[0].xyz;
    let y = world[1].xyz;
    let z = world[2].xyz;
    let c0 = cross(y, z);
    let c1 = cross(z, x);
    let c2 = cross(x, y);
    let determinant = dot(x, c0);
    return mat4x4<f32>(
        vec4<f32>(c0 / determinant, 0.0),
        vec4<f32>(c1 / determinant, 0.0),
        vec4<f32>(c2 / determinant, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
}
"#;
