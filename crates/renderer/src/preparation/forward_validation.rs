//! Asset, geometry, and template validation for Forward preparation.

use fluxel_assets::{AssetKind, AssetSnapshot};
use fluxel_material::MaterialResourceKind;
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::{
    BindingDeclaration, BindingKind, EntryPoint, ScalarType, ShaderCompositionRequest, ShaderStage,
    SourceUnit, VertexInputDeclaration, VertexType,
};

use crate::{RenderTransform, SceneAssetRef};

use super::{
    ColorTextureData, ForwardDrawSource, ForwardPreparationError, GeometryData, GeometryLayout,
    PreparedMaterial,
};

pub(super) fn resolve_asset<K: AssetKind, V>(
    snapshots: &[AssetSnapshot<K, V>],
    reference: SceneAssetRef<K>,
    asset: &'static str,
    source: ForwardDrawSource,
) -> Result<AssetSnapshot<K, V>, ForwardPreparationError> {
    let mut matches = snapshots.iter().filter(|snapshot| {
        snapshot.id() == reference.id() && snapshot.generation() == reference.generation()
    });
    let Some(snapshot) = matches.next() else {
        return Err(ForwardPreparationError::MissingAsset { asset, source });
    };
    if matches.next().is_some() {
        return Err(ForwardPreparationError::DuplicateAsset { asset, source });
    }
    Ok(snapshot.clone())
}

pub(super) fn validate_texture(
    texture: &ColorTextureData,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    let [width, height] = texture.extent;
    let Some(bytes) = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|texels| texels.checked_mul(4))
        .and_then(|bytes| usize::try_from(bytes).ok())
    else {
        return Err(ForwardPreparationError::InvalidTexture { source });
    };
    if width == 0 || height == 0 || texture.rgba8.len() != bytes {
        return Err(ForwardPreparationError::InvalidTexture { source });
    }
    Ok(())
}

pub(super) fn validate_cpu_geometry(
    geometry: &GeometryData,
    prepared: &crate::PreparedGeometry,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    if geometry.vertices.start > geometry.vertices.end {
        return Err(ForwardPreparationError::InvalidVertexRange { source });
    }
    if geometry.streams.len() != prepared.streams.len() {
        return Err(ForwardPreparationError::GeometryStreamCount {
            source,
            expected: prepared.streams.len(),
            actual: geometry.streams.len(),
        });
    }
    if geometry.vertices.is_empty() {
        return Ok(());
    }
    for (stream_index, (layout, bytes)) in
        prepared.streams.iter().zip(&geometry.streams).enumerate()
    {
        if !matches!(layout.step_mode, VertexStepMode::Vertex) {
            return Err(ForwardPreparationError::InstanceStreamUnsupported {
                source,
                stream: stream_index,
            });
        }
        let attribute_end = layout
            .attributes
            .iter()
            .map(|attribute| {
                forward_format_size(attribute.format)
                    .and_then(|size| attribute.offset.checked_add(size))
            })
            .collect::<Option<Vec<_>>>()
            .and_then(|ends| ends.into_iter().max());
        let Some(attribute_end) = attribute_end else {
            continue;
        };
        let required_end = u64::from(geometry.vertices.end - 1)
            .checked_mul(layout.stride)
            .and_then(|offset| offset.checked_add(attribute_end))
            .and_then(|end| usize::try_from(end).ok())
            .ok_or(ForwardPreparationError::GeometryPayloadTooShort {
                source,
                stream: stream_index,
            })?;
        if required_end > bytes.len() {
            return Err(ForwardPreparationError::GeometryPayloadTooShort {
                source,
                stream: stream_index,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_material_schema(
    prepared: &PreparedMaterial,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    let bindings = &prepared.material().contribution().bindings;
    let tint = bindings.iter().filter(|binding| {
        binding.location.group == 0
            && binding.location.binding == 0
            && binding.kind == MaterialResourceKind::UniformVec4
    });
    let texture = bindings.iter().filter(|binding| {
        binding.location.group == 0
            && binding.location.binding == 1
            && binding.kind == MaterialResourceKind::Texture2dF32
    });
    if bindings.len() != 2 || tint.count() != 1 || texture.count() != 1 {
        return Err(ForwardPreparationError::MaterialBindingSchema { source });
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) enum ForwardTemplate {
    Position,
    PositionUvColor,
}

impl ForwardTemplate {
    pub(super) fn request(self) -> ShaderCompositionRequest {
        match self {
            Self::Position => ShaderCompositionRequest::new(vec![SourceUnit::new(
                "forward-template-position",
                POSITION_TEMPLATE,
            )])
            .with_entry_point(EntryPoint::new("forward_vs", ShaderStage::Vertex))
            .with_entry_point(EntryPoint::new("forward_fs", ShaderStage::Fragment))
            .with_binding(
                BindingDeclaration::new(1, 0, BindingKind::Uniform)
                    .with_visibility([ShaderStage::Vertex]),
            )
            .with_vertex_input(VertexInputDeclaration::new(
                "POSITION",
                0,
                VertexType::Vec3(ScalarType::F32),
            ))
            .with_static_option("forward.vertex_inputs", "position"),
            Self::PositionUvColor => ShaderCompositionRequest::new(vec![SourceUnit::new(
                "forward-template-position-uv-color",
                POSITION_UV_COLOR_TEMPLATE,
            )])
            .with_entry_point(EntryPoint::new("forward_vs", ShaderStage::Vertex))
            .with_entry_point(EntryPoint::new("forward_fs", ShaderStage::Fragment))
            .with_binding(
                BindingDeclaration::new(1, 0, BindingKind::Uniform)
                    .with_visibility([ShaderStage::Vertex]),
            )
            .with_vertex_input(VertexInputDeclaration::new(
                "POSITION",
                0,
                VertexType::Vec3(ScalarType::F32),
            ))
            .with_vertex_input(VertexInputDeclaration::new(
                "UV",
                1,
                VertexType::Vec2(ScalarType::F32),
            ))
            .with_vertex_input(VertexInputDeclaration::new(
                "COLOR",
                2,
                VertexType::Vec4(ScalarType::F32),
            ))
            .with_static_option("forward.vertex_inputs", "position-uv-color"),
        }
    }
}

pub(super) fn forward_template(
    layout: &GeometryLayout,
    source: ForwardDrawSource,
) -> Result<ForwardTemplate, ForwardPreparationError> {
    let position = layout
        .streams
        .iter()
        .flat_map(|stream| &stream.attributes)
        .any(|attribute| attribute.semantic == "POSITION");
    let uv = layout
        .streams
        .iter()
        .flat_map(|stream| &stream.attributes)
        .any(|attribute| attribute.semantic == "UV");
    let color = layout
        .streams
        .iter()
        .flat_map(|stream| &stream.attributes)
        .any(|attribute| attribute.semantic == "COLOR");
    if !position {
        return Err(ForwardPreparationError::MissingPosition { source });
    }
    match (uv, color) {
        (false, false) => Ok(ForwardTemplate::Position),
        (true, true) => Ok(ForwardTemplate::PositionUvColor),
        _ => Err(ForwardPreparationError::IncompleteOptionalAttributes { source }),
    }
}

pub(super) fn finite_transform(transform: RenderTransform) -> bool {
    transform
        .columns()
        .iter()
        .flatten()
        .all(|value| value.is_finite())
}

pub(super) fn multiply_columns(left: RenderTransform, right: RenderTransform) -> [[f32; 4]; 4] {
    let left = left.columns();
    let right = right.columns();
    let mut product = [[0.0; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            product[column][row] = (0..4)
                .map(|inner| left[inner][row] * right[column][inner])
                .sum();
        }
    }
    product
}

fn forward_format_size(format: VertexFormat) -> Option<u64> {
    let (bytes, components): (u64, u8) = match format {
        VertexFormat::Float32(n) | VertexFormat::Uint32(n) | VertexFormat::Sint32(n) => (4, n),
        VertexFormat::Float16(n)
        | VertexFormat::Uint16(n)
        | VertexFormat::Sint16(n)
        | VertexFormat::Unorm16(n)
        | VertexFormat::Snorm16(n) => (2, n),
        VertexFormat::Uint8(n)
        | VertexFormat::Sint8(n)
        | VertexFormat::Unorm8(n)
        | VertexFormat::Snorm8(n) => (1, n),
        VertexFormat::Float64(n) | VertexFormat::Uint64(n) => (8, n),
    };
    (1..=4)
        .contains(&components)
        .then_some(bytes * u64::from(components))
}

const POSITION_TEMPLATE: &str = r#"
@group(1) @binding(0) var<uniform> forward_clip_from_local: mat4x4<f32>;

struct ForwardVertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn forward_vs(@location(0) forward_position: vec3<f32>) -> ForwardVertexOutput {
    var output: ForwardVertexOutput;
    output.position = forward_clip_from_local * vec4<f32>(forward_position, 1.0);
    return output;
}

@fragment
fn forward_fs() -> @location(0) vec4<f32> {
    return evaluate_material();
}
"#;

const POSITION_UV_COLOR_TEMPLATE: &str = r#"
@group(1) @binding(0) var<uniform> forward_clip_from_local: mat4x4<f32>;

struct ForwardVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn forward_vs(
    @location(0) forward_position: vec3<f32>,
    @location(1) forward_uv: vec2<f32>,
    @location(2) forward_color: vec4<f32>,
) -> ForwardVertexOutput {
    var output: ForwardVertexOutput;
    output.position = forward_clip_from_local * vec4<f32>(forward_position, 1.0);
    output.uv = forward_uv;
    output.color = forward_color;
    return output;
}

@fragment
fn forward_fs(input: ForwardVertexOutput) -> @location(0) vec4<f32> {
    let uv_factor = input.uv.x * 0.0 + 1.0;
    return evaluate_material() * input.color * uv_factor;
}
"#;
