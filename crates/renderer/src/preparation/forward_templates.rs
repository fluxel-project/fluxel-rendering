//! Pure Forward surface template constructors.
//!
//! These constructors contain no assets, Graph state, or preparation errors.
//! Geometry validation selects one of them in `forward_validation`; local WGSL
//! checks may reuse the exact same constructors without copying shader text.

use fluxel_shader::{
    BindingDeclaration, BindingKind, EntryPoint, ScalarType, ShaderCompositionRequest, ShaderStage,
    SourceUnit, VertexInputDeclaration, VertexType,
};

use super::forward_fragment::{ForwardFragmentInterface, with_ordinary_forward_fragment};

pub(crate) fn position_request() -> ShaderCompositionRequest {
    with_ordinary_forward_fragment(
        ShaderCompositionRequest::new(vec![
            SourceUnit::new("forward-template-position", POSITION_TEMPLATE),
            SourceUnit::new("forward-unlit-transform", UNLIT_TRANSFORM),
        ])
        .with_entry_point(EntryPoint::new("forward_vs", ShaderStage::Vertex))
        .with_entry_point(EntryPoint::new("forward_fs", ShaderStage::Fragment))
        .with_binding(
            BindingDeclaration::new(1, 0, BindingKind::Uniform)
                .with_visibility([ShaderStage::Vertex, ShaderStage::Fragment]),
        )
        .with_vertex_input(VertexInputDeclaration::new(
            "POSITION",
            0,
            VertexType::Vec3(ScalarType::F32),
        ))
        .with_static_option("forward.vertex_inputs", "position"),
        ForwardFragmentInterface::Position,
    )
}

pub(crate) fn position_uv0_request() -> ShaderCompositionRequest {
    with_ordinary_forward_fragment(
        ShaderCompositionRequest::new(vec![
            SourceUnit::new("forward-template-position-uv0", POSITION_UV0_TEMPLATE),
            SourceUnit::new("forward-unlit-transform", UNLIT_TRANSFORM),
        ])
        .with_entry_point(EntryPoint::new("forward_vs", ShaderStage::Vertex))
        .with_entry_point(EntryPoint::new("forward_fs", ShaderStage::Fragment))
        .with_binding(
            BindingDeclaration::new(1, 0, BindingKind::Uniform)
                .with_visibility([ShaderStage::Vertex, ShaderStage::Fragment]),
        )
        .with_vertex_input(VertexInputDeclaration::new(
            "POSITION",
            0,
            VertexType::Vec3(ScalarType::F32),
        ))
        .with_vertex_input(VertexInputDeclaration::new(
            "UV0",
            1,
            VertexType::Vec2(ScalarType::F32),
        ))
        .with_static_option("forward.vertex_inputs", "position-uv0"),
        ForwardFragmentInterface::PositionUv0,
    )
}

pub(crate) fn position_uv_color_request() -> ShaderCompositionRequest {
    with_ordinary_forward_fragment(
        ShaderCompositionRequest::new(vec![
            SourceUnit::new(
                "forward-template-position-uv-color",
                POSITION_UV_COLOR_TEMPLATE,
            ),
            SourceUnit::new("forward-unlit-transform", UNLIT_TRANSFORM),
        ])
        .with_entry_point(EntryPoint::new("forward_vs", ShaderStage::Vertex))
        .with_entry_point(EntryPoint::new("forward_fs", ShaderStage::Fragment))
        .with_binding(
            BindingDeclaration::new(1, 0, BindingKind::Uniform)
                .with_visibility([ShaderStage::Vertex, ShaderStage::Fragment]),
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
        ForwardFragmentInterface::PositionUvColor,
    )
}

const UNLIT_TRANSFORM: &str = r#"
struct ForwardTransform {
    clip_from_local: mat4x4<f32>,
    visibility: vec4<f32>,
};
@group(1) @binding(0) var<uniform> forward_transform: ForwardTransform;
"#;

const POSITION_TEMPLATE: &str = r#"
struct ForwardVertexOutput { @builtin(position) position: vec4<f32>, };
@vertex fn forward_vs(@location(0) forward_position: vec3<f32>) -> ForwardVertexOutput {
    var output: ForwardVertexOutput;
    output.position = forward_transform.clip_from_local * vec4<f32>(forward_position, 1.0);
    return output;
}
fn renderer_forward_shade() -> vec4<f32> {
    let material = evaluate_material();
    if material_alpha_rejected(material.a) { discard; }
    return vec4<f32>(material.rgb, material_output_alpha(material.a));
}
"#;

const POSITION_UV0_TEMPLATE: &str = r#"
struct ForwardUv0VertexOutput { @builtin(position) position: vec4<f32>, @location(0) uv0: vec2<f32>, };
@vertex fn forward_vs(@location(0) forward_position: vec3<f32>, @location(1) forward_uv0: vec2<f32>) -> ForwardUv0VertexOutput {
    var output: ForwardUv0VertexOutput;
    output.position = forward_transform.clip_from_local * vec4<f32>(forward_position, 1.0);
    output.uv0 = forward_uv0;
    return output;
}
fn renderer_forward_shade(input: ForwardUv0VertexOutput) -> vec4<f32> {
    let material = evaluate_material_uv0(input.uv0);
    if material_alpha_rejected(material.a) { discard; }
    return vec4<f32>(material.rgb, material_output_alpha(material.a));
}
"#;

const POSITION_UV_COLOR_TEMPLATE: &str = r#"
struct ForwardVertexOutput { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) color: vec4<f32>, };
@vertex fn forward_vs(@location(0) forward_position: vec3<f32>, @location(1) forward_uv: vec2<f32>, @location(2) forward_color: vec4<f32>) -> ForwardVertexOutput {
    var output: ForwardVertexOutput;
    output.position = forward_transform.clip_from_local * vec4<f32>(forward_position, 1.0);
    output.uv = forward_uv;
    output.color = forward_color;
    return output;
}
fn renderer_forward_shade(input: ForwardVertexOutput) -> vec4<f32> {
    let uv_factor = input.uv.x * 0.0 + 1.0;
    let material = evaluate_material() * input.color * uv_factor;
    if material_alpha_rejected(material.a) { discard; }
    return vec4<f32>(material.rgb, material_output_alpha(material.a));
}
"#;
