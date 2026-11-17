//! PBR Deferred request-shape tests without backend realization.

use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::{ShaderComposer, ShaderStage, SourceUnit};

use crate::{GeometryAttribute, GeometryLayout, GeometryStream};

fn pbr_layout() -> GeometryLayout {
    GeometryLayout {
        streams: vec![GeometryStream {
            stride: 24,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                GeometryAttribute {
                    semantic: "POSITION".into(),
                    offset: 0,
                    format: VertexFormat::Float32(3),
                },
                GeometryAttribute {
                    semantic: "NORMAL".into(),
                    offset: 12,
                    format: VertexFormat::Float32(3),
                },
            ],
        }],
    }
}

fn pbr_uv0_layout() -> GeometryLayout {
    let mut layout = pbr_layout();
    layout.streams[0].stride = 32;
    layout.streams[0].attributes.push(GeometryAttribute {
        semantic: "UV0".into(),
        offset: 24,
        format: VertexFormat::Float32(2),
    });
    layout
}

fn pbr_uv0_normal_layout() -> GeometryLayout {
    let mut layout = pbr_uv0_layout();
    layout.streams[0].stride = 48;
    layout.streams[0].attributes.push(GeometryAttribute {
        semantic: "TANGENT".into(),
        offset: 32,
        format: VertexFormat::Float32(4),
    });
    layout
}

fn add_material_evaluators(
    mut request: fluxel_shader::ShaderCompositionRequest,
) -> fluxel_shader::ShaderCompositionRequest {
    request.source_units.push(SourceUnit::new(
        "pbr-template-test-material",
        "fn evaluate_material() -> vec4<f32> { return vec4<f32>(0.8, 0.6, 0.4, 1.0); }\nfn evaluate_metallic_roughness() -> vec2<f32> { return vec2<f32>(0.2, 0.7); }",
    ));
    request
}

#[test]
fn pbr_deferred_gbuffer_request_requires_position_and_normal() {
    let request = crate::pbr::gbuffer_request(&pbr_layout()).expect("PBR layout is supported");
    assert_eq!(request.vertex_inputs.len(), 2);
    assert_eq!(request.vertex_inputs[0].semantic, "POSITION");
    assert_eq!(request.vertex_inputs[1].semantic, "NORMAL");
    let artifact = ShaderComposer::new()
        .compose(&add_material_evaluators(request))
        .expect("PBR G-buffer request composes");
    assert!(artifact.entry_point(ShaderStage::Vertex).is_some());
    assert!(artifact.entry_point(ShaderStage::Fragment).is_some());
}

#[test]
fn pbr_deferred_lighting_request_composes_with_three_gbuffer_inputs() {
    let request = crate::pbr::lighting_request();
    assert_eq!(request.bindings.len(), 3);
    let artifact = ShaderComposer::new()
        .compose(&request)
        .expect("PBR lighting request composes");
    assert!(artifact.entry_point(ShaderStage::Vertex).is_some());
    assert!(artifact.entry_point(ShaderStage::Fragment).is_some());
}

#[test]
fn pbr_uv0_forward_and_deferred_requests_pass_uv0_to_material_evaluators() {
    for request in [
        crate::pbr::forward_uv0_request(&pbr_uv0_layout()),
        crate::pbr::gbuffer_uv0_request(&pbr_uv0_layout()),
    ] {
        let mut request = request.expect("POSITION/NORMAL/UV0 layout");
        request.source_units.push(SourceUnit::new(
            "pbr-uv0-material",
            "fn evaluate_material_uv0(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(uv, 0.5, 1.0); }\nfn evaluate_metallic_roughness_uv0(uv: vec2<f32>) -> vec2<f32> { return uv; }",
        ));
        let artifact = ShaderComposer::new()
            .compose(&request)
            .expect("UV0 PBR composes");
        assert!(artifact.entry_point(ShaderStage::Vertex).is_some());
        assert_eq!(request.vertex_inputs[2].semantic, "UV0");
        assert_eq!(request.vertex_inputs[2].location, 2);
    }
}

#[test]
fn pbr_uv0_request_refuses_missing_or_wrong_uv0_semantic() {
    assert!(crate::pbr::forward_uv0_request(&pbr_layout()).is_none());
    let mut wrong = pbr_uv0_layout();
    wrong.streams[0].attributes[2].format = VertexFormat::Float32(3);
    assert!(crate::pbr::gbuffer_uv0_request(&wrong).is_none());
}

#[test]
fn pbr_uv0_normal_requests_require_tangent_and_compose_tbn_path() {
    for request in [
        crate::pbr::forward_uv0_normal_request(&pbr_uv0_normal_layout()),
        crate::pbr::gbuffer_uv0_normal_request(&pbr_uv0_normal_layout()),
    ] {
        let mut request = request.expect("POSITION/NORMAL/UV0/TANGENT layout");
        request.source_units.push(SourceUnit::new(
            "pbr-uv0-normal-material",
            "fn evaluate_material_uv0(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(uv, 0.5, 1.0); }\nfn evaluate_metallic_roughness_uv0(uv: vec2<f32>) -> vec2<f32> { return uv; }\nfn evaluate_normal_uv0(uv: vec2<f32>) -> vec3<f32> { return vec3<f32>(uv, 1.0); }",
        ));
        ShaderComposer::new()
            .compose(&request)
            .expect("normal-map PBR composes");
        assert_eq!(request.vertex_inputs[3].semantic, "TANGENT");
        assert_eq!(request.vertex_inputs[3].location, 3);
    }
    let mut missing = pbr_uv0_normal_layout();
    missing.streams[0].attributes.pop();
    assert!(crate::pbr::forward_uv0_normal_request(&missing).is_none());
    let mut wrong = pbr_uv0_normal_layout();
    wrong.streams[0].attributes[3].format = VertexFormat::Float32(3);
    assert!(crate::pbr::gbuffer_uv0_normal_request(&wrong).is_none());
}
