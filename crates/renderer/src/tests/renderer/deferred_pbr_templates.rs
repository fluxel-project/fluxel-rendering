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
