//! Source attribution and callable signatures are part of the public contract.

use fluxel_shader::{
    CallableDeclaration, EntryPoint, ScalarType, ShaderComposer, ShaderCompositionRequest,
    ShaderError, ShaderStage, SourceUnit, VertexType,
};

#[test]
fn parse_failure_identifies_the_original_source_unit_and_line() {
    let request = ShaderCompositionRequest::new(vec![
        SourceUnit::new("shared", "const VALUE: u32 = 1u;"),
        SourceUnit::new(
            "broken",
            "\n@compute @workgroup_size(1) fn main() { let = 3u; }",
        ),
    ])
    .with_entry_point(EntryPoint::new("main", ShaderStage::Compute));
    match ShaderComposer::new().compose(&request) {
        Err(ShaderError::Parse {
            source: Some(source),
            line: Some(line),
            ..
        }) => {
            assert_eq!(source.0, "broken");
            assert_eq!(line, 2);
        }
        _ => panic!("parse failure must identify its contributing source and line"),
    }
}

#[test]
fn declared_callable_signature_is_checked_against_actual_wgsl() {
    let source = "fn shade(value: vec2<f32>) -> vec4<f32> { return vec4<f32>(value, 0.0, 1.0); }\n@compute @workgroup_size(1) fn main() { _ = shade(vec2<f32>(0.0)); }";
    let request = ShaderCompositionRequest::new(vec![SourceUnit::new("callable", source)])
        .with_entry_point(EntryPoint::new("main", ShaderStage::Compute))
        .with_callable(CallableDeclaration {
            name: "shade".into(),
            arguments: vec![VertexType::Vec2(ScalarType::F32)],
            result: Some(VertexType::Vec4(ScalarType::F32)),
        });
    let artifact = ShaderComposer::new().compose(&request).unwrap();
    assert_eq!(artifact.source_map()[0].source.0, "callable");
    let mut incorrect = request;
    incorrect.callables[0].result = Some(VertexType::Scalar(ScalarType::F32));
    assert!(matches!(
        ShaderComposer::new().compose(&incorrect),
        Err(ShaderError::CallableMismatch { name }) if name == "shade"
    ));
}
