//! Public Shader-to-RHI feature requirement contract tests.

use fluxel_rhi::api::{SHADER_FEATURE_EARLY_DEPTH_TEST, SHADER_FEATURE_PRIMITIVE_INDEX};
use fluxel_shader::{
    EntryPoint, ShaderComposer, ShaderCompositionRequest, ShaderStage, SourceUnit,
};

fn compose(id: &str, source: &str, entry: &str) -> std::sync::Arc<fluxel_shader::ShaderArtifact> {
    let mut request = ShaderCompositionRequest::new(vec![SourceUnit::new(id, source)])
        .with_entry_point(EntryPoint::new(entry, ShaderStage::Fragment));
    request
        .requirements
        .names
        .insert("caller-must-remain".into());
    ShaderComposer::new().compose(&request).unwrap()
}

#[test]
fn primitive_index_inference_uses_the_selected_rhi_feature_name_and_keeps_caller_requirement() {
    let artifact = compose(
        "primitive-index",
        "@fragment fn main(@builtin(primitive_index) index: u32) -> @location(0) vec4<f32> { return vec4<f32>(f32(index)); }",
        "main",
    );
    assert_eq!(SHADER_FEATURE_PRIMITIVE_INDEX, "primitive-index");
    assert!(
        artifact
            .requirements()
            .names
            .contains(SHADER_FEATURE_PRIMITIVE_INDEX)
    );
    assert!(artifact.requirements().names.contains("caller-must-remain"));
}

#[test]
fn early_depth_test_inference_uses_the_selected_rhi_feature_name_and_keeps_caller_requirement() {
    let artifact = compose(
        "early-depth",
        "@early_depth_test(less_equal) @fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
        "main",
    );
    assert_eq!(SHADER_FEATURE_EARLY_DEPTH_TEST, "shader-early-depth-test");
    assert!(
        artifact
            .requirements()
            .names
            .contains(SHADER_FEATURE_EARLY_DEPTH_TEST)
    );
    assert!(artifact.requirements().names.contains("caller-must-remain"));
}
