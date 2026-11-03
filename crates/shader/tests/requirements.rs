use fluxel_shader::{
    EntryPoint, NagaCapability, ShaderComposer, ShaderCompositionRequest, ShaderStage, SourceUnit,
};

#[test]
fn f16_demand_is_exposed_for_consumer_admission() {
    let source =
        "enable f16; @compute @workgroup_size(1) fn main() { let value: f16 = 1.0h; _ = value; }";
    let request = ShaderCompositionRequest::new(vec![SourceUnit::new("f16", source)])
        .with_entry_point(EntryPoint::new("main", ShaderStage::Compute));
    let artifact = ShaderComposer::new().compose(&request).unwrap();
    assert!(
        artifact
            .requirements()
            .naga_capabilities
            .contains(&NagaCapability::ShaderFloat16)
    );
    assert!(
        artifact
            .requirements()
            .naga_capabilities
            .iter()
            .any(|capability| capability.name() == "shader-float16")
    );
}

#[test]
fn primitive_index_is_inferred_without_discarding_explicit_requirements() {
    let source = "@fragment fn main(@builtin(primitive_index) index: u32) -> @location(0) vec4<f32> { return vec4<f32>(f32(index)); }";
    let mut request = ShaderCompositionRequest::new(vec![SourceUnit::new("primitive", source)])
        .with_entry_point(EntryPoint::new("main", ShaderStage::Fragment));
    request
        .requirements
        .names
        .insert("consumer-requirement".into());
    let artifact = ShaderComposer::new().compose(&request).unwrap();
    assert!(artifact.requirements().names.contains("primitive-index"));
    assert!(
        artifact
            .requirements()
            .naga_capabilities
            .contains(&NagaCapability::PrimitiveIndex)
    );
    assert!(
        artifact
            .requirements()
            .names
            .contains("consumer-requirement")
    );
}

#[test]
fn explicit_capability_requirement_is_not_replaced_by_inference() {
    let source = "@compute @workgroup_size(1) fn main() {}";
    let mut request = ShaderCompositionRequest::new(vec![SourceUnit::new("plain", source)])
        .with_entry_point(EntryPoint::new("main", ShaderStage::Compute));
    request
        .requirements
        .naga_capabilities
        .insert(NagaCapability::PushConstant);
    let artifact = ShaderComposer::new().compose(&request).unwrap();
    assert!(
        artifact
            .requirements()
            .naga_capabilities
            .contains(&NagaCapability::PushConstant)
    );
}
