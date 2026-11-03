use std::sync::Arc;

use fluxel_shader::{
    EntryPoint, PipelineRequirements, ShaderComposer, ShaderCompositionRequest, ShaderStage,
    SourceUnit,
};

fn request(source: &str) -> ShaderCompositionRequest {
    ShaderCompositionRequest::new(vec![SourceUnit::new("unit", source)])
        .with_entry_point(EntryPoint::new("main", ShaderStage::Compute))
}

const COMPUTE: &str =
    "@compute @workgroup_size(1) fn main() {}\n@compute @workgroup_size(1) fn other() {}";

#[test]
fn equal_complete_requests_share_the_artifact() {
    let mut composer = ShaderComposer::new();
    let first = composer.compose(&request(COMPUTE)).unwrap();
    let second = composer.compose(&request(COMPUTE)).unwrap();
    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn every_logical_input_participates_in_cache_identity() {
    let mut composer = ShaderComposer::new();
    let base = composer.compose(&request(COMPUTE)).unwrap();
    let changed_source = composer
        .compose(&request("@compute @workgroup_size(2) fn main() {}"))
        .unwrap();
    let changed_entry = composer
        .compose(
            &ShaderCompositionRequest::new(vec![SourceUnit::new("unit", COMPUTE)])
                .with_entry_point(EntryPoint::new("other", ShaderStage::Compute)),
        )
        .unwrap();
    let changed_static = composer
        .compose(&request(COMPUTE).with_static_option("mode", "fast"))
        .unwrap();
    let mut requirements = PipelineRequirements::default();
    requirements.names.insert("storage-texture".into());
    let mut changed_requirements = request(COMPUTE);
    changed_requirements.requirements = requirements;
    let changed_requirements = composer.compose(&changed_requirements).unwrap();
    assert!(!Arc::ptr_eq(&base, &changed_source));
    assert!(!Arc::ptr_eq(&base, &changed_entry));
    assert!(!Arc::ptr_eq(&base, &changed_static));
    assert!(!Arc::ptr_eq(&base, &changed_requirements));
}
