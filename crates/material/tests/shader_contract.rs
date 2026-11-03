//! Public Material-to-Shader contract tests.

use std::collections::BTreeMap;

use fluxel_material::{
    BindingLocation, FragmentCall, MaterialCompiler, MaterialDefinition, MaterialError,
    MaterialFragment, MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType,
    StaticValue, SurfaceOutput,
};
use fluxel_shader::{
    BindingKind, EntryPoint, ShaderComposer, ShaderCompositionRequest, ShaderError, ShaderStage,
    SourceUnit,
};

const FRAGMENT_TEMPLATE: &str = r#"
@fragment
fn material_fragment() -> @location(0) vec4<f32> {
    return evaluate_material();
}
"#;

fn material_definition_with_static(root_wgsl: &str, enabled: bool) -> MaterialDefinition {
    let static_options = BTreeMap::from([("MATERIAL_ENABLED".into(), StaticValue::Bool(enabled))]);
    MaterialDefinition::new(
        vec![
            MaterialFragment {
                name: "sample_texel".into(),
                inputs: Vec::new(),
                output: MaterialType::Vec4,
                calls: Vec::new(),
                wgsl: "fn sample_texel() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0); }".into(),
            },
            MaterialFragment {
                name: "custom_tint".into(),
                inputs: Vec::new(),
                output: MaterialType::Vec4,
                calls: vec![FragmentCall {
                    fragment: "sample_texel".into(),
                    arguments: Vec::new(),
                }],
                wgsl: root_wgsl.into(),
            },
        ],
        "custom_tint",
        vec![MaterialParameter {
            name: "dynamic_tint".into(),
            ty: MaterialType::Vec4,
            resource: "material_tint".into(),
        }],
        vec![
            MaterialResource {
                name: "material_tint".into(),
                kind: MaterialResourceKind::UniformVec4,
                location: BindingLocation {
                    group: 0,
                    binding: 0,
                },
            },
            MaterialResource {
                name: "material_texture".into(),
                kind: MaterialResourceKind::Texture2dF32,
                location: BindingLocation {
                    group: 0,
                    binding: 1,
                },
            },
        ],
        static_options,
        SurfaceOutput::Color,
    )
}

fn material_definition(root_wgsl: &str) -> MaterialDefinition {
    material_definition_with_static(root_wgsl, true)
}

fn compose(
    contribution: &fluxel_material::MaterialShaderContribution,
) -> Result<std::sync::Arc<fluxel_shader::ShaderArtifact>, ShaderError> {
    let mut request = ShaderCompositionRequest::new(vec![
        contribution.source_unit(),
        SourceUnit::new("material-fragment-template", FRAGMENT_TEMPLATE),
    ])
    .with_entry_point(EntryPoint::new("material_fragment", ShaderStage::Fragment));
    request.bindings = contribution.binding_declarations();
    request.callables = contribution.callable_declarations();
    request.static_options = contribution.shader_static_options();
    ShaderComposer::new().compose(&request)
}

#[test]
fn multi_fragment_dynamic_uniform_and_texture_contribution_reflects_through_shader() {
    let variant = MaterialCompiler::compile(&material_definition(
        "fn custom_tint() -> vec4<f32> { return sample_texel() * dynamic_tint; }",
    ))
    .expect("authoring metadata is valid");

    let artifact =
        compose(variant.contribution()).expect("complete Material contribution is Naga-valid");
    assert_eq!(
        artifact.entry_point(ShaderStage::Fragment).unwrap().name,
        "material_fragment"
    );
    assert_eq!(artifact.interface().bindings().len(), 2);
    assert!(artifact.interface().bindings().iter().any(|binding| {
        binding.group == 0 && binding.binding == 0 && binding.kind == BindingKind::Uniform
    }));
    assert!(artifact.interface().bindings().iter().any(|binding| {
        binding.group == 0 && binding.binding == 1 && binding.kind == BindingKind::Texture
    }));
}

#[test]
fn pass_template_can_precede_the_material_contribution() {
    let variant = MaterialCompiler::compile(&material_definition(
        "fn custom_tint() -> vec4<f32> { return sample_texel() * dynamic_tint; }",
    ))
    .unwrap();
    let contribution = variant.contribution();
    let mut request = ShaderCompositionRequest::new(vec![
        SourceUnit::new("pass-template", FRAGMENT_TEMPLATE),
        contribution.source_unit(),
    ])
    .with_entry_point(EntryPoint::new("material_fragment", ShaderStage::Fragment));
    request.bindings = contribution.binding_declarations();
    request.callables = contribution.callable_declarations();
    request.static_options = contribution.shader_static_options();
    assert!(ShaderComposer::new().compose(&request).is_ok());
}

#[test]
fn function_and_resource_contract_disagreements_fail_structurally() {
    let invalid_signature = MaterialCompiler::compile(&material_definition(
        "fn custom_tint() -> f32 { return 1.0; }",
    ))
    .expect("Material metadata alone does not replace Shader validation");
    assert!(matches!(
        compose(invalid_signature.contribution()),
        Err(ShaderError::Parse { .. } | ShaderError::Validate { .. })
    ));

    let metadata_mismatch = MaterialDefinition::new(
        vec![
            MaterialFragment {
                name: "root".into(),
                inputs: Vec::new(),
                output: MaterialType::Vec4,
                calls: Vec::new(),
                wgsl: "fn root() -> vec4<f32> { return vec4<f32>(1.0); }".into(),
            },
            MaterialFragment {
                name: "unused".into(),
                inputs: vec![fluxel_material::FragmentInput {
                    name: "v".into(),
                    ty: MaterialType::F32,
                }],
                output: MaterialType::Vec4,
                calls: Vec::new(),
                wgsl: "fn unused() -> vec4<f32> { return vec4<f32>(1.0); }".into(),
            },
        ],
        "root",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::Color,
    );
    let mismatched = MaterialCompiler::compile(&metadata_mismatch).unwrap();
    assert!(
        matches!(compose(mismatched.contribution()), Err(ShaderError::CallableMismatch { name }) if name == "unused")
    );

    let valid = MaterialCompiler::compile(&material_definition(
        "fn custom_tint() -> vec4<f32> { return sample_texel() * dynamic_tint; }",
    ))
    .unwrap();
    let mut wrong_kind = valid.contribution().clone();
    wrong_kind
        .bindings
        .iter_mut()
        .find(|binding| binding.location.group == 0 && binding.location.binding == 1)
        .unwrap()
        .kind = MaterialResourceKind::UniformVec4;
    assert!(matches!(
        compose(&wrong_kind),
        Err(ShaderError::BindingMismatch {
            group: 0,
            binding: 1
        })
    ));

    let mut extra_binding = valid.contribution().clone();
    extra_binding
        .bindings
        .push(fluxel_material::MaterialBindingDeclaration {
            name: "undeclared".into(),
            location: BindingLocation {
                group: 0,
                binding: 9,
            },
            kind: MaterialResourceKind::Texture2dF32,
        });
    assert!(matches!(
        compose(&extra_binding),
        Err(ShaderError::BindingMismatch {
            group: 0,
            binding: 9
        })
    ));
}

#[test]
fn authoring_call_graph_failures_are_not_deferred_to_shader_text() {
    let missing = MaterialDefinition::new(
        Vec::new(),
        "not_present",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::Color,
    );
    assert!(matches!(
        MaterialCompiler::compile(&missing),
        Err(MaterialError::MissingFragment { .. })
    ));

    let invalid_call = MaterialDefinition::new(
        vec![MaterialFragment {
            name: "root".into(),
            inputs: Vec::new(),
            output: MaterialType::Vec4,
            calls: vec![FragmentCall {
                fragment: "not_present".into(),
                arguments: Vec::new(),
            }],
            wgsl: "fn root() -> vec4<f32> { return vec4<f32>(); }".into(),
        }],
        "root",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::Color,
    );
    assert!(matches!(
        MaterialCompiler::compile(&invalid_call),
        Err(MaterialError::InvalidCall { .. })
    ));

    let cycle = MaterialDefinition::new(
        vec![
            MaterialFragment {
                name: "first".into(),
                inputs: Vec::new(),
                output: MaterialType::Vec4,
                calls: vec![FragmentCall {
                    fragment: "second".into(),
                    arguments: Vec::new(),
                }],
                wgsl: "fn first() -> vec4<f32> { return second(); }".into(),
            },
            MaterialFragment {
                name: "second".into(),
                inputs: Vec::new(),
                output: MaterialType::Vec4,
                calls: vec![FragmentCall {
                    fragment: "first".into(),
                    arguments: Vec::new(),
                }],
                wgsl: "fn second() -> vec4<f32> { return first(); }".into(),
            },
        ],
        "first",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::Color,
    );
    assert!(matches!(
        MaterialCompiler::compile(&cycle),
        Err(MaterialError::CallCycle { .. })
    ));
}

#[test]
fn static_options_change_emitted_wgsl_and_shader_identity() {
    let body = "fn custom_tint() -> vec4<f32> { if (MATERIAL_ENABLED) { return sample_texel() * dynamic_tint; } return vec4<f32>(0.0); }";
    let enabled = MaterialCompiler::compile(&material_definition_with_static(body, true)).unwrap();
    let disabled =
        MaterialCompiler::compile(&material_definition_with_static(body, false)).unwrap();
    assert_ne!(enabled.key(), disabled.key());
    assert_ne!(
        enabled.contribution().source,
        disabled.contribution().source
    );
    let enabled_artifact = compose(enabled.contribution()).unwrap();
    let disabled_artifact = compose(disabled.contribution()).unwrap();
    assert_ne!(enabled_artifact.cache_key(), disabled_artifact.cache_key());
    assert!(
        enabled
            .contribution()
            .source
            .contains("const MATERIAL_ENABLED: bool = true;")
    );
    assert!(
        disabled
            .contribution()
            .source
            .contains("const MATERIAL_ENABLED: bool = false;")
    );
}
