use std::collections::BTreeMap;

use crate::{
    FragmentInput, MaterialCodeNode, MaterialCompiler, MaterialDefinition, MaterialError,
    MaterialFragment, MaterialType, SurfaceOutput,
};
use fluxel_shader::{
    EntryPoint, ShaderComposer, ShaderCompositionRequest, ShaderError, ShaderStage, SourceUnit,
};

const PBR_TEMPLATE: &str = r#"
@fragment
fn pbr_fragment() -> @location(0) vec4<f32> {
    let base = evaluate_material();
    let properties = evaluate_metallic_roughness();
    return vec4<f32>(base.rgb * properties.x, base.a * properties.y);
}
"#;

fn fragment(name: &str, output: MaterialType, wgsl: &str) -> MaterialFragment {
    MaterialFragment {
        name: name.into(),
        inputs: Vec::new(),
        output,
        calls: Vec::new(),
        wgsl: wgsl.into(),
    }
}

fn pbr_definition(properties: &str) -> MaterialDefinition {
    MaterialDefinition::new(
        vec![fragment(
            "base_color",
            MaterialType::Vec4,
            "fn base_color() -> vec4<f32> { return vec4<f32>(0.25, 0.5, 0.75, 1.0); }",
        )],
        "base_color",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::MetallicRoughness {
            properties: properties.into(),
        },
    )
}

fn compose(variant: &crate::CompiledMaterialVariant) {
    let contribution = variant.contribution();
    let mut units = contribution.source_units();
    units.push(SourceUnit::new("pbr-fragment-template", PBR_TEMPLATE));
    let mut request = ShaderCompositionRequest::new(units)
        .with_entry_point(EntryPoint::new("pbr_fragment", ShaderStage::Fragment));
    request.bindings = contribution.binding_declarations();
    request.callables = contribution.callable_declarations();
    request.static_options = contribution.shader_static_options();
    ShaderComposer::new()
        .compose(&request)
        .expect("both PBR evaluation wrappers compose into the fragment entry point");
}

#[test]
fn metallic_roughness_emits_and_composes_both_evaluation_wrappers() {
    let definition = pbr_definition("surface_properties").with_code_node(MaterialCodeNode {
        name: "surface_properties".into(),
        inputs: Vec::new(),
        output: MaterialType::Vec2,
        calls: Vec::new(),
        source: SourceUnit::new(
            "author:surface-properties",
            "fn surface_properties() -> vec2<f32> { return vec2<f32>(0.2, 0.7); }",
        ),
        resources: Vec::new(),
    });
    let variant = MaterialCompiler::compile(&definition).expect("valid PBR material");

    assert!(matches!(
        definition.output(),
        SurfaceOutput::MetallicRoughness { properties } if properties == "surface_properties"
    ));
    assert!(matches!(
        variant.output(),
        SurfaceOutput::MetallicRoughness { .. }
    ));
    let units = variant.contribution().source_units();
    assert!(
        units[0]
            .source
            .contains("fn evaluate_metallic_roughness() -> vec2<f32>")
    );
    assert_eq!(units[1].id.0, "author:surface-properties");
    assert_eq!(
        units[1].source,
        "fn surface_properties() -> vec2<f32> { return vec2<f32>(0.2, 0.7); }"
    );
    compose(&variant);
}

#[test]
fn metallic_roughness_rejects_missing_wrong_and_argument_property_callables() {
    let missing = pbr_definition("not_declared");
    assert!(matches!(
        MaterialCompiler::compile(&missing),
        Err(MaterialError::MissingSurfaceProperties { properties }) if properties == "not_declared"
    ));

    let wrong_type = pbr_definition("wrong_type").with_code_node(MaterialCodeNode {
        name: "wrong_type".into(),
        inputs: Vec::new(),
        output: MaterialType::Vec3,
        calls: Vec::new(),
        source: SourceUnit::new(
            "author:wrong-type",
            "fn wrong_type() -> vec3<f32> { return vec3<f32>(0.0); }",
        ),
        resources: Vec::new(),
    });
    assert!(matches!(
        MaterialCompiler::compile(&wrong_type),
        Err(MaterialError::SurfacePropertiesTypeMismatch { properties, actual: MaterialType::Vec3 }) if properties == "wrong_type"
    ));

    let with_input = pbr_definition("takes_input").with_code_node(MaterialCodeNode {
        name: "takes_input".into(),
        inputs: vec![FragmentInput {
            name: "value".into(),
            ty: MaterialType::F32,
        }],
        output: MaterialType::Vec2,
        calls: Vec::new(),
        source: SourceUnit::new(
            "author:takes-input",
            "fn takes_input(value: f32) -> vec2<f32> { return vec2<f32>(value); }",
        ),
        resources: Vec::new(),
    });
    assert!(matches!(
        MaterialCompiler::compile(&with_input),
        Err(MaterialError::SurfacePropertiesInputsUnsupported { properties }) if properties == "takes_input"
    ));
}

#[test]
fn metallic_roughness_reserves_its_generated_evaluator_names() {
    let reserved_property =
        pbr_definition("evaluate_metallic_roughness").with_code_node(MaterialCodeNode {
            name: "evaluate_metallic_roughness".into(),
            inputs: Vec::new(),
            output: MaterialType::Vec2,
            calls: Vec::new(),
            source: SourceUnit::new(
                "author:reserved-properties",
                "fn evaluate_metallic_roughness() -> vec2<f32> { return vec2<f32>(0.0); }",
            ),
            resources: Vec::new(),
        });

    assert!(matches!(
        MaterialCompiler::compile(&reserved_property),
        Err(MaterialError::TopLevelNameConflict { name }) if name == "evaluate_metallic_roughness"
    ));
}

#[test]
fn metallic_roughness_property_parse_errors_keep_the_author_source_domain() {
    let definition = pbr_definition("surface_properties").with_code_node(MaterialCodeNode {
        name: "surface_properties".into(),
        inputs: Vec::new(),
        output: MaterialType::Vec2,
        calls: Vec::new(),
        source: SourceUnit::new(
            "author:broken-properties",
            "\nfn surface_properties() -> vec2<f32> { let = 0.0; return vec2<f32>(0.0); }",
        ),
        resources: Vec::new(),
    });
    let variant = MaterialCompiler::compile(&definition).expect("metadata remains structural");
    let contribution = variant.contribution();
    let mut units = contribution.source_units();
    units.push(SourceUnit::new("pbr-fragment-template", PBR_TEMPLATE));
    let mut request = ShaderCompositionRequest::new(units)
        .with_entry_point(EntryPoint::new("pbr_fragment", ShaderStage::Fragment));
    request.bindings = contribution.binding_declarations();
    request.callables = contribution.callable_declarations();
    request.static_options = contribution.shader_static_options();

    match ShaderComposer::new().compose(&request) {
        Err(ShaderError::Parse {
            source: Some(source),
            line: Some(line),
            ..
        }) => {
            assert_eq!(source.0, "author:broken-properties");
            assert_eq!(line, 2);
        }
        other => panic!("expected attributed property parse error, got {other:?}"),
    }
}

#[test]
fn surface_domain_and_property_callable_affect_only_pbr_variant_identity() {
    let color = MaterialDefinition::new(
        vec![fragment(
            "base_color",
            MaterialType::Vec4,
            "fn base_color() -> vec4<f32> { return vec4<f32>(1.0); }",
        )],
        "base_color",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::Color,
    );
    let properties = || {
        [
            MaterialCodeNode {
                name: "properties_a".into(),
                inputs: Vec::new(),
                output: MaterialType::Vec2,
                calls: Vec::new(),
                source: SourceUnit::new(
                    "author:properties-a",
                    "fn properties_a() -> vec2<f32> { return vec2<f32>(0.0); }",
                ),
                resources: Vec::new(),
            },
            MaterialCodeNode {
                name: "properties_b".into(),
                inputs: Vec::new(),
                output: MaterialType::Vec2,
                calls: Vec::new(),
                source: SourceUnit::new(
                    "author:properties-b",
                    "fn properties_b() -> vec2<f32> { return vec2<f32>(1.0); }",
                ),
                resources: Vec::new(),
            },
        ]
    };
    let [first_node_a, first_node_b] = properties();
    let first = pbr_definition("properties_a")
        .with_code_node(first_node_a)
        .with_code_node(first_node_b);
    let [second_node_a, second_node_b] = properties();
    let second = pbr_definition("properties_b")
        .with_code_node(second_node_a)
        .with_code_node(second_node_b);

    let color = MaterialCompiler::compile(&color).unwrap();
    let first = MaterialCompiler::compile(&first).unwrap();
    let second = MaterialCompiler::compile(&second).unwrap();
    assert_ne!(color.key(), first.key());
    assert_ne!(first.key(), second.key());
    assert!(
        !color
            .key()
            .as_str()
            .contains("surface-output:metallic-roughness")
    );
}
