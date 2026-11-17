use std::collections::BTreeMap;

use crate::{
    BindingLocation, FragmentInput, MaterialCodeNode, MaterialCompiler, MaterialDefinition,
    MaterialError, MaterialFragment, MaterialNode, MaterialNodeGraph, MaterialNodeId,
    MaterialNodeInput, MaterialNodeInputSource, MaterialResource, MaterialResourceKind,
    MaterialType, SurfaceOutput,
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

const UV0_PBR_TEMPLATE: &str = r#"
@fragment
fn pbr_uv0_fragment() -> @location(0) vec4<f32> {
    let uv = vec2<f32>(0.25, 0.75);
    let base = evaluate_material_uv0(uv);
    let properties = evaluate_metallic_roughness_uv0(uv);
    let normal = evaluate_normal_uv0(uv);
    return vec4<f32>(base.rgb * properties.x * normal.z, base.a * properties.y);
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
fn uv0_graph_lowers_texture_sample_and_composes_both_pbr_evaluators() {
    let definition = MaterialDefinition::new(
        Vec::new(),
        "legacy_unused_root",
        Vec::new(),
        vec![
            MaterialResource {
                name: "base_texture".into(),
                kind: MaterialResourceKind::Texture2dF32,
                location: BindingLocation { group: 0, binding: 0 },
            },
            MaterialResource {
                name: "base_sampler".into(),
                kind: MaterialResourceKind::Sampler { comparison: false },
                location: BindingLocation { group: 0, binding: 1 },
            },
        ],
        BTreeMap::new(),
        SurfaceOutput::MetallicRoughnessUv0Normal {
            properties: "surface_properties_uv0".into(),
            normal: "sample_normal".into(),
        },
    )
    .with_code_node(MaterialCodeNode {
        name: "scale_uv".into(),
        inputs: vec![FragmentInput {
            name: "uv".into(),
            ty: MaterialType::Vec2,
        }],
        output: MaterialType::Vec2,
        calls: Vec::new(),
        source: SourceUnit::new(
            "author:scale-uv",
            "fn scale_uv(uv: vec2<f32>) -> vec2<f32> { return uv * 0.5; }",
        ),
        resources: Vec::new(),
    })
    .with_code_node(MaterialCodeNode {
        name: "sample_base".into(),
        inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }],
        output: MaterialType::Vec4,
        calls: Vec::new(),
        source: SourceUnit::new(
            "author:sample-base-uv0",
            "fn sample_base(uv: vec2<f32>) -> vec4<f32> { return textureSample(base_texture, base_sampler, uv); }",
        ),
        resources: Vec::new(),
    })
    .with_code_node(MaterialCodeNode {
        name: "surface_properties_uv0".into(),
        inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }],
        output: MaterialType::Vec2,
        calls: Vec::new(),
        source: SourceUnit::new(
            "author:properties-uv0",
            "fn surface_properties_uv0(uv: vec2<f32>) -> vec2<f32> { return vec2<f32>(uv.x, uv.y); }",
        ),
        resources: Vec::new(),
    })
    .with_code_node(MaterialCodeNode {
        name: "sample_normal".into(),
        inputs: vec![FragmentInput {
            name: "uv".into(),
            ty: MaterialType::Vec2,
        }],
        output: MaterialType::Vec3,
        calls: Vec::new(),
        source: SourceUnit::new(
            "author:sample-normal",
            "fn sample_normal(uv: vec2<f32>) -> vec3<f32> { return textureSample(base_texture, base_sampler, uv).xyz; }",
        ),
        resources: Vec::new(),
    })
    .with_node_graph(MaterialNodeGraph {
        nodes: vec![
            MaterialNode {
                id: MaterialNodeId(6),
                callable: "scale_uv".into(),
                inputs: vec![MaterialNodeInput {
                    input: "uv".into(),
                    source: MaterialNodeInputSource::Uv0,
                }],
            },
            MaterialNode {
                id: MaterialNodeId(7),
                callable: "sample_base".into(),
                inputs: vec![MaterialNodeInput {
                    input: "uv".into(),
                    source: MaterialNodeInputSource::Node(MaterialNodeId(6)),
                }],
            },
        ],
        output: MaterialNodeId(7),
    });
    let variant = MaterialCompiler::compile(&definition).expect("valid UV0 sampled PBR graph");
    let contribution = variant.contribution();
    assert!(
        contribution
            .source()
            .contains("fn fluxel_material_node_7(uv: vec2<f32>)")
    );
    assert!(contribution.source().contains("return scale_uv(uv);"));
    assert!(
        contribution
            .source()
            .contains("return sample_base(fluxel_material_node_6(uv));")
    );
    assert!(
        contribution
            .source()
            .contains("fn evaluate_material_uv0(uv: vec2<f32>)")
    );
    assert!(
        contribution
            .source()
            .contains("fn evaluate_normal_uv0(uv: vec2<f32>)")
    );

    let mut units = contribution.source_units();
    units.push(SourceUnit::new(
        "pbr-uv0-fragment-template",
        UV0_PBR_TEMPLATE,
    ));
    let mut request = ShaderCompositionRequest::new(units)
        .with_entry_point(EntryPoint::new("pbr_uv0_fragment", ShaderStage::Fragment));
    request.bindings = contribution.binding_declarations();
    request.callables = contribution.callable_declarations();
    ShaderComposer::new()
        .compose(&request)
        .expect("UV0 node graph and texture sample compose");
}

#[test]
fn tangent_space_normal_surface_rejects_wrong_and_argument_normal_callables() {
    let output = SurfaceOutput::MetallicRoughnessUv0Normal {
        properties: "properties".into(),
        normal: "normal".into(),
    };
    let definition = |normal: MaterialCodeNode| {
        MaterialDefinition::new(
            Vec::new(),
            "base",
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
            output.clone(),
        )
        .with_code_node(MaterialCodeNode {
            name: "base".into(),
            inputs: vec![FragmentInput {
                name: "uv".into(),
                ty: MaterialType::Vec2,
            }],
            output: MaterialType::Vec4,
            calls: Vec::new(),
            source: SourceUnit::new(
                "author:normal-base",
                "fn base(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(uv, 0.0, 1.0); }",
            ),
            resources: Vec::new(),
        })
        .with_code_node(MaterialCodeNode {
            name: "properties".into(),
            inputs: vec![FragmentInput {
                name: "uv".into(),
                ty: MaterialType::Vec2,
            }],
            output: MaterialType::Vec2,
            calls: Vec::new(),
            source: SourceUnit::new(
                "author:normal-properties",
                "fn properties(uv: vec2<f32>) -> vec2<f32> { return uv; }",
            ),
            resources: Vec::new(),
        })
        .with_code_node(normal)
    };
    let wrong_type = definition(MaterialCodeNode {
        name: "normal".into(),
        inputs: vec![FragmentInput {
            name: "uv".into(),
            ty: MaterialType::Vec2,
        }],
        output: MaterialType::Vec4,
        calls: Vec::new(),
        source: SourceUnit::new(
            "author:normal-wrong-type",
            "fn normal(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(uv, 0.0, 1.0); }",
        ),
        resources: Vec::new(),
    });
    assert!(matches!(
        MaterialCompiler::compile(&wrong_type),
        Err(MaterialError::SurfaceNormalTypeMismatch { normal, actual: MaterialType::Vec4 }) if normal == "normal"
    ));
    let without_uv = definition(MaterialCodeNode {
        name: "normal".into(),
        inputs: Vec::new(),
        output: MaterialType::Vec3,
        calls: Vec::new(),
        source: SourceUnit::new(
            "author:normal-without-uv",
            "fn normal() -> vec3<f32> { return vec3<f32>(0.0, 0.0, 1.0); }",
        ),
        resources: Vec::new(),
    });
    assert!(matches!(
        MaterialCompiler::compile(&without_uv),
        Err(MaterialError::SurfaceNormalInputsUnsupported { normal }) if normal == "normal"
    ));
}

#[test]
fn uv0_surface_rejects_roots_and_properties_without_exactly_one_vec2_input() {
    let root_without_uv = MaterialDefinition::new(
        vec![fragment(
            "base",
            MaterialType::Vec4,
            "fn base() -> vec4<f32> { return vec4<f32>(1.0); }",
        )],
        "base",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::MetallicRoughnessUv0 {
            properties: "properties".into(),
        },
    )
    .with_code_node(MaterialCodeNode {
        name: "properties".into(),
        inputs: vec![FragmentInput {
            name: "uv".into(),
            ty: MaterialType::Vec2,
        }],
        output: MaterialType::Vec2,
        calls: Vec::new(),
        source: SourceUnit::new(
            "author:valid-properties",
            "fn properties(uv: vec2<f32>) -> vec2<f32> { return uv; }",
        ),
        resources: Vec::new(),
    });
    assert!(matches!(
        MaterialCompiler::compile(&root_without_uv),
        Err(MaterialError::RootInputsUnsupported { fragment }) if fragment == "base"
    ));

    let properties_without_uv = MaterialDefinition::new(
        Vec::new(),
        "base",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::MetallicRoughnessUv0 {
            properties: "properties".into(),
        },
    )
    .with_code_node(MaterialCodeNode {
        name: "base".into(),
        inputs: vec![FragmentInput {
            name: "uv".into(),
            ty: MaterialType::Vec2,
        }],
        output: MaterialType::Vec4,
        calls: Vec::new(),
        source: SourceUnit::new(
            "author:valid-base",
            "fn base(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(uv, 0.0, 1.0); }",
        ),
        resources: Vec::new(),
    })
    .with_code_node(MaterialCodeNode {
        name: "properties".into(),
        inputs: Vec::new(),
        output: MaterialType::Vec2,
        calls: Vec::new(),
        source: SourceUnit::new(
            "author:invalid-properties",
            "fn properties() -> vec2<f32> { return vec2<f32>(0.0); }",
        ),
        resources: Vec::new(),
    });
    assert!(matches!(
        MaterialCompiler::compile(&properties_without_uv),
        Err(MaterialError::SurfacePropertiesInputsUnsupported { properties }) if properties == "properties"
    ));
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
