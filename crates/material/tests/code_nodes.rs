//! Public Code Node material-to-Shader contract tests.

use std::collections::BTreeMap;

use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, DynamicValue, FragmentCall, FragmentInput, MaterialAsset, MaterialAssetRef,
    MaterialCodeNode, MaterialCompiler, MaterialDefinition, MaterialError, MaterialFragment,
    MaterialInstance, MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType,
    StaticValue, SurfaceOutput,
};
use fluxel_shader::{
    EntryPoint, ShaderComposer, ShaderCompositionRequest, ShaderError, ShaderStage, SourceUnit,
};

const TEMPLATE: &str = r#"
@fragment
fn material_fragment() -> @location(0) vec4<f32> { return evaluate_material(); }
"#;

fn tint_resource() -> MaterialResource {
    MaterialResource {
        name: "material_tint".into(),
        kind: MaterialResourceKind::UniformVec4,
        location: BindingLocation {
            group: 0,
            binding: 0,
        },
    }
}

fn node(id: &str, source: &str) -> MaterialCodeNode {
    MaterialCodeNode {
        name: "apply_node".into(),
        inputs: vec![FragmentInput {
            name: "color".into(),
            ty: MaterialType::Vec4,
        }],
        output: MaterialType::Vec4,
        calls: Vec::new(),
        source: SourceUnit::new(id, source),
        resources: vec![tint_resource()],
    }
}

fn definition(node: MaterialCodeNode, enabled: bool) -> MaterialDefinition {
    MaterialDefinition::new(
        vec![
            MaterialFragment {
                name: "legacy_color".into(),
                inputs: Vec::new(),
                output: MaterialType::Vec4,
                calls: Vec::new(),
                wgsl: "fn legacy_color() -> vec4<f32> { return vec4<f32>(0.25, 0.5, 0.75, 1.0); }".into(),
            },
            MaterialFragment {
                name: "root".into(),
                inputs: Vec::new(),
                output: MaterialType::Vec4,
                calls: vec![
                    FragmentCall { fragment: "legacy_color".into(), arguments: Vec::new() },
                    FragmentCall { fragment: "apply_node".into(), arguments: vec![MaterialType::Vec4] },
                ],
                wgsl: "fn root() -> vec4<f32> { if (MATERIAL_ENABLED) { return apply_node(legacy_color()); } return vec4<f32>(0.0); }".into(),
            },
        ],
        "root",
        vec![MaterialParameter {
            name: "tint".into(),
            ty: MaterialType::Vec4,
            resource: "material_tint".into(),
        }],
        vec![tint_resource()],
        BTreeMap::from([("MATERIAL_ENABLED".into(), StaticValue::Bool(enabled))]),
        SurfaceOutput::Color,
    )
    .with_code_node(node)
}

fn compose(contribution: &fluxel_material::MaterialShaderContribution) -> Result<(), ShaderError> {
    let mut units = contribution.source_units();
    units.push(SourceUnit::new("code-node-test-template", TEMPLATE));
    let mut request = ShaderCompositionRequest::new(units)
        .with_entry_point(EntryPoint::new("material_fragment", ShaderStage::Fragment));
    request.bindings = contribution.binding_declarations();
    request.callables = contribution.callable_declarations();
    request.static_options = contribution.shader_static_options();
    ShaderComposer::new().compose(&request).map(|_| ())
}

#[test]
fn typed_code_node_and_legacy_adapter_compose_with_exact_shared_resource() {
    let definition = definition(
        node(
            "node:apply",
            "fn apply_node(color: vec4<f32>) -> vec4<f32> { return color * tint; }",
        ),
        true,
    );
    let compiled = MaterialCompiler::compile(&definition).expect("typed node validates");
    let contribution = compiled.contribution();
    let units = contribution.source_units();

    assert_eq!(units.len(), 2);
    assert_eq!(units[1].id.0, "node:apply");
    assert!(contribution.source().contains("fn apply_node"));
    assert!(contribution.source_unit().source.contains("fn apply_node"));
    assert_eq!(contribution.bindings.len(), 1);
    compose(contribution).expect("Shader composes generated glue with node source");
}

#[test]
fn source_and_static_choices_change_variant_but_dynamic_instance_values_do_not() {
    let source_a = definition(
        node(
            "node:a",
            "fn apply_node(color: vec4<f32>) -> vec4<f32> { return color * tint; }",
        ),
        true,
    );
    let source_b = definition(
        node(
            "node:b",
            "fn apply_node(color: vec4<f32>) -> vec4<f32> { return color + tint; }",
        ),
        true,
    );
    let static_b = definition(
        node(
            "node:a",
            "fn apply_node(color: vec4<f32>) -> vec4<f32> { return color * tint; }",
        ),
        false,
    );
    let first = MaterialCompiler::compile(&source_a).unwrap();
    assert_ne!(
        first.key(),
        MaterialCompiler::compile(&source_b).unwrap().key()
    );
    assert_ne!(
        first.key(),
        MaterialCompiler::compile(&static_b).unwrap().key()
    );

    let store = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let handle = store.create().unwrap();
    let snapshot = match store.acquire(&handle).unwrap() {
        Acquire::Producer(producer) => producer.commit(source_a, ResidentBytes::new(0)).unwrap(),
        _ => unreachable!(),
    };
    let variant = MaterialCompiler::compile_for(&snapshot).unwrap();
    let material = MaterialAssetRef::from_snapshot(&snapshot);
    let resources = BTreeMap::from([("material_tint".into(), "owner".into())]);
    let low = MaterialInstance::new(
        material,
        &variant,
        BTreeMap::from([("tint".into(), DynamicValue::Vec4([0.1; 4]))]),
        resources.clone(),
    )
    .unwrap();
    let high = MaterialInstance::new(
        material,
        &variant,
        BTreeMap::from([("tint".into(), DynamicValue::Vec4([0.9; 4]))]),
        resources,
    )
    .unwrap();
    assert_eq!(low.variant(), high.variant());
}

#[test]
fn independently_attributed_nodes_have_deterministic_ordering_in_the_variant() {
    let first = node(
        "node:apply",
        "fn apply_node(color: vec4<f32>) -> vec4<f32> { return color * tint; }",
    );
    let extra = MaterialCodeNode {
        name: "unused_node".into(),
        inputs: Vec::new(),
        output: MaterialType::Vec4,
        calls: Vec::new(),
        source: SourceUnit::new(
            "node:unused",
            "fn unused_node() -> vec4<f32> { return vec4<f32>(1.0); }",
        ),
        resources: Vec::new(),
    };
    let left = definition(first.clone(), true).with_code_node(extra.clone());
    let right = definition(extra, true).with_code_node(first);

    let left = MaterialCompiler::compile(&left).unwrap();
    let right = MaterialCompiler::compile(&right).unwrap();
    assert_eq!(left.key(), right.key());
    assert_eq!(
        left.contribution().source_units(),
        right.contribution().source_units()
    );
}

#[test]
fn node_callable_and_resource_conflicts_fail_without_legacy_fallback() {
    let missing = definition(
        node(
            "node:missing",
            "fn another() -> vec4<f32> { return vec4<f32>(1.0); }",
        ),
        true,
    );
    let missing = MaterialCompiler::compile(&missing).unwrap();
    assert!(compose(missing.contribution()).is_err());

    let wrong_signature = definition(
        node(
            "node:signature",
            "fn apply_node(color: f32) -> vec4<f32> { return vec4<f32>(color); }",
        ),
        true,
    );
    let wrong_signature = MaterialCompiler::compile(&wrong_signature).unwrap();
    assert!(compose(wrong_signature.contribution()).is_err());

    let mut conflicting = node(
        "node:binding",
        "fn apply_node(color: vec4<f32>) -> vec4<f32> { return color; }",
    );
    conflicting.resources[0].kind = MaterialResourceKind::Texture2dF32;
    assert!(matches!(
        MaterialCompiler::compile(&definition(conflicting, true)),
        Err(MaterialError::ConflictingResourceDeclaration { name }) if name == "material_tint"
    ));
}

#[test]
fn node_parse_diagnostic_keeps_author_source_identity_and_local_line() {
    let malformed = definition(
        node(
            "node:broken",
            "\nfn apply_node(color: vec4<f32>) -> vec4<f32> { let = color; return color; }",
        ),
        true,
    );
    let compiled = MaterialCompiler::compile(&malformed).unwrap();
    match compose(compiled.contribution()) {
        Err(ShaderError::Parse {
            source: Some(source),
            line: Some(line),
            ..
        }) => {
            assert_eq!(source.0, "node:broken");
            assert_eq!(line, 2);
        }
        other => panic!("expected attributed node parse error, got {other:?}"),
    }
}

#[test]
fn duplicate_or_empty_node_source_ids_are_not_silently_aggregated() {
    let first = node(
        "node:duplicate",
        "fn apply_node(color: vec4<f32>) -> vec4<f32> { return color; }",
    );
    let second = MaterialCodeNode {
        name: "unused_node".into(),
        inputs: Vec::new(),
        output: MaterialType::Vec4,
        calls: Vec::new(),
        source: SourceUnit::new(
            "node:duplicate",
            "fn unused_node() -> vec4<f32> { return vec4<f32>(1.0); }",
        ),
        resources: Vec::new(),
    };
    let duplicate = definition(first, true).with_code_node(second);
    assert!(matches!(
        MaterialCompiler::compile(&duplicate),
        Err(MaterialError::DuplicateCodeNodeSourceId { source_id }) if source_id == "node:duplicate"
    ));

    let empty = definition(
        node(
            "",
            "fn apply_node(color: vec4<f32>) -> vec4<f32> { return color; }",
        ),
        true,
    );
    assert!(matches!(
        MaterialCompiler::compile(&empty),
        Err(MaterialError::EmptyCodeNodeSourceId { node }) if node == "apply_node"
    ));
}
