use std::collections::BTreeMap;

use crate::{
    BindingLocation, DynamicValue, FragmentCall, FragmentInput, MaterialAsset, MaterialAssetRef,
    MaterialCompiler, MaterialDefinition, MaterialError, MaterialFragment, MaterialInstance,
    MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType, StaticValue,
    SurfaceOutput,
};
use fluxel_assets::{Acquire, AssetStore, Production, ResidentBytes};

fn fragment(
    name: &str,
    calls: Vec<FragmentCall>,
    output: MaterialType,
    wgsl: &str,
) -> MaterialFragment {
    MaterialFragment {
        name: name.into(),
        inputs: Vec::new(),
        output,
        calls,
        wgsl: wgsl.into(),
    }
}

fn definition(enabled: bool) -> MaterialDefinition {
    let mut options = BTreeMap::new();
    options.insert("MATERIAL_ENABLED".into(), StaticValue::Bool(enabled));
    MaterialDefinition::new(
        vec![
            fragment(
                "sample_color",
                Vec::new(),
                MaterialType::Vec4,
                "fn sample_color() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0); }",
            ),
            fragment(
                "custom_tint",
                vec![FragmentCall {
                    fragment: "sample_color".into(),
                    arguments: Vec::new(),
                }],
                MaterialType::Vec4,
                "fn custom_tint() -> vec4<f32> { return sample_color() * tint; }",
            ),
        ],
        "custom_tint",
        vec![MaterialParameter {
            name: "tint".into(),
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
        options,
        SurfaceOutput::Color,
    )
}

#[test]
fn multi_fragment_material_emits_deterministic_custom_wgsl() {
    let first = MaterialCompiler::compile(&definition(true)).unwrap();
    let second = MaterialCompiler::compile(&definition(true)).unwrap();
    assert_eq!(first, second);
    let source = &first.contribution().source;
    assert!(source.contains("@group(0) @binding(0) var<uniform> tint: vec4<f32>;"));
    assert!(source.contains("textureLoad(material_texture"));
    assert!(source.contains("return sample_color() * tint;"));
    assert!(source.contains("fn evaluate_material() -> vec4<f32>"));
    assert_eq!(first.contribution().bindings.len(), 2);
    assert_eq!(
        first.contribution().parameter_bindings,
        vec![crate::MaterialParameterBinding {
            parameter: "tint".into(),
            resource: "material_tint".into(),
            location: BindingLocation {
                group: 0,
                binding: 0,
            },
            ty: MaterialType::Vec4,
        }]
    );
}

#[test]
fn static_choice_and_asset_generation_invalidate_but_dynamic_values_share() {
    let enabled = MaterialCompiler::compile(&definition(true)).unwrap();
    let disabled = MaterialCompiler::compile(&definition(false)).unwrap();
    assert_ne!(enabled.key(), disabled.key());
    let (initial, replacement) = definition_snapshots();
    let material = MaterialAssetRef::from_snapshot(&initial);
    let first = MaterialCompiler::compile_for(&initial).unwrap();
    let second = MaterialCompiler::compile_for(&replacement).unwrap();
    assert_ne!(first.key(), second.key());
    assert!(matches!(
        MaterialInstance::new(
            MaterialAssetRef::from_snapshot(&replacement),
            &first,
            BTreeMap::new(),
            BTreeMap::new()
        ),
        Err(MaterialError::VariantMaterialMismatch { .. })
    ));
    let other_store = MaterialCompiler::compile_for(&other_store_snapshot()).unwrap();
    assert_ne!(first.key(), other_store.key());
    let resources = BTreeMap::from([
        ("material_tint".into(), "tint".into()),
        ("material_texture".into(), "texture".into()),
    ]);
    let low = MaterialInstance::new(
        material,
        &enabled,
        BTreeMap::from([("tint".into(), DynamicValue::Vec4([0.1; 4]))]),
        resources.clone(),
    )
    .unwrap();
    let high = MaterialInstance::new(
        material,
        &enabled,
        BTreeMap::from([("tint".into(), DynamicValue::Vec4([0.9; 4]))]),
        resources,
    )
    .unwrap();
    assert_eq!(low.variant(), high.variant());
}

#[test]
fn rejects_invalid_calls_cycles_types_bindings_outputs_and_values() {
    let missing = MaterialDefinition::new(
        Vec::new(),
        "missing",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::Color,
    );
    assert!(matches!(
        MaterialCompiler::compile(&missing),
        Err(MaterialError::MissingFragment { .. })
    ));
    let cycle = MaterialDefinition::new(
        vec![
            fragment(
                "a",
                vec![FragmentCall {
                    fragment: "b".into(),
                    arguments: Vec::new(),
                }],
                MaterialType::Vec4,
                "fn a() -> vec4<f32> { return b(); }",
            ),
            fragment(
                "b",
                vec![FragmentCall {
                    fragment: "a".into(),
                    arguments: Vec::new(),
                }],
                MaterialType::Vec4,
                "fn b() -> vec4<f32> { return a(); }",
            ),
        ],
        "a",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::Color,
    );
    assert!(matches!(
        MaterialCompiler::compile(&cycle),
        Err(MaterialError::CallCycle { .. })
    ));
    let mismatch = MaterialDefinition::new(
        vec![
            fragment(
                "root",
                vec![FragmentCall {
                    fragment: "child".into(),
                    arguments: vec![MaterialType::F32],
                }],
                MaterialType::Vec4,
                "fn root() -> vec4<f32> { return vec4<f32>(); }",
            ),
            MaterialFragment {
                name: "child".into(),
                inputs: vec![FragmentInput {
                    name: "color".into(),
                    ty: MaterialType::Vec4,
                }],
                output: MaterialType::Vec4,
                calls: Vec::new(),
                wgsl: "fn child(color: vec4<f32>) -> vec4<f32> { return color; }".into(),
            },
        ],
        "root",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::Color,
    );
    assert!(matches!(
        MaterialCompiler::compile(&mismatch),
        Err(MaterialError::CallTypeMismatch { .. })
    ));
    let binding = MaterialDefinition::new(
        vec![fragment(
            "root",
            Vec::new(),
            MaterialType::Vec4,
            "fn root() -> vec4<f32> { return vec4<f32>(); }",
        )],
        "root",
        Vec::new(),
        vec![
            MaterialResource {
                name: "one".into(),
                kind: MaterialResourceKind::UniformVec4,
                location: BindingLocation {
                    group: 0,
                    binding: 0,
                },
            },
            MaterialResource {
                name: "two".into(),
                kind: MaterialResourceKind::Texture2dF32,
                location: BindingLocation {
                    group: 0,
                    binding: 0,
                },
            },
        ],
        BTreeMap::new(),
        SurfaceOutput::Color,
    );
    assert!(matches!(
        MaterialCompiler::compile(&binding),
        Err(MaterialError::BindingConflict { .. })
    ));
    let bad_output = MaterialDefinition::new(
        vec![fragment(
            "root",
            Vec::new(),
            MaterialType::F32,
            "fn root() -> f32 { return 1.0; }",
        )],
        "root",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::Color,
    );
    assert!(matches!(
        MaterialCompiler::compile(&bad_output),
        Err(MaterialError::UnsupportedOutput { .. })
    ));
    let variant = MaterialCompiler::compile(&definition(true)).unwrap();
    assert!(matches!(
        MaterialInstance::new(
            material_reference(),
            &variant,
            BTreeMap::from([("tint".into(), DynamicValue::F32(0.0))]),
            BTreeMap::new()
        ),
        Err(MaterialError::ParameterTypeMismatch { .. })
    ));

    let missing_parameter_resource = MaterialDefinition::new(
        vec![fragment(
            "root",
            Vec::new(),
            MaterialType::Vec4,
            "fn root() -> vec4<f32> { return vec4<f32>(1.0); }",
        )],
        "root",
        vec![MaterialParameter {
            name: "tint".into(),
            ty: MaterialType::Vec4,
            resource: "not_declared".into(),
        }],
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::Color,
    );
    assert!(matches!(
        MaterialCompiler::compile(&missing_parameter_resource),
        Err(MaterialError::ParameterResourceMissing { .. })
    ));

    let incompatible_parameter_resource = MaterialDefinition::new(
        vec![fragment(
            "root",
            Vec::new(),
            MaterialType::Vec4,
            "fn root() -> vec4<f32> { return vec4<f32>(1.0); }",
        )],
        "root",
        vec![MaterialParameter {
            name: "tint".into(),
            ty: MaterialType::Vec4,
            resource: "texture".into(),
        }],
        vec![MaterialResource {
            name: "texture".into(),
            kind: MaterialResourceKind::Texture2dF32,
            location: BindingLocation {
                group: 0,
                binding: 0,
            },
        }],
        BTreeMap::new(),
        SurfaceOutput::Color,
    );
    assert!(matches!(
        MaterialCompiler::compile(&incompatible_parameter_resource),
        Err(MaterialError::ParameterResourceTypeMismatch { .. })
    ));

    let colliding_parameter_name = MaterialDefinition::new(
        vec![fragment(
            "root",
            Vec::new(),
            MaterialType::Vec4,
            "fn root() -> vec4<f32> { return vec4<f32>(1.0); }",
        )],
        "root",
        vec![MaterialParameter {
            name: "MATERIAL_ENABLED".into(),
            ty: MaterialType::Vec4,
            resource: "uniform".into(),
        }],
        vec![MaterialResource {
            name: "uniform".into(),
            kind: MaterialResourceKind::UniformVec4,
            location: BindingLocation {
                group: 0,
                binding: 0,
            },
        }],
        BTreeMap::from([("MATERIAL_ENABLED".into(), StaticValue::Bool(true))]),
        SurfaceOutput::Color,
    );
    assert!(matches!(
        MaterialCompiler::compile(&colliding_parameter_name),
        Err(MaterialError::ParameterNameConflict { .. })
    ));
}

fn material_reference() -> MaterialAssetRef {
    MaterialAssetRef::from_snapshot(&other_store_snapshot())
}

fn other_store_snapshot() -> fluxel_assets::AssetSnapshot<MaterialAsset, MaterialDefinition> {
    definition_snapshots().0
}

fn definition_snapshots() -> (
    fluxel_assets::AssetSnapshot<MaterialAsset, MaterialDefinition>,
    fluxel_assets::AssetSnapshot<MaterialAsset, MaterialDefinition>,
) {
    let store = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let handle = store.create().unwrap();
    let initial = match store.acquire(&handle).unwrap() {
        Acquire::Producer(permit) => permit
            .commit(definition(true), ResidentBytes::new(1))
            .unwrap(),
        _ => panic!("new material must nominate a producer"),
    };
    let replacement = match store.request_replacement(&handle).unwrap() {
        Production::Producer(permit) => permit,
        _ => panic!("replacement must nominate a producer"),
    };
    let current = replacement
        .commit(definition(false), ResidentBytes::new(1))
        .unwrap();
    (initial, current)
}
