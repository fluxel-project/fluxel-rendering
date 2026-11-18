//! Capture preparation must not resolve or manufacture an ordinary texture.

use super::*;
use crate::{FeatureIssuer, ForwardDrawSource, RenderTransform, RenderView, SceneAssetRef};
use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, FragmentInput, MaterialAsset, MaterialCodeNode, MaterialDefinition,
    MaterialFragment, MaterialNode, MaterialNodeGraph, MaterialNodeId, MaterialNodeInput,
    MaterialNodeInputSource, MaterialParameter, MaterialResource, MaterialResourceKind,
    MaterialType, SurfaceOutput,
};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::{ShaderComposer, SourceUnit};
use std::collections::BTreeMap;

fn snapshot<K: AssetKind, T: Send + Sync + 'static>(value: T) -> AssetSnapshot<K, T> {
    let store = AssetStore::<K, T, ()>::new();
    let handle = store.create().unwrap();
    match store.acquire(&handle).unwrap() {
        Acquire::Producer(producer) => producer.commit(value, ResidentBytes::new(0)).unwrap(),
        _ => unreachable!(),
    }
}

fn material() -> MaterialDefinition {
    MaterialDefinition::new(
        Vec::new(), "unused",
        vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "capture_tint".into() }],
        vec![
            MaterialResource { name: "capture_tint".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } },
            MaterialResource { name: "captured_image".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
            MaterialResource { name: "capture_sampler".into(), kind: MaterialResourceKind::Sampler { comparison: false }, location: BindingLocation { group: 0, binding: 2 } },
        ], BTreeMap::new(), SurfaceOutput::ColorUv0,
    )
    .with_code_node(MaterialCodeNode {
        name: "sample_capture".into(),
        inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }],
        output: MaterialType::Vec4, calls: Vec::new(), resources: Vec::new(),
        source: SourceUnit::new("capture-consumer", "fn sample_capture(uv: vec2<f32>) -> vec4<f32> { return textureSample(captured_image, capture_sampler, uv) * tint; }"),
    })
    .with_node_graph(MaterialNodeGraph {
        nodes: vec![MaterialNode { id: MaterialNodeId(1), callable: "sample_capture".into(), inputs: vec![MaterialNodeInput { input: "uv".into(), source: MaterialNodeInputSource::Uv0 }] }],
        output: MaterialNodeId(1),
    })
}

fn fixture(uv: bool) -> (ForwardCaptureResources, ForwardAssets) {
    let mut attributes = vec![GeometryAttribute {
        semantic: "POSITION".into(),
        offset: 0,
        format: VertexFormat::Float32(3),
    }];
    if uv {
        attributes.push(GeometryAttribute {
            semantic: "UV0".into(),
            offset: 12,
            format: VertexFormat::Float32(2),
        });
    }
    let mut bytes = Vec::new();
    for (position, texcoord) in [
        ([-1.0_f32, -1.0, 0.5], [0.0_f32, 0.0]),
        ([1.0, -1.0, 0.5], [1.0, 0.0]),
        ([0.0, 1.0, 0.5], [0.5, 1.0]),
    ] {
        for value in position {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        if uv {
            for value in texcoord {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    let geometry = snapshot::<GeometryAsset, _>(GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: if uv { 20 } else { 12 },
                step_mode: VertexStepMode::Vertex,
                attributes,
            }],
        },
        streams: vec![bytes],
        vertices: 0..3,
    });
    let material = snapshot::<MaterialAsset, _>(material());
    let resources = ForwardCaptureResources {
        geometry: SceneAssetRef::from_snapshot(&geometry),
        material: SceneAssetRef::from_snapshot(&material),
        textures: vec![crate::ForwardTextureBinding::capture(
            "captured_image",
            "captured_image",
        )],
        sampler: None,
        tint: [0.25, 0.5, 0.75, 1.0],
    };
    (
        resources,
        ForwardAssets {
            geometry: vec![geometry],
            materials: vec![material],
            textures: Vec::new(),
        },
    )
}

fn prepare(
    resources: &ForwardCaptureResources,
    assets: &ForwardAssets,
) -> Result<PreparedCaptureReceiver, ForwardPreparationError> {
    let source = ForwardDrawSource::Feature {
        feature: FeatureIssuer::new(0, 0).issue("capture-test").unwrap(),
        local_id: 11,
    };
    prepare_forward_capture_receiver(
        &RenderView::new(RenderTransform::identity(), [32, 24]).unwrap(),
        source,
        7,
        resources,
        RenderTransform::identity(),
        assets,
        &mut ShaderComposer::new(),
    )
}

fn mixed_material() -> MaterialDefinition {
    MaterialDefinition::new(
        Vec::new(),
        "unused",
        vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "tint".into() }],
        vec![
            MaterialResource { name: "tint".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } },
            MaterialResource { name: "capture_a".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
            MaterialResource { name: "capture_b".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 2 } },
            MaterialResource { name: "asset_image".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 3 } },
            MaterialResource { name: "image_sampler".into(), kind: MaterialResourceKind::Sampler { comparison: false }, location: BindingLocation { group: 0, binding: 4 } },
        ],
        BTreeMap::new(),
        SurfaceOutput::ColorUv0,
    )
    .with_code_node(MaterialCodeNode {
        name: "mixed".into(), inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }], output: MaterialType::Vec4, calls: Vec::new(), resources: Vec::new(),
        source: SourceUnit::new("mixed-capture", "fn mixed(uv: vec2<f32>) -> vec4<f32> { return (textureSample(capture_a, image_sampler, uv) + textureSample(capture_b, image_sampler, uv) + textureSample(asset_image, image_sampler, uv)) * tint; }"),
    })
    .with_node_graph(MaterialNodeGraph {
        nodes: vec![MaterialNode { id: MaterialNodeId(1), callable: "mixed".into(), inputs: vec![MaterialNodeInput { input: "uv".into(), source: MaterialNodeInputSource::Uv0 }] }], output: MaterialNodeId(1),
    })
}

#[test]
fn capture_receiver_mixes_asset_and_two_logical_aliases_of_one_producer() {
    let (mut resources, mut assets) = fixture(true);
    let material = snapshot::<MaterialAsset, _>(mixed_material());
    let asset = snapshot::<ColorTextureAsset, _>(ColorTextureData {
        extent: [1, 1],
        encoding: crate::TextureEncoding::Linear,
        mip_levels: 1,
        rgba8: vec![1, 2, 3, 255],
    });
    resources.material = SceneAssetRef::from_snapshot(&material);
    resources.textures = vec![
        crate::ForwardTextureBinding::capture("capture_b", "producer"),
        crate::ForwardTextureBinding::asset("asset_image", SceneAssetRef::from_snapshot(&asset)),
        crate::ForwardTextureBinding::capture("capture_a", "producer"),
    ];
    assets.materials = vec![material];
    assets.textures = vec![asset];
    let draw = prepare(&resources, &assets).unwrap().into_draw_input();
    assert_eq!(draw.resolved_textures().len(), 3);
    assert_eq!(draw.capture_texture_slot(0), Some("producer"));
    assert_eq!(draw.capture_texture_slot(1), Some("producer"));
    assert!(draw.asset_texture(2).is_some());
}

fn pbr_capture_material() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![
            MaterialFragment { name: "base".into(), inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }], output: MaterialType::Vec4, calls: Vec::new(), wgsl: "fn base(uv: vec2<f32>) -> vec4<f32> { return textureSample(base_image, image_sampler, uv) * tint; }".into() },
            MaterialFragment { name: "properties".into(), inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }], output: MaterialType::Vec2, calls: Vec::new(), wgsl: "fn properties(uv: vec2<f32>) -> vec2<f32> { return textureSample(orm, image_sampler, uv).gb; }".into() },
        ], "unused", vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "tint_resource".into() }],
        vec![
            MaterialResource { name: "base_image".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 0 } },
            MaterialResource { name: "normal_image".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
            MaterialResource { name: "orm".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 2 } },
            MaterialResource { name: "tint_resource".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 3 } },
            MaterialResource { name: "image_sampler".into(), kind: MaterialResourceKind::Sampler { comparison: false }, location: BindingLocation { group: 0, binding: 4 } },
        ], BTreeMap::new(), SurfaceOutput::MetallicRoughnessUv0Normal { properties: "properties".into(), normal: "normal".into() },
    ).with_code_node(MaterialCodeNode { name: "normal".into(), inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }], output: MaterialType::Vec3, calls: Vec::new(), resources: Vec::new(), source: SourceUnit::new("pbr-capture-normal", "fn normal(uv: vec2<f32>) -> vec3<f32> { return textureSample(normal_image, image_sampler, uv).xyz; }") })
    .with_node_graph(MaterialNodeGraph { nodes: vec![MaterialNode { id: MaterialNodeId(1), callable: "base".into(), inputs: vec![MaterialNodeInput { input: "uv".into(), source: MaterialNodeInputSource::Uv0 }] }], output: MaterialNodeId(1) })
}

#[test]
fn pbr_uv0_normal_capture_receiver_keeps_normal_transform_and_lighting_contract() {
    let (mut resources, mut assets) = fixture(true);
    let mut geometry = assets.geometry[0].value().clone();
    geometry.layout.streams[0].stride = 48;
    geometry.layout.streams[0].attributes.extend([
        GeometryAttribute {
            semantic: "NORMAL".into(),
            offset: 20,
            format: VertexFormat::Float32(3),
        },
        GeometryAttribute {
            semantic: "TANGENT".into(),
            offset: 32,
            format: VertexFormat::Float32(4),
        },
    ]);
    let bytes = [
        [-1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, -1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    ]
    .into_iter()
    .flatten()
    .flat_map(f32::to_le_bytes)
    .collect();
    geometry.streams = vec![bytes];
    let geometry = snapshot::<GeometryAsset, _>(geometry);
    let material = snapshot::<MaterialAsset, _>(pbr_capture_material());
    let asset = snapshot::<ColorTextureAsset, _>(ColorTextureData {
        extent: [1, 1],
        encoding: crate::TextureEncoding::Linear,
        mip_levels: 1,
        rgba8: vec![1, 2, 3, 255],
    });
    resources.geometry = SceneAssetRef::from_snapshot(&geometry);
    resources.material = SceneAssetRef::from_snapshot(&material);
    resources.textures = vec![
        crate::ForwardTextureBinding::capture("base_image", "base"),
        crate::ForwardTextureBinding::capture("normal_image", "normal"),
        crate::ForwardTextureBinding::asset("orm", SceneAssetRef::from_snapshot(&asset)),
    ];
    assets.geometry = vec![geometry];
    assets.materials = vec![material];
    assets.textures = vec![asset];
    let receiver = prepare(&resources, &assets).unwrap();
    assert!(receiver.normal_from_local.is_some());
    let draw = receiver
        .clone()
        .with_capture_lighting(&mut ShaderComposer::new(), false, false)
        .unwrap()
        .into_draw_input();
    assert_eq!(draw.transform_uniform_bytes(), crate::frame_data::forward::PBR_TRANSFORM_BYTES);
    let diffuse = receiver
        .clone()
        .with_capture_lighting(&mut ShaderComposer::new(), true, false)
        .expect("PBR capture receiver accepts the diffuse-lighting request")
        .into_draw_input();
    assert!(diffuse.normal_from_local().is_some());
    assert!(matches!(
        receiver.with_capture_lighting(&mut ShaderComposer::new(), false, true),
        Err(ForwardPreparationError::InvalidView)
    ));
}

#[test]
fn capture_receiver_prepares_without_any_cpu_texture_asset() {
    let (resources, assets) = fixture(true);
    assert!(assets.textures.is_empty());
    let draw = prepare(&resources, &assets)
        .expect("capture slot is the only texture input")
        .into_draw_input();
    assert!(draw.asset_texture(0).is_none());
    assert_eq!(draw.capture_texture_slot(0), Some("captured_image"));
    assert_eq!(draw.tint(), [0.25, 0.5, 0.75, 1.0]);
    assert_eq!(draw.extent(), [32, 24]);
    assert_eq!(draw.geometry().value().vertices, 0..3);
}

#[test]
fn capture_receiver_rejects_asset_only_but_accepts_logical_binding_to_producer_alias() {
    let (mut resources, mut assets) = fixture(true);
    let texture = snapshot::<ColorTextureAsset, _>(ColorTextureData {
        extent: [1, 1],
        encoding: crate::TextureEncoding::Linear,
        mip_levels: 1,
        rgba8: vec![255; 4],
    });
    assets.textures = vec![texture.clone()];
    resources.textures[0].source =
        ForwardTextureSource::Asset(SceneAssetRef::from_snapshot(&texture));
    assert!(matches!(
        prepare(&resources, &assets),
        Err(ForwardPreparationError::CaptureReceiverRequiresSlot { .. })
    ));
    resources.textures[0].source = ForwardTextureSource::CaptureSlot("another_image".into());
    let draw = prepare(&resources, &assets)
        .expect("logical resource names may differ from producer slots")
        .into_draw_input();
    assert_eq!(draw.capture_texture_slot(0), Some("another_image"));
}

#[test]
fn capture_receiver_requires_uv_payload_and_finite_parameters() {
    let (resources, assets) = fixture(false);
    assert!(matches!(
        prepare(&resources, &assets),
        Err(ForwardPreparationError::MissingUv0 { .. })
    ));
    let (mut resources, assets) = fixture(true);
    resources.tint[1] = f32::NAN;
    assert!(matches!(
        prepare(&resources, &assets),
        Err(ForwardPreparationError::InvalidTint { .. })
    ));
}

#[test]
fn capture_receiver_resolves_actual_geometry_generation() {
    let (resources, mut assets) = fixture(true);
    assets.geometry.clear();
    assert!(matches!(
        prepare(&resources, &assets),
        Err(ForwardPreparationError::MissingAsset {
            asset: "geometry",
            ..
        })
    ));
}
