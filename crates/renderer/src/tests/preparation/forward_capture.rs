//! Capture preparation must not resolve or manufacture an ordinary texture.

use super::*;
use crate::{FeatureIssuer, ForwardDrawSource, RenderTransform, RenderView, SceneAssetRef};
use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, FragmentInput, MaterialAsset, MaterialCodeNode, MaterialDefinition,
    MaterialNode, MaterialNodeGraph, MaterialNodeId, MaterialNodeInput, MaterialNodeInputSource,
    MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
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
fn capture_receiver_rejects_asset_substitution_and_wrong_logical_slot() {
    let (mut resources, assets) = fixture(true);
    let texture = snapshot::<ColorTextureAsset, _>(ColorTextureData {
        extent: [1, 1],
        encoding: crate::TextureEncoding::Linear,
        rgba8: vec![255; 4],
    });
    resources.textures[0].source =
        ForwardTextureSource::Asset(SceneAssetRef::from_snapshot(&texture));
    assert!(matches!(
        prepare(&resources, &assets),
        Err(ForwardPreparationError::CaptureReceiverRequiresSlot { .. })
    ));
    resources.textures[0].source = ForwardTextureSource::CaptureSlot("another_image".into());
    assert!(matches!(
        prepare(&resources, &assets),
        Err(ForwardPreparationError::InvalidCaptureSlot { .. })
    ));
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
