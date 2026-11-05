//! PBR remains explicitly outside the bounded position-only deformation route.

use super::*;

use std::{collections::BTreeMap, sync::Arc};

use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialDefinition, MaterialFragment, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_rhi::api::{Device, Error, VertexFormat, VertexStepMode};

use crate::{
    ColorTextureAsset, ColorTextureData, DeferredCpuPipeline, DeformationBinding, ForwardAssets,
    ForwardResources, GeometryAsset, GeometryAttribute, GeometryData, GeometryLayout,
    GeometryStream, RenderChange, RenderChangeSet, RenderRecord, RenderScene, RenderTransform,
    RenderView, SceneAssetRef,
};

fn pbr_material() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![
            MaterialFragment {
                name: "base_color".into(),
                inputs: vec![],
                output: MaterialType::Vec4,
                calls: vec![],
                wgsl: "fn base_color() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint; }".into(),
            },
            MaterialFragment {
                name: "surface_properties".into(),
                inputs: vec![],
                output: MaterialType::Vec2,
                calls: vec![],
                wgsl: "fn surface_properties() -> vec2<f32> { return vec2<f32>(0.25, 0.6); }".into(),
            },
        ],
        "base_color",
        vec![MaterialParameter {
            name: "tint".into(),
            ty: MaterialType::Vec4,
            resource: "material_tint".into(),
        }],
        vec![
            MaterialResource {
                name: "material_tint".into(),
                kind: MaterialResourceKind::UniformVec4,
                location: BindingLocation { group: 0, binding: 0 },
            },
            MaterialResource {
                name: "material_texture".into(),
                kind: MaterialResourceKind::Texture2dF32,
                location: BindingLocation { group: 0, binding: 1 },
            },
        ],
        BTreeMap::new(),
        SurfaceOutput::MetallicRoughness {
            properties: "surface_properties".into(),
        },
    )
}

fn pbr_deformation_device(script: Script) -> (Device<WitnessBackend>, Arc<Native>) {
    fixture_with_caps(script, |caps| {
        caps.compute = true;
        caps.storage_buffers = true;
        caps.limits.max_color_attachments = 2;
        caps.limits.max_bind_groups = 2;
        caps.limits.max_bindings_per_group = 3;
        caps.limits.max_vertex_buffers = 1;
        caps.limits.max_vertex_attributes = 2;
        caps.limits.max_compute_workgroups = [64, 1, 1];
        caps.limits.max_compute_workgroup_size = [64, 1, 1];
    })
}

fn pbr_scene() -> (
    RenderScene<ForwardResources>,
    crate::RenderViewId,
    ForwardAssets,
) {
    let geometries = AssetStore::<GeometryAsset, GeometryData, ()>::new();
    let materials = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let textures = AssetStore::<ColorTextureAsset, ColorTextureData, ()>::new();
    let mut bytes = Vec::new();
    for position in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        for value in position.into_iter().chain([0.0, 0.0, 1.0]) {
            bytes.extend(value.to_le_bytes());
        }
    }
    let geometry_handle = geometries.create().unwrap();
    let geometry = match geometries.acquire(&geometry_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(
                GeometryData {
                    layout: GeometryLayout {
                        streams: vec![GeometryStream {
                            stride: 24,
                            step_mode: VertexStepMode::Vertex,
                            attributes: vec![
                                GeometryAttribute {
                                    semantic: "POSITION".into(),
                                    offset: 0,
                                    format: VertexFormat::Float32(3),
                                },
                                GeometryAttribute {
                                    semantic: "NORMAL".into(),
                                    offset: 12,
                                    format: VertexFormat::Float32(3),
                                },
                            ],
                        }],
                    },
                    streams: vec![bytes],
                    vertices: 0..3,
                },
                ResidentBytes::new(72),
            )
            .unwrap(),
        _ => unreachable!(),
    };
    let material_handle = materials.create().unwrap();
    let material = match materials.acquire(&material_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(pbr_material(), ResidentBytes::new(0))
            .unwrap(),
        _ => unreachable!(),
    };
    let texture_handle = textures.create().unwrap();
    let texture = match textures.acquire(&texture_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(
                ColorTextureData {
                    extent: [1, 1],
                    rgba8: vec![128, 96, 64, 255],
                },
                ResidentBytes::new(4),
            )
            .unwrap(),
        _ => unreachable!(),
    };
    let mut scene = RenderScene::new().unwrap();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [8, 8]).unwrap(),
    ));
    changes.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(&geometry),
            material: SceneAssetRef::from_snapshot(&material),
            texture: SceneAssetRef::from_snapshot(&texture),
            tint: [1.0; 4],
        },
        transform: RenderTransform::identity(),
        bounds: bounds(),
    }));
    let receipt = scene.apply(&changes).unwrap();
    (
        scene,
        receipt.inserted_views()[0],
        ForwardAssets {
            geometry: vec![geometry],
            materials: vec![material],
            textures: vec![texture],
        },
    )
}

#[test]
fn pbr_control_succeeds_but_position_deformation_refuses_before_backend_work() {
    let (device, native) = pbr_deformation_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (scene, view, assets) = pbr_scene();
    let snapshot = scene.snapshot();
    let submission = block_on(renderer.submit_deferred(
        &DeferredCpuPipeline::new(BoundsVisibility),
        snapshot.clone(),
        view,
        bounds(),
        &assets,
    ))
    .expect("PBR Deferred positive control");
    block_on(renderer.readback(&submission)).expect("PBR Deferred completion");

    let (issuer, feature, input) = bind(&scene, &snapshot, SOURCE, [0.0; 4]);
    let before = counts(&native);
    let result = block_on(renderer.submit_deferred_with_deformation(
        &DeferredCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        DeformationBinding::new(&issuer, &feature, &input),
    ));
    assert!(matches!(
        result,
        Err(RendererError::Rhi(Error::Unsupported(_)))
    ));
    assert_eq!(counts(&native), before);
}

#[test]
fn pbr_surface_control_succeeds_but_position_deformation_refuses_before_backend_work() {
    let (device, native) = pbr_deformation_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (scene, view, assets) = pbr_scene();
    let snapshot = scene.snapshot();
    let mut surface = super::super::surface_tests::surface(&device, [8, 8]);
    let control = block_on(renderer.submit_deferred_surface(
        &DeferredCpuPipeline::new(BoundsVisibility),
        snapshot.clone(),
        view,
        bounds(),
        &assets,
        block_on(surface.acquire()).unwrap(),
    ))
    .expect("PBR Surface positive control");
    block_on(renderer.complete(&control)).expect("PBR Surface completion");

    let (issuer, feature, input) = bind(&scene, &snapshot, SOURCE, [0.0; 4]);
    let image = block_on(surface.acquire()).unwrap();
    let before = counts(&native);
    let result = block_on(renderer.submit_deferred_surface_with_deformation(
        &DeferredCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        DeformationBinding::new(&issuer, &feature, &input),
        image,
    ));
    assert!(matches!(
        result,
        Err(RendererError::Rhi(Error::Unsupported(_)))
    ));
    assert_eq!(counts(&native), before);
}
