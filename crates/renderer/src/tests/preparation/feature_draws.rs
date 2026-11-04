use super::*;

use std::collections::BTreeMap;

use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{MaterialAsset, MaterialDefinition, SurfaceOutput};
use fluxel_shader::ShaderComposer;

use crate::{
    ForwardCpuPipeline, ForwardDrawOrder, ForwardDrawSource, FramePipeline, RenderBounds,
    RenderChange, RenderChangeSet, RenderScene, RenderTransform, RenderView, SceneAssetRef,
};

fn resources() -> ForwardResources {
    let geometry_store = AssetStore::<GeometryAsset, GeometryData, ()>::new();
    let geometry_handle = geometry_store.create().unwrap();
    let geometry = match geometry_store.acquire(&geometry_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(
                GeometryData {
                    layout: GeometryLayout::default(),
                    streams: Vec::new(),
                    vertices: 0..0,
                },
                ResidentBytes::new(0),
            )
            .unwrap(),
        _ => unreachable!(),
    };
    let material_store = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let material_handle = material_store.create().unwrap();
    let material = match material_store.acquire(&material_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(
                MaterialDefinition::new(
                    Vec::new(),
                    "unused",
                    Vec::new(),
                    Vec::new(),
                    BTreeMap::new(),
                    SurfaceOutput::Color,
                ),
                ResidentBytes::new(0),
            )
            .unwrap(),
        _ => unreachable!(),
    };
    let texture_store = AssetStore::<ColorTextureAsset, ColorTextureData, ()>::new();
    let texture_handle = texture_store.create().unwrap();
    let texture = match texture_store.acquire(&texture_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(
                ColorTextureData {
                    extent: [1, 1],
                    rgba8: vec![0; 4],
                },
                ResidentBytes::new(4),
            )
            .unwrap(),
        _ => unreachable!(),
    };
    ForwardResources {
        geometry: SceneAssetRef::from_snapshot(&geometry),
        material: SceneAssetRef::from_snapshot(&material),
        texture: SceneAssetRef::from_snapshot(&texture),
        tint: [1.0; 4],
    }
}

#[test]
fn duplicate_feature_local_id_is_rejected_before_asset_resolution() {
    let mut scene = RenderScene::<ForwardResources>::new().unwrap();
    let issuer = scene.feature_issuer().unwrap();
    let feature = issuer.issue("feature-draws").unwrap();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [1, 1]).unwrap(),
    ));
    let view = scene.apply(&changes).unwrap().inserted_views()[0];
    let snapshot = scene.snapshot();
    let pipeline = ForwardCpuPipeline::<crate::BoundsVisibility>::default();
    let frame = pipeline
        .prepare(
            &snapshot,
            view,
            RenderBounds::new([0.0; 3], [1.0; 3]).unwrap(),
        )
        .unwrap();
    let request = ForwardDrawRequest {
        local_id: 7,
        order: 4,
        resources: resources(),
        transform: RenderTransform::identity(),
    };
    let mut composer = ShaderComposer::new();
    assert!(matches!(
        prepare_feature_draws(
            &frame,
            feature,
            &[request.clone(), request],
            &ForwardAssets::default(),
            &mut composer,
        ),
        Err(ForwardPreparationError::DuplicateFeatureLocalId {
            feature: actual,
            local_id: 7,
        }) if actual == feature
    ));
}

#[test]
fn feature_draw_order_keeps_feature_identity_in_its_own_namespace() {
    let scene = RenderScene::<ForwardResources>::new().unwrap();
    let feature = scene
        .feature_issuer()
        .unwrap()
        .issue("feature-draws")
        .unwrap();
    let source = ForwardDrawSource::Feature {
        feature,
        local_id: 12,
    };
    let order = ForwardDrawOrder::new(source, -3);

    assert_eq!(order.source(), source);
    assert_eq!(order.order(), -3);
    assert_eq!(order.sort_key().0, -3);
    assert_eq!(order.sort_key().1.0, 1);
}
