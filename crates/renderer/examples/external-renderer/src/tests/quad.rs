use super::quad::{
    quad_geometry_data, quad_marker_material_definition, QuadAssets, QuadFeature, QuadInput,
    QuadMarker,
};
use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{MaterialAsset, MaterialCompiler};
use fluxel_renderer::{
    prepare_forward, BoundsVisibility, ColorTextureAsset, ColorTextureData, CpuPhase, FeatureError,
    ForwardAssets, ForwardCpuPipeline, ForwardResources, FramePipeline, GeometryAsset,
    GeometryData, RenderBounds, RenderChange, RenderChangeSet, RenderFeature, RenderRecord,
    RenderScene, RenderTransform, RenderView, SceneAssetRef,
};
use fluxel_shader::ShaderComposer;

#[test]
fn quad_feature_prepares_only_its_own_visible_draws() {
    let assets = assets();
    let mut scene = RenderScene::<ForwardResources>::new().unwrap();
    let issuer = scene.feature_issuer().unwrap();
    let feature = QuadFeature::new(&issuer).unwrap();
    let view = insert_view_and_record(&mut scene, &assets);
    let snapshot = scene.snapshot();
    let context = issuer
        .context(&snapshot, CpuPhase::Reference, feature.handle())
        .unwrap();
    let mut storage = issuer.storage();
    let input = storage
        .publish(
            &context,
            QuadInput {
                markers: vec![
                    marker(7, [0.0, 0.0, 0.0], assets.clone()),
                    marker(8, [20.0, 0.0, 0.0], assets),
                ],
            },
        )
        .unwrap();
    let volume = RenderBounds::new([-2.0; 3], [2.0; 3]).unwrap();
    let frame = ForwardCpuPipeline::<BoundsVisibility>::default()
        .prepare(&snapshot, view, volume)
        .unwrap();

    let prepared = feature.prepare(&context, &input, &frame, volume).unwrap();
    assert_eq!(prepared.draws().len(), 1);
    assert_eq!(prepared.draws()[0].local_id, 7);
    assert_eq!(
        prepared.draws()[0].transform.columns()[3],
        [0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn quad_feature_rejects_dead_input_and_runtime_disable_is_empty() {
    let assets = assets();
    let mut scene = RenderScene::<ForwardResources>::new().unwrap();
    let issuer = scene.feature_issuer().unwrap();
    let mut feature = QuadFeature::new(&issuer).unwrap();
    let view = insert_view_and_record(&mut scene, &assets);
    let snapshot = scene.snapshot();
    let context = issuer
        .context(&snapshot, CpuPhase::Reference, feature.handle())
        .unwrap();
    let mut storage = issuer.storage();
    let mut invalid_marker = marker(3, [0.0; 3], assets.clone());
    invalid_marker.half_extent = [0.0, 1.0];
    let invalid = storage
        .publish(
            &context,
            QuadInput {
                markers: vec![invalid_marker],
            },
        )
        .unwrap();
    let volume = RenderBounds::new([-2.0; 3], [2.0; 3]).unwrap();
    let frame = ForwardCpuPipeline::<BoundsVisibility>::default()
        .prepare(&snapshot, view, volume)
        .unwrap();
    assert!(matches!(
        feature.prepare(&context, &invalid, &frame, volume),
        Err(FeatureError::InvalidPreparedData {
            detail: "marker values must be finite with positive half extent",
            ..
        })
    ));

    let valid = storage
        .publish(
            &context,
            QuadInput {
                markers: vec![marker(4, [0.0; 3], assets)],
            },
        )
        .unwrap();
    feature.set_enabled(false);
    assert!(!feature.is_enabled());
    assert!(feature
        .prepare(&context, &valid, &frame, volume)
        .unwrap()
        .draws()
        .is_empty());
}

#[test]
fn quad_material_is_a_real_compilable_custom_color_definition() {
    MaterialCompiler::compile(&quad_marker_material_definition())
        .expect("normal material compiler accepts quad fragment");
}

#[test]
fn quad_material_composes_through_real_forward_preparation() {
    let forward = forward_assets();
    let refs = QuadAssets {
        geometry: SceneAssetRef::from_snapshot(&forward.geometry[0]),
        material: SceneAssetRef::from_snapshot(&forward.materials[0]),
        texture: SceneAssetRef::from_snapshot(&forward.textures[0]),
    };
    let mut scene = RenderScene::<ForwardResources>::new().unwrap();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: refs.geometry,
            material: refs.material,
            texture: refs.texture,
            tint: [1.0, 0.0, 0.0, 1.0],
        },
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new([-0.5; 3], [0.5; 3]).unwrap(),
    }));
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [32, 32]).unwrap(),
    ));
    let view = scene.apply(&changes).unwrap().inserted_views()[0];
    let snapshot = scene.snapshot();
    let frame = ForwardCpuPipeline::<BoundsVisibility>::default()
        .prepare(
            &snapshot,
            view,
            RenderBounds::new([-1.0; 3], [1.0; 3]).unwrap(),
        )
        .unwrap();
    let mut composer = ShaderComposer::new();
    let prepared = prepare_forward(&frame, &forward, &mut composer)
        .expect("quad material must expose both group-0 Forward bindings to the composed Shader");
    assert_eq!(prepared.len(), 1);
}

fn marker(id: u64, center: [f32; 3], assets: QuadAssets) -> QuadMarker {
    QuadMarker {
        id,
        center,
        half_extent: [1.0, 1.0],
        tint: [0.2, 0.4, 0.8, 1.0],
        assets,
        order: 5,
    }
}

fn insert_view_and_record(
    scene: &mut RenderScene<ForwardResources>,
    assets: &QuadAssets,
) -> fluxel_renderer::RenderViewId {
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: assets.geometry,
            material: assets.material,
            texture: assets.texture,
            tint: [1.0; 4],
        },
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new([10.0; 3], [11.0; 3]).unwrap(),
    }));
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [32, 32]).unwrap(),
    ));
    scene.apply(&changes).unwrap().inserted_views()[0]
}

fn assets() -> QuadAssets {
    let forward = forward_assets();
    QuadAssets {
        geometry: SceneAssetRef::from_snapshot(&forward.geometry[0]),
        material: SceneAssetRef::from_snapshot(&forward.materials[0]),
        texture: SceneAssetRef::from_snapshot(&forward.textures[0]),
    }
}

fn forward_assets() -> ForwardAssets {
    let geometry_store = AssetStore::<GeometryAsset, GeometryData, ()>::new();
    let geometry_handle = geometry_store.create().unwrap();
    let geometry = match geometry_store.acquire(&geometry_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(quad_geometry_data(), ResidentBytes::new(216))
            .unwrap(),
        _ => unreachable!(),
    };
    let material_store =
        AssetStore::<MaterialAsset, fluxel_material::MaterialDefinition, ()>::new();
    let material_handle = material_store.create().unwrap();
    let material = match material_store.acquire(&material_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(quad_marker_material_definition(), ResidentBytes::new(0))
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
                    rgba8: vec![255; 4],
                },
                ResidentBytes::new(4),
            )
            .unwrap(),
        _ => unreachable!(),
    };
    ForwardAssets {
        geometry: vec![geometry],
        materials: vec![material],
        textures: vec![texture],
    }
}
