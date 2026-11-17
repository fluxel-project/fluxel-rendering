//! One-generation replacement within a four-logical-texture Forward record.

use super::*;

use fluxel_assets::{Acquire, AssetStore, Production, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialDefinition, MaterialFragment, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};

fn four_texture_material() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![MaterialFragment {
            name: "base".into(),
            inputs: Vec::new(),
            output: MaterialType::Vec4,
            calls: Vec::new(),
            wgsl: "fn base() -> vec4<f32> { let albedo = textureLoad(base_color, vec2<i32>(0, 0), 0); let normal_value = textureLoad(normal, vec2<i32>(0, 0), 0); let packed = textureLoad(orm, vec2<i32>(0, 0), 0); let emission = textureLoad(emissive, vec2<i32>(0, 0), 0); return (albedo + normal_value * 0.125 + packed * 0.0625 + emission * 0.03125) * tint; }".into(),
        }],
        "base",
        vec![MaterialParameter {
            name: "tint".into(),
            ty: MaterialType::Vec4,
            resource: "tint_resource".into(),
        }],
        vec![
            MaterialResource { name: "base_color".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 0 } },
            MaterialResource { name: "normal".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
            MaterialResource { name: "tint_resource".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 2 } },
            MaterialResource { name: "orm".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 3 } },
            MaterialResource { name: "emissive".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 4 } },
        ],
        BTreeMap::new(),
        SurfaceOutput::Color,
    )
}

fn snapshot<K, V>(
    store: &AssetStore<K, V, ()>,
    value: V,
    bytes: u64,
) -> fluxel_assets::AssetSnapshot<K, V>
where
    K: fluxel_assets::AssetKind,
    V: Send + Sync + 'static,
{
    let handle = store.create().unwrap();
    match store.acquire(&handle).unwrap() {
        Acquire::Producer(producer) => producer.commit(value, ResidentBytes::new(bytes)).unwrap(),
        _ => unreachable!(),
    }
}

fn generation_assets() -> (ForwardAssets, ForwardAssets) {
    let geometries = AssetStore::<GeometryAsset, GeometryData, ()>::new();
    let geometry = snapshot(&geometries, geometry(0.0, false), 96);
    let materials = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let material = snapshot(&materials, four_texture_material(), 0);
    let textures = AssetStore::<ColorTextureAsset, ColorTextureData, ()>::new();
    let mut handles = Vec::new();
    let mut old = Vec::new();
    for value in [
        [1, 2, 3, 255],
        [4, 5, 6, 255],
        [7, 8, 9, 255],
        [10, 11, 12, 255],
    ] {
        let handle = textures.create().unwrap();
        let image = match textures.acquire(&handle).unwrap() {
            Acquire::Producer(producer) => producer
                .commit(
                    ColorTextureData {
                        extent: [1, 1],
                        encoding: crate::TextureEncoding::Linear,
                        mip_levels: 1,
                        rgba8: value.to_vec(),
                    },
                    ResidentBytes::new(4),
                )
                .unwrap(),
            _ => unreachable!(),
        };
        handles.push(handle);
        old.push(image);
    }
    let mut replaced = old.clone();
    replaced[1] = match textures.request_replacement(&handles[1]).unwrap() {
        Production::Producer(producer) => producer
            .commit(
                ColorTextureData {
                    extent: [1, 1],
                    encoding: crate::TextureEncoding::Linear,
                    mip_levels: 1,
                    rgba8: vec![91, 92, 93, 255],
                },
                ResidentBytes::new(4),
            )
            .unwrap(),
        Production::Waiter(_) => unreachable!(),
    };
    assert_eq!(old[1].id(), replaced[1].id());
    assert_ne!(old[1].generation(), replaced[1].generation());
    let common = (geometry, material);
    (
        ForwardAssets {
            geometry: vec![common.0.clone()],
            materials: vec![common.1.clone()],
            textures: old,
        },
        ForwardAssets {
            geometry: vec![common.0],
            materials: vec![common.1],
            textures: replaced,
        },
    )
}

fn generation_scene(
    assets: &ForwardAssets,
) -> (crate::FrameSnapshot<ForwardResources>, crate::RenderViewId) {
    let mut scene = RenderScene::new().unwrap();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [8, 8]).unwrap(),
    ));
    changes.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(&assets.geometry[0]),
            material: SceneAssetRef::from_snapshot(&assets.materials[0]),
            textures: ["base_color", "normal", "orm", "emissive"]
                .into_iter()
                .zip(&assets.textures)
                .map(|(name, image)| {
                    crate::ForwardTextureBinding::asset(name, SceneAssetRef::from_snapshot(image))
                })
                .collect(),
            tint: [1.0; 4],
        },
        transform: RenderTransform::identity(),
        bounds: bounds(),
    }));
    let receipt = scene.apply(&changes).unwrap();
    (scene.snapshot(), receipt.inserted_views()[0])
}

fn submit_generation(
    renderer: &mut Renderer<WitnessBackend>,
    assets: &ForwardAssets,
) -> Submission {
    let (snapshot, view) = generation_scene(assets);
    block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        assets,
    ))
    .unwrap()
}

#[test]
fn one_named_texture_generation_replacement_keeps_pending_old_owner_and_exact_cache_isolation() {
    let (device, _) = device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (old, new) = generation_assets();
    let cold = submit_generation(&mut renderer, &old);
    block_on(renderer.readback(&cold)).unwrap();
    let hits = renderer.forward_cache_stats().texture_hits;
    let warm = submit_generation(&mut renderer, &old);
    assert!(renderer.forward_cache_stats().texture_hits >= hits + 4);
    block_on(renderer.readback(&warm)).unwrap();
    let old_pending = submit_generation(&mut renderer, &old);
    let new_frame = submit_generation(&mut renderer, &new);
    assert_eq!(renderer.pending(), 2);
    assert!(
        renderer.forward_cache_stats().texture_hits >= hits + 7,
        "three unchanged sources reuse their exact completed owners"
    );
    block_on(renderer.readback(&new_frame)).unwrap();
    assert_eq!(
        renderer.pending(),
        1,
        "new completion cannot retire the old generation"
    );
    block_on(renderer.readback(&old_pending)).unwrap();
    let returned_before = renderer.forward_cache_stats().texture_hits;
    let returned = submit_generation(&mut renderer, &old);
    assert!(renderer.forward_cache_stats().texture_hits >= returned_before + 4);
    block_on(renderer.readback(&returned)).unwrap();
}
