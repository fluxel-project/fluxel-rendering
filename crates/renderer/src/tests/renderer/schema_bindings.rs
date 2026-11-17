//! Submission consumers of reflected material binding locations.

use super::*;

#[test]
fn read_only_texture_queries_remain_valid_without_a_sampler() {
    let (device, _) = device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (_, _, mut assets) = multi_scene(true);
    let definition = MaterialDefinition::new(
        vec![MaterialFragment {
            name: "dimensions".into(),
            inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }],
            output: MaterialType::Vec4,
            calls: Vec::new(),
            wgsl: "fn dimensions(uv: vec2<f32>) -> vec4<f32> { return tint * vec4<f32>(vec2<f32>(textureDimensions(image)), 1.0, 1.0); }".into(),
        }],
        "dimensions",
        vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "values".into() }],
        vec![
            MaterialResource { name: "values".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 2 } },
            MaterialResource { name: "image".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 0 } },
        ],
        BTreeMap::new(),
        SurfaceOutput::ColorUv0,
    );
    let store = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let material = match store.acquire(&store.create().unwrap()).unwrap() {
        Acquire::Producer(p) => p.commit(definition, ResidentBytes::new(0)).unwrap(),
        _ => unreachable!(),
    };
    let mut scene = RenderScene::new().unwrap();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [8, 8]).unwrap(),
    ));
    changes.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(&assets.geometry[0]),
            material: SceneAssetRef::from_snapshot(&material),
            textures: vec![crate::ForwardTextureBinding::asset(
                "image",
                SceneAssetRef::from_snapshot(&assets.textures[0]),
            )],
            tint: [0.25; 4],
        },
        transform: RenderTransform::identity(),
        bounds: bounds(),
    }));
    assets.materials = vec![material];
    let receipt = scene.apply(&changes).unwrap();
    let snapshot = scene.snapshot();
    let view = receipt.inserted_views()[0];
    let forward = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot.clone(),
        view,
        bounds(),
        &assets,
    ))
    .expect("query-only Forward binding");
    block_on(renderer.readback(&forward)).expect("Forward completion");
    let deferred = block_on(renderer.submit_deferred(
        &crate::DeferredCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
    ))
    .expect("query-only Deferred binding");
    block_on(renderer.readback(&deferred)).expect("Deferred completion");
}

#[test]
fn permuted_material_locations_reach_both_pipelines_and_cache_reuse() {
    let (device, native) = device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) =
        multi_scene_at(true, true, crate::TextureEncoding::Srgb, [2, 0, 1]);
    let forward = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot.clone(),
        view,
        bounds(),
        &assets,
    ))
    .expect("permuted Forward bindings");
    assert!(native.command_calls.load(Ordering::SeqCst) > 0);
    block_on(renderer.readback(&forward)).expect("Forward completion");
    let pipeline = Counting {
        calls: Cell::new(0),
        declare: true,
    };
    let cold =
        block_on(renderer.submit_deferred(&pipeline, snapshot.clone(), view, bounds(), &assets))
            .expect("permuted Deferred bindings");
    block_on(renderer.readback(&cold)).expect("Deferred completion");
    let before = renderer.deferred_cache_stats();
    let warm = block_on(renderer.submit_deferred(&pipeline, snapshot, view, bounds(), &assets))
        .expect("permuted Deferred cache hit");
    assert!(renderer.deferred_cache_stats().texture_hits >= before.texture_hits + 2);
    assert!(renderer.deferred_cache_stats().gbuffer_program_hits > before.gbuffer_program_hits);
    assert_eq!(
        pipeline.calls.get(),
        2,
        "reuse consumes this frame's callback"
    );
    block_on(renderer.readback(&warm)).expect("warm completion");
    assert_eq!(renderer.pending(), 0);
}
