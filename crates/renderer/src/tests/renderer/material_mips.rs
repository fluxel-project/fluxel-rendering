//! Material mip generation through actual Renderer preparation and recording.
use super::*;
use crate::{BoundsVisibility, DeferredCpuPipeline, ForwardCpuPipeline};
use fluxel_assets::{AssetKind, AssetSnapshot};

fn snapshot<K: AssetKind, T: Send + Sync + 'static>(value: T) -> AssetSnapshot<K, T> {
    let store = AssetStore::<K, T, ()>::new();
    match store.acquire(&store.create().unwrap()).unwrap() {
        Acquire::Producer(producer) => producer.commit(value, ResidentBytes::new(0)).unwrap(),
        _ => unreachable!(),
    }
}

fn inputs() -> (
    crate::FrameSnapshot<ForwardResources>,
    crate::RenderViewId,
    ForwardAssets,
) {
    let (_, _, mut assets) = scene_and_assets();
    let image = snapshot::<ColorTextureAsset, _>(ColorTextureData {
        extent: [4, 4],
        mip_levels: 3,
        encoding: crate::TextureEncoding::Linear,
        rgba8: (0..16_u8)
            .flat_map(|x| [x * 11, 255 - x * 9, x * 7, 255])
            .collect(),
    });
    let material = snapshot::<MaterialAsset, _>(MaterialDefinition::new(
        vec![MaterialFragment { name: "mip_color".into(), inputs: vec![], output: MaterialType::Vec4,
            calls: vec![], wgsl: "fn mip_color() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0), 2) * tint; }".into() }],
        "mip_color", vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "material_tint".into() }],
        vec![MaterialResource { name: "material_tint".into(), kind: MaterialResourceKind::UniformVec4,
            location: BindingLocation { group: 0, binding: 0 } }, MaterialResource { name: "material_texture".into(),
            kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } }],
        BTreeMap::new(), SurfaceOutput::Color,
    ));
    assets.textures = vec![image];
    assets.materials = vec![material];
    let mut scene = RenderScene::new().unwrap();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [8, 8]).unwrap(),
    ));
    changes.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(&assets.geometry[0]),
            material: SceneAssetRef::from_snapshot(&assets.materials[0]),
            textures: vec![crate::ForwardTextureBinding::asset(
                "material_texture",
                SceneAssetRef::from_snapshot(&assets.textures[0]),
            )],
            samplers: Vec::new(),
            tint: [1.; 4],
        },
        transform: RenderTransform::identity(),
        bounds: bounds(),
    }));
    let receipt = scene.apply(&changes).unwrap();
    (scene.snapshot(), receipt.inserted_views()[0], assets)
}

fn submit_mips(
    renderer: &mut Renderer<WitnessBackend>,
    scene: &crate::FrameSnapshot<ForwardResources>,
    view: crate::RenderViewId,
    assets: &ForwardAssets,
    deferred: bool,
) -> std::result::Result<Submission, RendererError> {
    if deferred {
        block_on(renderer.submit_deferred(
            &DeferredCpuPipeline::new(BoundsVisibility),
            scene.clone(),
            view,
            bounds(),
            assets,
        ))
    } else {
        block_on(renderer.submit_forward(
            &ForwardCpuPipeline::new(BoundsVisibility),
            scene.clone(),
            view,
            bounds(),
            assets,
        ))
    }
}

#[test]
fn material_mips_generate_every_level_once_then_reuse_the_completed_chain_in_both_pipelines() {
    for deferred in [false, true] {
        let (device, native) = deferred_device(Script::AcceptedExact);
        let mut renderer = Renderer::new(device).unwrap();
        let (scene, view, assets) = inputs();
        let cold = submit_mips(&mut renderer, &scene, view, &assets, deferred).unwrap();
        assert_eq!(
            *native.blits.lock().unwrap(),
            vec![
                (0, 1, [4, 4, 1], [2, 2, 1], FilterMode::Linear),
                (1, 2, [2, 2, 1], [1, 1, 1], FilterMode::Linear),
            ]
        );
        block_on(renderer.readback(&cold)).unwrap();
        for _ in 0..2 {
            let warm = submit_mips(&mut renderer, &scene, view, &assets, deferred).unwrap();
            assert_eq!(
                native.blits.lock().unwrap().len(),
                2,
                "completed levels must not regenerate"
            );
            block_on(renderer.readback(&warm)).unwrap();
        }
        assert_eq!(renderer.pending(), 0);
    }
}

#[test]
fn material_mip_actual_owner_mutation_refuses_direct_blit_before_barriers_or_blits() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (scene, view, assets) = inputs();
    let submission = submit_mips(&mut renderer, &scene, view, &assets, false).unwrap();
    let pending = renderer.slots[submission.slot].as_mut().unwrap();
    let crate::renderer::PendingWork::Forward { _frame, plan, .. } = &mut pending.workload else {
        panic!("Forward")
    };
    assert!(plan.validate_frame(_frame).is_ok());
    let index = _frame.draws[0].textures[0];
    Arc::get_mut(&mut _frame.resident).unwrap().textures[index]
        .desc
        .mip_levels = 2;
    let counts = (
        native.transitions.load(Ordering::SeqCst),
        native.blits.lock().unwrap().len(),
    );
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let mut encoder = device.encoder(lane).unwrap();
    assert!(plan.record_mip(0, &mut encoder, _frame).is_err());
    assert_eq!(
        counts,
        (
            native.transitions.load(Ordering::SeqCst),
            native.blits.lock().unwrap().len()
        )
    );
}

#[test]
fn unsupported_mip_blit_retains_pending_owners_and_never_reuses_an_incomplete_chain() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (scene, view, assets) = inputs();
    native.blit_supported.store(false, Ordering::SeqCst);
    assert!(matches!(
        submit_mips(&mut renderer, &scene, view, &assets, false),
        Err(RendererError::RetainedExecution {
            recording_error: Some(Error::Unsupported(_)),
            ..
        })
    ));
    assert_eq!(renderer.pending(), 1);
    assert!(native.blits.lock().unwrap().is_empty());
    assert_eq!(native.texture_uploads.lock().unwrap().len(), 1);
    native.blit_supported.store(true, Ordering::SeqCst);
    let retry = submit_mips(&mut renderer, &scene, view, &assets, false).unwrap();
    assert_eq!(native.blits.lock().unwrap().len(), 2);
    assert_eq!(renderer.forward_cache_stats().texture_hits, 0);
    block_on(renderer.readback(&retry)).unwrap();
    assert_eq!(
        renderer.pending(),
        1,
        "first uncertain submission remains independently retained"
    );
    block_on(renderer.shutdown()).unwrap();
}

#[test]
fn completed_submission_with_execution_error_does_not_publish_mip_assets() {
    let (device, _) = deferred_device(Script::AcceptedWithError);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (scene, view, assets) = inputs();
    assert!(matches!(
        submit_mips(&mut renderer, &scene, view, &assets, false),
        Err(RendererError::RetainedExecution { .. })
    ));
    let slot = renderer.slots.iter().position(Option::is_some).unwrap();
    let serial = renderer.slots[slot].as_ref().unwrap().serial;
    let submission = Submission {
        owner: Arc::clone(&renderer.owner),
        device: device.id(),
        slot,
        serial,
    };
    block_on(renderer.readback(&submission)).unwrap();
    assert_eq!(renderer.pending(), 0);
    let key =
        crate::residency::ForwardTextureCacheKey::from_snapshot(&assets.textures[0], device.id());
    assert!(renderer.forward_residency.texture(key).is_none());
}

#[test]
fn material_mip_count_exceeding_actual_device_support_refuses_before_texture_creation() {
    for deferred in [false, true] {
        let (device, native) = deferred_device(Script::AcceptedExact);
        native.texture_max_mips.store(2, Ordering::SeqCst);
        let mut renderer = Renderer::new(device).unwrap();
        let (scene, view, assets) = inputs();
        assert!(submit_mips(&mut renderer, &scene, view, &assets, deferred).is_err());
        assert_eq!(native.created_textures.load(Ordering::SeqCst), 0);
        assert!(native.blits.lock().unwrap().is_empty());
        assert_eq!(renderer.pending(), 0);
    }
}

#[test]
fn captured_material_assets_generate_on_both_sides_and_share_only_completed_mips() {
    let (device, native) = capture_forward_tests::capture_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (scene, view, mut assets) = inputs();
    let receiver = capture_forward_tests::mixed_fixture::receiver(&mut assets, false);
    let cold = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        scene.clone(),
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("producer_capture").unwrap(),
        receiver.clone(),
    ))
    .unwrap();
    assert_eq!(
        native.blits.lock().unwrap().len(),
        4,
        "each independently allocated cold asset chain must be generated"
    );
    block_on(renderer.readback(&cold)).unwrap();
    let warm = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        scene,
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("producer_capture").unwrap(),
        receiver,
    ))
    .unwrap();
    assert_eq!(
        native.blits.lock().unwrap().len(),
        4,
        "shared complete chains are not regenerated"
    );
    block_on(renderer.readback(&warm)).unwrap();
}

#[test]
fn surface_material_mip_steps_are_covered_by_present_completion_and_warm_reuse() {
    for deferred in [false, true] {
        let (device, native) = deferred_device(Script::AcceptedExact);
        let mut surface = surface_tests::surface(&device, [8, 8]);
        let mut renderer = Renderer::new(device).unwrap();
        let (scene, view, assets) = inputs();
        for _ in 0..2 {
            let image = block_on(surface.acquire()).unwrap();
            let submitted = if deferred {
                block_on(renderer.submit_deferred_surface(
                    &DeferredCpuPipeline::new(BoundsVisibility),
                    scene.clone(),
                    view,
                    bounds(),
                    &assets,
                    image,
                ))
            } else {
                block_on(renderer.submit_forward_surface(
                    &ForwardCpuPipeline::new(BoundsVisibility),
                    scene.clone(),
                    view,
                    bounds(),
                    &assets,
                    image,
                ))
            }
            .unwrap();
            assert_eq!(native.blits.lock().unwrap().len(), 2);
            block_on(renderer.complete(&submitted)).unwrap();
            assert_eq!(renderer.pending(), 0);
        }
    }
}

#[test]
fn captured_surface_assets_include_generated_levels_in_the_single_submission() {
    let (device, native) = capture_forward_tests::capture_device(Script::AcceptedExact);
    let mut surface = surface_tests::surface(&device, [8, 8]);
    let mut renderer = Renderer::new(device).unwrap();
    let (scene, view, mut assets) = inputs();
    let receiver = capture_forward_tests::mixed_fixture::receiver(&mut assets, false);
    let submitted = block_on(renderer.submit_forward_capture_surface(
        &ForwardCpuPipeline::new(BoundsVisibility),
        scene,
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("producer_capture").unwrap(),
        receiver,
        block_on(surface.acquire()).unwrap(),
    ))
    .unwrap();
    assert_eq!(native.blits.lock().unwrap().len(), 4);
    block_on(renderer.complete(&submitted)).unwrap();
    assert_eq!(renderer.pending(), 0);
}

#[test]
fn failed_mip_completion_retains_the_generated_owner_without_cache_publication() {
    let (device, native) = deferred_device(Script::AcceptedCompletionDeviceLost);
    let key_device = device.id();
    let mut renderer = Renderer::new(device).unwrap();
    let (scene, view, assets) = inputs();
    let submitted = submit_mips(&mut renderer, &scene, view, &assets, false).unwrap();
    assert_eq!(native.blits.lock().unwrap().len(), 2);
    assert!(block_on(renderer.readback(&submitted)).is_err());
    assert_eq!(renderer.pending(), 1);
    let key =
        crate::residency::ForwardTextureCacheKey::from_snapshot(&assets.textures[0], key_device);
    assert!(renderer.forward_residency.texture(key).is_none());
}
