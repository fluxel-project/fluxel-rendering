//! Named texture-source admission failures over the actual Deferred submit path.

use super::*;

fn snapshot_with_texture_bindings(
    assets: &ForwardAssets,
    textures: Vec<crate::ForwardTextureBinding>,
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
            textures,
            samplers: Vec::new(),
            tint: [1.0; 4],
        },
        transform: RenderTransform::identity(),
        bounds: bounds(),
    }));
    let receipt = scene.apply(&changes).unwrap();
    (scene.snapshot(), receipt.inserted_views()[0])
}

fn bindings(assets: &ForwardAssets) -> Vec<crate::ForwardTextureBinding> {
    ["base_color", "normal_texture", "orm", "emissive"]
        .into_iter()
        .map(|name| {
            crate::ForwardTextureBinding::asset(
                name,
                SceneAssetRef::from_snapshot(&assets.textures[0]),
            )
        })
        .collect()
}

#[test]
fn named_texture_source_names_refuse_missing_duplicate_and_extra_before_gpu_work() {
    for mode in [0_u8, 1, 2] {
        let (device, native) = device(Script::AcceptedExact);
        let mut renderer = Renderer::new(device).unwrap();
        let (_, _, assets) = multi_scene(true);
        let mut sources = bindings(&assets);
        match mode {
            0 => {
                sources.pop();
            }
            1 => sources.push(crate::ForwardTextureBinding::asset(
                "base_color",
                SceneAssetRef::from_snapshot(&assets.textures[1]),
            )),
            2 => sources.push(crate::ForwardTextureBinding::asset(
                "not_declared",
                SceneAssetRef::from_snapshot(&assets.textures[1]),
            )),
            _ => unreachable!(),
        }
        let (snapshot, view) = snapshot_with_texture_bindings(&assets, sources);
        let before = (
            native.created_buffers.load(Ordering::SeqCst),
            native.created_textures.load(Ordering::SeqCst),
            native.command_calls.load(Ordering::SeqCst),
        );
        let result = block_on(renderer.submit_deferred(
            &DeferredCpuPipeline::new(BoundsVisibility),
            snapshot,
            view,
            bounds(),
            &assets,
        ));
        assert!(matches!(
            result,
            Err(RendererError::ForwardPreparation(
                crate::ForwardPreparationError::MissingTextureResource { .. }
                    | crate::ForwardPreparationError::DuplicateTextureResource { .. }
                    | crate::ForwardPreparationError::UnexpectedTextureResource { .. }
            ))
        ));
        assert_eq!(
            before,
            (
                native.created_buffers.load(Ordering::SeqCst),
                native.created_textures.load(Ordering::SeqCst),
                native.command_calls.load(Ordering::SeqCst),
            )
        );
    }
}

fn reject_nonfirst_actual_owner(
    label: &str,
    mutate: fn(&mut crate::residency::ForwardTexture<WitnessBackend>),
) {
    let (device, native) = device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (snapshot, view, assets) = multi_scene(true);
    let submission = block_on(renderer.submit_deferred(
        &DeferredCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
    ))
    .unwrap();
    {
        let pending = renderer.slots[submission.slot].as_mut().unwrap();
        let crate::renderer::PendingWork::Deferred { _frame, .. } = &mut pending.workload else {
            panic!("Deferred work")
        };
        let selected = _frame.draws[0].textures[2];
        assert_ne!(selected, _frame.draws[0].textures[0]);
        mutate(&mut Arc::get_mut(&mut _frame.resident).unwrap().textures[selected]);
    }
    let pending = renderer.pending_for(&submission).unwrap();
    let crate::renderer::PendingWork::Deferred { _frame, plan, .. } = &pending.workload else {
        panic!("Deferred work")
    };
    assert!(
        plan.validate_frame(_frame).is_err(),
        "actual owner mutation {label}"
    );
    let commands = native.command_calls.load(Ordering::SeqCst);
    let mut encoder = device.encoder(device.lanes()[0].id).unwrap();
    assert!(plan.record_geometry(&mut encoder, _frame).is_err());
    assert_eq!(commands, native.command_calls.load(Ordering::SeqCst));
}

#[test]
fn nonfirst_texture_format_extent_state_and_device_owners_refuse_before_recording() {
    reject_nonfirst_actual_owner("format", |texture| {
        texture.desc.format = fluxel_rhi::api::TextureFormat::Rgba8UnormSrgb;
    });
    reject_nonfirst_actual_owner("extent", |texture| texture.desc.extent[0] += 1);
    reject_nonfirst_actual_owner("layout", |texture| {
        texture.layout = fluxel_rhi::api::ImageLayout::ShaderReadOnly;
    });
    reject_nonfirst_actual_owner("device", |texture| {
        texture.device = fluxel_rhi::api::DeviceId::allocate().unwrap();
    });
}
