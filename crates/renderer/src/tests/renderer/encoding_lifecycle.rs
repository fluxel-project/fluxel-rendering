//! Encoding and completed-owner witnesses over real submission paths.

use super::*;

#[test]
fn wrong_encoding_in_a_completed_cache_owner_is_rejected_before_recording() {
    reject_cached_owner_mutation(|desc| desc.format = TextureFormat::Rgba8Unorm);
}

#[test]
fn wrong_extent_in_a_completed_cache_owner_is_rejected_before_recording() {
    reject_cached_owner_mutation(|desc| desc.extent[0] += 1);
}

fn reject_cached_owner_mutation(mutate: fn(&mut fluxel_rhi::api::TextureDesc)) {
    let (device, native) = device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (snapshot, view, assets) =
        multi_scene_with_effects_and_encoding(true, true, crate::TextureEncoding::Srgb);
    let pipeline = crate::DeferredCpuPipeline::new(BoundsVisibility);
    let cold =
        block_on(renderer.submit_deferred(&pipeline, snapshot.clone(), view, bounds(), &assets))
            .expect("cold encoded frame");
    block_on(renderer.readback(&cold)).expect("publish completed owners");
    let key =
        crate::residency::ForwardTextureCacheKey::from_snapshot(&assets.textures[0], device.id());
    let mut wrong = renderer
        .deferred_residency
        .texture(key)
        .expect("completed owner");
    mutate(&mut wrong.desc);
    renderer.deferred_residency = Default::default();
    assert!(renderer.deferred_residency.publish_texture(key, &wrong));
    let before = native.command_calls.load(Ordering::SeqCst);
    assert!(matches!(
        block_on(renderer.submit_deferred(&pipeline, snapshot, view, bounds(), &assets)),
        Err(RendererError::Rhi(Error::InvalidState))
    ));
    assert_eq!(native.command_calls.load(Ordering::SeqCst), before);
}

#[test]
fn encoded_assets_keep_srgb_owners_through_forward_deferred_and_deferred_cache_hits() {
    let (device, _) = device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) =
        multi_scene_with_effects_and_encoding(true, true, crate::TextureEncoding::Srgb);

    let forward = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot.clone(),
        view,
        bounds(),
        &assets,
    ))
    .expect("sRGB Forward frame");
    let pending = renderer.pending_for(&forward).expect("Forward pending");
    let crate::renderer::PendingWork::Forward { _frame, .. } = &pending.workload else {
        panic!("Forward work")
    };
    assert!(
        _frame
            .resident
            .textures
            .iter()
            .all(|texture| texture.desc.format == TextureFormat::Rgba8UnormSrgb)
    );
    block_on(renderer.readback(&forward)).expect("Forward completion");

    let pipeline = Counting {
        calls: Cell::new(0),
        declare: true,
    };
    let cold =
        block_on(renderer.submit_deferred(&pipeline, snapshot.clone(), view, bounds(), &assets))
            .expect("sRGB Deferred frame");
    block_on(renderer.readback(&cold)).expect("Deferred completion");
    let warm = block_on(renderer.submit_deferred(&pipeline, snapshot, view, bounds(), &assets))
        .expect("sRGB Deferred cache hit");
    let pending = renderer.pending_for(&warm).expect("Deferred pending");
    let crate::renderer::PendingWork::Deferred { _frame, .. } = &pending.workload else {
        panic!("Deferred work")
    };
    assert!(
        _frame
            .resident
            .textures
            .iter()
            .all(|texture| texture.desc.format == TextureFormat::Rgba8UnormSrgb)
    );
    assert!(renderer.deferred_cache_stats().texture_hits >= 2);
}
