//! Device and actual-owner boundaries for completion-gated Deferred reuse.

use super::*;

use std::sync::{Arc, atomic::Ordering};

use fluxel_rhi::api::Error;

fn complete_cached_frame(
    renderer: &mut Renderer<WitnessBackend>,
    pipeline: &CountingDeferredPipeline,
    snapshot: crate::FrameSnapshot<ForwardResources>,
    view: crate::RenderViewId,
    assets: &ForwardAssets,
) {
    let submission = submit_with_counting_pipeline(renderer, pipeline, snapshot, view, assets)
        .expect("Deferred submission");
    block_on(renderer.readback(&submission)).expect("Deferred readback");
}

#[test]
fn deferred_completed_products_are_device_isolated_and_old_device_keeps_its_hit() {
    let (old_device, _) = deferred_device(Script::AcceptedExact);
    let (new_device, _) = deferred_device(Script::AcceptedExact);
    let mut old_renderer = Renderer::new(old_device).expect("old renderer");
    let mut new_renderer = Renderer::new(new_device).expect("new renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let old_pipeline = CountingDeferredPipeline::complete();
    let new_pipeline = CountingDeferredPipeline::complete();

    complete_cached_frame(
        &mut old_renderer,
        &old_pipeline,
        snapshot.clone(),
        view,
        &assets,
    );
    let old_before_warm = old_renderer.deferred_cache_stats();
    complete_cached_frame(
        &mut old_renderer,
        &old_pipeline,
        snapshot.clone(),
        view,
        &assets,
    );
    assert!(old_renderer.deferred_cache_stats().geometry_hits > old_before_warm.geometry_hits);

    let new_before = new_renderer.deferred_cache_stats();
    complete_cached_frame(
        &mut new_renderer,
        &new_pipeline,
        snapshot.clone(),
        view,
        &assets,
    );
    assert_eq!(
        new_renderer.deferred_cache_stats().geometry_hits,
        new_before.geometry_hits
    );
    assert_eq!(
        new_renderer.deferred_cache_stats().texture_hits,
        new_before.texture_hits
    );
    let new_after = new_renderer.deferred_cache_stats();
    assert_eq!(
        new_after.gbuffer_program_hits,
        new_before.gbuffer_program_hits
    );
    assert_eq!(
        new_after.lighting_program_hits,
        new_before.lighting_program_hits
    );
    assert_eq!(new_after.graph_hits, new_before.graph_hits);
    assert_eq!(new_after.frame_reuses, new_before.frame_reuses);

    let old_before_return = old_renderer.deferred_cache_stats();
    complete_cached_frame(&mut old_renderer, &old_pipeline, snapshot, view, &assets);
    let old_after_return = old_renderer.deferred_cache_stats();
    assert!(old_after_return.geometry_hits > old_before_return.geometry_hits);
    assert!(old_after_return.texture_hits > old_before_return.texture_hits);
}

#[test]
fn deferred_device_loss_clears_completed_caches_but_retains_pending_owner() {
    let (device, native) = deferred_device(Script::AcceptedCompletionDeviceLost);
    native.fault_enabled.store(false, Ordering::SeqCst);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let pipeline = CountingDeferredPipeline::complete();

    complete_cached_frame(&mut renderer, &pipeline, snapshot.clone(), view, &assets);
    complete_cached_frame(&mut renderer, &pipeline, snapshot.clone(), view, &assets);
    assert!(!renderer.deferred_plans.is_empty());
    assert!(
        renderer
            .deferred_residency
            .geometry(
                crate::residency::ForwardResidentCache::<WitnessBackend>::geometry_key(
                    &prepare_forward(
                        &ForwardCpuPipeline::new(BoundsVisibility)
                            .prepare(&snapshot, view, bounds())
                            .expect("prepared frame"),
                        &assets,
                        &mut fluxel_shader::ShaderComposer::new(),
                    )
                    .expect("prepared draw")[0],
                    renderer.device.id(),
                )
            )
            .is_some()
    );
    assert!(renderer.deferred_frames.iter().any(Option::is_some));

    let pending =
        submit_with_counting_pipeline(&mut renderer, &pipeline, snapshot.clone(), view, &assets)
            .expect("accepted pending Deferred frame");
    native.fault_enabled.store(true, Ordering::SeqCst);
    assert!(matches!(
        block_on(renderer.readback(&pending)),
        Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert_eq!(renderer.pending(), 1);
    assert!(renderer.deferred_plans.is_empty());
    assert!(
        renderer
            .deferred_residency
            .geometry(
                crate::residency::ForwardResidentCache::<WitnessBackend>::geometry_key(
                    &prepare_forward(
                        &ForwardCpuPipeline::new(BoundsVisibility)
                            .prepare(&snapshot, view, bounds())
                            .expect("prepared frame"),
                        &assets,
                        &mut fluxel_shader::ShaderComposer::new(),
                    )
                    .expect("prepared draw")[0],
                    renderer.device.id(),
                )
            )
            .is_none()
    );
    assert!(renderer.deferred_frames.iter().all(Option::is_none));

    let before = counters(&native);
    assert!(matches!(
        submit_with_counting_pipeline(&mut renderer, &pipeline, snapshot, view, &assets),
        Err(RendererError::RendererClosed)
    ));
    assert_eq!(counters(&native), before);
}

#[test]
fn cached_deferred_plan_refuses_foreign_and_wrong_gbuffer_owners_before_work() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let pipeline = CountingDeferredPipeline::complete();
    complete_cached_frame(&mut renderer, &pipeline, snapshot.clone(), view, &assets);
    complete_cached_frame(&mut renderer, &pipeline, snapshot.clone(), view, &assets);
    let plan = Arc::clone(renderer.deferred_plans.last().expect("cached plan"));
    let mut cached_frame = prepared_deferred_frame(&mut renderer, &snapshot, view, &assets);
    assert!(cached_frame.resident_upload.bytes.is_empty());
    plan.validate_frame(&cached_frame)
        .expect("compatible warm frame before owner replacement");
    let original_resident = Arc::clone(&cached_frame.resident);

    let (other_device, _) = deferred_device(Script::AcceptedExact);
    let mut other_renderer = Renderer::new(other_device).expect("other renderer");
    let foreign_frame = prepared_deferred_frame(&mut other_renderer, &snapshot, view, &assets);
    cached_frame.resident = foreign_frame.resident;
    let before_foreign = counters(&native);
    assert!(matches!(
        plan.validate_frame(&cached_frame),
        Err(Error::WrongDevice)
    ));
    assert_eq!(counters(&native), before_foreign);

    cached_frame.resident = original_resident;
    plan.validate_frame(&cached_frame)
        .expect("restored warm frame before descriptor replacement");
    let mut wrong_descriptor = cached_frame;
    let [_, _, depth] = crate::residency::plan_deferred_gbuffer(&device, wrong_descriptor.extent)
        .expect("G-buffer plan")
        .allocate(&device)
        .expect("G-buffer allocation");
    wrong_descriptor.gbuffer[0] = depth;
    let before_descriptor = counters(&native);
    assert!(matches!(
        plan.validate_frame(&wrong_descriptor),
        Err(Error::InvalidState)
    ));
    assert_eq!(counters(&native), before_descriptor);
}
