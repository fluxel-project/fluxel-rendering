//! Presentation rejection and uncertain execution ownership for Deferred.

use super::*;
use fluxel_rhi::api::ExecutionStatus;

#[test]
fn deferred_surface_clean_present_rejection_releases_only_with_no_effect_evidence() {
    let (device, native) = deferred_device(Script::SurfacePresentRejected);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut target = surface(&device, [8, 8]);
    let (snapshot, view, assets) = scene_and_assets();
    let result = submit_surface(
        &mut renderer,
        snapshot,
        view,
        &assets,
        block_on(target.acquire()).unwrap(),
    );
    assert!(matches!(
        result,
        Err(RendererError::Rhi(Error::SurfaceOutdated))
    ));
    assert_eq!(renderer.pending(), 0);
    assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
    let allocated = native.created_buffers.load(Ordering::SeqCst);
    assert!(allocated > 0);
    assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), allocated);
}

#[test]
fn deferred_surface_present_loss_keeps_owners_and_closes_backend_reentry() {
    let (device, native) = deferred_device(Script::SurfacePresentDeviceLost);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut target = surface(&device, [8, 8]);
    let (snapshot, view, assets) = scene_and_assets();
    assert!(matches!(
        submit_surface(
            &mut renderer,
            snapshot.clone(),
            view,
            &assets,
            block_on(target.acquire()).unwrap()
        ),
        Err(RendererError::RetainedExecution {
            status: ExecutionStatus::Indeterminate,
            recording_error: Some(Error::DeviceLost),
            ..
        })
    ));
    assert_eq!(renderer.pending(), 1);
    assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 0);
    let image = block_on(target.acquire()).unwrap();
    let before = counters(&native);
    assert!(matches!(
        submit_surface(&mut renderer, snapshot, view, &assets, image),
        Err(RendererError::RendererClosed)
    ));
    assert_eq!(counters(&native), before);
}

#[test]
fn deferred_surface_partial_execution_retires_only_after_successful_idle() {
    for script in [
        Script::PartialAccepted,
        Script::PartialIssued,
        Script::Indeterminate,
    ] {
        let (device, native) = deferred_device(script);
        let mut renderer = Renderer::new(device.clone()).unwrap();
        let mut target = surface(&device, [8, 8]);
        let (snapshot, view, assets) = scene_and_assets();
        assert!(matches!(
            submit_surface(
                &mut renderer,
                snapshot,
                view,
                &assets,
                block_on(target.acquire()).unwrap()
            ),
            Err(RendererError::RetainedExecution { .. })
        ));
        assert_eq!(renderer.pending(), 1);
        assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 0);
        let report = block_on(renderer.shutdown()).expect("idle proves GPU retirement");
        assert_eq!(report.retired_submissions, 1);
        assert_eq!(renderer.pending(), 0);
        assert_eq!(native.idle_waits.load(Ordering::SeqCst), 1);
        assert_eq!(
            native.dropped_buffers.load(Ordering::SeqCst),
            native.created_buffers.load(Ordering::SeqCst)
        );
    }
}
