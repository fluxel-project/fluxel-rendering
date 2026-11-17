//! Combined capture-to-Surface completion ownership.

use super::*;
use crate::{BoundsVisibility, ForwardCpuPipeline};

#[test]
fn capture_surface_presents_after_one_six_scope_execution_and_retires_on_complete() {
    let (device, native) = capture_forward_tests::capture_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut surface = super::surface_tests::surface(&device, [8, 8]);
    let (snapshot, view, assets) = scene_and_assets();
    let (receiver, _) = capture_forward_tests::receiver();
    let submission = block_on(renderer.submit_forward_capture_surface(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        receiver,
        block_on(surface.acquire()).unwrap(),
    ))
    .unwrap();
    assert_eq!(renderer.pending(), 1);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 5);
    assert!(block_on(renderer.readback(&submission)).is_err());
    assert_eq!(renderer.pending(), 1);
    block_on(renderer.complete(&submission)).unwrap();
    assert_eq!(renderer.pending(), 0);
}

#[test]
fn capture_surface_rejected_present_after_composite_retains_returned_lease() {
    let (device, _) =
        capture_forward_tests::capture_device(Script::SurfacePresentRejectedAfterScopes);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut surface = super::surface_tests::surface(&device, [8, 8]);
    let (snapshot, view, assets) = scene_and_assets();
    let (receiver, _) = capture_forward_tests::receiver();
    assert!(matches!(
        block_on(renderer.submit_forward_capture_surface(
            &ForwardCpuPipeline::new(BoundsVisibility),
            snapshot,
            view,
            bounds(),
            &assets,
            receiver,
            block_on(surface.acquire()).unwrap(),
        )),
        Err(RendererError::RetainedExecution { .. })
    ));
    let pending = renderer.slots.iter().flatten().next().unwrap();
    let crate::renderer::PendingWork::CaptureForwardSurface { _present, .. } = &pending.workload
    else {
        panic!("capture surface pending")
    };
    assert!(
        _present.is_some(),
        "unconsumed present lease remains retained"
    );
    assert_eq!(renderer.pending(), 1);
}

#[test]
fn capture_surface_present_loss_retains_both_frames_and_composite_owner() {
    let (device, _) = capture_forward_tests::capture_device(Script::SurfacePresentDeviceLost);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut surface = super::surface_tests::surface(&device, [8, 8]);
    let (snapshot, view, assets) = scene_and_assets();
    let (receiver, _) = capture_forward_tests::receiver();
    assert!(matches!(
        block_on(renderer.submit_forward_capture_surface(
            &ForwardCpuPipeline::new(BoundsVisibility),
            snapshot,
            view,
            bounds(),
            &assets,
            receiver,
            block_on(surface.acquire()).unwrap(),
        )),
        Err(RendererError::RetainedExecution { .. })
    ));
    let pending = renderer.slots.iter().flatten().next().unwrap();
    let crate::renderer::PendingWork::CaptureForwardSurface {
        _capture,
        _receiver,
        _composite,
        ..
    } = &pending.workload
    else {
        panic!("capture surface pending")
    };
    assert!(!_capture.draws.is_empty());
    assert!(!_receiver.draws.is_empty());
    assert_eq!(renderer.pending(), 1);
}
