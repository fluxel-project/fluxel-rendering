use super::*;
use fluxel_rhi::api::Error;
use std::sync::atomic::Ordering;

#[test]
fn successful_shutdown_waits_idle_then_releases_pending_owners_and_plans() {
    let (device, native) = fixture(Script::AcceptedExact);
    let (source_snapshot, weak) = snapshot();
    let mut renderer = Renderer::new(device).unwrap();
    let _submission = submit(&mut renderer, source_snapshot).unwrap();

    let report = block_on(renderer.shutdown()).unwrap();
    assert_eq!(report.retired_submissions, 1);
    assert_eq!(report.released_cached_plans, 1);
    assert_eq!(renderer.pending(), 0);
    assert_eq!(renderer.compilations(), 0);
    assert_eq!(renderer.plan_id(), None);
    assert!(renderer.is_shutdown());
    assert!(weak.upgrade().is_none());
    assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 2);
    assert_eq!(native.idle_waits.load(Ordering::SeqCst), 1);
    assert!(matches!(
        submit(&mut renderer, snapshot().0),
        Err(RendererError::RendererClosed)
    ));

    assert_eq!(
        block_on(renderer.shutdown()).unwrap(),
        ShutdownReport::default()
    );
    assert_eq!(native.idle_waits.load(Ordering::SeqCst), 1);
}

#[test]
fn failed_idle_keeps_owners_and_refuses_new_work() {
    let (device, native) = fixture(Script::AcceptedIdleFailure);
    let (source_snapshot, weak) = snapshot();
    let mut renderer = Renderer::new(device).unwrap();
    let _submission = submit(&mut renderer, source_snapshot).unwrap();

    assert!(matches!(
        block_on(renderer.shutdown()),
        Err(RendererError::Rhi(Error::Backend(_)))
    ));
    assert_eq!(renderer.pending(), 1);
    assert_eq!(renderer.compilations(), 1);
    assert!(!renderer.is_shutdown());
    assert!(weak.upgrade().is_some());
    assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 0);
    assert_eq!(native.idle_waits.load(Ordering::SeqCst), 1);
    assert!(matches!(
        submit(&mut renderer, snapshot().0),
        Err(RendererError::RendererClosed)
    ));
}

#[test]
fn cancelled_idle_wait_keeps_owners_and_refuses_new_work() {
    let (device, native) = fixture(Script::AcceptedIdlePending);
    let (source_snapshot, weak) = snapshot();
    let mut renderer = Renderer::new(device).unwrap();
    let _submission = submit(&mut renderer, source_snapshot).unwrap();

    poll_once_pending(renderer.shutdown());
    assert_eq!(renderer.pending(), 1);
    assert_eq!(renderer.compilations(), 1);
    assert!(!renderer.is_shutdown());
    assert!(weak.upgrade().is_some());
    assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 0);
    assert_eq!(native.idle_waits.load(Ordering::SeqCst), 1);
    assert!(matches!(
        submit(&mut renderer, snapshot().0),
        Err(RendererError::RendererClosed)
    ));
}

#[test]
fn device_loss_during_idle_quarantines_owners_and_blocks_old_readback() {
    let (device, native) = fixture(Script::AcceptedIdleDeviceLost);
    let (source_snapshot, weak) = snapshot();
    let mut renderer = Renderer::new(device).unwrap();
    let submission = submit(&mut renderer, source_snapshot).unwrap();

    assert!(matches!(
        block_on(renderer.shutdown()),
        Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert_eq!(renderer.pending(), 1);
    assert_eq!(renderer.compilations(), 0);
    assert!(!renderer.is_shutdown());
    assert!(weak.upgrade().is_some());
    assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 0);
    assert!(matches!(
        block_on(renderer.readback(&submission)),
        Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert!(native.waited_completions.lock().unwrap().is_empty());
}

#[test]
fn device_loss_while_waiting_readback_quarantines_pending_owner() {
    let (device, native) = fixture(Script::AcceptedCompletionDeviceLost);
    let (source_snapshot, weak) = snapshot();
    let mut renderer = Renderer::new(device).unwrap();
    let submission = submit(&mut renderer, source_snapshot).unwrap();

    assert!(matches!(
        block_on(renderer.readback(&submission)),
        Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert_eq!(renderer.pending(), 1);
    assert_eq!(renderer.compilations(), 0);
    assert!(weak.upgrade().is_some());
    assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 0);
    assert!(matches!(
        submit(&mut renderer, snapshot().0),
        Err(RendererError::RendererClosed)
    ));
    assert!(matches!(
        block_on(renderer.readback(&submission)),
        Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert_eq!(*native.waited_completions.lock().unwrap(), [1]);
}

#[test]
fn device_loss_during_readback_releases_completed_frame_and_quarantines_other_frames() {
    let (device, native) = fixture(Script::AcceptedReadbackDeviceLost);
    let (first_snapshot, first_weak) = snapshot();
    let (second_snapshot, second_weak) = snapshot();
    let mut renderer = Renderer::new(device).unwrap();
    let first = submit(&mut renderer, first_snapshot).unwrap();
    let second = submit(&mut renderer, second_snapshot).unwrap();

    assert!(matches!(
        block_on(renderer.readback(&first)),
        Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert_eq!(renderer.pending(), 1);
    assert_eq!(renderer.compilations(), 0);
    assert!(first_weak.upgrade().is_none());
    assert!(second_weak.upgrade().is_some());
    assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 2);
    assert!(matches!(
        submit(&mut renderer, snapshot().0),
        Err(RendererError::RendererClosed)
    ));
    assert!(matches!(
        block_on(renderer.readback(&second)),
        Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert_eq!(*native.waited_completions.lock().unwrap(), [1]);
}
