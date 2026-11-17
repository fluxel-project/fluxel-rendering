//! Completion ownership boundary for the multi-record Deferred witness.

use super::*;

#[test]
fn deferred_multi_record_completion_failure_retains_every_owner() {
    let (device, _) = device(Script::AcceptedCompletionDeviceLost);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) = multi_scene(true);
    let submission = block_on(renderer.submit_deferred(
        &DeferredCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
    ))
    .unwrap();
    let pending = renderer.pending_for(&submission).unwrap();
    let crate::renderer::PendingWork::Deferred { _frame, .. } = &pending.workload else {
        panic!("Deferred work")
    };
    assert_eq!(_frame.draws.len(), 2);
    assert!(matches!(
        block_on(renderer.readback(&submission)),
        Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert_eq!(
        renderer.pending(),
        1,
        "completion failure may not retire owners"
    );
}
