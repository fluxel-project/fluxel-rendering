//! Actual-owner rejection tests for the combined capture plan.

use super::*;
use crate::{BoundsVisibility, ForwardCpuPipeline};

fn pending_capture(script: Script) -> (Renderer<WitnessBackend>, Submission) {
    let (device, _) = capture_forward_tests::capture_device(script);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) = scene_and_assets();
    let (receiver, _) = capture_forward_tests::receiver();
    let submission = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        receiver,
    ))
    .unwrap();
    (renderer, submission)
}

#[test]
fn capture_plan_rejects_a_mutated_receiver_texture_owner_before_recording() {
    let (mut renderer, submission) = pending_capture(Script::AcceptedExact);
    let pending = renderer.slots[submission.slot].as_mut().unwrap();
    let crate::renderer::PendingWork::CaptureForward { _receiver, .. } = &mut pending.workload
    else {
        panic!("capture pending work")
    };
    let resident = Arc::get_mut(&mut _receiver.resident).expect("pending owns receiver resident");
    resident.textures[0].instance = crate::graph::ResourceInstanceId::fresh().unwrap();
    let pending = renderer.pending_for(&submission).unwrap();
    let crate::renderer::PendingWork::CaptureForward {
        _capture,
        _receiver,
        plan,
        ..
    } = &pending.workload
    else {
        panic!("capture pending work")
    };
    assert!(plan.validate_frames(_capture, _receiver).is_err());
}

#[test]
fn capture_receiver_raster_refuses_mutated_actual_texture_usage() {
    let (mut renderer, submission) = pending_capture(Script::AcceptedExact);
    let pending = renderer.slots[submission.slot].as_mut().unwrap();
    let crate::renderer::PendingWork::CaptureForward { _receiver, .. } = &mut pending.workload
    else {
        panic!("capture pending work")
    };
    let resident = Arc::get_mut(&mut _receiver.resident).expect("pending owns receiver resident");
    resident.textures[0].desc.usage = TextureUsage::COPY_DST;
    let device = renderer.device.clone();
    let pending = renderer.pending_for(&submission).unwrap();
    let crate::renderer::PendingWork::CaptureForward {
        _capture,
        _receiver,
        plan,
        ..
    } = &pending.workload
    else {
        panic!("capture pending work")
    };
    let mut execution =
        unsafe { device.begin_execution(plan.submission().clone(), CompletionRequest::End) }
            .unwrap();
    assert!(
        execution
            .commands(plan.receiver_raster_step(), |encoder| {
                plan.record_receiver_raster(encoder, _capture, _receiver)
            })
            .is_err()
    );
    let _ = unsafe { execution.finish() };
}
