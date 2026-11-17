//! Actual-owner refusal tests for cached Forward depth bindings.

use super::forward_depth_selection_tests::{bounds, forward_device, scene_and_assets};
use super::*;
use super::{
    backend::{Native, Script, WitnessBackend},
    support::block_on,
};

use crate::{BoundsVisibility, ForwardCpuPipeline, Renderer, Submission};

fn pending_depth(script: Script) -> (Renderer<WitnessBackend>, Submission, Arc<Native>) {
    let (device, native) = forward_device(script);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let submission = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility).with_depth32_less_equal(),
        snapshot,
        view,
        bounds(),
        &assets,
    ))
    .expect("depth Forward submission");
    (renderer, submission, native)
}

fn assert_pending_depth_refuses_before_commands(
    renderer: &Renderer<WitnessBackend>,
    submission: &Submission,
    native: &Native,
) {
    let device = renderer.device.clone();
    let pending = renderer
        .pending_for(submission)
        .expect("pending Forward frame");
    let crate::renderer::PendingWork::Forward { _frame, plan, .. } = &pending.workload else {
        panic!("ordinary Forward pending work")
    };
    assert!(
        plan.validate_frame(&_frame).is_err(),
        "cached plan must refuse owner"
    );
    let before = native.command_calls.load(Ordering::SeqCst);
    let lane = device.lanes()[0].id;
    let mut encoder = device.encoder(lane).expect("test encoder");
    assert!(plan.record_raster(&mut encoder, &_frame).is_err());
    assert_eq!(
        native.command_calls.load(Ordering::SeqCst),
        before,
        "binding refusal precedes execution commands"
    );
}

#[test]
fn cached_plan_rejects_mutated_depth_descriptor_and_state_before_recording() {
    for mutation in [
        |depth: &mut crate::residency::ForwardDepthAttachment<WitnessBackend>| {
            depth.desc.usage = TextureUsage::COPY_DST;
        },
        |depth: &mut crate::residency::ForwardDepthAttachment<WitnessBackend>| {
            depth.desc.extent = [7, 8, 1];
        },
        |depth: &mut crate::residency::ForwardDepthAttachment<WitnessBackend>| {
            depth.desc.format = TextureFormat::Rgba8Unorm;
        },
        |depth: &mut crate::residency::ForwardDepthAttachment<WitnessBackend>| {
            depth.incoming = TextureUses {
                stages: StageMask::FRAGMENT,
                accesses: AccessMask::SHADER_READ,
            };
        },
        |depth: &mut crate::residency::ForwardDepthAttachment<WitnessBackend>| {
            depth.layout = ImageLayout::General;
        },
    ] {
        let (mut renderer, submission, native) = pending_depth(Script::AcceptedExact);
        let pending = renderer.slots[submission.slot]
            .as_mut()
            .expect("submitted Forward frame");
        let crate::renderer::PendingWork::Forward { _frame, .. } = &mut pending.workload else {
            panic!("ordinary Forward pending work")
        };
        mutation(
            _frame
                .frame
                .depth
                .as_mut()
                .expect("depth-enabled pending frame"),
        );
        assert_pending_depth_refuses_before_commands(&renderer, &submission, &native);
    }
}

#[test]
fn cached_plan_rejects_foreign_actual_depth_owner_before_recording() {
    let (mut renderer, submission, native) = pending_depth(Script::AcceptedExact);
    let (other_device, _) = forward_device(Script::AcceptedExact);
    let pending = renderer.slots[submission.slot]
        .as_mut()
        .expect("submitted Forward frame");
    let crate::renderer::PendingWork::Forward { _frame, .. } = &mut pending.workload else {
        panic!("ordinary Forward pending work")
    };
    _frame.frame.depth = Some(
        crate::residency::ForwardDepthAttachment::allocate(&other_device, _frame.extent)
            .expect("foreign depth owner"),
    );

    assert_pending_depth_refuses_before_commands(&renderer, &submission, &native);
}

#[test]
fn cached_plan_accepts_a_fresh_depth_instance_when_actual_facts_match() {
    let (mut renderer, submission, _) = pending_depth(Script::AcceptedExact);
    let pending = renderer.slots[submission.slot]
        .as_mut()
        .expect("submitted Forward frame");
    let crate::renderer::PendingWork::Forward { _frame, .. } = &mut pending.workload else {
        panic!("ordinary Forward pending work")
    };
    _frame
        .frame
        .depth
        .as_mut()
        .expect("depth-enabled pending frame")
        .instance = crate::graph::ResourceInstanceId::fresh().expect("fresh instance");

    let pending = renderer
        .pending_for(&submission)
        .expect("pending Forward frame");
    let crate::renderer::PendingWork::Forward { _frame, plan, .. } = &pending.workload else {
        panic!("ordinary Forward pending work")
    };
    assert!(plan.validate_frame(&_frame).is_ok());
}
