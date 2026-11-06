//! Forward pending-owner isolation and pre-submission upload cleanup tests.

use super::*;

use crate::{
    renderer::forward::ForwardPublication,
    residency::{ForwardFrame, ForwardFrameBuffers, ForwardResidentBuffers},
};

fn blank_frame(device: &Device<WitnessBackend>) -> ForwardFrame<WitnessBackend> {
    let (resident, resident_upload) =
        ForwardResidentBuffers::allocate(device, Vec::new(), Vec::new()).unwrap();
    let uniform_bytes = vec![0; 16];
    let (frame, uniform_upload) =
        ForwardFrameBuffers::allocate(device, [1, 1], 16, uniform_bytes.clone(), 0..16, 256)
            .unwrap();
    ForwardFrame {
        resident,
        resident_upload,
        frame,
        uniform_upload,
        uniform_bytes,
        draws: Vec::new(),
        draw_batches: Vec::new(),
        extent: [1, 1],
        readback_row_pitch: 256,
    }
}

fn execute(
    renderer: &mut Renderer<WitnessBackend>,
    retained: crate::renderer::PendingInputs,
    frame: ForwardFrame<WitnessBackend>,
) -> std::result::Result<Submission, RendererError> {
    block_on(renderer.execute_forward(retained, frame, ForwardPublication::default()))
}

fn owner_instances(frame: &ForwardFrame<WitnessBackend>) -> [crate::graph::ResourceInstanceId; 4] {
    [
        frame.frame.upload_instance,
        frame.frame.uniform_instance,
        frame.frame.target_instance,
        frame.frame.readback_instance,
    ]
}

fn pooled_instances(
    buffers: &crate::residency::ForwardFrameBuffers<WitnessBackend>,
) -> [crate::graph::ResourceInstanceId; 4] {
    [
        buffers.upload_instance,
        buffers.uniform_instance,
        buffers.target_instance,
        buffers.readback_instance,
    ]
}

#[test]
fn forward_three_pending_slots_isolate_real_owners_and_recycle_only_completed_slot() {
    let (device, _) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();

    let retained = [Arc::new(vec![1]), Arc::new(vec![2]), Arc::new(vec![3])];
    let weak = retained.each_ref().map(Arc::downgrade);
    let first_frame = blank_frame(&device);
    let first_expected = owner_instances(&first_frame);
    let first = execute(
        &mut renderer,
        crate::renderer::PendingInputs::Bytes {
            _bytes: Arc::clone(&retained[0]),
        },
        first_frame,
    )
    .unwrap();
    let second_frame = blank_frame(&device);
    let second_expected = owner_instances(&second_frame);
    let second = execute(
        &mut renderer,
        crate::renderer::PendingInputs::Bytes {
            _bytes: Arc::clone(&retained[1]),
        },
        second_frame,
    )
    .unwrap();
    let third_frame = blank_frame(&device);
    let third_expected = owner_instances(&third_frame);
    let third = execute(
        &mut renderer,
        crate::renderer::PendingInputs::Bytes {
            _bytes: Arc::clone(&retained[2]),
        },
        third_frame,
    )
    .unwrap();
    drop(retained);

    assert_ne!(first_expected, second_expected);
    assert_ne!(second_expected, third_expected);
    assert_ne!(first_expected, third_expected);

    // The submissions are still the actual owners: no completed frame may
    // enter the reusable pool merely because another slot was submitted.
    assert_eq!(renderer.pending(), 3);
    assert_ne!(first.slot, second.slot);
    assert_ne!(second.slot, third.slot);
    assert_ne!(first.slot, third.slot);
    assert!(renderer.forward_frames.iter().all(Option::is_none));
    assert!(weak.iter().all(|value| value.upgrade().is_some()));

    // The fixture's padded readback is deliberately invalid, but completion
    // and map succeeded. That makes the owner eligible for recycle without
    // conflating CPU pixel decoding with GPU retirement evidence.
    assert!(block_on(renderer.readback(&second)).is_err());
    assert_eq!(renderer.pending(), 2);
    assert!(renderer.forward_frames[first.slot].is_none());
    assert_eq!(
        pooled_instances(renderer.forward_frames[second.slot].as_ref().unwrap()),
        second_expected
    );
    assert!(renderer.forward_frames[third.slot].is_none());
    assert!(weak[0].upgrade().is_some());
    assert!(weak[1].upgrade().is_none());
    assert!(weak[2].upgrade().is_some());

    assert!(block_on(renderer.readback(&first)).is_err());
    assert_eq!(renderer.pending(), 1);
    assert_eq!(
        pooled_instances(renderer.forward_frames[first.slot].as_ref().unwrap()),
        first_expected
    );
    assert_eq!(
        pooled_instances(renderer.forward_frames[second.slot].as_ref().unwrap()),
        second_expected
    );
    assert!(renderer.forward_frames[third.slot].is_none());
    assert!(weak[0].upgrade().is_none());
    assert!(weak[2].upgrade().is_some());

    assert!(block_on(renderer.readback(&third)).is_err());
    assert_eq!(renderer.pending(), 0);
    assert_eq!(
        pooled_instances(renderer.forward_frames[third.slot].as_ref().unwrap()),
        third_expected
    );
    assert!(weak.iter().all(|value| value.upgrade().is_none()));
}

#[test]
fn forward_upload_map_or_flush_failure_drops_unsubmitted_owners_and_leaves_no_pending_slot() {
    for script in [Script::MapDeviceLost, Script::FlushDeviceLost] {
        let (device, native) = fixture(script);
        let mut renderer = Renderer::new(device.clone()).unwrap();
        let created_before = native.created_buffers.load(Ordering::SeqCst);
        let dropped_before = native.dropped_buffers.load(Ordering::SeqCst);
        let frame = blank_frame(&device);
        let retained = Arc::new(vec![1]);
        let weak = Arc::downgrade(&retained);

        assert!(matches!(
            execute(
                &mut renderer,
                crate::renderer::PendingInputs::Bytes { _bytes: retained },
                frame,
            ),
            Err(RendererError::ForwardOperation {
                error: Error::DeviceLost,
                ..
            }) | Err(RendererError::Rhi(Error::DeviceLost))
        ));
        assert_eq!(renderer.pending(), 0);
        assert!(renderer.forward_frames.iter().all(Option::is_none));
        assert!(weak.upgrade().is_none());
        assert_eq!(native.finishes.load(Ordering::SeqCst), 0);
        assert_eq!(native.command_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            native.created_buffers.load(Ordering::SeqCst) - created_before,
            native.dropped_buffers.load(Ordering::SeqCst) - dropped_before
        );
    }
}
