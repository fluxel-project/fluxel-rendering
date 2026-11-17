//! Actual-owner rejection tests for the combined capture plan.

use super::*;
use crate::{BoundsVisibility, ForwardCpuPipeline};

fn pending_capture(script: Script) -> (Renderer<WitnessBackend>, Submission, Arc<Native>) {
    let (device, native) = capture_forward_tests::capture_device(script);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) = scene_and_assets();
    let (receiver, _) = capture_forward_tests::receiver();
    let submission = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("captured").unwrap(),
        receiver,
    ))
    .unwrap();
    (renderer, submission, native)
}

fn recording_counts(native: &Native) -> (usize, usize, usize) {
    (
        native.copied_bytes.lock().unwrap().len(),
        native.texture_uploads.lock().unwrap().len(),
        native.uniform_offsets.lock().unwrap().len(),
    )
}

#[test]
fn capture_plan_rejects_a_mutated_receiver_texture_owner_before_recording() {
    let (mut renderer, submission, _) = pending_capture(Script::AcceptedExact);
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
    let (mut renderer, submission, native) = pending_capture(Script::AcceptedExact);
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
    let before = recording_counts(&native);
    let mut encoder = device.encoder(device.lanes()[0].id).unwrap();
    assert!(
        plan.record_receiver_raster(&mut encoder, _capture, _receiver)
            .is_err()
    );
    assert_eq!(recording_counts(&native), before);
}

#[test]
fn mixed_capture_rejects_a_nonfirst_saved_owner_before_recording() {
    let (device, _) = capture_forward_tests::capture_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, mut assets) = scene_and_assets();
    let receiver = capture_forward_tests::mixed_fixture::receiver(&mut assets, false);
    let submission = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("producer_capture").unwrap(),
        receiver,
    ))
    .unwrap();
    let pending = renderer.slots[submission.slot].as_mut().unwrap();
    let crate::renderer::PendingWork::CaptureForward { _receiver, .. } = &mut pending.workload
    else {
        panic!("capture pending work")
    };
    let capture_index = _receiver.draws[0].textures[3];
    Arc::get_mut(&mut _receiver.resident).unwrap().textures[capture_index].device =
        fluxel_rhi::api::DeviceId::allocate().unwrap();
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
fn mixed_capture_rejects_an_upload_targeting_the_capture_owner() {
    let (device, native) = capture_forward_tests::capture_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (snapshot, view, mut assets) = scene_and_assets();
    let receiver = capture_forward_tests::mixed_fixture::receiver(&mut assets, false);
    let submission = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("producer_capture").unwrap(),
        receiver,
    ))
    .unwrap();
    let pending = renderer.slots[submission.slot].as_mut().unwrap();
    let crate::renderer::PendingWork::CaptureForward { _receiver, .. } = &mut pending.workload
    else {
        panic!("capture pending work")
    };
    let mut upload = _receiver.resident_upload.textures[0].clone();
    upload.texture = _receiver.draws[0].textures[3];
    _receiver.resident_upload.textures.push(upload);
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
    let before = recording_counts(&native);
    let mut encoder = device.encoder(device.lanes()[0].id).unwrap();
    assert!(
        plan.record_receiver_upload(&mut encoder, _capture, _receiver)
            .is_err()
    );
    assert_eq!(recording_counts(&native), before);
}

#[test]
fn warm_shared_owner_descriptor_mutation_rejects_plan_and_direct_records() {
    let (device, native) = capture_forward_tests::capture_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (snapshot, view, mut assets) = scene_and_assets();
    let receiver = capture_forward_tests::mixed_fixture::receiver(&mut assets, false);
    let cold = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot.clone(),
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("producer_capture").unwrap(),
        receiver.clone(),
    ))
    .unwrap();
    block_on(renderer.readback(&cold)).unwrap();
    let warm = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("producer_capture").unwrap(),
        receiver,
    ))
    .unwrap();
    let pending = renderer.slots[warm.slot].as_mut().unwrap();
    let crate::renderer::PendingWork::CaptureForward {
        _capture,
        _receiver,
        plan,
        ..
    } = &mut pending.workload
    else {
        panic!("capture pending work")
    };
    assert!(plan.validate_frames(_capture, _receiver).is_ok());
    let texture_index = _receiver.draws[0].textures[0];
    let instance = _receiver.resident.textures[texture_index].instance;
    let resident = Arc::get_mut(&mut _receiver.resident).expect("pending owns receiver resident");
    resident.textures[texture_index].desc.usage = TextureUsage::COPY_DST;
    assert_eq!(resident.textures[texture_index].instance, instance);
    assert!(plan.validate_frames(_capture, _receiver).is_err());
    let before = recording_counts(&native);
    let mut encoder = device.encoder(device.lanes()[0].id).unwrap();
    assert!(
        plan.record_receiver_raster(&mut encoder, _capture, _receiver)
            .is_err()
    );
    assert_eq!(recording_counts(&native), before);
    assert!(
        plan.record_capture_upload(&mut encoder, _capture, _receiver)
            .is_err()
    );
    assert_eq!(recording_counts(&native), before);
}
