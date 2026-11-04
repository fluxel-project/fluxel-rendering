//! Core renderer submission lifecycle tests.

use super::*;

#[test]
fn accepted_exact_completion_retires_actual_owners_after_readback() {
    let (device, native) = fixture(Script::AcceptedExact);
    let (snapshot, weak) = snapshot();
    let mut renderer = Renderer::new(device).unwrap();
    let submission = submit(&mut renderer, snapshot).unwrap();
    assert_eq!(renderer.pending(), 1);
    assert_eq!(
        block_on(renderer.readback(&submission)).unwrap(),
        Vec::<u8>::new()
    );
    assert_eq!(native.waits.load(Ordering::SeqCst), 1);
    assert_eq!(renderer.pending(), 0);
    assert!(weak.upgrade().is_none());
    assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 2);
    assert!(native.outcome.lock().unwrap().upgrade().is_none());
}

#[test]
fn dynamic_copy_lengths_refresh_bindings_and_record_current_frame_ranges() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let first = submit(&mut renderer, snapshot_bytes(vec![1; 4]).0).unwrap();
    let plan = renderer.plan_id().unwrap();
    let second = submit(&mut renderer, snapshot_bytes(vec![2; 12]).0).unwrap();
    let third = submit(&mut renderer, snapshot_bytes(vec![3; 8]).0).unwrap();
    assert_eq!(renderer.pending(), 3);
    assert_eq!(renderer.compilations(), 1);
    assert_eq!(renderer.binding_refreshes(), 3);
    assert_eq!(renderer.plan_id(), Some(plan));
    assert_eq!(*native.copied_bytes.lock().unwrap(), [4, 12, 8]);
    assert_eq!(*native.readback_ranges.lock().unwrap(), [0..4, 0..12, 0..8]);
    for submission in [&first, &second, &third] {
        block_on(renderer.readback(submission)).unwrap();
    }
    assert_eq!(renderer.pending(), 0);
    assert_eq!(*native.waited_completions.lock().unwrap(), [1, 2, 3]);
}

#[test]
fn wrong_completion_frontier_refuses_before_wait_or_retirement() {
    let (device, native) = fixture(Script::AcceptedWrongFrontier);
    let (snapshot, weak) = snapshot();
    let mut renderer = Renderer::new(device).unwrap();
    let submission = submit(&mut renderer, snapshot).unwrap();
    assert!(matches!(
        block_on(renderer.readback(&submission)),
        Err(RendererError::MissingCompletion)
    ));
    assert_eq!(native.waits.load(Ordering::SeqCst), 0);
    assert_eq!(renderer.pending(), 1);
    assert!(weak.upgrade().is_some());
    assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 0);
}

#[test]
fn cancelled_wait_future_does_not_retire_pending_owners() {
    let (device, native) = fixture(Script::AcceptedWaitPending);
    let (snapshot, weak) = snapshot();
    let mut renderer = Renderer::new(device).unwrap();
    let submission = submit(&mut renderer, snapshot).unwrap();
    poll_once_pending(renderer.readback(&submission));
    assert_eq!(native.waits.load(Ordering::SeqCst), 1);
    assert_eq!(renderer.pending(), 1);
    assert!(weak.upgrade().is_some());
    assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 0);
}

#[test]
fn submissions_are_bounded_and_foreign_or_stale_handles_never_wait() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (first_snapshot, first_value) = snapshot();
    let (second_snapshot, second_value) = snapshot();
    let (third_snapshot, third_value) = snapshot();
    let first = submit(&mut renderer, first_snapshot).unwrap();
    let second = submit(&mut renderer, second_snapshot).unwrap();
    let third = submit(&mut renderer, third_snapshot).unwrap();
    assert_eq!(renderer.pending(), 3);
    assert_eq!(renderer.compilations(), 1);
    let plan = renderer.plan_id().unwrap();
    assert_eq!(renderer.plan_id(), Some(plan));
    assert_ne!(first.serial, second.serial);
    assert_ne!(second.serial, third.serial);
    assert_ne!(first.serial, third.serial);
    assert!(first_value.upgrade().is_some());
    assert!(second_value.upgrade().is_some());
    assert!(third_value.upgrade().is_some());
    assert!(matches!(
        submit(&mut renderer, snapshot().0),
        Err(RendererError::Backpressure)
    ));

    let mut other = Renderer::new(device).unwrap();
    assert!(matches!(
        block_on(other.readback(&first)),
        Err(RendererError::ForeignSubmission)
    ));
    assert_eq!(native.waits.load(Ordering::SeqCst), 0);

    block_on(renderer.readback(&first)).unwrap();
    assert_eq!(*native.waited_completions.lock().unwrap(), [1]);
    assert!(first_value.upgrade().is_none());
    assert!(second_value.upgrade().is_some());
    assert!(third_value.upgrade().is_some());
    assert!(matches!(
        block_on(renderer.readback(&first)),
        Err(RendererError::StaleSubmission)
    ));
    assert_eq!(native.waits.load(Ordering::SeqCst), 1);
    block_on(renderer.readback(&second)).unwrap();
    assert_eq!(*native.waited_completions.lock().unwrap(), [1, 2]);
    assert!(second_value.upgrade().is_none());
    assert!(third_value.upgrade().is_some());
    block_on(renderer.readback(&third)).unwrap();
    assert_eq!(*native.waited_completions.lock().unwrap(), [1, 2, 3]);
    assert!(third_value.upgrade().is_none());
}

#[test]
fn unsafe_outcomes_retain_actual_snapshot_native_buffers_and_original_outcome() {
    for script in [Script::RecordingFailureAfterPrefix, Script::Indeterminate] {
        let (device, native) = fixture(script);
        let (snapshot, weak) = snapshot();
        let mut renderer = Renderer::new(device).unwrap();
        match submit(&mut renderer, snapshot) {
            Err(RendererError::RetainedExecution { .. }) => {}
            Err(error) => panic!("unexpected renderer error: {error:?}"),
            Ok(_) => panic!("unsafe execution must not return a submission"),
        }
        assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
        if matches!(script, Script::RecordingFailureAfterPrefix) {
            assert_eq!(native.command_calls.load(Ordering::SeqCst), 2);
        }
        assert_eq!(renderer.pending(), 1);
        assert!(weak.upgrade().is_some());
        assert_eq!(native.created_buffers.load(Ordering::SeqCst), 2);
        assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 0);
        assert!(native.outcome.lock().unwrap().upgrade().is_some());
        if matches!(script, Script::RecordingFailureAfterPrefix) {
            let pending = renderer.slots[0]
                .as_ref()
                .expect("partial work stays quarantined");
            let facts = pending.outcome.facts();
            assert_eq!(facts.status, ExecutionStatus::PartiallyAccepted);
            assert_eq!(facts.accepted.map(|steps| steps.len()), Some(1));
            assert_eq!(facts.affected.map(|steps| steps.len()), Some(1));
            assert!(facts.checkpoints.is_none());
        }
        drop(renderer);
        // Retained uncertain work intentionally leaks native ownership at drop.
        assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn clean_rejection_finishes_and_releases_actual_owners() {
    let (device, native) = fixture(Script::CleanRejected);
    let (snapshot, weak) = snapshot();
    let mut renderer = Renderer::new(device).unwrap();
    assert!(matches!(
        submit(&mut renderer, snapshot),
        Err(RendererError::Rhi(_))
    ));
    assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
    assert_eq!(renderer.pending(), 0);
    assert!(weak.upgrade().is_none());
    assert_eq!(native.created_buffers.load(Ordering::SeqCst), 2);
    assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 2);
    assert!(native.outcome.lock().unwrap().upgrade().is_none());
}

#[test]
fn submit_stage_loss_quarantines_previous_owners_and_blocks_backend_reentry() {
    for script in [
        Script::AllocationDeviceLost,
        Script::MapDeviceLost,
        Script::FlushDeviceLost,
        Script::BeginDeviceLost,
        Script::RecordingDeviceLost,
        Script::FinishDeviceLost,
    ] {
        let (device, native) = fixture(script);
        native.fault_enabled.store(false, Ordering::SeqCst);
        let mut renderer = Renderer::new(device).unwrap();
        let (first, first_weak) = snapshot();
        let submission = submit(&mut renderer, first).unwrap();
        native.fault_enabled.store(true, Ordering::SeqCst);
        let (second, second_weak) = snapshot();
        assert!(submit(&mut renderer, second).is_err());
        let retained = matches!(
            script,
            Script::RecordingDeviceLost | Script::FinishDeviceLost
        );
        assert_eq!(renderer.pending(), if retained { 2 } else { 1 });
        assert_eq!(renderer.compilations(), 0);
        assert!(first_weak.upgrade().is_some());
        assert_eq!(second_weak.upgrade().is_some(), retained);
        assert_eq!(
            native.finishes.load(Ordering::SeqCst),
            if retained { 2 } else { 1 }
        );
        let counters = (
            native.created_buffers.load(Ordering::SeqCst),
            native.command_calls.load(Ordering::SeqCst),
            native.finishes.load(Ordering::SeqCst),
        );
        assert!(matches!(
            submit(&mut renderer, snapshot().0),
            Err(RendererError::RendererClosed)
        ));
        assert!(matches!(
            block_on(renderer.readback(&submission)),
            Err(RendererError::Rhi(Error::DeviceLost))
        ));
        assert!(matches!(
            block_on(renderer.shutdown()),
            Err(RendererError::Rhi(Error::DeviceLost))
        ));
        assert_eq!(
            counters,
            (
                native.created_buffers.load(Ordering::SeqCst),
                native.command_calls.load(Ordering::SeqCst),
                native.finishes.load(Ordering::SeqCst)
            )
        );
        assert!(native.waited_completions.lock().unwrap().is_empty());
        assert_eq!(native.idle_waits.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn nonaccepted_outcomes_without_recording_error_retain_until_successful_idle() {
    for (script, status) in [
        (Script::PartialAccepted, ExecutionStatus::PartiallyAccepted),
        (Script::PartialIssued, ExecutionStatus::Failed),
        (Script::Indeterminate, ExecutionStatus::Indeterminate),
    ] {
        let (device, native) = fixture(script);
        let mut renderer = Renderer::new(device).unwrap();
        let (source, weak) = snapshot();
        assert!(
            matches!(submit(&mut renderer, source), Err(RendererError::RetainedExecution { status: actual, .. }) if actual == status)
        );
        assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
        assert_eq!(renderer.pending(), 1);
        assert!(weak.upgrade().is_some());
        assert!(native.outcome.lock().unwrap().upgrade().is_some());
        let pending = renderer.slots[0].as_ref().unwrap();
        assert_eq!(pending.outcome.facts().status, status);
        if matches!(script, Script::PartialIssued) {
            assert_eq!(
                pending.outcome.facts().failed_step.unwrap().effect,
                StepEffect::Partial
            );
        }
        let report = block_on(renderer.shutdown()).unwrap();
        assert_eq!(report.retired_submissions, 1);
        assert_eq!(native.idle_waits.load(Ordering::SeqCst), 1);
        assert!(weak.upgrade().is_none());
        assert!(native.outcome.lock().unwrap().upgrade().is_none());
        assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 2);
    }
}
#[test]
fn contradictory_accepted_outcome_with_error_is_observable_and_retained() {
    let (device, native) = fixture(Script::AcceptedWithError);
    let mut renderer = Renderer::new(device).unwrap();
    let (source, weak) = snapshot();
    assert!(matches!(
        submit(&mut renderer, source),
        Err(RendererError::RetainedExecution {
            status: ExecutionStatus::Accepted,
            execution_error: Some(Error::Backend(_)),
            ..
        })
    ));
    assert_eq!(renderer.pending(), 1);
    assert!(weak.upgrade().is_some());
    assert!(native.outcome.lock().unwrap().upgrade().is_some());
    block_on(renderer.shutdown()).unwrap();
    assert!(weak.upgrade().is_none());
}

#[test]
fn copy_limit_refusal_happens_before_plan_allocation_mapping_or_recording() {
    let (device, native) = fixture_with_caps(Script::AcceptedExact, |caps| {
        caps.limits.max_buffer_bytes = 3;
    });
    let mut renderer = Renderer::new(device).unwrap();
    assert!(matches!(
        submit(&mut renderer, snapshot_bytes(vec![1, 2, 3, 4]).0),
        Err(RendererError::Rhi(Error::InvalidRange))
    ));
    assert_eq!(renderer.compilations(), 0);
    assert_eq!(native.created_buffers.load(Ordering::SeqCst), 0);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn compute_storage_capability_is_rejected_before_fixture_resources_exist() {
    for case in [
        "storage-buffers",
        "buffer-bytes",
        "bind-groups",
        "bindings-per-group",
        "workgroups-x",
        "workgroups-y",
        "workgroups-z",
        "workgroup-size-x",
        "workgroup-size-y",
        "workgroup-size-z",
    ] {
        let (device, native) = fixture_with_caps(Script::AcceptedExact, |caps| {
            caps.storage_buffers = true;
            caps.limits.max_compute_workgroups = [1, 1, 1];
            caps.limits.max_compute_workgroup_size = [64, 1, 1];
            match case {
                "storage-buffers" => caps.storage_buffers = false,
                "buffer-bytes" => caps.limits.max_buffer_bytes = 3,
                "bind-groups" => caps.limits.max_bind_groups = 0,
                "bindings-per-group" => caps.limits.max_bindings_per_group = 0,
                "workgroups-x" => caps.limits.max_compute_workgroups[0] = 0,
                "workgroups-y" => caps.limits.max_compute_workgroups[1] = 0,
                "workgroups-z" => caps.limits.max_compute_workgroups[2] = 0,
                "workgroup-size-x" => caps.limits.max_compute_workgroup_size[0] = 63,
                "workgroup-size-y" => caps.limits.max_compute_workgroup_size[1] = 0,
                "workgroup-size-z" => caps.limits.max_compute_workgroup_size[2] = 0,
                _ => unreachable!(),
            }
        });
        assert!(
            validate_compute_requirements(&device, 4, 1).is_err(),
            "{case}"
        );
        assert_eq!(native.created_buffers.load(Ordering::SeqCst), 0, "{case}");
        assert_eq!(native.command_calls.load(Ordering::SeqCst), 0, "{case}");
    }
}

// Keep the convenience executor with the witness backend that consumes it.
// Production code exposes only the real contribution path.
