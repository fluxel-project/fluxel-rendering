//! Queue submission, completion observation, and conformance fault injection.

use super::*;

/// Submits a copy command while retaining private staging allocations in the
/// same native submission bundle as the command buffer and target leases.
///
/// This is deliberately not a caller-side keepalive: an accepted submission
/// whose completion becomes unknowable must quarantine every referenced
/// allocation together.
pub(crate) fn submit_copy_with_staging(
    mut buffer: CopyCommandBuffer,
    leases: Vec<ResourceLease>,
    staging_buffers: Vec<OwnedBuffer>,
) -> Result<NativeCompletion, String> {
    let finished = buffer.native.take().expect("finished command buffer");
    #[cfg(any(test, feature = "test-support"))]
    let injected_submit_fault = take_submit_fault();
    #[cfg(not(any(test, feature = "test-support")))]
    let injected_submit_fault = 0;
    if injected_submit_fault == 1 {
        reset_finished(finished);
        return Err("injected pre-submit rejection".into());
    }
    let inject_accepted_unknown = injected_submit_fault == 2;
    let submission = match finished {
        #[cfg(feature = "dx12")]
        NativeFinished::Dx12 {
            owner,
            mut encoder,
            command_buffer,
            render_views,
        } => {
            // Keep this guard through both submit and the accepted-unknown
            // recovery `wait_for_idle` below. HAL requires those operations to
            // be externally synchronized with every operation on this queue.
            let queue_owner = Arc::clone(&owner);
            let _queue_guard = lock_queue_operations(&queue_owner.queue_operations);
            let NativeDevice::Dx12 { device, queue, .. } = &owner.native else {
                unreachable!()
            };
            // SAFETY: device is live and owns the new unsignaled fence.
            let fence = match unsafe { device.create_fence() } {
                Ok(fence) => fence,
                Err(error) => {
                    // No submission occurred. Reset first, then release views
                    // whose descriptors are recorded by this command buffer.
                    // SAFETY: fence creation failed before queue submission, so
                    // the uniquely owned encoder/buffer are not in flight.
                    unsafe { encoder.reset_all(core::iter::once(command_buffer)) };
                    destroy_render_views(&owner, render_views);
                    return Err(error.to_string());
                }
            };
            // SAFETY: command buffer and encoder belong to this device/queue and
            // remain owned by the returned completion until fence value 1.
            match unsafe { queue.submit(&[&command_buffer], &[], (&fence, 1)) } {
                Ok(()) => NativeSubmission::Dx12 {
                    owner,
                    encoder: Some(encoder),
                    command_buffer: Some(command_buffer),
                    fence: Some(fence),
                    leases,
                    staging_buffers,
                    render_views,
                    presentation_lease: None,
                    presentation_completion_hold: None,
                    failure: inject_accepted_unknown.then_some(CompletionFailure::DeviceLost),
                },
                Err(error) => {
                    let failure = completion_failure(&error);
                    // DX12 may execute command lists before a later fence signal
                    // reports this error. Only a successful queue-idle wait makes
                    // immediate command allocator reuse legal.
                    // SAFETY: `_queue_guard` externally serializes every operation
                    // on this live queue, as required by HAL `wait_for_idle`.
                    if unsafe { queue.wait_for_idle() }.is_ok() {
                        unsafe {
                            // SAFETY: successful idle proves the command buffer is
                            // no longer in flight; encoder and fence are unique and
                            // destroyed/reset exactly once on this terminal path.
                            encoder.reset_all(core::iter::once(command_buffer));
                            device.destroy_fence(fence);
                        }
                        destroy_render_views(&owner, render_views);
                        NativeSubmission::TerminalFailure(failure)
                    } else {
                        NativeSubmission::Dx12 {
                            owner,
                            encoder: Some(encoder),
                            command_buffer: Some(command_buffer),
                            fence: Some(fence),
                            leases,
                            staging_buffers,
                            render_views,
                            presentation_lease: None,
                            presentation_completion_hold: None,
                            failure: Some(CompletionFailure::DeviceLost),
                        }
                    }
                }
            }
        }
        #[cfg(feature = "vulkan")]
        NativeFinished::Vulkan {
            owner,
            mut encoder,
            command_buffer,
            render_views,
        } => {
            // See the DX12 arm: this is also held across error-path idle wait.
            let queue_owner = Arc::clone(&owner);
            let _queue_guard = lock_queue_operations(&queue_owner.queue_operations);
            let NativeDevice::Vulkan { device, queue, .. } = &owner.native else {
                unreachable!()
            };
            // SAFETY: device is live and owns the new unsignaled fence.
            let fence = match unsafe { device.create_fence() } {
                Ok(fence) => fence,
                Err(error) => {
                    // No submission occurred; this command allocator and every
                    // temporary render view may now be released exactly once.
                    // SAFETY: fence creation failed before queue submission, so
                    // the uniquely owned encoder/buffer are not in flight.
                    unsafe { encoder.reset_all(core::iter::once(command_buffer)) };
                    destroy_render_views(&owner, render_views);
                    return Err(error.to_string());
                }
            };
            // SAFETY: all submitted objects remain retained to completion.
            match unsafe { queue.submit(&[&command_buffer], &[], (&fence, 1)) } {
                Ok(()) => NativeSubmission::Vulkan {
                    owner,
                    encoder: Some(encoder),
                    command_buffer: Some(command_buffer),
                    fence: Some(NativeVulkanFence::Owned(fence)),
                    leases,
                    staging_buffers,
                    render_views,
                    presentation_lease: None,
                    failure: inject_accepted_unknown.then_some(CompletionFailure::DeviceLost),
                },
                Err(error) => {
                    let failure = completion_failure(&error);
                    // SAFETY: `_queue_guard` externally serializes every operation
                    // on this live queue, as required by HAL `wait_for_idle`.
                    if unsafe { queue.wait_for_idle() }.is_ok() {
                        unsafe {
                            // SAFETY: successful idle proves the command buffer is
                            // no longer in flight; encoder and fence are unique and
                            // destroyed/reset exactly once on this terminal path.
                            encoder.reset_all(core::iter::once(command_buffer));
                            device.destroy_fence(fence);
                        }
                        destroy_render_views(&owner, render_views);
                        NativeSubmission::TerminalFailure(failure)
                    } else {
                        NativeSubmission::Vulkan {
                            owner,
                            encoder: Some(encoder),
                            command_buffer: Some(command_buffer),
                            fence: Some(NativeVulkanFence::Owned(fence)),
                            leases,
                            staging_buffers,
                            render_views,
                            presentation_lease: None,
                            failure: Some(CompletionFailure::DeviceLost),
                        }
                    }
                }
            }
        }
    };
    Ok(NativeCompletion(Arc::new(std::sync::Mutex::new(
        submission,
    ))))
}

fn completion_failure(error: &wgpu_hal::DeviceError) -> CompletionFailure {
    if *error == wgpu_hal::DeviceError::Lost {
        CompletionFailure::DeviceLost
    } else {
        CompletionFailure::ExecutionFailed
    }
}

pub(crate) fn completion_status(completion: &NativeCompletion) -> Result<CompletionStatus, String> {
    #[cfg(any(test, feature = "test-support"))]
    if TEST_COMPLETION_PENDING.swap(false, std::sync::atomic::Ordering::SeqCst) {
        return Ok(CompletionStatus::Pending);
    }
    let mut submission = completion
        .0
        .lock()
        .map_err(|_| "completion lock poisoned".to_owned())?;
    let value = match &mut *submission {
        #[cfg(feature = "dx12")]
        NativeSubmission::Dx12 {
            owner,
            fence: Some(fence),
            presentation_completion_hold,
            failure,
            ..
        } => {
            if let Some(failure) = *failure {
                return Ok(CompletionStatus::Failed(failure));
            }
            if presentation_completion_is_held(presentation_completion_hold.as_ref()) {
                return Ok(CompletionStatus::Pending);
            }
            let NativeDevice::Dx12 { device, .. } = &owner.native else {
                unreachable!()
            };
            // SAFETY: the completion mutex gives exclusive access to this live
            // same-device fence, which remains owned by the submission.
            match unsafe { device.get_fence_value(fence) } {
                Ok(value) => value,
                Err(error) => {
                    let reason = completion_failure(&error);
                    *failure = Some(reason);
                    return Ok(CompletionStatus::Failed(reason));
                }
            }
        }
        #[cfg(feature = "vulkan")]
        NativeSubmission::Vulkan {
            owner,
            fence: Some(fence),
            failure,
            ..
        } => {
            if let Some(failure) = *failure {
                return Ok(CompletionStatus::Failed(failure));
            }
            let NativeDevice::Vulkan { device, .. } = &owner.native else {
                unreachable!()
            };
            let (fence, target) = match fence {
                NativeVulkanFence::Owned(fence) => (&*fence, 1),
                NativeVulkanFence::Presentation { sync, value } => (sync.fence()?, *value),
            };
            // SAFETY: the completion mutex gives exclusive access to this live
            // same-device fence, which remains owned by the submission.
            match unsafe { device.get_fence_value(fence) } {
                Ok(observed) => {
                    return Ok(if observed >= target {
                        CompletionStatus::Complete
                    } else {
                        CompletionStatus::Pending
                    });
                }
                Err(error) => {
                    let reason = completion_failure(&error);
                    *failure = Some(reason);
                    return Ok(CompletionStatus::Failed(reason));
                }
            }
        }
        NativeSubmission::TerminalFailure(failure) => {
            return Ok(CompletionStatus::Failed(*failure));
        }
        _ => {
            return Ok(CompletionStatus::Failed(
                fluxel_rendergraph::CompletionFailure::ExecutionFailed,
            ));
        }
    };
    #[cfg(feature = "dx12")]
    return Ok(if value >= 1 {
        CompletionStatus::Complete
    } else {
        CompletionStatus::Pending
    });
    #[cfg(not(feature = "dx12"))]
    #[allow(
        unreachable_code,
        reason = "the Vulkan-only match is structurally total and returns from every arm"
    )]
    {
        unreachable!("Vulkan completion status returns from its match arm");
    }
}

pub(crate) fn wait_completion(
    completion: &NativeCompletion,
    timeout: core::time::Duration,
) -> Result<CompletionStatus, String> {
    let mut submission = completion
        .0
        .lock()
        .map_err(|_| "completion lock poisoned".to_owned())?;
    let complete = match &mut *submission {
        #[cfg(feature = "dx12")]
        NativeSubmission::Dx12 {
            owner,
            fence: Some(fence),
            presentation_completion_hold,
            failure,
            ..
        } => {
            if let Some(failure) = *failure {
                return Ok(CompletionStatus::Failed(failure));
            }
            if presentation_completion_is_held(presentation_completion_hold.as_ref()) {
                return Ok(CompletionStatus::Pending);
            }
            let NativeDevice::Dx12 { device, .. } = &owner.native else {
                unreachable!()
            };
            // SAFETY: the completion mutex serializes waits and the live fence
            // belongs to this device/submission at the signaled value 1.
            match unsafe { device.wait(fence, 1, Some(timeout)) } {
                Ok(complete) => complete,
                Err(error) => {
                    let reason = completion_failure(&error);
                    *failure = Some(reason);
                    return Ok(CompletionStatus::Failed(reason));
                }
            }
        }
        #[cfg(feature = "vulkan")]
        NativeSubmission::Vulkan {
            owner,
            fence: Some(fence),
            failure,
            ..
        } => {
            if let Some(failure) = *failure {
                return Ok(CompletionStatus::Failed(failure));
            }
            let NativeDevice::Vulkan { device, .. } = &owner.native else {
                unreachable!()
            };
            let (fence, target) = match fence {
                NativeVulkanFence::Owned(fence) => (&*fence, 1),
                NativeVulkanFence::Presentation { sync, value } => (sync.fence()?, *value),
            };
            // SAFETY: the completion mutex serializes waits and the live fence
            // belongs to this device/submission at the signaled value 1.
            match unsafe { device.wait(fence, target, Some(timeout)) } {
                Ok(complete) => complete,
                Err(error) => {
                    let reason = completion_failure(&error);
                    *failure = Some(reason);
                    return Ok(CompletionStatus::Failed(reason));
                }
            }
        }
        NativeSubmission::TerminalFailure(failure) => {
            return Ok(CompletionStatus::Failed(*failure));
        }
        _ => {
            return Ok(CompletionStatus::Failed(
                fluxel_rendergraph::CompletionFailure::ExecutionFailed,
            ));
        }
    };
    Ok(if complete {
        CompletionStatus::Complete
    } else {
        CompletionStatus::Pending
    })
}

/// Keeps the test oracle local to presentation completion observation. The
/// caller passes this only from DX12 `NativeSubmission`; uploads and copies
/// have no hold field and cannot consume or observe an armed latch.
pub(crate) fn presentation_completion_is_held(
    hold: Option<&std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> bool {
    hold.is_some_and(|hold| hold.load(std::sync::atomic::Ordering::Acquire))
}

#[cfg(any(test, feature = "test-support"))]
static TEST_SUBMIT_FAULT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
/// Number of successful queue accepts to allow before consuming the configured
/// test-only submit fault. The fixture guard serializes configuration; the CAS
/// loop below makes individual submit consumption deterministic nonetheless.
#[cfg(any(test, feature = "test-support"))]
static TEST_SUBMIT_FAULT_AFTER: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(any(test, feature = "test-support"))]
static TEST_COMPLETION_PENDING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// Presentation-only accepted-unknown fault. It is consumed after a DX12
/// queue submit and present both succeed, never by generic submissions.
#[cfg(any(test, feature = "test-support"))]
static TEST_DX12_PRESENTATION_ACCEPTED_UNKNOWN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// FIFO weak handles armed by conformance code. A handle dropped before a
/// successful present is skipped, so stale tests cannot hold a later frame.
#[cfg(any(test, feature = "test-support"))]
static TEST_PRESENTATION_COMPLETION_HOLDS: std::sync::OnceLock<
    std::sync::Mutex<std::collections::VecDeque<std::sync::Weak<std::sync::atomic::AtomicBool>>>,
> = std::sync::OnceLock::new();

#[cfg(any(test, feature = "test-support"))]
pub(crate) fn inject_submit_rejected_once() {
    TEST_SUBMIT_FAULT.store(1, std::sync::atomic::Ordering::SeqCst);
    TEST_SUBMIT_FAULT_AFTER.store(0, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(any(test, feature = "test-support"))]
pub(crate) fn inject_submit_accepted_unknown_once() {
    TEST_SUBMIT_FAULT.store(2, std::sync::atomic::Ordering::SeqCst);
    TEST_SUBMIT_FAULT_AFTER.store(0, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(any(test, feature = "test-support"))]
pub(crate) fn inject_submit_rejected_after(successful_submits: usize) {
    TEST_SUBMIT_FAULT_AFTER.store(successful_submits, std::sync::atomic::Ordering::SeqCst);
    TEST_SUBMIT_FAULT.store(1, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(any(test, feature = "test-support"))]
pub(crate) fn inject_submit_accepted_unknown_after(successful_submits: usize) {
    TEST_SUBMIT_FAULT_AFTER.store(successful_submits, std::sync::atomic::Ordering::SeqCst);
    TEST_SUBMIT_FAULT.store(2, std::sync::atomic::Ordering::SeqCst);
}

/// Returns a configured test fault only after exactly the requested number of
/// earlier calls passed. AcqRel CAS makes each successful decrement correspond
/// to one submission attempt even if a fixture accidentally records in more
/// than one thread.
#[cfg(any(test, feature = "test-support"))]
pub(super) fn take_submit_fault() -> u8 {
    loop {
        let fault = TEST_SUBMIT_FAULT.load(std::sync::atomic::Ordering::Acquire);
        if fault == 0 {
            return 0;
        }
        let after = TEST_SUBMIT_FAULT_AFTER.load(std::sync::atomic::Ordering::Acquire);
        if after == 0 {
            if TEST_SUBMIT_FAULT
                .compare_exchange(
                    fault,
                    0,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                )
                .is_ok()
            {
                return fault;
            }
        } else if TEST_SUBMIT_FAULT_AFTER
            .compare_exchange(
                after,
                after - 1,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_ok()
        {
            return 0;
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
pub(crate) fn inject_completion_pending_once() {
    TEST_COMPLETION_PENDING.store(true, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(any(test, feature = "test-support"))]
pub(crate) fn inject_dx12_presentation_accepted_unknown_once() {
    TEST_DX12_PRESENTATION_ACCEPTED_UNKNOWN.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Called only after the presentation lowering has successfully submitted and
/// presented its acquired image. No upload/copy path may consume this fault.
#[cfg(any(test, feature = "test-support"))]
pub(crate) fn take_dx12_presentation_accepted_unknown() -> bool {
    TEST_DX12_PRESENTATION_ACCEPTED_UNKNOWN.swap(false, std::sync::atomic::Ordering::SeqCst)
}

#[cfg(not(any(test, feature = "test-support")))]
pub(crate) fn take_dx12_presentation_accepted_unknown() -> bool {
    false
}

#[cfg(any(test, feature = "test-support"))]
pub(crate) fn arm_next_dx12_presentation_completion_hold(
    hold: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    let holds = TEST_PRESENTATION_COMPLETION_HOLDS
        .get_or_init(|| std::sync::Mutex::new(std::collections::VecDeque::new()));
    holds
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push_back(std::sync::Arc::downgrade(hold));
}

/// Consumes one still-live hold only at the successful-present boundary.
/// Upload/copy submissions never call this helper.
#[cfg(any(test, feature = "test-support"))]
pub(crate) fn take_next_dx12_presentation_completion_hold()
-> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
    let holds = TEST_PRESENTATION_COMPLETION_HOLDS
        .get_or_init(|| std::sync::Mutex::new(std::collections::VecDeque::new()));
    let mut holds = holds
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    while let Some(hold) = holds.pop_front() {
        if let Some(hold) = hold.upgrade() {
            if hold.load(std::sync::atomic::Ordering::Acquire) {
                return Some(hold);
            }
        }
    }
    None
}

#[cfg(test)]
pub(crate) fn clear_dx12_presentation_completion_holds_for_test() {
    let holds = TEST_PRESENTATION_COMPLETION_HOLDS
        .get_or_init(|| std::sync::Mutex::new(std::collections::VecDeque::new()));
    holds
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
}

#[cfg(not(any(test, feature = "test-support")))]
pub(crate) fn take_next_dx12_presentation_completion_hold()
-> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
    None
}
