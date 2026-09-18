//! Step 9's first half: the fence, one submit on logical queue 0, and the
//! completion state machine.
//!
//! The retained execution model submits one ordered command buffer per graph
//! execution, so a submission here is exactly one [`Finished`] recording, one fence
//! and no semaphores: `vkQueueSubmit` is called once, with one [`vk::SubmitInfo`],
//! and the fence it signals is what completion is read from.
//!
//! # The completion state machine is `common`'s, not a second copy
//!
//! [`Disposition`](crate::common::base::submission::Disposition) already states who
//! holds a submission's leases while completion is not terminal, and
//! [`may_release`](crate::common::base::lifetime::may_release) already states which
//! statuses are terminal. This module only *reads* the fence and hands the answer to
//! those two rules; it never restates either. A terminal observation is the only
//! thing that frees the command buffer and destroys the fence, and re-observing a
//! released submission is refused rather than treated as idempotent.
//!
//! # Accepted-unknown work is quarantined, never released early
//!
//! `vkQueueSubmit` returning an error does **not** mean the driver accepted
//! nothing: on a device-lost result the submission may have executed in part. Only a
//! successful `vkQueueWaitIdle` proves nothing is in flight, and that is the single
//! path that releases a rejected recording. Where idle cannot be established, the
//! submission is returned live with [`CompletionStatus::Failed`] recorded and its
//! disposition abandoned, so the command buffer and the fence are deliberately never
//! freed. That is the preserved semantic from the plan's section 4, and the borrowed
//! path being replaced implements it the same way by forgetting its bundle.
//!
//! # What is deliberately not here
//!
//! No semaphores, no presentation, no queue-shape accessor and no non-blocking
//! retirement queue. A submission dropped before terminal completion quarantines its
//! command buffer and fence instead of destroying them; reclaiming that quarantine
//! is a retirement path's job, and the RHI layer that owns one arrives with the
//! submission integration.

use core::time::Duration;

use ash::vk;
use fluxel_rendergraph::{CompletionFailure, CompletionStatus};

use crate::common::base::submission::Disposition;

use super::command::Finished;

/// Why a submission could not be made, or could no longer be observed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SubmitError {
    /// The driver refused to create the fence, so nothing was submitted.
    Fence(vk::Result),
    /// The driver refused the submission and a queue-idle wait proved nothing was
    /// accepted, so the recording was released.
    ///
    /// Distinct from an accepted-unknown submission, which is returned as a live
    /// [`Submission`] reporting [`CompletionStatus::Failed`] and quarantines its
    /// resources rather than releasing them.
    Rejected(vk::Result),
    /// The submission already reached terminal completion, so its resources are
    /// gone and there is nothing left to observe.
    AlreadyTerminal,
}

/// Names the failure a driver result means, so a lost device is not mistaken for an
/// ordinary execution failure.
///
/// `CompletionFailure` is `#[non_exhaustive]`, but constructing its variants is
/// fine; what a crate outside `rendergraph` cannot do is match it exhaustively,
/// which is why nothing here does.
pub(crate) const fn failure_of(result: vk::Result) -> CompletionFailure {
    match result {
        vk::Result::ERROR_DEVICE_LOST => CompletionFailure::DeviceLost,
        _ => CompletionFailure::ExecutionFailed,
    }
}

/// Converts a portable wait duration into the nanosecond count `Vulkan` takes.
///
/// `u64::MAX` is the specification's "wait forever", so a duration too large to be
/// named in nanoseconds is clamped to it rather than wrapped -- a wrapped value
/// would turn a long wait into a short one, which is the unsafe direction for a
/// completion wait.
fn timeout_nanos(timeout: Duration) -> u64 {
    u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX)
}

/// What one fence wait answered, before the state machine interprets it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WaitOutcome {
    /// The fence signaled.
    Signaled,
    /// The timeout elapsed first; the work is still in flight.
    TimedOut,
    /// The driver could not answer, which is not a timeout and not a completion.
    Unavailable(vk::Result),
}

/// Separates a timeout from a failure, because they are different sentences: one
/// says the work is still running, the other says completion cannot be established.
fn wait_outcome(result: Result<(), vk::Result>) -> WaitOutcome {
    match result {
        Ok(()) => WaitOutcome::Signaled,
        Err(vk::Result::TIMEOUT) => WaitOutcome::TimedOut,
        Err(result) => WaitOutcome::Unavailable(result),
    }
}

/// One driver fence, unsignaled at creation and destroyed exactly once.
///
/// The fence is private to this module because its `Drop` destroys it
/// unconditionally, and that is only legal once the submission that named it is
/// terminal. [`Submission`] is the value that guarantees that: it destroys the fence
/// after a terminal observation, or forgets it when the submission is quarantined.
struct Fence {
    device: ash::Device,
    handle: vk::Fence,
}

impl Fence {
    /// Creates an **unsignaled** fence.
    ///
    /// It is never created signaled: a fence created in the signaled state would let
    /// a submission look complete before the driver had run anything.
    fn new(device: &ash::Device) -> Result<Self, SubmitError> {
        let create_info = vk::FenceCreateInfo::default();
        // SAFETY: no allocation callbacks are supplied, and the device is live for
        // this fence's lifetime because the caller owns it.
        let handle =
            unsafe { device.create_fence(&create_info, None) }.map_err(SubmitError::Fence)?;
        Ok(Self {
            device: device.clone(),
            handle,
        })
    }

    /// Returns the raw fence handle, for the one submission that names it.
    const fn handle(&self) -> vk::Fence {
        self.handle
    }

    /// Reports whether the fence has signaled, without waiting.
    fn is_signaled(&self) -> Result<bool, vk::Result> {
        // SAFETY: the fence belongs to this device and this module submits it to at
        // most one queue, which the caller externally synchronizes.
        unsafe { self.device.get_fence_status(self.handle) }
    }

    /// Waits up to `timeout` for the fence to signal.
    fn wait(&self, timeout: Duration) -> WaitOutcome {
        // SAFETY: the fence belongs to this device, and the slice points at a local
        // that outlives the call. Only one fence is waited on, so `wait_all` is
        // trivially satisfied.
        let result = unsafe {
            self.device
                .wait_for_fences(&[self.handle], true, timeout_nanos(timeout))
        };
        wait_outcome(result)
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        // SAFETY: this is the only owner of the fence; no allocation callbacks were
        // supplied, and `Submission` destroys it only after a terminal observation --
        // otherwise it is forgotten, so this body never runs on a pending fence.
        unsafe { self.device.destroy_fence(self.handle, None) };
    }
}

impl core::fmt::Debug for Fence {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.debug_struct("Fence").finish_non_exhaustive()
    }
}

/// One submitted execution: the fence that reports it and the recording it runs.
///
/// The recording is owned here rather than by the caller because it must stay alive
/// until the fence signals, and that is exactly the handoff the encoder's docs say
/// submission owns. Once completion is terminal the recording is dropped -- which
/// frees its command buffer -- and the fence is destroyed.
pub(crate) struct Submission {
    fence: Option<Fence>,
    encoder: Option<Finished>,
    disposition: Disposition,
    /// Set when completion cannot be established (an accepted submission whose
    /// driver result was a failure, or a fence the driver refused to report).
    ///
    /// It is not a second terminality rule: `Disposition` still decides ownership,
    /// and this value makes the ownership permanently unobservable so it stays
    /// quarantined.
    failure: Option<CompletionFailure>,
}

impl Submission {
    /// Whether terminal completion has been observed and the resources released.
    pub(crate) const fn is_terminal(&self) -> bool {
        matches!(self.disposition, Disposition::Terminal)
    }

    /// Reports completion without blocking.
    ///
    /// A submission that was accepted but whose completion cannot be established
    /// keeps answering [`CompletionStatus::Failed`]; it never becomes releasable,
    /// because nothing has established that the driver is done with its resources.
    pub(crate) fn status(&mut self) -> Result<CompletionStatus, SubmitError> {
        if self.is_terminal() {
            return Err(SubmitError::AlreadyTerminal);
        }
        if let Some(failure) = self.failure {
            return Ok(CompletionStatus::Failed(failure));
        }
        let Some(signaled) = self.fence.as_ref().map(Fence::is_signaled) else {
            return Err(SubmitError::AlreadyTerminal);
        };
        match signaled {
            Ok(true) => self.observe(CompletionStatus::Complete),
            Ok(false) => self.observe(CompletionStatus::Pending),
            Err(result) => {
                let failure = self.record_failure(result);
                Ok(CompletionStatus::Failed(failure))
            }
        }
    }

    /// Waits up to `timeout` for completion.
    ///
    /// A timeout answers [`CompletionStatus::Pending`] rather than an error, because
    /// work still in flight is information, not a failure. As with [`Self::status`],
    /// a submission whose completion cannot be established answers
    /// [`CompletionStatus::Failed`] without waiting.
    pub(crate) fn wait(&mut self, timeout: Duration) -> Result<CompletionStatus, SubmitError> {
        if self.is_terminal() {
            return Err(SubmitError::AlreadyTerminal);
        }
        if let Some(failure) = self.failure {
            return Ok(CompletionStatus::Failed(failure));
        }
        let Some(outcome) = self.fence.as_ref().map(|fence| fence.wait(timeout)) else {
            return Err(SubmitError::AlreadyTerminal);
        };
        match outcome {
            WaitOutcome::Signaled => self.observe(CompletionStatus::Complete),
            WaitOutcome::TimedOut => self.observe(CompletionStatus::Pending),
            WaitOutcome::Unavailable(result) => {
                let failure = self.record_failure(result);
                Ok(CompletionStatus::Failed(failure))
            }
        }
    }

    /// Hands `status` to `common`'s state machine and releases at terminality.
    ///
    /// Nothing here decides what terminal means: `Disposition::observe` delegates to
    /// `may_release`, so the status rule and the ownership rule cannot disagree.
    fn observe(&mut self, status: CompletionStatus) -> Result<CompletionStatus, SubmitError> {
        self.disposition = self
            .disposition
            .observe(status)
            .map_err(|_| SubmitError::AlreadyTerminal)?;
        if self.disposition.may_release() {
            self.release();
        }
        Ok(status)
    }

    /// Records that completion cannot be established, and quarantines ownership.
    ///
    /// The disposition is *abandoned* rather than observed terminal: the driver may
    /// still be using the resources, so this submission releases nothing even though
    /// its reported status is a terminal-looking failure.
    fn record_failure(&mut self, result: vk::Result) -> CompletionFailure {
        let failure = failure_of(result);
        self.failure = Some(failure);
        self.disposition = self.disposition.abandon();
        failure
    }

    /// Releases the recording and the fence once the driver is done with both.
    ///
    /// Order is the dependency order a rejection also uses: the command buffer is
    /// freed first, then the fence it would have signaled is destroyed.
    fn release(&mut self) {
        self.encoder = None;
        self.fence = None;
    }
}

impl Drop for Submission {
    fn drop(&mut self) {
        if self.is_terminal() {
            // `release` already freed the recording and destroyed the fence.
            return;
        }
        // Completion was never observed terminal, so the driver may still be using
        // the command buffer and the fence. Forgetting both quarantines them rather
        // than risking use after free: forgetting the fence keeps its `Drop` from
        // destroying a fence a pending submission still refers to, and forgetting
        // the recording keeps its command buffer allocated.
        //
        // Reclaiming a quarantine is a retirement path's job; until this backend has
        // one, a dropped non-terminal submission leaks two handles instead of
        // releasing resources the GPU may still read. That is the same direction the
        // borrowed path takes when it forgets its bundle.
        core::mem::forget(self.encoder.take());
        core::mem::forget(self.fence.take());
    }
}

impl core::fmt::Debug for Submission {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Submission")
            .field("disposition", &self.disposition)
            .field("failure", &self.failure)
            .finish_non_exhaustive()
    }
}

/// Submits exactly one finished recording to `queue` and owns its fence.
///
/// The caller passes the device's one queue, which is why "one submit per graph
/// execution on logical queue 0" is a property of the call site rather than a
/// parameter this backend can vary.
///
/// A driver rejection is reported as [`SubmitError::Rejected`] **only** after a
/// successful `vkQueueWaitIdle` has proved nothing was accepted; otherwise the
/// submission is returned live and quarantined, because the driver may have executed
/// part of it. See the module docs.
pub(crate) fn submit(
    device: &ash::Device,
    queue: vk::Queue,
    finished: Finished,
) -> Result<Submission, SubmitError> {
    let fence = Fence::new(device)?;
    let command_buffers = [finished.command_buffer()];
    // No wait semaphores and no signal semaphores: this backend is single-queue and
    // orders submissions on the queue itself, so a semaphore here would be a second
    // ordering mechanism nothing asked for.
    let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
    // SAFETY: the command buffer is in the executable state -- the `Finished` type
    // is the proof -- and belongs to this device; the fence is unsignaled and belongs
    // to this device; `submit_info` points at a local array that outlives the call;
    // and the caller externally synchronizes this one queue.
    let submitted = unsafe { device.queue_submit(queue, &[submit_info], fence.handle()) };
    match submitted {
        Ok(()) => Ok(Submission {
            fence: Some(fence),
            encoder: Some(finished),
            disposition: Disposition::accepted(),
            failure: None,
        }),
        Err(result) => {
            // The driver refused, but a refused `vkQueueSubmit` is not proof that
            // nothing ran. Only a successful idle wait is.
            // SAFETY: the queue belongs to this device and the caller externally
            // synchronizes it.
            if unsafe { device.queue_wait_idle(queue) }.is_ok() {
                // Nothing is in flight, so the recording and the fence are released
                // now: the command buffer first, then the fence it would have
                // signaled. The caller receives the driver's own refusal.
                drop(finished);
                drop(fence);
                Err(SubmitError::Rejected(result))
            } else {
                // Accepted-unknown: the resources may still be in use, so the
                // submission is returned live and quarantines them.
                let mut submission = Submission {
                    fence: Some(fence),
                    encoder: Some(finished),
                    disposition: Disposition::accepted(),
                    failure: None,
                };
                submission.record_failure(result);
                Ok(submission)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::vulkan::command::CommandPool;
    use crate::native::vulkan::resource::ResourceTable;
    use crate::native::vulkan::{allocator::GpuAllocator, memory, open};
    use crate::Validation;
    use fluxel_rendergraph::{
        BufferCopyRegion, BufferRange, BufferUsage, BufferUsageKind, ResourceAccessState,
    };

    /// Opens a headless device and a real command pool, or returns `None` where no
    /// adapter exists. Having no GPU is not what these tests are about.
    fn pool() -> Option<(open::OpenedVulkan, CommandPool)> {
        let opened = open::open(Validation::Disabled, 0).ok()?;
        let pool = CommandPool::new(
            opened.device.device(),
            opened.device.selected_queue().family,
        )
        .expect("a command pool on an opened device");
        Some((opened, pool))
    }

    fn declared_buffer(kinds: &[BufferUsageKind]) -> BufferUsage {
        BufferUsage::from_kinds(kinds.iter().copied())
    }

    #[test]
    fn a_timed_out_wait_is_pending_while_a_driver_result_is_unavailable() {
        // A timeout says the work is still running; a driver result says completion
        // cannot be established. Keeping them in one variant each is what stops a
        // timeout from being reported as a failure.
        assert_eq!(wait_outcome(Ok(())), WaitOutcome::Signaled);
        assert_eq!(
            wait_outcome(Err(vk::Result::TIMEOUT)),
            WaitOutcome::TimedOut
        );
        assert_eq!(
            wait_outcome(Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)),
            WaitOutcome::Unavailable(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        );
    }

    #[test]
    fn a_lost_device_is_named_and_every_other_result_is_an_execution_failure() {
        assert_eq!(
            failure_of(vk::Result::ERROR_DEVICE_LOST),
            CompletionFailure::DeviceLost
        );
        assert_eq!(
            failure_of(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY),
            CompletionFailure::ExecutionFailed
        );
        assert_eq!(
            failure_of(vk::Result::ERROR_SURFACE_LOST_KHR),
            CompletionFailure::ExecutionFailed
        );
    }

    #[test]
    fn a_timeout_longer_than_the_driver_can_name_waits_forever() {
        // `u64::MAX` is the specification's "forever", so an unrepresentable
        // duration clamps rather than wraps. A wrap would turn the longest wait into
        // the shortest one.
        assert_eq!(timeout_nanos(Duration::ZERO), 0);
        assert_eq!(timeout_nanos(Duration::from_millis(5)), 5_000_000);
        assert_eq!(timeout_nanos(Duration::from_secs(1)), 1_000_000_000);
        assert_eq!(timeout_nanos(Duration::MAX), u64::MAX);
    }

    #[test]
    fn a_real_submission_completes_then_releases_and_refuses_further_observation() {
        // Step 9 against the real driver: one pool, one recording that copies a
        // buffer, one fence, one submit, and a completion the state machine turns
        // into a release. Skips where no adapter exists.
        let Some((opened, pool)) = pool() else {
            return;
        };
        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let mut table =
            ResourceTable::new(opened.device.device(), opened.device.stamp(), allocator);

        let source = table
            .create_buffer(
                256,
                declared_buffer(&[BufferUsageKind::CopySource]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local copy source");
        let destination = table
            .create_buffer(
                256,
                declared_buffer(&[BufferUsageKind::CopyDestination]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local copy destination");
        let (source_handle, source_size) = (
            table.buffer_handle(source).expect("a live buffer"),
            table.buffer_size(source).expect("a created size"),
        );
        let (destination_handle, destination_size) = (
            table.buffer_handle(destination).expect("a live buffer"),
            table.buffer_size(destination).expect("a created size"),
        );

        let mut encoder = pool.begin().expect("a recording encoder");
        encoder
            .transition_buffer(
                source_handle,
                BufferRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::CopySource,
            )
            .expect("the source is readable");
        encoder
            .transition_buffer(
                destination_handle,
                BufferRange::Whole,
                ResourceAccessState::Undefined,
                ResourceAccessState::CopyDestination,
            )
            .expect("the destination is writable");
        encoder
            .copy_buffer(
                source_handle,
                source_size,
                destination_handle,
                destination_size,
                BufferCopyRegion {
                    source_offset: 0,
                    destination_offset: 0,
                    size: 256,
                },
            )
            .expect("a real buffer copy records");

        // `finish` ends the still-open recording and is the only way to obtain the
        // value submission accepts.
        let finished = encoder.finish().expect("the recording ends");
        let mut submission = submit(opened.device.device(), opened.device.queue(), finished)
            .expect("the driver accepts one submission");
        assert!(!submission.is_terminal());

        assert_eq!(
            submission.wait(Duration::from_secs(10)),
            Ok(CompletionStatus::Complete),
            "the fence signaled, so completion is terminal"
        );
        assert!(
            submission.is_terminal(),
            "a terminal observation releases the submission"
        );
        // The released submission still exists, but its resources are gone, so both
        // observations are refused rather than answered from a freed fence.
        assert_eq!(submission.status(), Err(SubmitError::AlreadyTerminal));
        assert_eq!(
            submission.wait(Duration::ZERO),
            Err(SubmitError::AlreadyTerminal)
        );
    }

    #[test]
    fn two_executions_each_submit_one_buffer_and_both_complete() {
        // "One submit per graph execution on logical queue 0" is the retained model,
        // so two executions are two submissions on one queue and both fences signal.
        // A second submission is also what proves the first was released before the
        // queue was reused.
        let Some((opened, pool)) = pool() else {
            return;
        };
        let allocator =
            GpuAllocator::new(opened.instance.instance(), &opened.device, opened.adapter)
                .expect("an allocator for an opened device");
        let memory_types = memory::types(opened.instance.instance(), opened.adapter);
        let mut table =
            ResourceTable::new(opened.device.device(), opened.device.stamp(), allocator);
        let buffer = table
            .create_buffer(
                64,
                declared_buffer(&[BufferUsageKind::Vertex]),
                &memory_types,
                memory::MemoryPurpose::DeviceLocal,
            )
            .expect("a device-local buffer");
        let handle = table.buffer_handle(buffer).expect("a live buffer");

        for _ in 0..2 {
            let mut encoder = pool.begin().expect("a recording encoder");
            // A same-state barrier is still a barrier, so the recording has real
            // work without needing a second resource.
            encoder
                .transition_buffer(
                    handle,
                    BufferRange::Whole,
                    ResourceAccessState::VertexRead,
                    ResourceAccessState::VertexRead,
                )
                .expect("a same-state transition records");
            let finished = encoder.finish().expect("the recording ends");
            let mut submission = submit(opened.device.device(), opened.device.queue(), finished)
                .expect("the driver accepts one submission");
            assert_eq!(
                submission.wait(Duration::from_secs(10)),
                Ok(CompletionStatus::Complete)
            );
            assert!(submission.is_terminal());
        }
    }
}
