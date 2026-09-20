//! Transactional Vulkan submission spine.
//!
//! A Fluxel submission has a stronger contract than Vulkan's incremental queue
//! API: returning `Err` proves that no work from the plan was accepted.  Vulkan
//! callers normally record and submit command buffers incrementally, but doing
//! that would make a later lowering failure ambiguous.  We therefore record,
//! begin and end every batch before the first `vkQueueSubmit`.
//!
//! Once that first submit is made, failure is execution-domain state, never the
//! return value of `submit`: a Vulkan implementation may have accepted part of
//! the call before reporting an error, and reporting `Err` would lie to the
//! portable caller.  The spine poisons the accepted serial range and reports
//! `Failed` (or `DeviceLost`) from completion instead.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::Waker;

use ash::vk;

use crate::api::command::record::{CopyRecord, RecordedPayload};
use crate::api::command::{ResourceUse, TextureUse, TextureUseIntent};
use crate::api::platform::DeviceLossInfo;
use crate::api::resource::transfer::ReadbackStatus;
use crate::api::submission::backend::{SubmissionOutcome, SubmissionRequest};
use crate::api::submission::plan::PlanBatch;
use crate::api::submission::{CompletionFailure, CompletionState};
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::ffi;
use crate::backend::vulkan::platform::device::VulkanShared;

use super::compute;
use super::raster;
use super::transfer::{self, ImageLayoutState, TransferRetention};

/// One single-queue Vulkan command domain.
pub(in crate::backend::vulkan) struct VulkanCommandSpine {
    inner: Arc<SpineInner>,
}

/// Queue-local native objects retained by completion waiter threads.
struct SpineInner {
    shared: Arc<VulkanShared>,
    command_pool: vk::CommandPool,
    state: Mutex<SpineState>,
}

/// Mutable submission state guarded as one transaction.
struct SpineState {
    /// Last serial that has entered Phase B.  Zero is deliberately never issued.
    issued: u64,
    /// Last serial known complete by an explicit fence query.
    completed: u64,
    /// Fence waiters are independent host threads and may report a later
    /// single-queue fence before an earlier waiter gets scheduled. Keep those
    /// observations here; the public completion frontier advances only across
    /// a contiguous prefix, after each batch's readbacks have been published.
    finished: BTreeSet<u64>,
    /// First serial whose queue outcome cannot safely be observed.  All later
    /// serials are behind it on the same queue and are terminal for the same
    /// reason.
    poison: Option<(u64, CompletionFailure)>,
    /// One fence per accepted batch.  Per-batch fences preserve v13's finer
    /// completion without exposing a Vulkan fence as a public token.
    pending: BTreeMap<u64, PendingBatch>,
    /// Last queue-accepted layout of every transferred texture subresource.
    /// ObjectId never aliases a later texture, unlike a recycled VkImage handle.
    /// Entries may outlive the logical texture; retirement is a bounded-memory
    /// optimization and must not weaken cross-submit layout correctness.
    image_layouts: Vec<ImageLayoutState>,
}

struct PendingBatch {
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
    /// Portable buffer handles and backend staging allocations referenced by
    /// the accepted command buffer.  Native Vulkan handles alone do not extend
    /// Fluxel resource lifetime, so releasing this only after the fence is
    /// terminal is part of the v13 ownership contract.
    retention: TransferRetention,
}

struct RecordedBatch {
    command_buffer: vk::CommandBuffer,
    retention: TransferRetention,
}

impl VulkanCommandSpine {
    /// Creates the private command pool for the device's selected graphics
    /// family.  The pool is reset only after its fence reaches completion; the
    /// initial implementation destroys completed command buffers instead of
    /// pooling them. This is deliberately a correctness baseline: a production
    /// command-buffer arena may replace it after measurements, without changing
    /// the transactional submission contract.
    pub(in crate::backend::vulkan) fn new(
        shared: Arc<VulkanShared>,
    ) -> Result<Self, VulkanFailure> {
        let create = vk::CommandPoolCreateInfo::default()
            .queue_family_index(shared.graphics_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: the queue family was selected while this VkDevice was created;
        // the create-info borrows only this stack value for the duration of call.
        let command_pool =
            unsafe { shared.device.create_command_pool(&create, None) }.map_err(|result| {
                VulkanFailure::Native(ffi::NativeError::new(
                    result,
                    "VulkanCommandSpine::create_command_pool",
                ))
            })?;
        Ok(Self {
            inner: Arc::new(SpineInner {
                shared: Arc::clone(&shared),
                command_pool,
                state: Mutex::new(SpineState {
                    issued: 0,
                    completed: 0,
                    finished: BTreeSet::new(),
                    poison: None,
                    pending: BTreeMap::new(),
                    image_layouts: Vec::new(),
                }),
            }),
        })
    }

    fn lock(&self) -> MutexGuard<'_, SpineState> {
        self.inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(in crate::backend::vulkan) fn poll(&self) {
        // Phase B installs a blocking waiter for every fence. Avoid querying a
        // fence here: host access to a fence is externally synchronized, and a
        // poll racing that waiter would violate Vulkan's synchronization rule.
    }

    /// The only blocking operation on this spine. Holding the queue-domain lock
    /// satisfies Vulkan's external synchronization rule for queue idle against
    /// concurrent submission on the same logical device.
    pub(in crate::backend::vulkan) fn wait_idle(&self) -> Result<(), VulkanFailure> {
        let _state = self.lock();
        unsafe { self.inner.shared.device.device_wait_idle() }.map_err(|result| {
            VulkanFailure::Native(ffi::NativeError::new(
                result,
                "VulkanCommandSpine::wait_idle",
            ))
        })?;
        // Fence waiters publish terminal completion independently of poll.
        drop(_state);
        Ok(())
    }

    /// Lowers one complete plan under the v13 two-phase acceptance rule.
    ///
    /// Buffer copy/upload/readback lowerings are completed entirely in Phase A.
    /// Any unsupported payload, allocation failure, map failure, or command
    /// recording failure therefore returns before the first `vkQueueSubmit`.
    pub(in crate::backend::vulkan) fn submit(
        &self,
        request: &SubmissionRequest<'_>,
    ) -> Result<SubmissionOutcome, VulkanFailure> {
        let mut state = self.lock();
        if self.inner.shared.loss_info().is_some() {
            return Err(VulkanFailure::Native(ffi::NativeError::new(
                vk::Result::ERROR_DEVICE_LOST,
                "VulkanCommandSpine::submit after device loss",
            )));
        }

        let recorded = self.allocate_and_record(request.batches, &state.image_layouts)?;
        let first_serial = state
            .issued
            .checked_add(1)
            .ok_or(VulkanFailure::Unsupported {
                what: "another Vulkan submission",
                why: "the backend completion serial space is exhausted",
            })?;
        let last_serial = first_serial
            .checked_add(recorded.len() as u64)
            .and_then(|serial| serial.checked_sub(1))
            .ok_or(VulkanFailure::Unsupported {
                what: "a submission with too many batches",
                why: "the backend completion serial space is exhausted",
            })?;

        // Allocate every completion fence before Phase B.  Failure here still
        // leaves the queue untouched, so it is an honest `Err`.
        let mut fences = Vec::with_capacity(recorded.len());
        for _ in &recorded {
            let create = vk::FenceCreateInfo::default();
            // SAFETY: the device remains alive through `shared`; no pointer is
            // retained by Vulkan beyond this creation call.
            let fence = match unsafe { self.inner.shared.device.create_fence(&create, None) } {
                Ok(fence) => fence,
                Err(result) => {
                    for fence in fences.drain(..) {
                        unsafe { self.inner.shared.device.destroy_fence(fence, None) };
                    }
                    unsafe {
                        self.inner.shared.device.free_command_buffers(
                            self.inner.command_pool,
                            &recorded
                                .iter()
                                .map(|batch| batch.command_buffer)
                                .collect::<Vec<_>>(),
                        )
                    };
                    return Err(VulkanFailure::Native(ffi::NativeError::new(
                        result,
                        "VulkanCommandSpine::create_fence",
                    )));
                }
            };
            fences.push(fence);
        }

        // Phase B: submit one batch at a time so each has an exact completion
        // fence.  Queue order supplies all same-queue and explicit-plan order;
        // multi-queue waits are intentionally not advertised yet.
        state.issued = last_serial;
        let mut poisoned = None;
        let mut recorded = recorded.into_iter();
        let mut fences = fences.into_iter();
        for index in 0..request.batches.len() {
            let batch_recording = recorded
                .next()
                .expect("Phase A allocated one buffer per batch");
            let buffer = batch_recording.command_buffer;
            let fence = fences
                .next()
                .expect("Phase A allocated one fence per batch");
            let serial = first_serial + index as u64;
            let command_buffers = [buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            // SAFETY: the command buffer was ended in Phase A, belongs to this
            // device/pool, and the fence is fresh and unsignalled.
            let result = unsafe {
                self.inner.shared.device.queue_submit(
                    self.inner.shared.graphics_queue,
                    &[submit],
                    fence,
                )
            };
            match result {
                Ok(()) => {
                    let readbacks = batch_recording.retention.readback_tickets();
                    state.image_layouts = batch_recording.retention.image_layouts();
                    state.pending.insert(
                        serial,
                        PendingBatch {
                            fence,
                            command_buffer: buffer,
                            retention: batch_recording.retention,
                        },
                    );
                    self.inner.shared.register_readbacks(&readbacks);
                    spawn_fence_waiter(Arc::clone(&self.inner), serial, fence);
                }
                Err(result) => {
                    // Do not return an error after queue submission was attempted.
                    // Retain the fence: a driver that accepted work despite the
                    // error may still use it, and destroying it early is unsafe.
                    let readbacks = batch_recording.retention.readback_tickets();
                    state.pending.insert(
                        serial,
                        PendingBatch {
                            fence,
                            command_buffer: buffer,
                            retention: batch_recording.retention,
                        },
                    );
                    self.inner.shared.register_readbacks(&readbacks);
                    let failure = VulkanFailure::Native(ffi::NativeError::new(
                        result,
                        "VulkanCommandSpine::queue_submit",
                    ));
                    let completion = CompletionFailure::new(failure.message());
                    state.poison = Some((serial, completion));
                    // Vulkan gives no portable proof that a failing submission
                    // accepted zero work. Fluxel's transactional API therefore
                    // cannot return Err here; poison this DeviceIdentity even
                    // for a non-DEVICE_LOST VkResult, wake every pending future,
                    // and require a fresh request_device for further work.
                    poisoned = Some(DeviceLossInfo::new(format!(
                        "Vulkan queue submission failed after work may have been accepted: {}",
                        failure.message()
                    )));
                    // No later batch was submitted. Its command buffers and
                    // fences are ordinary host-owned objects and can be freed
                    // immediately; their logical serials remain poisoned.
                    let remaining_buffers = recorded
                        .map(|batch| batch.command_buffer)
                        .collect::<Vec<_>>();
                    if !remaining_buffers.is_empty() {
                        unsafe {
                            self.inner
                                .shared
                                .device
                                .free_command_buffers(self.inner.command_pool, &remaining_buffers)
                        };
                    }
                    for fence in fences {
                        unsafe { self.inner.shared.device.destroy_fence(fence, None) };
                    }
                    break;
                }
            }
        }
        let points = request
            .batches
            .iter()
            .enumerate()
            .map(|(index, batch)| (batch.point, first_serial + index as u64))
            .collect();
        drop(state);
        if let Some(info) = poisoned {
            self.inner.shared.mark_lost(info);
        }
        Ok(SubmissionOutcome {
            completion: last_serial,
            points,
        })
    }

    /// Performs all native allocation, recording and close work in Phase A.
    fn allocate_and_record(
        &self,
        batches: &[PlanBatch],
        initial_image_layouts: &[ImageLayoutState],
    ) -> Result<Vec<RecordedBatch>, VulkanFailure> {
        let allocation = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.inner.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(u32::try_from(batches.len()).map_err(|_| {
                VulkanFailure::Unsupported {
                    what: "a submission with too many batches",
                    why: "Vulkan command-buffer count is u32",
                }
            })?);
        // SAFETY: the pool is private to this spine and locked across this Phase
        // A allocation; Vulkan writes handles into Ash-owned storage only.
        let buffers = unsafe {
            self.inner
                .shared
                .device
                .allocate_command_buffers(&allocation)
        }
        .map_err(|result| {
            VulkanFailure::Native(ffi::NativeError::new(
                result,
                "VulkanCommandSpine::allocate_command_buffers",
            ))
        })?;
        let mut recorded = Vec::with_capacity(buffers.len());
        // The plan's command buffers are submitted in this exact queue order.
        // Seed from the queue's accepted cross-submit state, then carry changes
        // across Phase-A batches. Phase B publishes a batch's final table only
        // after vkQueueSubmit accepts that batch.
        let mut plan_image_layouts = initial_image_layouts.to_vec();
        for (buffer, batch) in buffers.iter().copied().zip(batches) {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            // SAFETY: each newly allocated primary buffer is in initial state.
            if let Err(result) = unsafe {
                self.inner
                    .shared
                    .device
                    .begin_command_buffer(buffer, &begin)
            } {
                unsafe {
                    self.inner
                        .shared
                        .device
                        .free_command_buffers(self.inner.command_pool, &buffers)
                };
                return Err(VulkanFailure::Native(ffi::NativeError::new(
                    result,
                    "VulkanCommandSpine::begin_command_buffer",
                )));
            }
            let mut retention = TransferRetention::default();
            retention.seed_image_layouts(&plan_image_layouts);
            if let Err(error) = self.record_batch(buffer, batch, &mut retention) {
                unsafe {
                    self.inner
                        .shared
                        .device
                        .free_command_buffers(self.inner.command_pool, &buffers)
                };
                return Err(error);
            }
            // SAFETY: every lowered Vulkan command was recorded above; a
            // successful end makes the buffer immutable before Phase B.
            if let Err(result) = unsafe { self.inner.shared.device.end_command_buffer(buffer) } {
                unsafe {
                    self.inner
                        .shared
                        .device
                        .free_command_buffers(self.inner.command_pool, &buffers)
                };
                return Err(VulkanFailure::Native(ffi::NativeError::new(
                    result,
                    "VulkanCommandSpine::end_command_buffer",
                )));
            }
            recorded.push(RecordedBatch {
                command_buffer: buffer,
                retention,
            });
            plan_image_layouts = recorded
                .last()
                .expect("the just-recorded batch exists")
                .retention
                .image_layouts();
        }
        Ok(recorded)
    }

    /// Refuses unimplemented payloads explicitly.  Matching every variant is a
    /// compile-time tripwire: adding a portable command cannot silently become a
    /// no-op on Vulkan.
    fn record_batch(
        &self,
        command_buffer: vk::CommandBuffer,
        batch: &PlanBatch,
        retention: &mut TransferRetention,
    ) -> Result<(), VulkanFailure> {
        let mut raster_scope = None;
        for (work_index, work) in batch.work.iter().enumerate() {
            for (command_index, command) in work.commands().iter().enumerate() {
                match &command.payload {
                    RecordedPayload::RasterBegin(begin) => {
                        if raster_scope.is_some() {
                            return Err(VulkanFailure::Unsupported {
                                what: "nested Vulkan raster scopes",
                                why: "portable recording should keep raster scopes linear",
                            });
                        }
                        let shader_texture_uses =
                            collect_raster_shader_texture_uses(batch, work_index, command_index)?;
                        raster_scope = Some(raster::lower_raster_begin(
                            Arc::clone(&self.inner.shared),
                            command_buffer,
                            begin,
                            &shader_texture_uses,
                            retention,
                        )?);
                    }
                    RecordedPayload::RasterDraw(draw) => {
                        let scope = raster_scope.as_ref().ok_or(VulkanFailure::Unsupported {
                            what: "a Vulkan raster draw outside a render pass",
                            why: "portable recording should emit RasterBegin first",
                        })?;
                        let draw_retention = raster::lower_raster_draw(
                            &self.inner.shared,
                            command_buffer,
                            draw,
                            &command.uses,
                            scope,
                            retention,
                        )?;
                        retention.retain_raster(draw_retention);
                    }
                    RecordedPayload::RasterEnd => {
                        let scope = raster_scope.take().ok_or(VulkanFailure::Unsupported {
                            what: "a Vulkan raster-scope end without a begin",
                            why: "portable recording should keep raster scopes balanced",
                        })?;
                        let mut raster_retention = raster::RasterRetention::default();
                        raster::lower_raster_end(
                            command_buffer,
                            scope,
                            &mut raster_retention,
                            retention,
                        )?;
                        retention.retain_raster(raster_retention);
                    }
                    RecordedPayload::ComputeBegin(_) | RecordedPayload::ComputeEnd => {}
                    RecordedPayload::ComputeDispatch(dispatch) => {
                        let compute = compute::lower_compute_dispatch(
                            &self.inner.shared,
                            command_buffer,
                            dispatch,
                            &command.uses,
                            retention,
                        )?;
                        retention.retain_compute(compute);
                    }
                    RecordedPayload::Copy(CopyRecord::Buffer(copy)) => {
                        transfer::lower_buffer_copy(
                            &self.inner.shared,
                            command_buffer,
                            copy,
                            retention,
                        )?;
                    }
                    RecordedPayload::Copy(CopyRecord::BufferToTexture(copy)) => {
                        transfer::lower_buffer_texture_copy(
                            &self.inner.shared,
                            command_buffer,
                            copy,
                            true,
                            retention,
                        )?;
                    }
                    RecordedPayload::Copy(CopyRecord::TextureToBuffer(copy)) => {
                        transfer::lower_buffer_texture_copy(
                            &self.inner.shared,
                            command_buffer,
                            copy,
                            false,
                            retention,
                        )?;
                    }
                    RecordedPayload::Copy(CopyRecord::Texture(copy)) => {
                        transfer::lower_texture_copy(
                            &self.inner.shared,
                            command_buffer,
                            copy,
                            retention,
                        )?;
                    }
                    RecordedPayload::Upload(job) => match job.descriptor() {
                        crate::api::resource::transfer::UploadDescriptor::Buffer(_) => {
                            transfer::lower_upload(
                                &self.inner.shared,
                                command_buffer,
                                job,
                                retention,
                            )?
                        }
                        crate::api::resource::transfer::UploadDescriptor::Texture(_) => {
                            transfer::lower_texture_upload(
                                &self.inner.shared,
                                command_buffer,
                                job,
                                retention,
                            )?
                        }
                    },
                    RecordedPayload::Readback(ticket) => match ticket.request() {
                        crate::api::resource::transfer::ReadbackRequest::Buffer { .. } => {
                            transfer::lower_readback(
                                &self.inner.shared,
                                command_buffer,
                                ticket,
                                retention,
                            )?
                        }
                        crate::api::resource::transfer::ReadbackRequest::Texture { .. } => {
                            transfer::lower_texture_readback(
                                &self.inner.shared,
                                command_buffer,
                                ticket,
                                retention,
                            )?
                        }
                    },
                    // Debug markup has no execution semantics and no portable
                    // capability gate. Refusing it would make an otherwise
                    // supported workload fail merely because diagnostics were
                    // added. The Vulkan 1.0 baseline therefore accepts it as a
                    // no-op when VK_EXT_debug_utils is not enabled.
                    //
                    // TODO(tooling): enable VK_EXT_debug_utils when the instance
                    // advertises it and lower these three payloads (plus scope
                    // labels) to vkCmdBegin/End/InsertDebugUtilsLabelEXT. Keep
                    // this correct no-op fallback for loaders without the
                    // extension; debug labels must never become a required
                    // execution capability.
                    RecordedPayload::DebugPush(_)
                    | RecordedPayload::DebugPop
                    | RecordedPayload::DebugMarker(_) => {}
                    other => {
                        return Err(VulkanFailure::Unsupported {
                            what: payload_name(other),
                            why: "Vulkan command lowering for this payload has not been implemented",
                        });
                    }
                }
            }
        }
        if raster_scope.is_some() {
            return Err(VulkanFailure::Unsupported {
                what: "an unterminated Vulkan raster scope",
                why: "portable recording should emit RasterEnd before finish",
            });
        }
        Ok(())
    }

    /// Non-blocking completion query.  This path never waits for the GPU.
    pub(in crate::backend::vulkan) fn completion(&self, serial: u64) -> CompletionState {
        let state = self.lock();
        let answer = completion_from_state(&state, serial);
        let known_complete = serial != 0 && serial <= state.completed;
        drop(state);
        if let Some(info) = self.inner.shared.loss_info() {
            return if known_complete {
                CompletionState::Complete
            } else {
                CompletionState::DeviceLost(info)
            };
        }
        answer
    }

    /// Samples and registers a future waker under the same mutex used by
    /// completion advancement, preventing a completed fence from being missed
    /// between the sample and registration.
    pub(in crate::backend::vulkan) fn completion_or_register_waker(
        &self,
        serial: u64,
        waker: &Waker,
    ) -> CompletionState {
        let state = self.lock();
        let answer = completion_from_state(&state, serial);
        let known_complete = serial != 0 && serial <= state.completed;
        drop(state);
        if let Some(info) = self.inner.shared.loss_info() {
            return if known_complete {
                CompletionState::Complete
            } else {
                CompletionState::DeviceLost(info)
            };
        }
        if matches!(answer, CompletionState::Pending) {
            if let Err(info) = self.inner.shared.register_completion_waker(serial, waker) {
                return CompletionState::DeviceLost(info);
            }
            // Completion may have raced registration. Re-sample so a fence that
            // became signalled just before registration cannot strand a future.
            let state = self.lock();
            let answer = completion_from_state(&state, serial);
            drop(state);
            if !matches!(answer, CompletionState::Pending) {
                self.inner.shared.wake_completion(serial);
            }
            return answer;
        }
        answer
    }
}

impl Drop for SpineInner {
    fn drop(&mut self) {
        // Unlike COM-backed APIs, Vulkan command buffers and their pool may not
        // be destroyed while submitted work still uses them. The last public
        // Device owner can disappear without the caller awaiting its receipt,
        // so destruction itself must close that native lifetime. A lost device
        // may reject the wait; destruction is still the only remaining cleanup
        // path in that terminal domain.
        let _ = unsafe { self.shared.device.device_wait_idle() };
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Device shutdown drops the spine before VulkanShared.  The device may
        // already be lost; Vulkan destruction is still the required ownership
        // cleanup and does not turn a destructor into a recovery operation.
        for (_, pending) in std::mem::take(&mut state.pending) {
            unsafe { self.shared.device.destroy_fence(pending.fence, None) };
            unsafe {
                self.shared
                    .device
                    .free_command_buffers(self.command_pool, &[pending.command_buffer])
            };
        }
        unsafe {
            self.shared
                .device
                .destroy_command_pool(self.command_pool, None)
        };
    }
}

/// Waits independently of `Device::poll`, then advances both completion and
/// readback state.  This is intentionally one waiter per batch for the first
/// Vulkan vertical slice.  A shared fence waiter may replace it later without
/// changing the completion contract; this baseline's important property is that
/// a `CompletionPoint` or `ReadbackTicket` future always makes progress.
fn spawn_fence_waiter(inner: Arc<SpineInner>, serial: u64, fence: vk::Fence) {
    let worker = Arc::clone(&inner);
    let result = std::thread::Builder::new()
        .name("fluxel-vulkan-fence".into())
        .spawn(move || {
            // SAFETY: `worker` retains both the command pool and VulkanShared;
            // the fence was inserted into its state before this thread starts.
            let result = unsafe {
                worker
                    .shared
                    .device
                    .wait_for_fences(&[fence], true, u64::MAX)
            };
            finish_waited_batch(&worker, serial, result);
        });
    if result.is_err() {
        // Work was already accepted, so a thread-creation failure cannot be
        // returned as submit Err. Make the execution domain terminal instead;
        // this wakes all futures rather than leaving accepted work Pending.
        terminate_waiter_failure(&inner, serial, "could not start Vulkan fence waiter");
    }
}

fn finish_waited_batch(inner: &SpineInner, serial: u64, result: Result<(), vk::Result>) {
    let mut state = inner
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(pending) = state.pending.remove(&serial) else {
        return;
    };
    match result {
        Ok(()) => {
            let tickets = pending.retention.readback_tickets();
            let committed = inner.shared.commit_completion(&tickets, || {
                let mut terminal_loss = None;
                for readback in &pending.retention.readbacks {
                    if let Err(result) = transfer::publish_readback(readback) {
                        if result == vk::Result::ERROR_DEVICE_LOST {
                            terminal_loss = Some(
                                "Vulkan reported VK_ERROR_DEVICE_LOST while publishing completed readback data",
                            );
                            break;
                        }
                    }
                }
                terminal_loss
            });
            let terminal_loss = match committed {
                Ok(loss) => loss,
                Err(_) => {
                    // Device loss linearized first. The successful native wait
                    // does not get to change the terminal answer observers have
                    // already received for this still-pending logical point.
                    for readback in &pending.retention.readbacks {
                        readback.ticket.set_status(ReadbackStatus::DeviceLost);
                    }
                    unsafe {
                        inner.shared.device.destroy_fence(pending.fence, None);
                        inner
                            .shared
                            .device
                            .free_command_buffers(inner.command_pool, &[pending.command_buffer]);
                    }
                    drop(state);
                    return;
                }
            };
            unsafe {
                inner.shared.device.destroy_fence(pending.fence, None);
                inner
                    .shared
                    .device
                    .free_command_buffers(inner.command_pool, &[pending.command_buffer]);
            }
            if let Some(message) = terminal_loss {
                for other in state.pending.values() {
                    for readback in &other.retention.readbacks {
                        readback.ticket.set_status(ReadbackStatus::DeviceLost);
                    }
                }
                drop(state);
                inner
                    .shared
                    .mark_lost(DeviceLossInfo::new(message.to_owned()));
                return;
            }
            state.finished.insert(serial);
            let previous_frontier = state.completed;
            loop {
                let Some(next) = state.completed.checked_add(1) else {
                    break;
                };
                if !state.finished.remove(&next) {
                    break;
                }
                state.completed += 1;
            }
            let completed_frontier = state.completed;
            drop(state);
            for completed in previous_frontier + 1..=completed_frontier {
                inner.shared.wake_completion(completed);
            }
        }
        Err(result) if result == vk::Result::ERROR_DEVICE_LOST => {
            for readback in &pending.retention.readbacks {
                readback.ticket.set_status(ReadbackStatus::DeviceLost);
            }
            for other in state.pending.values() {
                for readback in &other.retention.readbacks {
                    readback.ticket.set_status(ReadbackStatus::DeviceLost);
                }
            }
            // Loss is not proof that DMA has stopped. Keep the native fence,
            // command buffer, staging, and portable resource owners retained
            // through execution-domain teardown rather than freeing memory a
            // removed device could conceivably still touch.
            state.pending.insert(serial, pending);
            drop(state);
            inner.shared.mark_lost(DeviceLossInfo::new(
                "Vulkan reported VK_ERROR_DEVICE_LOST while waiting for a completion fence"
                    .to_owned(),
            ));
        }
        Err(result) => {
            for readback in &pending.retention.readbacks {
                readback.ticket.set_status(ReadbackStatus::DeviceLost);
            }
            state.poison = Some((
                serial,
                CompletionFailure::new(format!(
                    "Vulkan fence wait became unobservable after submission: {result:?}"
                )),
            ));
            // An unobservable fence is likewise not permission to retire the
            // work's native ownership. It is released only during teardown.
            state.pending.insert(serial, pending);
            drop(state);
            inner.shared.mark_lost(DeviceLossInfo::new(format!(
                "Vulkan completion fence failed after submission and the execution domain can no longer prove progress: {result:?}"
            )));
        }
    }
}

fn terminate_waiter_failure(inner: &SpineInner, serial: u64, message: &'static str) {
    let state = inner
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for pending in state.pending.range(serial..) {
        for readback in &pending.1.retention.readbacks {
            readback.ticket.set_status(ReadbackStatus::DeviceLost);
        }
    }
    drop(state);
    inner
        .shared
        .mark_lost(DeviceLossInfo::new(message.to_owned()));
}

fn completion_from_state(state: &SpineState, serial: u64) -> CompletionState {
    if serial == 0 {
        CompletionState::Failed(CompletionFailure::new(
            "Vulkan completion serial zero is reserved and was never issued",
        ))
    } else if serial <= state.completed {
        CompletionState::Complete
    } else if let Some((first, failure)) = &state.poison {
        if serial >= *first {
            CompletionState::Failed(failure.clone())
        } else {
            CompletionState::Pending
        }
    } else if serial <= state.issued {
        CompletionState::Pending
    } else {
        CompletionState::Failed(CompletionFailure::new(
            "Vulkan completion was queried for a serial this device never issued",
        ))
    }
}

/// Collects exactly the shader image uses in one linear raster scope before it
/// is begun natively. Vulkan synchronization commands are invalid inside a
/// render pass, so raster lowering establishes descriptor layouts at the scope
/// boundary rather than when each draw is replayed.
fn collect_raster_shader_texture_uses(
    batch: &PlanBatch,
    begin_work: usize,
    begin_command: usize,
) -> Result<Vec<TextureUse>, VulkanFailure> {
    let mut result = Vec::new();
    // A Vulkan 1.0 render pass cannot insert a pipeline barrier between two
    // draws. Sample-only reuse is safe, but any storage-image use shared with
    // another draw would need an in-pass visibility dependency this baseline
    // does not create. Keep that shape out of the advertised/lowered subset.
    let mut prior_draw_images = Vec::new();
    let mut prior_draw_storage_images = Vec::new();
    for (work_index, work) in batch.work.iter().enumerate().skip(begin_work) {
        let first_command = if work_index == begin_work {
            begin_command + 1
        } else {
            0
        };
        for command in work.commands().iter().skip(first_command) {
            match &command.payload {
                RecordedPayload::RasterEnd => return Ok(result),
                RecordedPayload::RasterBegin(_) => {
                    return Err(VulkanFailure::Unsupported {
                        what: "nested Vulkan raster scopes",
                        why: "portable recording should keep raster scopes linear",
                    });
                }
                RecordedPayload::RasterDraw(_) => {
                    let shader_uses: Vec<_> = command
                        .uses
                        .iter()
                        .filter_map(|use_| match use_ {
                            ResourceUse::Texture(texture)
                                if matches!(
                                    texture.intent,
                                    TextureUseIntent::ShaderRead
                                        | TextureUseIntent::ShaderReadWrite
                                ) =>
                            {
                                Some(texture.clone())
                            }
                            _ => None,
                        })
                        .collect();
                    let current_images: Vec<_> = shader_uses
                        .iter()
                        .map(|texture| texture.texture.id())
                        .collect();
                    let current_storage_images: Vec<_> = shader_uses
                        .iter()
                        .filter(|texture| texture.intent == TextureUseIntent::ShaderReadWrite)
                        .map(|texture| texture.texture.id())
                        .collect();
                    let current_sampled_images: Vec<_> = shader_uses
                        .iter()
                        .filter(|texture| texture.intent == TextureUseIntent::ShaderRead)
                        .map(|texture| texture.texture.id())
                        .collect();
                    if current_storage_images
                        .iter()
                        .any(|id| current_sampled_images.contains(id))
                    {
                        return Err(VulkanFailure::Unsupported {
                            what: "one Vulkan raster draw binding the same texture as sampled and storage",
                            why: "one image cannot satisfy SHADER_READ_ONLY_OPTIMAL and GENERAL descriptors simultaneously",
                        });
                    }
                    if current_storage_images
                        .iter()
                        .any(|id| prior_draw_images.contains(id))
                        || prior_draw_storage_images
                            .iter()
                            .any(|id| current_images.contains(id))
                    {
                        return Err(VulkanFailure::Unsupported {
                            what: "a Vulkan raster storage image shared across draws in one render pass",
                            why: "this baseline has no in-render-pass shader memory barrier lowering",
                        });
                    }
                    prior_draw_images.extend(current_images);
                    prior_draw_storage_images.extend(current_storage_images);
                    result.extend(shader_uses);
                }
                _ => {}
            }
        }
    }
    Err(VulkanFailure::Unsupported {
        what: "an unterminated Vulkan raster scope",
        why: "portable recording should emit RasterEnd before finish",
    })
}

fn payload_name(payload: &RecordedPayload) -> &'static str {
    match payload {
        RecordedPayload::RasterBegin(_) => "a raster scope",
        RecordedPayload::RasterDraw(_) => "a raster draw",
        RecordedPayload::RasterEnd => "a raster-scope end",
        RecordedPayload::ComputeBegin(_) => "a compute scope",
        RecordedPayload::ComputeDispatch(_) => "a compute dispatch",
        RecordedPayload::ComputeEnd => "a compute-scope end",
        RecordedPayload::Copy(_) => "a copy command",
        RecordedPayload::Upload(_) => "an upload command",
        RecordedPayload::Readback(_) => "a readback command",
        RecordedPayload::DebugPush(_) => "a debug-group push",
        RecordedPayload::DebugPop => "a debug-group pop",
        RecordedPayload::DebugMarker(_) => "a debug marker",
    }
}
