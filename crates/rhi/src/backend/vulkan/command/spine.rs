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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::Waker;

use ash::vk;

use crate::api::command::record::{CopyRecord, RecordedPayload};
use crate::api::command::{ResourceUse, TextureUse, TextureUseIntent};
use crate::api::platform::DeviceLossInfo;
use crate::api::presentation::{FrameAttachment, PresentState};
use crate::api::resource::transfer::ReadbackStatus;
use crate::api::submission::backend::{SubmissionOutcome, SubmissionRequest};
use crate::api::submission::plan::PlanBatch;
use crate::api::submission::{CompletionFailure, CompletionState};
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::ffi;
use crate::backend::vulkan::platform::device::VulkanShared;
use crate::backend::vulkan::resource::{VulkanBuffer, VulkanQuerySet};

use super::compute;
use super::raster;
use super::transfer::{self, ImageLayoutState, TransferRetention};

#[cfg(any(windows, target_os = "android"))]
use crate::backend::vulkan::presentation::{VulkanFrameAttachment, VulkanPresentSync};
#[cfg(not(any(windows, target_os = "android")))]
#[derive(Clone, Copy)]
struct VulkanPresentSync {
    acquire_wait: vk::Semaphore,
    render_finished: vk::Semaphore,
}

/// One single-queue Vulkan command domain.
pub(in crate::backend::vulkan) struct VulkanCommandSpine {
    inner: Arc<SpineInner>,
    waiter: Option<std::thread::JoinHandle<()>>,
}

/// Queue-local native objects retained by completion waiter threads.
struct SpineInner {
    shared: Arc<VulkanShared>,
    command_pool: vk::CommandPool,
    state: Mutex<SpineState>,
    pending_changed: Condvar,
    shutdown: AtomicBool,
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

#[derive(Default)]
struct BatchPresentation {
    waits: Vec<vk::Semaphore>,
    signals: Vec<vk::Semaphore>,
    presents: Vec<usize>,
}

/// A fence wait is deliberately bounded even though the normal completion path
/// has no deadline.  `VulkanCommandSpine::drop` must be able to join its sole
/// waiter before it destroys the command pool, and Vulkan has no operation that
/// cancels an in-progress `vkWaitForFences`.  A finite wait is therefore the
/// shutdown observation point; it is *not* a completion timeout.
const FENCE_WAITER_POLL_NS: u64 = 50_000_000;

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
        let inner = Arc::new(SpineInner {
            shared: Arc::clone(&shared),
            command_pool,
            pending_changed: Condvar::new(),
            shutdown: AtomicBool::new(false),
            state: Mutex::new(SpineState {
                issued: 0,
                completed: 0,
                finished: BTreeSet::new(),
                poison: None,
                pending: BTreeMap::new(),
                image_layouts: Vec::new(),
            }),
        });
        let worker = Arc::clone(&inner);
        let waiter = std::thread::Builder::new()
            .name("fluxel-vulkan-fence".into())
            .spawn(move || fence_wait_loop(worker))
            .map_err(|_| {
                VulkanFailure::Native(ffi::NativeError::new(
                    vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                    "VulkanCommandSpine::spawn fence waiter",
                ))
            })?;
        Ok(Self {
            inner,
            waiter: Some(waiter),
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
        let issued = self.lock().issued;
        if issued == 0 {
            return Ok(());
        }
        {
            let _queue = self.inner.shared.queue_guard();
            unsafe { self.inner.shared.device.device_wait_idle() }.map_err(|result| {
                VulkanFailure::Native(ffi::NativeError::new(
                    result,
                    "VulkanCommandSpine::wait_idle",
                ))
            })?;
        }

        // `vkDeviceWaitIdle` proves the queue is idle, but it does not transfer
        // ownership of the worker's host fence wait or of its `PendingBatch`.
        // Let that one authority observe the signalled fences, publish readbacks,
        // wake futures, and free the native objects. Returning before this loop
        // would make `wait_idle().await` falsely imply a still-Pending completion
        // or readback. The bounded wait also observes a loss discovered by another
        // native entry point even if it did not notify this spine's condition
        // variable.
        let mut state = self.lock();
        while state.completed < issued {
            if self.inner.shared.loss_info().is_some() {
                return Err(VulkanFailure::Native(ffi::NativeError::new(
                    vk::Result::ERROR_DEVICE_LOST,
                    "VulkanCommandSpine::wait_idle after device loss",
                )));
            }
            let (next, _) = self
                .inner
                .pending_changed
                .wait_timeout(state, std::time::Duration::from_nanos(FENCE_WAITER_POLL_NS))
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next;
        }
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

        // An empty plan is a legal portable no-op with its own plan identity.
        // Keep native command-buffer allocation and queue submission entirely
        // out of this path; serial zero is the completed identity frontier of a
        // fresh device, and a later empty plan observes the current frontier.
        if request.batches.is_empty() {
            return Ok(SubmissionOutcome {
                completion: state.issued,
                points: Vec::new(),
            });
        }

        let recorded = self.allocate_and_record(request.batches, &state.image_layouts)?;
        // Presentation synchronization is part of Phase A. A foreign/non-Vulkan
        // frame must be refused before the first vkQueueSubmit, preserving the
        // portable `Err == zero native work accepted` contract.
        let presentation = prepare_presentations(request)?;
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
        let mut presented = vec![false; request.presents.len()];
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
            let batch_presentation = &presentation[index];
            let wait_stages = vec![
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
                batch_presentation.waits.len()
            ];
            let submit = vk::SubmitInfo::default()
                .command_buffers(&command_buffers)
                .wait_semaphores(&batch_presentation.waits)
                .wait_dst_stage_mask(&wait_stages)
                .signal_semaphores(&batch_presentation.signals);
            // SAFETY: the command buffer was ended in Phase A, belongs to this
            // device/pool, and the fence is fresh and unsignalled.
            let result = unsafe {
                let _queue = self.inner.shared.queue_guard();
                self.inner.shared.device.queue_submit(
                    self.inner.shared.graphics_queue,
                    &[submit],
                    fence,
                )
            };
            match result {
                Ok(()) => {
                    mark_batch_buffer_uses(&request.batches[index], serial);
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
                    self.inner.pending_changed.notify_one();
                    for &present_index in &batch_presentation.presents {
                        let present = &request.presents[present_index];
                        present.attachment.present(present.receipt);
                        presented[present_index] = true;
                    }
                    // vkQueuePresentKHR may be the first native call to report
                    // device loss. Do not feed later batches to a terminal
                    // queue; their receipts are completed below as DeviceLost.
                    if self.inner.shared.loss_info().is_some() {
                        let remaining_buffers = recorded
                            .by_ref()
                            .map(|batch| batch.command_buffer)
                            .collect::<Vec<_>>();
                        if !remaining_buffers.is_empty() {
                            unsafe {
                                self.inner.shared.device.free_command_buffers(
                                    self.inner.command_pool,
                                    &remaining_buffers,
                                )
                            };
                        }
                        for fence in fences.by_ref() {
                            unsafe { self.inner.shared.device.destroy_fence(fence, None) };
                        }
                        break;
                    }
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
            self.inner.pending_changed.notify_all();
        }
        if let Some(info) = self.inner.shared.loss_info() {
            for (index, present) in request.presents.iter().enumerate() {
                if !presented[index] {
                    present
                        .attachment
                        .terminate_present(present.receipt, PresentState::DeviceLost(info.clone()));
                }
            }
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
        // Legacy vkCmdResetQueryPool is forbidden inside a render pass. Reset
        // every slot used by this batch before replay can enter a raster scope;
        // reset is idempotent for duplicate references and preserves the
        // recorder's command order for begin/end/write operations themselves.
        for work in &batch.work {
            for command in work.commands() {
                let query = match &command.payload {
                    RecordedPayload::QueryBegin { set, index }
                    | RecordedPayload::TimestampWrite { set, index } => Some((set, *index)),
                    _ => None,
                };
                if let Some((set, index)) = query {
                    unsafe {
                        self.inner.shared.device.cmd_reset_query_pool(
                            command_buffer,
                            native_query_pool(set)?,
                            index,
                            1,
                        );
                    }
                }
            }
        }
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
                    RecordedPayload::RasterIndirect(draw) => {
                        let scope = raster_scope.as_ref().ok_or(VulkanFailure::Unsupported {
                            what: "a Vulkan raster indirect draw outside a render pass",
                            why: "portable recording should emit RasterBegin first",
                        })?;
                        let draw_retention = raster::lower_raster_indirect(
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
                    RecordedPayload::ComputeIndirect(dispatch) => {
                        let compute = compute::lower_compute_indirect(
                            &self.inner.shared,
                            command_buffer,
                            dispatch,
                            &command.uses,
                            retention,
                        )?;
                        retention.retain_compute(compute);
                        retention.buffers.push(dispatch.arguments.clone());
                    }
                    RecordedPayload::QueryBegin { set, index } => {
                        let pool = native_query_pool(set)?;
                        unsafe {
                            self.inner.shared.device.cmd_begin_query(
                                command_buffer,
                                pool,
                                *index,
                                vk::QueryControlFlags::empty(),
                            );
                        }
                        retention.query_sets.push(set.clone());
                    }
                    RecordedPayload::QueryEnd { set, index } => {
                        unsafe {
                            self.inner.shared.device.cmd_end_query(
                                command_buffer,
                                native_query_pool(set)?,
                                *index,
                            );
                        }
                        retention.query_sets.push(set.clone());
                    }
                    RecordedPayload::TimestampWrite { set, index } => {
                        let pool = native_query_pool(set)?;
                        unsafe {
                            self.inner.shared.device.cmd_write_timestamp(
                                command_buffer,
                                vk::PipelineStageFlags::ALL_COMMANDS,
                                pool,
                                *index,
                            );
                        }
                        retention.query_sets.push(set.clone());
                    }
                    RecordedPayload::QueryResolve(resolve) => {
                        unsafe {
                            self.inner.shared.device.cmd_copy_query_pool_results(
                                command_buffer,
                                native_query_pool(&resolve.set)?,
                                resolve.first_query,
                                resolve.query_count,
                                native_buffer(&resolve.destination)?,
                                resolve.destination_offset,
                                query_result_stride(&resolve.set)?,
                                // Timestamp facts explicitly say this slice
                                // does not promise non-blocking availability.
                                // WAIT turns the result copy into a defined
                                // producer/consumer dependency rather than
                                // exposing stale slots.
                                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                            );
                        }
                        retention.query_sets.push(resolve.set.clone());
                        retention.buffers.push(resolve.destination.clone());
                    }
                    RecordedPayload::Copy(CopyRecord::Buffer(copy)) => {
                        transfer::lower_buffer_copy(
                            &self.inner.shared,
                            command_buffer,
                            copy,
                            retention,
                        )?;
                    }
                    RecordedPayload::Copy(CopyRecord::ClearBuffer { buffer, range }) => {
                        transfer::lower_clear_buffer(
                            &self.inner.shared,
                            command_buffer,
                            buffer,
                            *range,
                            retention,
                        )?;
                    }
                    RecordedPayload::Copy(CopyRecord::ClearTexture {
                        texture,
                        subresources,
                    }) => {
                        transfer::lower_clear_texture(
                            &self.inner.shared,
                            command_buffer,
                            texture,
                            *subresources,
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
                    | RecordedPayload::DebugMarker(_) => {
                        return Err(VulkanFailure::Unsupported {
                            what: "a Vulkan debug-marker command",
                            why: "VK_EXT_debug_utils command lowering is not enabled by this device slice",
                        });
                    }
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

fn query_result_stride(set: &crate::api::query::QuerySet) -> Result<u64, VulkanFailure> {
    use crate::api::query::{PipelineStatistics, QueryType};
    let values = match set.descriptor().ty {
        QueryType::Occlusion | QueryType::Timestamp => 1,
        QueryType::PipelineStatistics(selection) => [
            PipelineStatistics::VERTEX_SHADER_INVOCATIONS,
            PipelineStatistics::CLIPPER_INVOCATIONS,
            PipelineStatistics::CLIPPER_PRIMITIVES_OUT,
            PipelineStatistics::FRAGMENT_SHADER_INVOCATIONS,
            PipelineStatistics::COMPUTE_SHADER_INVOCATIONS,
        ]
        .into_iter()
        .filter(|counter| selection.contains(*counter))
        .count(),
    };
    u64::try_from(values)
        .ok()
        .and_then(|values| values.checked_mul(8))
        .ok_or(VulkanFailure::Unsupported {
            what: "a Vulkan query-result stride",
            why: "the selected portable result layout overflowed Vulkan's stride type",
        })
}

impl Drop for VulkanCommandSpine {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Release);
        self.inner.pending_changed.notify_all();
        if let Some(waiter) = self.waiter.take() {
            let _ = waiter.join();
        }
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
        let _queue = self.shared.queue_guard();
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

/// One device-owned progress worker replaces one OS thread per batch. The
/// single queue issues serials in order, so waiting for the earliest pending
/// fence advances the exact portable frontier without sacrificing per-batch
/// completion points.
fn fence_wait_loop(inner: Arc<SpineInner>) {
    loop {
        let (serial, fence) = {
            let mut state = inner
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            loop {
                if inner.shutdown.load(Ordering::Acquire) || inner.shared.loss_info().is_some() {
                    return;
                }
                if let Some((&serial, pending)) = state.pending.first_key_value() {
                    break (serial, pending.fence);
                }
                state = inner
                    .pending_changed
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        };
        // SAFETY: the pending map retains this fence until this sole worker
        // publishes its terminal result. Shutdown joins this worker before the
        // command pool or fences are destroyed.
        let result = unsafe {
            inner
                .shared
                .device
                .wait_for_fences(&[fence], true, FENCE_WAITER_POLL_NS)
        };
        // A finite host wait only means this worker should resample shutdown,
        // loss, and the same earliest fence. It proves neither completion nor
        // failure, so in particular it must not remove `pending` or release the
        // command buffer/staging retained by that accepted batch.
        if fence_wait_timed_out(&result) {
            continue;
        }
        finish_waited_batch(&inner, serial, result);
        // The synchronous `wait_idle` path waits for the worker, rather than
        // touching a fence concurrently with it. Notify after every terminal
        // observation, including loss, so it need not wait for its bounded
        // resample interval in the usual case.
        inner.pending_changed.notify_all();
    }
}

/// Classifies the one non-terminal result of the bounded worker wait.
///
/// Keeping this separate makes it hard for a future refactor to accidentally
/// hand `TIMEOUT` to `finish_waited_batch`, where any error is deliberately a
/// terminal execution-domain failure.
fn fence_wait_timed_out(result: &Result<(), vk::Result>) -> bool {
    matches!(result, Err(vk::Result::TIMEOUT))
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
            inner.shared.advance_completed_serial(completed_frontier);
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

fn completion_from_state(state: &SpineState, serial: u64) -> CompletionState {
    if serial == 0 {
        CompletionState::Complete
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

/// Records the completion serial only after `vkQueueSubmit` accepted this
/// batch. Phase-A failures never reach here, preserving the map future's
/// guarantee that it waits for actual accepted GPU use rather than recorded
/// intent. Duplicate uses are harmless because the native buffer atomically
/// keeps the greatest serial.
fn mark_batch_buffer_uses(batch: &PlanBatch, serial: u64) {
    for work in &batch.work {
        for command in work.commands() {
            for use_ in &command.uses {
                if let ResourceUse::Buffer(buffer) = use_ {
                    if let Some(native) = buffer
                        .buffer
                        .native()
                        .as_any()
                        .downcast_ref::<VulkanBuffer>()
                    {
                        native.mark_accepted(serial);
                    }
                }
            }
        }
    }
}

fn prepare_presentations(
    request: &SubmissionRequest<'_>,
) -> Result<Vec<BatchPresentation>, VulkanFailure> {
    let mut batches = (0..request.batches.len())
        .map(|_| BatchPresentation::default())
        .collect::<Vec<_>>();
    for (present_index, present) in request.presents.iter().enumerate() {
        let sync = frame_present_sync(&present.attachment)?;
        let frame = present.attachment.frame_id();
        let first_use = request
            .batches
            .iter()
            .position(|batch| {
                batch.work.iter().any(|work| {
                    work.resource_uses()
                        .iter()
                        .any(|use_| matches!(use_, ResourceUse::Frame(use_) if use_.frame == frame))
                })
            })
            .ok_or(VulkanFailure::Unsupported {
                what: "a Vulkan presentation relation without frame work",
                why: "portable plan validation should require the presented frame to be used",
            })?;
        let present_batch = request
            .batches
            .iter()
            .position(|batch| batch.point == present.after)
            .ok_or(VulkanFailure::Unsupported {
                what: "a Vulkan presentation point outside the submitted plan",
                why: "portable plan validation should resolve every present-after point",
            })?;
        if first_use > present_batch {
            return Err(VulkanFailure::Unsupported {
                what: "a Vulkan presentation ordered before its first frame use",
                why: "portable plan validation should order every frame use before presentation",
            });
        }
        batches[first_use].waits.push(sync.acquire_wait);
        batches[present_batch].signals.push(sync.render_finished);
        batches[present_batch].presents.push(present_index);
    }
    Ok(batches)
}

#[cfg(any(windows, target_os = "android"))]
fn frame_present_sync(attachment: &FrameAttachment) -> Result<VulkanPresentSync, VulkanFailure> {
    attachment
        .native()
        .as_any()
        .downcast_ref::<VulkanFrameAttachment>()
        .map(VulkanFrameAttachment::sync)
        .ok_or(VulkanFailure::Unsupported {
            what: "a Vulkan presentation attachment",
            why: "its native drawable belongs to another backend",
        })
}

#[cfg(not(any(windows, target_os = "android")))]
fn frame_present_sync(_: &FrameAttachment) -> Result<VulkanPresentSync, VulkanFailure> {
    Err(VulkanFailure::Unsupported {
        what: "a Vulkan presentation attachment",
        why: "this platform has no Vulkan presentation lowering",
    })
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
                RecordedPayload::RasterDraw(_) | RecordedPayload::RasterIndirect(_) => {
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
        RecordedPayload::MeshDispatch(_) => "a mesh dispatch",
        RecordedPayload::MeshIndirect(_) => "an indirect mesh dispatch",
        RecordedPayload::RayTracingBegin(_) => "a ray-tracing scope",
        RecordedPayload::RayTracingDispatch(_) => "a ray dispatch",
        RecordedPayload::RayTracingEnd => "a ray-tracing scope end",
        RecordedPayload::AccelerationStructure(_) => "an acceleration-structure command",
        RecordedPayload::RasterBegin(_) => "a raster scope",
        RecordedPayload::RasterDraw(_) => "a raster draw",
        RecordedPayload::RasterEnd => "a raster-scope end",
        RecordedPayload::ComputeBegin(_) => "a compute scope",
        RecordedPayload::ComputeDispatch(_) => "a compute dispatch",
        RecordedPayload::RasterIndirect(_) => "an indirect raster draw",
        RecordedPayload::ComputeIndirect(_) => "an indirect compute dispatch",
        RecordedPayload::QueryBegin { .. } => "a query begin",
        RecordedPayload::QueryEnd { .. } => "a query end",
        RecordedPayload::TimestampWrite { .. } => "a timestamp write",
        RecordedPayload::QueryResolve(_) => "a query resolve",
        RecordedPayload::ComputeEnd => "a compute-scope end",
        RecordedPayload::Copy(_) => "a copy command",
        RecordedPayload::Upload(_) => "an upload command",
        RecordedPayload::Readback(_) => "a readback command",
        RecordedPayload::DebugPush(_) => "a debug-group push",
        RecordedPayload::DebugPop => "a debug-group pop",
        RecordedPayload::DebugMarker(_) => "a debug marker",
    }
}

fn native_query_pool(set: &crate::api::query::QuerySet) -> Result<vk::QueryPool, VulkanFailure> {
    set.native()
        .as_any()
        .downcast_ref::<VulkanQuerySet>()
        .map(VulkanQuerySet::pool)
        .ok_or(VulkanFailure::Unsupported {
            what: "a query set this Vulkan device did not create",
            why: "its VkQueryPool belongs to another backend",
        })
}

fn native_buffer(buffer: &crate::api::resource::Buffer) -> Result<vk::Buffer, VulkanFailure> {
    buffer
        .native()
        .as_any()
        .downcast_ref::<VulkanBuffer>()
        .map(VulkanBuffer::buffer)
        .ok_or(VulkanFailure::Unsupported {
            what: "a query resolve buffer this Vulkan device did not create",
            why: "its VkBuffer belongs to another backend",
        })
}

#[cfg(test)]
mod tests {
    use ash::vk;

    use super::fence_wait_timed_out;

    #[test]
    fn bounded_fence_wait_timeout_is_not_an_execution_failure() {
        assert!(fence_wait_timed_out(&Err(vk::Result::TIMEOUT)));
        assert!(!fence_wait_timed_out(&Ok(())));
        assert!(!fence_wait_timed_out(&Err(vk::Result::ERROR_DEVICE_LOST)));
    }
}
