//! The Direct3D 12 command spine: recording a batch, committing it, and
//! observing it finish.
//!
//! This module owns the four native objects Direct3D 12 requires before any work
//! can run — a command queue, command allocators, command lists, and a fence —
//! and the order a plan's batches are put through them. It decides nothing about
//! legality: every plan reaching it has passed section 40.5's checklist in the
//! portable layer, so a refusal produced here is one only Direct3D 12 can know.
//!
//! What a single step *is* — the transitions a copy needs, the staging an upload
//! or a readback allocates — lives in the sibling modules this chapter declares:
//! [`super::copy`], [`super::transfer`], [`super::transition`], and the shared
//! [`crate::backend::dx12::failure::Dx12Failure`]. This file is the machinery
//! those steps are recorded onto.
//!
//! # Record everything, then commit
//!
//! Section 41.3's Phase A is "an `Err` proves no native work was accepted", and
//! the way this module makes that literally true is by recording *every* batch of
//! the plan into its own command list before a single `ExecuteCommandLists` is
//! called. A payload this spine cannot lower is therefore refused with nothing
//! committed, rather than after an earlier batch is already on the queue — which
//! is the one outcome section 41.3 forbids.
//!
//! Committing per batch, rather than concatenating the whole plan into one list,
//! is what gives section 41.2's finer completion: each batch signals its own
//! fence value, so a readback ticket and an allocator's reuse both hang off their
//! own batch rather than off the slowest unrelated one. The cost is one
//! `ExecuteCommandLists` and one `Signal` per batch instead of one of each per
//! plan, which is the price of being able to answer "is *this* batch done".
//!
//! # The state invariant, and why there is no persistent tracker
//!
//! **Every command list this spine records leaves every buffer it touched in
//! `D3D12_RESOURCE_STATE_COMMON`.** Buffers are committed in `COMMON`
//! ([`crate::backend::dx12::resource`]), and each command transitions what it uses out of
//! `COMMON` and back before it is done.
//!
//! The alternative — a persistent per-resource tracker that remembers where each
//! resource was left — buys one thing: half the barriers. It costs a map that
//! must survive across submissions, must be rolled back exactly when a recording
//! is refused, and must be keyed by something stable across a resource's
//! lifetime. `CLAUDE.md` section 1 puts implementation simplicity below semantic
//! correctness but above execution efficiency, and the tracker's failure mode is
//! a hazard the driver reports as corruption rather than as an error. The
//! barriers are the price of an invariant that can be checked by reading one
//! sentence.
//!
//! Staging allocations are outside the invariant because Direct3D 12 forbids it:
//! an `UPLOAD` heap resource must be created in `GENERIC_READ` and a `READBACK`
//! heap resource in `COPY_DEST`, and neither heap permits a transition at all.
//! They are never named in a barrier, which is why the invariant stays true.
//!
//! # What this spine does not own
//!
//! - Whether a route exists. Section 9.4's answer comes from the portable
//!   layer's capability snapshot, and this module is only reached for a plan that
//!   already passed it.
//! - Whether a copy is legal. Section 34's checks ran at record time.
//! - Batch order and happens-before edges. [`crate::api::submission::backend`] documents why
//!   one native queue supplies all of them for free: the queue executes its lists
//!   in the order they were handed to it, and a batch is handed over before the
//!   next one is recorded.
//!
//! # Why a payload with no lowering is refused rather than skipped
//!
//! The recorder can also hold texture copies, resolves, presentation images, and
//! debug markup which this spine does not lower yet. Skipping one would leave the
//! caller holding a receipt for work that never happened — the silent substitution
//! discipline 3 and section 9.4 forbid in the route case, and which does not
//! become acceptable because the missing lowering is this backend's rather than
//! the platform's.
//!
//! # Performance upgrade map (backend-private)
//!
//! TODO(perf): This is deliberately the correctness-first spine. Its one
//! `Mutex<SpineState>` makes recording, fence serial issuance, submission and
//! retirement one linear transaction, and its `COMMON -> use -> COMMON` policy
//! makes each submitted list self-contained. A future batch-local state-diff
//! encoder may retain final states only while this plan records, then emit only
//! necessary transitions. It must still restore `COMMON` before an independent
//! plan uses the resource, or atomically publish authoritative post-submit state
//! with rollback for every pre-commit failure. `ResourceUse`, `PlanPoint`, and
//! `SubmissionPlan` already carry the portable information; no public API grows.
//!
//! TODO(perf): Submission can narrow this lock to slot/serial reservation and
//! the short Execute+Signal commit, recording into plan-owned slots outside it.
//! The invariant must survive: before the first Execute failure accepts no work;
//! afterwards every accepted batch owns exactly one ordered completion serial and
//! all retirement is keyed to that serial.
//!
//! TODO(perf): Replace per-serial waiter threads with one device-owned fence
//! waiter only when its lost-wakeup protocol registers under lock, arms after
//! registration, re-samples before sleeping, and wakes every waiter on both
//! completion and device loss. `CompletionPoint` is the full public seam.
//!
//! TODO(perf): Multi-queue lowering may map existing `SubmissionLane` and plan
//! dependencies to queue-local fences and waits when workloads prove overlap.
//! It must retain per-resource ordering/state ownership and must not advertise
//! concurrent lanes until those dependencies lower; no new RHI vocabulary is
//! required.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::Waker;

use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
    D3D12_COMMAND_QUEUE_PRIORITY_NORMAL, D3D12_FENCE_FLAG_NONE, ID3D12CommandAllocator,
    ID3D12CommandList, ID3D12CommandQueue, ID3D12Device, ID3D12Fence, ID3D12GraphicsCommandList,
    ID3D12PipelineState,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::PCWSTR;

use crate::api::command::record::{CopyRecord, RecordedPayload};
use crate::api::resource::transfer::ReadbackStatus;
use crate::api::submission::backend::{SubmissionOutcome, SubmissionRequest};
use crate::api::submission::plan::PlanBatch;
use crate::api::submission::{CompletionFailure, CompletionState};
use crate::backend::dx12::ffi;

use super::compute::lower_compute_dispatch;
use super::copy::lower_buffer_copy;
use super::raster::{RasterScopeState, lower_raster_begin, lower_raster_draw, lower_raster_end};
use super::transfer::{
    CommittedBatch, lower_buffer_texture_copy, lower_readback, lower_texture_copy, lower_upload,
    publish_readback,
};
use crate::backend::dx12::failure::{Dx12Failure, ref_native};

/// How long `wait_idle` will block before it reports that the GPU never got
/// there.
///
/// Thirty seconds is not a timeout anyone is meant to hit: it is the bound that
/// keeps `INFINITE` out of a library. A removed device can leave a fence value
/// that will never be signalled, and a blocking wait with no bound would then
/// hang the host inside a shutdown path instead of reporting one. The value is
/// generous on purpose — a long frame on a slow adapter is still a legal wait,
/// and the only thing being ruled out is waiting forever.
const WAIT_BOUND_MS: u32 = 30_000;

/// The Direct3D 12 objects one device submits through.
///
/// One queue, one fence, and a ring of command-list slots. The queue is the
/// device's only lane, which [`crate::backend::dx12::platform::provider`] already reports as
/// `SubmissionCapabilities`: Direct3D 12 exposes a compute queue and up to three
/// copy queues beside the direct one, but several *logical* lanes promise no
/// hardware overlap, so reporting them as lanes would claim a scheduling
/// structure this backend has not established.
pub(crate) struct Dx12CommandSpine {
    /// The one queue every batch is committed to.
    ///
    /// The direct (`DIRECT`) queue type, which is the only one that can execute
    /// every command this backend records. A copy queue would take the transfers
    /// and refuse everything else, and splitting a batch across queues would need
    /// the cross-queue synchronisation section 40.2 makes a plan's dependencies
    /// into work.
    queue: ID3D12CommandQueue,
    /// The fence every submission signals and completion is read from.
    ///
    /// Created at zero, and the first signal is `1`, so serial `0` is a serial no
    /// submission ever issued and is answered as such rather than as "not yet".
    fence: ID3D12Fence,
    /// The device, held so that slots and staging can be created on demand.
    ///
    /// A command list cannot be reset while the GPU is executing it, so the ring
    /// below grows rather than blocks: a device handed more concurrent batches
    /// than it has slots makes another one. Growth is bounded by the deepest
    /// pipeline the caller actually keeps in flight, which is the same quantity a
    /// swapchain's frame count bounds.
    device: ID3D12Device,
    /// Everything mutable, behind one lock.
    ///
    /// One lock rather than several: the fields below are read and written
    /// together — a slot's reuse is decided from the fence and its new deadline
    /// is written from the same submission — and splitting them would let a
    /// reader observe a slot reserved under one fence value and released under
    /// another.
    state: Mutex<SpineState>,
    /// Futures waiting for a fence transition. Kept separate from command state
    /// so a native event thread only needs a small portable waker registry.
    ///
    /// TODO(perf): The serial-keyed registry permits a future shared fence waiter
    /// without altering completion-future semantics; see the module upgrade map.
    completion_waiters: Arc<Mutex<BTreeMap<u64, Vec<Waker>>>>,
}

/// The mutable half of a spine.
struct SpineState {
    /// Command-list slots, in creation order.
    slots: Vec<Slot>,
    /// The highest serial handed to `Signal` so far, or zero before the first
    /// submission.
    ///
    /// A serial is taken for a batch before the batch's list is executed, so a
    /// query for a serial at or below this one is a query about work that has
    /// been committed even when the fence has not reached it yet.
    issued: u64,
    /// The first serial that was executed but could not be signalled.
    ///
    /// Set only by a `Signal` that failed after its `ExecuteCommandLists` had
    /// been called. Section 41.3 forbids reporting that as an `Err` — the work is
    /// on the queue — so it is recorded here and every serial at or beyond it
    /// answers terminally instead of staying `Pending` forever (section 41.8).
    ///
    /// Only the *first* such serial is kept, because it is the bound below which
    /// the fence still answers truthfully: later signals are queued behind the
    /// same broken queue and would each name a larger serial.
    unobservable: Option<(u64, Dx12Failure)>,
    /// Committed batches whose staging must outlive the fence reaching `serial`.
    ///
    /// In serial order, so the drain at the front is the whole of the reclaim
    /// policy: a batch's staging is released exactly when the fence reports that
    /// batch finished, and never earlier.
    pending: VecDeque<CommittedBatch>,
}

impl SpineState {
    /// Claims a slot free to record into, creating one if none is.
    ///
    /// `claimed` names the slots this submission has already taken, so one plan
    /// never records two batches into the same list — a list is closed before the
    /// next batch is recorded, and resetting it again would be a second recording
    /// into a list already on the queue.
    fn claim(
        &mut self,
        device: &ID3D12Device,
        completed: u64,
        claimed: &[usize],
    ) -> Result<usize, ffi::NativeError> {
        let free = (0..self.slots.len()).find(|index| {
            self.slots[*index].in_flight_until <= completed && !claimed.contains(index)
        });
        match free {
            Some(index) => Ok(index),
            None => {
                self.slots.push(Slot::new(device)?);
                Ok(self.slots.len() - 1)
            }
        }
    }
}

/// One command allocator and the command list recorded into it.
///
/// Paired rather than pooled separately because Direct3D 12 ties them: a list may
/// only be reset against an allocator that is itself free to reset, so a free
/// list with a busy allocator is not a reusable slot.
struct Slot {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    /// The serial of the last submission that recorded into this slot. The slot
    /// may be reused once the fence has reached it.
    in_flight_until: u64,
}

impl Slot {
    /// Creates an allocator and the list recorded into it, closed.
    ///
    /// The list is closed immediately, which is the pattern Direct3D 12's own
    /// samples use and not a formality: `CreateCommandList` hands back a list in
    /// the *recording* state, and this spine's first act on a slot is always a
    /// `Reset`. Leaving it open would make every slot's life begin in a state
    /// nothing here expects.
    ///
    /// The initial pipeline state is null, which is the documented way to say
    /// "no pipeline is bound".
    fn new(device: &ID3D12Device) -> Result<Self, ffi::NativeError> {
        // SAFETY: both calls write one interface pointer into the out-parameter
        // the binding owns and convert only on success. The command list type is
        // a plain enum value, the allocator outlives the call, and the null
        // initial state is the documented "none".
        unsafe {
            let allocator = device
                .CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(|error| ffi::NativeError::new(&error, "Dx12Device::submit"))?;
            let list = device
                .CreateCommandList::<_, _, ID3D12GraphicsCommandList>(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &allocator,
                    None::<&ID3D12PipelineState>,
                )
                .map_err(|error| ffi::NativeError::new(&error, "Dx12Device::submit"))?;
            list.Close()
                .map_err(|error| ffi::NativeError::new(&error, "Dx12Device::submit"))?;
            Ok(Self {
                allocator,
                list,
                in_flight_until: 0,
            })
        }
    }
}

/// An event handle that closes itself.
///
/// `wait_idle` is the only thing in this backend that needs one, and it needs it
/// for the duration of one call. A self-closing local is what keeps the handle
/// from being stored on a `Send + Sync` type: `HANDLE` is a raw pointer and
/// storing one would need an `unsafe impl Send`, which is a claim this code has
/// no way to justify.
struct OwnedEvent(HANDLE);

impl Drop for OwnedEvent {
    fn drop(&mut self) {
        // SAFETY: the handle came from `CreateEventW` in `wait_idle` and is owned
        // by this value, so this is the one and only close of it. The result is
        // discarded because a `Drop` cannot report, and a close that fails has no
        // consequence this backend could act on — the handle was already gone.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

impl Dx12CommandSpine {
    pub(crate) fn queue(&self) -> ID3D12CommandQueue {
        self.queue.clone()
    }

    /// Creates the queue and fence behind one device.
    ///
    /// The ring starts empty: slots are made on demand, so a device that never
    /// submits never pays for a command allocator, and a device that keeps a
    /// hundred batches in flight makes exactly as many as it needs.
    pub(crate) fn new(device: &ID3D12Device) -> Result<Self, ffi::NativeError> {
        let description = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            // Normal priority, and `Priority` is an `i32` here rather than the
            // enumerant the header names: Direct3D 12 also accepts any value in
            // `[-1, 100]` on the real-time-capable path, and the newtype carries
            // the enumerant's value.
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            // One node. Linked-node adapters are the multi-GPU feature this
            // backend does not expose, and the mask is how a queue says which
            // nodes may feed it.
            NodeMask: 0,
        };
        // SAFETY: `CreateCommandQueue` reads the descriptor it is given — a local
        // that outlives the call — and writes one interface pointer the binding
        // converts only on success. `CreateFence` takes two by-value arguments and
        // does the same. Neither takes a pointer from this code.
        unsafe {
            let queue = device
                .CreateCommandQueue::<ID3D12CommandQueue>(&description)
                .map_err(|error| ffi::NativeError::new(&error, "Dx12Provider::request_device"))?;
            let fence = device
                .CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE)
                .map_err(|error| ffi::NativeError::new(&error, "Dx12Provider::request_device"))?;
            Ok(Self {
                queue,
                fence,
                device: device.clone(),
                state: Mutex::new(SpineState {
                    slots: Vec::new(),
                    issued: 0,
                    unobservable: None,
                    pending: VecDeque::new(),
                }),
                completion_waiters: Arc::new(Mutex::new(BTreeMap::new())),
            })
        }
    }

    /// Borrows the state, surviving a poisoned lock.
    ///
    /// Recovering rather than propagating, for the reason
    /// [`crate::backend::dx12::platform::provider`]'s liveness cell gives: the guarded value is plain
    /// fields with no invariant a panicking holder could have left half-written.
    /// A panic here would also be reached from `Drop`-adjacent paths where
    /// unwinding is worse than continuing.
    fn lock(&self) -> MutexGuard<'_, SpineState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Lowers a whole plan and commits it.
    ///
    /// # Errors
    ///
    /// Only before the first `ExecuteCommandLists`: a payload with no lowering, a
    /// buffer this device did not allocate, or a driver failure while recording.
    /// Once anything is committed this returns `Ok` and reports trouble through
    /// [`Self::completion`] instead, which is section 41.3's invariant.
    pub(crate) fn submit(
        &self,
        request: &SubmissionRequest<'_>,
    ) -> Result<SubmissionOutcome, Dx12Failure> {
        let mut state = self.lock();
        // Read once: a slot may be reused exactly when the fence has passed the
        // batch last recorded into it.
        //
        // SAFETY: `GetCompletedValue` reads a counter from the fence and takes no
        // argument.
        let completed = unsafe { self.fence.GetCompletedValue() };

        // Phase A. Every batch is recorded into its own list before a single
        // list is executed, so a refusal below leaves the queue untouched — which
        // is what makes "an `Err` from submit proves nothing was accepted"
        // (section 41.3) true rather than merely intended.
        let mut claimed: Vec<usize> = Vec::with_capacity(request.batches.len());
        for _ in request.batches {
            let index = state
                .claim(&self.device, completed, &claimed)
                .map_err(Dx12Failure::Native)?;
            claimed.push(index);
        }

        let first_serial = state.issued + 1;
        let mut recorded: Vec<(ID3D12CommandList, CommittedBatch)> =
            Vec::with_capacity(request.batches.len());
        for (offset, batch) in request.batches.iter().enumerate() {
            let serial = first_serial + offset as u64;
            let slot = &mut state.slots[claimed[offset]];
            // SAFETY: the slot was claimed as free — the fence has passed the
            // batch last recorded into it — so this reset touches neither an
            // allocator nor a list the GPU can still be reading. `Reset` on the
            // list returns it to the recording state it must be in before
            // commands are appended, and the null initial state means no pipeline
            // is bound.
            unsafe {
                slot.allocator.Reset().map_err(|error| ref_native(&error))?;
                slot.list
                    .Reset(&slot.allocator, None::<&ID3D12PipelineState>)
                    .map_err(|error| ref_native(&error))?;
            }

            let mut committed = CommittedBatch {
                serial,
                staging: Vec::new(),
                readbacks: Vec::new(),
                compute_pipelines: Vec::new(),
                raster_pipelines: Vec::new(),
                raster_buffers: Vec::new(),
                raster_views: Vec::new(),
                raster_frames: Vec::new(),
                raster_textures: Vec::new(),
                raster_descriptor_heaps: Vec::new(),
                bind_groups: Vec::new(),
            };
            self.record_batch(&slot.list, batch, &mut committed)?;

            // SAFETY: closing a list in the recording state is always valid and
            // is what makes it executable; the list is not executed until the
            // loop below.
            unsafe { slot.list.Close() }.map_err(|error| ref_native(&error))?;
            slot.in_flight_until = serial;
            recorded.push((ID3D12CommandList::from(slot.list.clone()), committed));
        }

        // Phase B. From the first execute onward this may not fail: section 41.3
        // forbids telling a caller nothing happened once a queue has been fed.
        state.issued = first_serial + request.batches.len() as u64 - 1;
        let mut signals_intact = true;
        for (offset, (list, committed)) in recorded.into_iter().enumerate() {
            let serial = first_serial + offset as u64;
            // SAFETY: the list was closed above and is executed exactly once. The
            // binding copies the slice's pointers into the queue's own array for
            // the duration of the call, and the queue holds its own reference to
            // every list it is given.
            unsafe { self.queue.ExecuteCommandLists(&[Some(list)]) };

            // DXGI Present transfers ownership after the batch its plan point
            // names has entered the queue.  It cannot turn Phase B back into an
            // error: the frame backend records its terminal present state.
            let point = request.batches[offset].point;
            for present in request
                .presents
                .iter()
                .filter(|present| present.after == point)
            {
                present.attachment.present(present.receipt);
            }

            // Signalling after each execute is what makes per-batch completion
            // real: the signal is queued behind *this* list, so the fence
            // reaching `serial` means this batch is done rather than that the
            // whole plan is.
            if signals_intact {
                // SAFETY: `Signal` queues a fence write behind everything already
                // on this queue and takes no pointer from this code.
                match unsafe { self.queue.Signal(&self.fence, serial) } {
                    Ok(()) => {}
                    Err(error) => {
                        // Recorded, not returned. The list above is already on the
                        // queue, so section 41.3's Phase B applies and the caller
                        // must not be told the plan did not run. Further signals
                        // are skipped: they are queued behind the same broken
                        // queue, and a later failure would name a larger serial
                        // than the bound below which the fence still answers.
                        state.unobservable = Some((
                            serial,
                            Dx12Failure::Native(ffi::NativeError::new(
                                &error,
                                "Dx12Device::submit",
                            )),
                        ));
                        signals_intact = false;
                    }
                }
            }

            // Retained whether or not its signal landed: a batch with no
            // observable completion may still be reading its upload staging, and
            // freeing host memory the GPU is copying out of is a use-after-free
            // rather than a leak.
            state.pending.push_back(committed);
        }

        Ok(SubmissionOutcome {
            // The last serial of this plan, signalled or not. Reporting the last
            // one that *was* signalled would claim the whole plan complete while
            // later batches could still be running, which is the one direction
            // section 41.7 forbids.
            completion: state.issued,
            points: request
                .batches
                .iter()
                .enumerate()
                .map(|(offset, batch)| (batch.point, first_serial + offset as u64))
                .collect(),
        })
    }

    /// Records one batch's work into `list`.
    ///
    /// Every payload this spine cannot lower is refused here, before the list is
    /// closed and long before it is executed — which is what keeps section 41.3's
    /// Phase A honest for a plan whose first batch is a copy and whose second is a
    /// draw.
    ///
    /// The `device` each lowering takes is this spine's own rather than a
    /// parameter a caller supplies, which is the one reason this stays a method
    /// while the three lowerings are free functions.
    fn record_batch(
        &self,
        list: &ID3D12GraphicsCommandList,
        batch: &PlanBatch,
        committed: &mut CommittedBatch,
    ) -> Result<(), Dx12Failure> {
        let mut raster = None::<RasterScopeState>;
        for work in &batch.work {
            for command in work.commands() {
                match &command.payload {
                    RecordedPayload::RasterBegin(begin) => {
                        if raster.is_some() {
                            return Err(Dx12Failure::Unsupported {
                                what: "nested raster scopes",
                                why: "the portable recorder never emits them",
                            });
                        }
                        raster = Some(lower_raster_begin(&self.device, list, begin, committed)?);
                    }
                    RecordedPayload::RasterDraw(draw) => {
                        let Some(scope) = raster.as_ref() else {
                            return Err(Dx12Failure::Unsupported {
                                what: "a raster draw outside a raster scope",
                                why: "the portable recorder never emits it",
                            });
                        };
                        lower_raster_draw(list, draw, &command.uses, scope, committed)?;
                    }
                    RecordedPayload::RasterEnd => {
                        let Some(scope) = raster.take() else {
                            return Err(Dx12Failure::Unsupported {
                                what: "a raster-scope end without a scope",
                                why: "the portable recorder never emits it",
                            });
                        };
                        lower_raster_end(list, scope, committed);
                    }
                    RecordedPayload::Copy(CopyRecord::Buffer(copy)) => {
                        lower_buffer_copy(list, copy)?;
                    }
                    RecordedPayload::Copy(CopyRecord::Texture(copy)) => {
                        lower_texture_copy(list, copy)?;
                    }
                    RecordedPayload::Copy(CopyRecord::BufferToTexture(copy)) => {
                        lower_buffer_texture_copy(&self.device, list, copy, true)?;
                    }
                    RecordedPayload::Copy(CopyRecord::TextureToBuffer(copy)) => {
                        lower_buffer_texture_copy(&self.device, list, copy, false)?;
                    }
                    RecordedPayload::Upload(job) => {
                        lower_upload(&self.device, list, job, committed)?;
                    }
                    RecordedPayload::Readback(ticket) => {
                        lower_readback(&self.device, list, ticket, committed)?;
                    }
                    RecordedPayload::ComputeBegin(_) | RecordedPayload::ComputeEnd => {}
                    RecordedPayload::ComputeDispatch(dispatch) => {
                        lower_compute_dispatch(list, dispatch, &command.uses, committed)?;
                    }
                    other => {
                        return Err(Dx12Failure::Unsupported {
                            what: payload_name(other),
                            why: NOT_LOWERED,
                        });
                    }
                }
            }
        }
        if raster.is_some() {
            return Err(Dx12Failure::Unsupported {
                what: "an unterminated raster scope",
                why: "the portable recorder never emits it",
            });
        }
        Ok(())
    }

    /// Reports one serial's state, without blocking.
    ///
    /// Section 41.10 forbids a wait here, so this is a read of the fence's own
    /// counter plus this spine's own bookkeeping, and nothing else.
    ///
    /// # Why this query also drains
    ///
    /// Answering and publishing share one fence reading, so the answer cannot
    /// arrive ahead of the bytes it is the completion of. The ordering matters
    /// and a real-GPU run is what exposed it: a caller's idiom is "poll, ask
    /// section 41.7's question, then read the ticket", and if only `poll` drained,
    /// the fence could advance between `poll`'s read and this one. The caller would
    /// then observe this point `Complete` while the readback ticket under it was
    /// still `Pending` — and section 41.2's whole reason for a per-batch point is
    /// that a readback must not be forced to await the slowest unrelated batch, so
    /// a point that reaches `Complete` without publishing the ticket it was minted
    /// for would make the point in `completion_for` useless as a readiness signal.
    /// Draining with the very value the answer is computed from is what makes
    /// "this call reported the point complete" imply "the ticket under it is
    /// `Ready`".
    ///
    /// The rule itself lives in one place, `drain`; both verbs call it, which is
    /// section 65.3's one-authority requirement applied to "what a reached fence
    /// value implies" rather than duplicated across the two entry points.
    pub(crate) fn completion(&self, serial: u64) -> CompletionState {
        let mut state = self.lock();
        // SAFETY: `GetCompletedValue` reads a counter from the fence and takes no
        // argument.
        let reached = unsafe { self.fence.GetCompletedValue() };
        drain(&mut state, reached);

        // Serial zero is the "nothing has been submitted" identity: serials start
        // at one, so it is reachable only from the receipt of a plan that carried
        // no batches, and for that plan "everything submitted so far" really is
        // nothing. Answering `Complete` is what keeps an empty plan's receipt
        // pollable instead of sending a caller looking for a failure.
        if serial == 0 {
            return CompletionState::Complete;
        }

        // Checked before the fence, because a serial at or beyond this bound can
        // never be observed: the fence value it names was never written, so
        // asking the fence would answer `Pending` forever (section 41.8), and
        // asking it about a *later* serial would answer about work that is
        // unrelated to this one.
        if let Some((bound, failure)) = state.unobservable.as_ref() {
            if serial >= *bound {
                return CompletionState::Failed(CompletionFailure::new(failure.message()));
            }
        }

        if serial <= reached {
            CompletionState::Complete
        } else {
            CompletionState::Pending
        }
    }

    /// Samples one serial and subscribes a runtime waker if it remains pending.
    ///
    /// Direct3D 12 supplies completion as a fence event, whereas Rust futures
    /// supply a `Waker`. A tiny OS thread bridges exactly those two mechanisms:
    /// it sleeps in `WaitForSingleObject`, then wakes the executor; it never
    /// polls the fence and never assumes a particular async runtime.
    pub(crate) fn completion_or_register_waker(
        &self,
        serial: u64,
        waker: &Waker,
    ) -> CompletionState {
        let state = self.completion(serial);
        if !matches!(state, CompletionState::Pending) {
            return state;
        }

        let spawn_waiter = {
            let mut waiters = self
                .completion_waiters
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let entry = waiters.entry(serial).or_default();
            let first = entry.is_empty();
            entry.push(waker.clone());
            first
        };

        if spawn_waiter {
            let fence = self.fence.clone();
            let waiters = Arc::clone(&self.completion_waiters);
            std::thread::spawn(move || {
                // A fence event is one-shot. The bounded wait also guarantees a
                // device removal cannot strand an OS thread forever; the wake at
                // the bound causes a fresh state sample and, if still live,
                // registration for a new native event.
                let event = unsafe { CreateEventW(None, false, false, PCWSTR::null()) };
                if let Ok(event) = event {
                    let event = OwnedEvent(event);
                    if unsafe { fence.SetEventOnCompletion(serial, event.0) }.is_ok() {
                        let _ = unsafe { WaitForSingleObject(event.0, WAIT_BOUND_MS) };
                    }
                }
                wake_serial(&waiters, serial);
            });
        }
        CompletionState::Pending
    }

    /// Wakes all registered futures after device loss. Their next poll is what
    /// observes `DeviceLost` (or preserves an already-complete serial).
    pub(crate) fn wake_completion_waiters(&self) {
        let mut waiters = self
            .completion_waiters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let pending = core::mem::take(&mut *waiters);
        drop(waiters);
        for (_, wakers) in pending {
            for waker in wakers {
                waker.wake();
            }
        }
    }

    /// Terminates every readback retained by work whose fence can no longer
    /// advance, then wakes completion futures. Retained staging stays alive until
    /// device teardown: loss gives no proof that native DMA has stopped.
    pub(crate) fn terminate_pending_for_device_loss(&self) {
        let state = self.lock();
        for batch in &state.pending {
            for retention in &batch.readbacks {
                retention.ticket.set_status(ReadbackStatus::DeviceLost);
            }
        }
        drop(state);
        self.wake_completion_waiters();
    }

    /// Publishes whatever the fence has reported finished.
    ///
    /// Called from the device's `poll`, which is the portable layer's only
    /// progress verb, and from `wait_idle` after its wait. Both callers are
    /// non-blocking here, and this holds the lock for the whole drain so a
    /// concurrent `submit` cannot see a half-drained queue.
    pub(crate) fn advance(&self) {
        let mut state = self.lock();
        // SAFETY: `GetCompletedValue` reads a counter from the fence and takes no
        // argument.
        let reached = unsafe { self.fence.GetCompletedValue() };
        drain(&mut state, reached);
    }

    /// Blocks until every submitted batch has finished, or the bound expires.
    ///
    /// Section 6.7 confines this to shutdown, recovery, and diagnostics, which is
    /// why nothing in the frame path calls it. The wait is on an event rather
    /// than a spin over `GetCompletedValue`, because a spin would burn a core for
    /// the whole wait — a cost a shutdown path should not impose on the host.
    pub(crate) fn wait_idle(&self) -> Result<(), Dx12Failure> {
        let issued = self.lock().issued;
        if issued == 0 {
            // Nothing has ever been submitted, so there is no fence value to
            // reach and no event to wait on.
            return Ok(());
        }

        // SAFETY: `CreateEventW` returns an owned handle or an error, and the
        // binding reports the invalid value as an error rather than handing it
        // back. The auto-reset, initially-unsignalled event with no name and no
        // security attributes is the documented shape for a one-shot wait.
        let event =
            unsafe { CreateEventW(None, false, false, PCWSTR::null()) }.map_err(|error| {
                Dx12Failure::Native(ffi::NativeError::new(&error, "Device::wait_idle"))
            })?;
        let event = OwnedEvent(event);

        // SAFETY: `SetEventOnCompletion` records the handle on the fence's wait
        // list and signals it when the fence reaches `issued`; it returns an error
        // without recording anything if it cannot. The handle stays alive in
        // `event` for the whole wait below.
        unsafe { self.fence.SetEventOnCompletion(issued, event.0) }.map_err(|error| {
            Dx12Failure::Native(ffi::NativeError::new(&error, "Device::wait_idle"))
        })?;

        // SAFETY: `WaitForSingleObject` blocks this thread on a handle this
        // function owns. The bound is what keeps a removed device — whose fence
        // value will never be written — from hanging the host forever.
        let waited = unsafe { WaitForSingleObject(event.0, WAIT_BOUND_MS) };
        if waited != WAIT_OBJECT_0 {
            return Err(Dx12Failure::Stalled {
                bound_ms: WAIT_BOUND_MS,
            });
        }

        // The wait proves the fence moved, so the drain has something to publish.
        self.advance();
        Ok(())
    }
}

/// Removes and wakes the futures waiting for one fence value.
fn wake_serial(waiters: &Mutex<BTreeMap<u64, Vec<Waker>>>, serial: u64) {
    let registered = waiters
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(&serial)
        .unwrap_or_default();
    for waker in registered {
        waker.wake();
    }
}

/// Publishes everything a reached fence value implies, and terminates the rest.
///
/// The one body behind both [`Dx12CommandSpine::advance`] and
/// [`Dx12CommandSpine::completion`]. What a fence value implies — which batches'
/// staging may now be freed, which readback tickets may be published, which
/// unobservable serials have become terminal — is a single rule, and two copies
/// of it would be two authorities for one rule, which section 65.3 forbids.
///
/// `reached` is a parameter rather than a fresh fence read so that the caller
/// answering a *question* can drain with the very value it answers from. That is
/// what makes "the point is complete" imply "the ticket under it is `Ready`",
/// and it is the reason this is not folded into `advance` alone.
///
/// The caller must hold the lock, and the drain runs to completion under it, so a
/// concurrent `submit` cannot observe a half-drained queue.
fn drain(state: &mut SpineState, reached: u64) {
    loop {
        let Some(front) = state.pending.front() else {
            break;
        };
        if front.serial > reached {
            break;
        }
        let Some(finished) = state.pending.pop_front() else {
            break;
        };
        // The upload staging drops with `finished`, and this is the only moment it
        // may: the fence has reported that the batch reading it finished. The
        // readback staging is dropped the same way, after its bytes have been
        // copied out.
        for retention in finished.readbacks {
            publish_readback(&retention);
        }
    }

    // Serials the fence can never reach still have to terminate (section 41.8),
    // and their tickets are the one place this spine can say so: the device's own
    // completion answer is the provider's, and a ticket is queried through itself.
    // Their staging is deliberately *not* released — work that was executed and
    // never observed may still be reading it, and freeing host memory the GPU is
    // writing into would turn an unobservable submission into memory corruption.
    let terminal = state.unobservable.as_ref().map(|(bound, failure)| {
        (
            *bound,
            if failure.is_terminal() {
                ReadbackStatus::DeviceLost
            } else {
                ReadbackStatus::Failed
            },
        )
    });
    if let Some((bound, status)) = terminal {
        for batch in &state.pending {
            if batch.serial >= bound {
                for retention in &batch.readbacks {
                    retention.ticket.set_status(status);
                }
            }
        }
    }
}

/// The reason every unlifted payload reports.
const NOT_LOWERED: &str = "this Direct3D 12 spine lowers buffer copies, buffer uploads, and \
                           buffer readbacks; everything else the recorder can hold is not \
                           lowered yet, and section 9.4 forbids executing a plan while \
                           silently dropping part of it";

/// What an unlifted payload asked for, for the refusal's first clause.
///
/// Exhaustive rather than a catch-all, so adding a payload to
/// [`RecordedPayload`] is a compile error here instead of a refusal that names
/// the wrong thing.
fn payload_name(payload: &RecordedPayload) -> &'static str {
    match payload {
        RecordedPayload::RasterBegin(_) => "a raster scope",
        RecordedPayload::RasterDraw(_) => "a draw",
        RecordedPayload::RasterEnd => "the end of a raster scope",
        RecordedPayload::ComputeBegin(_) => "a compute scope",
        RecordedPayload::ComputeDispatch(_) => "a dispatch",
        RecordedPayload::ComputeEnd => "the end of a compute scope",
        RecordedPayload::Copy(_) => "a copy this spine has no lowering for",
        RecordedPayload::Upload(_) => "an upload",
        RecordedPayload::Readback(_) => "a readback",
        RecordedPayload::DebugPush(_) => "a debug group",
        RecordedPayload::DebugPop => "the end of a debug group",
        RecordedPayload::DebugMarker(_) => "a debug marker",
    }
}
