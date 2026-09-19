//! The Direct3D 12 command spine: recording a batch, committing it, and
//! observing it finish.
//!
//! This module owns the four native objects Direct3D 12 requires before any work
//! can run — a command queue, command allocators, command lists, and a fence —
//! and the lowering of the three payloads that can currently be recorded onto
//! them. It decides nothing about legality: every plan reaching it has passed
//! section 40.5's checklist in the portable layer, so a refusal produced here is
//! one only Direct3D 12 can know.
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
//! ([`super::resource`]), and each command transitions what it uses out of
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
//! - Batch order and happens-before edges. [`crate::base::command`] documents why
//!   one native queue supplies all of them for free: the queue executes its lists
//!   in the order they were handed to it, and a batch is handed over before the
//!   next one is recorded.
//!
//! # Why a payload with no lowering is refused rather than skipped
//!
//! The recorder can hold raster scopes, dispatches, texture copies, and debug
//! markup, and this spine lowers none of them yet. Skipping one would leave the
//! caller holding a receipt for a plan whose draw never happened — the silent
//! substitution discipline 3 and section 9.4 forbid in the route case, and which
//! does not become acceptable because the missing lowering is this backend's
//! rather than the platform's.

use std::collections::VecDeque;
use std::sync::{Mutex, MutexGuard};

use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
    D3D12_COMMAND_QUEUE_PRIORITY_NORMAL, D3D12_FENCE_FLAG_NONE, D3D12_RESOURCE_BARRIER,
    D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATES, D3D12_RESOURCE_TRANSITION_BARRIER, ID3D12CommandAllocator,
    ID3D12CommandList, ID3D12CommandQueue, ID3D12Device, ID3D12Fence, ID3D12GraphicsCommandList,
    ID3D12PipelineState, ID3D12Resource,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::PCWSTR;

use super::ffi;
use super::resource::{Dx12Buffer, StagingHeap, create_staging};
use crate::api::command::copy::BufferCopy;
use crate::api::command::record::{CopyRecord, RecordedPayload};
use crate::api::error::{RhiError, RhiErrorKind};
use crate::api::resource::buffer::Buffer;
use crate::api::resource::transfer::{
    ReadbackRequest, ReadbackStatus, ReadbackTicket, UploadDescriptor, UploadJob,
};
use crate::api::submission::plan::PlanBatch;
use crate::api::submission::{CompletionFailure, CompletionState};
use crate::base::command::{SubmissionOutcome, SubmissionRequest};

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

/// Why the spine could not lower or observe a plan.
///
/// Three variants, mirroring [`ffi::NativeFailure`]'s split: one is a fact about
/// what this backend has built, one is a fact about how long the GPU took, and
/// one is a fact about the driver. Only the third can end the device, which is
/// why the distinction survives to this type rather than being flattened into an
/// [`RhiError`] here.
pub(super) enum SpineFailure {
    /// The recording names something this spine has no lowering for.
    ///
    /// Reported as [`RhiErrorKind::Unsupported`] rather than as a silent skip:
    /// section 9.4 forbids substituting a path for one that does not exist, and
    /// a batch whose raster work was quietly dropped would execute as a
    /// copy-only plan while the caller believed it had drawn something.
    Unsupported {
        /// What the recording asked for, for the refusal's first clause.
        what: &'static str,
        /// Why this spine does not lower it.
        why: &'static str,
    },
    /// The GPU did not reach the last submitted serial inside the bound.
    ///
    /// Only [`Dx12CommandSpine::wait_idle`] produces this. It is deliberately
    /// not terminal: a GPU that is merely slow and a GPU that has hung are
    /// indistinguishable from here, and [`ffi::NativeFailure`] already records
    /// why treating a hung device as alive is the cheaper of the two mistakes.
    Stalled {
        /// The bound that expired, so the message states what was waited for.
        bound_ms: u32,
    },
    /// A Direct3D 12 call failed.
    Native(ffi::NativeError),
}

impl SpineFailure {
    /// Whether this failure ended the device.
    ///
    /// Neither `Unsupported` nor `Stalled` does. This backend not having built a
    /// lowering says nothing about the driver, and a slow frame is not a dead
    /// device; marking a healthy device lost on either would retire a usable
    /// device on a transient fact.
    pub(super) fn is_terminal(&self) -> bool {
        match self {
            Self::Unsupported { .. } | Self::Stalled { .. } => false,
            Self::Native(native) => native.failure().is_terminal(),
        }
    }

    /// The sentence this failure reports, without its operation tag.
    ///
    /// Read by [`super::provider`], which builds a [`DeviceLossInfo`] summary
    /// from it and renders it into the [`CompletionFailure`] a permanently
    /// unobservable serial answers with.
    ///
    /// [`DeviceLossInfo`]: crate::api::platform::DeviceLossInfo
    pub(super) fn message(&self) -> String {
        match self {
            Self::Unsupported { what, why } => format!("{what}: {why}"),
            Self::Stalled { bound_ms } => {
                format!("the GPU did not reach the last submitted serial within {bound_ms} ms")
            }
            Self::Native(native) => native.as_error().to_string(),
        }
    }

    /// Converts into the portable error a caller sees.
    pub(super) fn into_rhi(self) -> RhiError {
        match self {
            Self::Unsupported { what, why } => {
                RhiError::new(RhiErrorKind::Unsupported, format!("{what}: {why}"))
            }
            Self::Stalled { bound_ms } => RhiError::new(
                RhiErrorKind::BackendFailure,
                format!("the GPU did not reach the last submitted serial within {bound_ms} ms"),
            ),
            Self::Native(native) => return native.into_rhi(),
        }
        .at("Dx12Device::submit")
    }
}

/// The Direct3D 12 objects one device submits through.
///
/// One queue, one fence, and a ring of command-list slots. The queue is the
/// device's only lane, which [`super::provider`] already reports as
/// `SubmissionCapabilities`: Direct3D 12 exposes a compute queue and up to three
/// copy queues beside the direct one, but several *logical* lanes promise no
/// hardware overlap, so reporting them as lanes would claim a scheduling
/// structure this backend has not established.
pub(super) struct Dx12CommandSpine {
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
    unobservable: Option<(u64, SpineFailure)>,
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

/// A batch that has been committed, and the host-visible memory its command list
/// reads or writes.
///
/// Both halves are needed for the same reason: `ExecuteCommandLists` is
/// asynchronous, so a staging allocation whose last reference is dropped when
/// recording returns would be freed while the GPU is still copying out of it.
/// Retaining it here until the fence reports the batch finished is what keeps the
/// list's resource references valid for the list's whole life.
struct CommittedBatch {
    /// The serial that reports this batch's completion.
    serial: u64,
    /// Upload staging: written by the CPU before the commit, read by the GPU
    /// during it, and of no use afterwards.
    staging: Vec<Dx12Buffer>,
    /// Readback staging: written by the GPU, read by the CPU once the serial is
    /// reached, and then published to the ticket that asked for it.
    readbacks: Vec<ReadbackRetention>,
}

/// A readback's staging buffer and the ticket waiting on it.
struct ReadbackRetention {
    /// The `READBACK` heap allocation the GPU copies into.
    staging: Dx12Buffer,
    /// The ticket whose bytes these are.
    ticket: ReadbackTicket,
    /// How many bytes were copied, which is the range's size rather than the
    /// buffer's.
    size: u64,
}

/// The transition barriers one step of a recording needs, released together.
///
/// # Why this is a type and not four lines at each call site
///
/// `D3D12_RESOURCE_BARRIER`'s `pResource` is a `ManuallyDrop`, so a barrier built
/// the obvious way takes one reference to the resource and dropping the barrier
/// does **not** release it. `ResourceBarrier` copies the struct into the command
/// stream rather than taking ownership of it, so a caller that lets the barrier
/// fall out of scope leaks exactly one reference per barrier — a leak that grows
/// with every frame and never shows up as an error.
///
/// Reclaiming it is a `ManuallyDrop::drop` on a union member, which is `unsafe`
/// and easy to get wrong in the direction of a double release. Putting it in a
/// `Drop` impl means it is written once, with one rationale, and cannot be
/// forgotten at a call site. Note that `D3D12_RESOURCE_BARRIER`'s own `Clone` is
/// a `transmute_copy` that does *not* add a reference, so a clone of a barrier is
/// a second owner of one reference — this type never clones one.
#[derive(Default)]
struct Transitions {
    barriers: Vec<D3D12_RESOURCE_BARRIER>,
}

impl Transitions {
    /// Adds one transition from `before` to `after` on `resource`.
    fn push(
        &mut self,
        resource: &ID3D12Resource,
        before: D3D12_RESOURCE_STATES,
        after: D3D12_RESOURCE_STATES,
    ) {
        self.barriers.push(D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    // The one reference this barrier owns, released in `Drop`.
                    pResource: std::mem::ManuallyDrop::new(Some(resource.clone())),
                    // Every resource this backend copies through is a buffer, and
                    // a buffer has one subresource. The constant is Direct3D 12's
                    // own "all of them", which is the honest value for a resource
                    // that has exactly one.
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        });
    }

    /// Appends these transitions to `list` in the order they were pushed.
    ///
    /// Takes `&self` rather than consuming, which is what lets the caller keep
    /// the value alive until it drops and releases the references above.
    fn record(&self, list: &ID3D12GraphicsCommandList) {
        if self.barriers.is_empty() {
            return;
        }
        // SAFETY: `ResourceBarrier` reads the slice it is given and copies each
        // barrier into the command stream; the vector outlives the call, and the
        // resources it names are kept alive by the caller for as long as the
        // recorded list can execute.
        unsafe { list.ResourceBarrier(&self.barriers) };
    }
}

impl Drop for Transitions {
    fn drop(&mut self) {
        for barrier in &mut self.barriers {
            // SAFETY: every barrier in this vector was built by `push` above,
            // which is the only constructor, so `Transition` is the variant that
            // is live and reading it is not reading an inactive union member.
            // `pResource` is a `ManuallyDrop` because the generated struct has no
            // `Drop` of its own; releasing it here is the one release of the one
            // reference `push` took, and nothing else touches this vector.
            unsafe {
                let transition = &mut barrier.Anonymous.Transition;
                std::mem::ManuallyDrop::drop(&mut transition.pResource);
            }
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
    /// Creates the queue and fence behind one device.
    ///
    /// The ring starts empty: slots are made on demand, so a device that never
    /// submits never pays for a command allocator, and a device that keeps a
    /// hundred batches in flight makes exactly as many as it needs.
    pub(super) fn new(device: &ID3D12Device) -> Result<Self, ffi::NativeError> {
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
            })
        }
    }

    /// Borrows the state, surviving a poisoned lock.
    ///
    /// Recovering rather than propagating, for the reason
    /// [`super::provider`]'s liveness cell gives: the guarded value is plain
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
    pub(super) fn submit(
        &self,
        request: &SubmissionRequest<'_>,
    ) -> Result<SubmissionOutcome, SpineFailure> {
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
                .map_err(SpineFailure::Native)?;
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
                            SpineFailure::Native(ffi::NativeError::new(
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
    fn record_batch(
        &self,
        list: &ID3D12GraphicsCommandList,
        batch: &PlanBatch,
        committed: &mut CommittedBatch,
    ) -> Result<(), SpineFailure> {
        for work in &batch.work {
            for command in work.commands() {
                match &command.payload {
                    RecordedPayload::Copy(CopyRecord::Buffer(copy)) => {
                        self.lower_buffer_copy(list, copy)?;
                    }
                    RecordedPayload::Upload(job) => {
                        self.lower_upload(list, job, committed)?;
                    }
                    RecordedPayload::Readback(ticket) => {
                        self.lower_readback(list, ticket, committed)?;
                    }
                    other => {
                        return Err(SpineFailure::Unsupported {
                            what: payload_name(other),
                            why: NOT_LOWERED,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Lowers a buffer-to-buffer copy.
    ///
    /// Both resources are named in a barrier out of `COMMON` and back into it,
    /// which is the invariant the module documentation states. `CopyBufferRegion`
    /// itself does not transition anything: Direct3D 12 requires a copy's source
    /// to be in `COPY_SOURCE` and its destination in `COPY_DEST` when the list
    /// executes, and the two barriers are how that becomes true.
    fn lower_buffer_copy(
        &self,
        list: &ID3D12GraphicsCommandList,
        copy: &BufferCopy,
    ) -> Result<(), SpineFailure> {
        let source = dx12_buffer(&copy.src)?;
        let destination = dx12_buffer(&copy.dst)?;

        let mut entering = Transitions::default();
        entering.push(
            source.resource(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        entering.push(
            destination.resource(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        entering.record(list);

        // SAFETY: both resources are alive for at least as long as this call, the
        // two offsets and the size were validated against them at record time
        // (section 34), and the barriers immediately above and below put each
        // resource in the state the copy requires.
        unsafe {
            list.CopyBufferRegion(
                destination.resource(),
                copy.dst_offset,
                source.resource(),
                copy.src_offset,
                copy.size,
            );
        }

        let mut leaving = Transitions::default();
        leaving.push(
            source.resource(),
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        );
        leaving.push(
            destination.resource(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COMMON,
        );
        leaving.record(list);
        Ok(())
    }

    /// Lowers a buffer upload: staging copy from caller bytes, then GPU copy.
    ///
    /// # Why the CPU write happens here and not at `create_buffer_upload`
    ///
    /// The upload heap allocation could have been made and filled when the job
    /// was created, since the bytes are already retained and immutable
    /// (section 17.2). It is made here instead because the allocation must
    /// outlive the *commit*, not the job: a staging buffer created at job
    /// creation would be held by the job, which the caller may keep for as long
    /// as it likes, and a job encoded into several plans would need one staging
    /// buffer per plan anyway. Creating it inside the recording ties its life to
    /// the batch that reads it, which is exactly the lifetime the fence reports.
    fn lower_upload(
        &self,
        list: &ID3D12GraphicsCommandList,
        job: &UploadJob,
        committed: &mut CommittedBatch,
    ) -> Result<(), SpineFailure> {
        let UploadDescriptor::Buffer(descriptor) = job.descriptor() else {
            return Err(SpineFailure::Unsupported {
                what: "a texture upload",
                why: "this spine has no texture lowering at all, so there is no \
                      destination state, no region copy, and no host-layout repacking \
                      to write through",
            });
        };
        let destination = dx12_buffer(&descriptor.dst)?;

        let length = descriptor.bytes.len();
        let staging = create_staging(&self.device, length as u64, StagingHeap::Upload)
            .map_err(SpineFailure::Native)?;

        let mut pointer: *mut core::ffi::c_void = std::ptr::null_mut();
        // SAFETY: `Map` on an `UPLOAD` heap resource makes the whole allocation
        // CPU-writable and writes the address into `pointer`; the null read range
        // is what Direct3D 12 requires for a write-only heap. The mapping stays
        // live until the `Unmap` below.
        unsafe {
            staging
                .resource()
                .Map(0, None, Some(&mut pointer))
                .map_err(|error| ref_native(&error))?;
        }
        let Some(pointer) = std::ptr::NonNull::new(pointer.cast::<u8>()) else {
            return Err(SpineFailure::Native(
                ffi::NativeError::driver_contract_violation(
                    "Map reported success without producing a pointer",
                    "Dx12Device::submit",
                ),
            ));
        };
        // SAFETY: the mapping covers `length` bytes because that is the resource's
        // own width, which is what `create_staging` was asked for. The source is
        // the job's retained `Arc<[u8]>`, alive for the whole recording, and host
        // memory and a GPU allocation cannot overlap. `Unmap` follows the copy and
        // is the single matching call for the single mapping above.
        unsafe {
            std::ptr::copy_nonoverlapping(descriptor.bytes.as_ptr(), pointer.as_ptr(), length);
            staging.resource().Unmap(0, None);
        }

        let mut entering = Transitions::default();
        entering.push(
            destination.resource(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        entering.record(list);
        // SAFETY: the staging buffer was created in `GENERIC_READ` and stays
        // there — Direct3D 12 permits no transition in an upload heap — which is
        // a state the copy's source half may be in. The destination was put in
        // `COPY_DEST` by the barrier above, and `dst_offset` plus `length` was
        // validated against the destination's size at job creation (section 17.3).
        unsafe {
            list.CopyBufferRegion(
                destination.resource(),
                descriptor.dst_offset,
                staging.resource(),
                0,
                length as u64,
            );
        }
        let mut leaving = Transitions::default();
        leaving.push(
            destination.resource(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COMMON,
        );
        leaving.record(list);

        committed.staging.push(staging);
        Ok(())
    }

    /// Lowers a buffer readback: GPU copy into staging, then a ticket the drain
    /// publishes from.
    ///
    /// The reverse of [`Self::lower_upload`] in every respect, including which
    /// way the staging is retained: upload staging is dead the moment the batch
    /// finishes, while readback staging is the thing the batch's completion is
    /// *for*.
    fn lower_readback(
        &self,
        list: &ID3D12GraphicsCommandList,
        ticket: &ReadbackTicket,
        committed: &mut CommittedBatch,
    ) -> Result<(), SpineFailure> {
        let ReadbackRequest::Buffer { src, range, .. } = ticket.request() else {
            return Err(SpineFailure::Unsupported {
                what: "a texture readback",
                why: "this spine has no texture lowering at all, so there is no source \
                      state and no footprint to copy through",
            });
        };
        let source = dx12_buffer(src)?;

        let staging = create_staging(&self.device, range.size, StagingHeap::Readback)
            .map_err(SpineFailure::Native)?;

        let mut entering = Transitions::default();
        entering.push(
            source.resource(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        entering.record(list);
        // SAFETY: the staging buffer was created in `COPY_DEST` and stays there —
        // Direct3D 12 permits no transition in a readback heap — which is a state
        // the copy's destination half may be in. The source was put in
        // `COPY_SOURCE` by the barrier above, and the range was validated against
        // the source's size at record time (section 18.1).
        unsafe {
            list.CopyBufferRegion(
                staging.resource(),
                0,
                source.resource(),
                range.offset,
                range.size,
            );
        }
        let mut leaving = Transitions::default();
        leaving.push(
            source.resource(),
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        );
        leaving.record(list);

        committed.readbacks.push(ReadbackRetention {
            staging,
            ticket: ticket.clone(),
            size: range.size,
        });
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
    pub(super) fn completion(&self, serial: u64) -> CompletionState {
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

    /// Publishes whatever the fence has reported finished.
    ///
    /// Called from the device's `poll`, which is the portable layer's only
    /// progress verb, and from `wait_idle` after its wait. Both callers are
    /// non-blocking here, and this holds the lock for the whole drain so a
    /// concurrent `submit` cannot see a half-drained queue.
    pub(super) fn advance(&self) {
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
    pub(super) fn wait_idle(&self) -> Result<(), SpineFailure> {
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
                SpineFailure::Native(ffi::NativeError::new(&error, "Device::wait_idle"))
            })?;
        let event = OwnedEvent(event);

        // SAFETY: `SetEventOnCompletion` records the handle on the fence's wait
        // list and signals it when the fence reaches `issued`; it returns an error
        // without recording anything if it cannot. The handle stays alive in
        // `event` for the whole wait below.
        unsafe { self.fence.SetEventOnCompletion(issued, event.0) }.map_err(|error| {
            SpineFailure::Native(ffi::NativeError::new(&error, "Device::wait_idle"))
        })?;

        // SAFETY: `WaitForSingleObject` blocks this thread on a handle this
        // function owns. The bound is what keeps a removed device — whose fence
        // value will never be written — from hanging the host forever.
        let waited = unsafe { WaitForSingleObject(event.0, WAIT_BOUND_MS) };
        if waited != WAIT_OBJECT_0 {
            return Err(SpineFailure::Stalled {
                bound_ms: WAIT_BOUND_MS,
            });
        }

        // The wait proves the fence moved, so the drain has something to publish.
        self.advance();
        Ok(())
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

/// The native allocation behind a portable buffer.
///
/// # Errors
///
/// `Unsupported` when the buffer's backend is not this one. That is unreachable
/// for a plan this device accepted — the recorder compares device identity at
/// every encode, and section 3.3 makes identity the only answer to a cross-device
/// use — so this is a total function over an empty case rather than a reachable
/// refusal. It returns an error rather than unwrapping because the alternative to
/// a type-checked downcast is a panic in a library, and because a message naming
/// what was expected is worth more to whoever reaches it than an abort.
fn dx12_buffer(buffer: &Buffer) -> Result<&Dx12Buffer, SpineFailure> {
    buffer
        .native()
        .as_any()
        .downcast_ref::<Dx12Buffer>()
        .ok_or(SpineFailure::Unsupported {
            what: "a buffer this device did not allocate",
            why: "its native allocation belongs to another backend, and section 3.3 makes \
                  that a refusal rather than a migration",
        })
}

/// Copies one finished readback's bytes off the GPU and hands them to its ticket.
///
/// A failure here is reported as [`ReadbackStatus::Failed`] rather than as a
/// device loss: mapping a readback heap can fail for reasons that say nothing
/// about the device, and section 18.2 makes `Failed` exactly the terminal state
/// for a backend failure. The ticket carries no message, so the state is the
/// whole report.
fn publish_readback(retention: &ReadbackRetention) {
    // Widened before the mapping, so a range too large for this process's address
    // space fails the ticket instead of truncating to a plausible length.
    let Ok(length) = usize::try_from(retention.size) else {
        retention.ticket.set_status(ReadbackStatus::Failed);
        return;
    };

    let mut pointer: *mut core::ffi::c_void = std::ptr::null_mut();
    // SAFETY: `Map` on a `READBACK` heap resource makes the whole allocation
    // CPU-readable and writes the address into `pointer`; the null read range is
    // what Direct3D 12 requires for a read-only heap. The mapping stays live until
    // the `Unmap` below.
    unsafe {
        if retention
            .staging
            .resource()
            .Map(0, None, Some(&mut pointer))
            .is_err()
        {
            retention.ticket.set_status(ReadbackStatus::Failed);
            return;
        }
    }
    let Some(pointer) = std::ptr::NonNull::new(pointer.cast::<u8>()) else {
        // SAFETY: the mapping above is live and this is its one matching unmap.
        unsafe { retention.staging.resource().Unmap(0, None) };
        retention.ticket.set_status(ReadbackStatus::Failed);
        return;
    };

    // SAFETY: the mapping covers `length` bytes because that is the staging
    // resource's own width. The bytes are copied out before the unmap, which is
    // the single matching call for the single mapping above, and the ticket takes
    // ownership of the copy — so nothing borrows the mapping after it ends.
    let bytes = unsafe {
        let bytes = std::slice::from_raw_parts(pointer.as_ptr(), length).to_vec();
        retention.staging.resource().Unmap(0, None);
        bytes
    };
    // A buffer range is tightly packed by definition (section 18.3), which is why
    // there is no layout to report.
    retention.ticket.publish(bytes, None);
}

/// Builds a [`ffi::NativeError`] naming [`Dx12CommandSpine::submit`].
///
/// A free function rather than a closure at each `map_err`, because the operation
/// tag must be the same string at every one of them and a closure would have to
/// be re-typed to stay identical.
fn ref_native(error: &windows::core::Error) -> SpineFailure {
    SpineFailure::Native(ffi::NativeError::new(error, "Dx12Device::submit"))
}
