//! The common execution contract over a GL-family state machine.
//!
//! This is Layer 3's whole visible surface: one type that owns a Layer 2
//! machine and answers `fluxel_rendergraph::ExecutionBackend`, plus the three
//! things that contract needs and a GL-family context does not have.  Each of
//! the three is a module of its own, because each is a decision that can be
//! read, tested and changed without the other two:
//!
//! - [`transient`] lowers a compiled resource requirement onto a Layer 1
//!   creation descriptor, in both directions.
//! - [`retention`] is what keeps a transient alive until its last handle drops,
//!   and where its death is recorded.
//! - [`submission`] is the record that makes a completion query total without
//!   polling.
//!
//! # What a GL-family context does not have
//!
//! **A submission is not an object.**  The common contract's `submit` returns a
//! completion that can be polled later; this family has an immediate context,
//! where work is issued as it is recorded and made visible by a flush, and a
//! fence is the only thing afterwards that can be asked about.  So a submission
//! here is `flush` and then `create_fence`, in that order and for that reason: a
//! fence reports completion of the commands issued before it, so creating it
//! after the flush is what makes it report *this* submission.
//!
//! **A completion query cannot poll.**  `completion_status` is `&self` and
//! `poll_fence` needs `&mut self`, so no implementation of the former can reach
//! the latter.  [`submission`] records every outcome a poll obtained and answers
//! from the record; where there is no record it answers `Unknown`, which is the
//! contract's own fail-closed answer rather than an invention.
//!
//! **A transient's lifetime is a handle count, not a frame.**  The contract's
//! lease is documented as caller-owned *and* as retaining the resource until
//! completion, and the executor's transient pool holds one for the whole life of
//! a cached slot.  [`retention`] therefore keeps the object until the last clone
//! drops.
//!
//! # What this slice does not express
//!
//! Everything that would record a command refuses with `GlError::Unsupported`,
//! naming itself and giving one reason.  That includes the two transitions,
//! which is worth stating as a decision rather than an omission: this adapter
//! does report `TransitionCapabilities::BackendManaged`, but the contract still
//! requires `before == after` to be a real memory barrier, and a barrier is a
//! command like any other.  Accepting a transition as a no-op would be the one
//! thing a fail-closed layer must not do -- it would tell the executor a hazard
//! was resolved when nothing was emitted.
//!
//! The copy-pass brackets are the exception, and they are not a stub: a copy in
//! this family is a direct command with no scope around it, so `begin_copy` and
//! `end_copy` have nothing to bracket and are no-ops in the final implementation
//! too.  A graph that opens a copy pass therefore gets past the brackets and is
//! refused by the copy verb itself, which is where the work would have happened.
//!
//! # Teardown
//!
//! There is deliberately no `Drop` here.  The contract asks a backend that is
//! dropped with pending retirements to "either wait for them or perform native
//! device teardown that makes it safe to release every referenced object", and
//! the second is what dropping this type does: the leases in the ledger and the
//! queue protect objects owned by the machine's backend, and that backend's own
//! teardown invalidates them.  A `Drop` that destroyed them would have to reach
//! a provider from a lease, which is the arrangement [`retention`] exists to
//! avoid.

mod retention;
mod submission;
mod transient;

#[cfg(test)]
mod tests;

use std::ops::Range;
use std::rc::Rc;

use fluxel_rendergraph::{
    BoundBuffer, BoundTexture, BufferCopyRegion, BufferDesc, BufferRange, BufferUsage,
    CompletionStatus, DeviceCapabilities, DeviceIdentity, ExecutionBackend, IndexFormat,
    PresentationSubmission, QueueId, RasterPassDescriptor, ResourceAccessState, ScissorRect,
    TextureCopyRegion, TextureDesc, TextureRange, TextureUsage, Viewport,
};

use crate::webgl2::api::{
    BufferId, ContextStamp, GlContextLifecycle, GlError, GlFenceLease, GlFenceStatus, TextureId,
};
use crate::webgl2::state::{GlStateBackend, GlStateMachine, StateEvent};

use super::identity::DeviceIdentityMap;
use retention::{GlRetentionLease, ReleaseQueue, RetainedObject};
use submission::{Retirement, SubmissionLedger, failure};

/// Uninhabited raster-pipeline placeholder for this adapter.
///
/// Uninhabited rather than a stub: the common contract needs a name for the
/// object a renderer would select here, and this backend cannot currently
/// produce one, so the honest type is one no value can exist of.  A placeholder
/// that *could* be constructed would let an implementation hand back a pipeline
/// that selects nothing.
pub(crate) enum UnsupportedRasterPipeline {}

/// Uninhabited compute-pipeline placeholder for this adapter.
pub(crate) enum UnsupportedComputePipeline {}

/// Uninhabited binding placeholder for this adapter.
pub(crate) enum UnsupportedBindings {}

/// Uninhabited presentation-token placeholder for this adapter.
///
/// The contract's token is produced by acquiring an image, and this adapter
/// reports no surface in its capabilities and `present: false` on its queue, so
/// no acquisition can reach it.  Saying that with a type is stronger than
/// saying it with a check: the executor's presentation list is empty by
/// construction rather than by convention.
pub(crate) enum UnsupportedPresentationToken {}

/// One recording in progress.
///
/// It carries the context generation it was opened against and nothing else.
/// That is not a placeholder for state that will arrive: this family's context
/// is immediate, so recording *is* issuing, and the only thing the common
/// contract needs at this boundary is the ability to reject a command buffer
/// finished against a context generation that has since been replaced.
pub(crate) struct GlEncoder {
    context: ContextStamp,
}

/// One finished recording, ready to submit.
pub(crate) struct GlCommandBuffer {
    context: ContextStamp,
}

/// The reason every verb that would record a command refuses in this slice.
const NO_COMMAND_VOCABULARY: &str =
    "this adapter records no command yet, so the request cannot be made true in the driver";

/// The reason a semantic transition refuses in this slice.
///
/// Separate from [`NO_COMMAND_VOCABULARY`] because there is a second thing to
/// say: a transition whose `before` and `after` agree is not a no-op in this
/// contract -- it is a memory barrier -- so a backend that skipped it would be
/// dropping a hazard rather than saving a call.
const NO_TRANSITION_BARRIER: &str = "a transition is a memory barrier as well as a state change, and this adapter has no barrier command to emit";

/// A GL-family state machine presented as a common execution backend.
pub(crate) struct GlCompatibilityDevice<B: GlStateBackend> {
    machine: GlStateMachine<B>,
    identity: DeviceIdentityMap,
    capabilities: DeviceCapabilities,
    releases: Rc<ReleaseQueue>,
    submissions: SubmissionLedger,
}

impl<B: GlStateBackend> GlCompatibilityDevice<B> {
    /// The adapter over `backend`.
    ///
    /// The capability description and the context stamp are read from the
    /// backend *before* the machine takes it, because both are facts about the
    /// context rather than about the mirror: they come from the discovery
    /// snapshot, which the machine does not consult.
    pub(crate) fn new(backend: B) -> Self {
        let stamp = backend.context_stamp();
        let capabilities = super::capabilities::capabilities(backend.discovery());
        Self {
            machine: GlStateMachine::new(backend),
            identity: DeviceIdentityMap::new(stamp),
            capabilities,
            releases: ReleaseQueue::new(),
            submissions: SubmissionLedger::default(),
        }
    }

    /// Adopts a context generation change, if the provider reports one.
    ///
    /// A GL-family provider does not tell this adapter that its context was lost
    /// and restored; it reports a strictly newer stamp afterwards.  So the epoch
    /// is the observable, and everything here follows from it: the common device
    /// identity is reallocated (a lost context is a generation change and not a
    /// new device), the mirrors are invalidated, and the capability description
    /// is read again from the restored context's own discovery snapshot.
    ///
    /// Nothing of the previous generation is destroyed.  A restored context
    /// invalidated every object table, lease book and derived record before it
    /// reported the new stamp, so the identities held here name objects that no
    /// longer exist and destroying them would be a call against a dead epoch.
    /// The order of the two discards is load-bearing: the leases are dropped
    /// first and the queue is forgotten second, because dropping them pushes
    /// onto that same queue.
    fn refresh(&mut self) {
        let stamp = self.machine.backend().context_stamp();
        if !self.identity.adopt(stamp) {
            return;
        }
        drop(self.submissions.purge());
        self.releases.forget();
        self.capabilities = super::capabilities::capabilities(self.machine.backend().discovery());
        self.machine.invalidate(StateEvent::ContextRestored(stamp));
    }

    /// Destroys every object the release queue has collected since the last call.
    ///
    /// Each object is dispatched to the mirror *before* it is destroyed.  That
    /// ordering is the one Layer 2's event matrix requires and cannot enforce: a
    /// domain that still holds a binding naming the object has to drop it while
    /// the identity still means the object it describes, or a slot reused after
    /// the deletion could reach a record of its previous occupant.
    ///
    /// A context that accepts no commands does nothing here and keeps the
    /// records.  Every destroy verb preflights the lifecycle, so there is no
    /// call to make -- and "suspended" and "lost" are not the same fact: a
    /// suspended context returns and its objects are still its own to destroy,
    /// while a lost one is handled where the loss is observable, in
    /// [`Self::refresh`].
    fn release_pending(&mut self) -> Result<(), GlError> {
        if !self.machine.backend().lifecycle().accepts_commands() {
            return Ok(());
        }
        let mut first_error = None;
        for object in self.releases.drain() {
            let outcome = match object {
                RetainedObject::Texture(texture) => {
                    self.machine.invalidate(StateEvent::TextureDeleted(texture));
                    self.machine.backend().destroy_texture_resource(texture)
                }
                RetainedObject::Buffer(buffer) => {
                    self.machine.invalidate(StateEvent::BufferDeleted(buffer));
                    self.machine.backend().destroy_buffer_resource(buffer)
                }
            };
            if let Err(error) = outcome {
                first_error.get_or_insert(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// The common completion state one GL fence report stands for.
///
/// Exhaustive rather than wildcarded: `GlFenceStatus` is this crate's own type
/// and is not `#[non_exhaustive]`, so a new variant is a change to the
/// GL-family contract and should stop this lowering rather than fall into a
/// default.  `Failed` is the one that needs a second fact: the GL family says a
/// fence failed and nothing more, while the contract asks which of two failures
/// it was, and the lifecycle is where the context keeps that.
fn completion_status(status: GlFenceStatus, lifecycle: GlContextLifecycle) -> CompletionStatus {
    match status {
        GlFenceStatus::Pending => CompletionStatus::Pending,
        // The report says the signal did not happen.  Reporting a failure would
        // be inventing one, and the contract already has a name for not knowing.
        GlFenceStatus::Unknown => CompletionStatus::Unknown,
        GlFenceStatus::Complete => CompletionStatus::Complete,
        GlFenceStatus::Failed => CompletionStatus::Failed(failure(lifecycle)),
    }
}

impl<B: GlStateBackend> ExecutionBackend for GlCompatibilityDevice<B> {
    type Texture = TextureId;
    type Buffer = BufferId;
    type RasterPipeline = UnsupportedRasterPipeline;
    type ComputePipeline = UnsupportedComputePipeline;
    type Bindings = UnsupportedBindings;
    type Encoder = GlEncoder;
    type CommandBuffer = GlCommandBuffer;
    type Completion = GlFenceLease;
    type PresentationToken = UnsupportedPresentationToken;
    type Lease = GlRetentionLease;
    type Error = GlError;

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.identity.identity()
    }

    fn create_transient_texture(
        &mut self,
        descriptor: TextureDesc,
        usage: TextureUsage,
    ) -> Result<BoundTexture<Self::Texture, Self::Lease>, Self::Error> {
        self.refresh();
        self.release_pending()?;
        let lowered = transient::texture_descriptor(descriptor, usage)?;
        let physical = self.machine.backend().create_texture_resource(lowered)?;
        let lease =
            GlRetentionLease::new(RetainedObject::Texture(physical), Rc::clone(&self.releases));
        Ok(BoundTexture {
            device: self.identity.identity(),
            identity: transient::resource_identity(physical.slot, physical.generation),
            physical,
            descriptor,
            // Derived from the creation facts rather than echoed from the
            // request, because one GL usage bit covers more than one common
            // operation and the physical object really does permit all of them.
            usage: transient::texture_usage(lowered.usage, lowered.format),
            initial_state: ResourceAccessState::Undefined,
            lease,
        })
    }

    fn create_transient_buffer(
        &mut self,
        descriptor: BufferDesc,
        usage: BufferUsage,
    ) -> Result<BoundBuffer<Self::Buffer, Self::Lease>, Self::Error> {
        self.refresh();
        self.release_pending()?;
        let lowered = transient::buffer_descriptor(descriptor, usage)?;
        let physical = self.machine.backend().create_buffer_resource(lowered)?;
        let lease =
            GlRetentionLease::new(RetainedObject::Buffer(physical), Rc::clone(&self.releases));
        Ok(BoundBuffer {
            device: self.identity.identity(),
            identity: transient::resource_identity(physical.slot, physical.generation),
            physical,
            descriptor,
            usage: transient::buffer_usage(lowered.usage),
            initial_state: ResourceAccessState::Undefined,
            lease,
        })
    }

    fn begin_encoder(&mut self, queue: QueueId) -> Result<Self::Encoder, Self::Error> {
        self.refresh();
        if queue != QueueId::new(0) {
            return Err(GlError::Unsupported {
                operation: "begin-encoder",
                reason: "this backend records onto one immediate context on one owning thread, which the common contract names as queue zero",
            });
        }
        self.release_pending()?;
        let backend = self.machine.backend();
        backend.assert_ready("begin-encoder")?;
        Ok(GlEncoder {
            context: backend.context_stamp(),
        })
    }

    fn transition_texture(
        &mut self,
        _encoder: &mut Self::Encoder,
        _texture: &Self::Texture,
        _range: TextureRange,
        _before: ResourceAccessState,
        _after: ResourceAccessState,
    ) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "transition-texture",
            reason: NO_TRANSITION_BARRIER,
        })
    }

    fn transition_buffer(
        &mut self,
        _encoder: &mut Self::Encoder,
        _buffer: &Self::Buffer,
        _range: BufferRange,
        _before: ResourceAccessState,
        _after: ResourceAccessState,
    ) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "transition-buffer",
            reason: NO_TRANSITION_BARRIER,
        })
    }

    fn begin_raster(
        &mut self,
        _encoder: &mut Self::Encoder,
        _descriptor: &RasterPassDescriptor<'_, Self::Texture>,
    ) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "begin-raster",
            reason: NO_COMMAND_VOCABULARY,
        })
    }

    fn end_raster(&mut self, _encoder: &mut Self::Encoder) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "end-raster",
            reason: NO_COMMAND_VOCABULARY,
        })
    }

    fn begin_compute(
        &mut self,
        _encoder: &mut Self::Encoder,
        _label: &str,
    ) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "begin-compute",
            reason: NO_COMMAND_VOCABULARY,
        })
    }

    fn end_compute(&mut self, _encoder: &mut Self::Encoder) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "end-compute",
            reason: NO_COMMAND_VOCABULARY,
        })
    }

    fn begin_copy(
        &mut self,
        _encoder: &mut Self::Encoder,
        _label: &str,
    ) -> Result<(), Self::Error> {
        // Not a stub: there is no copy scope in this family to open -- see the
        // module documentation.
        Ok(())
    }

    fn end_copy(&mut self, _encoder: &mut Self::Encoder) -> Result<(), Self::Error> {
        Ok(())
    }

    fn set_raster_pipeline(
        &mut self,
        _encoder: &mut Self::Encoder,
        _pipeline: &Self::RasterPipeline,
    ) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "set-raster-pipeline",
            reason: NO_COMMAND_VOCABULARY,
        })
    }

    fn set_compute_pipeline(
        &mut self,
        _encoder: &mut Self::Encoder,
        _pipeline: &Self::ComputePipeline,
    ) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "set-compute-pipeline",
            reason: NO_COMMAND_VOCABULARY,
        })
    }

    fn set_bindings(
        &mut self,
        _encoder: &mut Self::Encoder,
        _bindings: &Self::Bindings,
    ) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "set-bindings",
            reason: NO_COMMAND_VOCABULARY,
        })
    }

    fn set_vertex_buffer(
        &mut self,
        _encoder: &mut Self::Encoder,
        _slot: u32,
        _buffer: &Self::Buffer,
        _offset: u64,
    ) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "set-vertex-buffer",
            reason: NO_COMMAND_VOCABULARY,
        })
    }

    fn set_index_buffer(
        &mut self,
        _encoder: &mut Self::Encoder,
        _buffer: &Self::Buffer,
        _offset: u64,
        _format: IndexFormat,
    ) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "set-index-buffer",
            reason: NO_COMMAND_VOCABULARY,
        })
    }

    fn set_viewport(
        &mut self,
        _encoder: &mut Self::Encoder,
        _viewport: Viewport,
    ) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "set-viewport",
            reason: NO_COMMAND_VOCABULARY,
        })
    }

    fn set_scissor(
        &mut self,
        _encoder: &mut Self::Encoder,
        _scissor: ScissorRect,
    ) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "set-scissor",
            reason: NO_COMMAND_VOCABULARY,
        })
    }

    fn draw(
        &mut self,
        _encoder: &mut Self::Encoder,
        _vertices: Range<u32>,
        _instances: Range<u32>,
    ) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "draw",
            reason: NO_COMMAND_VOCABULARY,
        })
    }

    fn draw_indexed(
        &mut self,
        _encoder: &mut Self::Encoder,
        _indices: Range<u32>,
        _base_vertex: i32,
        _instances: Range<u32>,
    ) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "draw-indexed",
            reason: NO_COMMAND_VOCABULARY,
        })
    }

    fn dispatch(
        &mut self,
        _encoder: &mut Self::Encoder,
        _groups: [u32; 3],
    ) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "dispatch",
            reason: NO_COMMAND_VOCABULARY,
        })
    }

    fn copy_texture(
        &mut self,
        _encoder: &mut Self::Encoder,
        _source: &Self::Texture,
        _destination: &Self::Texture,
        _region: TextureCopyRegion,
    ) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "copy-texture",
            reason: NO_COMMAND_VOCABULARY,
        })
    }

    fn copy_buffer(
        &mut self,
        _encoder: &mut Self::Encoder,
        _source: &Self::Buffer,
        _destination: &Self::Buffer,
        _region: BufferCopyRegion,
    ) -> Result<(), Self::Error> {
        Err(GlError::Unsupported {
            operation: "copy-buffer",
            reason: NO_COMMAND_VOCABULARY,
        })
    }

    fn finish_encoder(
        &mut self,
        encoder: Self::Encoder,
    ) -> Result<Self::CommandBuffer, Self::Error> {
        // Nothing to finish: recording in this family happened when the commands
        // were issued, which in this slice is never.  The generation the encoder
        // was opened against is carried forward so that submission can reject a
        // command buffer whose context has since been replaced.
        Ok(GlCommandBuffer {
            context: encoder.context,
        })
    }

    fn submit(
        &mut self,
        queue: QueueId,
        command_buffer: Self::CommandBuffer,
        presentations: Vec<PresentationSubmission<Self::PresentationToken>>,
    ) -> Result<Self::Completion, Self::Error> {
        self.refresh();
        // The tokens are answered first, and by being dropped.  The contract
        // requires every token to be left unconsumed on `Err` so that its `Drop`
        // performs the cancellation, and returning here does exactly that.
        if !presentations.is_empty() {
            return Err(GlError::Unsupported {
                operation: "submit",
                reason: "this backend reports no surface and acquires no image, so it has no path that could present one",
            });
        }
        if queue != QueueId::new(0) {
            return Err(GlError::Unsupported {
                operation: "submit",
                reason: "this backend has one ordered command path, which the common contract names as queue zero",
            });
        }
        self.release_pending()?;
        let backend = self.machine.backend();
        backend.validate_object_context("submit", command_buffer.context)?;
        // `flush` makes the commands issued before it visible to the device, and
        // `create_fence` then inserts a fence they are ordered before.  Swapping
        // the two would make the fence report the previous submission, which is
        // the one mistake this pair can make.
        backend.flush()?;
        let fence = backend.create_fence()?;
        self.submissions.record(fence);
        Ok(fence)
    }

    fn completion_status(&self, completion: &Self::Completion) -> CompletionStatus {
        self.submissions.outcome(completion)
    }

    fn retire(&mut self, completion: Self::Completion, leases: Vec<Self::Lease>) {
        match self.submissions.retire(completion, leases) {
            // The submission is still in flight: it holds them, and whichever
            // poll settles it releases them.
            Retirement::Held => {}
            // This is where they are released, and it is a real act: dropping a
            // retention lease records its object in the release queue, and the
            // frame's own `collect_retired` destroys it -- the call the executor
            // makes immediately after this one.
            Retirement::Release(leases) => drop(leases),
        }
    }

    fn collect_retired(&mut self) -> Result<usize, Self::Error> {
        self.refresh();
        // Every unsettled fence is polled under one borrow of the backend, and
        // the answers are applied outside it.  A fence the backend refuses to
        // describe keeps whatever outcome it already had, which is how a failed
        // poll leaves a submission quarantined instead of releasing work that
        // may still be running; the first refusal is what this call reports.
        let fences: Vec<GlFenceLease> = self.submissions.unsettled().collect();
        let mut observed = Vec::with_capacity(fences.len());
        let mut first_error = None;
        for fence in fences {
            match self.machine.backend().poll_fence(fence) {
                Ok(status) => observed.push((
                    fence,
                    completion_status(status, self.machine.backend().lifecycle()),
                )),
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        let (released, finished) = self.submissions.settle(&observed);
        // The count is the settlements and not the leases.  Every other backend
        // in this workspace reports how many retirement entries the poll
        // released, and one public number with two meanings is worse than a
        // slightly loose name.  It is a superset of theirs by construction: this
        // ledger tracks a submission from the moment it is issued, because
        // `completion_status` has to be answerable before anyone retires
        // anything, while a queue of retirements can only hold what was handed
        // over.
        let count = finished.len();
        // Released here rather than at the next entry point, so the frame that
        // learned the work is done is the frame that frees it.
        drop(released);
        for fence in finished {
            // A fence is a driver object and not a token.  Everything this
            // adapter can still be asked about a settled submission comes from
            // the recorded outcome, and the record is only ever compared against
            // -- never polled -- so the object goes as soon as it can be asked
            // nothing, and the key it leaves behind stays valid for exactly as
            // long as the record does.
            if let Err(error) = self.machine.backend().destroy_fence(fence) {
                first_error.get_or_insert(error);
            }
        }
        // Run whatever the release queue collected even when a poll or a destroy
        // failed: the objects in it are already unreachable, and deferring them
        // would only postpone the same call to a frame that may not come.
        let release_error = self.release_pending().err();
        match first_error.or(release_error) {
            Some(error) => Err(error),
            None => Ok(count),
        }
    }
}
