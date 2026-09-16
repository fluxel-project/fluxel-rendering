//! The common execution contract's verbs, one method each.
//!
//! This is the [`ExecutionBackend`] impl for [`GlCompatibilityDevice`], and it is
//! one `impl` block rather than two: a trait is implemented for a type in one
//! place (E0119), so the families a reader might expect as separate files --
//! resources and transients, passes and draws, submission -- are one block, in
//! the trait's own declaration order.  That order is not regrouped here, because
//! a second ordering of the same methods would be a second thing to keep true;
//! what each verb is for is documented next to it, and the decisions behind all
//! of them are argued in [`super`].
//!
//! # What this file is, and is not
//!
//! E0119 forces *one impl block*; it does not force *one file of bodies*, and
//! this file used to read the first as the second -- its own module doc argued
//! that the families "are one file" for the trait's reason, while three of its
//! verbs were already one-line delegations into [`super::compute`].  The
//! substance of a family's verbs belongs with that family: the raster bracket,
//! its recipe, its vertex input and its two draws are in [`super::raster`], the
//! compute bracket and its dispatch in [`super::compute`], the submission verbs
//! and the completion lowering in [`super::submission`], and a recording's
//! teardown in [`super::encoder`].  What remains here is the trait's own surface
//! -- the declaration order, the associated types, the verbs that are a
//! coordination body and nothing more, and the delegations -- which is the one
//! responsibility this file can state without a conjunction.
//!
//! The impl carries the adapter's compute witness as a type parameter, and that
//! is forced rather than chosen: a method cannot be stricter than its impl's own
//! bounds (E0276), so the optional `where B: GlOptionalComputeBackend` that the
//! compute verbs need has nowhere to go but a parameter of the type.  What the
//! witness buys is that the four compute verbs delegate to methods that exist for
//! every witness -- a refusal for [`NoCompute`](super::compute::NoCompute), the
//! real command for [`WithCompute`](super::compute::WithCompute) -- so this file
//! names no capability check of its own and each of those verbs is one line.  See
//! [`super::compute`].
//!
//! What the verbs share lives next door: the adapter's lifecycle and the
//! transition it accepts in [`super`], the recording machinery they drive in
//! [`super::encoder`].

use std::ops::Range;
use std::rc::Rc;

use fluxel_rendergraph::{
    BoundBuffer, BoundTexture, BufferCopyRegion, BufferDesc, BufferRange, BufferUsage,
    CompletionStatus, DeviceCapabilities, DeviceIdentity, ExecutionBackend, IndexFormat,
    PresentationSubmission, QueueId, RasterPassDescriptor, ResourceAccessState, ScissorRect,
    TextureCopyRegion, TextureDesc, TextureRange, TextureUsage, Viewport,
};

use crate::webgl2::api::{BufferId, GlError, GlFenceLease, TextureId};
use crate::webgl2::state::GlStateBackend;

use super::compute::ComputeDomain;
use super::encoder::{GlCommandBuffer, GlEncoder, InstalledPipeline};
use super::failure::malformed;
use super::object;
use super::pass;
use super::region;
use super::retention::{GlRetentionLease, RetainedObject};
use super::transient;
use super::{GlCompatibilityDevice, UnsupportedPresentationToken};

impl<B: GlStateBackend, C: ComputeDomain<B>> ExecutionBackend for GlCompatibilityDevice<B, C> {
    type Texture = TextureId;
    type Buffer = BufferId;
    type RasterPipeline = object::RasterPipeline;
    type ComputePipeline = object::ComputePipeline;
    type Bindings = object::Bindings;
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
        let attachment = pass::Attachment::of(&lowered);
        let physical = self.machine.backend().create_texture_resource(lowered)?;
        self.attachments.insert(physical, attachment);
        let lease = GlRetentionLease::new(
            [RetainedObject::Texture(physical)],
            Rc::clone(&self.releases),
        );
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
        let allocated = lowered.size;
        let physical = self.machine.backend().create_buffer_resource(lowered)?;
        // Recorded before anything else can name this identity, and for the same
        // reason the texture path records its attachment here: a storage binding
        // range authorized as *whole* has to be lowered to a real size, and the
        // creation descriptor is the only place that size is a fact.  See
        // [`GlCompatibilityDevice::storage_range`].
        self.buffers.insert(physical, allocated);
        let lease = GlRetentionLease::new(
            [RetainedObject::Buffer(physical)],
            Rc::clone(&self.releases),
        );
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
            pass: None,
        })
    }

    fn transition_texture(
        &mut self,
        encoder: &mut Self::Encoder,
        texture: &Self::Texture,
        _range: TextureRange,
        _before: ResourceAccessState,
        _after: ResourceAccessState,
    ) -> Result<(), Self::Error> {
        self.accept_transition("transition-texture", encoder.context, texture.context)
    }

    fn transition_buffer(
        &mut self,
        encoder: &mut Self::Encoder,
        buffer: &Self::Buffer,
        _range: BufferRange,
        _before: ResourceAccessState,
        _after: ResourceAccessState,
    ) -> Result<(), Self::Error> {
        self.accept_transition("transition-buffer", encoder.context, buffer.context)
    }

    /// Opens a raster pass over the descriptor's one colour attachment.
    ///
    /// The pass is *recorded* here and issued when the frame draws -- the module
    /// documentation says why -- so this verb's work is to admit the pass, derive
    /// a framebuffer for its attachment, and open the boundary.  Admission and
    /// attachment lowering both happen before the derivation, because a
    /// framebuffer Layer 2 reports this pass owns has no second name: the only
    /// place it can be destroyed is the pass that derived it, so a refusal made
    /// after the derivation would leak it.
    fn begin_raster(
        &mut self,
        encoder: &mut Self::Encoder,
        descriptor: &RasterPassDescriptor<'_, Self::Texture>,
    ) -> Result<(), Self::Error> {
        self.open_raster_pass(encoder, descriptor)
    }

    /// Closes the pass this encoder has open.
    ///
    /// The executor calls this *unconditionally* once a pass was opened -- after
    /// the callback and after a failed one, both inside a catch -- so this verb
    /// is reached on paths where draws failed or never happened, and it closes
    /// whatever is open without consulting what the frame did.  That is why the
    /// ownership bookkeeping lives here and not in the draws: this is the one
    /// call guaranteed to run for every pass that opened.
    ///
    /// A call with no pass open is refused rather than absorbed, because the only
    /// way to reach it is a caller that never opened one -- and absorbing it
    /// would let a frame that believes it rendered see a clean result for a pass
    /// that never existed.
    fn end_raster(&mut self, encoder: &mut Self::Encoder) -> Result<(), Self::Error> {
        self.close_raster_pass(encoder)
    }

    /// Opens the compute pass a dispatch will be recorded in.
    ///
    /// The label is *ignored*, and that is the same choice `begin_raster` makes
    /// about its descriptor's name and `begin_copy` about its own label: this
    /// family's context has no debug-group annotation to attach one to, so the
    /// name a frame gives a scope is a fact about the *plan* and not about any
    /// command.  It is deliberately not refused: the executor passes the pass's
    /// real name (`execution/recording/orchestration.rs`), so a refusal here would
    /// reject every compute pass in every frame.
    ///
    /// Everything else is the witness's, which is why this verb is one line: a
    /// witness with no compute domain refuses with the ledger's reason, and one
    /// with a domain refuses when the discovery snapshot did not prove the
    /// capability -- both before any side effect, and neither reachable from here.
    fn begin_compute(
        &mut self,
        encoder: &mut Self::Encoder,
        _label: &str,
    ) -> Result<(), Self::Error> {
        self.open_compute_pass(encoder, "begin-compute")
    }

    fn end_compute(&mut self, encoder: &mut Self::Encoder) -> Result<(), Self::Error> {
        self.close_compute_pass(encoder, "end-compute")
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

    /// Records the recipe this pass draws with.
    ///
    /// Nothing is resolved or installed here, and that is the module
    /// documentation's point: a [`GlRasterPipeline`] is a program, a vertex array
    /// and a rasterization state as one value, so the two ids it needs cannot be
    /// resolved until the frame has also said what it binds.  Recording the
    /// recipe is also what lets [`Self::commit`] name the artifact when it
    /// refuses a binding set resolved for a different one.
    fn set_raster_pipeline(
        &mut self,
        encoder: &mut Self::Encoder,
        pipeline: &Self::RasterPipeline,
    ) -> Result<(), Self::Error> {
        self.record_raster_pipeline(encoder, pipeline)
    }

    fn set_compute_pipeline(
        &mut self,
        encoder: &mut Self::Encoder,
        pipeline: &Self::ComputePipeline,
    ) -> Result<(), Self::Error> {
        // The mirror of the verb above, and the compute half's own narrowing is
        // [`GlEncoder::compute`]: recording a compute recipe into a raster pass
        // would make the next draw install it.
        self.record_compute_pipeline(encoder, "set-compute-pipeline", pipeline)
    }

    /// Records the set this pass draws or dispatches with.
    ///
    /// One verb for both families, because what a set is checked against is the
    /// artifact it was resolved for and that is the same question either way --
    /// while a set *resolved* for one family's artifact and installed in the
    /// other's pass is a different recipe and is refused by name below.  Both
    /// refusals name the operation, so a frame is told which verb asked.
    ///
    /// The check is made here as well as at the commit, and the two are not the
    /// same check: this one catches a set resolved for another artifact at the
    /// call that got it wrong, while the commit catches a pipeline installed
    /// *after* the set, which would otherwise bind resources at the wrong
    /// numbers.
    fn set_bindings(
        &mut self,
        encoder: &mut Self::Encoder,
        bindings: &Self::Bindings,
    ) -> Result<(), Self::Error> {
        let pass = self.open_pass(encoder, "set-bindings")?;
        let Some(installed) = pass.pipeline.as_ref().map(InstalledPipeline::recipe) else {
            return Err(malformed(
                "set-bindings",
                "a binding set is resolved for the artifact that reads it, and no pipeline is installed in this pass",
            ));
        };
        if bindings.recipe() != installed {
            return Err(malformed(
                "set-bindings",
                "the binding set was resolved for a different artifact than the one installed in this pass",
            ));
        }
        pass.bindings = Some((installed, bindings.slots().to_vec()));
        Ok(())
    }

    /// Records one vertex buffer, replacing whatever this pass had in that slot.
    ///
    /// Replacing rather than appending because a slot holds one buffer at a time
    /// in this family too, and a frame that set a slot twice meant the second
    /// one; keeping both would make the draw depend on which the search found
    /// first.
    fn set_vertex_buffer(
        &mut self,
        encoder: &mut Self::Encoder,
        slot: u32,
        buffer: &Self::Buffer,
        offset: u64,
    ) -> Result<(), Self::Error> {
        self.record_vertex_buffer(encoder, slot, buffer, offset)
    }

    fn set_index_buffer(
        &mut self,
        encoder: &mut Self::Encoder,
        buffer: &Self::Buffer,
        offset: u64,
        format: IndexFormat,
    ) -> Result<(), Self::Error> {
        self.record_index_buffer(encoder, buffer, offset, format)
    }

    fn set_viewport(
        &mut self,
        encoder: &mut Self::Encoder,
        viewport: Viewport,
    ) -> Result<(), Self::Error> {
        let shape = self.raster_pass(encoder, "set-viewport")?;
        shape.viewport = pass::viewport(viewport);
        Ok(())
    }

    fn set_scissor(
        &mut self,
        encoder: &mut Self::Encoder,
        scissor: ScissorRect,
    ) -> Result<(), Self::Error> {
        let shape = self.raster_pass(encoder, "set-scissor")?;
        shape.scissor = Some(pass::scissor(scissor));
        Ok(())
    }

    /// Draws the pass's artifact without an index buffer.
    ///
    /// Every check this verb can make is made before [`Self::commit`], and that
    /// order is the point: a commit installs a program, a vertex array and a
    /// pipeline, so a draw refused after one would leave the backend holding
    /// state for a command that never happened.
    fn draw(
        &mut self,
        encoder: &mut Self::Encoder,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) -> Result<(), Self::Error> {
        self.issue_draw(encoder, vertices, instances)
    }

    /// Draws the pass's artifact from its index buffer.
    ///
    /// `base_vertex` is refused when it is not zero rather than folded into the
    /// first index: this family adds it to each index inside the shader pipeline,
    /// and the verb that expresses that is an optional one this adapter has no
    /// lowering for -- so a non-zero value here would be a draw offset the caller
    /// asked for and the driver never made.
    fn draw_indexed(
        &mut self,
        encoder: &mut Self::Encoder,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> Result<(), Self::Error> {
        self.issue_indexed_draw(encoder, indices, base_vertex, instances)
    }

    /// Dispatches the pass's artifact.
    ///
    /// This is the compute half's commit point, on `draw`'s terms exactly: every
    /// check is made before the program is linked and the bindings applied, so a
    /// dispatch refused after one would leave the backend holding state for a
    /// command that never happened.
    ///
    /// A dispatch of no workgroups is refused one layer down rather than here,
    /// unlike a draw of no vertices, and the difference is which layer owns the
    /// rule.  A draw's vertex range is a fact this adapter is the only one to see
    /// -- Layer 1's draw verb takes a count and validates nothing about it -- so
    /// the empty case is refused above with a sentence that says what it means.
    /// A group count is the opposite: `GlDispatchGroups::validate` owns both the
    /// zero rule and the axis-limit rule, at the verb the dispatch reaches, so a
    /// second check here would be a coarser copy of a rule already enforced --
    /// and one that could disagree with it the first time an axis limit moved.
    fn dispatch(
        &mut self,
        encoder: &mut Self::Encoder,
        groups: [u32; 3],
    ) -> Result<(), Self::Error> {
        self.issue_dispatch(encoder, "dispatch", groups)
    }

    fn copy_texture(
        &mut self,
        encoder: &mut Self::Encoder,
        source: &Self::Texture,
        destination: &Self::Texture,
        region: TextureCopyRegion,
    ) -> Result<(), Self::Error> {
        self.refresh();
        let backend = self.machine.backend();
        backend.validate_object_context("copy-texture", encoder.context)?;
        backend.validate_object_context("copy-texture", source.context)?;
        backend.validate_object_context("copy-texture", destination.context)?;
        backend.copy_texture_region(
            region::texture_region(
                *source,
                region.source_mip_level,
                region.source_origin,
                region.extent,
            ),
            region::texture_region(
                *destination,
                region.destination_mip_level,
                region.destination_origin,
                region.extent,
            ),
        )
    }

    fn copy_buffer(
        &mut self,
        encoder: &mut Self::Encoder,
        source: &Self::Buffer,
        destination: &Self::Buffer,
        region: BufferCopyRegion,
    ) -> Result<(), Self::Error> {
        self.refresh();
        let backend = self.machine.backend();
        backend.validate_object_context("copy-buffer", encoder.context)?;
        backend.validate_object_context("copy-buffer", source.context)?;
        backend.validate_object_context("copy-buffer", destination.context)?;
        backend.copy_buffer_range(
            region::buffer_range(*source, region.source_offset, region.size),
            region::buffer_range(*destination, region.destination_offset, region.size),
        )
    }

    fn finish_encoder(
        &mut self,
        encoder: Self::Encoder,
    ) -> Result<Self::CommandBuffer, Self::Error> {
        self.finish_recording(encoder)
    }

    fn submit(
        &mut self,
        queue: QueueId,
        command_buffer: Self::CommandBuffer,
        presentations: Vec<PresentationSubmission<Self::PresentationToken>>,
    ) -> Result<Self::Completion, Self::Error> {
        self.submit_commands(queue, command_buffer, presentations)
    }

    fn completion_status(&self, completion: &Self::Completion) -> CompletionStatus {
        self.submissions.outcome(completion)
    }

    fn retire(&mut self, completion: Self::Completion, leases: Vec<Self::Lease>) {
        self.retire_completion(completion, leases);
    }

    fn collect_retired(&mut self) -> Result<usize, Self::Error> {
        self.collect_terminal_outcomes()
    }
}
