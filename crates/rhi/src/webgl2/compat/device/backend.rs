//! The common execution contract's verbs, one method each.
//!
//! This is the [`ExecutionBackend`] impl for [`GlCompatibilityDevice`], and it is
//! one `impl` block rather than two: a trait is implemented for a type in one
//! place (E0119), so the families a reader might expect as separate files --
//! resources and transients, passes and draws, submission -- are one file, in the
//! trait's own declaration order.  That order is not regrouped here, because a
//! second ordering of the same methods would be a second thing to keep true; what
//! each verb is for is documented next to it, and the decisions behind all of
//! them are argued in [`super`].
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
//! [`super::encoder`].  The two free helpers at the foot are the lowering two of
//! the verbs do before they touch the machine.

use std::ops::Range;
use std::rc::Rc;

use fluxel_rendergraph::{
    BoundBuffer, BoundTexture, BufferCopyRegion, BufferDesc, BufferRange, BufferUsage,
    CompletionStatus, DeviceCapabilities, DeviceIdentity, ExecutionBackend, IndexFormat,
    PresentationSubmission, QueueId, RasterPassDescriptor, ResourceAccessState, ScissorRect,
    TextureCopyRegion, TextureDesc, TextureRange, TextureUsage, Viewport,
};

use crate::webgl2::api::{
    BufferId, GlContextLifecycle, GlDrawCommand, GlError, GlFenceLease, GlFenceStatus,
    GlIndexBinding, GlIndexedDraw, GlNonIndexedDraw, GlRenderPassDescriptor, GlVertexBufferBinding,
    TextureId,
};
use crate::webgl2::state::GlStateBackend;

use super::compute::ComputeDomain;
use super::encoder::{GlCommandBuffer, GlEncoder, InstalledPipeline, OpenPass, PassShape};
use super::failure::{self, malformed, pass_open, unsupported};
use super::object;
use super::pass;
use super::region;
use super::retention::{GlRetentionLease, RetainedObject};
use super::submission::{Retirement, failure as submission_failure};
use super::transient;
use super::{GlCompatibilityDevice, UnsupportedPresentationToken};

/// The instance count of a draw that asks for exactly one instance.
///
/// This family's instanced draw is `draw_advanced_raster`, an optional verb with
/// per-instance offsets this adapter has no lowering for, so a range naming more
/// than one instance is refused by name rather than silently drawn once.
///
/// A range naming *none* is refused as well, and by this verb rather than by the
/// backend.  Zero instances is a legal command that rasterizes nothing, so the
/// check belongs last -- but "last" here means inside the provider, and a zero
/// that reached it would be reported against `draw-raster`, an operation name
/// the frame never issued.  So the refusal is made where the name is still the
/// caller's.  It is also a request the executor cannot produce: it refuses an
/// empty instance range before any backend sees one, which leaves a caller that
/// bypassed the executor as the only way to arrive here.
fn single_instance(operation: &'static str, instances: &Range<u32>) -> Result<u32, GlError> {
    match instances.len() {
        0 => Err(malformed(
            operation,
            "a draw that runs no instance has nothing to rasterize",
        )),
        1 => Ok(1),
        _ => Err(unsupported(
            operation,
            "this family's instanced draw is an optional verb this adapter has no lowering for, so a draw asks for at most one instance",
        )),
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
        GlFenceStatus::Failed => CompletionStatus::Failed(submission_failure(lifecycle)),
    }
}

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
        self.refresh();
        if encoder.pass.is_some() {
            return Err(pass_open("begin-raster"));
        }
        self.machine
            .backend()
            .validate_object_context("begin-raster", encoder.context)?;
        let attachment = pass::admit(descriptor.colors, descriptor.depth_stencil.as_ref())?;
        // The adapter's own record of what it created.  A texture this device did
        // not create has no shape to lower, and the refusal names that rather
        // than reporting a missing attachment.
        let facts = *self.attachments.get(attachment.texture).ok_or_else(|| {
            malformed(
                "begin-raster",
                "the pass attaches a texture this device did not create",
            )
        })?;
        self.machine
            .backend()
            .validate_object_context("begin-raster", attachment.texture.context)?;
        let views = vec![pass::view(*attachment.texture, facts, attachment.range)];
        let colors = pass::attachments(descriptor.colors, &views)?;
        let requested = pass::framebuffer(views, None);
        let (framebuffer, owned) = self
            .machine
            .framebuffer_for(&requested)
            .map_err(failure::into_gl_error)?;

        let mut open = OpenPass::raster(facts);
        if owned {
            // Narrowed by hand rather than through `OpenPass::raster`, which
            // returns a `Result`: the pass was built as a raster pass one line
            // above, so the narrowing cannot refuse, and this is the one place in
            // this file where an error path would have to be invented rather than
            // reported.  The framebuffer is already derived and has no second
            // name, so a `?` here would leak it.
            if let PassShape::Raster(shape) = &mut open.shape {
                shape.owned_framebuffers.push(framebuffer);
            }
        }
        encoder.pass = Some(open);

        let render_pass = GlRenderPassDescriptor {
            framebuffer,
            color_attachments: colors,
            depth_stencil_attachment: None,
        };
        if let Err(error) = self
            .machine
            .begin_pass(render_pass)
            .map_err(failure::into_gl_error)
        {
            // A pass that never opened has no `end_raster` coming: the executor
            // returns on this error rather than bracketing the callback, so
            // whatever this encoder came to own is destroyed here instead of by
            // a close that will not happen.
            if let Some(abandoned) = encoder.pass.take() {
                let _ = self.destroy_owned(abandoned);
            }
            return Err(error);
        }
        Ok(())
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
        self.refresh();
        let pass = encoder.take_raster("end-raster")?;
        let closed = self.machine.end_pass().map_err(failure::into_gl_error);
        // Destroyed whether or not the boundary closed: these objects are
        // unreachable either way, and a failure to end the pass is not a reason
        // to leak them as well.  Not destroyed, though, when the context was
        // replaced while the pass was open -- their identities belong to an epoch
        // the backend no longer accepts, the context's own teardown already
        // released them, and asking would turn a context loss into a second,
        // unrelated failure on the frame's error path.
        let released = if encoder.context == self.machine.backend().context_stamp() {
            self.destroy_owned(pass)
        } else {
            Ok(())
        };
        closed.and(released)
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
        let pass = self.open_pass(encoder, "set-raster-pipeline")?;
        // Refused when the open pass is a compute pass, which is what the
        // narrowing here is for: the two kinds do not share a pipeline slot, and a
        // raster recipe recorded in a compute pass would be installed by the next
        // dispatch.
        pass.raster_shape("set-raster-pipeline")?;
        // The recorded bindings are *kept*, and checked against this recipe
        // rather than dropped with the previous one: they are facts about what
        // the frame resolved, and a recipe that does not read them is a mistake
        // worth reporting at the bind that made it rather than a silent discard.
        pass.pipeline = Some(InstalledPipeline::Raster(pipeline.clone()));
        Ok(())
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
        let shape = self.raster_pass(encoder, "set-vertex-buffer")?;
        shape.vertex.retain(|bound| bound.slot != slot);
        shape.vertex.push(GlVertexBufferBinding {
            slot,
            buffer: *buffer,
            offset,
        });
        Ok(())
    }

    fn set_index_buffer(
        &mut self,
        encoder: &mut Self::Encoder,
        buffer: &Self::Buffer,
        offset: u64,
        format: IndexFormat,
    ) -> Result<(), Self::Error> {
        let shape = self.raster_pass(encoder, "set-index-buffer")?;
        shape.index = Some(GlIndexBinding {
            buffer: *buffer,
            format: pass::index_format(format),
            offset,
        });
        Ok(())
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
        self.refresh();
        let kernel = encoder.kernel("draw")?;
        if pass::indexed(kernel) {
            return Err(malformed(
                "draw",
                "this artifact takes its vertices from an index buffer, so it is drawn with draw-indexed",
            ));
        }
        let instance_count = single_instance("draw", &instances)?;
        if vertices.is_empty() {
            return Err(malformed(
                "draw",
                "a draw with no vertices has nothing to rasterize",
            ));
        }
        self.commit("draw", encoder)?;
        self.machine
            .backend()
            .draw_raster(GlDrawCommand::NonIndexed(GlNonIndexedDraw {
                first_vertex: vertices.start,
                vertex_count: vertices.len() as u32,
                instance_count,
            }))
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
        self.refresh();
        let kernel = encoder.kernel("draw-indexed")?;
        if !pass::indexed(kernel) {
            return Err(malformed(
                "draw-indexed",
                "this artifact draws its vertices without an index buffer, so it is drawn with draw",
            ));
        }
        let instance_count = single_instance("draw-indexed", &instances)?;
        if indices.is_empty() {
            return Err(malformed(
                "draw-indexed",
                "an indexed draw with no indices has nothing to rasterize",
            ));
        }
        if base_vertex != 0 {
            return Err(unsupported(
                "draw-indexed",
                "this family adds the base vertex to each index inside the shader pipeline, which is an optional verb this adapter has no lowering for",
            ));
        }
        self.commit("draw-indexed", encoder)?;
        self.machine
            .backend()
            .draw_raster(GlDrawCommand::Indexed(GlIndexedDraw {
                first_index: indices.start,
                index_count: indices.len() as u32,
                instance_count,
            }))
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
        // Recording in this family happens when the commands are issued, which
        // for a raster pass is its draws, so there is no buffer to build.  The
        // generation the encoder was opened against is carried forward so that
        // submission can reject a command buffer whose context has since been
        // replaced.
        //
        // A pass still open is refused rather than closed silently.  This
        // family's context runs one pass at a time and the executor brackets
        // every pass it opens, so an open one at this point means a frame
        // abandoned it -- and a command buffer that reported success would be a
        // frame claiming a completed render for a boundary it never crossed.
        //
        // The refusal is reported, and the *pass is still unwound*: this is the
        // last call that can reach it, and a provider whose pass is still open
        // refuses the next `begin-pass` ("render pass already active" on both
        // executable providers and on the recorder), which would turn one
        // abandoned pass into a context no later frame can render on.  So the
        // boundary is closed for the backend's sake while the frame is told what
        // went wrong; the close's own failure is not reported, because it is
        // cleanup for a mistake already named and a second error would replace
        // the diagnosis with its consequence.  Whatever the pass came to own is
        // destroyed either way, since there is no later call that could name it.
        //
        // Only a raster pass has a boundary to unwind, and that is a fact of its
        // shape rather than a policy: a compute pass opened nothing in Layer 1,
        // because this family's pass boundary is a framebuffer's.  So the unwind
        // is conditioned on the shape, and the *message* is not: an abandoned pass
        // is the same mistake either way, and a frame reading it should not have
        // to work out which kind it left open.
        let mut encoder = encoder;
        if let Some(abandoned) = encoder.pass.take() {
            if !abandoned.is_compute() {
                let _ = self.machine.end_pass();
            }
            let _ = self.destroy_owned(abandoned);
            return Err(malformed(
                "finish-encoder",
                "a pass was left open on this encoder, and a command buffer cannot be finished inside one",
            ));
        }
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
