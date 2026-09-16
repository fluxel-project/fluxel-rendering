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
//! - [`region`] lowers the addressing half of a copy onto the values Layer 1's
//!   copy verbs take.
//! - [`retention`] is what keeps a transient alive until its last handle drops,
//!   and where its death is recorded.
//! - [`submission`] is the record that makes a completion query total without
//!   polling.
//!
//! Two more modules are what the *renderer* side of the contract needs, and they
//! are split from the four above along the same line -- one decision each:
//!
//! - [`object`] is what a raster pipeline and a binding set are here, and
//!   [`registry`] is how the executor resolves either one by identity.
//! - [`pass`] lowers one recorded raster pass onto Layer 1's pass vocabulary.
//!   It is pure, and that is deliberate: every decision about what a pass
//!   *becomes* is checkable without a context, which is what keeps the verbs
//!   below readable as the order they drive the machine in.
//!
//! The registry is a value rather than a field of the adapter, and
//! [`GlCompatibilityDevice::object_registry`] is the only constructor: it
//! captures the device identity and the context profile the recipes will be
//! lowered for, because a provider is reached through `&self` while the recorder
//! holds `&mut` on the adapter and therefore cannot ask it anything.
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
//! # What this slice expresses, and what it still refuses
//!
//! Two things that a reader would expect to be commands are real here and issue
//! none.
//!
//! A **transition** is accepted and emits nothing, because neither half of what
//! the contract asks for needs a command in this family: a GL-family resource
//! carries no access state of its own, and the ordering and visibility the
//! contract calls a memory barrier are carried by the execution model this
//! adapter declares (`SynchronizationCapabilities::SingleQueueOrdering`).  The
//! argument is made where it is needed rather than here -- see
//! [`Self::accept_transition`] -- because "this is accepted and nothing happens"
//! is exactly the claim a fail-closed layer must not make without showing its
//! work.
//!
//! A **copy** is the one command this adapter issues.  `copy_texture` and
//! `copy_buffer` lower the contract's addressing onto Layer 1's and hand it to
//! the provider, which checks the region against the real descriptor -- the only
//! place one exists.  [`region`] owns that lowering and states what it decides.
//!
//! A **raster pass and its draws** are real.  `begin_raster` derives the
//! framebuffer and opens the pass, the verbs inside it record what the frame
//! asked for, and a **draw is the commit point**: it is there, and not at each
//! verb, that Layer 2 is driven, because a GL-family pipeline carries its vertex
//! array and its rasterization state as one value
//! (`GlRasterPipeline { program, vertex_array, state }`) while the contract
//! supplies those pieces in the order a pipeline-first API supplies them.  See
//! [`GlCompatibilityDevice::commit`] for what that costs and why it is the
//! correct order rather than a workaround.
//!
//! Everything that would record a *compute* command still refuses with
//! `GlError::Unsupported`, naming itself and giving one reason: this adapter has
//! no compute pipeline object for such a pass to select.
//!
//! The copy-pass brackets are not among those refusals, and they are not a stub:
//! a copy in this family is a direct command with no scope around it, so
//! `begin_copy` and `end_copy` have nothing to bracket and are no-ops in the
//! final implementation too.
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

mod failure;
mod object;
mod pass;
mod region;
mod registry;
mod retention;
mod submission;
mod transient;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::ops::Range;
use std::rc::Rc;

use fluxel_rendergraph::{
    BoundBuffer, BoundTexture, BufferCopyRegion, BufferDesc, BufferRange, BufferUsage,
    CompletionStatus, DeviceCapabilities, DeviceIdentity, ExecutionBackend, IndexFormat,
    PresentationSubmission, QueueId, RasterPassDescriptor, ResourceAccessState, ScissorRect,
    TextureCopyRegion, TextureDesc, TextureRange, TextureUsage, Viewport,
};

use crate::resource::RasterKernel;
use crate::webgl2::api::{
    BufferId, ContextStamp, FramebufferId, GlContextLifecycle, GlDrawCommand, GlError,
    GlFenceLease, GlFenceStatus, GlIndexBinding, GlIndexedDraw, GlNonIndexedDraw, GlRasterPipeline,
    GlRenderPassDescriptor, GlScissorRect, GlVertexBufferBinding, GlViewport, ProgramId, TextureId,
};
use crate::webgl2::state::{GlStateBackend, GlStateMachine, StateEvent};

use super::identity::DeviceIdentityMap;
use failure::{malformed, no_pass, pass_open, unsupported};
use object::BindingSlot;
use registry::GlObjectRegistry;
use retention::{GlRetentionLease, ReleaseQueue, RetainedObject};
use submission::{Retirement, SubmissionLedger, failure as submission_failure};

/// Uninhabited compute-pipeline placeholder for this adapter.
///
/// Still uninhabited, and now the only one of the three placeholders that is.
/// Compute is F4's slice and has no recipes to lower, while the raster pipeline
/// and the binding set are real objects with real lowered contents, so their
/// placeholders are gone.  What is left is the case where the contract needs a
/// name and this backend genuinely cannot have a value of it -- and uninhabited
/// rather than a stub is still the point: a placeholder that *could* be
/// constructed would let an implementation hand back a pipeline that selects
/// nothing.
pub(crate) enum UnsupportedComputePipeline {}

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
/// It carries the context generation it was opened against and, while a pass is
/// open, what the frame has recorded inside it.  The context stamp is what lets
/// submission reject a command buffer finished against a generation that has
/// since been replaced.
///
/// # Why a pass is recorded and not issued as its verbs arrive
///
/// This family's context is immediate, so a verb *could* issue -- and the copy
/// verbs do.  A raster pass cannot, and the reason is a shape difference rather
/// than a policy: Layer 2 installs rasterization state as one value that names
/// its vertex array (`GlRasterPipeline { program, vertex_array, state }`), and
/// deriving that array needs the vertex input, while the contract supplies the
/// pipeline first and the vertex buffers after it.  So the verbs record, and the
/// draw drives Layer 2 once everything is in hand.
///
/// That is not deferred execution: nothing here is queued for a later frame, and
/// a pass that records a draw issues it before `draw` returns.  What the record
/// buys is that the *order* Layer 2 needs is the order the commit uses, instead
/// of an order this vocabulary cannot express.
pub(crate) struct GlEncoder {
    context: ContextStamp,
    pass: Option<OpenPass>,
}

/// What the frame has recorded inside the pass this encoder has open.
///
/// Every field is per-pass and cleared where the pass ends, which is the same
/// lifetime Layer 1's own pass scope has: a binding recorded in one pass is not
/// in force in the next, and neither is a viewport.
struct OpenPass {
    /// The colour attachment's shape, which the viewport defaults to and the
    /// pipeline's sample count is taken from.
    target: pass::Attachment,
    /// The recipe the frame selected, if it has selected one.
    pipeline: Option<object::RasterPipeline>,
    /// The binding set the frame resolved, with the artifact it was resolved
    /// *for*: the kernel is kept beside the slots because the slots alone cannot
    /// answer whether they belong to the recipe installed at the draw, and a
    /// pipeline installed after a set is the one way the two can disagree.
    /// Replaced rather than accumulated: a fixed artifact declares one set, so a
    /// second one is a different recipe's.
    bindings: Option<(RasterKernel, Vec<BindingSlot>)>,
    /// The vertex buffers the frame bound, by slot.
    vertex: Vec<GlVertexBufferBinding>,
    /// The index buffer the frame bound.
    index: Option<GlIndexBinding>,
    /// The viewport, defaulting to the whole attachment.
    viewport: GlViewport,
    /// The scissor, absent until the frame sets one.
    scissor: Option<GlScissorRect>,
    /// Framebuffers and programs whose ownership Layer 2 handed to this encoder.
    /// Destroyed when the pass ends, because a caller-owned object is one no
    /// cache will ever free.
    owned_framebuffers: Vec<FramebufferId>,
    owned_programs: Vec<ProgramId>,
}

impl GlEncoder {
    /// The recipe this pass has selected, refusing when it has none.
    ///
    /// Asked before the commit rather than after it, so that a draw issued
    /// through the wrong verb of the pair -- a non-indexed draw of an indexed
    /// artifact, say -- is refused before anything is installed for it.
    fn kernel(&self, operation: &'static str) -> Result<RasterKernel, GlError> {
        self.pass
            .as_ref()
            .and_then(|pass| pass.pipeline.as_ref())
            .map(object::RasterPipeline::kernel)
            .ok_or_else(|| {
                malformed(
                    operation,
                    "no raster pipeline is installed in this pass, so there is nothing to draw with",
                )
            })
    }
}

/// One finished recording, ready to submit.
pub(crate) struct GlCommandBuffer {
    context: ContextStamp,
}

/// The reason the compute verbs refuse in this slice.
///
/// Narrowed from "every verb that would record a command": the raster pass and
/// its draws are issued now, and what is left is compute -- the dispatch, the
/// compute pipeline and the labels around them -- which is the next slice's
/// subject rather than this one's.  The reason does not name a slice number,
/// because it is a claim about the adapter and not about a schedule.
const NO_COMMAND_VOCABULARY: &str =
    "this adapter records no compute command yet, so the request cannot be made true in the driver";

/// A GL-family state machine presented as a common execution backend.
pub(crate) struct GlCompatibilityDevice<B: GlStateBackend> {
    machine: GlStateMachine<B>,
    identity: DeviceIdentityMap,
    capabilities: DeviceCapabilities,
    releases: Rc<ReleaseQueue>,
    submissions: SubmissionLedger,
    /// What each texture this adapter created was created as.
    ///
    /// Layer 1 has no query that answers a texture's format or extent -- an
    /// identity names an object rather than describing one -- and a pass
    /// descriptor carries only identities and ranges, so a pass cannot be
    /// lowered without this.  Keyed by the identity the creation returned and
    /// dropped where the object is destroyed, so an identity reused after a
    /// deletion cannot resolve to its predecessor's shape.
    attachments: HashMap<TextureId, pass::Attachment>,
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
            attachments: HashMap::new(),
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
        // The attachment records describe objects of the superseded generation,
        // which the restored context has already invalidated: keeping them would
        // let an identity minted by the new generation resolve to a shape that
        // belonged to the old one.
        self.attachments.clear();
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
                    // Dropped before the object is destroyed, so that a slot
                    // reused afterwards cannot be described by the shape of its
                    // predecessor.
                    self.attachments.remove(&texture);
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

    /// The object registry a frame recorded through this adapter resolves
    /// against.
    ///
    /// It captures three facts rather than borrowing them, and each has to be
    /// captured for the same reason: a provider is reached through `&self` while
    /// the recorder holds `&mut` on this adapter, so a registry that read them
    /// per call would be a second borrow of the machine at the moment it is
    /// already borrowed -- F1's self-deadlock finding, answered by construction
    /// rather than by a lock.
    ///
    /// - the **device identity**, so that a frame's own cross-device check is
    ///   against the generation the registry was built for.  A context
    ///   restoration reallocates it ([`Self::refresh`]), and a registry left over
    ///   from before one therefore reports the superseded generation and the
    ///   resolver refuses its objects -- a staleness the renderer can see rather
    ///   than a bind against a dead epoch.
    /// - the **profile**, because it is what the recipes are lowered for.
    /// - the **release queue**, because the objects it collects are destroyed at
    ///   this adapter's next `&mut self` entry point, and a queue of the
    ///   registry's own would be one nothing ever drained.
    ///
    /// A renderer rebuilds this whenever it re-reads capabilities, which is the
    /// same signal: a context whose generation changed is one whose registered
    /// objects were lowered for a context that no longer exists.
    pub(crate) fn object_registry(&mut self) -> GlObjectRegistry<B> {
        GlObjectRegistry::new(
            self.identity.identity(),
            self.machine.backend().profile(),
            Rc::clone(&self.releases),
        )
    }

    /// Accepts one semantic transition after checking the two things that can be
    /// wrong about it at this boundary, and issues nothing.
    ///
    /// The contract asks two things of a transition, and this family satisfies
    /// both without a command -- which is a claim that has to be shown and not
    /// asserted, so here is each half.
    ///
    /// The *state change* half has nothing to change.  A GL-family texture or
    /// buffer has no access state of its own: what a resource is being used for
    /// is a property of the call that uses it, decided by the binding the Layer 2
    /// mirror reconciles at that call, not a mode the resource is put into.  The
    /// graph is telling this backend about a hazard it will not have, because the
    /// hazard exists for backends whose resources do carry state -- the native
    /// ones track it per subresource and lower each transition to a real barrier
    /// (`imp/command/copy.rs`).
    ///
    /// The *memory barrier* half, including the `before == after` case the
    /// contract calls out, is carried by the execution model rather than by a
    /// command.  This adapter declares
    /// [`SynchronizationCapabilities::SingleQueueOrdering`]: one immediate context
    /// on one queue, where commands are issued in order and a write is visible to
    /// a later read, and where two submissions are ordered by `flush` and the
    /// fence [`Self::submit`] inserts between them.  There is no barrier to emit
    /// for texture or buffer access in this family -- Layer 1's only barrier verb
    /// is the compute domain's, documented as a shader-storage coherency barrier
    /// that WebGL2 providers do not implement (`api/compute.rs`), and it serves a
    /// different hazard than this one.
    ///
    /// So the verb's real work here is the check that the transition names
    /// resources and an encoder of the *current* context generation: a
    /// transition carrying a superseded identity is exactly the stale-plan
    /// mistake the common contract wants rejected locally, and it is the one
    /// thing about a transition this backend can determine to be wrong.
    ///
    /// Accepting a range and not consulting it is deliberate.  The range narrows
    /// where a barrier applies, and there is no barrier; a version of this verb
    /// that consulted the range would have to invent an operation to justify the
    /// narrowing, which is the same mistake as refusing the transition for the
    /// wrong reason.  That includes the aspect a `TextureRange::Subresources`
    /// carries: with no barrier there is nothing for an aspect to select, and an
    /// aspect is a claim about a resource the same way the copy region's would
    /// be -- see [`region`] for why this adapter does not make one.
    fn accept_transition(
        &mut self,
        operation: &'static str,
        encoder: ContextStamp,
        object: ContextStamp,
    ) -> Result<(), GlError> {
        self.refresh();
        let backend = self.machine.backend();
        backend.validate_object_context(operation, encoder)?;
        backend.validate_object_context(operation, object)
    }

    /// The pass an encoder has open, or the refusal that it has none.
    ///
    /// Every verb that records into a pass reads it through here, so that the
    /// four of them report the same mistake the same way and none of them can
    /// reach the state without asking.
    fn open_pass<'e>(
        &mut self,
        encoder: &'e mut GlEncoder,
        operation: &'static str,
    ) -> Result<&'e mut OpenPass, GlError> {
        encoder.pass.as_mut().ok_or_else(|| no_pass(operation))
    }

    /// Drives Layer 2 so the backend holds everything the pass's draw needs.
    ///
    /// Called at the draw and not at each verb, because a GL-family pipeline
    /// carries its program, its vertex array and its rasterization state as one
    /// value while the contract supplies those three in the order a
    /// pipeline-first API supplies them -- so the pieces can only be assembled
    /// once all of them have arrived.  See the module documentation.
    ///
    /// The order inside is the one Layer 2 documents, and each step is here for a
    /// reason rather than by convention:
    ///
    /// 1. **The program**, because `program_for` links when its cache misses and a
    ///    link invalidates the mirror's installed pipeline -- so it has to happen
    ///    before the install below rather than after it, or the install would be
    ///    recorded against a pipeline the link already forgot.
    /// 2. **The vertex input**, because the pipeline has to *name* the array, and
    ///    the identity comes from the geometry domain's derivation.
    /// 3. **The pipeline**, which installs that array -- and binds it *without*
    ///    re-emitting its attribute description, which is why the geometry domain
    ///    drops its claim here.
    /// 4. **The vertex input again**, which is the only thing that re-enables the
    ///    attributes after that install.
    /// 5. **The bindings**, one at a time, because the texture domain's request is
    ///    single-valued: `active_texture`, `bind_texture` and `bind_sampler` each
    ///    *overwrite* the previous request, so a whole set recorded and applied
    ///    once would bind only its last unit.
    fn commit(&mut self, operation: &'static str, encoder: &mut GlEncoder) -> Result<(), GlError> {
        let Some(pass) = encoder.pass.as_mut() else {
            return Err(no_pass(operation));
        };
        let recipe = pass.pipeline.clone().ok_or_else(|| {
            malformed(
                operation,
                "no raster pipeline is installed in this pass, so there is nothing to draw with",
            )
        })?;
        if let Some((resolved_for, _)) = pass.bindings.as_ref() {
            // A pipeline installed *after* the set is the one way the two can
            // disagree: `set_bindings` checks the set against the pipeline that
            // was installed when it was called, and this is the check that holds
            // for the pipeline installed now.
            if *resolved_for != recipe.kernel() {
                return Err(malformed(
                    operation,
                    "the binding set in this pass was resolved for a different artifact than the one installed at the draw",
                ));
            }
        }
        let input = pass::vertex_input(&recipe, &pass.vertex, pass.index, operation)?;

        let (program, _reflection, owned) = self
            .machine
            .program_for(recipe.descriptor())
            .map_err(failure::into_gl_error)?;
        if owned {
            pass.owned_programs.push(program);
        }

        self.machine.set_vertex_input(input);
        let _ = self
            .machine
            .apply_geometry()
            .map_err(failure::into_gl_error)?;
        let vertex_array = self
            .machine
            .geometry()
            .applied_vertex_array()
            .ok_or_else(|| {
                malformed(
                    operation,
                    "the vertex input was bound without naming a vertex array",
                )
            })?;

        self.machine.set_pipeline(&GlRasterPipeline {
            program,
            vertex_array,
            state: pass::raster_state(
                recipe.kernel(),
                pass.viewport,
                pass.scissor,
                pass.target.sample_count(),
            ),
        });
        let _ = self
            .machine
            .apply_pipeline()
            .map_err(failure::into_gl_error)?;

        let _ = self
            .machine
            .apply_geometry()
            .map_err(failure::into_gl_error)?;
        match pass.bindings.as_ref() {
            Some((_, slots)) => self.bind_slots(operation, slots),
            // No set was recorded, and that is refused rather than read as an
            // empty one.  The set is also what names the artifact its resources
            // were resolved for -- [`Self::set_bindings`] is where that recipe
            // check is made -- so a draw without one is a draw whose bindings were
            // never checked against the pipeline installed after them.  It is not
            // a shape a frame can produce: a renderer resolves a set per draw
            // whatever the artifact reads, and the bare triangle's is empty and is
            // still recorded.
            None => Err(malformed(
                operation,
                "no binding set is recorded in this pass, and the set is what names the artifact its resources were resolved for",
            )),
        }
    }

    /// Binds one registered set, one logical binding at a time.
    ///
    /// Each binding is applied as soon as it is recorded, for the reason step 5
    /// of [`Self::commit`] gives: the texture domain's request is single-valued,
    /// so applying per binding is the only order in which a set of more than one
    /// arrives intact.
    fn bind_slots(
        &mut self,
        operation: &'static str,
        slots: &[BindingSlot],
    ) -> Result<(), GlError> {
        for slot in slots {
            match *slot {
                BindingSlot::Uniform {
                    binding,
                    buffer,
                    range,
                } => {
                    let (offset, size) = uniform_range(operation, range)?;
                    self.machine
                        .bind_uniform_buffer(binding, Some(buffer), offset, size);
                    self.machine
                        .apply_buffers()
                        .map_err(failure::into_gl_error)?;
                }
                BindingSlot::Texture {
                    binding,
                    texture,
                    range,
                } => {
                    // This family has no texture view, so a sampled binding can
                    // only name a whole texture.  Lowering a subresource range
                    // would mean sampling the wrong mips, which is a wrong image
                    // rather than a missing feature.
                    if !matches!(range, TextureRange::Whole) {
                        return Err(unsupported(
                            operation,
                            "this family has no texture view, so a sampled binding names a whole texture",
                        ));
                    }
                    let facts = *self.attachments.get(&texture).ok_or_else(|| {
                        malformed(
                            operation,
                            "a binding names a texture this device did not create",
                        )
                    })?;
                    let target = facts.target()?;
                    self.machine.bind_texture(binding, target, Some(texture));
                    self.machine
                        .apply_textures()
                        .map_err(failure::into_gl_error)?;
                }
            }
        }
        Ok(())
    }

    /// Destroys the objects a pass came to own, reporting the first failure.
    ///
    /// Every object is attempted even after one fails: they are already
    /// unreachable -- the pass that derived them is the last thing that named
    /// them -- so stopping at the first would leave the rest with no name at all.
    /// The failure is reported rather than counted, unlike the domains' own
    /// cleanup, because this is a verb's error path and not a silent pass over a
    /// cache.
    fn destroy_owned(&mut self, pass: OpenPass) -> Result<(), GlError> {
        let mut first: Option<GlError> = None;
        for framebuffer in pass.owned_framebuffers {
            if let Err(error) = self.machine.backend().destroy_framebuffer(framebuffer) {
                first.get_or_insert(error);
            }
        }
        for program in pass.owned_programs {
            if let Err(error) = self.machine.backend().destroy_program(program) {
                first.get_or_insert(error);
            }
        }
        match first {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

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

/// The byte range a uniform binding point is given, from the range the graph
/// authorized.
///
/// `Whole` becomes `(0, 0)`, which is Layer 1's own spelling of "from the offset
/// through the end of the allocation" and not a range of no bytes.  An explicit
/// range wider than an indexed binding point can express is refused: this
/// family's offset and size are 32-bit, and truncating a 64-bit one would bind a
/// range the graph never authorized.
fn uniform_range(operation: &'static str, range: BufferRange) -> Result<(u32, u32), GlError> {
    let (offset, size) = match range {
        BufferRange::Whole => (0, 0),
        BufferRange::Bytes { offset, size } => (
            u32::try_from(offset).map_err(|_| {
                malformed(
                    operation,
                    "the authorized offset is beyond what an indexed binding point can address",
                )
            })?,
            u32::try_from(size).map_err(|_| {
                malformed(
                    operation,
                    "the authorized size is beyond what an indexed binding point can address",
                )
            })?,
        ),
    };
    Ok((offset, size))
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

impl<B: GlStateBackend> ExecutionBackend for GlCompatibilityDevice<B> {
    type Texture = TextureId;
    type Buffer = BufferId;
    type RasterPipeline = object::RasterPipeline;
    type ComputePipeline = UnsupportedComputePipeline;
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
        let physical = self.machine.backend().create_buffer_resource(lowered)?;
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

        let mut open = OpenPass {
            target: facts,
            pipeline: None,
            bindings: None,
            vertex: Vec::new(),
            index: None,
            // The attachment's own extent, so that a pass whose frame sets no
            // viewport renders into the whole of what it attached rather than
            // into whatever the previous pass left selected.
            viewport: pass::whole_extent(facts),
            scissor: None,
            owned_framebuffers: Vec::new(),
            owned_programs: Vec::new(),
        };
        if owned {
            open.owned_framebuffers.push(framebuffer);
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
        let Some(pass) = encoder.pass.take() else {
            return Err(no_pass("end-raster"));
        };
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
        // The recorded bindings are *kept*, and checked against this recipe
        // rather than dropped with the previous one: they are facts about what
        // the frame resolved, and a recipe that does not read them is a mistake
        // worth reporting at the bind that made it rather than a silent discard.
        pass.pipeline = Some(pipeline.clone());
        Ok(())
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

    /// Records the set this pass draws with.
    ///
    /// The recipe check is made here as well as at the draw, and the two are not
    /// the same check: this one catches a set resolved for another artifact at
    /// the call that got it wrong, while [`Self::commit`] catches a pipeline
    /// installed *after* the set, which would otherwise bind resources at the
    /// wrong numbers.
    fn set_bindings(
        &mut self,
        encoder: &mut Self::Encoder,
        bindings: &Self::Bindings,
    ) -> Result<(), Self::Error> {
        let pass = self.open_pass(encoder, "set-bindings")?;
        let pipeline = pass.pipeline.as_ref().ok_or_else(|| {
            malformed(
                "set-bindings",
                "a binding set is resolved for the artifact that reads it, and no raster pipeline is installed in this pass",
            )
        })?;
        if bindings.kernel() != pipeline.kernel() {
            return Err(malformed(
                "set-bindings",
                "the binding set was resolved for a different artifact than the one installed in this pass",
            ));
        }
        pass.bindings = Some((bindings.kernel(), bindings.slots().to_vec()));
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
        let pass = self.open_pass(encoder, "set-vertex-buffer")?;
        pass.vertex.retain(|bound| bound.slot != slot);
        pass.vertex.push(GlVertexBufferBinding {
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
        let pass = self.open_pass(encoder, "set-index-buffer")?;
        pass.index = Some(GlIndexBinding {
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
        let pass = self.open_pass(encoder, "set-viewport")?;
        pass.viewport = pass::viewport(viewport);
        Ok(())
    }

    fn set_scissor(
        &mut self,
        encoder: &mut Self::Encoder,
        scissor: ScissorRect,
    ) -> Result<(), Self::Error> {
        let pass = self.open_pass(encoder, "set-scissor")?;
        pass.scissor = Some(pass::scissor(scissor));
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
        let mut encoder = encoder;
        if let Some(abandoned) = encoder.pass.take() {
            let _ = self.machine.end_pass();
            let _ = self.destroy_owned(abandoned);
            return Err(malformed(
                "finish-encoder",
                "a raster pass was left open on this encoder, and a command buffer cannot be finished inside one",
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
