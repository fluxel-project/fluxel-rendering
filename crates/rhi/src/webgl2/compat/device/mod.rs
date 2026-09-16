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
//! Two modules hold the machinery the verbs drive, and they are split along
//! the same line -- one decision each:
//!
//! - [`encoder`] is one recording in progress: what it holds while a pass is
//!   open, and what the device does with it -- admitting a pass, committing at
//!   the draw, applying a set one binding at a time, and destroying what a pass
//!   came to own.
//! - [`backend`] is the contract itself: the `ExecutionBackend` impl, one method
//!   per verb, in the trait's own declaration order.
//!
//! The verbs are not split out of [`backend`] by family, and the reason is the
//! language rather than the file size: a trait has one impl block per type
//! (E0119), so the families a reader might expect as separate files -- resources
//! and transients, passes and draws, submission -- are one `impl` in one file.
//! What *is* separable is the machinery the verbs use, which is [`encoder`] --
//! and the addressing and retention half of what they compute, which are already
//! [`transient`], [`region`] and [`retention`].
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

mod backend;
mod encoder;
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
use std::rc::Rc;

use fluxel_rendergraph::DeviceCapabilities;

use crate::webgl2::api::{ContextStamp, GlError, TextureId};
use crate::webgl2::state::{GlStateBackend, GlStateMachine, StateEvent};

use super::identity::DeviceIdentityMap;
use registry::GlObjectRegistry;
use retention::{ReleaseQueue, RetainedObject};
use submission::SubmissionLedger;

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
}
