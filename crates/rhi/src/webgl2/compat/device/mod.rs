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
//! - [`upload`] and [`readback`] are how host bytes cross this boundary in each
//!   direction -- into a buffer or a texture this device created, and back out of
//!   one -- which is the half of a caller-owned resource the contract has no verb
//!   for either way.  [`transfer`] is the rectangle and the client encoding both
//!   directions share, so that the shape a frame's picture is written with and
//!   the shape it is read back with cannot drift apart.
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
//! [`compute`] is the one module that is neither a lowering nor a recording, and
//! it exists because this family's compute domain is *optional*: it holds the
//! witness type that says whether a given adapter has one, and the two refusals a
//! compute verb can give -- the backend has no such domain, or the context's
//! discovery snapshot did not prove the capability.  Its own documentation argues
//! why that is a type parameter and not a flag.
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
//! A **compute pass and its dispatch** are real too, and only on an adapter whose
//! witness says so -- one whose backend is a [`GlOptionalComputeBackend`], which
//! the browser provider is not.  Layer 1 declares no pass boundary for compute in
//! this family, so the brackets record nothing and the **dispatch is the commit
//! point**, on the draw's terms and for the draw's reason.  See [`compute`].
//!
//! [`GlOptionalComputeBackend`]: crate::webgl2::state::GlOptionalComputeBackend
//!
//! A **presentation token** exists, and the acquisition and presentation slices
//! put it between them.  It used to be an uninhabited type and that was stronger
//! than a check; it cannot stay one, because a type that carries an acquisition
//! to submission has to hold the acquisition, and the acquisition is a lease and
//! a texture.  What replaces the old guarantee is narrow and worth stating
//! exactly: a token can only be minted by the acquisition verb, that verb refuses
//! fail-closed while `DeviceCapabilities::surface` is `None`, and the
//! advertisement is filled from the observed drawable rather than asserted -- so
//! the reachable states are "no surface, no token" and "a surface, and a token
//! whose lease came out of the drawable".  [`super::capabilities`] is where the
//! advertisement is made from the observed drawable, and the answer it gives for
//! a drawable it could not read is the `None` this verb refuses on.
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
mod compute;
mod encoder;
mod failure;
// The measurement entry lives here rather than beside `webgl2::conformance`
// because of module privacy and not preference: `mod device` is private inside
// `compat`, so the adapter and its verbs are nameable only from inside this
// module.  It is `pub(crate)` so that `compat/mod.rs` can hand it to
// `test_support`, which is where it becomes reachable at all.
// The gate names both native surfaces rather than the Windows one alone.  What
// lives here is the report vocabulary and the request parsing every native
// entry shares, and none of it is WGL-shaped; only `drive_desktop_gl4_draws`
// is, and it carries that gate itself.  Leaving the module on the Windows gate
// would have locked the shared vocabulary behind one surface, which is the same
// defect `webgl2::conformance` had and is fixed the same way.
#[cfg(all(
    feature = "test-support",
    any(
        all(target_os = "windows", feature = "native-gl-wgl"),
        all(not(target_arch = "wasm32"), feature = "native-gles-egl")
    )
))]
pub(crate) mod harness;
mod object;
mod pass;
mod raster;
mod readback;
mod region;
mod registry;
mod retention;
mod submission;
mod surface;
mod transfer;
mod transient;
mod upload;
// The workload both measurement surfaces drive.  It is here rather than beside
// either of them because it is a fact about the adapter and the verbs and not
// about WGL or a canvas, and both callers are inside this module tree: the
// desktop entry beside it, and -- through the `pub(crate)` re-export in
// `compat/mod.rs` -- the browser draw test, which lives above the layers because
// `compat` may not name a browser type and `api/browser` may not name `compat`.
// `pub(crate)` for the same reason `harness` is: without it that re-export
// cannot path through a private module.
pub(crate) mod workload;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;

use fluxel_rendergraph::DeviceCapabilities;

use crate::webgl2::api::{BufferId, ContextStamp, GlError, TextureId};
use crate::webgl2::state::{
    ExecutionMode, GlStateBackend, GlStateMachine, StateCounters, StateEvent,
};

use super::identity::DeviceIdentityMap;
use compute::{ComputeDomain, NoCompute};
use registry::GlObjectRegistry;
use retention::{ReleaseQueue, RetainedObject};
use submission::SubmissionLedger;
use surface::GlSurfaceToken;

/// A GL-family state machine presented as a common execution backend.
///
/// The second parameter is the *witness* for the optional compute domain, and it
/// defaults to the one that refuses: a caller that names no witness gets the
/// adapter whose compute verbs all refuse, which is the shape every provider
/// without a compute command domain needs and the fail-closed one for a provider
/// that has it.  [`compute`] argues why the choice is a type rather than a flag;
/// the short version is that a method cannot be stricter than its impl's own
/// bounds (E0276) and a trait has one impl per type (E0119), so the optional
/// bound has nowhere else to live.
///
/// `C` occupies no storage, and the marker is `PhantomData<fn() -> C>` rather
/// than `PhantomData<C>` on purpose: the adapter neither owns nor borrows a `C`,
/// it only *calls* one, and `fn() -> C` is the marker that says exactly that --
/// covariant in `C`, and carrying no drop-check or auto-trait obligation that a
/// witness type should not be able to impose on the device.
pub(crate) struct GlCompatibilityDevice<B: GlStateBackend, C: ComputeDomain<B> = NoCompute> {
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
    /// How many bytes each buffer this adapter created was created with.
    ///
    /// The same record as `attachments` for the same reason, and it exists for
    /// one caller: a storage binding range that the graph authorized as *whole*
    /// has to be lowered to a real size, because Layer 1 validates a storage
    /// range against the allocation while an indexed uniform range spells "to
    /// the end" as zero.  Kept and dropped beside `attachments`, so an identity
    /// reused after a deletion cannot resolve to its predecessor's size.
    buffers: HashMap<BufferId, u64>,
    /// The compute witness, which carries the bound rather than a value.  See
    /// the type's own documentation.
    witness: PhantomData<fn() -> C>,
}

impl<B: GlStateBackend, C: ComputeDomain<B>> GlCompatibilityDevice<B, C> {
    /// The adapter over `backend`.
    ///
    /// The capability description and the context stamp are read from the
    /// backend *before* the machine takes it, because both are facts about the
    /// context rather than about the mirror: they come from the discovery
    /// snapshot, which the machine does not consult.
    pub(crate) fn new(backend: B) -> Self {
        Self::with_mode(backend, ExecutionMode::Optimized)
    }

    /// The adapter over `backend`, running Layer 2 in `mode`.
    ///
    /// The mode is a parameter here rather than a flag on the machine because it
    /// is fixed for the machine's whole life: it decides whether a redundant call
    /// may be skipped, and a machine that changed its mind part-way through would
    /// have a trace whose middle was filtered by one rule and whose end by
    /// another.  So the only way to run the same frame both ways is to build two
    /// adapters, which is what a cached-versus-uncached differential does.
    ///
    /// It is `pub(crate)` and not public on purpose.  Which mode a *renderer*
    /// runs is not a choice the common contract offers -- an optimized adapter is
    /// the only production adapter -- so this exists for the differential and for
    /// nothing else.  `new` is the same call with [`ExecutionMode::Optimized`].
    pub(crate) fn with_mode(backend: B, mode: ExecutionMode) -> Self {
        let stamp = backend.context_stamp();
        let capabilities = super::capabilities::capabilities(backend.discovery());
        Self {
            machine: GlStateMachine::with_mode(backend, mode),
            identity: DeviceIdentityMap::new(stamp),
            capabilities,
            releases: ReleaseQueue::new(),
            submissions: SubmissionLedger::default(),
            attachments: HashMap::new(),
            buffers: HashMap::new(),
            witness: PhantomData,
        }
    }

    /// The mode this adapter's machine was built with.
    pub(crate) fn execution_mode(&self) -> ExecutionMode {
        self.machine.mode()
    }

    /// What the machine has emitted, skipped and cached so far.
    ///
    /// Read-only, and deliberately not a `counters_mut`: every counter is the
    /// machine's own tally of what it did, so a caller that could write one could
    /// make the tally disagree with the trace it is supposed to describe.
    ///
    /// This is the *only* place a caller outside `compat::device` can reach these
    /// numbers, and it exists because the differential needs them: a candidate is
    /// accepted on emitted calls falling as predicted, and that prediction is
    /// checked against this, not against a log.
    pub(crate) fn counters(&self) -> &StateCounters {
        self.machine.counters()
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
        // The attachment and allocation records describe objects of the
        // superseded generation, which the restored context has already
        // invalidated: keeping them would let an identity minted by the new
        // generation resolve to a shape that belonged to the old one.
        self.attachments.clear();
        self.buffers.clear();
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
                    // Dropped for the texture arm's reason, and here it matters
                    // twice over: the record is what a storage binding is lowered
                    // against, so a stale one would give a later binding of a
                    // reused identity a size its predecessor had.
                    self.buffers.remove(&buffer);
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
    pub(crate) fn object_registry(&mut self) -> GlObjectRegistry<B, C> {
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
