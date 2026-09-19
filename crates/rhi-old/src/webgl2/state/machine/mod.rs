//! The state machine: one backend, one mirror per domain, and the order the
//! domains run in.
//!
//! This module is the integration point and nothing else.  It owns the backend,
//! the execution mode, the counters and the domain fields; it decides *when*
//! each domain reconciles and what else a transition implies; and it does not
//! know how any domain makes the driver match.  A domain module can be read,
//! tested and changed without this file, which is the property that lets one
//! file per domain stay honest.
//!
//! # The domain contract
//!
//! Every state domain is a module under this one declaring a `<Domain>State`
//! with the same four things.  A new domain is added by writing those four and a
//! field here; nothing about the existing domains changes.
//!
//! ```text
//! struct <Domain>State { desired: ..., applied: ..., <its own caches> }
//!
//! fn new(...) -> Self
//!     The initial state, which is always "the driver holds nothing this layer
//!     put there" -- never a belief about GL defaults.
//!
//! fn <entry>(&mut self, ...)
//!     Records what the caller wants.  Emits nothing: the redundancy decision
//!     belongs in reconcile, so it is made in exactly one place.
//!
//! fn reconcile(&mut self, backend: &mut impl GlStateBackend, counters: &mut StateCounters)
//!     -> Result<..., StateError>
//!     Emits whatever calls are needed to make the applied state agree with the
//!     desired state, counts a request and either an emit or a skip, and leaves
//!     the applied state describing the driver *after* the call.  A failure
//!     leaves this domain's applied state unknown rather than unchanged.
//!
//! fn invalidate(&mut self, backend: &mut impl GlStateBackend, event: &StateEvent,
//!               counters: &mut StateCounters)
//!     Reacts to the events the domain owns and ignores the rest.  The matrix in
//!     [`super::event`] is the list of which events a domain owns.
//! ```
//!
//! Three rules keep the domains from reaching into each other.  A domain never
//! calls another domain: where one domain's transition implies another's
//! invalidation, the first reports it in its reconcile result and the machine
//! applies it -- [`SessionEffects`] is the worked example.  A domain never
//! re-validates what Layer 1 already rejects, because two definitions of
//! validity drift and the cheaper one wins.  And no domain may reach the
//! backend except through the parameters it is handed, so a domain's tests can
//! run it against any provider without building a machine.
//!
//! # The order within one transition
//!
//! The domains do not run in parallel and their order is not arbitrary.  A draw
//! requires an open pass, an installed pipeline and bound geometry; a pass
//! requires its framebuffer; a group upload requires the buffer bindings it
//! expands into.  The machine therefore runs the domains in dependency order at
//! each entry point, and an entry point that cannot be satisfied fails before
//! any of the later domains emit.  That is what makes a rejected request
//! genuinely pre-side-effect rather than mostly so.
//!
//! # Teardown
//!
//! A derived object is a backend object this layer created, so the machine owns
//! destroying it.  [`GlStateMachine::shutdown`] drains every cache and destroys
//! what it holds, counting failures rather than raising them; there is
//! deliberately no `Drop` that does the same, because a `Drop` cannot report and
//! a silently-failing cleanup is indistinguishable from a leak.  The distinction
//! between draining and purging is the one [`super::cache`] documents: shutdown
//! destroys through live identities, an epoch change destroys nothing.

use crate::webgl2::api::{
    BufferId, FramebufferId, GlBufferRange, GlFramebufferDescriptor, GlIndexBinding,
    GlProgramDescriptor, GlProgramReflection, GlRasterPipeline, GlRenderPassDescriptor,
    GlStorageBufferApi, GlStorageBufferRange, GlStorageImageBinding, GlTextureTarget,
    GlVertexBufferBinding, GlVertexLayout, ProgramId, SamplerId, TextureId,
};

use super::GlStateBackend;
use super::backend::GlOptionalComputeBackend;
use super::buffers::BuffersState;
use super::cache::DEFAULT_BUDGET;
use super::compute::ComputeState;
use super::counters::StateCounters;
use super::error::StateError;
use super::event::StateEvent;
use super::geometry::{GeometryEffects, GeometryState, VertexInput};
use super::knowledge::ExecutionMode;
use super::pipeline::{PipelineEffects, PipelineState};
use super::session::{SessionEffects, SessionState};
use super::textures::TexturesState;

/// A backend plus the mirror of what it holds.
#[derive(Debug)]
pub(crate) struct GlStateMachine<B: GlStateBackend> {
    backend: B,
    mode: ExecutionMode,
    counters: StateCounters,
    session: SessionState,
    pipeline: PipelineState,
    geometry: GeometryState,
    textures: TexturesState,
    buffers: BuffersState,
    compute: ComputeState,
}

impl<B: GlStateBackend> GlStateMachine<B> {
    /// A machine over `backend` that has applied nothing yet.
    ///
    /// Every domain starts with an empty applied state, which means the first
    /// request for each of them emits the call that establishes it.  That is
    /// deliberate and it is the reason this layer never has to assume a GL
    /// default: the cost is one redundant call per domain per context, paid
    /// once, in exchange for a mirror that cannot be wrong about a value it
    /// never set.
    pub(crate) fn new(backend: B) -> Self {
        Self::with_mode(backend, ExecutionMode::Optimized)
    }

    /// A machine that runs `mode` for its whole life.
    ///
    /// The mode is fixed at construction because the alternative -- switching
    /// mid-life -- would have to decide what happens to the derived objects the
    /// previous mode retained.  Destroying them would make the switch a
    /// teardown, and keeping them would leave the oracle running against a
    /// cache the optimized trace filled, which is exactly the comparison the
    /// differential test must not make.  The differential test therefore builds
    /// two machines over two backends and compares their traces.
    pub(crate) fn with_mode(backend: B, mode: ExecutionMode) -> Self {
        Self {
            backend,
            mode,
            counters: StateCounters::default(),
            // Every domain is constructed with the *mode* rather than the cache
            // policy it implies, because the two are separate decisions: the
            // retention half reaches a domain's caches, and the skipping half
            // decides whether a call it can prove redundant is emitted.  A domain
            // given only retention would skip unconditionally, and an oracle
            // machine that skipped would emit a trace no mirror-free machine could
            // have produced -- which is the comparison it exists for.
            session: SessionState::new(DEFAULT_BUDGET, mode),
            pipeline: PipelineState::new(DEFAULT_BUDGET, mode),
            geometry: GeometryState::new(DEFAULT_BUDGET, mode),
            textures: TexturesState::new(mode),
            buffers: BuffersState::new(mode),
            compute: ComputeState::new(mode),
        }
    }

    /// The backend, for the object lifetime calls this layer does not mirror.
    pub(crate) fn backend(&mut self) -> &mut B {
        &mut self.backend
    }

    /// The execution mode this machine runs.
    pub(crate) const fn mode(&self) -> ExecutionMode {
        self.mode
    }

    /// The counters, for a report or a test.
    pub(crate) const fn counters(&self) -> &StateCounters {
        &self.counters
    }

    /// The counters, mutably, for a test that needs a clean slate.
    pub(crate) fn counters_mut(&mut self) -> &mut StateCounters {
        &mut self.counters
    }

    /// The session domain, for the pass boundary and the framebuffer records.
    pub(crate) fn session(&mut self) -> &mut SessionState {
        &mut self.session
    }

    /// The pipeline domain, for the installed pipeline and the program records.
    pub(crate) fn pipeline(&mut self) -> &mut PipelineState {
        &mut self.pipeline
    }

    /// The geometry domain, for the bound vertex input and the vertex arrays.
    pub(crate) fn geometry(&mut self) -> &mut GeometryState {
        &mut self.geometry
    }

    /// The texture domain, for the active unit and the per-unit bindings.
    pub(crate) fn textures(&mut self) -> &mut TexturesState {
        &mut self.textures
    }

    /// The buffer domain, for the indexed uniform and storage binding points.
    pub(crate) fn buffers(&mut self) -> &mut BuffersState {
        &mut self.buffers
    }

    /// Records that the caller wants this pass open, and applies the boundary.
    pub(crate) fn begin_pass(
        &mut self,
        descriptor: GlRenderPassDescriptor,
    ) -> Result<SessionEffects, StateError> {
        self.session.begin_pass(descriptor);
        let effects = self
            .session
            .reconcile(&mut self.backend, &mut self.counters)?;
        self.apply_session_effects(effects);
        Ok(effects)
    }

    /// Records that the caller wants no pass open, and applies the boundary.
    ///
    /// The effects are reported *and* applied: Layer 1's `end_render_pass`
    /// forgets the installed pipeline, so a mirror that kept believing a pipeline
    /// was installed would skip the `set_raster_pipeline` the next draw needs.
    /// The caller still gets the value, because a caller that has to re-establish
    /// a pipeline of its own may want to know that this is why.
    pub(crate) fn end_pass(&mut self) -> Result<SessionEffects, StateError> {
        self.session.end_pass();
        let effects = self
            .session
            .reconcile(&mut self.backend, &mut self.counters)?;
        self.apply_session_effects(effects);
        Ok(effects)
    }

    /// Applies the consequences a pass boundary has for the other domains.
    ///
    /// A domain never calls another domain: the one that knows the consequence
    /// reports it, and the machine applies it.  A pass that ended is the worked
    /// case -- Layer 1's `end_render_pass` forgets the installed pipeline, so the
    /// pipeline mirror must forget it too or the next draw skips the install it
    /// needs.
    ///
    /// A pass that *began* has no consequence here.  Layer 1's
    /// `begin_render_pass` binds a framebuffer and issues clears, and neither act
    /// reaches a value any other domain mirrors.
    fn apply_session_effects(&mut self, effects: SessionEffects) {
        if effects.pass_ended {
            self.pipeline.pass_ended();
        }
    }

    /// Returns the framebuffer for an attachment set, building it if needed.
    ///
    /// The second half of the pair is whether the caller owns the result: a
    /// framebuffer the cache could not retain -- because the cache is disabled
    /// or because its budget could not be met -- is still perfectly usable, and
    /// the caller destroys it when the pass is done.
    /// See [`super::session::framebuffer::FramebufferCache::framebuffer_for`].
    pub(crate) fn framebuffer_for(
        &mut self,
        descriptor: &GlFramebufferDescriptor,
    ) -> Result<(FramebufferId, bool), StateError> {
        self.session
            .framebuffer_for(&mut self.backend, descriptor, &mut self.counters)
    }

    /// Records that the caller wants `pipeline` installed.
    pub(crate) fn set_pipeline(&mut self, pipeline: &GlRasterPipeline) {
        self.pipeline.set_pipeline(pipeline);
    }

    /// Installs the desired pipeline, and applies what installing it implies.
    ///
    /// Layer 1's `set_raster_pipeline` binds the vertex array the pipeline names
    /// *without* re-emitting its attribute description, so the geometry domain's
    /// belief about which input is in force may no longer describe the driver --
    /// and the pipeline may have named an array this domain never derived.  The
    /// consequence is applied here rather than inside either domain, because a
    /// domain never calls another domain.
    ///
    /// The claim is dropped conservatively, even when the pipeline installed the
    /// very array the geometry domain derived, so the cost is one re-bind after
    /// each pipeline install.  Refining that would need the pipeline to report
    /// *which* array it installed, and [`PipelineEffects`] deliberately reports
    /// only that it did; the merge of two domains' beliefs is not a thing either
    /// domain can do alone, and a wrong guess here is a silently wrong image.
    pub(crate) fn apply_pipeline(&mut self) -> Result<PipelineEffects, StateError> {
        let effects = self
            .pipeline
            .reconcile(&mut self.backend, &mut self.counters)?;
        if effects.pipeline_installed {
            // Dropping the claim is a *release*: an array the cache could not
            // retain is named by nothing but this claim, so forgetting it here
            // would leak the object rather than merely forget where it is.
            self.geometry
                .vertex_input_unknown(&mut self.backend, &mut self.counters);
        }
        Ok(effects)
    }

    /// The program for a descriptor, linking it if needed.
    ///
    /// The second and third halves are the reflection Layer 1 recorded and
    /// whether the caller owns the program, on the same terms as
    /// [`GlStateMachine::framebuffer_for`].
    pub(crate) fn program_for(
        &mut self,
        descriptor: &GlProgramDescriptor,
    ) -> Result<(ProgramId, GlProgramReflection, bool), StateError> {
        self.pipeline
            .program_for(&mut self.backend, descriptor, &mut self.counters)
    }

    /// Records that the caller wants this vertex input bound.
    pub(crate) fn set_vertex_input(&mut self, input: VertexInput) {
        self.geometry.set_vertex_input(input);
    }

    /// Binds the desired vertex input.
    ///
    /// The effects report the two binding points Layer 1 leaves at values it does
    /// not promise; nothing in this layer mirrors either, so the machine records
    /// that they are unknown and the caller decides.  `#[must_use]` on
    /// [`GeometryEffects`] is what makes that decision explicit.
    pub(crate) fn apply_geometry(&mut self) -> Result<GeometryEffects, StateError> {
        self.geometry
            .reconcile(&mut self.backend, &mut self.counters)
    }

    /// Records that the caller wants `unit` to be the active texture unit.
    pub(crate) fn active_texture(&mut self, unit: u32) {
        self.textures.active_texture(unit);
    }

    /// Records that the caller wants `texture` bound at `unit` for `target`.
    pub(crate) fn bind_texture(
        &mut self,
        unit: u32,
        target: GlTextureTarget,
        texture: Option<TextureId>,
    ) {
        self.textures.bind_texture(unit, target, texture);
    }

    /// Records that the caller wants `sampler` bound at `unit`.
    pub(crate) fn bind_sampler(&mut self, unit: u32, sampler: Option<SamplerId>) {
        self.textures.bind_sampler(unit, sampler);
    }

    /// Applies the desired texture and sampler binding.
    pub(crate) fn apply_textures(&mut self) -> Result<(), StateError> {
        self.textures
            .reconcile(&mut self.backend, &mut self.counters)
    }

    /// Records that the caller wants `index` to hold `buffer`'s byte range.
    pub(crate) fn bind_uniform_buffer(
        &mut self,
        index: u32,
        buffer: Option<BufferId>,
        offset: u32,
        size: u32,
    ) {
        self.buffers
            .bind_uniform_buffer(index, buffer, offset, size, &mut self.counters);
    }

    /// Records that the caller wants `binding` to hold `range`.
    ///
    /// Gated on the optional bound for the reason the storage role exists at all:
    /// the verb it mirrors is not in the required profile, so a machine over a
    /// profile without it must not be able to record a want it could never settle.
    pub(crate) fn bind_storage_buffer(&mut self, binding: u32, range: GlStorageBufferRange)
    where
        B: GlOptionalComputeBackend,
    {
        self.buffers
            .bind_storage_buffer(binding, range, &mut self.counters);
    }

    /// Applies the desired indexed uniform bindings.
    ///
    /// This is the whole of the required role.  A caller on a profile with the
    /// optional command domains settles the storage role with
    /// [`GlStateMachine::apply_storage_buffers`] afterwards, so a required-role
    /// failure is reported before the optional role emits anything.
    pub(crate) fn apply_buffers(&mut self) -> Result<(), StateError> {
        self.buffers
            .reconcile(&mut self.backend, &mut self.counters)
    }

    /// Applies the desired indexed storage bindings, on a profile that has them.
    ///
    /// The bound is on this method rather than on the machine, which is what makes
    /// "this profile has no storage binding points" a compile-time property instead
    /// of a run-time flag: a machine over a backend without the optional command
    /// domains has no way to call this, and none of its callers can forget to
    /// check.  The alternative -- a helper trait with a no-op impl for every
    /// backend and a real one for the optional backends -- does not compile, because
    /// the two blanket impls overlap and Rust has no specialization to resolve it.
    pub(crate) fn apply_storage_buffers(&mut self) -> Result<(), StateError>
    where
        B: GlOptionalComputeBackend,
    {
        self.buffers
            .reconcile_storage(&mut self.backend, &mut self.counters)
    }

    /// Records that the caller wants `binding` to hold `image`.
    ///
    /// Gated on the optional bound for the reason the domain exists at all: the
    /// verb it mirrors is not in the required profile, so a machine over a
    /// profile without it must not be able to record a want it could never
    /// settle.  There is deliberately no unbounded `compute()` accessor beside
    /// the other domains' -- an accessor that handed the domain out would hand
    /// its unbounded entry point out with it, and the bound is the whole reason
    /// the group is optional.
    pub(crate) fn bind_storage_image(&mut self, binding: u32, image: GlStorageImageBinding)
    where
        B: GlOptionalComputeBackend,
    {
        self.compute
            .bind_storage_image(binding, image, &mut self.counters);
    }

    /// Applies the desired image-unit bindings, on a profile that has them.
    ///
    /// Bounded on the same terms as [`GlStateMachine::apply_storage_buffers`]: a
    /// machine over a backend without the optional command domains has no way to
    /// call this, and none of its callers can forget to check.
    pub(crate) fn apply_storage_images(&mut self) -> Result<(), StateError>
    where
        B: GlOptionalComputeBackend,
    {
        self.compute
            .reconcile(&mut self.backend, &mut self.counters)
    }

    /// Dispatches an invalidation to every domain.
    ///
    /// A deletion must be dispatched *before* the backend is asked to delete:
    /// the domain that holds a record naming the object has to drop it while the
    /// identity still means the object it describes, or a name reused after the
    /// deletion could hit that record.  The machine cannot enforce the ordering
    /// of a call it does not make, so it is part of this layer's contract and is
    /// checked by the differential tests.
    ///
    /// Every domain is dispatched, unconditionally, and each reacts to the rows
    /// it owns and ignores the rest -- the matrix in [`super::event`] is the list
    /// of which those are.  Dispatching on a mask built here instead would put a
    /// second copy of that matrix in this file, where it could drift from the one
    /// the domains act on.
    ///
    /// Each domain counts its own whole-mirror invalidation, so
    /// `lifecycle.domain_invalidations` reads as the number of *domain*
    /// invalidations, not the number of events.
    pub(crate) fn invalidate(&mut self, event: StateEvent) {
        self.session
            .invalidate(&mut self.backend, &event, &mut self.counters);
        self.pipeline
            .invalidate(&mut self.backend, &event, &mut self.counters);
        self.geometry
            .invalidate(&mut self.backend, &event, &mut self.counters);
        self.textures
            .invalidate(&mut self.backend, &event, &mut self.counters);
        self.buffers
            .invalidate(&mut self.backend, &event, &mut self.counters);
        self.compute
            .invalidate(&mut self.backend, &event, &mut self.counters);
    }

    /// Destroys every derived object this machine created.
    ///
    /// A caller that has decided to stop using a context calls this before
    /// releasing it, because these objects are this layer's and nobody else can
    /// destroy them.  A failed deletion is counted in
    /// [`StateCounters::lifecycle`]'s `driver_errors` rather than returned: the
    /// caller has no decision left to make about it, and the count is what a
    /// leak report reads.
    ///
    /// Only the domains that *derive* objects are drained, which is the three that
    /// own a cache: the session's framebuffers, the pipeline's programs and the
    /// geometry's vertex arrays.  The texture and buffer domains create nothing --
    /// they mirror bindings -- so there is nothing of theirs to destroy, and a
    /// `shutdown` that called into them would be looking for a drain they do not
    /// have.
    pub(crate) fn shutdown(&mut self) {
        self.session.shutdown(&mut self.backend, &mut self.counters);
        self.pipeline
            .shutdown(&mut self.backend, &mut self.counters);
        self.geometry
            .shutdown(&mut self.backend, &mut self.counters);
    }
}

#[cfg(test)]
mod tests;
