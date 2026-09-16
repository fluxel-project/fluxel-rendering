//! The pipeline state domain: which raster pipeline is installed, and the
//! linked programs derived for the descriptors that ask for one.
//!
//! Responsibility: make the backend's installed program, raster pipeline, and
//! per-pipeline rasterization values agree with what the caller asked for.
//!
//! Not owned here: the pass boundary (session), the vertex bindings (geometry),
//! the texture and buffer bindings (textures, buffers), and the linked-program
//! objects themselves (Layer 1's shader tables).  This domain holds the derived
//! program cache and the installed-state mirror, and nothing else.
//!
//! # The unit of redundancy is the whole pipeline
//!
//! Layer 1 offers this domain one verb, `set_raster_pipeline`, and it carries a
//! program, a vertex array and the complete rasterization state together.  There
//! is no per-value verb to skip and nothing to ask the driver for: the provider
//! makes the program current, binds the vertex array, and applies every
//! rasterization value in that one call.  The mirror therefore compares whole
//! pipelines, which is also the only comparison that stays sound when one call
//! establishes all of those values at once -- a per-value mirror would have to
//! prove that each value was last written by the call it came from, and a single
//! partial failure would break that proof for all of them.
//!
//! # Two Layer 1 facts this domain exists for
//!
//! `end_render_pass` forgets the installed pipeline.  The provider clears its
//! own record of it, so a mirror that kept believing a pipeline was installed
//! would skip the install the next draw needs.  The session reports that in
//! [`SessionEffects`](super::session::SessionEffects), the machine applies it
//! here through [`PipelineState::pass_ended`], and this domain never reads the
//! session.
//!
//! `create_program` makes the program it linked current and then leaves no
//! program selected, because the sampler-unit assignment it performs has to run
//! against the bound program.  Linking therefore also clears the installed
//! pipeline as far as the driver is concerned, which is why
//! [`PipelineState::program_for`] invalidates the mirror on every link it makes
//! rather than only when a new object appears.
//!
//! # The consequence that leaves this domain
//!
//! A pipeline install binds the pipeline's vertex array, and the bound vertex
//! array is the geometry domain's to mirror.  So an install clobbers geometry's
//! belief exactly as a pass end clobbers this domain's, and the effect is
//! reported in [`PipelineEffects`] for the machine to apply to that domain --
//! the same arrangement, in the other direction, as the session's report.
//!
//! # The one driver fact this domain shares and does not own alone
//!
//! The install also selects the pipeline's program, and GL has exactly one
//! current program.  A compute program install selects the same slot, so the
//! driver can stop holding the program this domain installed without this
//! domain having seen anything -- and that is deliberate rather than a hole: the
//! claim this mirror makes is that the pipeline's *effect* is present, and a
//! compute install leaves the two halves of that effect it does not touch (the
//! bound vertex array and the applied rasterization values) exactly as they were.
//! Layer 1 re-asserts the program at the top of every draw, so a skip this mirror
//! grants stays correct even when another domain's verb moved the raw selection
//! in between.  The alternative -- a second mirror of "which program is current"
//! here, kept in step by a cross-domain effect -- would be two owners of one
//! driver fact, which is the drift this layer's grouping exists to prevent.
//!
//! # What a pass end costs
//!
//! Nothing is emitted at the pass end itself, because there is no verb for
//! "uninstall a pipeline": the driver forgets it on its own.  The cost shows up
//! at the next reconcile, as one recovery counted in
//! [`unknown_recoveries`](super::counters::DomainCounters::unknown_recoveries)
//! and one re-install, which is the honest accounting -- the mirror had no
//! belief left, and that is a different fact from a caller asking for something
//! new.
//!
//! # Deliberately not here
//!
//! The pass boundary, the vertex bindings, the render area from the pass
//! extent, and the pipeline's vertex-array *object*.  This domain names a
//! vertex array that the geometry domain derived and Layer 1 created; it neither
//! builds one from a layout nor decides which one a draw should use.

use crate::webgl2::api::{GlProgramDescriptor, GlProgramReflection, GlRasterPipeline, ProgramId};

pub(super) mod cache;

use super::GlStateBackend;
use super::cache::CacheBudget;
use super::counters::StateCounters;
use super::error::{PartialApplication, StateError};
use super::event::StateEvent;
use super::knowledge::{DriverKnowledge, ExecutionMode, StateDomain};
use cache::{ProgramCache, ProgramKey, ProgramRecord};

/// What changed about the installed pipeline when a reconcile ran.
///
/// The machine has to act on this: a pipeline install also binds the pipeline's
/// vertex array, and the bound vertex array is the geometry domain's mirror, so
/// a geometry reconcile that kept believing its array was still bound would skip
/// the `bind_vertex_array` the next draw needs.  Returning it rather than
/// reaching into that domain keeps the two from depending on each other, and
/// `#[must_use]` is what makes forgetting to read it a compile-time warning
/// instead of a silent wrong-geometry bug.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[must_use = "an installed pipeline also binds its vertex array, so a caller that ignores this lets the geometry mirror skip a binding it needs"]
pub(crate) struct PipelineEffects {
    /// A pipeline was installed into the backend by this reconcile.
    pub pipeline_installed: bool,
}

impl PipelineEffects {
    /// Nothing about the installed pipeline changed.
    const NONE: Self = Self {
        pipeline_installed: false,
    };
}

/// What the caller asked the installed pipeline to be.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Desired {
    /// The pipeline to be installed, or `None` for "none asked for yet".
    ///
    /// There is no way to ask for *no* pipeline: Layer 1 has no verb that
    /// uninstalls one, and ending a pass is what clears it.  So `None` means the
    /// caller has not asked for anything yet, and a reconcile in that state --
    /// a pass that only clears, or a compute-only frame -- emits nothing.
    pipeline: Option<GlRasterPipeline>,
}

/// What the backend's installed pipeline currently is.
///
/// The value is a [`DriverKnowledge`] rather than an `Option` because "no
/// pipeline is installed" and "this mirror does not know what is installed" are
/// different facts and only the first may be compared against a request.  A
/// freshly constructed domain knows nothing, and a pass that ended puts it back
/// to knowing nothing, so the first request after either emits the call that
/// establishes the value instead of being skipped against an assumed default.
#[derive(Debug)]
struct Applied {
    /// The pipeline the backend was last left holding.
    pipeline: DriverKnowledge<GlRasterPipeline>,
}

impl Default for Applied {
    fn default() -> Self {
        Self {
            pipeline: DriverKnowledge::Unknown,
        }
    }
}

/// The installed pipeline and the linked programs derived for it.
#[derive(Debug)]
pub(crate) struct PipelineState {
    desired: Desired,
    applied: Applied,
    programs: ProgramCache,
    mode: ExecutionMode,
}

impl PipelineState {
    /// A pipeline state that has installed nothing and linked nothing.
    pub(crate) fn new(budget: CacheBudget, mode: ExecutionMode) -> Self {
        Self {
            desired: Desired::default(),
            applied: Applied::default(),
            programs: ProgramCache::new(budget),
            mode,
        }
    }

    /// The execution mode this domain runs.
    ///
    /// The domain holds the mode rather than a [`CacheMode`] because the two are
    /// not the same decision: retention governs whether a linked program this
    /// domain derived is kept, and skipping governs whether an install it can
    /// prove redundant is re-emitted.  A domain given only the retention half
    /// would skip unconditionally, which is what makes an oracle trace unusable
    /// as a comparison.  The retention policy is derived where a cache needs it,
    /// via [`ExecutionMode::cache`].
    ///
    /// [`CacheMode`]: super::cache::CacheMode
    pub(crate) const fn mode(&self) -> ExecutionMode {
        self.mode
    }

    /// Records that the caller wants this pipeline installed.
    ///
    /// Nothing is emitted here.  The next [`PipelineState::reconcile`] decides
    /// whether a call is needed, which is what keeps the redundancy decision in
    /// one place instead of once per entry point.  The caller's descriptor is
    /// not re-validated either: Layer 1 validates it on every path, including
    /// the calls this layer skips, and a second definition of validity would
    /// drift from the first.
    pub(crate) fn set_pipeline(&mut self, pipeline: &GlRasterPipeline) {
        self.desired.pipeline = Some(pipeline.clone());
    }

    /// The pipeline the caller last asked for, if any.
    pub(crate) fn desired_pipeline(&self) -> Option<&GlRasterPipeline> {
        self.desired.pipeline.as_ref()
    }

    /// The pipeline this mirror believes the backend holds, if it knows.
    pub(crate) fn applied_pipeline(&self) -> Option<&GlRasterPipeline> {
        self.applied.pipeline.get()
    }

    /// Whether this mirror believes a pipeline is installed.
    pub(crate) fn pipeline_installed(&self) -> bool {
        self.applied.pipeline.is_known()
    }

    /// The number of retained program records.
    pub(crate) fn program_records(&self) -> usize {
        self.programs.len()
    }

    /// Returns the linked program for a descriptor, linking it if needed.
    ///
    /// The third half of the return is whether the caller owns the program and
    /// must destroy it when it is done, exactly as
    /// [`super::session::SessionState::framebuffer_for`] reports ownership of a
    /// framebuffer.  That is true in two cases, and both are ordinary rather
    /// than error paths: the oracle mode, where nothing is retained by
    /// construction, and a budget the record does not fit.  The program was
    /// still linked, because the caller needs
    /// one to install; it is simply not kept, which is a slower frame rather
    /// than a refusal.
    pub(crate) fn program_for(
        &mut self,
        backend: &mut impl GlStateBackend,
        descriptor: &GlProgramDescriptor,
        counters: &mut StateCounters,
    ) -> Result<(ProgramId, GlProgramReflection, bool), StateError> {
        // The key is built only when the cache is allowed to reuse, so the
        // oracle neither looks a program up nor pays for the key it would need.
        let key = self.mode.cache().may_reuse().then(|| {
            counters.allocated();
            ProgramKey::new(backend.context_stamp(), descriptor.clone())
        });
        if let Some(key) = key.as_ref() {
            if let Some((program, reflection)) = self.programs.record_for(key, counters) {
                counters.allocated();
                return Ok((program, reflection, false));
            }
        }

        let (program, reflection) = link(backend, descriptor, counters)?;
        // Linking made this program current and then cleared the selection, so
        // whatever pipeline this mirror believed was installed is no longer
        // what the driver holds.  Forgetting it here costs at most one install
        // and is the only sound answer: the provider's bind scope is invisible
        // from here.
        self.applied.pipeline.invalidate();

        let Some(key) = key else {
            return Ok((program, reflection, true));
        };
        let retained = self.programs.retain(
            backend,
            key,
            ProgramRecord {
                program,
                reflection: reflection.clone(),
            },
            counters,
        );
        if retained {
            // The record keeps a reflection of its own, which is the
            // steady-state cost of a cached program.
            counters.allocated();
        }
        Ok((program, reflection, !retained))
    }

    /// Whether the mirror proves this request redundant under this mode.
    ///
    /// The two halves are one decision.  An oracle machine runs the same domains
    /// with skipping disabled, so a domain that skipped in oracle mode would emit
    /// a trace no mirror-free machine could have produced.
    fn skippable(&self, agrees: bool) -> bool {
        self.mode.may_skip() && agrees
    }

    /// Makes the backend's installed pipeline match the desired one.
    pub(crate) fn reconcile(
        &mut self,
        backend: &mut impl GlStateBackend,
        counters: &mut StateCounters,
    ) -> Result<PipelineEffects, StateError> {
        counters.domain(StateDomain::Pipeline).request();

        let Some(wanted) = self.desired.pipeline.as_ref() else {
            // No pipeline was asked for.  There is no verb that uninstalls one,
            // so this is not a redundancy the mirror proved -- it is a request
            // model with nothing in it, and it counts as a skip in both modes.
            counters.domain(StateDomain::Pipeline).skip();
            return Ok(PipelineEffects::NONE);
        };
        if self.skippable(self.applied.pipeline.agrees(wanted)) {
            counters.domain(StateDomain::Pipeline).skip();
            return Ok(PipelineEffects::NONE);
        }
        if !self.applied.pipeline.is_known() {
            // Nothing this layer installed is in the driver any more -- the
            // first request of a context, or the first after a pass ended.
            counters.domain(StateDomain::Pipeline).recover();
        }

        match backend.set_raster_pipeline(wanted) {
            Ok(()) => {
                counters.domain(StateDomain::Pipeline).emit();
                // Storing the installed pipeline costs one allocation, and the
                // plan's third optimization candidate is judged on this counter.
                counters.allocated();
                self.applied.pipeline.set(wanted.clone());
                Ok(PipelineEffects {
                    pipeline_installed: true,
                })
            }
            Err(source) => {
                // The provider binds the program and the vertex array before it
                // applies the rasterization values, so a failure partway
                // through leaves the driver in a state this layer cannot name.
                // Forgetting the pipeline is the only sound mirror, and the
                // next reconcile re-installs it in full.
                self.applied.pipeline.invalidate();
                counters.lifecycle.driver_errors += 1;
                Err(StateError::backend(
                    StateDomain::Pipeline,
                    "set-raster-pipeline",
                    PartialApplication::new(0, 1),
                    source,
                ))
            }
        }
    }

    /// Applies the session's report that a pass ended.
    ///
    /// Layer 1's `end_render_pass` clears its own record of the installed
    /// pipeline, so the driver no longer holds what this mirror believes.  The
    /// *desired* pipeline is kept: the caller asked for it, and the next
    /// reconcile after the next pass opens is what installs it again.
    pub(crate) fn pass_ended(&mut self) {
        self.applied.pipeline.invalidate();
    }

    /// Reacts to an invalidation.
    ///
    /// The events this domain owns are a program deletion, a vertex-array
    /// deletion, and the two ways the whole mirror stops being knowable: an epoch
    /// change, and a raw scope that said it may have touched pipeline state.  A
    /// shader-object deletion is owned here by the matrix and is satisfied by
    /// doing nothing, because Layer 1 links from lowered source content and its
    /// providers delete the shader objects as soon as the link succeeds -- a
    /// deleted shader object names nothing this domain holds.
    ///
    /// Both object deletions are the same argument, which is why they are one
    /// match arm apart: a pipeline verb's effect includes binding the pipeline's
    /// vertex array, so a pipeline that *names* a deleted object is no longer a
    /// description of what the driver holds.  The vertex array's own mirror is
    /// geometry's, and geometry invalidates its claim independently -- but this
    /// mirror's claim is about the whole verb, and a mirror that kept it would
    /// skip the re-install that the deletion made necessary.
    ///
    /// The backend is needed only for the raw-scope arm: a whole-mirror epoch
    /// event purges records whose identities are no longer callable, while a
    /// scope leaves a live context whose programs this layer still has to
    /// delete.
    pub(crate) fn invalidate(
        &mut self,
        backend: &mut impl GlStateBackend,
        event: &StateEvent,
        counters: &mut StateCounters,
    ) {
        self.programs.invalidate(event, counters);

        match event {
            StateEvent::ProgramDeleted(program) => {
                // An installed pipeline that names the program being deleted is
                // no longer installable, and the driver's selection of it is not
                // something a caller can rely on once the object is gone.
                if self
                    .applied
                    .pipeline
                    .get()
                    .is_some_and(|installed| installed.program == *program)
                {
                    self.applied.pipeline.invalidate();
                }
            }
            StateEvent::VertexArrayDeleted(vertex_array) => {
                // The mirror's claim covers the whole verb, and the verb binds
                // this array.  Deleting it leaves that binding undone while the
                // program and the rasterization values stay installed, so the
                // claim is no longer a description of the driver -- and keeping it
                // would skip the install that the deletion made necessary.
                if self
                    .applied
                    .pipeline
                    .get()
                    .is_some_and(|installed| installed.vertex_array == *vertex_array)
                {
                    self.applied.pipeline.invalidate();
                }
            }
            StateEvent::ShaderDeleted(_) => {}
            StateEvent::ScopedRawAccess(scope)
                if scope.domains().contains(StateDomain::Pipeline) =>
            {
                // A raw access may have installed anything, and it may also have
                // deleted a program this layer linked.  The context is still
                // alive, so those programs are still deletable through their
                // identities: the cache drains them rather than dropping them
                // silently.
                self.programs.drop_all(backend, counters);
                self.applied.pipeline.invalidate();
                counters.lifecycle.domain_invalidations += 1;
            }
            event if event.invalidates_everything() => {
                // The identities in these records belong to a context epoch the
                // backend no longer accepts, so there is nothing to destroy
                // through them; asking would be a call with a stale object.
                self.applied.pipeline.invalidate();
                counters.lifecycle.domain_invalidations += 1;
            }
            _ => {}
        }
    }

    /// Destroys the programs this domain linked.
    ///
    /// Called when the caller is done with the context, which is the last moment
    /// the identities in these records can still be deleted through.  Unlike the
    /// whole-mirror path, this is not recoverable and both halves of the mirror
    /// are cleared: shutdown says the caller is finished, so a later reconcile
    /// must not install a pipeline the caller never asked for again.
    pub(crate) fn shutdown(
        &mut self,
        backend: &mut impl GlStateBackend,
        counters: &mut StateCounters,
    ) {
        self.programs.drop_all(backend, counters);
        self.desired = Desired::default();
        self.applied = Applied::default();
    }
}

/// Links one program, recording the failure against the pipeline domain.
fn link(
    backend: &mut impl GlStateBackend,
    descriptor: &GlProgramDescriptor,
    counters: &mut StateCounters,
) -> Result<(ProgramId, GlProgramReflection), StateError> {
    match backend.create_program(descriptor) {
        Ok(linked) => {
            counters.domain(StateDomain::Pipeline).emit();
            Ok(linked)
        }
        Err(source) => {
            counters.lifecycle.driver_errors += 1;
            Err(StateError::backend(
                StateDomain::Pipeline,
                "create-program",
                PartialApplication::new(0, 1),
                source,
            ))
        }
    }
}

#[cfg(test)]
mod tests;
