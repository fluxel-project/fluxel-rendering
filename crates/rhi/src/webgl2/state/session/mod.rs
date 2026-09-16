//! The session state domain: which pass is open, and the framebuffers derived
//! for passes to render into.
//!
//! This domain owns the *pass boundary* and nothing inside it.  Everything a
//! pass contains -- the installed pipeline, the vertex bindings, the texture
//! and uniform bindings, the compute program -- belongs to the domain that sets
//! it, because those values change many times within one pass and re-deriving
//! them from the pass would make every one of them depend on this file.
//!
//! # Why the session is the domain that opens passes
//!
//! Layer 1's `begin_render_pass` binds a framebuffer, applies the draw-buffer
//! selection, disables scissor testing for the load clears, and issues those
//! clears; `end_render_pass` re-binds the default framebuffer and forgets the
//! installed pipeline.  Both are pass-scoped acts, and a second domain that
//! also opened passes would have to know the same ordering.  So the session
//! owns the pair, and the one consequence that reaches into another domain --
//! that ending a pass invalidates the pipeline mirror -- is reported out
//! through [`SessionEffects`] rather than applied here.
//!
//! # What is deliberately not here
//!
//! The drawable.  A pass that renders to the default framebuffer needs a
//! framebuffer id, and Layer 1's descriptor validation rejects an
//! attachment-less framebuffer, so there is no such id to hand a pass.  The
//! series plan carries that as an open item; this module does not paper over it
//! with a pseudo-id, because a pass that silently targeted an attachment
//! framebuffer instead of the drawable would be a wrong-image bug with no error
//! to trace.
//!
//! The render area.  The viewport and scissor are rasterization state, set by
//! the pipeline, and are bounds-checked against the pass extent Layer 1 already
//! recorded.  The session reports the pass extent so a caller can build a
//! pipeline for it; it does not own the values.

use crate::webgl2::api::{FramebufferId, GlFramebufferDescriptor, GlRenderPassDescriptor};

pub(super) mod framebuffer;

use super::GlStateBackend;
use super::cache::CacheBudget;
use super::counters::StateCounters;
use super::error::{PartialApplication, StateError};
use super::event::StateEvent;
use super::knowledge::{ExecutionMode, StateDomain};
use framebuffer::FramebufferCache;

/// What changed about the pass boundary when a reconcile ran.
///
/// The machine has to act on this: a pass that ended invalidated the installed
/// pipeline, and a mirror that kept believing its pipeline was still installed
/// would skip the `set_raster_pipeline` the next draw needs.  Returning it
/// rather than reaching into the pipeline domain keeps the two domains from
/// depending on each other, and `#[must_use]` is what makes forgetting to read
/// it a compile-time warning instead of a silent wrong-image bug.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[must_use = "a pass that ended invalidated the installed pipeline; a caller that ignores this will skip the pipeline it needs"]
pub(crate) struct SessionEffects {
    /// A pass was opened by this reconcile.
    pub pass_began: bool,
    /// A pass was closed by this reconcile.
    pub pass_ended: bool,
}

impl SessionEffects {
    /// Nothing about the pass boundary changed.
    const NONE: Self = Self {
        pass_began: false,
        pass_ended: false,
    };
}

/// What the caller asked the pass boundary to be.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Desired {
    /// The pass to be open, or `None` for no pass.
    pass: Option<GlRenderPassDescriptor>,
}

/// What the backend's pass boundary currently is.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Applied {
    /// The pass the backend has open, or `None` when none is.
    ///
    /// Holding the whole descriptor rather than only an "open" flag is what
    /// makes the redundant-open check possible: a caller that asks for the pass
    /// already open must not pay for ending and re-beginning it, and comparing
    /// descriptors is the only way to know two asks are the same one.
    pass: Option<GlRenderPassDescriptor>,
}

/// The pass boundary and the framebuffers derived for it.
#[derive(Debug)]
pub(crate) struct SessionState {
    desired: Desired,
    applied: Applied,
    framebuffers: FramebufferCache,
    mode: ExecutionMode,
}

impl SessionState {
    /// A session that has opened no pass.
    pub(crate) fn new(budget: CacheBudget, mode: ExecutionMode) -> Self {
        Self {
            desired: Desired::default(),
            applied: Applied::default(),
            framebuffers: FramebufferCache::new(budget),
            mode,
        }
    }

    /// The execution mode this session runs.
    ///
    /// The domain holds the mode rather than a [`CacheMode`] because the two are
    /// not the same decision: retention governs whether a framebuffer this
    /// session derived is kept, and skipping governs whether a pass it can prove
    /// unchanged is re-opened.  A domain given only the retention half would skip
    /// unconditionally, which is what makes an oracle trace unusable as a
    /// comparison.  The retention policy is derived where a cache needs it, via
    /// [`ExecutionMode::cache`].
    ///
    /// [`CacheMode`]: super::cache::CacheMode
    pub(crate) const fn mode(&self) -> ExecutionMode {
        self.mode
    }

    /// Records that the caller wants this pass open.
    ///
    /// Nothing is emitted here.  The next [`SessionState::reconcile`] decides
    /// whether a call is needed, which is what keeps the redundancy check in one
    /// place instead of once per entry point.
    pub(crate) fn begin_pass(&mut self, descriptor: GlRenderPassDescriptor) {
        self.desired.pass = Some(descriptor);
    }

    /// Records that the caller wants no pass open.
    pub(crate) fn end_pass(&mut self) {
        self.desired.pass = None;
    }

    /// The pass the caller last asked for, if any.
    pub(crate) fn desired_pass(&self) -> Option<&GlRenderPassDescriptor> {
        self.desired.pass.as_ref()
    }

    /// The pass the backend has open, if any.
    pub(crate) fn applied_pass(&self) -> Option<&GlRenderPassDescriptor> {
        self.applied.pass.as_ref()
    }

    /// Whether a pass is currently open in the backend.
    pub(crate) fn pass_open(&self) -> bool {
        self.applied.pass.is_some()
    }

    /// The number of retained framebuffer records.
    pub(crate) fn framebuffer_records(&self) -> usize {
        self.framebuffers.len()
    }

    /// Returns the framebuffer for an attachment set, building it if needed.
    ///
    /// The second half of the pair is whether the caller owns the result and
    /// must destroy it: true when the cache is disabled, and true when the
    /// budget could not keep it.  See [`FramebufferCache::framebuffer_for`].
    pub(crate) fn framebuffer_for(
        &mut self,
        backend: &mut impl GlStateBackend,
        descriptor: &GlFramebufferDescriptor,
        counters: &mut StateCounters,
    ) -> Result<(FramebufferId, bool), StateError> {
        self.framebuffers
            .framebuffer_for(backend, descriptor, self.mode.cache(), counters)
    }

    /// Whether the mirror proves this request redundant under this mode.
    ///
    /// The two halves are one decision.  An oracle machine runs the same domains
    /// with skipping disabled, so a domain that skipped in oracle mode would emit
    /// a trace no mirror-free machine could have produced.
    fn skippable(&self, agrees: bool) -> bool {
        self.mode.may_skip() && agrees
    }

    /// Makes the backend's pass boundary match the desired one.
    pub(crate) fn reconcile(
        &mut self,
        backend: &mut impl GlStateBackend,
        counters: &mut StateCounters,
    ) -> Result<SessionEffects, StateError> {
        counters.domain(StateDomain::Session).request();

        let applied_open = self.applied.pass.is_some();
        // A boundary with nothing on either side is not a redundancy this layer
        // proved -- there is no verb that would close nothing -- so it counts as
        // a skip in both modes, exactly as a domain with no request does.
        if self.desired.pass.is_none() && !applied_open {
            counters.domain(StateDomain::Session).skip();
            return Ok(SessionEffects::NONE);
        }
        // Re-opening the pass that is already open *is* a proven redundancy, so
        // only a mode that may skip acts on it.  An oracle falls through and
        // re-establishes the boundary, which is what a mirror-free machine given
        // the same two requests would have done.
        let unchanged = match (&self.desired.pass, &self.applied.pass) {
            (Some(wanted), Some(open)) => wanted == open,
            _ => false,
        };
        if self.skippable(unchanged) {
            counters.domain(StateDomain::Session).skip();
            return Ok(SessionEffects::NONE);
        }

        // Close first, and only clear the applied pass once the backend agreed:
        // a failed end leaves the pass open, and a mirror that forgot it would
        // then open a second one over it.
        let closing = applied_open;
        if applied_open {
            end(backend, counters)?;
            self.applied.pass = None;
        }
        let Some(wanted) = self.desired.pass.clone() else {
            return Ok(SessionEffects {
                pass_began: false,
                pass_ended: closing,
            });
        };
        // Storing the descriptor costs one allocation, and the plan's third
        // optimization candidate is judged on exactly this counter.
        counters.allocated();
        begin(backend, &wanted, counters)?;
        self.applied.pass = Some(wanted);
        Ok(SessionEffects {
            pass_began: true,
            pass_ended: closing,
        })
    }

    /// Destroys the framebuffers this session derived.
    ///
    /// Called when the caller is done with the context, which is the last moment
    /// the identities in these records can still be destroyed through.  Unlike
    /// the whole-mirror path, this is not recoverable and both halves of the
    /// boundary are cleared: shutdown says the caller is finished, so a later
    /// reconcile must not re-open a pass the caller never asked for again.
    pub(crate) fn shutdown(
        &mut self,
        backend: &mut impl GlStateBackend,
        counters: &mut StateCounters,
    ) {
        self.framebuffers.drop_all(backend, counters);
        self.desired = Desired::default();
        self.applied = Applied::default();
    }

    /// Reacts to an invalidation.
    ///
    /// The pass boundary itself is only invalidated by a whole-mirror event:
    /// a texture or framebuffer deletion does not close a pass, but it does
    /// invalidate the framebuffer records derived from it, which is why the
    /// cache is dispatched to unconditionally and the boundary is not.
    pub(crate) fn invalidate(
        &mut self,
        backend: &mut impl GlStateBackend,
        event: &StateEvent,
        counters: &mut StateCounters,
    ) {
        self.framebuffers.invalidate(backend, event, counters);

        if event.invalidates_everything() {
            // The backend's pass state is gone with the context, so the mirror
            // must not claim a pass is open -- but nothing is emitted, because
            // there is no context left to emit into.  The desired pass is kept:
            // the caller asked for it, and the next reconcile after restoration
            // is what opens it again.
            self.applied = Applied::default();
            counters.lifecycle.domain_invalidations += 1;
        }
    }
}

/// Opens a pass, recording the failure against the session domain.
fn begin(
    backend: &mut impl GlStateBackend,
    descriptor: &GlRenderPassDescriptor,
    counters: &mut StateCounters,
) -> Result<(), StateError> {
    match backend.begin_render_pass(descriptor) {
        Ok(()) => {
            counters.domain(StateDomain::Session).emit();
            counters.submissions.passes += 1;
            counters.submissions.pass_loads += 1;
            Ok(())
        }
        Err(source) => {
            counters.lifecycle.driver_errors += 1;
            Err(StateError::backend(
                StateDomain::Session,
                "begin-pass",
                PartialApplication::new(0, 1),
                source,
            ))
        }
    }
}

/// Closes a pass, recording the failure against the session domain.
fn end(backend: &mut impl GlStateBackend, counters: &mut StateCounters) -> Result<(), StateError> {
    match backend.end_render_pass() {
        Ok(()) => {
            counters.domain(StateDomain::Session).emit();
            counters.submissions.pass_stores += 1;
            Ok(())
        }
        Err(source) => {
            counters.lifecycle.driver_errors += 1;
            Err(StateError::backend(
                StateDomain::Session,
                "end-pass",
                PartialApplication::new(0, 1),
                source,
            ))
        }
    }
}

/// Tests for the pass boundary and the framebuffer records.
///
/// Every test here runs the domain against the mock provider rather than a
/// hand-written double, because what is being tested is the *trace* -- which
/// calls were emitted, in what order, and how many -- and a double written for
/// these tests would answer with whatever trace the test expected.  The mock is
/// the same provider the Layer 1 contract suites run against, so a change in
/// Layer 1's own call surface shows up here as a failing trace rather than as a
/// mirror that quietly agrees with itself.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::webgl2::api::tests::{snapshot, texture_desc, texture_view};
    use crate::webgl2::api::{
        GlAttachmentTarget, GlColorAttachment, GlColorClearValue, GlFamilyApi, GlFamilyProfile,
        GlFramebufferApi, GlLoadOp, GlResourceApi, GlStoreOp, MockCall, MockGlFamilyApi,
    };
    use crate::webgl2::state::cache::DEFAULT_BUDGET;

    /// A backend, an attachment set, and the framebuffer and pass over it.
    ///
    /// The framebuffer is built here rather than by the domain because the
    /// session never creates one on its own initiative: a caller asks for the
    /// framebuffer it wants to render into and names it in the pass descriptor.
    fn attachment_set(
        backend: &mut MockGlFamilyApi,
    ) -> (GlFramebufferDescriptor, GlRenderPassDescriptor) {
        let texture = backend
            .create_texture_resource(texture_desc(1))
            .expect("attachment texture");
        let descriptor = GlFramebufferDescriptor {
            color_attachments: vec![texture_view(texture, 1)],
            depth_stencil_attachment: None,
            draw_buffers: vec![],
        };
        let framebuffer = backend
            .create_framebuffer(&descriptor)
            .expect("framebuffer over the attachment set");
        let pass = GlRenderPassDescriptor {
            framebuffer,
            color_attachments: vec![GlColorAttachment {
                view: texture_view(texture, 1),
                resolve_target: None,
                load: GlLoadOp::Clear,
                store: GlStoreOp::Store,
                clear: GlColorClearValue {
                    red: 0,
                    green: 0,
                    blue: 0,
                    alpha: 0,
                },
            }],
            depth_stencil_attachment: None,
        };
        (descriptor, pass)
    }

    fn backend() -> MockGlFamilyApi {
        MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2))
    }

    fn session(mode: ExecutionMode) -> SessionState {
        SessionState::new(DEFAULT_BUDGET, mode)
    }

    /// Applies the boundary for a setup step whose effect the test does not
    /// assert.
    ///
    /// [`SessionEffects`] is `#[must_use]` because a caller that ignores a pass
    /// that ended will skip the pipeline it needs.  A test step that deliberately
    /// does not care states that here, once, instead of at each call site.
    fn apply(
        session: &mut SessionState,
        backend: &mut MockGlFamilyApi,
        counters: &mut StateCounters,
    ) {
        let _ = session
            .reconcile(backend, counters)
            .expect("the boundary applies");
    }

    fn calls_of(backend: &MockGlFamilyApi, matched: fn(&MockCall) -> bool) -> usize {
        backend.calls().iter().filter(|call| matched(call)).count()
    }

    fn begins(backend: &MockGlFamilyApi) -> usize {
        calls_of(backend, |call| matches!(call, MockCall::BeginRenderPass(_)))
    }

    fn ends(backend: &MockGlFamilyApi) -> usize {
        calls_of(backend, |call| matches!(call, MockCall::EndRenderPass))
    }

    fn created_framebuffers(backend: &MockGlFamilyApi) -> usize {
        calls_of(backend, |call| {
            matches!(call, MockCall::CreateFramebuffer(_))
        })
    }

    fn destroyed_framebuffers(backend: &MockGlFamilyApi) -> usize {
        calls_of(backend, |call| {
            matches!(call, MockCall::DestroyFramebuffer(_))
        })
    }

    #[test]
    fn an_unchanged_pass_is_not_reopened() {
        let mut backend = backend();
        let (_, pass) = attachment_set(&mut backend);
        let mut session = session(ExecutionMode::Optimized);
        let mut counters = StateCounters::default();

        session.begin_pass(pass);
        let first = session
            .reconcile(&mut backend, &mut counters)
            .expect("the first begin emits");
        assert_eq!(
            first,
            SessionEffects {
                pass_began: true,
                pass_ended: false,
            }
        );

        // The caller asking for the pass that is already open is the common
        // case in a renderer that re-enters the same pass to add work.
        let second = session
            .reconcile(&mut backend, &mut counters)
            .expect("an unchanged boundary is not a failure");
        assert_eq!(second, SessionEffects::NONE);
        assert_eq!(begins(&backend), 1);
        assert_eq!(ends(&backend), 0);
        assert_eq!(counters.domain_counts(StateDomain::Session).emitted, 1);
        assert_eq!(counters.domain_counts(StateDomain::Session).skipped, 1);
        assert_eq!(counters.submissions.passes, 1);
    }

    #[test]
    fn a_pass_over_a_different_attachment_set_replaces_the_open_one() {
        let mut backend = backend();
        let (_, first_pass) = attachment_set(&mut backend);
        let (_, second_pass) = attachment_set(&mut backend);
        let mut session = session(ExecutionMode::Optimized);
        let mut counters = StateCounters::default();

        session.begin_pass(first_pass);
        apply(&mut session, &mut backend, &mut counters);
        session.begin_pass(second_pass);
        let effects = session
            .reconcile(&mut backend, &mut counters)
            .expect("a different pass replaces the open one");

        assert_eq!(
            effects,
            SessionEffects {
                pass_began: true,
                pass_ended: true,
            }
        );
        // Order matters: Layer 1 refuses a begin while a pass is active, so a
        // reconcile that opened before closing would fail on the second pass.
        assert_eq!(begins(&backend), 2);
        assert_eq!(ends(&backend), 1);
        assert_eq!(counters.submissions.passes, 2);
        assert_eq!(counters.submissions.pass_loads, 2);
        assert_eq!(counters.submissions.pass_stores, 1);
    }

    #[test]
    fn ending_a_pass_reports_the_effect_that_invalidates_the_pipeline() {
        let mut backend = backend();
        let (_, pass) = attachment_set(&mut backend);
        let mut session = session(ExecutionMode::Optimized);
        let mut counters = StateCounters::default();

        session.begin_pass(pass.clone());
        apply(&mut session, &mut backend, &mut counters);
        session.end_pass();
        let effects = session
            .reconcile(&mut backend, &mut counters)
            .expect("the end emits");

        assert!(effects.pass_ended);
        assert!(!effects.pass_began);
        assert!(!session.pass_open());
        assert_eq!(ends(&backend), 1);

        // Ending a pass that is already closed is not a driver call, and not a
        // failure: the caller states the boundary it wants, not a transition.
        session.end_pass();
        let again = session
            .reconcile(&mut backend, &mut counters)
            .expect("an already-closed boundary is not a failure");
        assert_eq!(again, SessionEffects::NONE);
        assert_eq!(ends(&backend), 1);
        assert_eq!(counters.submissions.pass_stores, 1);

        // The mirror must not treat "closed" as "already open", so the same
        // descriptor re-opens the pass rather than being skipped.
        session.begin_pass(pass);
        let reopened = session
            .reconcile(&mut backend, &mut counters)
            .expect("a closed pass re-opens");
        assert_eq!(
            reopened,
            SessionEffects {
                pass_began: true,
                pass_ended: false,
            }
        );
        assert_eq!(begins(&backend), 2);
    }

    #[test]
    fn the_same_attachment_set_reuses_one_framebuffer_object() {
        let mut backend = backend();
        let (descriptor, _) = attachment_set(&mut backend);
        let mut session = session(ExecutionMode::Optimized);
        let mut counters = StateCounters::default();
        let built_by_the_fixture = created_framebuffers(&backend);

        let (first, owned) = session
            .framebuffer_for(&mut backend, &descriptor, &mut counters)
            .expect("a framebuffer is built on a miss");
        assert!(!owned, "a retained framebuffer belongs to the cache");
        let (second, owned) = session
            .framebuffer_for(&mut backend, &descriptor, &mut counters)
            .expect("a framebuffer is served from the cache");
        assert_eq!(first, second);
        assert!(!owned);

        assert_eq!(created_framebuffers(&backend), built_by_the_fixture + 1);
        assert_eq!(counters.caches.hits, 1);
        assert_eq!(counters.caches.created, 1);
        assert_eq!(session.framebuffer_records(), 1);
    }

    #[test]
    fn a_disabled_cache_never_reuses_and_hands_ownership_back() {
        let mut backend = backend();
        let (descriptor, _) = attachment_set(&mut backend);
        let mut session = session(ExecutionMode::Oracle);
        let mut counters = StateCounters::default();
        let built_by_the_fixture = created_framebuffers(&backend);

        let (_, owned) = session
            .framebuffer_for(&mut backend, &descriptor, &mut counters)
            .expect("the oracle still builds a usable framebuffer");
        assert!(
            owned,
            "the oracle retains nothing, so the caller destroys it"
        );
        let (_, owned) = session
            .framebuffer_for(&mut backend, &descriptor, &mut counters)
            .expect("the oracle builds again");
        assert!(owned);

        assert_eq!(created_framebuffers(&backend), built_by_the_fixture + 2);
        assert_eq!(session.framebuffer_records(), 0);
        assert_eq!(counters.caches.hits, 0);
        assert_eq!(counters.caches.created, 0);
    }

    #[test]
    fn deleting_an_attachment_drops_the_framebuffer_built_from_it() {
        let mut backend = backend();
        let (descriptor, _) = attachment_set(&mut backend);
        let texture = match descriptor.color_attachments[0].target {
            GlAttachmentTarget::Texture(texture) => texture,
            other => panic!("the fixture attaches a texture, not {other:?}"),
        };
        let mut session = session(ExecutionMode::Optimized);
        let mut counters = StateCounters::default();
        session
            .framebuffer_for(&mut backend, &descriptor, &mut counters)
            .expect("a framebuffer is built");
        assert_eq!(session.framebuffer_records(), 1);

        session.invalidate(
            &mut backend,
            &StateEvent::TextureDeleted(texture),
            &mut counters,
        );

        // The record goes before the backend is asked to delete the attachment,
        // which is the ordering that keeps a reused name from hitting it.
        assert_eq!(session.framebuffer_records(), 0);
        assert_eq!(destroyed_framebuffers(&backend), 1);
        assert_eq!(counters.caches.invalidated, 1);
        assert_eq!(
            counters.lifecycle.domain_invalidations, 0,
            "a single deletion is not a whole-domain invalidation"
        );
    }

    #[test]
    fn a_context_loss_purges_records_without_destroying_them() {
        let mut backend = backend();
        let (descriptor, pass) = attachment_set(&mut backend);
        let mut session = session(ExecutionMode::Optimized);
        let mut counters = StateCounters::default();
        session
            .framebuffer_for(&mut backend, &descriptor, &mut counters)
            .expect("a framebuffer is built");
        session.begin_pass(pass);
        apply(&mut session, &mut backend, &mut counters);
        assert!(session.pass_open());
        let before = destroyed_framebuffers(&backend);

        session.invalidate(&mut backend, &StateEvent::ContextLost, &mut counters);

        assert_eq!(session.framebuffer_records(), 0);
        assert_eq!(
            destroyed_framebuffers(&backend),
            before,
            "an identity from a lost epoch is not callable, so nothing may be destroyed through it"
        );
        assert!(counters.caches.purged >= 1);
        assert_eq!(counters.lifecycle.domain_invalidations, 1);
        // The mirror must stop claiming a pass the context took with it, while
        // keeping what the caller asked for: restoration re-opens it.
        assert!(!session.pass_open());
        assert!(session.desired_pass().is_some());
    }

    #[test]
    fn a_pass_that_cannot_be_opened_leaves_no_applied_pass() {
        let mut backend = backend();
        let (_, pass) = attachment_set(&mut backend);
        let mut session = session(ExecutionMode::Optimized);
        let mut counters = StateCounters::default();
        backend.context_lost().expect("the mock records the loss");

        session.begin_pass(pass);
        let error = session
            .reconcile(&mut backend, &mut counters)
            .expect_err("a lost context refuses the pass");

        assert_eq!(error.domain(), StateDomain::Session);
        assert_eq!(error.operation(), "begin-pass");
        assert!(
            !session.pass_open(),
            "a failed begin must not be mirrored as an open pass"
        );
        assert_eq!(counters.lifecycle.driver_errors, 1);
        assert_eq!(counters.submissions.passes, 0);
        assert_eq!(begins(&backend), 0);
    }
}
