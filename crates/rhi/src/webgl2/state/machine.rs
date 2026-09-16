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

use crate::webgl2::api::{FramebufferId, GlFramebufferDescriptor, GlRenderPassDescriptor};

use super::backend::GlStateBackend;
use super::cache::DEFAULT_BUDGET;
use super::counters::StateCounters;
use super::error::StateError;
use super::event::StateEvent;
use super::knowledge::ExecutionMode;
use super::session::{SessionEffects, SessionState};

/// A backend plus the mirror of what it holds.
#[derive(Debug)]
pub(crate) struct GlStateMachine<B: GlStateBackend> {
    backend: B,
    mode: ExecutionMode,
    counters: StateCounters,
    session: SessionState,
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
            session: SessionState::new(DEFAULT_BUDGET, mode.cache()),
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

    /// Records that the caller wants this pass open, and applies the boundary.
    pub(crate) fn begin_pass(
        &mut self,
        descriptor: GlRenderPassDescriptor,
    ) -> Result<SessionEffects, StateError> {
        self.session.begin_pass(descriptor);
        self.session
            .reconcile(&mut self.backend, &mut self.counters)
    }

    /// Records that the caller wants no pass open, and applies the boundary.
    ///
    /// The effects are reported rather than applied because a pass that ended
    /// invalidated whatever pipeline was installed: Layer 1's `end_render_pass`
    /// forgets it, so a mirror that kept believing a pipeline was still
    /// installed would skip the `set_raster_pipeline` the next draw needs.
    pub(crate) fn end_pass(&mut self) -> Result<SessionEffects, StateError> {
        self.session.end_pass();
        self.session
            .reconcile(&mut self.backend, &mut self.counters)
    }

    /// Returns the framebuffer for an attachment set, building it if needed.
    ///
    /// The second half of the pair is whether the caller owns the result: a
    /// framebuffer the cache could not retain -- because the cache is disabled
    /// or because its budget could not be met without breaking a lease -- is
    /// still perfectly usable, and the caller destroys it when the pass is done.
    /// See [`super::cache::framebuffer::FramebufferCache::framebuffer_for`].
    pub(crate) fn framebuffer_for(
        &mut self,
        descriptor: &GlFramebufferDescriptor,
    ) -> Result<(FramebufferId, bool), StateError> {
        self.session
            .framebuffer_for(&mut self.backend, descriptor, &mut self.counters)
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
    /// Each domain counts its own whole-mirror invalidation, so
    /// `lifecycle.domain_invalidations` reads as the number of *domain*
    /// invalidations, not the number of events.
    pub(crate) fn invalidate(&mut self, event: StateEvent) {
        self.session
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
    pub(crate) fn shutdown(&mut self) {
        self.session.shutdown(&mut self.backend, &mut self.counters);
    }
}

/// Tests for the integration point: the wiring the domain modules are written
/// against, exercised through one domain so that a change to the domain contract
/// shows up here before it shows up in seven other modules.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::webgl2::api::tests::{snapshot, texture_desc, texture_view};
    use crate::webgl2::api::{
        GlColorAttachment, GlColorClearValue, GlFamilyProfile, GlFramebufferApi, GlLoadOp,
        GlResourceApi, GlStoreOp, MockCall, MockGlFamilyApi,
    };
    use crate::webgl2::state::cache::CacheMode;

    /// A machine over the mock and a pass that names a framebuffer the machine
    /// built for it, which is the ordering a real caller uses.
    fn machine_with_a_pass() -> (
        GlStateMachine<MockGlFamilyApi>,
        GlFramebufferDescriptor,
        GlRenderPassDescriptor,
    ) {
        let mut machine = GlStateMachine::new(MockGlFamilyApi::from_discovery(snapshot(
            GlFamilyProfile::WebGl2,
        )));
        let texture = machine
            .backend()
            .create_texture_resource(texture_desc(1))
            .expect("attachment texture");
        let descriptor = GlFramebufferDescriptor {
            color_attachments: vec![texture_view(texture, 1)],
            depth_stencil_attachment: None,
            draw_buffers: vec![],
        };
        let (framebuffer, _) = machine
            .framebuffer_for(&descriptor)
            .expect("the machine derives the framebuffer");
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
        (machine, descriptor, pass)
    }

    fn calls_of(
        machine: &mut GlStateMachine<MockGlFamilyApi>,
        matched: fn(&MockCall) -> bool,
    ) -> usize {
        machine
            .backend()
            .calls()
            .iter()
            .filter(|call| matched(call))
            .count()
    }

    /// Applies a boundary whose effect the test does not assert.
    ///
    /// [`SessionEffects`] is `#[must_use]` because a caller that ignores a pass
    /// that ended will skip the pipeline it needs.  A setup step that
    /// deliberately does not care states that here rather than at each call.
    fn apply(machine: &mut GlStateMachine<MockGlFamilyApi>, pass: Option<GlRenderPassDescriptor>) {
        let boundary = match pass {
            Some(pass) => machine.begin_pass(pass),
            None => machine.end_pass(),
        };
        let _ = boundary.expect("the boundary applies");
    }

    #[test]
    fn the_machine_derives_one_framebuffer_and_opens_the_pass_over_it() {
        let (mut machine, descriptor, pass) = machine_with_a_pass();

        let effects = machine.begin_pass(pass.clone()).expect("the pass opens");
        assert!(effects.pass_began);
        // The derived framebuffer is the machine's to destroy, and asking for
        // the same attachment set again returns the same object.
        let (again, owned) = machine
            .framebuffer_for(&descriptor)
            .expect("the framebuffer is served from the cache");
        assert_eq!(again, pass.framebuffer);
        assert!(!owned);
        assert_eq!(
            calls_of(&mut machine, |call| matches!(
                call,
                MockCall::CreateFramebuffer(_)
            )),
            1
        );

        apply(&mut machine, None);
        assert_eq!(machine.session().framebuffer_records(), 1);
    }

    #[test]
    fn shutdown_destroys_what_the_machine_derived() {
        let (mut machine, _, pass) = machine_with_a_pass();
        apply(&mut machine, Some(pass));
        apply(&mut machine, None);
        assert_eq!(
            calls_of(&mut machine, |call| matches!(
                call,
                MockCall::DestroyFramebuffer(_)
            )),
            0
        );

        machine.shutdown();

        assert_eq!(
            calls_of(&mut machine, |call| matches!(
                call,
                MockCall::DestroyFramebuffer(_)
            )),
            1
        );
        assert_eq!(machine.session().framebuffer_records(), 0);
        assert_eq!(
            machine.counters().lifecycle.driver_errors,
            0,
            "the mock accepts every deletion"
        );
    }

    #[test]
    fn an_oracle_machine_runs_the_same_domains_without_retaining_anything() {
        let mut oracle = GlStateMachine::with_mode(
            MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2)),
            ExecutionMode::Oracle,
        );
        assert_eq!(oracle.mode(), ExecutionMode::Oracle);
        assert_eq!(
            oracle.session().mode(),
            CacheMode::Disabled,
            "the oracle's domains must be the ones that retain nothing"
        );
    }
}
