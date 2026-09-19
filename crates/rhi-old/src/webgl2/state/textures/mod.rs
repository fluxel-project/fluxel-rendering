//! The texture state domain: the active texture unit, the texture bound at each
//! unit and target, and the sampler bound at each unit.
//!
//! Responsibility: make the backend's texture-unit and sampler bindings agree
//! with what the caller asked for.
//!
//! Not owned here: the texture images and sampler objects themselves, which are
//! Layer 1's allocation tables, and the framebuffers built from texture views,
//! which are the session domain's.  This domain creates nothing, destroys
//! nothing and validates nothing of its own -- see below.
//!
//! # The Layer 1 call surface this mirror is built on
//!
//! Read off the executable providers rather than assumed:
//!
//! 1. `bind_texture` **selects the unit it binds into**.  Both providers emit the
//!    unit selection and then the bind, and they do it on the unbind path too, so
//!    `bind_texture(unit, target, None)` moves the active unit exactly as
//!    `Some(texture)` does.  The selection is a side effect of a verb whose
//!    caller named a unit, a target and a texture -- not of the verb whose only
//!    argument is a unit.
//! 2. `bind_sampler` **does not** select a unit.  It is one call addressed by
//!    unit index, so a sampler bind leaves the active unit where it was.  A
//!    mirror that treated the two verbs alike would be wrong in both directions.
//! 3. The two verbs share one unit index space and hold two binding namespaces.
//!    Layer 1 bounds both by the discovered combined texture-unit count with the
//!    same rule, so a unit index means the same unit in either verb; inside one
//!    unit the sampler binding and the four target bindings are independent
//!    slots, and binding one never disturbs another.
//! 4. The target is part of the binding identity.  A unit holds one binding per
//!    target, so the same texture at the same unit under a different target is a
//!    different binding, and a mirror keyed by unit alone would skip a bind the
//!    driver still needs.
//! 5. Rebinding an identical slot with identical arguments is legal, and Layer 1
//!    deliberately does not deduplicate it: that decision is this layer's.
//!
//! The unit itself stays Layer 1's to validate.  It rejects an out-of-range unit
//! on every path, including the paths this domain skips, so this domain holds no
//! unit-count knowledge and no second opinion about range.
//!
//! # Why the active unit is recorded from the emission, never from the request
//!
//! The active unit is the one value here that a *different* verb mutates, and the
//! asymmetry is easy to get wrong: it moves when a bind is emitted, and it does
//! not move when a bind is skipped, because a skipped bind emitted nothing.  A
//! mirror that recorded the requested unit instead would claim a unit the driver
//! is not on, and the failure that follows has no error to trace -- a later
//! selection of the unit the driver actually holds would be skipped, and every
//! bind after it would land on the wrong texture.
//!
//! # Why this domain has no derived cache
//!
//! The session domain keeps one because a pass *derives* a framebuffer: a backend
//! object this layer created, so this layer must destroy it, and that is what
//! makes a drain distinguishable from a purge.  A binding is not an object.
//! Nothing here is created or destroyed, the mirror's whole content is the
//! driver's own state, and its key is the unit the driver already keys that state
//! by, so a derived table over the same key would be a second copy of [`Applied`]
//! with no lifecycle of its own.  This domain's `invalidate` is therefore only
//! ever a forgetting, and it emits nothing because it has nothing to destroy.

use std::collections::BTreeMap;

use crate::webgl2::api::{GlError, GlTextureTarget, SamplerId, TextureId};

use super::GlStateBackend;
use super::counters::StateCounters;
use super::error::{PartialApplication, StateError};
use super::event::StateEvent;
use super::knowledge::{DriverKnowledge, ExecutionMode, StateDomain};

#[cfg(test)]
mod tests;

/// What the caller last asked this domain for.
///
/// One request at a time rather than a table of every binding the caller has ever
/// asked for, for two reasons.  A reconcile has to be able to report one request
/// and one outcome, and a table would turn one request into a group of calls
/// whose size the caller never asked for.  And the oracle mode -- which exists so
/// a differential test can compare this layer's trace against the trace of a
/// machine with no mirror at all -- has to emit exactly what such a machine
/// would, which is one call per request; a table would emit the caller's whole
/// history on every reconcile.
///
/// What persists is [`Applied`]: this is the request, not the state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Request {
    /// The caller has asked for nothing since this domain was built.
    #[default]
    Nothing,
    /// Make `unit` the active texture unit.
    ActiveUnit(u32),
    /// Bind or unbind `texture` at `unit` for `target`.
    ///
    /// `None` is an unbind, which the providers still implement by selecting the
    /// unit first, so it is not a no-op for the active unit.
    Texture {
        unit: u32,
        target: GlTextureTarget,
        texture: Option<TextureId>,
    },
    /// Bind or unbind `sampler` at `unit`.
    Sampler {
        unit: u32,
        sampler: Option<SamplerId>,
    },
}

/// What the driver is known to hold.
///
/// Absence from either map is `Unknown`, never "unbound": a slot this domain has
/// never applied says nothing at all about the driver, and folding the two
/// together is how a mirror starts believing a default.  An entry whose value is
/// `None` is the one honest statement that the slot is known to be unbound.
#[derive(Debug)]
struct Applied {
    /// The active texture unit.
    active: DriverKnowledge<u32>,
    /// The texture bound at each (unit, target) this domain has applied.
    textures: BTreeMap<(u32, GlTextureTarget), Option<TextureId>>,
    /// The sampler bound at each unit this domain has applied.
    samplers: BTreeMap<u32, Option<SamplerId>>,
}

impl Applied {
    /// The mirror of a driver this domain has said nothing about.
    ///
    /// There is deliberately no default-valued constructor: the two values a
    /// fresh mirror could be tempted to assume -- unit zero, and an unbound slot
    /// -- are exactly the driver facts a Host, a debug layer or a previous
    /// context owner may already have changed.
    fn unknown() -> Self {
        Self {
            active: DriverKnowledge::Unknown,
            textures: BTreeMap::new(),
            samplers: BTreeMap::new(),
        }
    }

    /// What the driver is known to hold at one texture slot.
    fn texture(&self, unit: u32, target: GlTextureTarget) -> DriverKnowledge<Option<TextureId>> {
        match self.textures.get(&(unit, target)) {
            Some(bound) => DriverKnowledge::Known(*bound),
            None => DriverKnowledge::Unknown,
        }
    }

    /// What the driver is known to hold at one sampler slot.
    fn sampler(&self, unit: u32) -> DriverKnowledge<Option<SamplerId>> {
        match self.samplers.get(&unit) {
            Some(bound) => DriverKnowledge::Known(*bound),
            None => DriverKnowledge::Unknown,
        }
    }
}

/// The unit-axis mirror: one active unit, one texture binding per (unit, target),
/// one sampler binding per unit.
#[derive(Debug)]
pub(crate) struct TexturesState {
    desired: Request,
    applied: Applied,
    mode: ExecutionMode,
}

impl TexturesState {
    /// A domain that has applied nothing.
    ///
    /// The mode is fixed at construction for the reason the machine fixes its own:
    /// an oracle that began skipping mid-life would produce a trace no
    /// mirror-free machine could have produced, which is the one comparison the
    /// differential test makes.
    pub(crate) fn new(mode: ExecutionMode) -> Self {
        Self {
            desired: Request::Nothing,
            applied: Applied::unknown(),
            mode,
        }
    }

    /// The execution mode this domain runs.
    pub(crate) const fn mode(&self) -> ExecutionMode {
        self.mode
    }

    /// Records that the caller wants `unit` to be the active texture unit.
    ///
    /// Nothing is emitted here.  The next [`TexturesState::reconcile`] decides
    /// whether the selection is needed, which is what keeps the redundancy check
    /// in one place instead of once per entry point.
    pub(crate) fn active_texture(&mut self, unit: u32) {
        self.desired = Request::ActiveUnit(unit);
    }

    /// Records that the caller wants `texture` bound at `unit` for `target`.
    pub(crate) fn bind_texture(
        &mut self,
        unit: u32,
        target: GlTextureTarget,
        texture: Option<TextureId>,
    ) {
        self.desired = Request::Texture {
            unit,
            target,
            texture,
        };
    }

    /// Records that the caller wants `sampler` bound at `unit`.
    pub(crate) fn bind_sampler(&mut self, unit: u32, sampler: Option<SamplerId>) {
        self.desired = Request::Sampler { unit, sampler };
    }

    /// The unit the caller last asked to be active, if that was the last request.
    ///
    /// A texture or sampler request answers `None`, because neither of them is a
    /// request about the active unit -- the texture one only happens to move it.
    pub(crate) fn desired_active_unit(&self) -> Option<u32> {
        match self.desired {
            Request::ActiveUnit(unit) => Some(unit),
            _ => None,
        }
    }

    /// What the driver is known to hold as the active texture unit.
    pub(crate) fn applied_active_unit(&self) -> DriverKnowledge<u32> {
        self.applied.active
    }

    /// What the driver is known to hold at one texture slot.
    pub(crate) fn applied_texture(
        &self,
        unit: u32,
        target: GlTextureTarget,
    ) -> DriverKnowledge<Option<TextureId>> {
        self.applied.texture(unit, target)
    }

    /// What the driver is known to hold at one sampler slot.
    pub(crate) fn applied_sampler(&self, unit: u32) -> DriverKnowledge<Option<SamplerId>> {
        self.applied.sampler(unit)
    }

    /// Makes the backend's bindings match the request the caller last recorded.
    ///
    /// One request, one outcome: either the verb is emitted, or the mirror proves
    /// it redundant and it is skipped.  Both the request and the outcome are
    /// counted, and an emit forced by an unknown value is counted as a recovery,
    /// which is the counter that says how much of this domain's traffic is
    /// repairing knowledge rather than moving state.
    pub(crate) fn reconcile(
        &mut self,
        backend: &mut impl GlStateBackend,
        counters: &mut StateCounters,
    ) -> Result<(), StateError> {
        counters.domain(StateDomain::Textures).request();

        match self.desired {
            // A domain nobody has asked anything of emits nothing.  It is the
            // only reconcile that is not a skip: there was no request to skip.
            Request::Nothing => {
                counters.domain(StateDomain::Textures).skip();
                Ok(())
            }
            Request::ActiveUnit(unit) => {
                if self.skippable(self.applied.active.agrees(&unit)) {
                    counters.domain(StateDomain::Textures).skip();
                    return Ok(());
                }
                if !self.applied.active.is_known() {
                    counters.domain(StateDomain::Textures).recover();
                }
                self.select_unit(backend, unit, counters)
            }
            Request::Texture {
                unit,
                target,
                texture,
            } => {
                if self.skippable(self.applied.texture(unit, target).agrees(&texture)) {
                    counters.domain(StateDomain::Textures).skip();
                    return Ok(());
                }
                if !self.applied.texture(unit, target).is_known() {
                    counters.domain(StateDomain::Textures).recover();
                }
                self.apply_texture(backend, unit, target, texture, counters)
            }
            Request::Sampler { unit, sampler } => {
                if self.skippable(self.applied.sampler(unit).agrees(&sampler)) {
                    counters.domain(StateDomain::Textures).skip();
                    return Ok(());
                }
                if !self.applied.sampler(unit).is_known() {
                    counters.domain(StateDomain::Textures).recover();
                }
                self.apply_sampler(backend, unit, sampler, counters)
            }
        }
    }

    /// Reacts to an invalidation.
    ///
    /// The backend is unused and is still taken, because the domain contract
    /// fixes this signature for every domain.  It is unused for a reason worth
    /// stating: this domain has no derived object, so a deletion leaves it
    /// nothing to destroy, and it emits no unbind either -- the driver clears the
    /// bindings to an object it is deleting, so an unbind here would be a call
    /// made on the strength of driver behavior this layer would have to assume,
    /// and would move the active unit as a side effect of a path the caller does
    /// not expect to have one.
    pub(crate) fn invalidate(
        &mut self,
        _backend: &mut impl GlStateBackend,
        event: &StateEvent,
        counters: &mut StateCounters,
    ) {
        match event {
            StateEvent::TextureDeleted(texture) => self.forget_texture(*texture),
            StateEvent::SamplerDeleted(sampler) => self.forget_sampler(*sampler),
            StateEvent::ScopedRawAccess(scope) => {
                // A scope that declared nothing is a whole-mirror event and is
                // handled below; a scope that declared other domains left this
                // one alone, which is the entire reason for declaring them.
                if scope.domains().contains(StateDomain::Textures) {
                    self.purge(counters);
                }
            }
            event => {
                if event.invalidates_everything() {
                    self.purge(counters);
                }
            }
        }
    }

    /// Whether the mirror proves this request redundant under this mode.
    ///
    /// The two halves are one decision.  An oracle machine runs the same domains
    /// with skipping disabled, so a domain that skipped in oracle mode would emit
    /// a trace no mirror-free machine could have produced.
    fn skippable(&self, agrees: bool) -> bool {
        self.mode.may_skip() && agrees
    }

    /// Selects the active unit, recording the failure against this domain.
    fn select_unit(
        &mut self,
        backend: &mut impl GlStateBackend,
        unit: u32,
        counters: &mut StateCounters,
    ) -> Result<(), StateError> {
        match backend.active_texture(unit) {
            Ok(()) => {
                counters.domain(StateDomain::Textures).emit();
                self.applied.active.set(unit);
                Ok(())
            }
            Err(source) => Err(self.poison(counters, "active-texture", source)),
        }
    }

    /// Binds or unbinds one texture, recording the failure against this domain.
    fn apply_texture(
        &mut self,
        backend: &mut impl GlStateBackend,
        unit: u32,
        target: GlTextureTarget,
        texture: Option<TextureId>,
        counters: &mut StateCounters,
    ) -> Result<(), StateError> {
        match backend.bind_texture(unit, target, texture) {
            Ok(()) => {
                counters.domain(StateDomain::Textures).emit();
                // The provider selects `unit` before it binds, so the driver's
                // active unit is `unit` now, and this is the only place that may
                // claim so: a caller that only asks for a bind has not asked for
                // a selection, and a skipped bind never reaches this line.
                self.applied.active.set(unit);
                if !self.applied.textures.contains_key(&(unit, target)) {
                    counters.allocated();
                }
                self.applied.textures.insert((unit, target), texture);
                Ok(())
            }
            Err(source) => Err(self.poison(counters, "bind-texture", source)),
        }
    }

    /// Binds or unbinds one sampler, recording the failure against this domain.
    fn apply_sampler(
        &mut self,
        backend: &mut impl GlStateBackend,
        unit: u32,
        sampler: Option<SamplerId>,
        counters: &mut StateCounters,
    ) -> Result<(), StateError> {
        match backend.bind_sampler(unit, sampler) {
            Ok(()) => {
                counters.domain(StateDomain::Textures).emit();
                // Deliberately no update of the active unit: this verb is
                // addressed by unit index and selects nothing, so a mirror that
                // moved the unit here would skip a selection the driver still
                // needs.
                if self.applied.samplers.insert(unit, sampler).is_none() {
                    counters.allocated();
                }
                Ok(())
            }
            Err(source) => Err(self.poison(counters, "bind-sampler", source)),
        }
    }

    /// Records a failed application and marks the whole mirror unknown.
    ///
    /// The whole mirror, not only the slot the request named.  `bind_texture` is
    /// a unit selection followed by a bind, so a failure may have moved the active
    /// unit before it was reported, and this layer cannot tell that case from one
    /// where the refusal happened before any side effect.  An unknown mirror costs
    /// one re-applied request; a mirror that keeps claiming the unit it asked for
    /// costs a wrong-texture frame with no error to trace.
    ///
    /// The request is deliberately left in place.  The error contract is that a
    /// failed group is unknown and is re-applied in full on the next reconcile,
    /// which is what makes the failure recoverable without the caller having to
    /// ask again.
    fn poison(
        &mut self,
        counters: &mut StateCounters,
        operation: &'static str,
        source: GlError,
    ) -> StateError {
        counters.lifecycle.driver_errors += 1;
        self.applied = Applied::unknown();
        StateError::backend(
            StateDomain::Textures,
            operation,
            // One Layer 1 verb was the whole group, and it was the call that
            // failed, so nothing was emitted before the failure.
            PartialApplication::new(0, 1),
            source,
        )
    }

    /// Forgets every slot that named `texture`, in the mirror and in the request.
    ///
    /// This runs before the backend deletes the name, which is the ordering that
    /// matters: an entry still naming the identity after the deletion could be hit
    /// by a name the driver reuses.  The slots become unknown rather than known
    /// unbound, because whether the driver clears the binding of a texture it is
    /// deleting is a driver fact this layer would have to assume, and `Unknown`
    /// costs one emit where a wrong `Known(None)` costs a missing one.
    ///
    /// A request naming the deleted identity is dropped rather than kept.  The
    /// request can no longer be satisfied, so keeping it would turn a caller's
    /// harmless ordering -- ask, then delete -- into a doomed call that Layer 1
    /// refuses on the next reconcile.  The caller's next request for that slot
    /// re-establishes it, and that one it makes knowing what is live.
    fn forget_texture(&mut self, texture: TextureId) {
        self.applied
            .textures
            .retain(|_, bound| *bound != Some(texture));
        if matches!(self.desired, Request::Texture { texture: Some(requested), .. } if requested == texture)
        {
            self.desired = Request::Nothing;
        }
    }

    /// Forgets every unit that named `sampler`, in the mirror and in the request.
    fn forget_sampler(&mut self, sampler: SamplerId) {
        self.applied
            .samplers
            .retain(|_, bound| *bound != Some(sampler));
        if matches!(self.desired, Request::Sampler { sampler: Some(requested), .. } if requested == sampler)
        {
            self.desired = Request::Nothing;
        }
    }

    /// Forgets the whole mirror, keeping the request.
    ///
    /// Nothing is emitted: the identities in this mirror belong to a context or an
    /// epoch the backend no longer accepts, and the request is kept because the
    /// caller asked for it -- the next reconcile, after restoration, is what
    /// applies it again.
    fn purge(&mut self, counters: &mut StateCounters) {
        self.applied = Applied::unknown();
        counters.lifecycle.domain_invalidations += 1;
    }
}
