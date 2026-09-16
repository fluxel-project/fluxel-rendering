//! The geometry state domain: which vertex-array object is bound, and the
//! vertex-array objects derived from a complete vertex layout.
//!
//! Responsibility: make the backend's bound vertex array agree with what the
//! caller asked for, and reuse one derived vertex-array object for one layout.
//!
//! Not owned here: the buffer contents and the buffer bindings a vertex array
//! records (buffers), and the pipeline's attribute declarations (pipeline).
//! The derivation is keyed by the layout value the caller supplies, and the
//! buffers it names are the entry's dependencies.
//!
//! # The Layer 1 call surface this domain is built on
//!
//! - *Creating* a vertex array takes the structural layout and stores it.  The
//!   object is a pure function of that layout, so one layout needs one object
//!   however many times it is bound, and two layouts that differ in any part of
//!   the structure need two.
//! - *Binding* re-emits the whole attribute description from the layout the
//!   object stores together with the buffer bindings the call was given, and
//!   records the index binding on the object.  A vertex array's driver state is
//!   therefore the input it was last bound with, not the layout it was created
//!   from -- which is why both the desired value and the applied claim below
//!   carry the whole input rather than only the layout.
//! - A bind leaves the generic array-buffer binding point and the element-array
//!   binding point at values Layer 1 does not promise, because re-emitting the
//!   attribute description reaches the array-buffer point once per slot.  This
//!   domain mirrors neither point and no verb it can call names either, so the
//!   reconcile result reports them as unknown rather than holding an opinion.
//! - An *indexed draw* takes its index binding from the vertex array the
//!   installed pipeline recorded, not from the one currently bound, so a caller
//!   must hand a pipeline the same array it asks this domain to bind.
//! - *Installing a pipeline* binds that array without re-emitting the attribute
//!   description.  It cannot be relied on to establish a vertex input: the bind
//!   this domain emits is what enables the attributes, and a caller that skips
//!   it gets an array with nothing enabled.
//!
//! # Why one derived object per layout
//!
//! The layout is the whole of what creating an array depends on, and a renderer
//! that draws many meshes with one vertex layout -- each mesh with its own
//! vertex buffer -- must not pay for a new array per mesh.  So the derivation is
//! keyed by the layout alone, and the buffers an array was pointed at are the
//! entry's *dependencies* rather than part of its key.  A key that left out any
//! part of the structure would return an array that describes a different
//! layout, and the bind that follows would re-emit that wrong description with
//! no error anywhere.
//!
//! # What this domain does not do
//!
//! It does not hold the buffer contents, the buffer bindings, or the attribute
//! declarations a program makes: the first two belong to the buffer domain and
//! the third to the pipeline domain.  It does not decide what a draw needs, and
//! it does not validate: Layer 1 rejects an invalid layout, a binding to a
//! buffer that is not live, an out-of-range offset and a stale identity on every
//! path including the ones this domain skips.

use crate::webgl2::api::{
    BufferId, GlIndexBinding, GlVertexBufferBinding, GlVertexLayout, VertexArrayId,
};

pub(super) mod cache;

use super::GlStateBackend;
use super::cache::CacheBudget;
use super::counters::StateCounters;
use super::error::{PartialApplication, StateError};
use super::event::StateEvent;
use super::knowledge::{DriverKnowledge, ExecutionMode, StateDomain};
use cache::VertexArrayCache;

/// The complete input that establishes one vertex-array binding.
///
/// The layout is what a vertex array is derived from; the bindings and the
/// index source are what a bind points it at.  Together they are the whole of
/// the backend's vertex input, which is why both the desired value and the
/// applied claim hold this type rather than a summary of it: a request that
/// changed one buffer offset is a different input, and comparing anything less
/// than the whole value would skip the bind that offset needs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct VertexInput {
    /// The structural layout the array must describe.
    pub layout: GlVertexLayout,
    /// One entry per slot of the layout: the buffer that slot reads, and the
    /// byte offset within it that the slot starts at.
    pub bindings: Vec<GlVertexBufferBinding>,
    /// The index source, or `None` for a non-indexed input.
    pub index: Option<GlIndexBinding>,
}

/// What a geometry reconcile changed about the backend.
///
/// The two binding points a bind reaches are reported rather than applied
/// because they are not this domain's to own: this layer can only say that they
/// are no longer describable, not what they now hold.  `#[must_use]` is what
/// makes a caller that mirrors either of them decide what to do about it here,
/// instead of silently keeping a belief the bind just invalidated.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[must_use = "a bind leaves the generic array-buffer and element-array binding points at values Layer 1 does not promise; a caller that mirrors either one must treat it as unknown"]
pub(crate) struct GeometryEffects {
    /// This reconcile emitted the bind that establishes the desired input.
    pub vertex_array_bound: bool,
    /// The bind reached the generic array-buffer binding point, because the
    /// bound layout has at least one slot.  Layer 1 leaves that point at the
    /// last slot's buffer and does not promise which.  The element-array
    /// binding point is reached by every bind that is emitted, for the same
    /// reason and with the same consequence.
    pub array_buffer_binding_unknown: bool,
}

impl GeometryEffects {
    /// Nothing about the backend's vertex input changed.
    const NONE: Self = Self {
        vertex_array_bound: false,
        array_buffer_binding_unknown: false,
    };
}

/// What the caller asked the vertex input to be.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Desired {
    /// The input to have bound, or `None` when the caller has asked for none.
    input: Option<VertexInput>,
}

/// The input the backend's vertex input was last established with.
#[derive(Clone, Debug, Eq, PartialEq)]
struct AppliedInput {
    /// The whole input, so the redundant-request check compares the value the
    /// caller supplied rather than a summary of it.
    input: VertexInput,
    /// The array object the bind named.
    vertex_array: VertexArrayId,
    /// Whether this domain created the array and must therefore destroy it.
    ///
    /// True only when the derivation could not be retained -- the oracle mode,
    /// or a budget the record does not fit.  In that case
    /// the object is this domain's alone: no cache holds it and no caller has
    /// seen it, so dropping the claim without destroying it would leak it.
    owned: bool,
}

/// What the backend's vertex input currently is.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Applied {
    input: DriverKnowledge<AppliedInput>,
}

impl Applied {
    /// The backend holds no vertex input this layer put there.
    fn unknown() -> Self {
        Self {
            input: DriverKnowledge::Unknown,
        }
    }

    /// Whether the backend is known to hold exactly this input.
    ///
    /// This is [`DriverKnowledge`]'s own rule -- an unknown value never agrees,
    /// and a known one agrees by equality -- applied to the part of the claim
    /// that describes driver state.  The array identity and the ownership flag
    /// are this domain's bookkeeping and take no part in the comparison, so
    /// they cannot make two different inputs look alike.
    fn agrees(&self, desired: &VertexInput) -> bool {
        self.input.get().is_some_and(|held| held.input == *desired)
    }
}

/// The bound vertex input and the vertex arrays derived for it.
#[derive(Debug)]
pub(crate) struct GeometryState {
    desired: Desired,
    applied: Applied,
    vertex_arrays: VertexArrayCache,
    mode: ExecutionMode,
}

impl GeometryState {
    /// A geometry domain that has bound nothing.
    pub(crate) fn new(budget: CacheBudget, mode: ExecutionMode) -> Self {
        Self {
            desired: Desired::default(),
            applied: Applied::unknown(),
            vertex_arrays: VertexArrayCache::new(budget),
            mode,
        }
    }

    /// The execution mode this domain runs.
    ///
    /// The domain holds the mode rather than a [`CacheMode`] because the two are
    /// not the same decision: retention governs whether a vertex array this
    /// domain derived is kept, and skipping governs whether a bind it can prove
    /// redundant is re-emitted.  A domain given only the retention half would
    /// skip unconditionally, which is what makes an oracle trace unusable as a
    /// comparison.  The retention policy is derived where a cache needs it, via
    /// [`ExecutionMode::cache`].
    ///
    /// [`CacheMode`]: super::cache::CacheMode
    pub(crate) const fn mode(&self) -> ExecutionMode {
        self.mode
    }

    /// Records that the caller wants this input bound.
    ///
    /// Nothing is emitted here.  The next [`GeometryState::reconcile`] decides
    /// whether a call is needed, which is what keeps the redundancy check in one
    /// place instead of once per entry point.
    pub(crate) fn set_vertex_input(&mut self, input: VertexInput) {
        self.desired.input = Some(input);
    }

    /// The input the caller last asked for, if any.
    pub(crate) fn desired_input(&self) -> Option<&VertexInput> {
        self.desired.input.as_ref()
    }

    /// Releases the claim on the driver's input, keeping what the caller asked
    /// for.
    ///
    /// The machine calls this after a pipeline install, which is the one act
    /// outside this domain that can change the driver's vertex input: Layer 1
    /// binds the array a pipeline names without re-emitting its attribute
    /// description, and that array may be one this domain never derived.  The
    /// want is kept, so the next [`GeometryState::reconcile`] re-binds and the
    /// driver ends up holding the input the caller asked for; dropping the want
    /// instead would leave whatever the pipeline bound with no error anywhere.
    ///
    /// The claim is *released* rather than forgotten, and that is not the same
    /// act: a claim can be the only name an array object has.  An array the cache
    /// could not retain is one no record keys, so the claim is the last thing
    /// holding it -- forgetting it there would leak the object silently, which is
    /// the failure mode [`GeometryState::release_claim`] exists to prevent.
    /// Nothing is lost by destroying it: the next reconcile derives an array for
    /// this input again, and the array it derives is the one it binds.
    pub(crate) fn vertex_input_unknown(
        &mut self,
        backend: &mut impl GlStateBackend,
        counters: &mut StateCounters,
    ) {
        self.release_claim(backend, counters);
    }

    /// The input the backend is known to hold, if any.
    pub(crate) fn applied_input(&self) -> Option<&VertexInput> {
        self.applied.input.get().map(|held| &held.input)
    }

    /// The array object the applied input is held in, if any.
    ///
    /// A caller needs this to name the same array in a pipeline, which is the
    /// one thing an indexed draw takes its index binding from.
    pub(crate) fn applied_vertex_array(&self) -> Option<VertexArrayId> {
        self.applied.input.get().map(|held| held.vertex_array)
    }

    /// The number of retained vertex-array records.
    pub(crate) fn vertex_array_records(&self) -> usize {
        self.vertex_arrays.len()
    }

    /// Returns the vertex array for an input, deriving it if needed.
    ///
    /// The second half of the pair is whether the caller owns the result and
    /// must destroy it: true when the cache is disabled, and true when its
    /// budget could not keep the array.  Both are ordinary rather than error
    /// paths -- the caller gets a usable array either way, and only pays for
    /// building it again next time.  See
    /// [`cache::VertexArrayCache::vertex_array_for`].
    pub(crate) fn vertex_array_for(
        &mut self,
        backend: &mut impl GlStateBackend,
        input: &VertexInput,
        counters: &mut StateCounters,
    ) -> Result<(VertexArrayId, bool), StateError> {
        let derivation =
            self.vertex_arrays
                .vertex_array_for(backend, input, self.mode.cache(), counters)?;
        for (_, vertex_array) in derivation.removed {
            // The derivation made room for itself by evicting records, and one
            // of them may be the array the mirror believes is bound.  An eviction
            // is the only way a claim can come to name an object that no longer
            // exists, because this loop is the only place a derived array is
            // destroyed.
            if self.names_array(vertex_array) {
                self.applied.input.invalidate();
            }
            destroy(backend, vertex_array, counters);
        }
        Ok((derivation.vertex_array, derivation.owned))
    }

    /// Whether the mirror proves this request redundant under this mode.
    ///
    /// The two halves are one decision.  An oracle machine runs the same domains
    /// with skipping disabled, so a domain that skipped in oracle mode would emit
    /// a trace no mirror-free machine could have produced.
    fn skippable(&self, agrees: bool) -> bool {
        self.mode.may_skip() && agrees
    }

    /// Makes the backend's vertex input match the desired one.
    ///
    /// The derive-and-bind pair is one transition: the array comes from the
    /// layout, and the bind is what points it at the buffers.  A caller that
    /// only wants the identity -- to name it in a pipeline -- asks
    /// [`GeometryState::vertex_array_for`] instead, and is then responsible for
    /// asking this domain to bind it before a draw, because installing a
    /// pipeline binds the array without enabling anything in it.
    pub(crate) fn reconcile(
        &mut self,
        backend: &mut impl GlStateBackend,
        counters: &mut StateCounters,
    ) -> Result<GeometryEffects, StateError> {
        counters.domain(StateDomain::Geometry).request();

        let Some(wanted) = self.desired.input.as_ref() else {
            // No input was asked for.  There is no verb that unbinds a vertex
            // input, and a caller that wants none simply does not draw, so the
            // request is answered with nothing rather than with a call.
            counters.domain(StateDomain::Geometry).skip();
            return Ok(GeometryEffects::NONE);
        };
        if self.skippable(self.applied.agrees(wanted)) {
            counters.domain(StateDomain::Geometry).skip();
            return Ok(GeometryEffects::NONE);
        }
        if !self.applied.input.is_known() {
            // The bind is being emitted because the driver's input is not known
            // rather than because the caller changed it: the first request after
            // creation, or the first after an invalidation.
            counters.domain(StateDomain::Geometry).recover();
        }

        // Storing the input costs one allocation, and the plan's third
        // optimization candidate is judged on exactly this counter.
        let wanted = wanted.clone();
        let array_buffer_reached = !wanted.layout.buffers.is_empty();
        // The claim this reconcile is about to replace is released first, so the
        // applied state is never a mixture of two inputs, and a derivation that
        // fails leaves the driver's input unknown rather than described.
        self.release_claim(backend, counters);
        let derivation =
            self.vertex_arrays
                .vertex_array_for(backend, &wanted, self.mode.cache(), counters)?;
        // The claim was released above, so a record this derivation evicted is
        // one nothing here still refers to: it is destroyed rather than
        // repaired, and the identity the bind below names is the new one.
        for (_, evicted) in derivation.removed {
            destroy(backend, evicted, counters);
        }
        match backend.bind_vertex_array(derivation.vertex_array, &wanted.bindings, wanted.index) {
            Ok(()) => {
                counters.domain(StateDomain::Geometry).emit();
                counters.allocated();
                self.applied.input.set(AppliedInput {
                    input: wanted,
                    vertex_array: derivation.vertex_array,
                    owned: derivation.owned,
                });
                Ok(GeometryEffects {
                    vertex_array_bound: true,
                    array_buffer_binding_unknown: array_buffer_reached,
                })
            }
            Err(source) => {
                counters.lifecycle.driver_errors += 1;
                // The bind did not happen, so what the driver holds is unknown
                // rather than unchanged -- it may hold half of this input.
                if derivation.owned {
                    destroy(backend, derivation.vertex_array, counters);
                }
                Err(StateError::backend(
                    StateDomain::Geometry,
                    "bind-vertex-array",
                    PartialApplication::new(0, 1),
                    source,
                ))
            }
        }
    }

    /// Reacts to an invalidation.
    ///
    /// The two halves are handled separately because they are invalidated by
    /// different things: a record is dropped when a resource it depends on is
    /// deleted, while the *claim* is dropped when the input it describes is no
    /// longer one the driver can be believed to hold -- a deleted buffer, a
    /// failed buffer domain, or a scope that declared it touched this domain.
    pub(crate) fn invalidate(
        &mut self,
        backend: &mut impl GlStateBackend,
        event: &StateEvent,
        counters: &mut StateCounters,
    ) {
        if event.invalidates_everything() {
            // The identities in these records belong to an epoch the backend no
            // longer accepts, so there is nothing to destroy through them, and
            // asking would be a call with a stale object.  The desired input is
            // kept: the caller asked for it, and the next reconcile after
            // restoration is what binds it again.
            self.applied.input.invalidate();
            counters.lifecycle.domain_invalidations += 1;
            self.vertex_arrays.invalidate(event, counters);
            return;
        }

        // A claim whose input names a buffer that is going away describes state
        // the driver cannot be believed to still hold: the array's attribute
        // description points into a name that may be reused.  Nothing has to be
        // dropped for that -- the bind re-emits the whole description -- but the
        // claim has to stop being one this domain skips on.
        match event {
            StateEvent::BufferDeleted(buffer) => {
                if self.applied_input_names_buffer(*buffer) {
                    self.release_claim(backend, counters);
                }
            }
            // A buffer domain that failed partway leaves which bindings are in
            // force unknown, so no claim built from buffer bindings survives it.
            // The derived arrays themselves do not: a record is keyed by a
            // structure that names no buffer, so a binding failure cannot make
            // one describe the wrong layout.
            StateEvent::DomainFailed(StateDomain::Buffers) => self.release_claim(backend, counters),
            // A scope that declares it touched this domain may have bound any
            // array, so the claim is gone.  The derived arrays are kept: they
            // are objects this layer created, and a scope that deleted one
            // reports the deletion through the event above rather than by
            // leaving this domain to guess.
            StateEvent::ScopedRawAccess(scope)
                if scope.domains().contains(StateDomain::Geometry) =>
            {
                self.release_claim(backend, counters);
            }
            _ => {}
        }

        // A `VertexArrayDeleted` names an object its owner has already
        // destroyed, so the record holding it is dropped without being destroyed
        // through: asking to delete it again would be a call on an identity the
        // backend no longer accepts.
        let already_gone = match event {
            StateEvent::VertexArrayDeleted(vertex_array) => Some(*vertex_array),
            _ => None,
        };
        for (_, vertex_array) in self.vertex_arrays.invalidate(event, counters) {
            if already_gone != Some(vertex_array) {
                destroy(backend, vertex_array, counters);
            }
            if self.names_array(vertex_array) {
                self.applied.input.invalidate();
            }
        }
    }

    /// Destroys the arrays this domain derived.
    ///
    /// Called when the caller is done with the context, which is the last moment
    /// the identities in these records can still be destroyed through.  Unlike
    /// the whole-mirror path, this is not recoverable and both halves are
    /// cleared: shutdown says the caller is finished, so a later reconcile must
    /// not re-bind an input the caller never asked for again.
    pub(crate) fn shutdown(
        &mut self,
        backend: &mut impl GlStateBackend,
        counters: &mut StateCounters,
    ) {
        self.release_claim(backend, counters);
        for (_, vertex_array) in self.vertex_arrays.drain(counters) {
            destroy(backend, vertex_array, counters);
        }
        self.desired = Desired::default();
        self.applied = Applied::unknown();
    }

    /// Whether the claim names this array object.
    fn names_array(&self, vertex_array: VertexArrayId) -> bool {
        self.applied
            .input
            .get()
            .is_some_and(|held| held.vertex_array == vertex_array)
    }

    /// Whether the claim's input reads from this buffer, as a slot source or as
    /// the index source.
    fn applied_input_names_buffer(&self, buffer: BufferId) -> bool {
        self.applied.input.get().is_some_and(|held| {
            held.input
                .bindings
                .iter()
                .any(|binding| binding.buffer == buffer)
                || held.input.index.is_some_and(|index| index.buffer == buffer)
        })
    }

    /// Forgets the claim, destroying the array it owned.
    ///
    /// An owned array is one no cache holds, so forgetting the claim is the last
    /// reference to it and the only chance to destroy it.  A retained array is
    /// the cache's and must not be destroyed here: the record that holds it
    /// outlives the claim.
    fn release_claim(&mut self, backend: &mut impl GlStateBackend, counters: &mut StateCounters) {
        let held = core::mem::replace(&mut self.applied.input, DriverKnowledge::Unknown);
        if let DriverKnowledge::Known(applied) = held {
            if applied.owned {
                destroy(backend, applied.vertex_array, counters);
            }
        }
    }
}

/// Destroys one derived vertex array, counting a failure rather than raising it.
///
/// A failed deletion is not something the caller can act on -- the record that
/// named the object is already gone -- and the alternative to counting it is
/// leaking the object silently.  The count is what a leak report reads;
/// propagating it would turn a cleanup detail into a frame failure.
fn destroy(
    backend: &mut impl GlStateBackend,
    vertex_array: VertexArrayId,
    counters: &mut StateCounters,
) {
    if backend.destroy_vertex_array(vertex_array).is_err() {
        counters.lifecycle.driver_errors += 1;
    }
}

#[cfg(test)]
mod tests;
