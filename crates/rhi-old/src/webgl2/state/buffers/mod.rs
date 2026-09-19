//! The buffer state domain: which buffer is bound in each binding role, and
//! what range of it is bound.
//!
//! Responsibility: make the backend's buffer binding points agree with what the
//! caller asked for, per role, and leave no scratch binding behind.
//!
//! Not owned here: the buffer objects and their storage (Layer 1's resource
//! tables), and the vertex-array objects that record their own bindings
//! (geometry).  This domain has no derived cache, and the reason is the same one
//! that gives it two maps rather than one: a binding is not an object.  A cache
//! exists to hand back an identity this layer created and must therefore
//! destroy, to key a derivation by a structural value, and to bound the bytes it
//! retains.  Nothing here creates an object, so there is nothing to key, budget,
//! evict or destroy -- and a domain that grew a cache would be buying that whole
//! machinery for a lookup a map already answers.
//!
//! # The two roles, and why they are two index spaces
//!
//! Layer 1's mirrorable set is exactly the binding points a verb can name -- a
//! binding point no verb can name is one this layer must not hold an opinion
//! about -- and for buffers that is two of them:
//!
//! | Role | Verb | Index space | Entry |
//! |---|---|---|---|
//! | uniform | `bind_uniform_buffer(index, buffer, offset, size)` | the indexed uniform binding points, as wide as the discovered uniform binding count | the buffer (or an explicit unbind), the offset and the size |
//! | storage | `bind_storage_buffer(binding, range)` | the indexed storage binding points, as wide as the discovered storage binding count -- a *different* count | the buffer, the offset, the size and the declared usage |
//!
//! The index spaces are unrelated, and on the shipped profiles they have
//! different widths, so an index legal for one role is out of range for the
//! other.  A single table keyed by index and tagged with a role would bind the
//! wrong slot the moment both roles used the same number -- a wrong image with
//! no error anywhere, which is the failure this layer exists to make
//! impossible.  One map per role is the whole of the fix, and it is also why the
//! role is a compile-time fact here rather than a run-time field.
//!
//! The roles differ in shape too, and the mirror keeps the difference rather
//! than flattening it: a uniform binding point has an *unbind* form, so its
//! entry distinguishes "the driver holds this slot unbound" from "the driver's
//! value here is not known", while a storage binding always names a buffer and
//! has no unbind form at all.
//!
//! The machinery that holds both maps is [`super::binding`]'s, shared with the
//! compute domain's storage-image role because the two are the same problem:
//! what belongs to this module is *which* index spaces exist, which verb settles
//! each one, what an invalidation forgets, and which domain the settle is
//! accounted against.
//!
//! # What the mirror holds, and what makes two requests the same one
//!
//! An entry holds everything the verb carried.  For the uniform role that is the
//! buffer, the offset and the size *verbatim*: `size == 0` means "through the end
//! of the allocation" and is a different request from `size == N` even when the
//! two resolve to the same bytes, because the verb carries the number the caller
//! wrote.  Two requests are the same one only when the whole entry compares
//! equal, and the predicate is [`DriverKnowledge::agrees`] -- never a structural
//! shortcut, and never a comparison against a value the mirror merely *assumed*
//! the driver holds.
//!
//! For the storage role the entry also holds the declared usage, which Layer 1
//! validates against the buffer's allocation and never hands to a binding point:
//! the binding point holds a buffer and a byte range, and the usage is a
//! descriptor fact about how the shader reads it.  Holding it costs nothing and
//! comparing it can only emit a call that was not strictly needed -- one
//! redundant call in a rare case, never a skipped call that was needed -- which
//! is the direction this layer errs in, and it keeps the skip predicate the one
//! predicate [`DriverKnowledge::agrees`] defines for every domain.
//!
//! # What is deliberately not mirrored
//!
//! The generic scratch targets.  Buffer allocation, upload, readback and
//! buffer-to-buffer copy all use binding points the caller cannot name: each
//! provider binds the target its own transfer needs immediately before the call
//! and leaves it bound, and no verb in that domain accepts a target.  The
//! transfer domain records that contract on its own trait; this module's half of
//! it is that the mirror covers exactly the two binding points a verb can name.
//!
//! The pixel-store parameters.  Layer 1's transfer verbs take an explicit
//! per-call layout and scope the pixel-store state around the transfer, and the
//! only pixel-store method on the trait is a getter, so there is no desired value
//! a caller could state and no verb for this domain to emit.  A mirror of values
//! no verb can set would be a second, weaker copy of state Layer 1 already owns.
//!
//! # Invalidations
//!
//! Three of [`super::event`]'s rows reach this domain.
//! [`StateEvent::BufferDeleted`] forgets the slots that still name the buffer in
//! *both* roles -- the belief and the want -- and emits nothing: a binding is not
//! an object, so there is nothing to destroy.  [`StateEvent::DomainFailed`]
//! naming this domain has nothing left to forget, because the emit that failed
//! already cleared the applied state before it returned.  And a whole-mirror
//! event -- or a raw scope that declared this domain -- forgets what the driver
//! holds everywhere.
//!
//! The two forget *different amounts*, and the difference is whether the want is
//! still satisfiable.  A deletion takes the want with the belief: no call can
//! establish a binding to an object that is gone, so keeping the want would have
//! this layer re-emit a request the caller never made and then report Layer 1's
//! refusal as a failure of the caller's next transition -- repeatedly, until the
//! caller happened to rebind that slot.  A whole-mirror event keeps every want:
//! every object it names still exists, so the next settle re-establishes exactly
//! what the caller asked for, which is the rule the session domain gives a
//! context loss for the same reason.  [`super::binding`] states the rule once for
//! both roles and every domain that shares them.

use crate::webgl2::api::{BufferId, GlStorageBufferApi, GlStorageBufferRange};

use super::GlStateBackend;
use super::binding::BindingPoints;
use super::counters::StateCounters;
use super::error::StateError;
use super::event::StateEvent;
use super::knowledge::{DirtyDomains, ExecutionMode, StateDomain};

/// One indexed uniform binding point: everything `bind_uniform_buffer` carried.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct UniformBinding {
    /// The buffer the slot names, or `None` for the unbind form.
    ///
    /// The unbind form is a request of its own rather than the absence of one: a
    /// slot the caller unbound is a slot the driver is *known* to hold nothing
    /// in, which is why the entry carries this flag instead of the slot being
    /// removed from the map.
    buffer: Option<BufferId>,
    /// Byte offset of the bound range; zero for an unbind.
    offset: u32,
    /// Byte size of the bound range; zero means through the allocation end, and
    /// is a distinct request from an explicit size that reaches the same end.
    size: u32,
}

/// The indexed buffer binding points, one mirror per role.
#[derive(Debug)]
pub(crate) struct BuffersState {
    /// The required role: the indexed uniform binding points.
    uniform: BindingPoints<UniformBinding>,
    /// The optional role: the indexed storage binding points.
    storage: BindingPoints<GlStorageBufferRange>,
    mode: ExecutionMode,
}

impl BuffersState {
    /// A domain that has applied nothing.
    ///
    /// Both roles start with no desired and no applied entry, which means the
    /// first request for any slot emits the call that establishes it.  That is
    /// the trade the machine makes for every domain: one redundant call per slot
    /// per context, in exchange for a mirror that cannot be wrong about a binding
    /// point it never set.
    pub(crate) fn new(mode: ExecutionMode) -> Self {
        Self {
            uniform: BindingPoints::default(),
            storage: BindingPoints::default(),
            mode,
        }
    }

    /// The execution mode this domain runs.
    ///
    /// This domain derives no objects and so has no cache and no retention
    /// policy to derive: the mode reaches it for the skipping half alone.  A
    /// domain that skipped in oracle mode would emit a trace no mirror-free
    /// machine could have produced, which is what makes the differential
    /// comparison meaningless.
    pub(crate) const fn mode(&self) -> ExecutionMode {
        self.mode
    }

    /// Records that the caller wants `index` to hold `buffer`'s byte range.
    ///
    /// The arguments are the verb's own, in the verb's own order, so a machine
    /// entry point forwards them without deciding anything -- including
    /// `buffer: None`, which is the unbind form and carries no range.
    pub(crate) fn bind_uniform_buffer(
        &mut self,
        index: u32,
        buffer: Option<BufferId>,
        offset: u32,
        size: u32,
        counters: &mut StateCounters,
    ) {
        self.uniform.record(
            index,
            UniformBinding {
                buffer,
                offset,
                size,
            },
            counters,
        );
    }

    /// Records that the caller wants `binding` to hold `range`.
    ///
    /// Only reachable through [`BuffersState::reconcile_storage`], because the
    /// verb it mirrors is optional: a caller on a profile without that domain has
    /// no way to record a storage want, and no empty storage group exists for one
    /// to sit in.
    pub(crate) fn bind_storage_buffer(
        &mut self,
        binding: u32,
        range: GlStorageBufferRange,
        counters: &mut StateCounters,
    ) {
        self.storage.record(binding, range, counters);
    }

    /// Makes the backend's indexed uniform bindings agree with the caller's.
    ///
    /// This is the whole of the required domain.  Nothing is validated here: the
    /// index, the offset alignment, the range and the buffer's usage are Layer 1's
    /// checks, made on every path including the ones this reconcile skips, and a
    /// second definition of validity here would drift from that one and win by
    /// being cheaper.
    pub(crate) fn reconcile(
        &mut self,
        backend: &mut impl GlStateBackend,
        counters: &mut StateCounters,
    ) -> Result<(), StateError> {
        self.uniform.settle(
            StateDomain::Buffers,
            "bind-uniform-buffer",
            self.mode,
            counters,
            |index, binding| {
                backend.bind_uniform_buffer(index, binding.buffer, binding.offset, binding.size)
            },
        )
    }

    /// Makes the backend's indexed storage bindings agree with the caller's.
    ///
    /// Separate from [`BuffersState::reconcile`] rather than folded into it,
    /// because the storage role's verb lives on a trait the command backend bound
    /// does not include: a profile without that domain must have no storage state
    /// group at all, not an empty one, and a bound is what keeps that a
    /// compile-time fact.  A machine whose backend has the optional domains calls
    /// this after [`BuffersState::reconcile`], so a failure in the required role
    /// is reported before the optional one emits.
    ///
    /// It counts its own request: the two roles are two logical requests even when
    /// one transition asks for both.
    pub(crate) fn reconcile_storage(
        &mut self,
        backend: &mut impl GlStorageBufferApi,
        counters: &mut StateCounters,
    ) -> Result<(), StateError> {
        self.storage.settle(
            StateDomain::Buffers,
            "bind-storage-buffer",
            self.mode,
            counters,
            |binding, range| backend.bind_storage_buffer(binding, *range),
        )
    }

    /// Reacts to an invalidation.
    ///
    /// The backend is unused: a binding is not an object, so this domain owns
    /// nothing to destroy and never reaches the driver from here.  The parameter
    /// stays because the machine dispatches every domain's invalidation the same
    /// way, and a domain that dropped it would need a special case at the one call
    /// site that exists to be uniform.
    pub(crate) fn invalidate(
        &mut self,
        _backend: &mut impl GlStateBackend,
        event: &StateEvent,
        counters: &mut StateCounters,
    ) {
        match event {
            StateEvent::BufferDeleted(buffer) => {
                // Both roles, before the caller asks the backend to delete the
                // name: a slot that still named the buffer must stop claiming to
                // know what the driver holds before that name can belong to
                // something else, and the want for it goes too, because no call
                // can satisfy a want that names a deleted object.  The two
                // predicates differ because the two entry types carry the buffer
                // differently -- one as a field that may be the unbind form, one
                // as a plain identity.
                self.uniform
                    .forget_object_where(|binding| binding.buffer == Some(*buffer));
                self.storage
                    .forget_object_where(|range| range.buffer == *buffer);
            }
            // A group that failed partway is already unknown -- the emit that
            // failed cleared this domain's applied state before it returned -- so
            // there is nothing left for this arm to forget.  The event exists to
            // warn the domains that depended on the failed one's result.
            StateEvent::DomainFailed(domain) if *domain == StateDomain::Buffers => {}
            _ => {}
        }

        // The whole-mirror events, and the raw scopes that named this domain.  An
        // undeclared scope reports every domain, which is the contract: a caller
        // that cannot say what it touched must be assumed to have touched all of
        // it.  Both readings fold into one mask so that a scope invalidating
        // everything is still one domain invalidation here.
        let scope = match event {
            StateEvent::ScopedRawAccess(scope) => scope.domains(),
            _ if event.invalidates_everything() => DirtyDomains::ALL,
            _ => DirtyDomains::EMPTY,
        };
        if scope.contains(StateDomain::Buffers) {
            self.uniform.forget_applied();
            self.storage.forget_applied();
            counters.lifecycle.domain_invalidations += 1;
        }
    }
}

#[cfg(test)]
mod tests;
