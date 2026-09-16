//! The compute state domain: the storage-image bindings a dispatch reads and
//! writes.
//!
//! Responsibility: make the backend's image-unit bindings agree with what the
//! caller asked for.
//!
//! Not owned here: the compute program, the dispatch verbs, the storage-buffer
//! half of the same dispatch, and the texture objects themselves.
//!
//! # Why this mirror is one index space and nothing else
//!
//! A storage-image binding names an image unit, and the image-unit index space
//! is its own: it is not the storage-buffer binding count and not a texture
//! unit, and on the shipped profiles the three have different widths, so an
//! index legal for one is out of range for another.  That is why the three are
//! three [`BindingPoints`] rather than one table tagged with a role -- a wrong
//! image with no error anywhere is the failure this layer exists to make
//! impossible -- and it is why the domain is one field rather than a group.
//!
//! The entry is the whole [`GlStorageImageBinding`]: the texture, the level, the
//! sample count, the layering and its layer, the format and the declared access.
//! Two requests are the same request only when all of them compare equal, and
//! holding the format and the access -- which Layer 1 validates against the
//! texture and the format table and never hands to a binding point -- costs
//! nothing and can only emit a call that was not strictly needed, which is the
//! direction this layer errs in.
//!
//! Unlike the uniform binding role there is no unbind form: every bind names a
//! texture, so a unit once named is never *known to be empty*.  The mirror
//! therefore has no "cleared" entry, and the only way a unit stops being claimed
//! is an invalidation.
//!
//! # Deliberately not here
//!
//! The compute program.  GL has one current program, shared with the raster
//! pipeline domain, so a mirror of it here would be a second owner of a fact
//! another domain also moves; [`super::knowledge`]'s `Compute` row records the
//! resolution, and Layer 1 re-asserts the program every verb is about to use, so
//! the skip such a mirror would buy is already free.
//!
//! Memory visibility.  A barrier is a command whose effect is per invocation --
//! it orders the writes that have already happened -- so there is no driver value
//! to compare a request against and nothing a mirror could prove redundant.
//! Skipping one would not be an optimization; it would be dropping the ordering
//! the caller asked for.
//!
//! The storage-buffer half of the same dispatch.  Those bindings are the buffer
//! domain's storage role, which the same optional bound reaches, and a second
//! mirror of them here would be two owners of one index space.
//!
//! # Reachability
//!
//! The two entry points are bounded on [`super::GlOptionalComputeBackend`] and
//! [`GlStorageImageApi`], because the verbs they mirror live on a trait the
//! command-backend bound does not include.  The *state* is a field of every
//! machine: a struct's fields cannot be conditional on a bound, and the two
//! alternatives are worse than two empty maps -- a second machine type would
//! duplicate every domain wiring for one optional group, and an `Option` field
//! would put a runtime branch where the bound already makes the request
//! inexpressible.  What the bound removes is not the field but the ability to
//! *ask*: a profile without the optional domains has no way to record a want
//! here and no way to reconcile one, so this group can never hold anything and
//! never emits a call.
//!
//! # Invalidations
//!
//! [`StateEvent::TextureDeleted`] forgets every unit that named the texture --
//! both what the driver was believed to hold and what the caller asked for -- and
//! emits nothing: a binding is not an object, so there is nothing to destroy.  The
//! want goes because no call can satisfy it; keeping it would make this domain
//! re-emit a request the caller never made and report Layer 1's refusal as a
//! failure of the caller's transition.  [`StateEvent::DomainFailed`] naming this
//! domain has nothing left to forget, because the emit that failed already cleared
//! the applied state before it returned.  And a whole-mirror event -- or a raw
//! scope that declared this domain -- forgets what the driver holds everywhere
//! while keeping every want, because every object those name still exists.
//! [`super::binding`] draws the line once for both binding domains.

use crate::webgl2::api::{GlStorageImageApi, GlStorageImageBinding};

use super::GlStateBackend;
use super::binding::BindingPoints;
use super::counters::StateCounters;
use super::error::StateError;
use super::event::StateEvent;
use super::knowledge::{DirtyDomains, ExecutionMode, StateDomain};

/// The image units a compute dispatch can bind, as one mirror.
#[derive(Debug)]
pub(crate) struct ComputeState {
    /// The indexed storage-image binding points.
    images: BindingPoints<GlStorageImageBinding>,
    mode: ExecutionMode,
}

impl ComputeState {
    /// A domain that has applied nothing.
    ///
    /// The mirror starts empty, so the first request for any unit emits the call
    /// that establishes it.  That is the trade the machine makes for every
    /// domain: one redundant call per unit per context, in exchange for a mirror
    /// that cannot be wrong about a unit it never set.
    pub(crate) fn new(mode: ExecutionMode) -> Self {
        Self {
            images: BindingPoints::default(),
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

    /// Records that the caller wants `binding` to hold `image`.
    ///
    /// The arguments are the verb's own, in the verb's own order, so a machine
    /// entry point forwards them without deciding anything.  Nothing is
    /// validated here either: the unit, the layering and the exact format-access
    /// fact are Layer 1's checks, made on every path including the ones this
    /// reconcile skips.
    pub(crate) fn bind_storage_image(
        &mut self,
        binding: u32,
        image: GlStorageImageBinding,
        counters: &mut StateCounters,
    ) {
        self.images.record(binding, image, counters);
    }

    /// Makes the backend's image units agree with the caller's.
    ///
    /// Only reachable through the optional bound, because the verb it mirrors is
    /// optional: a caller on a profile without that domain has no way to record
    /// a storage-image want, and no empty group exists for one to sit in.
    pub(crate) fn reconcile(
        &mut self,
        backend: &mut impl GlStorageImageApi,
        counters: &mut StateCounters,
    ) -> Result<(), StateError> {
        self.images.settle(
            StateDomain::Compute,
            "bind-storage-image",
            self.mode,
            counters,
            |binding, image| backend.bind_storage_image(binding, *image),
        )
    }

    /// Reacts to an invalidation.
    ///
    /// The backend is unused: a binding is not an object, so this domain owns
    /// nothing to destroy and never reaches the driver from here.  The parameter
    /// stays because the machine dispatches every domain's invalidation the same
    /// way, and a domain that dropped it would need a special case at the one
    /// call site that exists to be uniform.
    pub(crate) fn invalidate(
        &mut self,
        _backend: &mut impl GlStateBackend,
        event: &StateEvent,
        counters: &mut StateCounters,
    ) {
        match event {
            StateEvent::TextureDeleted(texture) => {
                // Before the caller asks the backend to delete the name: a unit
                // that still named the texture must stop claiming to know what
                // the driver holds before that name can belong to something else.
                // The want goes with the belief -- no call can bind a deleted
                // texture -- which is the rule [`super::binding`] states once for
                // this domain and the buffer one.
                self.images
                    .forget_object_where(|image| image.texture == *texture);
            }
            // A group that failed partway is already unknown -- the emit that
            // failed cleared this domain's applied state before it returned --
            // so there is nothing left for this arm to forget.
            StateEvent::DomainFailed(domain) if *domain == StateDomain::Compute => {}
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
        if scope.contains(StateDomain::Compute) {
            self.images.forget_applied();
            counters.lifecycle.domain_invalidations += 1;
        }
    }
}

#[cfg(test)]
mod tests;
