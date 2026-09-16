//! The geometry domain's vertex-array cache.
//!
//! A renderer draws many meshes through the same vertex layout, and each mesh
//! usually has its own vertex buffer.  Creating a vertex array per mesh would
//! make the driver repeat work for a structure that has not changed, so the
//! geometry domain keeps one array per layout and points it at whatever buffers
//! the caller binds.
//!
//! # Why the key is the whole layout
//!
//! `GlVertexLayout` is the complete structural input of a derivation: every
//! slot, its stride and step mode, and every attribute's location, source slot,
//! format and offset.  Two layouts that differ in any of those need different
//! arrays, and two that agree in all of them do not -- which is exactly the
//! equality Layer 1 derives.  A key that omitted one part of the structure would
//! return an array that describes a different layout, and the bind that follows
//! re-emits the array's own description, so the wrong description would reach
//! the driver with no error anywhere to trace it.
//!
//! The context stamp is carried transitively: the derived identity contains it,
//! a derived array cannot be reused across an epoch change even in principle,
//! and the epoch change purges this cache anyway.
//!
//! # Why the derivation takes the whole input
//!
//! The buffers a derived array reads come from the bind input, not from the
//! layout, so they cannot be read off the key.  They are the entry's
//! dependencies, and the derive therefore takes the whole [`VertexInput`]: an
//! entry's dependencies are fixed when the entry is created and this cache has
//! no way to revise them afterwards.  A later bind that points the same array at
//! different buffers leaves the record's dependency set describing the buffers
//! the record was created from; that is deliberate, because a record is dropped
//! conservatively and the *claim* is what tracks which buffers are in force.
//!
//! # Who destroys what
//!
//! Nobody but this cache's caller.  Every call that can remove a record hands
//! the removed pairs back rather than destroying them, because the domain is the
//! only place that knows which of the removed arrays the bound-input claim
//! names, and a cache that destroyed them itself could only report identities
//! whose objects are already gone.
//!
//! # What this module does not do
//!
//! It does not decide what is bound, does not emit the bind, and does not know
//! what a pipeline declares.  It holds one table and one derivation.

use crate::webgl2::api::{
    GlVertexAttribute, GlVertexBufferBinding, GlVertexBufferLayout, GlVertexLayout, VertexArrayId,
};

use super::super::GlStateBackend;
use super::super::cache::{CacheBudget, CacheMode, DependencySet, ResourceRef, StructuralCache};
use super::super::counters::StateCounters;
use super::super::error::{PartialApplication, StateError};
use super::super::event::StateEvent;
use super::super::knowledge::StateDomain;
use super::VertexInput;

/// The estimated retained cost of one vertex-array record.
///
/// The value is a layout-shaped estimate, not a measurement of driver memory:
/// what is being bounded is this layer's own bookkeeping, and the driver's
/// vertex-array storage is not something this layer can see.  The number is
/// dominated by the per-slot and per-attribute parts, which are the parts that
/// actually grow with the key.
fn record_bytes(layout: &GlVertexLayout) -> u64 {
    let slot = core::mem::size_of::<GlVertexBufferLayout>() as u64
        + core::mem::size_of::<GlVertexBufferBinding>() as u64;
    let attribute = core::mem::size_of::<GlVertexAttribute>() as u64;
    layout.buffers.len() as u64 * slot + layout.attributes.len() as u64 * attribute
}

/// The identities a vertex-array record depends on.
///
/// Both the attribute sources and the index source are recorded: the array's
/// description points at the first, and the array's index binding names the
/// second, so a deletion of either leaves a record describing an input whose
/// buffers are gone.
///
/// The array object itself is recorded too, which is the one entry in this
/// layer's dependency table that is a record's own value rather than an input it
/// was derived from.  It is there because every vertex array in flight was
/// created by this domain, so a deletion of one is always a deletion of a
/// record's value: without this, the record would survive and hand out an
/// identity whose object is gone, and the failure would surface later as a
/// refused bind instead of being dropped where it happened.
fn dependencies_of(input: &VertexInput, vertex_array: VertexArrayId) -> DependencySet {
    let mut dependencies = DependencySet::new();
    for binding in &input.bindings {
        dependencies.insert(ResourceRef::Buffer(binding.buffer));
    }
    if let Some(index) = input.index {
        dependencies.insert(ResourceRef::Buffer(index.buffer));
    }
    dependencies.insert(ResourceRef::VertexArray(vertex_array));
    dependencies
}

/// What one derivation produced, and what it removed to produce it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Derivation {
    /// The array for the input's layout, newly created or already retained.
    pub vertex_array: VertexArrayId,
    /// Whether the caller owns the array and must destroy it.
    ///
    /// True when the cache is disabled, and true when the budget could not be
    /// met: the array was still built, because the caller needs one, but it is
    /// not kept.
    pub owned: bool,
    /// Records this call evicted to respect the budget, in creation order.
    ///
    /// The caller destroys each one.  A record removed here is one the caller
    /// may hold a claim about, and the object is gone either way, so the claim
    /// has to be dropped rather than repaired.
    pub removed: Vec<(GlVertexLayout, VertexArrayId)>,
}

/// The vertex arrays the geometry domain derived, keyed by their layout.
#[derive(Debug)]
pub(crate) struct VertexArrayCache {
    entries: StructuralCache<GlVertexLayout, VertexArrayId>,
}

impl VertexArrayCache {
    /// An empty cache with the given budget.
    pub(crate) fn new(budget: CacheBudget) -> Self {
        Self {
            entries: StructuralCache::new(budget),
        }
    }

    /// The number of live vertex-array records.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Looks the input's layout up, or builds the array and retains it.
    ///
    /// A hit is a hit on the layout alone: the array is a pure function of the
    /// layout, and pointing it at different buffers is what the bind does.  A
    /// miss creates the array, and the input that caused the miss also supplies
    /// the record's dependencies.
    pub(crate) fn vertex_array_for(
        &mut self,
        backend: &mut impl GlStateBackend,
        input: &VertexInput,
        mode: CacheMode,
        counters: &mut StateCounters,
    ) -> Result<Derivation, StateError> {
        if mode.may_reuse() {
            if let Some(vertex_array) = self.entries.get(&input.layout, &mut counters.caches) {
                return Ok(Derivation {
                    vertex_array: *vertex_array,
                    owned: false,
                    removed: Vec::new(),
                });
            }
        }

        let vertex_array = match backend.create_vertex_array(&input.layout) {
            Ok(vertex_array) => vertex_array,
            Err(source) => {
                counters.lifecycle.driver_errors += 1;
                return Err(StateError::backend(
                    StateDomain::Geometry,
                    "create-vertex-array",
                    PartialApplication::new(0, 1),
                    source,
                ));
            }
        };
        counters.domain(StateDomain::Geometry).emit();

        if !mode.may_reuse() {
            return Ok(Derivation {
                vertex_array,
                owned: true,
                removed: Vec::new(),
            });
        }
        // Keying the record clones the layout, which is this layer's allocation.
        counters.allocated();
        let mutation = self.entries.insert(
            input.layout.clone(),
            vertex_array,
            record_bytes(&input.layout),
            dependencies_of(input, vertex_array),
            &mut counters.caches,
        );
        Ok(Derivation {
            vertex_array,
            owned: !mutation.retained,
            removed: mutation.removed,
        })
    }

    /// Drops the records an event invalidates, reporting what was dropped.
    ///
    /// Called before the caller deletes the resource, because a name reused
    /// after deletion must not be able to hit a record that still describes the
    /// old occupant.
    pub(crate) fn invalidate(
        &mut self,
        event: &StateEvent,
        counters: &mut StateCounters,
    ) -> Vec<(GlVertexLayout, VertexArrayId)> {
        match event {
            StateEvent::BufferDeleted(buffer) => {
                self.entries
                    .invalidate_resource(ResourceRef::Buffer(*buffer), &mut counters.caches)
                    .removed
            }
            // A derived array its owner deleted: the record that holds it would
            // otherwise hand out an identity whose object is gone.
            StateEvent::VertexArrayDeleted(vertex_array) => {
                self.entries
                    .invalidate_resource(
                        ResourceRef::VertexArray(*vertex_array),
                        &mut counters.caches,
                    )
                    .removed
            }
            event if event.invalidates_everything() => {
                // The identities in these records belong to a context epoch the
                // backend no longer accepts, so there is nothing to hand back.
                self.entries.purge(&mut counters.caches);
                Vec::new()
            }
            // A deletion of anything else, and a declared scope, leave the
            // records alone: an array is what this cache derives and never what
            // it derives from, so only the buffers it reads and the array object
            // itself can make a record wrong.
            _ => Vec::new(),
        }
    }

    /// Empties the cache for a normal teardown, reporting everything held.
    ///
    /// This is the teardown path, not the invalidation path: the objects are
    /// still destructible through their identities, so dropping the records
    /// without destroying them would leak exactly the objects this cache exists
    /// to reuse.
    pub(crate) fn drain(
        &mut self,
        counters: &mut StateCounters,
    ) -> Vec<(GlVertexLayout, VertexArrayId)> {
        self.entries.drain(&mut counters.caches)
    }
}
