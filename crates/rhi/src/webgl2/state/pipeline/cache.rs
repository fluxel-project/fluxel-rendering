//! The pipeline domain's linked-program cache.
//!
//! A renderer installs the same handful of pipelines every frame, and each one
//! names a linked program.  Linking is the most expensive setup operation a GL
//! implementation offers -- it compiles every stage and runs the linker -- while
//! the value the caller hands over is a complete, hashable descriptor.  A
//! domain that linked the program on every pipeline install would pay the linker
//! once per draw call, so the domain keeps the programs it linked keyed by the
//! descriptor that produced them.
//!
//! # Why the key carries the context stamp
//!
//! Every other derived key in this layer can rely on the stamp arriving
//! transitively, because the objects in the key are Layer 1 identities that
//! already contain one.  `GlProgramDescriptor` is the exception: it is built
//! entirely from lowered shader text, a pipeline layout and a debug name, and
//! names no object at all.  Two descriptors that differ only in which context
//! their sources were lowered for compare equal, so the stamp has to be part of
//! the key rather than something the value implies.  A program is not reusable
//! across an epoch in principle either -- the driver object does not survive one
//! -- and an epoch change purges this cache, so the stamp is a second line of
//! defence rather than the only one.
//!
//! # What one record depends on
//!
//! Exactly one thing: the program it names.  The descriptor's shader sources are
//! kept as *content* -- text, hash, entry point and dialect -- and never as
//! shader object identities, because Layer 1 links from the content and its
//! providers delete the shader objects the moment the link succeeds.  That is
//! why a shader-object deletion is not a cache event here: a deleted shader
//! object names nothing this cache holds.  Recording the program as the record's
//! own dependency is what makes a program deletion drop its record, and it is
//! the only lever that does so, because the underlying table offers no
//! removal-by-key and no iteration.
//!
//! # Who destroys a program
//!
//! A record's value is a program this layer asked Layer 1 to link, so this layer
//! is the one that has to delete it -- but only along the two paths where nobody
//! else is deleting it: an eviction, and the machine's teardown.  The third path
//! is [`StateEvent::ProgramDeleted`], which is the owner saying it is deleting
//! the program itself; destroying through that identity here would be the second
//! deletion of one object, so that path forgets the record and hands nothing
//! back.  The two are told apart by *who is deleting*, which is why
//! [`ProgramCache::invalidate`] needs no backend while every other removal path
//! does.
//!
//! A context loss is the one removal that destroys nothing, and that is what
//! [`StructuralCache::purge`] exists for: the identities in these records belong
//! to an epoch the backend no longer accepts.  A raw scope is the other way
//! round.  It invalidates *state* without ending the context, so the programs it
//! may have deleted are still deletable through their identities, and the cache
//! drains them for the caller to destroy rather than dropping them silently.

use crate::webgl2::api::{
    ContextStamp, GlLogicalBinding, GlProgramDescriptor, GlProgramKind, GlProgramReflection,
    ProgramId,
};

use super::super::GlStateBackend;
use super::super::cache::{CacheBudget, DependencySet, ResourceRef, StructuralCache};
use super::super::counters::StateCounters;
use super::super::event::StateEvent;

/// The complete structural input of one linked program.
///
/// Both halves are required: the descriptor says which program was asked for,
/// and the stamp says which context was asked.  See this module's documentation.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ProgramKey {
    stamp: ContextStamp,
    descriptor: GlProgramDescriptor,
}

impl ProgramKey {
    /// A key for one descriptor in one context epoch.
    pub(crate) fn new(stamp: ContextStamp, descriptor: GlProgramDescriptor) -> Self {
        Self { stamp, descriptor }
    }
}

/// One linked program and the link evidence that came back with it.
///
/// The reflection is kept beside the identity because it is the other half of
/// what linking produced: a caller that wants to assign this program's uniform
/// blocks has to resolve the logical bindings against the executable locations
/// the linker chose, and re-deriving that would mean asking the driver again.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProgramRecord {
    /// The program this layer linked.
    pub(crate) program: ProgramId,
    /// The assignments and stage interfaces the linker reported.
    pub(crate) reflection: GlProgramReflection,
}

/// The estimated retained cost of one program record.
///
/// The value is a descriptor-shaped estimate, not a measurement of driver
/// memory: what is being bounded is this layer's own bookkeeping, and the
/// driver's compiled program is not something this layer can see.  The number is
/// dominated by the lowered source text, which is the part that actually grows
/// with the key.
fn record_bytes(descriptor: &GlProgramDescriptor) -> u64 {
    let sources = match &descriptor.kind {
        GlProgramKind::Raster { vertex, fragment } => {
            vertex.text.len() as u64 + fragment.text.len() as u64
        }
        GlProgramKind::Compute { shader } => shader.text.len() as u64,
    };
    let bindings =
        descriptor.layout.bindings.len() as u64 * core::mem::size_of::<GlLogicalBinding>() as u64;
    let name = descriptor
        .debug_name
        .as_ref()
        .map_or(0, |name| name.len() as u64);
    sources + bindings + name
}

/// The resources one record was derived from.
///
/// The program is its own dependency: the record is the mapping from a
/// descriptor to a program, so the program going away is the one event that
/// makes the mapping wrong.
fn dependencies_of(record: &ProgramRecord) -> DependencySet {
    let mut dependencies = DependencySet::new();
    dependencies.insert(ResourceRef::Program(record.program));
    dependencies
}

/// The linked programs this domain derived, keyed by their descriptor.
#[derive(Debug)]
pub(crate) struct ProgramCache {
    entries: StructuralCache<ProgramKey, ProgramRecord>,
}

impl ProgramCache {
    /// An empty cache with the given budget.
    pub(crate) fn new(budget: CacheBudget) -> Self {
        Self {
            entries: StructuralCache::new(budget),
        }
    }

    /// The number of live program records.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Serves one record from the cache, cloning what the caller needs out of
    /// it.
    ///
    /// The clone is deliberate and has to be: the caller holds the answer across
    /// a mutable borrow of this cache, so the reflection cannot be lent out of
    /// the table.  It is the price of a cache hit that a miss does not pay
    /// twice, which is why the caller counts it.
    pub(crate) fn record_for(
        &mut self,
        key: &ProgramKey,
        counters: &mut StateCounters,
    ) -> Option<(ProgramId, GlProgramReflection)> {
        self.entries
            .get(key, &mut counters.caches)
            .map(|record| (record.program, record.reflection.clone()))
    }

    /// Retains a record, destroying whatever the budget pushed out.
    ///
    /// Returns whether the cache kept the record.  `false` means the budget
    /// could not be met without evicting a leased one: nothing was inserted and
    /// the caller still owns the program it just linked.
    pub(crate) fn retain(
        &mut self,
        backend: &mut impl GlStateBackend,
        key: ProgramKey,
        record: ProgramRecord,
        counters: &mut StateCounters,
    ) -> bool {
        let bytes = record_bytes(&key.descriptor);
        let dependencies = dependencies_of(&record);
        let mutation = self
            .entries
            .insert(key, record, bytes, dependencies, &mut counters.caches);
        // Whatever the budget pushed out is this layer's to destroy, in the
        // creation order the cache reports.
        for (_, evicted) in mutation.removed {
            destroy(backend, evicted.program, counters);
        }
        mutation.retained
    }

    /// Drops the records an event invalidated, destroying nothing.
    ///
    /// Every removal this reacts to either destroys nothing -- a purge after an
    /// epoch change, whose identities are no longer callable -- or must not
    /// destroy, because the owner is deleting the program itself.  That is why
    /// this takes no backend: the cache's event handling makes no driver call at
    /// all.
    pub(crate) fn invalidate(&mut self, event: &StateEvent, counters: &mut StateCounters) {
        match event {
            StateEvent::ProgramDeleted(program) => {
                // The record must go before the name can be reused, and the
                // deletion itself is the owner's act rather than this cache's.
                let _ = self
                    .entries
                    .invalidate_resource(ResourceRef::Program(*program), &mut counters.caches);
            }
            event if event.invalidates_everything() => {
                self.entries.purge(&mut counters.caches);
            }
            _ => {}
        }
    }

    /// Destroys and forgets every record.
    ///
    /// This is a teardown path, not an invalidation path: the objects are still
    /// destructible through their identities, so dropping the records without
    /// destroying them would leak exactly the programs this cache exists to
    /// reuse.  It is used both by the machine's shutdown and by a raw scope that
    /// touched this domain, because in both cases the programs are this layer's
    /// to delete and the context is still alive to delete them.
    pub(crate) fn drop_all(
        &mut self,
        backend: &mut impl GlStateBackend,
        counters: &mut StateCounters,
    ) {
        let drained = self.entries.drain(&mut counters.caches);
        for (_, record) in drained {
            destroy(backend, record.program, counters);
        }
    }
}

/// Deletes one linked program, counting a failure rather than raising it.
///
/// A failed deletion is not something the caller can act on: the frame that
/// needed the program has already been recorded, and the alternative to counting
/// it is leaking the object silently.  The count is what a leak report reads;
/// propagating it would turn a cleanup detail into a frame failure.
fn destroy(backend: &mut impl GlStateBackend, program: ProgramId, counters: &mut StateCounters) {
    if backend.destroy_program(program).is_err() {
        counters.lifecycle.driver_errors += 1;
    }
}
