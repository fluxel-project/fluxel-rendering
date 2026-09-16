//! The session domain's framebuffer cache.
//!
//! A pass renders into a framebuffer built from an attachment set, and a
//! renderer usually renders into the same handful of attachment sets every
//! frame.  Rebuilding the framebuffer object each time would re-run attachment
//! validation, the completeness check, and the driver's own allocation for a
//! shape that has not changed, so the session keeps the ones it built.
//!
//! # Why the key is the whole descriptor
//!
//! `GlFramebufferDescriptor` is the complete structural input: the attachment
//! views, the optional depth-stencil view, and the explicit draw-buffer
//! selection.  Two descriptors that differ in any of those need different
//! framebuffer objects, and two that agree in all of them do not -- which is
//! exactly the equality Layer 1 derives.  The context stamp is carried
//! transitively, because every attachment view names an object identity that
//! already contains it; a framebuffer cannot be reused across an epoch change
//! even in principle, and the epoch change purges this cache anyway.
//!
//! # Dependency direction
//!
//! Every attachment is a dependency of the entry, so deleting a texture or a
//! renderbuffer drops the framebuffers built from it rather than leaving a
//! cached object that names a dead attachment.  The session dispatches the
//! deletion to this cache *before* asking the backend to delete the attachment,
//! which is the ordering [`super::super::event`] requires.
//!
//! # What this module does not do
//!
//! It does not decide when a pass begins or ends, and it does not know about
//! the drawable -- not as an omission but because there is nothing here for it
//! to name.  Layer 1's attachment vocabulary has exactly two storage classes, a
//! texture and a renderbuffer (`GlAttachmentTarget`), and the window system's
//! drawable is neither: a graph reaches the screen by rendering into the
//! texture its present root names and publishing that texture afterwards, which
//! is `GlSurfacePresentationApi`'s half and not this cache's.  Every entry this
//! cache can hold therefore names storage it can also destroy.

use crate::webgl2::api::{
    FramebufferId, GlAttachmentTarget, GlFramebufferDescriptor, GlTextureView,
};

use super::super::GlStateBackend;
use super::super::cache::{CacheBudget, CacheMode, DependencySet, ResourceRef, StructuralCache};
use super::super::counters::StateCounters;
use super::super::error::{PartialApplication, StateError};
use super::super::event::StateEvent;
use super::super::knowledge::StateDomain;

/// The estimated retained cost of one framebuffer record.
///
/// The value is a descriptor-shaped estimate, not a measurement of driver
/// memory: what is being bounded is this layer's own bookkeeping, and the
/// driver's framebuffer storage is not something this layer can see.  The
/// number is dominated by the per-attachment views, which is the part that
/// actually grows with the key.
fn record_bytes(descriptor: &GlFramebufferDescriptor) -> u64 {
    let views = descriptor.color_attachments.len() as u64
        + u64::from(descriptor.depth_stencil_attachment.is_some());
    let view = core::mem::size_of::<GlTextureView>() as u64;
    views * view + core::mem::size_of::<u32>() as u64 * descriptor.draw_buffers.len() as u64
}

/// The attachment identities a framebuffer record depends on.
fn dependencies_of(descriptor: &GlFramebufferDescriptor) -> DependencySet {
    let mut dependencies = DependencySet::new();
    for view in descriptor
        .color_attachments
        .iter()
        .chain(descriptor.depth_stencil_attachment.iter())
    {
        match view.target {
            GlAttachmentTarget::Texture(texture) => {
                dependencies.insert(ResourceRef::Texture(texture));
            }
            GlAttachmentTarget::Renderbuffer(renderbuffer) => {
                dependencies.insert(ResourceRef::Renderbuffer(renderbuffer));
            }
        }
    }
    dependencies
}

/// The framebuffer objects the session derived, keyed by their descriptor.
#[derive(Debug)]
pub(crate) struct FramebufferCache {
    entries: StructuralCache<GlFramebufferDescriptor, FramebufferId>,
}

impl FramebufferCache {
    /// An empty cache with the given budget.
    pub(crate) fn new(budget: CacheBudget) -> Self {
        Self {
            entries: StructuralCache::new(budget),
        }
    }

    /// The number of live framebuffer records.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Looks the descriptor up, or builds it and retains the result.
    ///
    /// The second half of the return is whether the caller owns the framebuffer
    /// and must destroy it when it is done.  That is true in two cases, and
    /// both are the ordinary ones rather than error paths:
    ///
    /// - The oracle mode, where nothing is retained by construction.
    /// - A budget the record does not fit.  The framebuffer was still built,
    ///   because the caller needs one to render into; it is simply not kept,
    ///   which is a slower frame rather than a refusal.
    pub(crate) fn framebuffer_for(
        &mut self,
        backend: &mut impl GlStateBackend,
        descriptor: &GlFramebufferDescriptor,
        mode: CacheMode,
        counters: &mut StateCounters,
    ) -> Result<(FramebufferId, bool), StateError> {
        if mode.may_reuse() {
            if let Some(framebuffer) = self.entries.get(descriptor, &mut counters.caches) {
                return Ok((*framebuffer, false));
            }
        }

        let framebuffer = match backend.create_framebuffer(descriptor) {
            Ok(framebuffer) => framebuffer,
            Err(source) => {
                counters.lifecycle.driver_errors += 1;
                return Err(StateError::backend(
                    StateDomain::Session,
                    "create-framebuffer",
                    PartialApplication::new(0, 1),
                    source,
                ));
            }
        };
        counters.domain(StateDomain::Session).emit();

        if !mode.may_reuse() {
            return Ok((framebuffer, true));
        }
        let mutation = self.entries.insert(
            descriptor.clone(),
            framebuffer,
            record_bytes(descriptor),
            dependencies_of(descriptor),
            &mut counters.caches,
        );
        // Whatever the budget pushed out is this layer's to destroy, in the
        // creation order the cache reports.
        for (_, evicted) in mutation.removed {
            destroy(backend, evicted, counters);
        }
        Ok((framebuffer, !mutation.retained))
    }

    /// Drops the records built from `resource`, before the caller deletes it.
    ///
    /// Called before the backend deletes the attachment, because a name reused
    /// after deletion must not be able to hit a record that still describes the
    /// old occupant.
    pub(crate) fn invalidate_resource(
        &mut self,
        backend: &mut impl GlStateBackend,
        resource: ResourceRef,
        counters: &mut StateCounters,
    ) {
        let mutation = self
            .entries
            .invalidate_resource(resource, &mut counters.caches);
        for (_, framebuffer) in mutation.removed {
            destroy(backend, framebuffer, counters);
        }
    }

    /// Reacts to an invalidation the session dispatched to its caches.
    pub(crate) fn invalidate(
        &mut self,
        backend: &mut impl GlStateBackend,
        event: &StateEvent,
        counters: &mut StateCounters,
    ) {
        match event {
            StateEvent::TextureDeleted(texture) => {
                self.invalidate_resource(backend, ResourceRef::Texture(*texture), counters);
            }
            StateEvent::RenderbufferDeleted(renderbuffer) => {
                self.invalidate_resource(
                    backend,
                    ResourceRef::Renderbuffer(*renderbuffer),
                    counters,
                );
            }
            // A framebuffer the cache itself derived was deleted by its owner,
            // so the record naming it must go before the name can be reused.
            StateEvent::FramebufferDeleted(framebuffer) => {
                self.invalidate_resource(backend, ResourceRef::Framebuffer(*framebuffer), counters);
            }
            // An attachment's extent, sample count or replacement changed, so
            // every record built from a view of it may now be incomplete.
            // There is no cheaper answer than rebuilding: completeness is a
            // driver property, and asking the driver about it is the call the
            // cache exists to avoid.
            StateEvent::AttachmentChanged => self.drop_all(backend, counters),
            event if event.invalidates_everything() => {
                // The identities in these records belong to a context epoch the
                // backend no longer accepts, so there is nothing to destroy
                // through them; asking would be a call with a stale object.
                self.entries.purge(&mut counters.caches);
            }
            _ => {}
        }
    }

    /// Destroys and forgets every record.
    ///
    /// This is the teardown path, not the invalidation path: the objects are
    /// still destructible through their identities, so dropping the records
    /// without destroying them would leak exactly the objects this cache exists
    /// to reuse.
    pub(crate) fn drop_all(
        &mut self,
        backend: &mut impl GlStateBackend,
        counters: &mut StateCounters,
    ) {
        let drained = self.entries.drain(&mut counters.caches);
        for (_, framebuffer) in drained {
            destroy(backend, framebuffer, counters);
        }
    }
}

/// Destroys one derived framebuffer, counting a failure rather than raising it.
///
/// A failed deletion is not something the caller can act on -- the pass that
/// used the framebuffer has already ended, and the alternative to counting it
/// is leaking the object silently.  The count is what a leak report reads;
/// propagating it would turn a cleanup detail into a frame failure.
fn destroy(
    backend: &mut impl GlStateBackend,
    framebuffer: FramebufferId,
    counters: &mut StateCounters,
) {
    if backend.destroy_framebuffer(framebuffer).is_err() {
        counters.lifecycle.driver_errors += 1;
    }
}
