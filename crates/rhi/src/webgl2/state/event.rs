//! The events that invalidate mirrored state, and the domain masks they touch.
//!
//! Every domain implements one `invalidate(&mut self, event: &StateEvent)`
//! which matches the events it owns and ignores the rest.  Centralising the
//! *table* -- which event invalidates which domains -- in one place would make
//! every domain's correctness depend on a file none of them owns, which is the
//! arrangement the invalidation matrix exists to avoid.
//!
//! The table below is therefore documentation of the plan's invalidation
//! matrix, and the implementation of each row lives in the domain that has to
//! act on it:
//!
//! | Event | Domains that must act |
//! |---|---|
//! | [`StateEvent::BufferDeleted`] | buffers, geometry, groups, sync |
//! | [`StateEvent::TextureDeleted`] | textures, session, groups, compute |
//! | [`StateEvent::SamplerDeleted`] | textures, groups |
//! | [`StateEvent::ShaderDeleted`] | pipeline |
//! | [`StateEvent::ProgramDeleted`] | pipeline, groups |
//! | [`StateEvent::VertexArrayDeleted`] | geometry, pipeline |
//! | [`StateEvent::FramebufferDeleted`] | session |
//! | [`StateEvent::QueryDeleted`] | sync |
//! | [`StateEvent::SyncDeleted`] | sync |
//! | [`StateEvent::RenderbufferDeleted`] | session |
//! | [`StateEvent::AttachmentResized`] | session |
//! | [`StateEvent::DomainFailed`] | the named domain, which is already unknown |
//! | [`StateEvent::ScopedRawAccess`] | every domain the scope declared |
//! | [`StateEvent::ContextLost`] | every domain |
//! | [`StateEvent::ContextRestored`] | every domain, plus every cache |
//! | [`StateEvent::DeviceReplaced`] | every domain, plus every cache |
//!
//! A domain appears on a deletion row when it *mirrors a binding that named the
//! deleted object*, not when the deleted object belongs to some group.  That is
//! why the compute row is the texture one: the image units a dispatch fills are
//! filled with textures, and the storage-buffer half of the same dispatch is the
//! buffer domain's storage role, so a deleted buffer reaches compute through no
//! entry of its own.
//!
//! A deletion is dispatched to the mirror *before* the backend is asked to
//! delete, because a name reused after deletion must not be able to hit a
//! mirror entry that still names the old occupant.

use crate::webgl2::api::{
    BufferId, ContextStamp, FramebufferId, ProgramId, QueryId, RenderbufferId, SamplerId, ShaderId,
    SyncId, TextureId, VertexArrayId,
};

use super::knowledge::{DirtyDomains, StateDomain};

/// A change that may make some mirrored driver state wrong.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StateEvent {
    /// A buffer is about to be deleted, or has been retired by its owner.
    BufferDeleted(BufferId),
    /// A texture is about to be deleted.
    TextureDeleted(TextureId),
    /// A sampler is about to be deleted.
    SamplerDeleted(SamplerId),
    /// A shader object is about to be deleted.
    ShaderDeleted(ShaderId),
    /// A linked program is about to be deleted.
    ProgramDeleted(ProgramId),
    /// A vertex-array object is about to be deleted.
    VertexArrayDeleted(VertexArrayId),
    /// A framebuffer object is about to be deleted.
    FramebufferDeleted(FramebufferId),
    /// A renderbuffer is about to be deleted.
    RenderbufferDeleted(RenderbufferId),
    /// A query object is about to be deleted.
    QueryDeleted(QueryId),
    /// A sync object is about to be deleted.
    SyncDeleted(SyncId),
    /// An attachment's extent, sample count or replacement changed, so any
    /// framebuffer record built from it is stale.
    AttachmentChanged,
    /// A domain's group failed partway through and is now unknown.
    ///
    /// Domains that depend on the failed one's *result* -- a framebuffer built
    /// from an attachment, a geometry record built from a buffer binding -- must
    /// treat this like a deletion of that result.  A domain that does not
    /// depend on it ignores the event, which is why the domain is named rather
    /// than the whole mask being dirtied.
    DomainFailed(StateDomain),
    /// A scoped raw-context access ran.
    ///
    /// `declared` is the mask the scope declared, or [`ScopedRawAccess::all`]
    /// when it declared nothing precise.  An undeclared scope invalidates
    /// everything: the contract is that a caller which cannot say what it
    /// touched must be assumed to have touched all of it.
    ScopedRawAccess(ScopedRawAccess),
    /// The backend reported the context lost.
    ContextLost,
    /// The context was restored onto a strictly newer stamp.
    ContextRestored(ContextStamp),
    /// Residency replaced the device, so every old-generation binding is gone.
    DeviceReplaced(ContextStamp),
}

impl StateEvent {
    /// A short stable name for diagnostics.
    pub(crate) const fn name(&self) -> &'static str {
        match self {
            Self::BufferDeleted(_) => "buffer-deleted",
            Self::TextureDeleted(_) => "texture-deleted",
            Self::SamplerDeleted(_) => "sampler-deleted",
            Self::ShaderDeleted(_) => "shader-deleted",
            Self::ProgramDeleted(_) => "program-deleted",
            Self::VertexArrayDeleted(_) => "vertex-array-deleted",
            Self::FramebufferDeleted(_) => "framebuffer-deleted",
            Self::RenderbufferDeleted(_) => "renderbuffer-deleted",
            Self::QueryDeleted(_) => "query-deleted",
            Self::SyncDeleted(_) => "sync-deleted",
            Self::AttachmentChanged => "attachment-changed",
            Self::DomainFailed(_) => "domain-failed",
            Self::ScopedRawAccess(_) => "scoped-raw-access",
            Self::ContextLost => "context-lost",
            Self::ContextRestored(_) => "context-restored",
            Self::DeviceReplaced(_) => "device-replaced",
        }
    }

    /// Whether this event invalidates every domain and every derived cache.
    ///
    /// The three whole-mirror events are the ones a domain cannot reason about
    /// locally: the context is gone, the epoch moved, or a scope declared that
    /// it may have touched anything.
    pub(crate) const fn invalidates_everything(&self) -> bool {
        match self {
            Self::ContextLost | Self::ContextRestored(_) | Self::DeviceReplaced(_) => true,
            Self::ScopedRawAccess(scope) => scope.is_everything(),
            _ => false,
        }
    }
}

/// What a scoped raw-context access declared about the state it touched.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ScopedRawAccess {
    declared: Option<DirtyDomains>,
}

impl ScopedRawAccess {
    /// A scope that declared precisely these domains.
    pub(crate) const fn declaring(domains: DirtyDomains) -> Self {
        Self {
            declared: Some(domains),
        }
    }

    /// A scope that declared nothing, and is therefore assumed to have touched
    /// every domain.
    pub(crate) const fn all() -> Self {
        Self { declared: None }
    }

    /// Whether this scope must invalidate the whole mirror.
    pub(crate) const fn is_everything(&self) -> bool {
        self.declared.is_none()
    }

    /// The domains this scope invalidates.
    pub(crate) fn domains(&self) -> DirtyDomains {
        self.declared.unwrap_or(DirtyDomains::ALL)
    }
}

#[cfg(test)]
mod tests {
    use super::{ScopedRawAccess, StateEvent};
    use crate::webgl2::state::knowledge::{DirtyDomains, StateDomain};

    #[test]
    fn only_whole_mirror_events_invalidate_everything() {
        assert!(StateEvent::ContextLost.invalidates_everything());
        assert!(StateEvent::ScopedRawAccess(ScopedRawAccess::all()).invalidates_everything());
        assert!(!StateEvent::AttachmentChanged.invalidates_everything());
        // A scope that named its domains is not a whole-mirror event, which is
        // the entire reason for declaring them.
        assert!(
            !StateEvent::ScopedRawAccess(ScopedRawAccess::declaring(DirtyDomains::of(
                StateDomain::Textures
            )))
            .invalidates_everything()
        );
    }

    #[test]
    fn an_undeclared_scope_invalidates_every_domain() {
        assert_eq!(ScopedRawAccess::all().domains(), DirtyDomains::ALL);

        let declared = ScopedRawAccess::declaring(DirtyDomains::of(StateDomain::Pipeline));
        assert_eq!(declared.domains().to_string(), "{pipeline}");
    }
}
