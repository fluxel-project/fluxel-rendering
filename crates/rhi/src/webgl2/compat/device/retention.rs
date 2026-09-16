//! The strong lease one transient is handed out with, and the queue its death
//! lands in.
//!
//! The common contract's `Lease` is documented as keeping a transient alive
//! until submission completion *and* as caller-owned in the same breath
//! (`rendergraph/src/execution/run/exports.rs:26`: "Caller-owned strong lease;
//! completion alone does not invalidate it"), and the executor's transient pool
//! takes it at its word: each cached slot holds a clone of the lease for the
//! slot's whole life and frees the slot only after polling a terminal
//! completion.  So the native object must stay alive until the *last* handle
//! drops -- not until the frame's completion -- and it must be destroyed exactly
//! once, after that.
//!
//! That is why this is an `Rc<LeaseInner>` and not a plain handle.  Layer 1's
//! destroy verbs take `&mut` on the provider, and a `Drop` has no provider to
//! take them from; so the drop that discovers "nobody holds this any more"
//! cannot destroy anything itself.  It does the one thing it can -- it records
//! the identity in a queue shared with the adapter -- and the adapter destroys
//! everything in that queue at its next `&mut self` entry, which is a place that
//! does have a provider.  The alternative, destroying from `Drop`, would need
//! the provider to be reachable from a lease, and a lease that can reach the
//! provider can outlive it.
//!
//! Nothing here counts, orders or interprets.  Whether an object *may* be
//! released is the completion ledger's answer, and it answers by holding the
//! lease rather than by telling this queue anything.

use std::cell::RefCell;
use std::rc::Rc;

use crate::webgl2::api::{BufferId, TextureId};

/// One native object whose last lease has dropped.
///
/// The two kinds travel in one queue because the queue's consumer destroys them
/// through two different Layer 1 verbs, and a queue that had to be asked
/// per-kind would be a second place the "which verb" decision lives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RetainedObject {
    /// A physical texture created by `create_texture_resource`.
    Texture(TextureId),
    /// A physical buffer created by `create_buffer_resource`.
    Buffer(BufferId),
}

/// The objects released since the adapter last looked.
///
/// One queue per adapter, shared by `Rc` with every lease that adapter minted.
/// It is deliberately not `Sync`: the adapter is a single immediate context on
/// its owning thread (`OwnerThreadIdentity`), and a lease that could be dropped
/// from another thread would make the destroy verbs reachable off that thread.
#[derive(Debug)]
pub(super) struct ReleaseQueue {
    pending: RefCell<Vec<RetainedObject>>,
}

impl ReleaseQueue {
    /// A queue with nothing in it.
    pub(super) fn new() -> Rc<Self> {
        Rc::new(Self {
            pending: RefCell::new(Vec::new()),
        })
    }

    /// Records one object as released.
    fn push(&self, object: RetainedObject) {
        self.pending.borrow_mut().push(object);
    }

    /// Takes every object released since the last call, oldest first.
    ///
    /// The order is the order the leases dropped, which is the only order with
    /// any meaning here: an object released earlier is at least as dead as one
    /// released later.
    pub(super) fn drain(&self) -> Vec<RetainedObject> {
        core::mem::take(&mut *self.pending.borrow_mut())
    }

    /// Drops every record without acting on it.
    ///
    /// This is what a context-generation change calls, and it is not the same
    /// act as draining.  A restored context reports a strictly newer
    /// `ContextStamp`, and the provider invalidates every object table, lease
    /// book and derived record *before* it reports `Active` again
    /// (`api/native/provider.rs`): the objects named here are already gone, so
    /// destroying them would be a call against identities of a dead epoch.  The
    /// records are what has to go, and only the records.
    pub(super) fn forget(&self) {
        self.pending.borrow_mut().clear();
    }
}

/// The lease a transient is handed out with.
///
/// `pub(crate)` rather than `pub(super)` because it is bound to a public
/// foreign trait's associated type: `ExecutionBackend::Lease` is declared in
/// `fluxel_rendergraph`, and a binding narrower than the crate is a restricted
/// type in a public interface (E0446).  The whole module chain above this one is
/// crate-private, so nothing here is nameable from outside the crate either way;
/// the wider spelling is what the language requires, not a widening of the API.
#[derive(Clone, Debug)]
pub(crate) struct GlRetentionLease(Rc<LeaseInner>);

/// The shared half of a retention lease: the object, and where its death goes.
///
/// `Drop` is on this and not on [`GlRetentionLease`] because it must run once,
/// at the last drop, and a `Drop` on the outer type would run at every clone's
/// drop.  This is the whole reason the outer type is an `Rc` rather than a
/// plain `Copy` handle.
#[derive(Debug)]
struct LeaseInner {
    object: RetainedObject,
    releases: Rc<ReleaseQueue>,
}

impl Drop for LeaseInner {
    fn drop(&mut self) {
        self.releases.push(self.object);
    }
}

impl GlRetentionLease {
    /// Retains `object` until the last clone of this lease is dropped.
    pub(super) fn new(object: RetainedObject, releases: Rc<ReleaseQueue>) -> Self {
        Self(Rc::new(LeaseInner { object, releases }))
    }
}
