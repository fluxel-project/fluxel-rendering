//! Crate-private backend contract for the resource API (sections 11 through 15).
//!
//! Portable validation runs in the public API before these traits are reached.
//! This module adds the ownership rule that makes the
//! resource seam different from the platform one:
//!
//! # What a resource handle already owns, and what is left for the backend
//!
//! `crate::api::platform`'s handles are thin — a `Device` is an identity and a
//! shared backend object — so nearly everything about a device is a backend
//! question. A resource handle is the opposite. [`crate::api::resource::Buffer`]
//! already carries its [`crate::api::identity::ObjectId`], its
//! [`crate::api::identity::DeviceIdentity`], and the descriptor
//! it was created from, and section 3 makes all three the *portable* layer's
//! answers: identity is minted by the layer that enforces generation rules, and
//! the descriptor is kept exactly as the caller wrote it — validation is a check,
//! not a normalization, so there is no canonical form for a backend to be handed
//! instead.
//!
//! What is left over is exactly one thing: the native allocation. It has no
//! portable spelling — section 59 excludes `GpuAddress`, and
//! ADR-0002 excludes native handles generally — so the trait below
//! is the only place it can be named at all, and it names it as an opaque
//! [`Any`] rather than as a type.
//!
//! # Why that is a downcast and not a method per operation
//!
//! The obvious alternative is a trait with a method per native operation:
//! `copy_from`, `map`, `gpu_address`. That shape puts the operation's *policy* on
//! the object being operated on, and almost every operation in this chapter
//! involves two objects that must belong to the same device — a copy takes a
//! source and a destination. A trait method can only be reached through one of
//! them, which makes the choice of receiver arbitrary and hides a device check
//! inside a call that looks like it touches one object. The DX12 copy path
//! instead reaches *the device's* backend and downcasts both operands there,
//! which is the same place section 3.3's device comparison already lives.
//!
//! So the seam carries the object and nothing else, and the operations belong to
//! the device backend. A backend that finds itself wanting a second method here
//! should read that as a sign the operation was not per-object after all.

use std::any::Any;
use std::task::{Context, Poll};

use crate::api::error::RhiResult;

/// The native allocation behind one [`crate::api::resource::Buffer`].
///
/// Implemented by a backend, held by the portable handle, and never reachable
/// from outside the crate. See the module documentation for why it carries the
/// object rather than the operations.
pub(crate) trait BufferBackend: Send + Sync + 'static {
    /// This allocation as an opaque native object.
    ///
    /// The cast is the seam's whole purpose: a caller that knows the backend
    /// family — which, inside this crate, means that backend's own device
    /// implementation — downcasts to reach the native resource, and every other
    /// caller can only see that *something* is there. `Any` rather than a
    /// backend-declared trait object because a trait declared here would have to
    /// name the operations, which is the shape the module documentation rejects.
    ///
    /// The downcast's first caller is the DX12 command spine, which reaches a
    /// backend's own buffer type from a portable handle to record a copy. It is
    /// still unreached in a build with no backend compiled, so the expectation
    /// below is gated on the backend feature list rather than deleted: the method
    /// is part of the seam's contract, and a configuration with nothing on the
    /// far side of the seam has nothing that could call it.
    // Backend lowerers downcast this seam when a backend is enabled.  An
    // API-only build has no such consumer, but the method remains part of the
    // shared private contract for the feature matrix.
    #[allow(dead_code)]
    fn as_any(&self) -> &dyn Any;
}

/// One active backend mapping lease.
///
/// The lease owns whatever native map/unmap token is required.  It is not
/// `Send`: several native APIs associate mapping lifetime with external command
/// synchronization, and the portable layer deliberately does not invent a
/// cross-thread promise for it.
pub(crate) trait MappedBufferBackend: 'static {
    fn bytes(&self) -> &[u8];
    fn bytes_mut(&mut self) -> Option<&mut [u8]>;
    fn flush(&mut self) -> RhiResult<()>;
    fn invalidate(&mut self) -> RhiResult<()>;
}

/// One native request to acquire a mapping lease.
///
/// Mapping is an asynchronous completion-domain operation: a backend may need
/// to wait for earlier GPU uses to retire before the host may touch the range.
/// The request therefore owns native waiting state and must register `cx`'s
/// waker whenever it returns [`Poll::Pending`].  Device loss is terminal: it
/// must wake every registered mapping request, after which the portable future
/// returns `DeviceLost` before publishing a mapped view.
///
/// Dropping a *pending* request cancels the wait.  Once `poll` yields a mapping
/// lease, dropping the now-consumed request must not undo that lease; ownership
/// has transferred to the returned [`MappedBufferBackend`]. Backends release
/// any pending native reservation in `Drop`; the portable future releases its
/// corresponding exclusive buffer lease in its own `Drop` implementation.
pub(crate) trait MappingRequestBackend: 'static {
    /// Advances the native wait and, once ready, yields the native map lease.
    fn poll(&mut self, context: &mut Context<'_>) -> Poll<RhiResult<Box<dyn MappedBufferBackend>>>;
}

/// The native allocation behind one portable texture.
///
/// This mirrors [`BufferBackend`]: texture state transitions and copies belong to
/// the device lowering, while the object merely carries the backend allocation
/// that those operations downcast to.
pub(crate) trait TextureBackend: Send + Sync + 'static {
    fn as_any(&self) -> &dyn Any;
}

/// The native descriptor or view object behind a texture view.
pub(crate) trait TextureViewBackend: Send + Sync + 'static {
    fn as_any(&self) -> &dyn Any;
}

/// The native descriptor or object behind a sampler.
pub(crate) trait SamplerBackend: Send + Sync + 'static {
    fn as_any(&self) -> &dyn Any;
}

/// Native allocation behind one portable query set.
///
/// Query storage is device-owned native state rather than buffer memory.  It is
/// nevertheless held by the portable handle so a recorded command retains the
/// exact pool/heap it addresses until submission is terminal.
pub(crate) trait QuerySetBackend: Send + Sync + 'static {
    /// Exposes the native object only to its owning backend lowering.
    fn as_any(&self) -> &dyn Any;
}

/// Native acceleration-structure allocation held by a portable handle.
///
/// The public API intentionally never exposes a GPU virtual address.  A command
/// lowering downcasts this object together with its owning device only after all
/// portable ownership, usage, and build-range checks have succeeded.
pub(crate) trait AccelerationStructureBackend: Send + Sync + 'static {
    /// Exposes the backend object only to its owning lowering implementation.
    fn as_any(&self) -> &dyn Any;
}
