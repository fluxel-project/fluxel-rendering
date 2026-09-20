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
//! `design-rhi.md:59-60` excludes native handles generally — so the trait below
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
    #[cfg_attr(
        not(any(
            test,
            // The backend features that actually compile a lowering. A feature
            // that selects nothing must not appear here: it would remove this
            // expectation in a configuration where the item really is dead, and
            // the gate would then be silent about it. When Vulkan lands and starts
            // calling this, its feature joins the list — which is rule 4.6's
            // "the matrix gets the row" applied to the attribute itself.
            feature = "dx12"
        )),
        expect(
            dead_code,
            reason = "called by a backend's own lowering, which is the only code that may \
                      downcast across the seam; a build with no backend compiled has none"
        )
    )]
    fn as_any(&self) -> &dyn Any;
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
