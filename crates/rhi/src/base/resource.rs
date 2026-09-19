//! The resource chapter's seam (specification module 02, sections 11 through 15).
//!
//! Read the four disciplines in [`crate::base`] first; they bind every trait
//! here. This module adds one thing to them, and it is the thing that makes the
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
    /// The downcast's first caller is the transfer lowering, which is not
    /// written, so no non-test build reaches this method yet even though every
    /// backend implements it. The expectation is the honest form of that: the
    /// method is part of the seam's contract and its absence of callers is a
    /// schedule fact, not a design one.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the transfer lowering is the first downcaster and is not written; until it lands, only the contract tests call this"
        )
    )]
    fn as_any(&self) -> &dyn Any;
}
