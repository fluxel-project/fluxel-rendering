//! Recording, committing and observing Direct3D 12 work.
//!
//! Five files, split by what a reader has to hold in mind at once rather than by
//! native call:
//!
//! * [`spine`] owns the queue, the fence, the command-list ring, and the order a
//!   plan's batches go through them. It is the lifecycle, not the steps.
//! * [`copy`], [`transfer`], [`compute`], and [`raster`] lower the corresponding
//!   recorded payloads. Unsupported texture transfers, resolves, presentation
//!   images, and debug markup remain explicit pre-commit refusals.
//! * [`transition`] is the barrier lifetime problem, which is a resource-ownership
//!   question and not a copy question: every step needs it and none of them own
//!   it.
//!
//! The vocabulary all of the above refuse through is
//! [`crate::backend::dx12::failure::Dx12Failure`], which sits at the chapter root
//! rather than in here: [`crate::backend::dx12::binding`] and
//! [`crate::backend::dx12::pipeline`] refuse through it too, and a type those two
//! had to import *from* `command` would point the dependency backwards.
//!
//! The re-export is the chapter's inside face, so the device chapter says
//! `command::Dx12CommandSpine` rather than naming the file.
//!
//! # What this chapter does not own
//!
//! Whether a plan is legal (section 40.5, decided in the portable layer), and
//! what the portable recorder can hold (section 39). This chapter is reached only
//! for a plan that already passed both, so every refusal it produces is one only
//! Direct3D 12 could know.

mod compute;
mod copy;
mod query;
mod raster;
mod spine;
mod transfer;
mod transition;

pub(crate) use spine::Dx12CommandSpine;

use crate::api::resource::buffer::Buffer;
use crate::backend::dx12::failure::Dx12Failure;
use crate::backend::dx12::resource::Dx12Buffer;

/// The native allocation behind a portable buffer.
///
/// The one helper [`copy`] and [`transfer`] share, which is why it sits at the
/// chapter root rather than in either of them: neither owns the bridge from the
/// portable handle to the native allocation, and a copy of it in both files would
/// be two authorities for what a cross-backend buffer answers (section 65.3).
///
/// # Errors
///
/// `Unsupported` when the buffer's backend is not this one. That is unreachable
/// for a plan this device accepted — the recorder compares device identity at
/// every encode, and section 3.3 makes identity the only answer to a cross-device
/// use — so this is a total function over an empty case rather than a reachable
/// refusal. It returns an error rather than unwrapping because the alternative to
/// a type-checked downcast is a panic in a library, and because a message naming
/// what was expected is worth more to whoever reaches it than an abort.
pub(super) fn dx12_buffer(buffer: &Buffer) -> Result<&Dx12Buffer, Dx12Failure> {
    buffer
        .native()
        .as_any()
        .downcast_ref::<Dx12Buffer>()
        .ok_or(Dx12Failure::Unsupported {
            what: "a buffer this device did not allocate",
            why: "its native allocation belongs to another backend, and section 3.3 makes \
                  that a refusal rather than a migration",
        })
}

use crate::api::resource::texture::Texture;
use crate::backend::dx12::resource::Dx12Texture;

/// The native allocation behind a portable texture.
pub(super) fn dx12_texture(texture: &Texture) -> Result<&Dx12Texture, Dx12Failure> {
    texture
        .native()
        .as_any()
        .downcast_ref::<Dx12Texture>()
        .ok_or(Dx12Failure::Unsupported {
            what: "a texture this device did not allocate",
            why: "its native allocation belongs to another backend",
        })
}
