//! Resource, route, upload, and readback contract tests (specification
//! sections 9 through 18).
//!
//! The tests are partitioned by specification section, one file per
//! section group, because a single file had grown past the size at which
//! a reader can hold it. The conventions below apply to every submodule,
//! so they are stated once, here:
//!
//! Every validation rule a specification list states is exercised twice —
//! once with a descriptor that must be accepted, once with one that must be
//! refused — and each refusal asserts the exact
//! [`RhiErrorKind`](crate::api::error::RhiErrorKind), because
//! the kind is the part a caller may branch on.
//!
//! Two conventions run through these tests:
//!
//! * Capability answers are *built* here rather than probed, because no device
//!   exists yet. That is exactly why the validators take them as parameters: the
//!   rule is decidable without a backend, so it is tested without one.
//! * Objects are assembled through the crate-private constructors the device
//!   verbs will call, the same way `tests/identity.rs` reaches the token
//!   constructors. A test that could not name an object could not test what an
//!   accessor reports about it.

mod buffer;
mod route;
mod sampler;
mod subresource;
mod texture;
mod transfer;
mod view;
use crate::api::error::{RhiErrorKind, RhiResult};
use crate::api::format::{TextureFormat, TextureSupport, TextureSupportLimits};
use crate::api::identity::{DeviceGeneration, DeviceIdentity, DeviceInstanceId, ObjectId};
use crate::api::resource::buffer::{
    Buffer, BufferDescriptor, BufferSupport, BufferSupportLimits, BufferUsage,
};
use crate::api::resource::route::BufferCopyLayoutLimits;
use crate::api::resource::texture::{Extent3d, Texture, TextureDescriptor, TextureUsage};

fn identity(instance: u64, generation: u64) -> DeviceIdentity {
    DeviceIdentity::new(
        DeviceInstanceId::new(instance),
        DeviceGeneration::new(generation),
    )
}

fn object(value: u64) -> ObjectId {
    ObjectId::new(value)
}

/// The identity every fixture object belongs to.
fn device() -> DeviceIdentity {
    identity(1, 1)
}

/// Asserts a validation result is the exact kind the specification's mapping
/// requires, and prints the message when it is not.
fn assert_kind(result: RhiResult<()>, expected: RhiErrorKind) {
    match result {
        Ok(()) => panic!("expected {expected}, but the operation was accepted"),
        Err(error) => assert_eq!(error.kind(), expected, "{}", error.message()),
    }
}

/// A capability answer broad enough that only the rule under test can refuse.
fn generous_buffer_support() -> BufferSupport {
    BufferSupport::Supported(BufferSupportLimits::new(1 << 30))
}

/// The texture counterpart of [`generous_buffer_support`].
fn generous_texture_support() -> TextureSupport {
    TextureSupport::Supported(TextureSupportLimits::new(
        Extent3d::d3(16384, 16384, 2048),
        15,
        2048,
    ))
}

/// A 4x4 single-sampled 2D RGBA texture descriptor usable for whatever usage the
/// caller names.
fn simple_texture_descriptor(usage: TextureUsage) -> TextureDescriptor {
    TextureDescriptor::new_2d(4, 4, TextureFormat::Rgba8Unorm, usage)
}

fn buffer_with(usage: BufferUsage, size: u64) -> Buffer {
    Buffer::new(object(1), device(), BufferDescriptor::new(size, usage))
}

fn texture_with(descriptor: TextureDescriptor) -> Texture {
    Texture::new(object(2), device(), descriptor)
}

/// Section 16.1's copy-layout envelope, in the shape a device reports it.
fn copy_limits() -> BufferCopyLayoutLimits {
    BufferCopyLayoutLimits::new(4, 4)
}
