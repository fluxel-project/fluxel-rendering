//! Handles a test assembles directly, and the native side they need.
//!
//! # Why these exist rather than `Device::create_buffer`
//!
//! Most of the contract tests in this directory examine a rule that takes a
//! handle as an *argument*: a range checked against a buffer's size, an ownership
//! comparison against a target device, an alignment limit read off a route. None
//! of them is about creation, and reaching one through `Device::create_buffer`
//! would require a device, a capability snapshot that answers `buffer_support`
//! completely, and an allocation — which would make a test about range arithmetic
//! depend on a GPU backend and on the whole creation path being finished.
//!
//! So they assemble the handle as a struct literal through the crate-private
//! constructor, which is exactly what that constructor's own documentation says
//! it is for.
//!
//! # The token, and why it is honest
//!
//! A buffer handle has a native side, and a fixture has none to give it. What it
//! gives instead is [`FixtureBuffer`]: a type whose only job is to exist, so that
//! the seam has something to hand over and the field is not a lie. Nothing in
//! these tests reads it, and a test that *does* need to observe an allocation
//! wants a real device and the mock backend — `api::tests::mock`'s `MockBuffer` — not
//! this.
//!
//! Saying that plainly is the point of the type having a name rather than being
//! an `unimplemented!()` behind an `Arc`: a fixture that panics when read would
//! turn "these tests do not look at allocations" from a fact into a trap.

use std::any::Any;
use std::sync::Arc;

use crate::api::identity::{DeviceIdentity, ObjectId};
use crate::api::resource::backend::BufferBackend;
use crate::api::resource::buffer::{Buffer, BufferDescriptor};

/// The native side of a fixture buffer: a token that is never read.
pub(crate) struct FixtureBuffer;

impl BufferBackend for FixtureBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A buffer handle assembled without a device.
///
/// The identity, the device, and the descriptor are the caller's, because those
/// are the three facts each test is actually about; only the allocation is
/// supplied by this function.
pub(crate) fn buffer(id: ObjectId, device: DeviceIdentity, descriptor: BufferDescriptor) -> Buffer {
    Buffer::new(id, device, descriptor, Arc::new(FixtureBuffer))
}
