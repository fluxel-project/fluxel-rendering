//! Resource identity: which device a resource belongs to, and which kind it is.
//!
//! A [`ResourceId`] answers two questions and deliberately not a third:
//!
//! 1. **Which device?** The stamp carries the device identity and its generation,
//!    so an id from a device that has since been replaced is rejected rather than
//!    silently accepted by the replacement.
//! 2. **Which kind?** The kind parameter makes a buffer id and a texture id
//!    different types, so passing one where the other is expected does not
//!    compile. There is no run-time "wrong kind" error because there is no way to
//!    express one.
//!
//! The third question -- "is this the same physical resource generation?" -- is
//! answered by the opaque [`PhysicalResourceIdentity`] the id carries, which is
//! the portable vocabulary's own notion of one physical resource generation. How a
//! backend produces that value is the backend's business, and the two live
//! approaches already differ: the native path derives it from a monotonic counter
//! for owned resources and from `(generation, slot)` for transients, while the GL
//! family guards raw-name reuse with a slot generation of its own. The base fixes
//! the *question*, not the recipe.
//!
//! # What is deliberately absent
//!
//! No name, no handle, no pointer, no backend tag. An id is a value a backend
//! associates with its own object in a private table; it is never the object. That
//! is what keeps a raw handle or a driver name from reaching a caller, and it is
//! why the base can promise "no backend-specific object escapes" without auditing
//! every accessor.

use std::marker::PhantomData;

use fluxel_rendergraph::PhysicalResourceIdentity;

use crate::common::base::stamp::{DeviceStamp, StampMismatch};

/// The kinds of resource the base knows how to identify.
///
/// The set is closed because the base's resource model is: vertex, index, uniform
/// and storage buffers on one side, and textures on the other. A backend does not
/// add kinds here; a kind arrives only with the resource model that needs it.
pub(crate) trait ResourceKind: 'static {
    /// The kind's name, for diagnostics.
    const NAME: &'static str;
}

/// A buffer resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BufferKind;

impl ResourceKind for BufferKind {
    const NAME: &'static str = "buffer";
}

/// A texture resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TextureKind;

impl ResourceKind for TextureKind {
    const NAME: &'static str = "texture";
}

/// Identity of one resource on one device generation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ResourceId<K: ResourceKind> {
    stamp: DeviceStamp,
    identity: PhysicalResourceIdentity,
    kind: PhantomData<K>,
}

/// Identity of a buffer.
pub(crate) type BufferId = ResourceId<BufferKind>;

/// Identity of a texture.
pub(crate) type TextureId = ResourceId<TextureKind>;

impl<K: ResourceKind> ResourceId<K> {
    /// Creates an id for `identity` on the device generation `stamp` names.
    pub(crate) const fn new(stamp: DeviceStamp, identity: PhysicalResourceIdentity) -> Self {
        Self {
            stamp,
            identity,
            kind: PhantomData,
        }
    }

    /// Returns the device generation this id belongs to.
    pub(crate) const fn stamp(self) -> DeviceStamp {
        self.stamp
    }

    /// Returns the opaque physical resource generation.
    pub(crate) const fn identity(self) -> PhysicalResourceIdentity {
        self.identity
    }

    /// Returns the kind's name, for diagnostics.
    pub(crate) const fn kind_name(self) -> &'static str {
        K::NAME
    }

    /// Verifies that this id may be used by a device in generation `current`.
    ///
    /// The failure is the stamp's own, so an id and a raw stamp report the same
    /// two facts in the same words: a foreign device and a replaced generation are
    /// different mistakes and neither is collapsed into "invalid resource".
    pub(crate) fn verify(self, current: DeviceStamp) -> Result<(), StampMismatch> {
        self.stamp.verify(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxel_rendergraph::DeviceIdentity;

    fn stamp(value: u64) -> DeviceStamp {
        DeviceStamp::initial(DeviceIdentity::new(value))
    }

    fn raw(value: u64) -> PhysicalResourceIdentity {
        PhysicalResourceIdentity::new(value)
    }

    fn buffer(value: u64) -> BufferId {
        BufferId::new(stamp(1), raw(value))
    }

    #[test]
    fn an_id_verifies_against_its_own_device_generation() {
        let id = buffer(7);
        assert_eq!(id.verify(stamp(1)), Ok(()));
        assert_eq!(id.identity(), raw(7));
        assert_eq!(id.kind_name(), "buffer");
    }

    #[test]
    fn an_id_from_another_device_is_rejected() {
        let id = buffer(7);
        assert_eq!(id.verify(stamp(2)), Err(StampMismatch::ForeignDevice));
    }

    #[test]
    fn an_id_from_a_replaced_generation_is_rejected() {
        let id = buffer(7);
        let replacement = stamp(1).next_generation();
        assert_eq!(
            id.verify(replacement),
            Err(StampMismatch::StaleGeneration {
                object: 0,
                current: 1,
            })
        );
    }

    #[test]
    fn two_resources_of_one_kind_differ_by_identity_alone() {
        assert_ne!(buffer(1), buffer(2));
        // The same physical identity on the same stamp is the same id, which is
        // what lets a backend use the id as its private table key.
        assert_eq!(buffer(1), buffer(1));
    }

    #[test]
    fn the_kind_is_the_type_rather_than_a_field() {
        // A buffer id and a texture id with the same stamp and identity are still
        // different types, so this function cannot be handed the wrong one.
        fn takes_buffer(_: BufferId) {}
        takes_buffer(buffer(1));
        let texture = TextureId::new(stamp(1), raw(1));
        assert_eq!(texture.kind_name(), "texture");
        assert_ne!(texture.identity(), raw(2));
    }
}
