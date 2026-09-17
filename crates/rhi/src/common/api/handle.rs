//! What a negotiated family handle can answer about itself.
//!
//! Every family handle -- graphics, compute, storage, indirect -- owes its caller
//! one answer that has nothing to do with its own vocabulary: **which device
//! generation it belongs to.**
//!
//! # Why the handle's stamp, and not a freshly read one
//!
//! A resource id is verified against a [`DeviceStamp`]. The question is which
//! stamp to verify against, and there is a wrong answer that looks right: reading
//! the device's current stamp at the point of use. If the device was replaced
//! between negotiation and use, that re-read returns the *replacement's* stamp,
//! and a stale id from the retired generation would then verify successfully --
//! the exact mixing `DeviceStamp` exists to prevent.
//!
//! So a handle reports the stamp it was negotiated on, and callers verify against
//! that. It is a lifetime rather than a lookup: the handle cannot outlive the
//! device it borrows, so its stamp cannot outlive the ledger that proved the
//! family either.
//!
//! # Why this is a supertrait rather than a field
//!
//! A family handle is a backend's own type, so the base cannot put a field in it.
//! Making this a supertrait of every family's vocabulary means a handle that
//! cannot answer the question cannot be constructed, which is cheaper than
//! discovering at a call site that the answer is missing.

use crate::common::base::resource::{BufferId, TextureId};
use crate::common::base::stamp::{DeviceStamp, StampMismatch};

/// A negotiated handle onto one capability family.
pub(crate) trait FamilyApi {
    /// Returns the device generation this handle was negotiated on.
    fn stamp(&self) -> DeviceStamp;
}

/// Verifies a resource id against the device generation a handle was negotiated on.
///
/// Free rather than a method so it works for any handle without the family traits
/// having to repeat it, and so the one correct source of the stamp -- the handle
/// itself -- is the only input it accepts.
pub(crate) fn verify_buffer<A: FamilyApi + ?Sized>(
    api: &A,
    id: BufferId,
) -> Result<(), StampMismatch> {
    id.verify(api.stamp())
}

/// Verifies a texture id against the device generation a handle was negotiated on.
pub(crate) fn verify_texture<A: FamilyApi + ?Sized>(
    api: &A,
    id: TextureId,
) -> Result<(), StampMismatch> {
    id.verify(api.stamp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxel_rendergraph::{DeviceIdentity, PhysicalResourceIdentity};

    /// A handle that reports the stamp it was negotiated on.
    #[derive(Debug)]
    struct MockHandle {
        stamp: DeviceStamp,
    }

    impl FamilyApi for MockHandle {
        fn stamp(&self) -> DeviceStamp {
            self.stamp
        }
    }

    fn stamp(value: u64) -> DeviceStamp {
        DeviceStamp::initial(DeviceIdentity::new(value))
    }

    fn buffer(stamp: DeviceStamp) -> BufferId {
        BufferId::new(stamp, PhysicalResourceIdentity::new(7))
    }

    #[test]
    fn an_id_from_the_handles_own_generation_verifies() {
        let api = MockHandle { stamp: stamp(1) };
        assert_eq!(verify_buffer(&api, buffer(stamp(1))), Ok(()));
        assert_eq!(
            verify_texture(
                &api,
                TextureId::new(stamp(1), PhysicalResourceIdentity::new(9))
            ),
            Ok(())
        );
    }

    #[test]
    fn an_id_from_another_device_is_rejected() {
        let api = MockHandle { stamp: stamp(1) };
        assert_eq!(
            verify_buffer(&api, buffer(stamp(2))),
            Err(StampMismatch::ForeignDevice)
        );
    }

    #[test]
    fn a_stale_id_is_rejected_against_the_replacement_generation() {
        // The device was replaced after the resource was created. A handle from
        // the replacement generation must refuse the retired id, and it does so
        // precisely because it reports its *own* stamp rather than reading
        // whatever the device currently is.
        let negotiated = stamp(1);
        let retired = buffer(negotiated);
        let replacement = MockHandle {
            stamp: negotiated.next_generation(),
        };
        assert_eq!(
            verify_buffer(&replacement, retired),
            Err(StampMismatch::StaleGeneration {
                object: 0,
                current: 1,
            })
        );
    }

    #[test]
    fn the_handle_still_accepts_its_own_generations_resources() {
        // The other half of the previous test: a handle from the retired
        // generation remains usable with resources of that generation, which is
        // what lets work already in flight finish instead of being torn down.
        let negotiated = stamp(1);
        let api = MockHandle { stamp: negotiated };
        assert_eq!(verify_buffer(&api, buffer(negotiated)), Ok(()));
    }
}
