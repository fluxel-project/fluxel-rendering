//! The identity stamp every device-affine backend object carries.
//!
//! # Why two facts and not one
//!
//! A device identity alone answers "did this object come from another device?".
//! It cannot answer "did this object come from the device that has since been
//! replaced?", and that second question is the one browser and mobile platforms
//! make unavoidable: a lost WebGPU device is recovered into a *new* device, and a
//! lost GL context is restored into a new context epoch. Objects from the old
//! generation are not merely foreign, they are dead, and a backend that compared
//! only the identity would happily hand a dead object to the new generation.
//!
//! So a stamp is `(identity, generation)`, and the generation is opaque: this
//! layer never derives meaning from its value beyond equality, because the
//! platforms disagree about what a generation counts.
//!
//! # Where the two shapes already exist
//!
//! - the native path stamps owned resources with
//!   `fluxel_rendergraph::PhysicalResourceIdentity` and one opened device;
//! - the GL family stamps ids with a context stamp of
//!   `DeviceIdentity` plus a context epoch.
//!
//! Those are the same two facts under two names, which is why one stamp is
//! extracted here rather than a third spelling being added.

use fluxel_rendergraph::DeviceIdentity;

/// The generation of a device that was created rather than replaced.
///
/// Zero is the first generation because the GL family's context epoch already
/// starts there, and a stamp's value is opaque, so starting anywhere else would
/// be a gratuitous difference between two implementations of the same idea.
pub(crate) const INITIAL_GENERATION: u32 = 0;

/// The device a backend object belongs to, and that device's generation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DeviceStamp {
    identity: DeviceIdentity,
    generation: u32,
}

impl DeviceStamp {
    /// Creates the stamp of the first generation of `identity`.
    pub(crate) const fn initial(identity: DeviceIdentity) -> Self {
        Self {
            identity,
            generation: INITIAL_GENERATION,
        }
    }

    /// Creates a stamp for `identity` at `generation`.
    pub(crate) const fn new(identity: DeviceIdentity, generation: u32) -> Self {
        Self {
            identity,
            generation,
        }
    }

    /// Returns the device identity.
    pub(crate) const fn identity(self) -> DeviceIdentity {
        self.identity
    }

    /// Returns the opaque generation.
    pub(crate) const fn generation(self) -> u32 {
        self.generation
    }

    /// Returns the next generation of the same device identity.
    ///
    /// Used when a device is recovered into a replacement. The generation is
    /// monotonic within one identity, so an object from a retired generation can
    /// never be confused with one from the replacement.
    pub(crate) const fn next_generation(self) -> Self {
        Self {
            identity: self.identity,
            generation: self.generation.wrapping_add(1),
        }
    }

    /// Verifies that an object stamped `self` may be used by `current`.
    ///
    /// Returns the reason rather than a boolean, because the two failures are
    /// different sentences to a caller: a foreign device is a mixing mistake,
    /// while a stale generation is a lifetime mistake. The device is checked
    /// first: an object from another device is foreign whatever generation it
    /// claims, and reporting a generation mismatch for it would name the wrong
    /// mistake.
    pub(crate) fn verify(self, current: Self) -> Result<(), StampMismatch> {
        if self.identity != current.identity {
            return Err(StampMismatch::ForeignDevice);
        }
        if self.generation != current.generation {
            return Err(StampMismatch::StaleGeneration {
                object: self.generation,
                current: current.generation,
            });
        }
        Ok(())
    }
}

/// Why a stamped object may not be used by the current device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StampMismatch {
    /// The object belongs to a different device identity.
    ForeignDevice,
    /// The object belongs to a generation of this device that has been replaced.
    StaleGeneration {
        /// The generation the object was created in.
        object: u32,
        /// The generation the device is currently in.
        current: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(value: u64) -> DeviceIdentity {
        DeviceIdentity::new(value)
    }

    #[test]
    fn the_same_device_and_generation_verifies() {
        let stamp = DeviceStamp::initial(identity(1));
        assert_eq!(stamp.verify(stamp), Ok(()));
    }

    #[test]
    fn a_foreign_device_is_reported_before_any_generation_difference() {
        let object = DeviceStamp::new(identity(1), 7);
        let current = DeviceStamp::initial(identity(2));
        assert_eq!(object.verify(current), Err(StampMismatch::ForeignDevice));
    }

    #[test]
    fn a_replaced_generation_reports_both_numbers() {
        let object = DeviceStamp::initial(identity(1));
        let current = object.next_generation();
        assert_eq!(
            object.verify(current),
            Err(StampMismatch::StaleGeneration {
                object: 0,
                current: 1,
            })
        );
    }

    #[test]
    fn a_replacement_keeps_the_identity_and_advances_the_generation() {
        let first = DeviceStamp::initial(identity(3));
        let second = first.next_generation();
        assert_eq!(second.identity(), first.identity());
        assert_eq!(second.generation(), INITIAL_GENERATION + 1);
        // The replacement does not accept the retired generation, and vice versa.
        assert!(first.verify(second).is_err());
        assert!(second.verify(first).is_err());
    }

    #[test]
    fn the_stamp_is_the_pair_rather_than_either_half() {
        let a = DeviceStamp::new(identity(1), 2);
        let b = DeviceStamp::new(identity(1), 3);
        let c = DeviceStamp::new(identity(2), 2);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_eq!(a, DeviceStamp::new(identity(1), 2));
    }
}
