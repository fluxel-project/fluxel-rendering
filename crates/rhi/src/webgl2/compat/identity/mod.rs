//! The common device identity one live GL-family context reports.
//!
//! The common contract identifies a device with an opaque `u64` that "need only
//! be unique among simultaneously live device instances"
//! (`fluxel_rendergraph::DeviceIdentity`); the GL family identifies a context
//! with an allocator-owned device identity plus a monotonic epoch
//! (`ContextStamp`).  Those are different kinds of fact, and the mapping between
//! them is an allocation rather than a conversion.
//!
//! Deliberately the *same* allocation the native backends already use:
//! `Device::open` takes its identity from this crate's process-local counter, and
//! taking this one from the same counter is what keeps a GL context and a DX12
//! device from ever colliding.  Deriving the common identity from the stamp
//! instead would be deterministic, and wrong in exactly the case the identity
//! exists for: a stamp is not unique to a live context -- two contexts can be
//! created on one GL device from one allocator metadata value -- and the common
//! identity is what the graph rejects a foreign bound resource with.  A derived
//! identity would make those two contexts the same device and quietly disable
//! that rejection.
//!
//! A lost context is a generation change and not a new device: the GL device
//! identity survives it and the epoch advances.  This type follows the epoch, so
//! the common identity changes with the generation -- which is what lets 0.14
//! residency see the recreation instead of reusing resources the new context
//! never held.

use fluxel_rendergraph::DeviceIdentity;

use crate::webgl2::api::ContextStamp;

/// The common identity of one live GL-family context, and the stamp it came from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DeviceIdentityMap {
    stamp: ContextStamp,
    identity: DeviceIdentity,
}

impl DeviceIdentityMap {
    /// Takes the common identity for `stamp` from the crate's device allocator.
    pub(crate) fn new(stamp: ContextStamp) -> Self {
        Self {
            stamp,
            identity: allocate(),
        }
    }

    /// The identity every bound object of this context is checked against.
    pub(crate) fn identity(&self) -> DeviceIdentity {
        self.identity
    }

    /// The context generation this identity currently stands for.
    pub(crate) fn stamp(&self) -> ContextStamp {
        self.stamp
    }

    /// Adopts `stamp` and reports whether the common identity changed with it.
    ///
    /// The caller reads the provider's stamp after a loss and a restoration
    /// rather than predicting it, so this is the only place the two can
    /// disagree; `false` means the generation is the one already held, and every
    /// derived bound object stays valid.
    pub(crate) fn adopt(&mut self, stamp: ContextStamp) -> bool {
        if stamp == self.stamp {
            return false;
        }
        self.stamp = stamp;
        self.identity = allocate();
        true
    }
}

/// One identity from this crate's process-local device counter.
fn allocate() -> DeviceIdentity {
    DeviceIdentity::new(crate::next_identity())
}

#[cfg(test)]
mod tests;
