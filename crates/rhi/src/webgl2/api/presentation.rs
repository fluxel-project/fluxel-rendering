//! Host-sized surface lifecycle with generation-bound acquire leases.

use super::{GlError, GlFamilyApi, SurfaceImageId};
use std::collections::BTreeSet;
use std::num::NonZeroU64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlSurfaceSize {
    pub width: u32,
    pub height: u32,
}
impl GlSurfaceSize {
    pub const fn is_zero(self) -> bool {
        self.width == 0 || self.height == 0
    }
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlSurfaceGeneration(NonZeroU64);
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlSurfaceLease {
    pub image: SurfaceImageId,
    pub generation: GlSurfaceGeneration,
    serial: NonZeroU64,
    pub size: GlSurfaceSize,
}
/// Tracks the leases owned by the current surface generation.
#[derive(Debug)]
pub(crate) struct GlSurfaceLeaseBook {
    generation: GlSurfaceGeneration,
    next_serial: u64,
    live: BTreeSet<u64>,
}
impl GlSurfaceLeaseBook {
    pub(crate) fn new() -> Self {
        Self {
            generation: GlSurfaceGeneration(NonZeroU64::MIN),
            next_serial: 0,
            live: BTreeSet::new(),
        }
    }
    pub(crate) fn acquire(
        &mut self,
        image: SurfaceImageId,
        size: GlSurfaceSize,
    ) -> Result<GlSurfaceLease, GlError> {
        self.next_serial = self
            .next_serial
            .checked_add(1)
            .ok_or_else(|| GlError::Validation {
                operation: "acquire_surface_image",
                message: "surface lease serial exhausted".into(),
            })?;
        let serial = NonZeroU64::new(self.next_serial).unwrap();
        self.live.insert(serial.get());
        Ok(GlSurfaceLease {
            image,
            generation: self.generation,
            serial,
            size,
        })
    }
    pub(crate) fn validate(&self, lease: GlSurfaceLease) -> Result<(), GlError> {
        (lease.generation == self.generation && self.live.contains(&lease.serial.get()))
            .then_some(())
            .ok_or_else(|| GlError::Validation {
                operation: "present_surface",
                message: "surface acquire lease is stale or already consumed".into(),
            })
    }
    pub(crate) fn consume(&mut self, lease: GlSurfaceLease) -> Result<(), GlError> {
        self.validate(lease)?;
        self.live.remove(&lease.serial.get());
        Ok(())
    }
    /// Resize, suspension, and resume all require this before another acquire.
    pub(crate) fn invalidate_generation(&mut self) -> Result<(), GlError> {
        let next = self
            .generation
            .0
            .get()
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .ok_or_else(|| GlError::Validation {
                operation: "surface_lifecycle",
                message: "surface generation exhausted".into(),
            })?;
        self.generation = GlSurfaceGeneration(next);
        self.live.clear();
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlSurfaceAcquire {
    Lease(GlSurfaceLease),
    Suspended,
}

/// RHI owns the surface executor; Host owns the native window. Resize,
/// suspend, and resume invalidate all old leases before they return.
pub(crate) trait GlSurfacePresentationApi: GlFamilyApi {
    fn acquire_surface_image(&mut self) -> Result<GlSurfaceAcquire, GlError>;
    fn resize_surface(&mut self, size: GlSurfaceSize) -> Result<(), GlError>;
    fn suspend_surface(&mut self) -> Result<(), GlError>;
    fn resume_surface(&mut self) -> Result<(), GlError>;
    fn present_surface(&mut self, lease: GlSurfaceLease) -> Result<(), GlError>;
}

#[cfg(test)]
mod tests {
    use super::{GlSurfaceLeaseBook, GlSurfaceSize};
    use crate::webgl2::api::{ContextEpoch, ContextStamp, DeviceIdentity, SurfaceImageId};
    #[test]
    fn resize_invalidates_an_old_acquire_lease() {
        let s = ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL);
        let mut b = GlSurfaceLeaseBook::new();
        let lease = b
            .acquire(
                SurfaceImageId::new(s, 1, 1),
                GlSurfaceSize {
                    width: 1,
                    height: 1,
                },
            )
            .unwrap();
        b.invalidate_generation().unwrap();
        assert!(b.validate(lease).is_err());
    }
}
