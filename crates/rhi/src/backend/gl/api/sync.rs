//! Fence lifetime and bounded-progress contracts without native handles.

use super::{GlError, GlFamilyApi, SyncId};
use std::collections::BTreeSet;
use std::num::NonZeroU64;

/// Visibility classes accepted by `glMemoryBarrier`.  A fence orders command
/// completion; it never substitutes for one of these cache-visibility barriers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlMemoryBarrier(pub u32);
impl GlMemoryBarrier {
    pub const VERTEX_ATTRIB_ARRAY: Self = Self(1 << 0);
    pub const ELEMENT_ARRAY: Self = Self(1 << 1);
    pub const UNIFORM: Self = Self(1 << 2);
    pub const TEXTURE_FETCH: Self = Self(1 << 3);
    pub const SHADER_IMAGE_ACCESS: Self = Self(1 << 4);
    pub const COMMAND: Self = Self(1 << 5);
    pub const PIXEL_BUFFER: Self = Self(1 << 6);
    pub const TEXTURE_UPDATE: Self = Self(1 << 7);
    pub const BUFFER_UPDATE: Self = Self(1 << 8);
    pub const FRAMEBUFFER: Self = Self(1 << 9);
    pub const TRANSFORM_FEEDBACK: Self = Self(1 << 10);
    pub const ATOMIC_COUNTER: Self = Self(1 << 11);
    pub const SHADER_STORAGE: Self = Self(1 << 12);
    pub const CLIENT_MAPPED_BUFFER: Self = Self(1 << 13);
    pub const QUERY_BUFFER: Self = Self(1 << 14);
    pub const BY_REGION_LEGAL: Self =
        Self(Self::TEXTURE_FETCH.0 | Self::SHADER_IMAGE_ACCESS.0 | Self::FRAMEBUFFER.0);
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
    pub(crate) fn validate_nonempty(self) -> Result<(), GlError> {
        (self.0 != 0)
            .then_some(())
            .ok_or_else(|| GlError::Validation {
                operation: "memory_barrier",
                message: "at least one barrier class is required".into(),
            })
    }
    pub(crate) fn validate_by_region(self) -> Result<(), GlError> {
        self.validate_nonempty()?;
        (self.0 & !Self::BY_REGION_LEGAL.0 == 0)
            .then_some(())
            .ok_or_else(|| GlError::Validation {
                operation: "memory_barrier_by_region",
                message: "barrier class is not legal for by-region visibility".into(),
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlFenceStatus {
    Pending,
    Complete,
    Failed,
    Unknown,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlWaitBound {
    pub nanoseconds: u64,
}
impl GlWaitBound {
    pub const POLL: Self = Self { nanoseconds: 0 };
}
/// A fence identity may only be used through a currently live lease.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlFenceLease {
    pub fence: SyncId,
    serial: NonZeroU64,
}
/// Fluxel-side lease table: destroy or context loss invalidates outstanding leases.
#[derive(Debug, Default)]
pub(crate) struct GlFenceLeaseBook {
    next: u64,
    live: BTreeSet<u64>,
}
impl GlFenceLeaseBook {
    pub(crate) fn issue(&mut self, fence: SyncId) -> Result<GlFenceLease, GlError> {
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| GlError::Validation {
                operation: "create_fence",
                message: "fence lease serial exhausted".into(),
            })?;
        let serial = NonZeroU64::new(self.next).unwrap();
        self.live.insert(serial.get());
        Ok(GlFenceLease { fence, serial })
    }
    pub(crate) fn validate(&self, lease: GlFenceLease) -> Result<(), GlError> {
        self.live
            .contains(&lease.serial.get())
            .then_some(())
            .ok_or_else(|| GlError::Validation {
                operation: "fence",
                message: "fence lease is stale or was destroyed".into(),
            })
    }
    pub(crate) fn revoke(&mut self, lease: GlFenceLease) {
        self.live.remove(&lease.serial.get());
    }
    pub(crate) fn revoke_all(&mut self) {
        self.live.clear();
    }
}

/// All fence methods preflight owner thread, active lifecycle, context stamp,
/// allocation-table liveness, and lease liveness before a driver/browser call.
pub(crate) trait GlSyncApi: GlFamilyApi {
    /// Inserts a live fence and returns its sole current lease.
    fn create_fence(&mut self) -> Result<GlFenceLease, GlError>;
    /// Revokes the lease and destroys its fence; later use must fail locally.
    fn destroy_fence(&mut self, fence: GlFenceLease) -> Result<(), GlError>;
    fn poll_fence(&mut self, fence: GlFenceLease) -> Result<GlFenceStatus, GlError>;
    fn wait_fence(
        &mut self,
        fence: GlFenceLease,
        bound: GlWaitBound,
    ) -> Result<GlFenceStatus, GlError>;
    /// Flush submits work but never implies fence completion.
    fn flush(&mut self) -> Result<(), GlError>;
    fn memory_barrier(&mut self, barriers: GlMemoryBarrier) -> Result<(), GlError>;
    fn memory_barrier_by_region(&mut self, barriers: GlMemoryBarrier) -> Result<(), GlError>;
    /// Texture feedback ordering, distinct from memory visibility barriers.
    fn texture_barrier(&mut self) -> Result<(), GlError>;
}

#[cfg(test)]
mod tests {
    use super::{GlFenceLeaseBook, GlMemoryBarrier};
    use crate::backend::gl::api::{ContextEpoch, ContextStamp, DeviceIdentity, SyncId};
    #[test]
    fn revoked_lease_is_rejected() {
        let s = ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL);
        let mut book = GlFenceLeaseBook::default();
        let lease = book.issue(SyncId::new(s, 1, 1)).unwrap();
        book.revoke(lease);
        assert!(book.validate(lease).is_err());
    }
    #[test]
    fn by_region_refuses_non_local_visibility_classes() {
        assert!(GlMemoryBarrier::TEXTURE_FETCH.validate_by_region().is_ok());
        assert!(GlMemoryBarrier::COMMAND.validate_by_region().is_err());
    }
}
