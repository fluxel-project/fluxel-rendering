//! Fence lifetime and bounded-progress contracts without native handles.

use super::{GlError, GlFamilyApi, SyncId};
use std::collections::BTreeSet;
use std::num::NonZeroU64;

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
}

#[cfg(test)]
mod tests {
    use super::GlFenceLeaseBook;
    use crate::webgl2::api::{ContextEpoch, ContextStamp, DeviceIdentity, SyncId};
    #[test]
    fn revoked_lease_is_rejected() {
        let s = ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL);
        let mut book = GlFenceLeaseBook::default();
        let lease = book.issue(SyncId::new(s, 1, 1)).unwrap();
        book.revoke(lease);
        assert!(book.validate(lease).is_err());
    }
}
