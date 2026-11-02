use std::sync::Arc;

use fluxel_rhi::api::DeviceId;

/// An opaque reference to one pending fixture submission.
#[derive(Clone, Debug)]
pub struct Submission {
    pub(super) owner: Arc<()>,
    pub(super) device: DeviceId,
    pub(super) slot: usize,
    pub(super) serial: u64,
}
