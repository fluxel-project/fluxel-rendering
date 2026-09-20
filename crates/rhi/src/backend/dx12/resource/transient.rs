//! Direct3D 12's current transient allocation baseline.
//!
//! The public transient contract is already frozen. This backend currently uses
//! dedicated committed allocations as its correct implementation; D3D12 heap
//! placement and aliasing barriers can replace that lowering later without a
//! public API change.

use crate::api::resource::transient::{TransientAllocationSupport, TransientCapabilities};

/// Reports the transient strategy actually implemented by this DX12 backend.
///
/// `Aliasing` must not be reported merely because D3D12 supports placed
/// resources: until allocation placement and aliasing barriers are lowered, the
/// only truthful strategy is the portable Dedicated baseline.
pub(crate) fn transient_capabilities() -> TransientCapabilities {
    TransientCapabilities {
        buffers: TransientAllocationSupport::Dedicated,
        textures: TransientAllocationSupport::Dedicated,
        mixed_resource_aliasing: false,
    }
}
