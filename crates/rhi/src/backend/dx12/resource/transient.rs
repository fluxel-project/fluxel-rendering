//! Direct3D 12's current transient allocation baseline.
//!
//! The public transient contract is already frozen. This backend currently uses
//! dedicated committed allocations as its correct implementation; D3D12 heap
//! placement and aliasing barriers can replace that lowering later without a
//! public API change.
//!
//! TODO(perf): Upgrade to placed resources only after a device-owned heap/page
//! allocator proves non-overlap from `TransientLifetime` and ordered `PlanPoint`s.
//! Reusing a physical range also needs the D3D12 aliasing barrier from old to new
//! logical resource, normal transitions from actual `ResourceUse`, and retirement
//! through the final release-frontier completion rather than Rust `Drop`. Until
//! all are present `Aliasing` is a correctness lie. The frozen transient API has
//! all vocabulary this private lowering needs.

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
