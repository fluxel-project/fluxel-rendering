//! Native execution of portable RenderGraph copy, fixed-compute, and fixed-raster plans.
//!
//! This boundary validates plan declarations against device-bound opaque RHI
//! resources, records native commands privately, and retains every referenced
//! lease until submission completion. It neither defines RenderGraph semantics
//! nor exposes HAL values. Active encoder scopes are exclusive, and recipe or
//! pipeline changes invalidate recipe-specific binding state before recording
//! can continue.

pub(super) use core::{fmt, time::Duration};
pub(super) use std::collections::HashMap;
pub(super) use std::ops::Range;

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub(super) use fluxel_rendergraph::SurfaceCapabilities;
pub(super) use fluxel_rendergraph::{
    BoundBuffer, BoundTexture, BufferCapabilities, BufferCopyRegion, BufferDesc, BufferRange,
    BufferUsage, BufferUsageKind, CompletionFailure, CompletionStatus, DeviceCapabilities,
    DeviceLimits, ExecutionBackend, IndexFormat, LoadOp, PresentationSubmission, QueueCapabilities,
    QueueDescriptor, QueueId, RasterPassDescriptor, RecordingCapabilities, RecordingModel,
    ResourceAccessState, ScissorRect, StoreOp, SynchronizationCapabilities, TextureCopyRegion,
    TextureDesc, TextureFormat, TextureFormatCapabilities, TextureRange, TextureUsage,
    TextureUsageKind, TimestampCapabilities, TransientResourceCapabilities, TransitionCapabilities,
    Viewport,
};

pub(super) use crate::{
    Buffer, BufferDescriptor, ComputeBindings, ComputePipeline, Device, MemoryPolicy,
    PresentationToken, RasterNormalBindings, RasterPipeline, RasterTextureBindings,
    RasterUniformBindings, RasterUvLinearClampTextureBindings, RasterUvTextureBindings,
    RasterVertexColorBindings, ResourceCreateError, ResourceLease, Texture, TextureDescriptor,
    TexturePackBindings,
};

mod compute;
#[cfg(test)]
#[allow(
    unused_imports,
    reason = "preserves the former crate-internal contract-test path"
)]
pub(in crate::execution) use tests::contract as contract_tests;
mod copy;
mod errors;
mod helpers;
mod provider;
mod raster;
#[cfg(any(test, feature = "test-support"))]
mod test_support;
#[cfg(test)]
mod tests;
mod types;

pub use compute::*;
pub(in crate::execution) use copy::copy_capabilities;
pub use errors::*;
pub use provider::*;
#[cfg(test)]
pub(in crate::execution) use raster::capabilities::raster_capabilities_from_limit_and_filterability;
pub use raster::*;
#[cfg(any(test, feature = "test-support"))]
#[allow(
    unused_imports,
    reason = "no-backend unit-test builds retain the sibling-fixture API without consuming it"
)]
pub use test_support::*;
pub use types::*;
