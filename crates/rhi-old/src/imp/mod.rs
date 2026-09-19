//! Private native boundary containing all HAL objects and unsafe lowering.

// Windows-only containment of the `wgpu-hal` bootstrap API.

#![cfg_attr(
    any(
        all(feature = "dx12", not(feature = "vulkan")),
        all(feature = "vulkan", not(feature = "dx12"))
    ),
    allow(
        dead_code,
        irrefutable_let_patterns,
        unreachable_patterns,
        unused_mut,
        unused_variables,
        reason = "single-backend builds intentionally collapse exhaustive native enums and omit the dual-backend hardware fixture"
    )
)]

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
use crate::{
    Backend, BufferDescriptor, BufferUploadStage, DeviceKind, DeviceOptions, HardwareCapabilities,
    HardwareInfo, OpenError, ResourceCreateError, ResourceLease, TextureDescriptor,
    TextureUploadStage, Validation,
};
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
use fluxel_rendergraph::{
    BufferCopyRegion, BufferUsage, BufferUsageKind, CompletionFailure, CompletionStatus,
    IndexFormat, ResourceAccessState, TextureAspect, TextureCopyRegion, TextureDesc,
    TextureDimension, TextureFormat, TextureRange, TextureUsage, TextureUsageKind,
};
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
use std::{
    borrow::Cow,
    collections::HashMap,
    num::NonZeroU64,
    sync::{Arc, Mutex, MutexGuard},
};
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
use wgpu_hal::{Adapter as _, CommandEncoder as _, Device as _, Instance as _, Queue as _};
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
use wgpu_types as wgt;

// Each native concern is a real child module.  The façade below is the only
// path exported to the rest of RHI; leaves may communicate through
// `pub(crate)` contracts but never expose HAL values outside this boundary.
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
mod bindings;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
mod command;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
mod common;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
mod device_open;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
mod diagnostics;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
mod lowering;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
mod pipeline;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
mod presentation;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
mod resource;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
mod submission;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan"), test))]
mod tests;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
mod upload;

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub(crate) use bindings::*;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub(crate) use command::*;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub(crate) use common::*;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub(crate) use device_open::*;
#[cfg(all(
    windows,
    any(feature = "dx12", feature = "vulkan"),
    any(test, feature = "test-support")
))]
pub(crate) use diagnostics::*;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub(crate) use lowering::*;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub(crate) use pipeline::*;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub(crate) use presentation::*;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub(crate) use resource::*;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub(crate) use submission::*;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub(crate) use upload::*;

#[cfg(any(
    not(windows),
    all(windows, not(any(feature = "dx12", feature = "vulkan")))
))]
mod stub;
#[cfg(any(
    not(windows),
    all(windows, not(any(feature = "dx12", feature = "vulkan")))
))]
pub(crate) use stub::*;
