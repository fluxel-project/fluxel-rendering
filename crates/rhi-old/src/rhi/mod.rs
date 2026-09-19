//! Fluxel RHI API v1.
//!
//! This is the portable rendering hardware interface: one vocabulary for
//! devices, resources, pipelines, recording, submission, and presentation, with
//! every backend difference expressed as a declared *fact* rather than as a
//! separate entry point.
//!
//! # What it is
//!
//! The public surface is the module list below plus the identity and error
//! vocabulary re-exported here. A caller never names a native type:
//! `ID3D12Device`, `VkDevice`, `MTLDevice`, `GPUDevice`, `WebGL2RenderingContext`,
//! `HWND`, and `VkSurfaceKHR` all live behind backend-private implementations.
//!
//! # The three layers, kept distinct
//!
//! ```text
//! PlatformProvider -> AdapterInfo + AvailableCapabilities   what is offered
//! Device           -> EnabledCapabilities                   what was enabled
//! Surface fact     -> PresentationTargetCapabilities        what this pair can do
//! ```
//!
//! Correctness always reads the layer that owns the question. An adapter
//! snapshot never substitutes for `Device::capabilities`, and presentation
//! legality is never folded into device capability, because the same device
//! answers differently for two presentation targets.
//!
//! # What it deliberately does not own
//!
//! There is no session, token, lease, or runtime object in this model. Resource
//! ownership is device identity plus generation, which is the same model as
//! `ID3D12Device`, `VkDevice`, `MTLDevice`, `GPUDevice`, and a WebGL2 context.
//! Passing a resource to the wrong device is a structured error from this
//! library rather than something a browser or driver is trusted to catch.

mod hash;

pub mod binding;
pub mod capability;
pub mod command;
pub mod diagnostics;
pub mod format;
pub mod graph_bridge;
#[cfg(any(test, feature = "test-support"))]
pub mod mock;
pub mod platform;
pub mod pipeline;
pub mod presentation;
pub mod resource;
mod retirement;
pub mod shader;
pub mod statistics;
pub mod submission;
#[doc(hidden)]
pub mod tooling;

pub use capability::{
    AvailableCapabilities, CapabilityCompatibilityId, CapabilityFingerprint, DeviceLimits,
    EnabledCapabilities,
};
pub use platform::{
    AdapterId, AdapterInfo, AdapterSelection, BackendKind, Device, DeviceIdentity, DeviceInstanceId,
    DeviceGeneration, DeviceLossInfo, DeviceRequest, DeviceRequestDescriptor, DeviceRequirements,
    DeviceStatus, Label, LimitKey, LimitRequirement, ObjectId, OptionalFeature, PlatformProvider,
    RequestStatus, RhiError, RhiErrorKind, RhiResult,
};
