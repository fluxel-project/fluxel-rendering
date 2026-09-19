//! Platform, adapter, and device (specification sections 5 through 7).
//!
//! This module is the composition entry for the chapter that answers "what
//! hardware is there, and what did I get". It owns the tree's shape and the
//! names developers type; the rules themselves live in the submodules:
//!
//! ```text
//! provider      adapter discovery, and the provider that owns it   (5.2 - 5.6)
//! requirements  what a caller asks for before a device exists      (5.7)
//! request       asking for a device, and waiting for the answer    (5.8 - 5.9)
//! device        the shared logical execution domain                (6)
//! ```
//!
//! Capability *facts* are not here — they are [`crate::api::capability`], which
//! both the adapter snapshot and the device answer in the same vocabulary. That
//! symmetry is deliberate: it is what lets a caller compare what was available
//! against what was enabled without learning two shapes.

pub mod device;
pub mod provider;
pub mod request;
pub mod requirements;

pub use device::{Device, DeviceLossInfo, DeviceStatus};
pub use provider::{AdapterId, AdapterInfo, AdapterSelection, BackendKind, PlatformProvider};
pub use request::{DeviceRequest, DeviceRequestDescriptor, RequestStatus};
pub use requirements::{DeviceRequirements, LimitKey, LimitRequirement, OptionalFeature};

#[cfg(test)]
mod tests;
