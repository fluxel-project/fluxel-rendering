//! Safe RHI resource creation, validation, and lifetime contracts.
//!
//! This module turns validated portable descriptors into opaque, device-bound
//! native resources and closed fixed-pipeline bindings. It does not expose HAL
//! objects or command recording; [`crate::execution`] consumes its leases.
//! Resource identity distinguishes physical generations, and leases retain
//! every native dependency until an accepted submission is complete.

use core::fmt;
use std::sync::Arc;

use fluxel_rendergraph::{
    BufferDesc, BufferUsage, BufferUsageKind, CompletionStatus, IndexFormat,
    PhysicalResourceIdentity, ResourceAccessState, TextureDesc, TextureDimension, TextureFormat,
    TextureUsage, TextureUsageKind,
};

use crate::{Backend, Device, NativeCompletion};

mod artifact;
mod bindings;
mod creation;
mod lease;
mod owned;
mod raster;
mod upload;
mod validation;

pub use artifact::*;
pub use bindings::*;
pub use lease::*;
pub use owned::*;
pub use upload::*;

pub use raster::*;

#[cfg(test)]
mod tests;
