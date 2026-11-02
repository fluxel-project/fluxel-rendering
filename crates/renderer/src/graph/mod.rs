//! Private logical resources, versions, dependencies, validation, and execution.
//!
//! Graph will lower directly to portable RHI recording and completion contracts.
//! It does not expose a public scheduler or duplicate backend GPU semantics.

mod access;
mod binding;
pub(crate) mod compiler;
mod compute;
mod copy;
mod definition;
mod execution;
mod raster;
pub(crate) mod read_set_raster;
mod state;

pub(crate) use binding::{BindingError, ResourceInstanceId};
pub(crate) use compute::{ComputeFixtureBindings, ComputePlan};
pub(crate) use copy::{CopyFrame, CopyPlan};
pub(crate) use raster::{RasterPlan, SurfaceRasterPlan};
pub(crate) use read_set_raster::{ReadSetRasterBindings, ReadSetRasterFrame, ReadSetRasterPlan};
