//! Native command recording organized by copy, compute, and raster responsibilities.
//!
//! The safe execution layer owns pass ordering; these leaves lower validated operations
//! and retain native objects through command completion.

use super::*;

mod compute;
mod copy;
mod raster;

pub(crate) use compute::*;
pub(crate) use copy::*;
pub(crate) use raster::*;
