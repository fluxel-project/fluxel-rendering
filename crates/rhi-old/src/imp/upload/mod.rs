//! Immutable upload façade.
//!
//! Production submission is separate from staging mechanics and test-only observers, so
//! test readback cannot become part of normal upload lifetime handling.

use super::*;

mod production;
mod staging;
#[cfg(any(test, feature = "test-support"))]
mod test_support;

pub(crate) use production::*;
use staging::*;
pub(crate) use staging::{copy_base, texture_subresources};
#[cfg(any(test, feature = "test-support"))]
pub(crate) use test_support::*;
