//! Canonical GL-family profile vocabulary.
//!
//! The migrated typed API owns this backend-private vocabulary. This module is
//! only the capability-probe import seam; keeping a second profile enum here
//! would let native, browser and capability discovery disagree.

pub(crate) use super::api::{GlFamilyProfile as GlProfile, GlVersion};
