//! Device entry points that lower portable resource contracts into native objects.
//!
//! Each child owns one construction family. They share validation and opaque
//! resource types from the parent module, but keep independent artifact recipes
//! from accumulating in one implementation block.

mod compute_artifact;
mod immutable_upload;
mod owned_resource;
mod raster_artifact;
