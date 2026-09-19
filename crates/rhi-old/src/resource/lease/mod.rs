//! Safe leases and immutable-upload completion handles.
//!
//! This module exposes opaque resource and binding lifetime handles; it does
//! not create native objects or encode commands. A lease keeps the device and
//! every native dependency alive through submission. Upload children retain
//! accepted work until completion is proven, so dropping a pending safe handle
//! cannot release native staging storage too early.

use super::*;

mod buffer_upload;
mod common;
mod texture_upload;

#[allow(
    unused_imports,
    reason = "the parent module re-exports this stable resource API"
)]
pub use buffer_upload::*;
pub use common::*;
#[allow(
    unused_imports,
    reason = "the parent module re-exports this stable resource API"
)]
pub use texture_upload::*;
