//! Resource allocation on Direct3D 12.
//!
//! One file today — [`buffer`] — because the only portable resource this backend
//! can allocate is a buffer. Textures, texture views and samplers join it as
//! their own files rather than as further sections of one, since each has its own
//! native object, its own creation-time flags and its own reason to exist.
//!
//! The re-exports below are the chapter's inside face: `dx12` is a private module
//! of the crate, so `pub(crate)` here is narrower than it looks and reaches only
//! this backend's other chapters. They exist so that a caller says
//! `resource::create_buffer` rather than naming the file, which is what lets
//! [`buffer`] be split further without touching its callers.

mod buffer;

pub(crate) use buffer::{Dx12Buffer, StagingHeap, create_buffer, create_staging};
