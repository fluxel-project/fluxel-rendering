//! Vulkan lowering for the frozen Fluxel v13 execution contract.
//!
//! Native Vulkan vocabulary stays below this module. Public resources retain
//! only device identity and opaque backend objects; queue-family indices,
//! memory types, image layouts, pipeline stages, access masks, semaphores and
//! fences are all backend-private mechanics.
//!
//! The implementation grows by end-to-end vertical slices. Dedicated buffer,
//! texture, view and sampler ownership exists; the published slice covers
//! buffer/texture transfer, SPIR-V shaders, immutable descriptor sets, and
//! buffer-backed compute dispatch. Raster command/pipeline lowering and
//! presentation remain closed until their vertical slices are complete. An
//! available Vulkan feature is not yet a Fluxel capability by itself.

pub(crate) mod binding;
pub(crate) mod command;
pub(crate) mod failure;
pub(crate) mod ffi;
mod format;
pub(crate) mod pipeline;
pub(crate) mod platform;
pub(crate) mod resource;
pub(crate) mod shader;

#[cfg(test)]
mod compute_tests;
