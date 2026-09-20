//! Vulkan lowering for the frozen Fluxel v13 execution contract.
//!
//! Native Vulkan vocabulary stays below this module. Public resources retain
//! only device identity and opaque backend objects; queue-family indices,
//! memory types, image layouts, pipeline stages, access masks, semaphores and
//! fences are all backend-private mechanics.
//!
//! The implementation grows by end-to-end vertical slices. Dedicated buffer,
//! texture, view and sampler ownership exists; the published slice covers
//! buffer/texture transfer, SPIR-V shaders, immutable descriptor sets, compute
//! buffer/image dispatch, offscreen raster lowering, and Win32 presentation.
//! An available Vulkan feature is not yet a Fluxel capability by itself.

pub(crate) mod binding;
pub(crate) mod command;
pub(crate) mod failure;
pub(crate) mod ffi;
mod format;
pub(crate) mod pipeline;
pub(crate) mod platform;
#[cfg(windows)]
pub(crate) mod presentation;
pub(crate) mod resource;
pub(crate) mod shader;

#[cfg(test)]
mod compute_tests;
#[cfg(test)]
mod image_binding_tests;
#[cfg(all(test, windows, feature = "dx12"))]
mod presentation_tests;
#[cfg(test)]
mod raster_tests;
