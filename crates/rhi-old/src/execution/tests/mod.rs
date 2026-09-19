//! Native execution fixtures partitioned by contract family.

// The contract suite lives in this real test tree. `execution::contract_tests`
// re-exports it for existing crate-internal call sites; Rust test names follow
// the defining module (`execution::tests::contract`), which is the necessary
// consequence of the normal `tests/mod.rs` layout.
pub(in crate::execution) mod contract;

#[cfg(all(test, windows, feature = "dx12", feature = "vulkan"))]
mod native_common;
#[cfg(all(test, windows, feature = "dx12", feature = "vulkan"))]
mod native_compute;
#[cfg(all(test, windows, feature = "dx12", feature = "vulkan"))]
mod native_compute_negative;
#[cfg(all(test, windows, feature = "dx12", feature = "vulkan"))]
mod native_copy;
#[cfg(all(test, windows, feature = "dx12", feature = "vulkan"))]
mod native_negative_support;
#[cfg(all(test, windows, feature = "dx12", feature = "vulkan"))]
mod native_raster;
#[cfg(all(test, windows, feature = "dx12", feature = "vulkan"))]
mod native_raster_recipes;
#[cfg(all(test, windows, feature = "dx12", feature = "vulkan"))]
mod native_raster_state;
